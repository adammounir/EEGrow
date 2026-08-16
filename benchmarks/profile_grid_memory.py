"""Peak GPU memory of every (model x dataset) cell of the grid, measured not guessed.

WHY THIS EXISTS
---------------
The packed runner has ONE global ``K`` -- co-tenants per GPU -- and K is a memory
bound: ``K x peak_reserved`` must fit on the card. But peak memory is a property of
the *cell*, not of the runner. Two figures already measured on real cells differ by a
factor of nine (``grow_shallow`` 726 MiB, ``grow_eegnex`` 6410 MiB), and one cell was
once seen reserving 43 GB. A single K therefore either wastes nine tenths of the
fleet or OOMs, and which one you get depends on the cell that happens to be claimed.

There is a second reason to measure now rather than reuse the old figures. Those were
taken *before* the growth cap was connected: ``GrowingShallowFBCSPNet`` and
``GrowingDeepEEGNet`` had no ``target_width``, so gromo grew them without any bound
(8 -> 17 measured, 8 -> 77 against a target of 32). Their memory was the memory of an
unbounded model. Every number for those two arms is now stale in the safe direction,
and the exclusion list can only shrink -- but "can only shrink" is not a measurement.

WHAT IT MEASURES
----------------
For each (model, dataset): peak ``allocated`` and peak ``reserved`` on the device,
plus the width the model ended at and whether it reached its target.

``reserved`` is the operative number, not ``allocated``. The caching allocator never
returns a block to the driver, so the device -- and therefore a co-tenant -- only ever
sees the reserved figure. ``allocated`` is what the model needs; ``reserved`` is what
it holds.

HOW, AND WHAT THAT COSTS IN FIDELITY
------------------------------------
Synthetic data at the cell's real shapes, not the real epochs. GPU memory depends on
``(batch, n_chans, n_times, n_outputs)`` and the architecture -- never on the values in
the tensors -- so random data measures the same peak for a fraction of the I/O. What
this does NOT measure is host RAM, which does depend on the real data volume; that
constraint is separate and already known (``lee2019_mi`` is host-RAM-bound, see
``pack_run.sh``).

Two deliberate deviations from the production pipeline, both of which only make the
probe reach the peak sooner:

* ``grow_every=1`` instead of 5. The growth *step* is what peaks, and its cost depends
  on the layer geometry, not on how many epochs preceded it. Growing every epoch
  reaches the cap in a fifth of the epochs and peaks identically.
* ``EarlyStopping`` removed. On random labels validation accuracy never improves, so
  patience would stop the fit at epoch 16 -- before the model has grown to its target,
  i.e. before the peak. Removing it is what makes the measurement an upper bound
  rather than an accident of the noise.

Each cell runs in its own subprocess. This is not tidiness: ``reserved`` is a
process-global high-water mark that ``empty_cache()`` does not reset, so measuring
model B after model A in one process reports A's blocks as B's.

USAGE
-----
    python benchmarks/profile_grid_memory.py --out profile_mem/grid_memory.json

    # one cell, as the driver invokes it (prints one JSON line on stdout)
    python benchmarks/profile_grid_memory.py --worker --model grow_eegnex \
        --dataset schirrmeister2017
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIG = HERE / "config"
# The deep arms only. The ML arms never touch the GPU, so a memory class for them
# would be a row of zeros; their bound is CPU time and host RAM.
MODELS = ["grow_shallow", "bd_shallow", "grow_sccnet", "bd_sccnet",
          "grow_eegnex", "bd_eegnex", "grow_deep", "fix_deepeeg", "bd_deep4"]
DATASETS = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004",
            "bnci2015_001", "cho2017", "lee2019_mi", "physionetmi",
            "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016"]

# Enough batches for a growth step to see a representative epoch; the tensors gromo
# builds are sized by the layer geometry, so more samples would cost time and change
# nothing. 8 x batch_size.
N_SAMPLES = 512
MAX_EPOCHS = 60


# --------------------------------------------------------------------- shapes
def _shapes(cache_path: str, memo: Path, wanted: list[str]) -> dict:
    """``dataset -> (n_chans, n_times, n_outputs)``, from the warm MOABB cache.

    Read from one subject per dataset and memoised to ``memo``: the shapes are a
    property of the paradigm and the resampling grid, not of the run, and re-deriving
    them costs minutes of I/O for numbers that cannot change between two invocations.
    Only the datasets actually asked for are derived, and the memo is extended rather
    than replaced -- so a two-cell smoke test does not pay for all twelve.
    """
    out = json.loads(memo.read_text()) if memo.exists() else {}
    missing = [d for d in wanted if d not in out]
    if not missing:
        return out

    import moabb.datasets as mds
    import moabb.paradigms as mpar
    from omegaconf import OmegaConf

    from utils import cache_config, set_data_dir

    set_data_dir(None)
    for name in missing:
        dcfg = OmegaConf.load(CONFIG / "dataset" / f"{name}.yaml")
        ds = getattr(mds, dcfg.moabb_class)(**(OmegaConf.to_container(
            dcfg.get("kwargs")) or {}))
        pkw = {}
        if dcfg.get("resample"):
            pkw["resample"] = float(dcfg.resample)
        if dcfg.get("tmin") is not None:
            pkw["tmin"] = float(dcfg.tmin)
        if dcfg.get("tmax") is not None:
            pkw["tmax"] = float(dcfg.tmax)
        par = getattr(mpar, dcfg.paradigm)(fmin=8, fmax=32, **pkw)
        # OmegaConf, not a plain dict: cache_config reads `cfg.path` by attribute.
        ccfg = cache_config(OmegaConf.create(
            {"enabled": True, "path": cache_path}))
        subj = ds.subject_list[0]
        X, y, _ = par.get_data(dataset=ds, subjects=[subj],
                               **({"cache_config": ccfg} if ccfg else {}))
        out[name] = {"n_chans": int(X.shape[1]), "n_times": int(X.shape[2]),
                     # From one subject: MOABB's paradigm fixes the label set for the
                     # whole dataset, so a subject is representative of the classes.
                     "n_outputs": int(len(set(y))),
                     "sfreq": float(dcfg.resample) if dcfg.get("resample") else 250.0}
        print(f"  shapes {name}: {out[name]}", flush=True)
        del X, y
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------- worker
def worker(model: str, dataset: str, shapes: dict, n_samples: int = N_SAMPLES) -> dict:
    """Measure one cell and return its record. Runs in its own process.

    ``n_samples`` is a parameter and not the module constant because the constant is
    exactly what this probe got wrong. The growth step used to materialise the whole
    epoch on the growth device, so a growing arm's device peak scaled with the size of
    the training fold -- and 512 synthetic samples are not a fold. Sweeping this
    argument is how one checks that the peak no longer depends on it; if two very
    different values give the same peak, the probe's central assumption (memory is a
    function of shapes and architecture, not of the data) has become true again for
    the growing arms, and a single measurement per cell is once more sufficient.
    """
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from pipelines import build_pipeline

    s = shapes[dataset]
    mcfg = OmegaConf.to_container(
        OmegaConf.load(CONFIG / "model" / f"{model}.yaml"), resolve=True)
    if mcfg.get("grow_every"):
        mcfg["grow_every"] = 1        # see module docstring
    tcfg = {"lr": 6.25e-4, "max_epochs": MAX_EPOCHS, "batch_size": 64}

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, s["n_chans"], s["n_times"])).astype("float32")
    y = rng.integers(0, s["n_outputs"], n_samples).astype("int64")

    pipe = build_pipeline(mcfg, tcfg, n_chans=s["n_chans"], n_times=s["n_times"],
                          n_outputs=s["n_outputs"], sfreq=s["sfreq"],
                          device="cuda", seed=0)
    clf = pipe.named_steps["clf"]
    # Drop EarlyStopping: on random labels it fires long before the model has grown.
    clf.callbacks = [c for c in clf.callbacks
                     if not type(c if not isinstance(c, tuple) else c[1])
                     .__name__.startswith("EarlyStopping")]

    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    t0 = time.time()
    err = None
    try:
        pipe.fit(X, y)
    except torch.cuda.OutOfMemoryError as exc:
        err = f"OOM: {str(exc)[:200]}"
    except Exception as exc:                      # a probe must report, not crash
        err = f"{type(exc).__name__}: {str(exc)[:200]}"

    mod = getattr(clf, "module_", None)
    return {
        "model": model, "dataset": dataset, "n_samples": n_samples,
        "n_chans": s["n_chans"], "n_times": s["n_times"],
        "n_outputs": s["n_outputs"],
        "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "card_total_mib": round(total / 2**20, 1),
        "width_end": getattr(mod, "growable_width", None) if mod else None,
        "target_width": getattr(mod, "target_width", None) if mod else None,
        "n_params": (int(sum(p.numel() for p in mod.parameters())) if mod else None),
        "seconds": round(time.time() - t0, 1),
        "device_name": torch.cuda.get_device_name(0),
        "error": err,
    }


# --------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--dataset")
    ap.add_argument("--out", default=str(HERE / "profile_mem" / "grid_memory.json"))
    ap.add_argument("--cache", default="/scratch/amounir/moabb_cache")
    ap.add_argument("--shapes", default=str(HERE / "profile_mem" / "shapes.json"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help="synthetic trials; sweep it to test that the peak no longer "
                         "depends on the fold size (see worker)")
    args = ap.parse_args()

    # No per-process cap while probing: the point is the peak the cell *wants*, and a
    # cap would report the ceiling instead of the demand.
    os.environ.pop("EEGROW_CUDA_FRACTION", None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    datasets = args.datasets.split(",")
    shapes = _shapes(args.cache, Path(args.shapes),
                     [args.dataset] if args.worker else datasets)

    if args.worker:
        print("@@JSON@@" + json.dumps(
            worker(args.model, args.dataset, shapes, args.n_samples)))
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume: a driver that dies after 80 of 108 cells should not redo the 80.
    done = {}
    if out_path.exists():
        done = {(r["model"], r["dataset"]): r
                for r in json.loads(out_path.read_text())
                if r.get("n_samples", N_SAMPLES) == args.n_samples}

    models = args.models.split(",")
    total = len(models) * len(datasets)
    i = 0
    for ds in datasets:
        for m in models:
            i += 1
            if (m, ds) in done:
                print(f"[{i}/{total}] {m} x {ds}: cached", flush=True)
                continue
            cmd = [sys.executable, __file__, "--worker", "--model", m,
                   "--dataset", ds, "--cache", args.cache, "--shapes", args.shapes,
                   "--n-samples", str(args.n_samples)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            rec = None
            for line in p.stdout.splitlines():
                if line.startswith("@@JSON@@"):
                    rec = json.loads(line[len("@@JSON@@"):])
            if rec is None:
                # A worker killed outright (OOM killer, segfault) leaves no line. That
                # is itself the result: the cell could not be measured on this card.
                rec = {"model": m, "dataset": ds, "error":
                       f"worker died rc={p.returncode}: {p.stderr.strip()[-300:]}"}
            done[(m, ds)] = rec
            out_path.write_text(json.dumps(list(done.values()), indent=2))
            print(f"[{i}/{total}] {m:13s} x {ds:19s} "
                  f"reserved={rec.get('peak_reserved_mib', '?')} MiB "
                  f"alloc={rec.get('peak_alloc_mib', '?')} MiB "
                  f"width={rec.get('width_end')}/{rec.get('target_width')} "
                  f"{rec.get('seconds', '?')}s {rec.get('error') or ''}", flush=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
