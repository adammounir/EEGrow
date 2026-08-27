"""Peak host RAM of one tenant per dataset, measured not guessed.

WHY THIS EXISTS, GIVEN profile_grid_memory.py ALREADY RAN
---------------------------------------------------------
The GPU profile deliberately used synthetic data at the cell's real shapes, because
device memory depends on ``(batch, n_chans, n_times)`` and the architecture, never on
the values in the tensors. Host RAM is the opposite: it is dominated by the epoched
array MOABB materialises, which is exactly the thing synthetic data replaces. So the
GPU numbers say nothing here, and its docstring says so.

This matters because host RAM, not device memory, is what killed the first packed
campaign's ``lee2019_mi`` cells -- 30 of them, SIGKILLed mid-load by the cgroup OOM
killer, no traceback, GPU idle. A K derived from device memory alone reproduces that
failure exactly.

WHY THE BOUND IS PER-DATASET AND NOT PER-PROTOCOL
-------------------------------------------------
One might expect ``within_session`` to hold one subject at a time. It does not.
``BaseEvaluation._process_parallel`` loads::

    subjects_to_load = (dataset.subject_list if self._needs_all_subjects
                        else list(work_plan.keys()))

``CrossSubjectEvaluation`` sets ``_needs_all_subjects = True``, so it loads everything
unconditionally. The other two leave it False -- but ``work_plan`` is the set of
subjects that still have work, and on a fresh campaign run with ``overwrite=true``
that is every subject in the dataset. All three protocols therefore hold the whole
dataset, and one measurement per dataset covers the grid.

The array then stays resident for the life of the cell: ``n_jobs=1`` (config.yaml), so
``Parallel`` runs in-process and joblib's auto-memmap -- which is what the "passed as
positional args for joblib auto-mmap" comment upstream is about -- never triggers.

WHAT IS MEASURED
----------------
``ru_maxrss``: the high-water mark of resident memory, which is the quantity the
cgroup OOM killer compares against the limit. Not the array size. The two differ by
the transient cost of decoding raw files into epochs, and it is the transient that
kills -- the first campaign's cells died *during* the load, not after it.

One subprocess per dataset, for the same reason the GPU probe used one: ``ru_maxrss``
is a high-water mark that never comes back down, so a second dataset measured in the
same process inherits the first one's peak.

No GPU is requested and no CUDA call is made. This probe competes with nothing on the
tau queue, which is why it can run while the device-side questions are still open.

USAGE
-----
    python benchmarks/profile_host_ram.py --out profile_mem/host_ram.json

    # one dataset, as the driver invokes it (prints one JSON line on stdout)
    python benchmarks/profile_host_ram.py --worker --dataset lee2019_mi
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIG = HERE / "config"
DATASETS = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004",
            "bnci2015_001", "cho2017", "lee2019_mi", "physionetmi",
            "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016"]


def _rss_mib() -> float:
    """Peak RSS of this process. ``ru_maxrss`` is KiB on Linux, bytes on macOS."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((ru / 1024) if sys.platform != "darwin" else (ru / 2**20), 1)


# --------------------------------------------------------------------- worker
def worker(dataset: str, cache_path: str) -> dict:
    """Load one whole dataset the way the runner does, and report what it cost."""
    import moabb.datasets as mds
    import moabb.paradigms as mpar
    import torch  # noqa: F401
    from omegaconf import OmegaConf

    from utils import cache_config, set_data_dir

    # torch is imported and never used on purpose. The measured floor is what a tenant
    # costs to merely exist, and on the small datasets that floor is the whole story:
    # alexmi's epoched array is 44 MiB against a 700 MiB interpreter. A production
    # tenant imports torch, so a floor measured without it understates every cell by
    # the size of the torch runtime -- and it is the floor, multiplied by G*K, that
    # decides how many tenants fit. No CUDA context is created: nothing calls
    # torch.cuda here, so this stays a CPU-only probe.

    set_data_dir(None)
    dcfg = OmegaConf.load(CONFIG / "dataset" / f"{dataset}.yaml")
    ds = getattr(mds, dcfg.moabb_class)(**(OmegaConf.to_container(
        dcfg.get("kwargs")) or {}))
    if dcfg.get("subjects"):
        ds.subject_list = list(dcfg.subjects)
    # Same MOABB session off-by-one neutralised as in run_moabb_hydra.py: with the
    # filter left in place Lee2019 serves half its trials, and half the trials is
    # half the memory -- the measurement would be wrong in the dangerous direction.
    if getattr(ds, "_selected_sessions", None) is not None:
        ds._selected_sessions = None

    pkw = {}
    if dcfg.get("resample"):
        pkw["resample"] = float(dcfg.resample)
    if dcfg.get("tmin") is not None:
        pkw["tmin"] = float(dcfg.tmin)
    if dcfg.get("tmax") is not None:
        pkw["tmax"] = float(dcfg.tmax)
    par = getattr(mpar, dcfg.paradigm)(fmin=8, fmax=32, **pkw)
    ccfg = cache_config(OmegaConf.create({"enabled": True, "path": cache_path}))

    rss_before = _rss_mib()
    t0 = time.time()
    err = None
    X = y = None
    try:
        # No `subjects=`: this is the whole-dataset load the evaluation performs.
        X, y, _ = par.get_data(dataset=ds,
                               **({"cache_config": ccfg} if ccfg else {}))
    except MemoryError as exc:
        err = f"MemoryError: {str(exc)[:200]}"
    except Exception as exc:                      # a probe must report, not crash
        err = f"{type(exc).__name__}: {str(exc)[:200]}"

    rec = {
        "dataset": dataset,
        "n_subjects": len(ds.subject_list),
        "rss_before_mib": rss_before,
        "peak_rss_mib": _rss_mib(),
        "seconds": round(time.time() - t0, 1),
        "error": err,
    }
    if X is not None:
        rec.update({
            "n_trials": int(X.shape[0]),
            "n_chans": int(X.shape[1]),
            "n_times": int(X.shape[2]),
            "dtype": str(X.dtype),
            "array_mib": round(X.nbytes / 2**20, 1),
            "n_outputs": int(len(set(y))),
        })
    return rec


# --------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--out", default=str(HERE / "profile_mem" / "host_ram.json"))
    ap.add_argument("--cache", default="/scratch/amounir/moabb_cache")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    if args.worker:
        print("@@JSON@@" + json.dumps(worker(args.dataset, args.cache)))
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        done = {r["dataset"]: r for r in json.loads(out_path.read_text())}

    datasets = args.datasets.split(",")
    for i, ds in enumerate(datasets, 1):
        if ds in done:
            print(f"[{i}/{len(datasets)}] {ds}: cached", flush=True)
            continue
        p = subprocess.run(
            [sys.executable, __file__, "--worker", "--dataset", ds,
             "--cache", args.cache], capture_output=True, text=True)
        rec = None
        for line in p.stdout.splitlines():
            if line.startswith("@@JSON@@"):
                rec = json.loads(line[len("@@JSON@@"):])
        if rec is None:
            # Killed outright -- almost certainly by the OOM killer, which is itself
            # the answer: this dataset does not fit in the memory the probe was given.
            rec = {"dataset": ds, "error":
                   f"worker died rc={p.returncode}: {p.stderr.strip()[-300:]}"}
        done[ds] = rec
        out_path.write_text(json.dumps(list(done.values()), indent=2))
        print(f"[{i}/{len(datasets)}] {ds:19s} "
              f"peak_rss={rec.get('peak_rss_mib', '?')} MiB "
              f"array={rec.get('array_mib', '?')} MiB "
              f"trials={rec.get('n_trials', '?')} "
              f"dtype={rec.get('dtype', '?')} "
              f"{rec.get('seconds', '?')}s {rec.get('error') or ''}", flush=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
