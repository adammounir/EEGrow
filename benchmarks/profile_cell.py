"""Measure where the wall time of one benchmark cell actually goes.

The production launcher gives a whole GPU to every ``(eval x dataset x model x
seed)`` point. Whether that is wasteful, and by how much, is not a matter of
opinion: it depends on three numbers this script measures on a real cell.

    preprocessing vs training   MOABB re-derives the epochs from the raw files on
                                every job. If that dominates, no amount of GPU
                                packing helps and the fix is ``cache_config``.
    peak GPU memory             sets how many processes can share one GPU (K).
    GPU utilisation             a device sitting at 10 % between two batches is a
                                device that can host other tenants.

Usage (one cell, same config names as ``run_moabb_hydra.py``)::

    python benchmarks/profile_cell.py model=grow_shallow dataset=bnci2014_001 \
        eval=within_session profile.cache=false

Set ``profile.cache=true`` to run the same cell through MOABB's on-disk epoch
cache, which is the whole point of the comparison: the second run of any cell in
a grid of 14 models x 5 seeds should not pay the preprocessing again.

Writes one JSON per run under ``benchmarks/profile/`` so that a cold run and a
warm run can be diffed afterwards rather than eyeballed from logs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipelines import build_pipeline  # noqa: E402
from utils import (  # noqa: E402
    cache_config,
    cap_cuda_fraction,
    logger,
    pick_device,
    results_path,
    set_data_dir,
    set_seed,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


class GpuSampler(threading.Thread):
    """Poll ``nvidia-smi`` for utilisation and memory on this process' device.

    Sampling from outside the process is deliberate: ``torch.cuda`` reports what
    the allocator holds, which is what *we* asked for, whereas the question here
    is what the device is actually doing -- including the idle gaps between
    batches that make co-tenancy profitable.
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.util: list[int] = []
        self.mem: list[int] = []
        self._stop_evt = threading.Event()
        # CUDA_VISIBLE_DEVICES renumbers devices for us but not for nvidia-smi.
        self.index = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]

    def run(self) -> None:
        # No --id filter. Under SLURM, CUDA_VISIBLE_DEVICES is not always the
        # small integer nvidia-smi expects -- some configurations set it to a
        # GPU UUID, and `--id=GPU-1c4f...` makes every call fail silently, which
        # is how the first profiling run came back with zero samples. Query
        # everything the job can see and keep the first row: a --gres=gpu:1 job
        # sees exactly one device, and the packed runner pins one device per
        # process through CUDA_VISIBLE_DEVICES anyway.
        query = ("nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits")
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(query, capture_output=True, text=True,
                                     timeout=5).stdout.strip()
                u, m = (int(v) for v in out.splitlines()[0].split(","))
                self.util.append(u)
                self.mem.append(m)
            except Exception:  # a missing/busy nvidia-smi must not kill the run
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> dict:
        self._stop_evt.set()
        self.join(timeout=5)
        if not self.util:
            return {"samples": 0}
        srt = sorted(self.util)
        return {
            "samples": len(self.util),
            "util_mean": round(sum(self.util) / len(self.util), 1),
            "util_median": srt[len(srt) // 2],
            "util_max": max(self.util),
            # Share of the run where the device is essentially idle. This is the
            # headroom a second tenant would fill.
            "util_below_20pct": round(sum(1 for u in self.util if u < 20)
                                      / len(self.util), 3),
            "mem_used_max_mib": max(self.mem),
        }


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run(cfg: DictConfig) -> None:
    import moabb.datasets as mds
    import moabb.paradigms as mpar
    import torch
    from moabb import evaluations as mev

    set_seed(int(cfg.seed))
    set_data_dir(cfg.get("data_dir"))
    dcfg = cfg.dataset
    label = str(cfg.model.label)
    use_cache = bool(cfg.cache.get("enabled"))
    logger.info("PROFILE eval=%s dataset=%s model=%s cache=%s",
                cfg.eval.name, dcfg.name, label, use_cache)

    t = {}
    t0 = time.perf_counter()

    dataset = getattr(mds, dcfg.moabb_class)(**(OmegaConf.to_container(
        dcfg.get("kwargs")) or {}))
    if dcfg.get("subjects"):
        dataset.subject_list = list(dcfg.subjects)
    if getattr(dataset, "_selected_sessions", None) is not None:
        dataset._selected_sessions = None
    if cfg.profile.get("n_subjects"):
        dataset.subject_list = dataset.subject_list[: int(cfg.profile.n_subjects)]
    n_subjects = len(dataset.subject_list)

    pkw = {}
    if dcfg.get("resample"):
        pkw["resample"] = float(dcfg.resample)
    if dcfg.get("tmin") is not None:
        pkw["tmin"] = float(dcfg.tmin)
    if dcfg.get("tmax") is not None:
        pkw["tmax"] = float(dcfg.tmax)
    paradigm = getattr(mpar, dcfg.paradigm)(
        fmin=float(cfg.paradigm.fmin), fmax=float(cfg.paradigm.fmax), **pkw)

    ccfg = cache_config(cfg.get("cache"))

    # ---- phase 1: preprocessing, every subject, nothing to do with the GPU ----
    # Timed on its own because it is the part a cache removes and the part that
    # is repeated identically by every model and every seed of the grid.
    tp = time.perf_counter()
    X0 = None
    for subj in dataset.subject_list:
        kw = {"cache_config": ccfg} if ccfg else {}
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subj], **kw)
        if X0 is None:
            X0, y0 = X, y
    t["preprocess_all_subjects_s"] = round(time.perf_counter() - tp, 2)

    n_chans, n_times = int(X0.shape[1]), int(X0.shape[2])
    n_outputs = int(len(set(y0)))
    sfreq = float(dcfg.resample) if dcfg.get("resample") else 250.0

    device = pick_device(cfg.model)
    cap_cuda_fraction()
    pipeline = build_pipeline(
        OmegaConf.to_container(cfg.model, resolve=True),
        OmegaConf.to_container(cfg.train, resolve=True),
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=sfreq,
        device=device, seed=int(cfg.seed))

    # ---- phase 2: the evaluation itself ------------------------------------
    # Preprocessing is now warm in the OS page cache (and in MOABB's cache when
    # enabled), so what this measures is dominated by the training.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()

    out_dir = results_path(str(cfg.results_dir), cfg.eval.name, dcfg.name)
    common = dict(paradigm=paradigm, datasets=[dataset], overwrite=True,
                  random_state=int(cfg.seed), n_jobs=int(cfg.n_jobs),
                  hdf5_path=str(out_dir), suffix=f"profile_{cfg.suffix}")
    if ccfg:
        common["cache_config"] = ccfg
    if cfg.eval.name == "within_session":
        ev = mev.WithinSessionEvaluation(n_splits=int(cfg.eval.n_splits), **common)
    elif cfg.eval.name == "cross_session":
        ev = mev.CrossSessionEvaluation(**common)
    else:
        ev = mev.CrossSubjectEvaluation(**common)

    te = time.perf_counter()
    results = ev.process({label: pipeline})
    t["evaluate_s"] = round(time.perf_counter() - te, 2)
    gpu = sampler.stop()
    t["total_s"] = round(time.perf_counter() - t0, 2)

    rec = {
        "eval": cfg.eval.name, "dataset": dcfg.name, "model": label,
        "seed": int(cfg.seed), "cache": use_cache, "device": str(device),
        "n_subjects": n_subjects, "n_chans": n_chans, "n_times": n_times,
        "sfreq": sfreq, "max_epochs": int(cfg.train.max_epochs),
        "batch_size": int(cfg.train.batch_size),
        "n_score_rows": len(results),
        "mean_score": round(float(results["score"].mean()), 4),
        "fit_time_sum_s": round(float(results["time"].sum()), 2),
        **t,
        "preprocess_share": round(t["preprocess_all_subjects_s"]
                                  / max(t["total_s"], 1e-9), 3),
        "gpu": gpu,
        "torch_peak_alloc_mib": (round(torch.cuda.max_memory_allocated() / 2**20)
                                 if torch.cuda.is_available() else None),
        "torch_peak_reserved_mib": (round(torch.cuda.max_memory_reserved() / 2**20)
                                    if torch.cuda.is_available() else None),
        "node": os.environ.get("SLURMD_NODENAME", "local"),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
    }

    dest = Path(cfg.profile.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stem = (f"{cfg.eval.name}__{dcfg.name}__{label}__seed{cfg.seed}"
            f"__cache{int(use_cache)}")
    (dest / f"{stem}.json").write_text(json.dumps(rec, indent=2))
    logger.info("PROFILE %s", json.dumps(rec, indent=2))


if __name__ == "__main__":
    run()
