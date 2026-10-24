"""Time one fold of one benchmark cell for a few epochs, to plan campaign A in GPU-hours.

A full cross-subject cell is up to tens of GPU-hours, so it cannot be its own probe.
This trains ONLY fold 0 of the protocol (cross_subject: every subject but the first;
cross_session: subject 1 on its sessions but the first; within_session: 80 % of subject
1's first session) for ``--epochs`` epochs, with the validation campaign's exact
pipeline (same compose, same dataset/paradigm build as run_moabb_hydra.main, same
protocol overrides as valid_battery.sbatch), and writes:

    n_train, wall seconds of the fit, seconds per training trial per epoch, peak GPU
    memory, GPU name and compute capability.

Cost of a cell is then extrapolated as  s_per_trial_epoch x n_train_fold x 200 x folds,
with the growth overhead taken from the full within-session validation cells (a 10-epoch
probe sees only the first, narrow, growth events).

Usage::

    python timing_probe.py --eval cross_subject --epochs 10 --out probe.json -- \
        model=bd_shallow dataset=lee2019_mi seed=0 train.lr=6.25e-4 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOCOL = ["train.selection_monitor=valid_acc", "train.smooth_window=5",
            "train.tie_break=earliest", "train.restore_selected=false",
            "train.save_selected=false", "train.growth_holdout=0.1",
            "train.lr_schedule=null"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=["within_session", "cross_session", "cross_subject"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()

    import moabb.datasets as mds
    import moabb.paradigms as mpar
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from pipelines import build_pipeline
    from utils import cache_config, cap_cuda_fraction, pick_device, set_data_dir, set_seed

    ov = list(a.overrides) + PROTOCOL + [f"train.max_epochs={a.epochs}", f"train.patience={a.epochs}"]
    with initialize_config_dir(config_dir=str(HERE / "config"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=ov)
    set_seed(int(cfg.seed))
    set_data_dir(cfg.get("data_dir"))
    dcfg = cfg.dataset
    dataset = getattr(mds, dcfg.moabb_class)(**(OmegaConf.to_container(dcfg.get("kwargs")) or {}))
    if dcfg.get("subjects"):
        dataset.subject_list = list(dcfg.subjects)
    if getattr(dataset, "_selected_sessions", None) is not None:
        dataset._selected_sessions = None
    pkw = {}
    if dcfg.get("resample"):
        pkw["resample"] = float(dcfg.resample)
    if dcfg.get("tmin") is not None:
        pkw["tmin"] = float(dcfg.tmin)
    if dcfg.get("tmax") is not None:
        pkw["tmax"] = float(dcfg.tmax)
    paradigm = getattr(mpar, dcfg.paradigm)(fmin=float(cfg.paradigm.fmin), fmax=float(cfg.paradigm.fmax), **pkw)
    cc = cache_config(cfg.get("cache"))
    kw = {"cache_config": cc} if cc else {}

    t_load = time.perf_counter()
    subjects = list(dataset.subject_list) if a.eval == "cross_subject" else [dataset.subject_list[0]]
    Xs, ys, metas = [], [], []
    for s in subjects:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[s], **kw)
        Xs.append(X); ys.append(y); metas.append(meta)
    t_load = time.perf_counter() - t_load
    X, y = np.concatenate(Xs), np.concatenate(ys)
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)  # as MOABB's evaluations do before fit
    import pandas as pd
    meta = pd.concat(metas, ignore_index=True)
    first = meta["subject"].iloc[0]
    sessions = sorted(meta["session"].unique())
    if a.eval == "cross_subject":
        mask = (meta["subject"] != first).to_numpy()
    elif a.eval == "cross_session":
        mask = (meta["session"] != sessions[0]).to_numpy()
    else:
        idx = np.flatnonzero((meta["session"] == sessions[0]).to_numpy())
        rng = np.random.default_rng(0)
        mask = np.zeros(len(meta), bool)
        mask[rng.permutation(idx)[: int(0.8 * len(idx))]] = True
    Xtr, ytr = X[mask], y[mask]
    n_chans, n_times = int(X.shape[1]), int(X.shape[2])
    sfreq = float(dcfg.resample) if dcfg.get("resample") else 250.0

    device = pick_device(cfg.model)
    cap_cuda_fraction()
    pipe = build_pipeline(OmegaConf.to_container(cfg.model, resolve=True),
                          OmegaConf.to_container(cfg.train, resolve=True),
                          n_chans=n_chans, n_times=n_times, n_outputs=int(len(set(y))), sfreq=sfreq,
                          device=device, seed=int(cfg.seed), record_path=None)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_s = time.perf_counter() - t0
    rec = {"eval": a.eval, "dataset": dcfg.name, "model": str(cfg.model.label), "seed": int(cfg.seed),
           "epochs": a.epochs, "n_train": int(mask.sum()), "n_chans": n_chans, "n_times": n_times,
           "n_subjects_loaded": len(subjects), "load_s": round(t_load, 1), "fit_s": round(fit_s, 2),
           "s_per_trial_epoch": fit_s / (int(mask.sum()) * a.epochs),
           "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
           "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else None,
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
           "node": os.environ.get("SLURMD_NODENAME"), "overrides": ov}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
