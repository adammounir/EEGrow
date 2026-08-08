"""Small helpers shared by the Hydra benchmark: seeding, device, results IO."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("eegrow.benchmark")


def set_seed(seed: int, *, cuda: bool = True) -> None:
    """Seed python/numpy/torch (braindecode's helper if available)."""
    try:
        from braindecode.util import set_random_seeds
        set_random_seeds(seed=seed, cuda=cuda)
    except Exception:  # pragma: no cover - braindecode always present here
        import random

        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def pick_device(model_cfg) -> str:
    """ML baselines run on CPU; deep arms use CUDA when present, else CPU.

    MPS is deliberately *not* selected on the cluster path -- growth's ``eigh`` has no
    MPS kernel; on a GPU node CUDA is the right (and available) device anyway.
    """
    if model_cfg["kind"] == "ml":
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_data_dir(data_dir: str | None) -> None:
    """Point MNE/MOABB at a shared dataset cache (the cluster's titanic_1/datasets).

    No-op when ``data_dir`` is null -> MNE's default ``~/mne_data`` is used (local).
    """
    if not data_dir:
        return
    path = str(Path(data_dir).expanduser())
    os.environ["MNE_DATA"] = path
    os.environ.setdefault("MOABB_RESULTS", path)
    try:
        from mne import set_config
        set_config("MNE_DATA", path, set_env=True)
    except Exception:
        pass
    logger.info("dataset cache -> %s", path)


def default_results_root() -> Path:
    """``benchmarks/results``, located from this file rather than from the cwd.

    The config used to say ``${hydra:runtime.cwd}/benchmarks/results``, which makes the
    destination depend on the directory the process happens to be launched from. Under
    slurm that is not something the sbatch script controls: the same
    ``cd .../benchmarks`` before ``srun`` sent some jobs' results to
    ``benchmarks/results`` and others to ``benchmarks/benchmarks/results``. A whole arm
    of the grid landed in the second one, invisible to every analysis.

    This file sits in ``benchmarks/``, so its own location is the anchor. An explicit
    ``results_dir=`` on the command line still wins.
    """
    return Path(__file__).resolve().parent / "results"


def results_path(results_dir, eval_name: str, dataset: str) -> Path:
    """``<results_dir>/<eval>/<dataset>/`` (created); one folder per (eval, dataset)."""
    root = default_results_root() if results_dir in (None, "", "None", "null") \
        else Path(str(results_dir)).expanduser()
    p = root / eval_name / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p
