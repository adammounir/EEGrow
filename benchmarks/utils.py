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
    # Falling back to CPU is right on a laptop and wrong under the packed runner,
    # which pins every deep cell to a card with CUDA_VISIBLE_DEVICES. If that
    # variable is set and CUDA is still unavailable, the device it names does not
    # exist -- and the cell would run ~20x slower on CPU while producing a
    # perfectly ordinary CSV that nothing downstream could tell apart.
    #
    # Not hypothetical: margpu021 advertises gpu:turing:3 to SLURM and carries
    # two cards. G came from SLURM_GPUS_ON_NODE, so a third of the tenants were
    # pinned to device 2, silently dropped to CPU, and turned an 8-minute
    # allocation into a 56-minute one with both real GPUs at 0 % utilisation.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} but "
            "torch.cuda.is_available() is False: the pinned device does not "
            "exist. Refusing to fall back to CPU.")
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cap_cuda_fraction() -> None:
    """Honour ``EEGROW_CUDA_FRACTION``: a per-process ceiling on the device.

    Only meaningful when several processes share one GPU. The caching allocator
    never hands a block back, so without a ceiling the first co-tenant to run a
    large batch keeps the card and its neighbours OOM at some arbitrary later
    point. Declaring the fraction turns that into a failure of the process that
    actually exceeded its share, at the moment it exceeds it.
    """
    frac = os.environ.get("EEGROW_CUDA_FRACTION")
    if not frac:
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac))
        logger.info("per-process CUDA memory fraction capped at %s", frac)


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


def cache_config(cfg) -> dict | None:
    """MOABB ``cache_config`` for the preprocessed epochs, or ``None`` when off.

    Why this matters more than any GPU tuning: the grid asks each dataset for the
    *same* epochs 70 times (14 pipelines x 5 seeds), and without a cache MOABB
    re-reads the raw files and redoes the band-pass, the resampling and the
    epoching every single time -- single-threaded, on the CPU, while the GPU
    waits. Caching the epochs makes the first job pay for the dataset and the
    other 69 read an array back.

    ``save_epochs`` and ``save_array`` are both on: the array is what
    ``get_data`` actually returns, and the epochs level is what a
    ``return_epochs`` evaluation would want. The raw level is *not* cached -- it
    would duplicate the 112 GB of ``MNE_DATA`` for no gain, since nothing here
    re-reads raws once the epochs exist.

    The ``overwrite_*`` flags stay False on purpose: a cache that rewrites itself
    on every hit is not a cache. Invalidating it is a deliberate act (delete the
    directory), because the cache key covers the preprocessing parameters and a
    silent overwrite is exactly how a mixed-preprocessing grid happens.
    """
    if cfg is None or not cfg.get("enabled"):
        return None
    cc = {"save_raw": False, "save_epochs": True, "save_array": True, "use": True,
          "overwrite_raw": False, "overwrite_epochs": False,
          "overwrite_array": False}
    if cfg.get("path"):
        cc["path"] = str(Path(str(cfg.path)).expanduser())
    logger.info("epoch cache ON -> %s", cc.get("path", "MNE_DATA default"))
    return cc


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
