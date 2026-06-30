"""One MOABB benchmark job, configured by Hydra.

A *single* job is one ``(eval x dataset x model x seed)`` point: it builds the
pipeline, runs the chosen MOABB evaluation on the chosen dataset, and dumps the
result rows. Parallelism lives at the **sweep** level -- ``--multirun`` expands the
cartesian product and a launcher (joblib locally, submitit/SLURM on Margaret) runs
the jobs concurrently, one model per process. That is Bruno's "parallelise at the
pipeline level": no model loops inside the script.

Local smoke test (1 job, tiny training)::

    python benchmarks/run_moabb_hydra.py model=grow_sccnet dataset=bnci2014_001 \
        eval=within_session train.max_epochs=2

Local parallel sweep (every model, both protocols, 3 seeds)::

    python benchmarks/run_moabb_hydra.py -m hydra/launcher=joblib \
        model=glob(*) dataset=bnci2014_001 eval=within_session,cross_subject \
        seed=0,1,2

Cluster sweep (Margaret, SLURM ``tau`` partition)::

    python benchmarks/run_moabb_hydra.py -m hydra/launcher=tau \
        model=glob(*) dataset=glob(*) eval=within_session,cross_subject seed=0,1,2
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipelines import build_pipeline  # noqa: E402
from utils import (  # noqa: E402
    logger,
    pick_device,
    results_path,
    set_data_dir,
    set_seed,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _make_evaluation(cfg, paradigm, dataset, hdf5_path):
    """Build the MOABB evaluation object named by ``cfg.eval.name``."""
    from moabb import evaluations as mev

    common = dict(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=bool(cfg.overwrite),
        random_state=int(cfg.seed),
        n_jobs=int(cfg.n_jobs),
        hdf5_path=str(hdf5_path),
        suffix=str(cfg.suffix),
    )
    name = cfg.eval.name
    if name == "within_session":
        return mev.WithinSessionEvaluation(n_splits=int(cfg.eval.n_splits), **common)
    if name == "cross_session":
        return mev.CrossSessionEvaluation(**common)
    if name == "cross_subject":
        return mev.CrossSubjectEvaluation(**common)
    raise ValueError(f"unknown eval.name: {name!r}")


def _infer_sfreq(cfg, dataset, paradigm, n_times: int) -> float:
    """Sampling rate of the served epochs.

    Order: explicit ``paradigm.resample`` > explicit ``dataset.sfreq`` (override) >
    derived from the epoch length. Deriving ``n_times / interval_duration`` avoids
    hard-coding a rate per dataset (12+ of them); MOABB's ``dataset.interval`` is the
    epoch window in seconds, so the native rate is the sample count over that window.
    """
    if getattr(paradigm, "resample", None):
        return float(paradigm.resample)
    if cfg.dataset.get("sfreq"):
        return float(cfg.dataset.sfreq)
    interval = getattr(dataset, "interval", None)
    if interval and (interval[1] - interval[0]) > 0:
        return float(round(n_times / (interval[1] - interval[0])))
    return 250.0


def main(cfg: DictConfig) -> pd.DataFrame:
    import moabb.datasets as mds
    import moabb.paradigms as mpar

    set_seed(int(cfg.seed))
    set_data_dir(cfg.get("data_dir"))

    dcfg = cfg.dataset
    label = str(cfg.model.label)
    logger.info("job: eval=%s dataset=%s model=%s seed=%s",
                cfg.eval.name, dcfg.name, label, cfg.seed)

    # ---- dataset + paradigm ------------------------------------------------
    dataset = getattr(mds, dcfg.moabb_class)(**(OmegaConf.to_container(
        dcfg.get("kwargs")) or {}))
    if dcfg.get("subjects"):
        dataset.subject_list = list(dcfg.subjects)
    pkw = {}
    if dcfg.get("resample"):
        pkw["resample"] = float(dcfg.resample)
    paradigm = getattr(mpar, dcfg.paradigm)(
        fmin=float(cfg.paradigm.fmin), fmax=float(cfg.paradigm.fmax), **pkw)

    # ---- infer input dims once (on the first subject; cached afterwards) ----
    X0, y0, _ = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]])
    n_chans, n_times = int(X0.shape[1]), int(X0.shape[2])
    n_outputs = int(len(set(y0)))
    sfreq = _infer_sfreq(cfg, dataset, paradigm, n_times)
    logger.info("dims: chans=%d times=%d outputs=%d sfreq=%.1f",
                n_chans, n_times, n_outputs, sfreq)

    device = pick_device(cfg.model)
    pipeline = build_pipeline(
        OmegaConf.to_container(cfg.model, resolve=True),
        OmegaConf.to_container(cfg.train, resolve=True),
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=sfreq,
        device=device, seed=int(cfg.seed))
    logger.info("pipeline ready (device=%s)", device)

    # ---- evaluate ----------------------------------------------------------
    out_dir = results_path(str(cfg.results_dir), cfg.eval.name, dcfg.name)
    evaluation = _make_evaluation(cfg, paradigm, dataset, out_dir)
    results = evaluation.process({label: pipeline})

    results["eval"] = cfg.eval.name
    results["model"] = label
    results["seed"] = int(cfg.seed)
    stem = f"{label}__seed{cfg.seed}"
    results.to_csv(out_dir / f"{stem}.csv", index=False)
    joblib.dump(results, out_dir / f"{stem}.joblib")
    logger.info("saved %d rows -> %s", len(results), out_dir / f"{stem}.csv")
    logger.info("mean score = %.4f", float(results["score"].mean()))
    return results


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run(cfg: DictConfig) -> None:
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    try:
        main(cfg)
        logger.info("job finished")
    except Exception as e:  # one failed job must not kill a sweep
        logger.error("job FAILED: %s", e)
        traceback.print_exc()


if __name__ == "__main__":
    run()
