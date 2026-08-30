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
from aligned_paradigm import make_aligned_paradigm  # noqa: E402
from pipelines import build_pipeline  # noqa: E402
from subject_stamp import stamped  # noqa: E402
from utils import (  # noqa: E402
    align_tag,
    cache_config,
    cell_stem,
    cap_cuda_fraction,
    logger,
    pick_device,
    provenance,
    results_path,
    set_data_dir,
    set_seed,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _align_tag(cfg) -> str:
    """Filename/suffix marker for the alignment arm ("" when raw).

    Delegates to ``utils.align_tag`` so the rule has exactly one definition: the grid
    generators emit the resulting stem into the TSV and ``pack_run.sh`` tests for it to
    decide a cell is done, so a second copy of this logic that drifted would make the
    packer re-run the campaign on top of itself in silence.
    """
    return align_tag(cfg.align)


def _make_evaluation(cfg, paradigm, dataset, hdf5_path, stamp=False):
    """Build the MOABB evaluation object named by ``cfg.eval.name``.

    ``stamp`` wraps the class so every fit record names its held-out subject (see
    ``subject_stamp.py``): MOABB owns the subject loop, so that identity is not
    reachable from a skorch callback and has to be injected on the evaluation side.
    Off for the ML arms, which write no fit records to stamp.
    """
    from moabb import evaluations as mev

    tag = _align_tag(cfg)
    common = dict(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=bool(cfg.overwrite),
        random_state=int(cfg.seed),
        n_jobs=int(cfg.n_jobs),
        hdf5_path=str(hdf5_path),
        # the tag also goes in MOABB's own cache suffix, so a cached raw result can
        # never be served for an aligned run
        suffix=f"{cfg.suffix}_{tag}" if tag else str(cfg.suffix),
    )
    cc = cache_config(cfg.get("cache"))
    if cc:
        common["cache_config"] = cc
    wrap = stamped if stamp else (lambda c: c)
    name = cfg.eval.name
    if name == "within_session":
        return wrap(mev.WithinSessionEvaluation)(
            n_splits=int(cfg.eval.n_splits), **common)
    if name == "cross_session":
        return wrap(mev.CrossSessionEvaluation)(**common)
    if name == "cross_subject":
        return wrap(mev.CrossSubjectEvaluation)(**common)
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

    # MOABB 1.5.0 silently drops sessions on the Lee2019 family. BaseDataset.get_data
    # filters sessions with {str(s) for s in self._selected_sessions} -- {'1','2'} for
    # Lee2019 -- while Lee2019._get_single_subject_data names its sessions in base 0
    # (session_name = str(session - 1)), i.e. {'0','1'}. The intersection is {'1'}, so
    # session '0' disappears without a warning even though n_sessions still says 2.
    # Consequences: CrossSessionEvaluation skips every subject ("Only one session
    # available") and within/cross_subject silently train on half the trials.
    # Audited over the 12 datasets of this grid: Lee2019_MI is the ONLY one with a
    # non-None _selected_sessions, so neutralising the filter is a no-op elsewhere.
    # A benchmark always wants every session, which is also MOABB's own default.
    if getattr(dataset, "_selected_sessions", None) is not None:
        logger.warning("dropping dataset._selected_sessions=%s (MOABB off-by-one on "
                       "session naming) so that all sessions are loaded",
                       dataset._selected_sessions)
        dataset._selected_sessions = None
    pkw = {}
    if dcfg.get("resample"):
        pkw["resample"] = float(dcfg.resample)
    # Fixed epoch window (tmin/tmax) when a dataset needs it. Without an explicit
    # tmax the MI paradigms fall back to each event's annotation duration, which is
    # not constant on some datasets (e.g. physionetmi) and breaks the array concat
    # in get_data. Passing the dataset native interval makes every epoch equal-length.
    if dcfg.get("tmin") is not None:
        pkw["tmin"] = float(dcfg.tmin)
    if dcfg.get("tmax") is not None:
        pkw["tmax"] = float(dcfg.tmax)
    # Trial alignment (align=euclidean) is a property of the *data*, not of the
    # estimator: it needs the subject ids, which only exist in the metadata frame
    # returned by get_data. So it is wired in as a paradigm subclass, not as a step
    # of the sklearn pipeline (see aligned_paradigm for the full argument).
    paradigm_cls = getattr(mpar, dcfg.paradigm)
    if cfg.align.name == "euclidean":
        paradigm_cls = make_aligned_paradigm(
            paradigm_cls, level=str(cfg.align.level),
            preserve_scale=bool(cfg.align.preserve_scale),
            rcond=float(cfg.align.rcond))
    elif cfg.align.name != "none":
        raise ValueError(f"unknown align.name: {cfg.align.name!r}")
    paradigm = paradigm_cls(
        fmin=float(cfg.paradigm.fmin), fmax=float(cfg.paradigm.fmax), **pkw)

    # ---- infer input dims once (on the first subject; cached afterwards) ----
    # Goes through the cache as well: without it this probe alone re-derives one
    # subject's epochs from the raw files in every single job of the grid.
    _cc = cache_config(cfg.get("cache"))
    X0, y0, _ = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]],
                                  **({"cache_config": _cc} if _cc else {}))
    n_chans, n_times = int(X0.shape[1]), int(X0.shape[2])
    n_outputs = int(len(set(y0)))
    sfreq = _infer_sfreq(cfg, dataset, paradigm, n_times)
    logger.info("dims: chans=%d times=%d outputs=%d sfreq=%.1f",
                n_chans, n_times, n_outputs, sfreq)

    device = pick_device(cfg.model)
    cap_cuda_fraction()
    out_dir = results_path(str(cfg.results_dir), cfg.eval.name, dcfg.name)
    logger.info("results -> %s", out_dir)
    # The alignment tag belongs in `stem` HERE, not further down where the CSV is
    # written, because `record_path` below is derived from it too. The raw and aligned
    # arms of the ablation are the same (eval, dataset, model, seed) point, so an
    # untagged stem makes the second arm overwrite the first -- silently, and for the
    # fit records as well as for the results.
    stem = cell_stem(label, _align_tag(cfg), cfg.seed)
    # One JSONL per cell: this process is its only writer, so appends from the
    # successive folds cannot interleave. Deep arms only -- the ML pipelines have
    # neither epochs nor a width. See eegrow.training.recording for why the growth
    # trajectory has to be written from inside the fit.
    record_path = (None if cfg.model.kind == "ml"
                   else out_dir / f"{stem}__fits.jsonl")
    pipeline = build_pipeline(
        OmegaConf.to_container(cfg.model, resolve=True),
        OmegaConf.to_container(cfg.train, resolve=True),
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=sfreq,
        device=device, seed=int(cfg.seed), record_path=record_path)
    logger.info("pipeline ready (device=%s)", device)

    # ---- evaluate ----------------------------------------------------------
    evaluation = _make_evaluation(cfg, paradigm, dataset, out_dir,
                                  stamp=record_path is not None)
    results = evaluation.process({label: pipeline})

    results["eval"] = cfg.eval.name
    results["model"] = label
    results["seed"] = int(cfg.seed)
    # The ablation arm, carried in the rows and not only in the filename: a column
    # survives a concat where a naming convention does not, and pairing raw against
    # aligned is the whole point of the arm.
    results["align"] = str(cfg.align.name)
    results["align_level"] = str(cfg.align.get("level") or "")
    # Regime + provenance ON THE ROW. `sfreq` is the rate the epochs were actually
    # served at, not the rate a config says they should have been: a pair split across
    # two rates measures preprocessing, not growth, and that is exactly what the
    # published grid could no longer rule out once its Hydra records were deleted.
    results["sfreq"] = float(sfreq)
    results["resample_cfg"] = (float(dcfg.resample) if dcfg.get("resample") else None)
    results["fmin"] = float(cfg.paradigm.fmin)
    results["fmax"] = float(cfg.paradigm.fmax)
    results["n_chans_in"] = n_chans
    results["n_times_in"] = n_times
    results["device"] = device
    for k, v in provenance().items():
        results[k] = v
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
    except Exception as e:
        # Log then re-raise: each sweep point is an isolated process (submitit array
        # task on SLURM, joblib worker locally), so failing loudly marks *this* job
        # failed without taking the others down -- and a swallowed error would leave a
        # silent hole in the results grid that is hard to spot afterwards.
        logger.error("job FAILED: %s", e)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run()
