"""Stage 2: does training across datasets beat training on one?

MOABB has no cross-dataset evaluation -- ``CrossSubjectEvaluation`` loops over datasets
and never pools them -- so the protocol is here. Three arms, always scored on the *same*
held-out target subjects, so the only thing that differs between them is which trials
were available for training:

``within``  train on the target dataset's other subjects. The matched baseline. It is
            *not* the published single-dataset benchmark: that one ran BNCI2014_001 as
            4-class MotorImagery on 64 native channels, and here everything is 2-class
            left/right on 22 harmonised channels. Comparing across those three changes
            at once would mostly measure "2 classes are easier than 4", so the baseline
            is re-run under this protocol. That re-run is the point of this arm.
``lodo``    train on every *other* pool dataset; the target is never seen. Zero-shot
            transfer. Fold-independent by construction, so it is trained once and scored
            on every target subject -- the same model, N test sets.
``pooled``  the other datasets *plus* the target's training subjects. This is the arm
            that can actually beat ``within``, because it is the only one with strictly
            more data than the baseline. ``pooled`` vs ``within`` is the answer to
            "better than the first benchmark"; ``lodo`` vs ``within`` says whether the
            transfer works at all.

The alignment axis has three levels, not two
-------------------------------------------
Pooling raw datasets is confounded by amplitude: different amplifiers, references and
impedances put the datasets on different scales, so a single dataset can dominate the
loss for reasons that have nothing to do with EEG. Euclidean alignment removes that as a
side effect of whitening. If the comparison were only ``none`` vs ``euclidean``, a gain
would be unattributable -- whitening or plain rescaling? So ``scale`` (divide each
subject by its own global std, nothing else) sits between them as the control. Only
``euclidean`` minus ``scale`` is evidence about whitening.

    python benchmarks/cross_dataset.py --target bnci2014_001 --arm pooled \
        --model grow_shallow --align euclidean --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool as poolmod  # noqa: E402
from pipelines import build_pipeline  # noqa: E402
from utils import default_results_root, logger, pick_device, set_seed  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "config"
ARMS = ("within", "lodo", "pooled")


def results_root() -> Path:
    return default_results_root().parent / "results_cross_dataset"


def _model_cfg(name: str) -> dict:
    cfg = yaml.safe_load((CONFIG / "model" / f"{name}.yaml").read_text())
    return cfg


def _train_cfg(max_epochs: int | None) -> dict:
    root = yaml.safe_load((CONFIG / "config.yaml").read_text())
    t = dict(root["train"])
    if max_epochs is not None:
        t["max_epochs"] = max_epochs
    return t


def target_subjects(target: str) -> list[str]:
    d = poolmod.pool_root() / target
    return sorted((f.stem.split("sub-")[1] for f in d.glob("sub-*.npz")),
                  key=lambda s: (len(s), s))


def training_sets(arm: str, target: str, pool_names: list[str], held_out: list[str]):
    """Which (dataset, subject) pairs an arm may train on.

    Returns ``(names, exclude, subjects)`` for :func:`pool.load`. ``held_out`` is the
    list of target subjects under test, always excluded -- passed explicitly rather than
    filtered later, so a leak has to be written on purpose.
    """
    others = [n for n in pool_names if n != target]
    excl = [(target, s) for s in held_out]
    if arm == "within":
        return [target], excl, None
    if arm == "lodo":
        return others, [], None
    if arm == "pooled":
        return pool_names, excl, None
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def _fit_and_score(Xtr, ytr, tests, model_cfg, train_cfg, seed, device):
    """Train once, score on each ``(label, X, y)`` in ``tests``."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    pipe = build_pipeline(
        model_cfg, train_cfg,
        n_chans=Xtr.shape[1], n_times=Xtr.shape[2], n_outputs=2,
        sfreq=poolmod.SFREQ, device=device, seed=seed)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    fit_s = time.time() - t0

    rows = []
    for label, Xte, yte in tests:
        pred = pipe.predict(Xte)
        acc = float(accuracy_score(yte, pred))
        try:
            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(Xte)[:, 1]
            else:  # a decision_function-only estimator still ranks
                p = pipe.decision_function(Xte)
            auc = float(roc_auc_score(yte, p))
        except Exception as e:  # never lose the accuracy over a scoring detail
            logger.warning("auc unavailable for %s: %s", label, e)
            auc = float("nan")
        rows.append({"subject": label, "score": acc, "accuracy": acc, "auc": auc,
                     "n_test": int(len(yte)), "fit_seconds": fit_s})
        logger.info("  test %-24s acc=%.4f auc=%.4f (n=%d)", label, acc, auc, len(yte))
    return rows


def run(target: str, arm: str, model: str, align: str, seed: int, *,
        pool_tier: str = "core", max_epochs: int | None = None,
        folds: int | None = None) -> pd.DataFrame:
    set_seed(seed)
    pool_names = sorted(poolmod.tier(pool_tier))
    if target not in pool_names:
        raise ValueError(f"target {target!r} not in pool {pool_names}")
    subs = target_subjects(target)
    if not subs:
        raise FileNotFoundError(f"{target} not built -- run `pool.py build`")

    model_cfg = _model_cfg(model)
    train_cfg = _train_cfg(max_epochs)
    device = pick_device(model_cfg)
    logger.info("cross-dataset: target=%s arm=%s model=%s align=%s seed=%d "
                "pool=%s (%d datasets) device=%s",
                target, arm, model, align, seed, pool_tier, len(pool_names), device)

    # Leave-one-subject-out by default, matching MOABB's CrossSubjectEvaluation so the
    # `within` arm is directly comparable to the published grid's protocol. --folds
    # groups subjects instead, for targets where LOSO x pooled training is too costly.
    if folds and folds < len(subs):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(subs))
        groups = [[subs[i] for i in order[k::folds]] for k in range(folds)]
    else:
        groups = [[s] for s in subs]

    # The zero-shot arm's training set does not depend on the fold, so it is trained
    # once and scored on every subject. Retraining it per fold would burn N times the
    # compute on N identical models -- and, worse, N different random inits, making the
    # per-subject spread look like fold variance when it is seed variance.
    if arm == "lodo":
        groups = [[s for g in groups for s in g]]

    all_rows = []
    for fold, held in enumerate(groups):
        names, excl, restrict = training_sets(arm, target, pool_names, held)
        Xtr, ytr, gtr = poolmod.load(names, align=align, exclude=excl,
                                     subjects=restrict)
        tests = []
        for s in held:
            Xte, yte, _ = poolmod.load([target], align=align,
                                       subjects={target: [s]})
            tests.append((f"{target}|{s}", Xte, yte))
        logger.info("fold %d/%d: train %d trials / %d subjects, test %s",
                    fold + 1, len(groups), len(ytr), len(set(gtr)),
                    [t[0] for t in tests])

        leaked = {f"{target}|{s}" for s in held} & set(gtr)
        if leaked:  # cheap, and the one error that would silently invalidate everything
            raise RuntimeError(f"held-out subjects present in training: {leaked}")

        rows = _fit_and_score(Xtr, ytr, tests, model_cfg, train_cfg, seed, device)
        for r in rows:
            r.update(fold=fold, n_train_trials=int(len(ytr)),
                     n_train_subjects=int(len(set(gtr))))
        all_rows += rows
        del Xtr, ytr

    df = pd.DataFrame(all_rows)
    df["target"] = target
    df["arm"] = arm
    df["model"] = model_cfg["label"]
    df["align"] = align
    df["seed"] = int(seed)
    df["pool_tier"] = pool_tier
    df["pool_datasets"] = ",".join(pool_names)
    df["eval"] = "cross_dataset"
    df["sfreq"] = float(poolmod.SFREQ)
    df["n_times"] = int(poolmod.N_TIMES)
    df["n_chans"] = len(poolmod.SENSORIMOTOR_22)

    out = results_root() / target
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{model_cfg['label']}__{arm}__{align}__{pool_tier}__seed{seed}"
    df.to_csv(out / f"{stem}.csv", index=False)
    logger.info("saved %d rows -> %s", len(df), out / f"{stem}.csv")
    logger.info("mean acc = %.4f  mean auc = %.4f",
                df["accuracy"].mean(), df["auc"].mean())
    return df


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--model", required=True, help="a name under config/model/")
    ap.add_argument("--align", default="none",
                    choices=("none", "scale", "euclidean"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-tier", default="core",
                    help="core / core+interp / comma list -- the interpolation ablation")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--folds", type=int,
                    help="grouped K-fold over target subjects instead of LOSO")
    a = ap.parse_args(argv)
    try:
        run(a.target, a.arm, a.model, a.align, a.seed, pool_tier=a.pool_tier,
            max_epochs=a.max_epochs, folds=a.folds)
    except Exception as e:
        logger.error("run FAILED: %s", e)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
