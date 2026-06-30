"""BNCI2014_001 benchmark through MOABB's evaluation framework: growable vs fixed.

Instead of a hand-rolled split + loop, this uses MOABB's
:class:`~moabb.evaluations.WithinSessionEvaluation` -- the standard, cross-validated
protocol of the library (per subject, per session, stratified k-fold). The three arms
are ordinary scikit-learn pipelines, each ending in a braindecode ``EEGClassifier``:

  * ``fixed-small``  : SCCNet frozen at ``start`` spatial filters (cheap baseline);
  * ``growable``     : SCCNet that grows ``start -> target`` via the ``GromoGrowth``
                       callback during ``fit`` (no width tuning);
  * ``fixed-target`` : SCCNet frozen at ``target`` (from-scratch oracle width).

Why MOABB's evaluation (and not our own split)? It is the field-standard estimator:
stratified k-fold within each session, aggregated over both sessions and all subjects,
so the numbers are comparable to the rest of the MOABB ecosystem and the variance is a
real cross-validation spread -- not a single lucky split. The module is passed to
``EEGClassifier`` as a **class** (``module=GrowingSCCNet``) with ``module__*`` params,
so every CV fold builds a *fresh* model (no state leaks across folds).

Run:  python examples/benchmark_moabb.py                       # subjects 1-9, 5-fold
      python examples/benchmark_moabb.py --subjects 1,2,3 --folds 3 --epochs 20
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from eegrow import GromoGrowth
from eegrow.models.growing_sccnet import GrowingSCCNet

warnings.filterwarnings("ignore")

ARMS = ("fixed-small", "growable", "fixed-target")


def to_float32(X):
    """MOABB serves float64 epochs; braindecode/skorch convs want float32."""
    return np.asarray(X, dtype="float32")


def parse_ints(spec: str) -> list[int]:
    """'1-9' -> [1..9]; '1,3,5' -> [1,3,5]; mixes allowed."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def make_pipeline(kind, *, n_chans, n_classes, n_times, sfreq, start, target,
                  epochs, grow_every, lr, batch_size, device):
    """One benchmark arm as a sklearn ``Pipeline`` (cast -> growable EEGClassifier).

    ``module`` is the *class* (+ ``module__*`` params) so each CV fold instantiates a
    fresh SCCNet. The growable arm gets the ``GromoGrowth`` callback and a target
    width; the fixed arms have no target (frozen) and no callback.
    """
    from braindecode import EEGClassifier

    width = {"fixed-small": start, "growable": start, "fixed-target": target}[kind]
    target_w = target if kind == "growable" else None  # None => frozen
    callbacks = ([("gromo", GromoGrowth(grow_every=grow_every, verbose=False))]
                 if kind == "growable" else [])

    clf = EEGClassifier(
        module=GrowingSCCNet,
        module__n_chans=n_chans,
        module__n_outputs=n_classes,
        module__n_times=n_times,
        module__sfreq=sfreq,
        module__n_spatial_filters=width,
        module__n_spatial_filters_smooth=16,
        module__target_n_spatial_filters=target_w,
        module__device=device,
        criterion=torch.nn.CrossEntropyLoss,  # our models output raw logits
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        train_split=None,  # MOABB owns the CV split; train on the whole fold
        iterator_train__shuffle=True,
        callbacks=callbacks,
        device=device,
        verbose=0,
    )
    return Pipeline([("cast", FunctionTransformer(to_float32)), ("clf", clf)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", default="1-9")
    p.add_argument("--folds", type=int, default=5, help="within-session CV folds")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--grow-every", type=int, default=5)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--target", type=int, default=16)
    p.add_argument("--lr", type=float, default=6.25e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="examples/results_bnci2014_001.md")
    args = p.parse_args()

    from braindecode.util import set_random_seeds
    from moabb.datasets import BNCI2014_001
    from moabb.evaluations import WithinSessionEvaluation
    from moabb.paradigms import MotorImagery

    set_random_seeds(seed=args.seed, cuda=False)
    subjects = parse_ints(args.subjects)

    dataset = BNCI2014_001()
    dataset.subject_list = subjects
    paradigm = MotorImagery()

    # Infer the input dims once (cached after the first download). BNCI2014_001 is
    # 250 Hz and MotorImagery does not resample by default, so sfreq is the native
    # rate (used by SCCNet to size its temporal pooling).
    X0, y0, _ = paradigm.get_data(dataset=dataset, subjects=subjects[:1])
    n_chans, n_times = X0.shape[1], X0.shape[2]
    n_classes = len(set(y0))
    sfreq = float(paradigm.resample) if paradigm.resample else 250.0
    print(f"BNCI2014_001: chans={n_chans} times={n_times} classes={n_classes} "
          f"sfreq={sfreq} | subjects={subjects} | {args.folds}-fold within-session")

    pipelines = {
        arm: make_pipeline(
            arm, n_chans=n_chans, n_classes=n_classes, n_times=n_times, sfreq=sfreq,
            start=args.start, target=args.target, epochs=args.epochs,
            grow_every=args.grow_every, lr=args.lr, batch_size=args.batch_size,
            device=args.device)
        for arm in ARMS
    }

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=[dataset], overwrite=True,
        random_state=args.seed, n_splits=args.folds, suffix="eegrow",
    )

    t0 = time.time()
    results = evaluation.process(pipelines)  # one row per (subject, session, pipeline)
    elapsed = time.time() - t0

    # ---- aggregate: per-subject mean over sessions, then mean +/- std over subjects
    # MOABB returns ``subject`` as a string, so compare on str (not the int from CLI).
    def subj_scores(arm):
        sub = results[results.pipeline == arm]
        return [float(sub[sub.subject.astype(str) == str(s)].score.mean())
                for s in subjects]

    per_arm = {arm: subj_scores(arm) for arm in ARMS}
    overall = {arm: (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)
               for arm, v in per_arm.items()}

    # ---- markdown report ----
    L = [
        "# BNCI2014_001 within-session benchmark (SCCNet)\n",
        f"MOABB `WithinSessionEvaluation` ({args.folds}-fold, both sessions), "
        f"4-class motor imagery, subjects {args.subjects}. Growable vs fixed "
        f"baselines, all via braindecode `EEGClassifier`. Width "
        f"{args.start}->{args.target}, {args.epochs} epochs, grow_every="
        f"{args.grow_every}. Score = accuracy.\n",
        "Per-subject accuracy (mean over sessions x CV folds):\n",
        "| subject | " + " | ".join(ARMS) + " |",
        "|" + "---|" * (len(ARMS) + 1),
    ]
    for i, s in enumerate(subjects):
        L.append(f"| S{s} | " + " | ".join(f"{per_arm[a][i]:.3f}" for a in ARMS)
                 + " |")
    L.append("| **mean (over subjects)** | " +
             " | ".join(f"**{overall[a][0]:.3f} ± {overall[a][1]:.3f}**"
                        for a in ARMS) + " |")
    L.append(f"\n_Spread is across subjects. {len(subjects)} subjects x 2 sessions x "
             f"{args.folds}-fold CV. Generated in {elapsed:.0f}s._\n")
    report = "\n".join(L) + "\n"

    with open(args.out, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"(written to {args.out})")


if __name__ == "__main__":
    main()
