"""Benchmark: growable models vs fixed baselines, trained via braindecode EEGClassifier.

Goal (per Bruno's request): show the growable models train end-to-end through the
*standard* braindecode/skorch API, and put quick numbers on the table.

We compare three classifiers, all trained the same way (``EEGClassifier.fit``):

  * ``growable``      : starts narrow (width = ``start``), grows towards ``target``;
  * ``fixed-small``   : frozen at ``start`` (the cheap baseline you would settle for
                        if you never tuned the width);
  * ``fixed-target``  : frozen at ``target`` (the "oracle" width, the cost of knowing
                        the right size up front).

The growth uses a *held-out* line search (the scaling factor is chosen on data the
new neurons did not fit) and a hard width cap, both from
:mod:`eegrow.training.skorch_integration`.

Models: ``--model sccnet`` (default) or ``--model eegnex`` -- the same integration
drives both, which is the point (the growth callback is model-agnostic).

Data:
  * default       : a small synthetic but **learnable** EEG set (each class = a
                    temporal frequency on a class-specific channel pair), so the
                    spatial width matters and the script runs in ~1 min on CPU.
  * ``--moabb``   : a real braindecode ``MOABBDataset`` (BNCI2014_001, motor imagery),
                    one subject. Requires a download the first time.

Run:  python examples/benchmark_eegclassifier.py
      python examples/benchmark_eegclassifier.py --model eegnex
      python examples/benchmark_eegclassifier.py --moabb --subject 1
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

from eegrow import GrowingEEGNeX, GrowingSCCNet
from eegrow.training.skorch_integration import make_eeg_classifier

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------- data
def make_synthetic(n=600, n_chans=16, n_times=128, sfreq=128.0, n_classes=4, seed=0):
    """Learnable synthetic EEG where *spatial width matters*.

    Each class is defined by BOTH a temporal frequency AND a distinct pair of source
    channels (a topography). Separating the classes needs several spatial filters: a
    narrow model captures only a couple of topographies. Signal is weak vs noise so
    the task is non-trivial. Returns (X float32, y int64, sfreq).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    y = rng.integers(0, n_classes, n).astype("int64")
    X = (rng.standard_normal((n, n_chans, n_times)) * 1.0).astype("float32")
    amp = 0.6
    for i in range(n):
        c = y[i]
        freq = 6 + 4 * c
        chans = [(2 * c) % n_chans, (2 * c + 1) % n_chans]
        X[i, chans] += amp * np.sin(2 * np.pi * freq * t).astype("float32")
    return X, y, sfreq


def load_moabb(subject=1):
    """Load one subject of BNCI2014_001 (motor imagery) as (X, y, sfreq) arrays.

    Standard braindecode pipeline: pick EEG, to microvolts, band-pass, exponential
    moving standardisation, then windows from events. Needs a download the first run.
    """
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
        Preprocessor,
        create_windows_from_events,
        exponential_moving_standardize,
        preprocess,
    )

    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject])
    low_hz, high_hz, factor = 4.0, 38.0, 1e6
    preprocess(ds, [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda d: np.multiply(d, factor)),
        Preprocessor("filter", l_freq=low_hz, h_freq=high_hz),
        Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000),
    ])
    sfreq = ds.datasets[0].raw.info["sfreq"]
    windows = create_windows_from_events(
        ds, trial_start_offset_samples=int(-0.5 * sfreq),
        trial_stop_offset_samples=0, preload=True,
    )
    X = np.stack([windows[i][0] for i in range(len(windows))]).astype("float32")
    y = np.asarray([windows[i][1] for i in range(len(windows))], dtype="int64")
    return X, y, float(sfreq)


def split(X, y, frac=0.75, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(frac * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


# ------------------------------------------------------------------- models
def build(family, kind, *, n_chans, n_classes, n_times, sfreq, start, target, device):
    """Build one benchmark arm of the chosen model family.

    kind is 'growable' (start->target), 'fixed-small' (frozen at start) or
    'fixed-target' (frozen at target). 'growable_width' maps to n_spatial_filters
    (SCCNet) or filter_1 (EEGNeX).
    """
    width = {"growable": start, "fixed-small": start, "fixed-target": target}[kind]
    tgt = target if kind == "growable" else None  # None => frozen
    torch.manual_seed(0)
    if family == "sccnet":
        return GrowingSCCNet(
            n_chans=n_chans, n_outputs=n_classes, n_times=n_times, sfreq=sfreq,
            n_spatial_filters=width, n_spatial_filters_smooth=16,
            target_n_spatial_filters=tgt, device=device,
        )
    return GrowingEEGNeX(
        n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
        filter_1=width, filter_2=16, target_filter_1=tgt, device=device,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sccnet", "eegnex"], default="sccnet")
    p.add_argument("--moabb", action="store_true", help="use real BNCI2014_001 data")
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--grow-every", type=int, default=8)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--target", type=int, default=24)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.moabb:
        print(f"loading BNCI2014_001 subject {args.subject} (may download)...")
        X, y, SF = load_moabb(args.subject)
    else:
        X, y, SF = make_synthetic()
    C, T, K = X.shape[1], X.shape[2], int(y.max()) + 1
    Xtr, ytr, Xte, yte = split(X, y)
    print(f"model={args.model}  data: train {Xtr.shape}  test {Xte.shape}  "
          f"classes={K} (chance={1/K:.2f})\n")

    rows = []
    for kind in ("fixed-small", "growable", "fixed-target"):
        model = build(args.model, kind, n_chans=C, n_classes=K, n_times=T, sfreq=SF,
                      start=args.start, target=args.target, device=args.device)
        clf = make_eeg_classifier(
            model, max_epochs=args.epochs, grow_every=args.grow_every,
            batch_size=32, device=args.device, verbose=(kind == "growable"),
        )
        clf.fit(Xtr, ytr)
        tr = float((clf.predict(Xtr) == ytr).mean())
        te = float((clf.predict(Xte) == yte).mean())
        params = sum(pp.numel() for pp in model.parameters())
        rows.append((kind, model.growable_width, params, tr, te))

    print("\n" + "=" * 60)
    print(f"{'arm':<14}{'width':>7}{'params':>10}{'train':>10}{'test':>9}")
    print("-" * 60)
    for kind, width, params, tr, te in rows:
        print(f"{kind:<14}{width:>7}{params:>10,}{tr:>10.3f}{te:>9.3f}")
    print("=" * 60)
    print(
        "Read it honestly: the growable model trains end-to-end through "
        "EEGClassifier and auto-sizes from `start` to `target` (no width tuning), "
        "with a held-out line search and a hard width cap. Whether it closes the gap "
        "to a from-scratch `fixed-target` is exactly what this benchmark measures -- "
        "compare the `test` column."
    )


if __name__ == "__main__":
    main()
