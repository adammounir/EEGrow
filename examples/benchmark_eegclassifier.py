"""Benchmark: growable models vs fixed baselines, trained via braindecode EEGClassifier.

Goal (per Bruno's request): show the growable models train end-to-end through the
*standard* braindecode/skorch API, and put quick numbers on the table.

We compare three classifiers, all trained the same way (``EEGClassifier.fit``):

  * ``growable``      : starts narrow (width = ``start``), grows towards ``target``;
  * ``fixed-small``   : frozen at ``start`` (the cheap baseline you would settle for
                        if you never tuned the width);
  * ``fixed-target``  : frozen at ``target`` (the "oracle" width, the cost of knowing
                        the right size up front).

The point of growth: approach the ``fixed-target`` accuracy while spending most of
training at a smaller width (cheaper) and *discovering* the width automatically.

Data: a small synthetic but **learnable** EEG set (each class is a distinct temporal
frequency injected on a few channels), so accuracy is meaningful and the script runs
in ~1 min on CPU with no download. Swap in a real braindecode ``MOABBDataset`` /
windowing pipeline where indicated to benchmark on real data.

Run:  python examples/benchmark_eegclassifier.py
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

from eegrow import GrowingSCCNet
from eegrow.training.skorch_integration import make_eeg_classifier

warnings.filterwarnings("ignore")


def make_dataset(n=600, n_chans=16, n_times=128, sfreq=128.0, n_classes=4, seed=0):
    """Learnable synthetic EEG where *spatial width matters*.

    Each class is defined by BOTH a temporal frequency AND a distinct pair of source
    channels (a topography). Separating the classes therefore needs several spatial
    filters: a narrow model can only capture a couple of topographies. The signal is
    weak relative to the noise so the task is non-trivial. Returns float32/int64.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    y = rng.integers(0, n_classes, n).astype("int64")
    X = (rng.standard_normal((n, n_chans, n_times)) * 1.0).astype("float32")
    amp = 0.6
    for i in range(n):
        c = y[i]
        freq = 6 + 4 * c
        chans = [(2 * c) % n_chans, (2 * c + 1) % n_chans]  # class-specific topography
        X[i, chans] += amp * np.sin(2 * np.pi * freq * t).astype("float32")
    return X, y


def split(X, y, frac=0.75, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(frac * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def build(kind, *, n_chans, n_classes, n_times, sfreq, start, target, device):
    """One GrowingSCCNet per benchmark arm (growable / fixed-small / fixed-target)."""
    width = {"growable": start, "fixed-small": start, "fixed-target": target}[kind]
    tgt = target if kind == "growable" else None  # None => frozen
    torch.manual_seed(0)
    return GrowingSCCNet(
        n_chans=n_chans, n_outputs=n_classes, n_times=n_times, sfreq=sfreq,
        n_spatial_filters=width, n_spatial_filters_smooth=16,
        target_n_spatial_filters=tgt, device=device,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--grow-every", type=int, default=8)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--target", type=int, default=24)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    C, T, SF, K = 16, 128, 128.0, 4
    X, y = make_dataset(n_chans=C, n_times=T, sfreq=SF, n_classes=K)
    Xtr, ytr, Xte, yte = split(X, y)
    print(f"data: train {Xtr.shape}  test {Xte.shape}  classes={K} (chance={1/K:.2f})\n")

    rows = []
    for kind in ("fixed-small", "growable", "fixed-target"):
        model = build(kind, n_chans=C, n_classes=K, n_times=T, sfreq=SF,
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
        "EEGClassifier and auto-sizes from `start` to `target` (no width tuning). "
        "On this synthetic task it fits the train set but a from-scratch "
        "`fixed-target` still generalises best -- gromo's line search greedily "
        "minimises the *train* loss, so closing that gap (held-out line search / "
        "growth regularisation) and a real-data (MOABB) benchmark are the next steps."
    )


if __name__ == "__main__":
    main()
