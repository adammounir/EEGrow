"""Why is every cross-dataset accuracy exactly 0.5000?

One fold, one model, three input scalings, training curve printed. The point is to
separate two hypotheses that produce the same symptom (a constant prediction):

* **numerical**: MOABB serves volts, so the arrays have std ~5e-6. A convnet whose first
  layers are linear sees activations at 1e-6, and ShallowFBCSPNet then *squares* them
  (1e-12) before taking a log -- a regime where the gradient is numerically dead. If this
  is the cause, multiplying by a constant fixes it, and the constant does not matter.
* **statistical**: cross-subject 2-class left/right really is at chance with 8 training
  subjects and this recipe. Then no scaling helps and the grid's design is what needs
  revisiting, not its arithmetic.

A third possibility the curve also settles: early stopping (patience 20 on valid_acc)
firing before the net leaves its initial plateau, which would be a recipe bug rather
than either of the above.

    python benchmarks/diag_scale.py --model bd_shallow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool as poolmod  # noqa: E402


def one_fold(target="bnci2014_001", held="1"):
    subs = [f.stem.split("sub-")[1] for f in (poolmod.pool_root() / target).glob("sub-*.npz")]
    tr = sorted(s for s in subs if s != held)
    Xtr, ytr, _ = poolmod.load([target], subjects={target: tr})
    Xte, yte, _ = poolmod.load([target], subjects={target: [held]})
    return Xtr, ytr, Xte, yte


def fit_report(Xtr, ytr, Xte, yte, *, tag, model, epochs, lr):
    import torch
    from braindecode import EEGClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    if model == "bd_shallow":
        from braindecode.models import ShallowFBCSPNet as M
        kw = {}
    else:
        from eegrow import GrowingShallowFBCSPNet as M
        kw = {"module__n_filters_time": 8, "module__n_filters_spat": 40,
              "module__target_n_filters_time": 40, "module__device": "cuda"}

    clf = EEGClassifier(
        module=M, module__n_chans=Xtr.shape[1], module__n_outputs=2,
        module__n_times=Xtr.shape[2], **kw,
        criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
        optimizer__lr=lr, max_epochs=epochs, batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        classes=[0, 1], verbose=0,
    )
    clf.fit(Xtr.astype("float32"), ytr)
    hist = clf.history
    tr_loss = [h["train_loss"] for h in hist]
    va_loss = [h["valid_loss"] for h in hist]
    pred = clf.predict(Xte.astype("float32"))
    proba = clf.predict_proba(Xte.astype("float32"))[:, 1]
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    # a constant prediction is the signature we are hunting; report it explicitly
    frac1 = float(np.mean(pred == 1))
    print(f"[{tag:22s}] std_in={Xtr.std():.3e} epochs_run={len(hist)} "
          f"train_loss {tr_loss[0]:.4f}->{tr_loss[-1]:.4f} "
          f"valid_loss {va_loss[0]:.4f}->{va_loss[-1]:.4f} "
          f"acc={acc:.4f} auc={auc:.4f} frac_pred_1={frac1:.3f}")
    return acc, auc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bd_shallow")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=6.25e-4)
    ap.add_argument("--held", default="1")
    a = ap.parse_args(argv)

    Xtr, ytr, Xte, yte = one_fold(held=a.held)
    print(f"train {Xtr.shape} test {Xte.shape} "
          f"balance train={np.bincount(ytr)} test={np.bincount(yte)}")

    # The three candidate scalings. If the numerical hypothesis is right, x1e6 and
    # unit-variance both work and x1 does not; the exact constant is irrelevant.
    variants = {
        "volts (as cached)": lambda X: X,
        "x1e6 (microvolts)": lambda X: X * 1e6,
        "unit global std": lambda X: X / Xtr.std(),
    }
    for tag, f in variants.items():
        fit_report(f(Xtr), ytr, f(Xte), yte, tag=tag, model=a.model,
                   epochs=a.epochs, lr=a.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
