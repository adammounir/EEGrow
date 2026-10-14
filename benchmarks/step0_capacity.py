"""Step-0 capacity criterion: can the net memorise 32 trials? (PREREG § 5.1)

WHY THIS EXISTS
---------------
A net that cannot drive its loss to ~0 on a tiny training set has a descent problem
(wrong init scale, a broken layer, an optimizer setting that stalls) and every
accuracy it produces downstream measures that problem, not the architecture. The
previous campaign mistook exactly this for a property of Deep4 ("the only net that
cannot fit its own training data", median final train loss 1.01) until a budget
change showed it was early stopping. Overfitting a small subset is the cheapest test
that separates the two, and it does not read any held-out accuracy.

PROTOCOL (frozen in the PREREG; nothing here is tuned on its outcome)
--------------------------------------------------------------------
* data: bnci2014_001, subject 1, session ``0train``, the benchmark's own MOABB
  paradigm (MotorImagery 8-32 Hz, resampled to 250 Hz), float32;
* 32 trials, 8 per class, drawn with ``numpy.random.RandomState(seed)`` per class;
* the arm's module, built exactly as ``pipelines._build_dl`` builds it, and the arm's
  candidate optimizer (AdamW, lr, weight decay, eps, batch size from the train
  config, constant lr);
* no validation split, no early stopping, no selection: 200 epochs, drop_last False
  (batch 64 > 32 trials, so one full-batch step per epoch);
* measure: cross-entropy of the final net in EVAL mode on the 32 trials (dropout off,
  BatchNorm on running statistics) -- the train-mode epoch loss of a dropout net is
  noisy by construction and would make the threshold a coin toss;
* refused if that loss is >= 0.05.

Growing arms (PREREG addendum 1, 14/10): the net grows during the test, with its own
``GromoGrowth`` built as ``pipelines._build_dl`` builds it (grow_every, wide grid,
min_singular_ratio). The line search needs trials it never trains on, so a growing arm
draws 10 trials per class instead of 8 and sets 1 per class aside as a disjoint
held-out (``grow_holdout = "disjoint"``): it trains on 36 trials and the measure is
the eval-mode loss on those 36. The ``bd`` path is unchanged (its results were read
before this addendum).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.05
N_PER_CLASS = 8
MAX_EPOCHS = 200


def balanced_subset(y, n_per_class, seed):
    """Indices of ``n_per_class`` trials per class, sorted, reproducible from seed."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        if len(pool) < n_per_class:
            raise ValueError(f"class {c} has {len(pool)} trials < {n_per_class}")
        idx.extend(rng.choice(pool, n_per_class, replace=False))
    return np.sort(np.asarray(idx))


N_PER_CLASS_GROWING = 10   # 9 trained on + 1 held out for the growth line search


class FixedHoldoutSplit:
    """First ``n_train`` samples train, the rest is the growth held-out; no validation.

    ``holdout`` is non-zero so ``GromoGrowth`` refuses to fall back to training batches
    if ``holdout_`` were ever missing.
    """

    holdout = 1.0

    def __init__(self, n_train):
        self.n_train = n_train

    def __call__(self, dataset, y=None, groups=None):
        import torch

        n = len(dataset)
        self.holdout_ = torch.utils.data.Subset(dataset, range(self.n_train, n))
        return torch.utils.data.Subset(dataset, range(self.n_train)), None


def growing_subset(y, seed):
    """(train indices, held-out indices): 10 per class, the first drawn of each held out."""
    rng = np.random.RandomState(seed)
    train, hold = [], []
    for c in np.unique(y):
        pick = rng.choice(np.flatnonzero(y == c), N_PER_CLASS_GROWING, replace=False)
        hold.append(pick[0])
        train.extend(pick[1:])
    return np.sort(np.asarray(train)), np.sort(np.asarray(hold))


def capacity_fit(model_cfg, train_cfg, X, y, *, seed, max_epochs=MAX_EPOCHS,
                 device="cpu", X_hold=None, y_hold=None):
    """Fit the arm's module on (X, y) with no validation; return the measures.

    Growing arms also take the growth held-out (``X_hold``, ``y_hold``).
    """
    import torch
    from braindecode import EEGClassifier
    from braindecode.util import set_random_seeds

    if train_cfg.get("lr_schedule"):
        raise ValueError("step 0 freezes a constant lr (train.lr_schedule=null)")
    module_params = {"module__n_chans": X.shape[1], "module__n_outputs": len(np.unique(y)),
                     "module__n_times": X.shape[2]}
    for k, v in (model_cfg.get("module_kwargs") or {}).items():
        module_params[f"module__{k}"] = v
    callbacks, train_split, X_fit, y_fit = [], None, X, y
    if model_cfg["kind"] == "bd":
        import braindecode.models as bdm
        module = getattr(bdm, model_cfg["arch"])
    elif model_cfg["kind"] == "growing":
        import eegrow
        from eegrow import GromoGrowth
        from eegrow.training.loop import MIN_SINGULAR_RATIO

        if X_hold is None or len(np.unique(y_hold)) != len(np.unique(y)):
            raise ValueError("a growing arm needs a held-out holding every class")
        module = getattr(eegrow, model_cfg["arch"])
        module_params["module__device"] = device
        callbacks = [("gromo", GromoGrowth(
            grow_every=model_cfg["grow_every"],
            scaling_grid=train_cfg.get("scaling_grid"),
            normalization=train_cfg.get("growth_normalization"),
            min_singular_ratio=float(model_cfg.get("min_singular_ratio",
                                                   MIN_SINGULAR_RATIO)),
            verbose=False))]
        train_split = FixedHoldoutSplit(len(X))
        X_fit, y_fit = np.concatenate([X, X_hold]), np.concatenate([y, y_hold])
    else:
        raise NotImplementedError(f"kind={model_cfg['kind']!r}")

    set_random_seeds(seed, cuda=str(device).startswith("cuda"))
    net = EEGClassifier(
        module=module, **module_params,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(train_cfg["lr"]),
        optimizer__eps=float(train_cfg.get("optimizer_eps", 1e-8)),
        optimizer__weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        batch_size=int(train_cfg["batch_size"]),
        iterator_train__drop_last=False,
        train_split=train_split, max_epochs=max_epochs, callbacks=callbacks,
        device=device, classes=list(range(len(np.unique(y)))), verbose=0)
    net.fit(X_fit, y_fit)
    net.module_.eval()
    with torch.no_grad():
        logits = net.module_(torch.as_tensor(X, device=net.device))
        loss = torch.nn.functional.cross_entropy(
            logits, torch.as_tensor(y, device=net.device)).item()
        acc = float((logits.argmax(1).cpu().numpy() == y).mean())
    curve = [float(h["train_loss"]) for h in net.history]
    growth = {}
    if model_cfg["kind"] == "growing":
        events = [h for h in net.history if "grow_s" in h]
        growth = {"width_start": int(net.history[0].get("width", -1))
                  if "width" in net.history[0] else None,
                  "width_end": int(net.module_.growable_width),
                  "target_width": int(net.module_.target_width),
                  "grow_holdout": sorted({str(h.get("grow_holdout")) for h in events}),
                  "n_events": len(events)}
    return {"final_loss_eval": loss, "final_acc_eval": acc, **growth,
            "train_loss_curve": curve, "first_epoch_train_loss": curve[0],
            "passes": bool(math.isfinite(loss) and loss < THRESHOLD)}


def load_subject_session(subject=1, session="0train"):
    """The benchmark's own data path: MOABB paradigm as in run_moabb_hydra."""
    import yaml
    import moabb.datasets as mds
    import moabb.paradigms as mpar

    root = yaml.safe_load((HERE / "config" / "config.yaml").read_text())
    dcfg = yaml.safe_load((HERE / "config" / "dataset" / "bnci2014_001.yaml").read_text())
    dataset = getattr(mds, dcfg["moabb_class"])()
    paradigm = getattr(mpar, dcfg["paradigm"])(
        fmin=float(root["paradigm"]["fmin"]), fmax=float(root["paradigm"]["fmax"]),
        resample=float(dcfg["resample"]))
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    keep = (meta["session"].astype(str) == session).to_numpy()
    classes = sorted(set(labels))
    y = np.array([classes.index(v) for v in labels])[keep]
    return X[keep].astype("float32"), y.astype("int64"), classes


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--candidate", required=True, help="label of the optimizer candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                    help="growing arms: line-search grid (addendum 2 widens it)")
    ap.add_argument("--normalization", default=None,
                    help="growing arms: growth_normalization (addendum 2)")
    ap.add_argument("--out", required=True, help="JSONL, one line appended per run")
    a = ap.parse_args(argv)

    model_cfg = yaml.safe_load((HERE / "config" / "model" / f"{a.arm}.yaml").read_text())
    train_cfg = yaml.safe_load((HERE / "config" / "config.yaml").read_text())["train"]
    train_cfg.update(lr=a.lr, weight_decay=a.weight_decay, lr_schedule=None)
    X, y, classes = load_subject_session()
    hold = None
    if model_cfg["kind"] == "growing":
        # Same line-search grid as the battery (PREREG § 3.1).
        train_cfg.update(scaling_grid=list(a.grid), growth_normalization=a.normalization)
        idx, hold = growing_subset(y, a.seed)
        res = capacity_fit(model_cfg, train_cfg, X[idx], y[idx], seed=a.seed,
                           device=a.device, X_hold=X[hold], y_hold=y[hold])
    else:
        idx = balanced_subset(y, N_PER_CLASS, a.seed)
        res = capacity_fit(model_cfg, train_cfg, X[idx], y[idx], seed=a.seed,
                           device=a.device)
    rec = {"arm": a.arm, "candidate": a.candidate, "seed": a.seed, "lr": a.lr,
           "weight_decay": a.weight_decay, "subject": 1, "session": "0train",
           "trial_indices": idx.tolist(), "n_trials": int(len(idx)),
           "holdout_indices": None if hold is None else hold.tolist(),
           "grid": list(a.grid) if hold is not None else None,
           "normalization": a.normalization if hold is not None else None,
           "threshold": THRESHOLD, **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{a.arm} [{a.candidate}] final eval loss {res['final_loss_eval']:.4f} "
          f"-> {'pass' if res['passes'] else 'REFUSED'}")
    return 0 if res["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
