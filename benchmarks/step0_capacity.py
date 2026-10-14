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

Growing arms enter by addendum: the capacity test of a net that changes width needs
its growth callback and a held-out, and this script refuses them rather than testing
a frozen-width stand-in under their name.
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


def capacity_fit(model_cfg, train_cfg, X, y, *, seed, max_epochs=MAX_EPOCHS,
                 device="cpu"):
    """Fit the arm's module on (X, y) with no validation; return the measures."""
    import torch
    from braindecode import EEGClassifier
    from braindecode.util import set_random_seeds

    if model_cfg["kind"] != "bd":
        raise NotImplementedError(
            f"kind={model_cfg['kind']!r}: growing arms enter by addendum with their "
            "growth callback; a frozen-width stand-in would test another net")
    if train_cfg.get("lr_schedule"):
        raise ValueError("step 0 freezes a constant lr (train.lr_schedule=null)")
    import braindecode.models as bdm

    module = getattr(bdm, model_cfg["arch"])
    module_params = {"module__n_chans": X.shape[1], "module__n_outputs": len(np.unique(y)),
                     "module__n_times": X.shape[2]}
    for k, v in (model_cfg.get("module_kwargs") or {}).items():
        module_params[f"module__{k}"] = v

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
        train_split=None, max_epochs=max_epochs, callbacks=[],
        device=device, classes=list(range(len(np.unique(y)))), verbose=0)
    net.fit(X, y)
    net.module_.eval()
    with torch.no_grad():
        logits = net.module_(torch.as_tensor(X, device=net.device))
        loss = torch.nn.functional.cross_entropy(
            logits, torch.as_tensor(y, device=net.device)).item()
        acc = float((logits.argmax(1).cpu().numpy() == y).mean())
    curve = [float(h["train_loss"]) for h in net.history]
    return {"final_loss_eval": loss, "final_acc_eval": acc,
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
    ap.add_argument("--out", required=True, help="JSONL, one line appended per run")
    a = ap.parse_args(argv)

    model_cfg = yaml.safe_load((HERE / "config" / "model" / f"{a.arm}.yaml").read_text())
    train_cfg = yaml.safe_load((HERE / "config" / "config.yaml").read_text())["train"]
    train_cfg.update(lr=a.lr, weight_decay=a.weight_decay, lr_schedule=None)
    X, y, classes = load_subject_session()
    idx = balanced_subset(y, N_PER_CLASS, a.seed)
    res = capacity_fit(model_cfg, train_cfg, X[idx], y[idx], seed=a.seed,
                       device=a.device)
    rec = {"arm": a.arm, "candidate": a.candidate, "seed": a.seed, "lr": a.lr,
           "weight_decay": a.weight_decay, "subject": 1, "session": "0train",
           "trial_indices": idx.tolist(), "n_trials": int(len(idx)),
           "threshold": THRESHOLD, **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{a.arm} [{a.candidate}] final eval loss {res['final_loss_eval']:.4f} "
          f"-> {'pass' if res['passes'] else 'REFUSED'}")
    return 0 if res["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
