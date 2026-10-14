"""Live check of smoothed selection (RestoreBestModel(smooth_window=5)) on real fits.

bnci2014_001, within_session, the benchmark's own pipeline (`pipelines._build_dl`) with
patience=200, selection_monitor=valid_acc, smooth_window=5. One extra callback scores
the fold's held-out TEST set at the end of every epoch, so the test accuracy of the
epoch picked by any rule can be read off one fit.

Criteria (fixed before running):
  A fidelity  restored_epoch == offline sm5 pick on the recorded valid_acc curve, and
              the returned model's test accuracy == test_acc recorded at that epoch.
  B edges     < 20 % of fits scored before epoch 5, < 20 % in the last 5 epochs.
  C utility   test acc at the sm5 pick not below orig/late (subject = unit, bootstrap).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EEGROW = Path("/Users/adammounir/Desktop/Inria/Exploration/eegrow")
sys.path.insert(0, str(EEGROW / "benchmarks"))

import torch  # noqa: E402
import yaml  # noqa: E402
from moabb.datasets import BNCI2014_001  # noqa: E402
from moabb.paradigms import MotorImagery  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from skorch.callbacks import Callback  # noqa: E402

from pipelines import _build_dl  # noqa: E402


class TestAcc(Callback):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def on_epoch_end(self, net, **kwargs):
        pred = net.predict(self.X)
        net.history.record("test_acc", float((pred == self.y).mean()))


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="grow_shallow")
ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--max-epochs", type=int, default=200)
ap.add_argument("--max-fits", type=int, default=None)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

cfg = yaml.safe_load(open(EEGROW / "benchmarks/config/config.yaml"))
train_cfg = dict(cfg["train"], patience=200, selection_monitor="valid_acc",
                 smooth_window=5, max_epochs=args.max_epochs)
model_cfg = yaml.safe_load(open(EEGROW / f"benchmarks/config/model/{args.model}.yaml"))

paradigm = MotorImagery(fmin=8, fmax=32, resample=250.0)
dataset = BNCI2014_001()
done = set()
if args.out.exists():
    for line in open(args.out):
        r = json.loads(line)
        done.add((r["subject"], r["session"], r["fold"]))
n_fits = 0
for subject in args.subjects:
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    y = LabelEncoder().fit_transform(y)
    for session in sorted(meta.session.unique()):
        idx = np.where(meta.session.to_numpy() == session)[0]
        Xs, ys = X[idx].astype("float32"), y[idx]
        cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
        for fold, (tr, te) in enumerate(cv.split(Xs, ys)):
            if (subject, str(session), fold) in done:
                continue
            if args.max_fits is not None and n_fits >= args.max_fits:
                sys.exit(0)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            pipe = _build_dl(model_cfg, train_cfg, n_chans=Xs.shape[1],
                             n_times=Xs.shape[2], n_outputs=4, sfreq=250.0,
                             device=args.device, seed=args.seed)
            clf = pipe.named_steps["clf"]
            cbs = list(clf.callbacks)
            pos = [i for i, c in enumerate(cbs)
                   if isinstance(c, tuple) and c[0] == "restore"][0]
            cbs.insert(pos, ("testacc", TestAcc(Xs[te], ys[te])))
            clf.set_params(callbacks=cbs)
            t0 = time.time()
            pipe.fit(Xs[tr], ys[tr])
            final = float((pipe.predict(Xs[te]) == ys[te]).mean())
            h = clf.history
            rec = dict(model=args.model, seed=args.seed, subject=subject,
                       session=str(session), fold=fold, n_train=len(tr),
                       n_test=len(te), seconds=round(time.time() - t0, 1),
                       epochs=len(h), restored_epoch=clf.restored_epoch_,
                       final_test_acc=final,
                       valid_acc=[r["valid_acc"] for r in h],
                       test_acc=[r["test_acc"] for r in h],
                       width=[r.get("width") for r in h])
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n_fits += 1
            print(f"s{subject} {session} f{fold} {rec['seconds']}s "
                  f"restored={rec['restored_epoch']} test={final:.3f}", flush=True)
