"""Line-search behaviour of the previous campaign, by arm and held-out (13/10/2026).

Local, zero GPU: dynamics_final/gd_events.csv.gz (542 187 growth events) joined to
gd_fits.csv.gz for n_train. GromoGrowth holds out the last 20 % of the epoch's batches
for the line search only when the epoch has >= 4 batches, i.e. ceil(0.8 n_train / 64)
>= 4; otherwise the line search runs on the very batches the statistics used.

Result (13/10):
  * 92.7 % of events had NO held-out at all.
  * The held-out changes decisions even though it is made of already-trained batches:
    while the net is still learning (train loss > 0.1), abstentions go 0 -> 7.5 %
    (grow_deep), 0.3 -> 20.5 % (grow_sccnet), 2.8 -> 23.9 % (grow_shallow).
  * When a step applies, s hits the TOP of the grid {0, 0.1, 0.5, 1} on 97.8-100 % of
    them, held-out or not: the grid is censored, its maximum says nothing about the
    optimum amplitude.

    python line_search_censoring.py      # from benchmarks/analysis
"""
import numpy as np
import pandas as pd

D = "dynamics_final/"
e = pd.read_csv(D + "gd_events.csv.gz", low_memory=False,
                usecols=["eval", "dataset", "model", "align_tag", "seed", "fit", "epoch",
                         "applied", "subject", "session", "grow_s", "train_loss_before"])
f = pd.read_csv(D + "gd_fits.csv.gz", low_memory=False,
                usecols=["eval", "dataset", "model", "align_tag", "seed", "fit",
                         "subject", "session", "n_train"])
for d in (e, f):
    d["align_tag"] = d["align_tag"].fillna("")
    d["session"] = d["session"].astype(str)
    d["subject"] = d["subject"].astype(str)
m = e.merge(f, on=["eval", "dataset", "model", "align_tag", "seed", "fit", "subject",
                   "session"], how="left")
m["holdout"] = np.ceil(0.8 * m.n_train / 64) >= 4
print("share of events with a held-out:", round(m.holdout.mean(), 3))
for (model, ho), g in m[m.train_loss_before > 0.1].groupby(["model", "holdout"]):
    a = g[g.applied == True]  # noqa: E712
    print(f"{model:13s} holdout={ho!s:5s} n {len(g):6d} "
          f"abstain {1 - len(a) / len(g):5.1%}  s at grid top: {(a.grow_s == 1.0).mean():5.1%}")
