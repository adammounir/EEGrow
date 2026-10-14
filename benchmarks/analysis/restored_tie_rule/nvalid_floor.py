"""Part de fits scorés < ép. 5 (et >= budget-5) sous orig/late, par tranche de n_valid.
grow_shallow within_session, 7 datasets. Sert à fixer le n_valid minimal du préreg."""
import json, glob
import numpy as np, pandas as pd
from re_cv import n_valid_of
rows = []
for p in glob.glob("results_*/within_session/*/grow_shallow__seed*__fits.jsonl"):
    ds = p.split("/")[2]
    for line in open(p):
        r = json.loads(line); h = r["history"]
        acc = np.array([x["valid_acc"] for x in h]); ep = np.array([x["epoch"] for x in h])
        b = acc.max(); first = ep[np.argmax(acc)]; last = ep[np.where(acc == b)[0][-1]]
        gmax = max((x.get("grad_norm_max", 0) or 0) for x in h)
        rows.append(dict(ds=ds, nv=n_valid_of(acc), n_train=r["n_train"], E=len(ep),
                         orig=first, late=last, gmax=gmax))
f = pd.DataFrame(rows)
f["early_o"] = f.orig < 5; f["early_l"] = f.late < 5
f["end_l"] = f.late >= f.E - 5; f["end_o"] = f.orig >= f.E - 5
bins = [0, 5, 8, 12, 16, 24, 32, 48, 1000]
f["bin"] = pd.cut(f.nv, bins)
print(f.groupby("bin", observed=True).agg(n=("ds", "size"), ds=("ds", lambda s: ",".join(sorted(set(s)))),
      ntrain=("n_train", "median"), early_o=("early_o", "mean"), early_l=("early_l", "mean"),
      end_l=("end_l", "mean")).round(3).to_string())
print()
print(f.groupby("ds").agg(nv=("nv", "median"), ntrain=("n_train", "median"),
      early_l=("early_l", "mean"), end_l=("end_l", "mean")).round(3).to_string())
f.to_csv("nvalid_floor_fits.csv", index=False)
