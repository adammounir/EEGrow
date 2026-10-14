"""Règles de sélection de l'époque sur valid_acc, comparées sur le COMPORTEMENT.

orig  : 1er argmax (égalité -> plus tôt), ce qui a tourné
late  : dernier argmax (égalité -> plus tardif), décision du 06/10
smW   : argmax (1er) de la moyenne glissante CENTRÉE de largeur W (bords : fenêtre tronquée)

Critères de choix (comportement, écrits avant de regarder la sonde) : part de fits scorés
< ép. 5, part >= budget-5, ICC(1,k) de restored_epoch par sujet. La partielle q_cent|acc
n'est affichée qu'en contrôle (choisir dessus = calibrer sur le résultat).
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from re_cv import DATASETS, A, dm, dp

W = (5, 11, 21)


def pick(acc, ep):
    out = {"orig": ep[np.argmax(acc)], "late": ep[np.where(acc == acc.max())[0][-1]]}
    s = pd.Series(acc)
    for w in W:
        m = s.rolling(w, center=True, min_periods=w // 2 + 1).mean().to_numpy()
        out[f"sm{w}"] = ep[int(np.nanargmax(m))]
    return out


rows = []
for ds, (tree, stage_c) in DATASETS.items():
    for seed in (0, 1, 2):
        for line in open(f"{tree}/within_session/{ds}/grow_shallow__seed{seed}__fits.jsonl"):
            r = json.loads(line); h = r["history"]
            acc = np.array([x["valid_acc"] for x in h], float); ep = np.array([x["epoch"] for x in h])
            rows.append(dict(ds=ds, stage_c=stage_c, subject=int(r["subject"]), seed=seed,
                             E=len(ep), **pick(acc, ep)))
f = pd.DataFrame(rows)
rules = ["orig", "late"] + [f"sm{w}" for w in W]
out, pooled = [], {k: [] for k in rules}
for ds, g in f.groupby("ds"):
    line = {"ds": ds}
    rk = pd.read_csv(A / f"ranking_{ds}.csv")
    d = dm.load_matrix(A / f"dxr_{ds}"); q, _ = dm.donor_quality(d, dm.METRIC)
    subj = g.groupby("subject")[rules].mean().reset_index()
    m = subj.merge(rk[["subject", "acc_probe"]], on="subject").merge(
        q[["donor", "q_cent"]], left_on="subject", right_on="donor").dropna()
    for k in rules:
        line[f"early_{k}"] = (g[k] < 5).mean()
        line[f"end_{k}"] = (g[k] >= g.E - 5).mean()
        i, _, kk = dp.icc1([v.to_numpy(float) for _, v in g.groupby("subject")[k]])
        line[f"icc_{k}"] = kk * i / (1 + (kk - 1) * i) if i > 0 else 0.0
        r_ = dp.partial_spearman(m[k].to_numpy(float), m.q_cent.to_numpy(float), m.acc_probe.to_numpy(float))
        line[f"par_{k}"] = r_
        if g.stage_c.iloc[0]:
            pooled[k].append((r_, len(m)))
    out.append(line)
t = pd.DataFrame(out).set_index("ds")
pd.set_option("display.width", 250)
for crit in ("early", "end", "icc", "par"):
    print(f"\n{crit}"); print(t[[f"{crit}_{k}" for k in rules]].round(3).to_string())
print("\nétage C groupé (contrôle seulement)")
for k, lst in pooled.items():
    z = np.arctanh([r for r, _ in lst]); w = np.array([n - 4 for _, n in lst], float)
    zb = (w * z).sum() / w.sum(); se = 1 / np.sqrt(w.sum())
    print(f"  {k:5s} {np.tanh(zb):+.3f} [{np.tanh(zb-1.96*se):+.3f}, {np.tanh(zb+1.96*se):+.3f}]")
t.to_csv("smooth_rule_summary.csv")
f.to_csv("smooth_rule_fits.csv", index=False)
