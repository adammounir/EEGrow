"""Read the 23/10 Shallow learning-rate campaign (PREREG_shallow_lr_path.md).

Committed BEFORE any result of the campaign is read. Same unit, pairing and bootstrap as
valid_read.py (imported): the subject, paired by subject, percentile bootstrap over
subjects (10 000 resamples, default_rng(0)).

Per dataset, lr* = the lr where bd_shallow's mean subject score is highest (over the 5
lrs, 6.25e-4 = the validation cells). Selecting lr* on bd's own test score favours bd:
the primary contrast is therefore conservative against growth.
Primary (PREREG § 4): C = tiny(lr*) - bd(lr*), paired by subject, per dataset:
  SURVIT      CI lower bound > 0 on ALL 3 datasets (intersection-union)
  DISPARAÎT   CI lower bound <= 0 on ALL 3 datasets
  PARTIEL     otherwise
Reported, not decisional: bd(lr*) - bd(6.25e-4) (was bd under-optimised?), the full
lr x arm table, best-tiny - best-bd, stability and width at target.

Usage::

    python lrshallow_read.py --root /scratch/amounir/eegrow_valid --out read/lrshallow
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ratio_read import widths
from valid_read import DATASETS, paired, scores

LRS = [("", 6.25e-4), ("_lr2e-4", 2e-4), ("_lr2e-3", 2e-3), ("_lr6.25e-3", 6.25e-3),
       ("_lr2e-2", 2e-2)]


def ci(c):
    return "n/a" if not c else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"cells": {}, "primary": {}}
    L = ["# Shallow : l'avantage de la croissance survit-il au lr ? — lecture", "",
         "lr* = lr où bd_shallow a le meilleur score moyen (sélection favorable à bd).", ""]
    prim = {}
    for ds, planned in DATASETS.items():
        L += [f"## {ds}", "", "| lr | bd (moy.) | tiny (moy.) | tiny − bd [IC95] | stable bd | "
              "stable tiny | tiny à la cible | complet |", "|---|---|---|---|---|---|---|---|"]
        s = {}
        for tag, lr in LRS:
            b, bs, _, _ = scores(a.root, f"shallow__bd{tag}", ds)
            t, ts, _, _ = scores(a.root, f"shallow__tiny{tag}", ds)
            wb, wt = widths(a.root, f"shallow__bd{tag}", ds), widths(a.root, f"shallow__tiny{tag}", ds)
            c = paired(t, b) if t and b else None
            complete = bool(b) and bool(t) and set(b) == set(t) and bs == ts == planned
            s[lr] = (b, t, complete)
            out["cells"][f"{ds}/{lr}"] = {"tiny_minus_bd": c, "bd_mean": float(np.mean(list(b.values()))) if b else None,
                                          "tiny_mean": float(np.mean(list(t.values()))) if t else None,
                                          "w_bd": wb, "w_tiny": wt, "complete": complete}
            m = lambda d: "—" if not d else f"{np.mean(list(d.values())):.3f}"
            at = "—" if wt["at_target"] is None else f"{wt['at_target']:.0%}"
            L.append(f"| {lr:g} | {m(b)} | {m(t)} | {ci(c)} | {wb['n_fits'] - wb['unstable']}/{wb['n_fits']} | "
                     f"{wt['n_fits'] - wt['unstable']}/{wt['n_fits']} | {at} | {'✅' if complete else '❌'} |")
        ok = {lr: v for lr, v in s.items() if v[0]}
        lr_star = max(ok, key=lambda lr: np.mean(list(ok[lr][0].values()))) if ok else None
        lr_t = max((lr for lr, v in s.items() if v[1]),
                   key=lambda lr: np.mean(list(s[lr][1].values())), default=None)
        c = paired(s[lr_star][1], s[lr_star][0]) if lr_star and s[lr_star][1] else None
        gain_bd = paired(s[lr_star][0], s[6.25e-4][0]) if lr_star and s[6.25e-4][0] else None
        best = paired(s[lr_t][1], s[lr_star][0]) if lr_t and lr_star else None
        complete = all(v[2] for v in s.values())
        prim[ds] = {"lr_star": lr_star, "lr_tiny_best": lr_t, "C": c, "bd_gain_from_lr": gain_bd,
                    "best_tiny_minus_best_bd": best, "complete": complete}
        L += ["", f"lr* = {lr_star:g} ; bd(lr*) − bd(6.25e-4) = {ci(gain_bd)} ; "
              f"meilleur tiny (lr {lr_t:g}) − meilleur bd = {ci(best)}" if lr_star and lr_t else "lr* indisponible", ""]
    L += ["## Critère principal (PREREG § 4) : tiny(lr*) − bd(lr*)", "",
          "| dataset | lr* | C [IC95] |", "|---|---|---|"]
    for ds, p in prim.items():
        lr_txt = "—" if p["lr_star"] is None else f"{p['lr_star']:g}"
        L.append(f"| {ds} | {lr_txt} | {ci(p['C'])} |")
    cs = [p["C"] for p in prim.values()]
    if not all(p["complete"] for p in prim.values()) or any(c is None for c in cs):
        v = "INCOMPLET"
    elif all(c["lo"] > 0 for c in cs):
        v = "SURVIT"
    elif all(c["lo"] <= 0 for c in cs):
        v = "DISPARAÎT"
    else:
        v = "PARTIEL"
    out["primary"] = {"per_dataset": prim, "verdict": v}
    L += ["", f"**Verdict : {v}**"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
