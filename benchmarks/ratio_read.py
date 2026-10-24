"""Read the 23/10 censoring campaign (PREREG_growth_ratio_censoring.md).

Committed BEFORE any result of the campaign is read. Same unit, pairing and bootstrap as
valid_read.py (imported): the subject, paired by subject, percentile bootstrap over
subjects (10 000 resamples, default_rng(0)).

Per (model, ratio, dataset), against the two reference arms of the validation campaign:
  vs r0.1   mean(r - tiny@0.1): does growing further change the score?
  vs bd     mean(r - bd): non-inferiority as in the validation (margin 0.02)
  S         every fit finite, grad_norm_max_fit <= 1e6
  width     fraction of fits at target, median final width / target
Primary (PREREG § 4): ratio 0 (every proposed neuron kept) against 0.1, per model:
  GAIN    CI lower bound > 0 on BOTH datasets (intersection-union, no correction)
  LOSS    CI upper bound < 0 on ANY dataset
  NO GAIN DETECTED otherwise -- reported with its CI, never read as "equivalent"
The three intermediate ratios are a dose-response, exploratory.

Usage::

    python ratio_read.py --root /scratch/amounir/eegrow_valid --out read/ratio
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from valid_read import _dir, paired, scores

MODELS = {"conformer": "grow_conformer_all", "ctnet": "grow_ctnet_all",
          "msvtnet": "grow_msvtnet_all"}
RATIOS = [("tiny", 0.1), ("tiny_r003", 0.03), ("tiny_r001", 0.01),
          ("tiny_r0001", 0.001), ("tiny_r0", 0.0)]
DATASETS = {"bnci2014_001": [0, 1, 2], "lee2019_mi": [0]}
DELTA = 0.02
SPIKE = 1e6


def widths(root, label, ds):
    n = bad = at = 0
    frac = []
    for f in glob.glob(f"{_dir(root, label, ds)}/*__fits.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            g = r.get("grad_norm_max_fit")
            bad += (any(not (isinstance(h.get(k), (int, float)) and math.isfinite(h[k]))
                        for h in r["history"] for k in ("train_loss", "valid_loss"))
                    or not (isinstance(g, (int, float)) and math.isfinite(g) and g <= SPIKE))
            if r.get("width_end") is None or r.get("target_width") is None:
                continue  # fixed arm: stability only, no width
            at += r["width_end"] >= r["target_width"]
            frac.append(r["width_end"] / r["target_width"])
    return {"n_fits": n, "unstable": bad, "at_target": at / n if n else None,
            "width_frac_median": float(np.median(frac)) if frac else None}


def ci(c):
    return "n/a" if not c else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"cells": {}, "primary": {}}
    L = ["# Censure de la croissance (plancher relatif) — lecture", "",
         "r = `min_singular_ratio` ; 0.1 = réglage de la validation (référence) ; "
         "0 = tous les neurones proposés gardés.", ""]
    for m in MODELS:
        L += [f"## {m}", "", "| dataset | r | n | à la cible | largeur méd. | stable | "
              "r − r0.1 [IC95] | r − bd [IC95] | NI |", "|---|---|---|---|---|---|---|---|---|"]
        for ds, planned in DATASETS.items():
            bd = scores(a.root, f"{m}__bd", ds)[0]
            ref = scores(a.root, f"{m}__tiny", ds)[0]
            for lab, r in RATIOS:
                s, seeds, _, _ = scores(a.root, f"{m}__{lab}", ds)
                w = widths(a.root, f"{m}__{lab}", ds)
                c_ref = paired(s, ref) if s and ref and lab != "tiny" else None
                c_bd = paired(s, bd) if s and bd else None
                complete = bool(s) and set(s) == set(bd) and seeds == planned
                out["cells"][f"{m}/{r}/{ds}"] = {"vs_ref": c_ref, "vs_bd": c_bd, "widths": w,
                                                 "complete": complete}
                ni = ("—" if not c_bd else "✅" if c_bd["lo"] > -DELTA else
                      "❌ inférieur" if c_bd["hi"] < 0 else "❔")
                at = "—" if w["at_target"] is None else f"{w['at_target']:.0%}"
                wf = "—" if w["width_frac_median"] is None else f"{w['width_frac_median']:.2f}"
                L.append(f"| {ds} | {r} | {len(s)}{'' if complete else ' (incomplet)'} | "
                         f"{at} | {wf} | "
                         f"{w['n_fits'] - w['unstable']}/{w['n_fits']} | "
                         f"{'réf.' if lab == 'tiny' else ci(c_ref)} | {ci(c_bd)} | {ni} |")
        L.append("")
    L += ["## Critère principal (PREREG § 4) : r = 0 contre r = 0.1", "",
          "| modèle | bnci2014_001 | lee2019_mi | verdict |", "|---|---|---|---|"]
    for m in MODELS:
        cs = [out["cells"][f"{m}/0.0/{ds}"] for ds in DATASETS]
        vs = [c["vs_ref"] for c in cs]
        if not all(c["complete"] for c in cs) or any(v is None for v in vs):
            v = "INCOMPLET"
        elif all(x["lo"] > 0 for x in vs):
            v = "GAIN"
        elif any(x["hi"] < 0 for x in vs):
            v = "PERTE"
        else:
            v = "PAS DE GAIN DÉTECTÉ"
        out["primary"][m] = v
        L.append(f"| {m} | {ci(vs[0])} | {ci(vs[1])} | {v} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
