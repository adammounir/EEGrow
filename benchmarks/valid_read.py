"""Read the 21/10 validation campaign (PREREG_validation_growth_rule.md).

Committed BEFORE any result of the campaign is read; it computes exactly the criteria of
the PREREG § 4 and nothing else is decisional. Unit = the subject: each arm's score is
averaged over sessions and seeds per subject, contrasts are paired by subject, 95 % CI
by percentile bootstrap over subjects (10 000 resamples, numpy default_rng(0)).

Per (model, rule, dataset):
  S  stability   every fit finite (train_loss and valid_loss at every epoch) and
                 grad_norm_max_fit <= 1e6 (the step-0 spike threshold)
  G  growth      >= 90 % of fits end wider than they started (width_end > width_start)
  NI non-infer.  PASS if the CI lower bound of mean(grow - bd) > -0.02; INFERIOR if the
                 CI upper bound < 0 (significantly worse); else INCONCLUSIVE (the data
                 cannot tell: never read as a failure of the rule, never as a pass)
  C  complete    the arm has exactly the bd twin's subjects, with the planned seeds
Readability of a (model, dataset): CI lower bound of mean(bd - chance) > 0 (chance =
0.5 for ROC-AUC on 2 classes, 1/n_classes for accuracy). A rule is VALIDATED on a model
when S, G, NI and C pass on all 3 datasets (intersection-union: each test at 5 %, no
correction needed for the conjunction); FAILED when any S, G, C fails or any NI is
INFERIOR; otherwise INCONCLUSIVE. Unreadable (model, dataset) -> the model is "not
readable" there, never "validated".

Usage::

    python valid_read.py --root /scratch/amounir/eegrow_valid --out read/valid
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os

import numpy as np

MODELS = ["shallow", "eegnex", "tidnet", "msvtnet", "ctnet", "conformer"]
RULES = ["tiny", "m025", "varA"]
DATASETS = {"bnci2014_001": [0, 1, 2], "cho2017": [0], "lee2019_mi": [0]}
DELTA = 0.02
SPIKE = 1e6
GROW_FRAC = 0.90
N_BOOT = 10_000


def _dir(root, label, ds):
    return f"{root}/valid_results/{label}/results/within_session/{ds}"


def scores(root, label, ds):
    per, seeds, ncls = collections.defaultdict(list), set(), set()
    for f in glob.glob(f"{_dir(root, label, ds)}/*__seed*.csv"):
        for r in csv.DictReader(open(f)):
            per[r["subject"]].append(float(r["score"]))
            seeds.add(int(r["seed"]))
            ncls.add(int(float(r["n_classes"])))
    return ({k: float(np.mean(v)) for k, v in per.items()}, sorted(seeds), ncls,
            {k: len(v) for k, v in per.items()})


def fits(root, label, ds):
    n = nonfinite = spikes = grew = 0
    wfrac = []
    for f in glob.glob(f"{_dir(root, label, ds)}/*__fits.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            bad = any(not (isinstance(h.get(k), (int, float)) and math.isfinite(h[k]))
                      for h in r["history"] for k in ("train_loss", "valid_loss"))
            nonfinite += bad
            g = r.get("grad_norm_max_fit")
            spikes += not (isinstance(g, (int, float)) and math.isfinite(g) and g <= SPIKE)
            if r.get("width_start") is not None and r.get("width_end") is not None:
                grew += r["width_end"] > r["width_start"]
                if r.get("target_width"):
                    wfrac.append(r["width_end"] / r["target_width"])
    return {"n_fits": n, "nonfinite": nonfinite, "spikes": spikes, "grew": grew,
            "width_frac_median": float(np.median(wfrac)) if wfrac else None}


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"n": len(d), "mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "n_neg": int((d < 0).sum())}


def paired(a, b):
    ks = sorted(set(a) & set(b), key=lambda k: int(k))
    return boot([a[k] - b[k] for k in ks]) if ks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"cells": {}, "readable": {}, "verdict": {}, "explo": {}}
    L = ["# Validation de la règle de croissance — lecture", "",
         f"NI : borne basse IC95 de (grow − bd) > −{DELTA} ; S : 0 non-fini, 0 pic > 1e6 ;"
         f" G : ≥ {GROW_FRAC:.0%} des fits ont grandi ; C : mêmes sujets et graines que bd.", ""]
    for m in MODELS:
        L += [f"## {m}", "", "| dataset | bd (− chance [IC]) | règle | n | grow − bd [IC95] (neg) |"
              " NI | S (nonfini/pics/fits) | G (grandi, largeur méd.) | C |",
              "|---|---|---|---|---|---|---|---|---|"]
        for ds, planned in DATASETS.items():
            bd, bd_seeds, ncls, _ = scores(a.root, f"{m}__bd", ds)
            chance = (0.5 if ncls == {2} else 1 / max(ncls)) if ncls else None
            rd = boot([v - chance for v in bd.values()]) if bd else None
            readable = bool(rd and rd["lo"] > 0 and bd_seeds == planned)
            out["readable"][f"{m}/{ds}"] = {"bd_minus_chance": rd, "readable": readable,
                                            "chance": chance, "bd_seeds": bd_seeds}
            bdtxt = ("absent" if not rd else
                     f"{np.mean(list(bd.values())):.3f} ({rd['mean']:+.3f} [{rd['lo']:+.3f}, {rd['hi']:+.3f}])")
            for rule in RULES:
                lab = f"{m}__{rule}"
                g, seeds, _, _ = scores(a.root, lab, ds)
                fz = fits(a.root, lab, ds)
                c = paired(g, bd) if g and bd else None
                ok_c = bool(g) and set(g) == set(bd) and seeds == planned
                ok_s = fz["n_fits"] > 0 and fz["nonfinite"] == 0 and fz["spikes"] == 0
                ok_g = fz["n_fits"] > 0 and fz["grew"] >= GROW_FRAC * fz["n_fits"]
                ok_ni = bool(c and c["lo"] > -DELTA)
                inferior = bool(c and c["hi"] < 0)
                out["cells"][f"{m}/{rule}/{ds}"] = {"contrast": c, "fits": fz, "C": ok_c,
                                                    "S": ok_s, "G": ok_g, "NI": ok_ni,
                                                    "inferior": inferior,
                                                    "readable": readable}
                ct = ("n/a" if not c else
                      f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ({c['n_neg']}/{c['n']})")
                yn = lambda b: "✅" if b else "❌"
                ni_txt = "✅" if ok_ni else ("❌ inférieur" if inferior else "❔ non concluant")
                wf = fz["width_frac_median"]
                L.append(f"| {ds} | {bdtxt} | {rule} | {len(g)} | {ct} | {ni_txt} | "
                         f"{yn(ok_s)} ({fz['nonfinite']}/{fz['spikes']}/{fz['n_fits']}) | "
                         f"{yn(ok_g)} ({fz['grew']}/{fz['n_fits']}, "
                         f"{'—' if wf is None else f'{wf:.2f}'}) | {yn(ok_c)} |")
        L.append("")
    L += ["## Verdict (PREREG § 4.3)", "", "| règle | " + " | ".join(MODELS) + " |",
          "|---" * (len(MODELS) + 1) + "|"]
    for rule in RULES:
        row = []
        for m in MODELS:
            cs = [out["cells"][f"{m}/{rule}/{ds}"] for ds in DATASETS]
            if not all(out["readable"][f"{m}/{ds}"]["readable"] for ds in DATASETS):
                v = "NON LISIBLE"
            elif all(c["S"] and c["G"] and c["NI"] and c["C"] for c in cs):
                v = "VALIDÉE"
            else:
                why = ",".join(f"{ds.split('_')[0][:4]}:" + "".join(
                    k for k in ("S", "G", "NI", "C") if not c[k])
                    for ds, c in zip(DATASETS, cs) if not all(c[k] for k in ("S", "G", "NI", "C")))
                hard = any(not (c["S"] and c["G"] and c["C"]) or c["inferior"] for c in cs)
                v = ("ÉCHEC (" if hard else "NON CONCLUANT (") + why + ")"
            out["verdict"][f"{rule}/{m}"] = v
            row.append(v)
        L.append(f"| {rule} | " + " | ".join(row) + " |")
    L += ["", "## Exploratoire (non décisionnel) : règle contre règle, apparié par sujet", "",
          "| modèle | dataset | tiny − varA | m025 − varA | tiny − m025 |", "|---|---|---|---|---|"]
    for m in MODELS:
        for ds in DATASETS:
            s = {r: scores(a.root, f"{m}__{r}", ds)[0] for r in RULES}
            cells = []
            for x, y in (("tiny", "varA"), ("m025", "varA"), ("tiny", "m025")):
                c = paired(s[x], s[y]) if s[x] and s[y] else None
                out["explo"][f"{m}/{ds}/{x}-{y}"] = c
                cells.append("n/a" if not c else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
            L.append(f"| {m} | {ds} | " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
