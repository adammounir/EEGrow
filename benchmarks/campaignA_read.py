"""Read campaign A (analysis/PREREG_campaign_A.md, § 3). Committed with the PREREG, BEFORE
any cell of the grid; it computes the PREREG's criteria and nothing else is decisional.

Unit = the subject. A subject's score in an arm = mean of its CSV rows (sessions) per
seed, then mean over the planned seeds. Contrasts are paired by subject; 95 % CIs are
percentile bootstraps over subjects (10 000 resamples, numpy default_rng(0)). Score =
the CSV's (MOABB: accuracy > 2 classes, ROC-AUC at 2), i.e. the LAST epoch.

Per case (model, dataset, protocol), Q1: A = tiny, B = bd; Q2: A = tiny, B = random:
  SUPÉRIEUR      CI lower bound of mean(A - B) > 0
  NON INFÉRIEUR  CI lower bound > -0.02
  INFÉRIEUR      CI upper bound < 0 and lower bound <= -0.02
  NON CONCLUANT  otherwise (reported with its MDE)
  LISIBLE        CI lower bound of mean(bd - chance) > 0 (chance 0.5 in ROC-AUC, 1/k in acc)
  COMPLET        both arms hold every subject of the manifest with every planned seed
A case not readable or not complete is reported and leaves the counts and the pools.

Pooled, per (model, protocol), 18 per question: D = mean over the readable and complete
datasets (equal weights) of their per-dataset mean delta; CI by a bootstrap stratified
by dataset (subjects resampled within each dataset). One-sided bootstrap p = share of
resamples <= threshold (0 for superiority, -0.02 for non-inferiority). Holm at
alpha = 0.025 one-sided (the 95 % two-sided CI's tail), one family of 18 per test kind
and per question.

Refuses (exit 2) on a JSONL whose growth rows mix directions or carry another direction
than the arm's (the 25/10 smoke: tiny and random once shared a file).

    python campaignA_read.py --cells campaignA_cells.txt --manifest ../manifest --out ../read/campaignA
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from campaignA_check import parse_cells

DELTA = 0.02
SPIKE = 1e6
N_BOOT = 10_000
ALPHA = 0.025
Z_MDE = 1.959964 + 0.841621  # two-sided 5 %, power 80 %
MODELS = ["shallow", "eegnex", "tidnet", "msvtnet", "ctnet", "conformer"]
PROTOCOLS = ["within_session", "cross_session", "cross_subject"]
FAMILY = (("V100", "volta"), ("P100", "pascal"), ("A100", "ampere"), ("A40", "ampere"),
          ("A30", "ampere"), ("A10", "ampere"), ("H100", "hopper"), ("RTX A", "ampere"),
          ("2080", "turing"), ("RTX", "turing"), ("Quadro", "turing"))


def family(name):
    return next((f for k, f in FAMILY if name and k in name), f"other:{name}")


def boot(d, rng=None):
    d = np.asarray(d, float)
    rng = rng or np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"n": len(d), "mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "n_neg": int((d < 0).sum()),
            "mde": float(Z_MDE * d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}


def verdict(c):
    if c["lo"] > 0:
        return "SUPÉRIEUR"
    if c["lo"] > -DELTA:
        return "NON INFÉRIEUR"
    if c["hi"] < 0:
        return "INFÉRIEUR"
    return "NON CONCLUANT"


def holm(ps, alpha=ALPHA):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, m = [False] * len(ps), len(ps)
    for r, i in enumerate(order):
        if ps[i] <= alpha / (m - r):
            rej[i] = True
        else:
            break
    return rej


def load(cells_path):
    """{(prot, ds, model, arm): {subj: {seed: score}}}, fit counters, gpu families."""
    scores = collections.defaultdict(lambda: collections.defaultdict(dict))
    fits = collections.defaultdict(collections.Counter)
    fams = collections.defaultdict(lambda: collections.defaultdict(set))
    planned = collections.defaultdict(lambda: collections.defaultdict(set))
    ncls = {}
    for c in parse_cells(cells_path):
        key = (c["prot"], c["ds"], c["model"], c["arm"])
        planned[key][c["subj"]].add(c["seed"])
        if not os.path.exists(c["csv"]):
            continue
        df = pd.read_csv(c["csv"])
        scores[key][c["subj"]][c["seed"]] = float(df["score"].mean())
        ncls[c["ds"]] = int(df["n_classes"].iloc[0]) if "n_classes" in df else None
        for line in open(c["csv"][:-4] + "__fits.jsonl"):
            r = json.loads(line)
            dirs = {h.get("grow_direction") for h in r["history"] if "grow_s" in h}
            want = set() if c["arm"] == "bd" else {c["arm"]}
            if dirs != want:
                print(f"REFUSED: {c['csv']} fit {r['fit_uid'][:8]} directions {dirs} != {want}")
                sys.exit(2)
            f = fits[key]
            f["fits"] += 1
            f["nonfinite"] += not all(isinstance(h.get(k), (int, float)) and math.isfinite(h[k])
                                      for h in r["history"] for k in ("train_loss", "valid_loss"))
            g = r.get("grad_norm_max_fit")
            f["spikes"] += not (isinstance(g, (int, float)) and math.isfinite(g) and g <= SPIKE)
            if c["arm"] != "bd":
                f["grew"] += bool(r.get("width_end") and r.get("width_start")
                                  and r["width_end"] > r["width_start"])
            fams[key][c["subj"]].add(family(r.get("gpu_name")))
    return scores, fits, fams, planned, ncls


def per_subject(sc, planned):
    """{subj: mean over seeds}, complete flag (every planned subject and seed present)."""
    complete = set(sc) == set(planned) and all(set(sc[s]) == planned[s] for s in planned)
    return {s: float(np.mean(list(v.values()))) for s, v in sc.items()}, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {Path(f).stem: json.load(open(f)) for f in Path(a.manifest).glob("*.json")}
    scores, fits, fams, planned, ncls = load(a.cells)
    datasets = sorted({k[1] for k in planned})
    out = {"cases": {}, "pooled": {}}
    L = ["# Campagne A — lecture (PREREG_campaign_A.md § 3)", "",
         "Score = dernière époque. Δ = moyenne par sujet ; IC95 bootstrap sur les sujets. "
         f"NI : borne basse > −{DELTA}. Case hors comptes si non lisible ou incomplète.", ""]
    for prot in PROTOCOLS:
        for m in MODELS:
            for ds in datasets:
                key = lambda arm: (prot, ds, m, arm)
                if key("bd") not in planned:
                    continue
                arms = {arm: per_subject(scores.get(key(arm), {}), planned[key(arm)])
                        for arm in ("bd", "tiny", "random")}
                k = len(man[ds]["classes"])
                chance = 0.5 if k == 2 else 1 / k
                bd = arms["bd"][0]
                rd = boot([v - chance for v in bd.values()]) if bd else None
                case = {"readable": bool(rd and rd["lo"] > 0), "bd_minus_chance": rd,
                        "fits": {arm: dict(fits[key(arm)]) for arm in arms}}
                for q, (A, B) in (("Q1", ("tiny", "bd")), ("Q2", ("tiny", "random"))):
                    (sa, ca), (sb, cb) = arms[A], arms[B]
                    complete = ca and cb
                    ks = sorted(set(sa) & set(sb), key=str)
                    c = boot([sa[s] - sb[s] for s in ks]) if ks else None
                    v = ("INCOMPLET" if not complete else "NON LISIBLE" if not case["readable"]
                         else verdict(c))
                    case[q] = {"contrast": c, "complete": complete, "verdict": v,
                               "deltas": {s: sa[s] - sb[s] for s in ks}}
                    if q == "Q1" and ks:
                        same = [s for s in ks if len(fams[key("tiny")][s] | fams[key("bd")][s]) == 1]
                        cs = boot([sa[s] - sb[s] for s in same]) if len(same) > 1 else None
                        case["Q1_same_family"] = {"n": len(same), "contrast": cs,
                                                  "verdict": verdict(cs) if cs else None}
                out["cases"][f"{prot}/{m}/{ds}"] = case
    # pooled endpoints
    for q in ("Q1", "Q2"):
        rows = []
        for prot in PROTOCOLS:
            for m in MODELS:
                use = {ds: c[q]["deltas"] for k_, c in out["cases"].items()
                       for p_, m_, ds in [k_.split("/")]
                       if p_ == prot and m_ == m and c[q]["verdict"] not in ("INCOMPLET", "NON LISIBLE")}
                if not use:
                    rows.append({"prot": prot, "model": m, "n_datasets": 0})
                    continue
                rng = np.random.default_rng(0)
                arr = {ds: np.array(list(d.values())) for ds, d in use.items()}
                bm = np.mean([d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
                              for d in arr.values()], axis=0)
                D = float(np.mean([d.mean() for d in arr.values()]))
                rows.append({"prot": prot, "model": m, "n_datasets": len(use), "datasets": sorted(use),
                             "D": D, "lo": float(np.percentile(bm, 2.5)), "hi": float(np.percentile(bm, 97.5)),
                             "p_sup": float((bm <= 0).mean()), "p_ni": float((bm <= -DELTA).mean()),
                             "mde": float(Z_MDE * bm.std())})
        live = [r for r in rows if r["n_datasets"]]
        for kind in ("sup", "ni"):
            for r, rej in zip(live, holm([r[f"p_{kind}"] for r in live])):
                r[f"holm_{kind}"] = rej
        for r in rows:
            if not r["n_datasets"]:
                r["verdict"] = "AUCUN DATASET"
            elif r["holm_sup"]:
                r["verdict"] = "SUPÉRIEUR" if q == "Q1" else "LA DIRECTION COMPTE"
            elif q == "Q1" and r["holm_ni"]:
                r["verdict"] = "NON INFÉRIEUR"
            elif r["hi"] < 0 and r["lo"] <= -DELTA:
                r["verdict"] = "INFÉRIEUR" if q == "Q1" else "RANDOM MEILLEUR"
            else:
                r["verdict"] = "NON CONCLUANT"
        out["pooled"][q] = rows
    # report
    for q, title in (("Q1", "Q1 — tiny − bd"), ("Q2", "Q2 — tiny − random")):
        L += [f"## {title}", "", "| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict |"
              + (" même famille de carte |" if q == "Q1" else ""),
              "|---|---|---|---|---|---|---|" + ("---|" if q == "Q1" else "")]
        for k_, c in out["cases"].items():
            p_, m_, ds = k_.split("/")
            x = c[q]["contrast"]
            ct = "n/a" if not x else f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] ({x['n_neg']}/{x['n']})"
            mde = "—" if not x or x["mde"] is None else f"{x['mde']:.3f}"
            extra = ""
            if q == "Q1":
                sf = c.get("Q1_same_family") or {}
                extra = f" {sf.get('n', 0)} : {sf.get('verdict') or '—'} |"
            L.append(f"| {p_} | {m_} | {ds} | {0 if not x else x['n']} | {ct} | {mde} | {c[q]['verdict']} |{extra}")
        L += ["", f"### {q} groupé (Holm, 18 tests, α = {ALPHA} unilatéral)", "",
              "| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for r in out["pooled"][q]:
            if not r["n_datasets"]:
                L.append(f"| {r['prot']} | {r['model']} | 0 | — | — | — | — | {r['verdict']} |")
                continue
            L.append(f"| {r['prot']} | {r['model']} | {r['n_datasets']} | {r['D']:+.3f} [{r['lo']:+.3f}, "
                     f"{r['hi']:+.3f}] | {r['p_sup']:.4f} | {r['p_ni']:.4f} | {r['mde']:.3f} | {r['verdict']} |")
        L.append("")
    L += ["## Stabilité et croissance (comptes, par bras)", "",
          "| protocole | modèle | dataset | bras | fits | non finis | pics > 1e6 | ont grandi |",
          "|---|---|---|---|---|---|---|---|"]
    for k_, c in out["cases"].items():
        p_, m_, ds = k_.split("/")
        for arm, f in c["fits"].items():
            L.append(f"| {p_} | {m_} | {ds} | {arm} | {f.get('fits', 0)} | {f.get('nonfinite', 0)} | "
                     f"{f.get('spikes', 0)} | {'—' if arm == 'bd' else f.get('grew', 0)} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"... -> {a.out}.md")


if __name__ == "__main__":
    main()
