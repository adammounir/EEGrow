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
alpha = 0.025 one-sided (the 95 % two-sided CI's tail), one family per test kind: the
endpoints with >= 1 readable complete dataset (18 at most; 15 while TIDNet is out)
and per question.

Shallow init control (PREREG_shallow_init_control.md § 3), cells of a second ``--cells``
file (campaignS_grid), same per-case verdicts and pooling:
  C1 = tiny_bdinit - bd  ENDPOINT, one per protocol (3), Holm at alpha = 0.025 one-sided
                         over these 3, per test kind; verdicts as Q1
  C2 = tiny - tiny_bdinit, C3 = fix - bd  reported, nothing decided
  C1 + C2 = Q1 subject by subject: an identity, checked (exit 2 if it fails: the reader
  would not be pairing the same subjects)
and the reading of Shallow's Q1 the PREREG prescribes: C1 SUPÉRIEUR -> attributable to
growth; Q1 SUPÉRIEUR without C1 -> confounded by the init, the published verdict is C1.

Refuses (exit 2) on a JSONL whose growth rows mix directions or carry another direction
than the arm's (the 25/10 smoke: tiny and random once shared a file).

    python campaignA_read.py --cells campaignA_grid/campaignA_cells_planned_av5.txt \
        --cells campaignS_grid/campaignA_cells_planned_av5.txt --manifest ../manifest_A \
        --out ../read/campaignA
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from campaignA_check import GROWS, parse_cells

DELTA = 0.02
SPIKE = 1e6
N_BOOT = 10_000
ALPHA = 0.025
Z_MDE = 1.959964 + 0.841621  # two-sided 5 %, power 80 %
MODELS = ["shallow", "eegnex", "tidnet", "msvtnet", "ctnet", "conformer"]
PROTOCOLS = ["within_session", "cross_session", "cross_subject"]
ARMS = ["bd", "tiny", "random", "tiny_bdinit", "fix"]
# (question, A, B): contrast A - B. Q1 / Q2: PREREG_campaign_A § 3; C1-C3: the Shallow
# init control. Holm families: one per decided question and test kind.
CONTRASTS = (("Q1", "tiny", "bd"), ("Q2", "tiny", "random"), ("C1", "tiny_bdinit", "bd"),
             ("C2", "tiny", "tiny_bdinit"), ("C3", "fix", "bd"))
DECIDED = ("Q1", "Q2", "C1")
TITLES = {"Q1": "Q1 — tiny − bd", "Q2": "Q2 — tiny − random",
          "C1": "C1 — tiny_bdinit − bd (endpoint du contrôle d'init de Shallow)",
          "C2": "C2 — tiny − tiny_bdinit (rapporté, ne décide rien)",
          "C3": "C3 — fix − bd (rapporté, ne décide rien)"}
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


def load(cells_paths):
    """{(prot, ds, model, arm): {subj: {seed: score}}}, fit counters, gpu families."""
    scores = collections.defaultdict(lambda: collections.defaultdict(dict))
    fits = collections.defaultdict(collections.Counter)
    fams = collections.defaultdict(lambda: collections.defaultdict(set))
    planned = collections.defaultdict(lambda: collections.defaultdict(set))
    ncls = {}
    for c in (c for p in cells_paths for c in parse_cells(p)):
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
            want = set() if GROWS[c["arm"]] is None else {GROWS[c["arm"]]}
            if dirs != want:
                print(f"REFUSED: {c['csv']} fit {r['fit_uid'][:8]} directions {dirs} != {want}")
                sys.exit(2)
            f = fits[key]
            f["fits"] += 1
            f["nonfinite"] += not all(isinstance(h.get(k), (int, float)) and math.isfinite(h[k])
                                      for h in r["history"] for k in ("train_loss", "valid_loss"))
            g = r.get("grad_norm_max_fit")
            f["spikes"] += not (isinstance(g, (int, float)) and math.isfinite(g) and g <= SPIKE)
            # scored as the constant predictor (subject_stamp.constant_score, PREREG § 3.1)
            lg = glob.glob(f"{c['csv'][:-4]}__fits__logits/{r['fit_uid']}*.npz")
            f["diverged"] += any(bool(np.load(p)["diverged"]) for p in lg)
            if GROWS[c["arm"]] is not None:
                f["grew"] += bool(r.get("width_end") and r.get("width_start")
                                  and r["width_end"] > r["width_start"])
            fams[key][c["subj"]].add(family(r.get("gpu_name")))
    return scores, fits, fams, planned, ncls


def per_subject(sc, planned):
    """{subj: mean over seeds}, complete flag (every planned subject and seed present)."""
    complete = set(sc) == set(planned) and all(set(sc[s]) == planned[s] for s in planned)
    return {s: float(np.mean(list(v.values()))) for s, v in sc.items()}, complete


def pool(cases, q, models):
    """Pooled rows of question q, per (protocol, model) that has q in >= 1 case."""
    rows = []
    for prot in PROTOCOLS:
        for m in models:
            have = {k_: c for k_, c in cases.items()
                    for p_, m_, ds in [k_.split("/")] if p_ == prot and m_ == m and q in c}
            if not have and not q.startswith("Q"):
                continue
            use = {k_.split("/")[2]: c[q]["deltas"] for k_, c in have.items()
                   if c[q]["verdict"] not in ("INCOMPLET", "NON LISIBLE")}
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
    return rows


def decide(rows, q):
    live = [r for r in rows if r["n_datasets"]]
    if q in DECIDED:
        for kind in ("sup", "ni"):
            for r, rej in zip(live, holm([r[f"p_{kind}"] for r in live])):
                r[f"holm_{kind}"] = rej
    q1like = q in ("Q1", "C1")
    for r in rows:
        if not r["n_datasets"]:
            r["verdict"] = "AUCUN DATASET"
        elif q not in DECIDED:
            r["verdict"] = "RAPPORTÉ"
        elif r["holm_sup"]:
            r["verdict"] = "SUPÉRIEUR" if q1like else "LA DIRECTION COMPTE"
        elif q1like and r["holm_ni"]:
            r["verdict"] = "NON INFÉRIEUR"
        elif r["hi"] < 0 and r["lo"] <= -DELTA:
            r["verdict"] = "INFÉRIEUR" if q1like else "RANDOM MEILLEUR"
        else:
            r["verdict"] = "NON CONCLUANT"
    return rows


def shallow_q1_reading(pooled):
    """PREREG_shallow_init_control.md § 3, per protocol."""
    out = {}
    for prot in PROTOCOLS:
        q1 = next((r for r in pooled.get("Q1", []) if r["prot"] == prot and r["model"] == "shallow"), None)
        c1 = next((r for r in pooled.get("C1", []) if r["prot"] == prot), None)
        if not q1 or not c1:
            continue
        if c1["verdict"] == "SUPÉRIEUR":
            txt = "C1 SUPÉRIEUR : l'effet de Shallow est attribuable à la croissance"
        elif q1["verdict"] == "SUPÉRIEUR":
            txt = f"Q1 SUPÉRIEUR sans C1 : confondu par l'init ; verdict publié = C1 ({c1['verdict']})"
        else:
            txt = f"Q1 {q1['verdict']}, C1 {c1['verdict']} : Q1 rapporté tel quel"
        out[prot] = {"Q1": q1["verdict"], "C1": c1["verdict"], "reading": txt}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, action="append",
                    help="planned cells; repeat it for the Shallow init control grid")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {Path(f).stem: json.load(open(f)) for f in Path(a.manifest).glob("*.json")}
    scores, fits, fams, planned, ncls = load(a.cells)
    datasets = sorted({k[1] for k in planned})
    out = {"cases": {}, "pooled": {}}
    L = ["# Campagne A — lecture (PREREG_campaign_A.md § 3, PREREG_shallow_init_control.md § 3)", "",
         "Score = dernière époque. Δ = moyenne par sujet ; IC95 bootstrap sur les sujets. "
         f"NI : borne basse > −{DELTA}. Case hors comptes si non lisible ou incomplète.", ""]
    for prot in PROTOCOLS:
        for m in MODELS:
            for ds in datasets:
                key = lambda arm: (prot, ds, m, arm)
                if key("bd") not in planned:
                    continue
                arms = {arm: per_subject(scores.get(key(arm), {}), planned[key(arm)])
                        for arm in ARMS if key(arm) in planned}
                k = len(man[ds]["classes"])
                chance = 0.5 if k == 2 else 1 / k
                bd = arms["bd"][0]
                rd = boot([v - chance for v in bd.values()]) if bd else None
                case = {"readable": bool(rd and rd["lo"] > 0), "bd_minus_chance": rd,
                        "fits": {arm: dict(fits[key(arm)]) for arm in arms}}
                for q, A, B in CONTRASTS:
                    if A not in arms or B not in arms:
                        continue
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
                if all(q in case for q in ("Q1", "C1", "C2")):
                    d1, d2, dq = case["C1"]["deltas"], case["C2"]["deltas"], case["Q1"]["deltas"]
                    both = set(d1) & set(d2)
                    err = max((abs(d1[s] + d2[s] - dq[s]) for s in both if s in dq), default=0.0)
                    if (set(both) - set(dq)) or err > 1e-9 or (
                            case["Q1"]["complete"] and case["C1"]["complete"] and case["C2"]["complete"]
                            and both != set(dq)):
                        print(f"REFUSED: {prot}/{m}/{ds} C1 + C2 != Q1 (subjects "
                              f"{len(both)} vs {len(dq)}, max |error| {err:.2e})")
                        sys.exit(2)
                    case["identity_C1_C2_Q1"] = {"n": len(both), "max_abs_error": err}
                out["cases"][f"{prot}/{m}/{ds}"] = case
    for q, _, _ in CONTRASTS:
        if q.startswith("C") and not any(q in c for c in out["cases"].values()):
            continue
        out["pooled"][q] = decide(pool(out["cases"], q, MODELS), q)
    out["shallow_q1_reading"] = shallow_q1_reading(out["pooled"])
    # report
    for q in out["pooled"]:
        L += [f"## {TITLES[q]}", "", "| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict |"
              + (" même famille de carte |" if q == "Q1" else ""),
              "|---|---|---|---|---|---|---|" + ("---|" if q == "Q1" else "")]
        for k_, c in out["cases"].items():
            if q not in c:
                continue
            p_, m_, ds = k_.split("/")
            x = c[q]["contrast"]
            ct = "n/a" if not x else f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] ({x['n_neg']}/{x['n']})"
            mde = "—" if not x or x["mde"] is None else f"{x['mde']:.3f}"
            extra = ""
            if q == "Q1":
                sf = c.get("Q1_same_family") or {}
                extra = f" {sf.get('n', 0)} : {sf.get('verdict') or '—'} |"
            L.append(f"| {p_} | {m_} | {ds} | {0 if not x else x['n']} | {ct} | {mde} | {c[q]['verdict']} |{extra}")
        n_live = sum(1 for r in out["pooled"][q] if r["n_datasets"])
        head = (f"Holm sur les {n_live} endpoints lisibles, α = {ALPHA} unilatéral" if q in DECIDED
                else "sans test : rapporté")
        L += ["", f"### {q} groupé ({head})", "",
              "| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for r in out["pooled"][q]:
            if not r["n_datasets"]:
                L.append(f"| {r['prot']} | {r['model']} | 0 | — | — | — | — | {r['verdict']} |")
                continue
            L.append(f"| {r['prot']} | {r['model']} | {r['n_datasets']} | {r['D']:+.3f} [{r['lo']:+.3f}, "
                     f"{r['hi']:+.3f}] | {r['p_sup']:.4f} | {r['p_ni']:.4f} | {r['mde']:.3f} | {r['verdict']} |")
        L.append("")
    if out["shallow_q1_reading"]:
        L += ["## Lecture de Q1 Shallow (PREREG_shallow_init_control.md § 3)", "",
              "| protocole | Q1 | C1 | lecture |", "|---|---|---|---|"]
        for prot, r in out["shallow_q1_reading"].items():
            L.append(f"| {prot} | {r['Q1']} | {r['C1']} | {r['reading']} |")
        idn = [c["identity_C1_C2_Q1"] for c in out["cases"].values() if "identity_C1_C2_Q1" in c]
        L += ["", f"Identité C1 + C2 = Q1 vérifiée sur {len(idn)} cases, "
              f"{sum(i['n'] for i in idn)} sujets, erreur max {max((i['max_abs_error'] for i in idn), default=0):.1e}.", ""]
    L += ["## Stabilité et croissance (comptes, par bras)", "",
          "| protocole | modèle | dataset | bras | fits | divergés (score constant) | non finis | pics > 1e6 | ont grandi |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k_, c in out["cases"].items():
        p_, m_, ds = k_.split("/")
        for arm, f in c["fits"].items():
            L.append(f"| {p_} | {m_} | {ds} | {arm} | {f.get('fits', 0)} | {f.get('diverged', 0)} | {f.get('nonfinite', 0)} | "
                     f"{f.get('spikes', 0)} | {'—' if GROWS[arm] is None else f.get('grew', 0)} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"... -> {a.out}.md")


if __name__ == "__main__":
    main()
