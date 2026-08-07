"""Follow-up: absolute levels + sign test, to tell 'growth helps' from 'the fixed
reference collapsed'. A +0.13 delta over a baseline sitting at chance is a different
claim from a +0.03 over a baseline that trains fine.

Like where_grow_wins.py, this reads ONE alignment arm at a time (default the raw one):
aligned and raw cells share the same (eval, dataset, subject, session, seed), so a
merge that ignores ``align`` returns a cartesian product instead of a pairing.

Usage:  python where_grow_wins2.py [results_root] [align]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
ALIGN = sys.argv[2] if len(sys.argv) > 2 else "none"
PAIRS = [("grow_shallow", "bd_shallow"), ("grow_sccnet", "bd_sccnet"),
         ("grow_eegnex", "bd_eegnex"), ("grow_deep", "bd_deep4")]
DEEP = {m for p in PAIRS for m in p}
KEY = ["eval", "dataset", "subject", "session", "seed"]
AUC_DS = {"bnci2014_004", "cho2017", "lee2019_mi", "physionetmi", "shin2017a",
          "weibo2014"}
# chance level per dataset, to judge whether an arm actually learned anything
CHANCE = {"alexmi": 1 / 3, "bnci2014_001": 0.25, "bnci2014_002": 0.5,
          "bnci2015_001": 0.5, "schirrmeister2017": 0.25, "zhou2016": 1 / 3}

rows = []
for f in glob.glob(os.path.join(ROOT, "*", "*", "*.csv")):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if "score" not in d.columns or "model" not in d.columns or d.empty:
        continue
    if d["model"].iloc[0] not in DEEP:
        continue
    ev, ds, _ = f.split(os.sep)[-3:]
    d["eval"], d["dataset"] = ev, ds
    # the raw grid predates the align axis, so its CSVs carry no such column
    d["align"] = d["align"].fillna("none") if "align" in d.columns else "none"
    rows.append(d)

df = pd.concat(rows, ignore_index=True)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
# `align` belongs in the dedup key: accuracy scores are quantised (0.25, 0.5, ...), so
# a raw and an aligned row of the same cell can legitimately carry the same value, and
# without it one of the two would be silently dropped.
df = df.dropna(subset=["score"]).drop_duplicates(
    subset=KEY + ["model", "align", "score"])
df = df[df["align"] == ALIGN]
print(f"bras align={ALIGN} : {len(df)} lignes")
if df.empty:
    raise SystemExit(f"aucune ligne pour align={ALIGN!r}")

# --- garde-fou prétraitement : une paire dont les deux bras n'ont pas tourné a la
# meme frequence d'echantillonnage compare du pretraitement, pas de la croissance.
from regime_guard import assert_paired  # noqa: E402

assert_paired(PAIRS, bench_root=os.path.dirname(os.path.abspath(ROOT)) or ".")


recs = []
for g, b in PAIRS:
    G = df[df["model"] == g][KEY + ["score"]].rename(columns={"score": "g"})
    B = df[df["model"] == b][KEY + ["score"]].rename(columns={"score": "b"})
    M = G.merge(B, on=KEY, how="inner")
    M["pair"] = g
    recs.append(M)
M = pd.concat(recs, ignore_index=True)

out = []
for (pair, ev, ds), sub in M.groupby(["pair", "eval", "dataset"]):
    o = sub.groupby(["subject", "session"])[["g", "b"]].mean()
    d = o["g"].values - o["b"].values
    p = np.nan
    if len(o) >= 6 and not np.allclose(d, 0):
        p = wilcoxon(o["g"].values, o["b"].values).pvalue
    ch = CHANCE.get(ds, 0.5)
    out.append(dict(pair=pair, eval=ev, dataset=ds, n=len(o),
                    grow=o["g"].mean(), fixed=o["b"].mean(), delta=d.mean(), p=p,
                    chance=ch, metric="AUC" if ds in AUC_DS else "acc",
                    fixed_above_chance=o["b"].mean() - ch))
res = pd.DataFrame(out)

print("=" * 112)
print("A) LES CELLULES OU GROW GAGNE (p<0.05) — avec les NIVEAUX ABSOLUS")
print("=" * 112)
w = res[(res["p"] < 0.05) & (res["delta"] > 0)].sort_values("delta", ascending=False)
print(w[["pair", "eval", "dataset", "n", "grow", "fixed", "delta", "chance",
         "fixed_above_chance", "metric"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 112)
print("B) TEST DE SIGNE PAR PAIRE (sur les deltas au niveau (eval, dataset))")
print("   -> une paire consistante gagne dans une large majorite de cellules")
print("=" * 112)
for pair in [p[0] for p in PAIRS]:
    s = res[res["pair"] == pair]
    per_cell = s.groupby(["eval", "dataset"])["delta"].mean()
    pos, n = int((per_cell > 0).sum()), len(per_cell)
    pv = binomtest(pos, n, 0.5).pvalue
    med = per_cell.median()
    print(f"  {pair:14s} {pos:2d}/{n:2d} cellules positives   "
          f"mediane delta={med:+.4f}   test de signe p={pv:.2e}")

print("\n" + "=" * 112)
print("C) PAR PROTOCOLE, TOUTES PAIRES CONFONDUES")
print("=" * 112)
for ev in ["within_session", "cross_session", "cross_subject"]:
    s = res[res["eval"] == ev]
    per_cell = s.groupby(["pair", "dataset"])["delta"].mean()
    pos, n = int((per_cell > 0).sum()), len(per_cell)
    print(f"  {ev:15s} {pos:2d}/{n:2d} positives   mediane={per_cell.median():+.4f}")

print("\n" + "=" * 112)
print("D) EFFET DE LA TAILLE DU DATASET")
print("=" * 112)
n_subj = df.groupby("dataset")["subject"].nunique().rename("n_sujets")
agg = res.groupby("dataset")["delta"].agg(["mean", "median", "count"]).join(n_subj)
print(agg.sort_values("n_sujets").to_string(float_format=lambda v: f"{v:+.4f}"))

print("\n" + "=" * 112)
print("E) MEME CHOSE POUR grow_shallow SEUL (la paire la plus consistante)")
print("=" * 112)
s = res[res["pair"] == "grow_shallow"].sort_values("delta", ascending=False)
print(s[["eval", "dataset", "n", "grow", "fixed", "delta", "p", "metric"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 112)
print("F) DIAGNOSTIC : la reference fixe s'entraine-t-elle ? (ecart au hasard)")
print("=" * 112)
diag = res.groupby(["pair", "eval"])["fixed_above_chance"].mean().unstack()
print(diag.to_string(float_format=lambda v: f"{v:+.4f}"))
