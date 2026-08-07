"""Where does growing actually beat its fixed twin?

Strictly paired analysis. A comparison is only valid between grow_X and bd_X on the
*same* (eval, dataset, subject, session, seed): same data, same split, same seed. We
merge on that key and look at the per-observation delta, never at two independent means.

Aggregation for the test: average the 5 seeds within each (subject, session) first, so
one observation = one held-out subject/session -- seeds are replicates of the same
measurement, not independent samples. Wilcoxon signed-rank on those observations.

One arm at a time. The alignment ablation writes aligned cells next to the raw ones
under the same (eval, dataset, subject, session, seed), so a merge that ignores the
``align`` column silently returns a cartesian product -- every raw grow row paired with
both the raw and the aligned fixed row. We therefore select a single arm (default the
raw one, ``align=none``); the aligned-vs-raw contrast is a different question and lives
in ea_interaction.py.

Usage:  python where_grow_wins.py [results_root] [align]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
ALIGN = sys.argv[2] if len(sys.argv) > 2 else "none"

PAIRS = [
    ("grow_shallow", "bd_shallow"),
    ("grow_sccnet", "bd_sccnet"),
    ("grow_eegnex", "bd_eegnex"),
    ("grow_deep", "bd_deep4"),  # loose pair (VGG-style vs Deep4Net) -- flag it
]
DEEP = {m for p in PAIRS for m in p}
KEY = ["eval", "dataset", "subject", "session", "seed"]

rows = []
for f in glob.glob(os.path.join(ROOT, "*", "*", "*.csv")):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if "score" not in d.columns or "model" not in d.columns:
        continue
    if d.empty or d["model"].iloc[0] not in DEEP:
        continue
    ev, ds, _ = f.split(os.sep)[-3:]
    d["eval"], d["dataset"] = ev, ds
    # the raw grid was produced before the align axis existed, so its CSVs have no
    # such column; those rows are the unaligned arm by construction
    d["align"] = d["align"].fillna("none") if "align" in d.columns else "none"
    rows.append(d)

df = pd.concat(rows, ignore_index=True)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])
# the shared-hdf5 artefact duplicated some rows verbatim; harmless but drop them
before = len(df)
df = df.drop_duplicates(subset=KEY + ["model", "align", "score"])
n_dedup = before - len(df)
seen = sorted(df["align"].unique())
df = df[df["align"] == ALIGN]
print(f"bras align={ALIGN} (presents dans results: {', '.join(seen)})")
print(f"lignes deep: {len(df)} sur ce bras (dedup: -{n_dedup})")
if df.empty:
    raise SystemExit(f"aucune ligne pour align={ALIGN!r}")

# --- garde-fou prétraitement : une paire dont les deux bras n'ont pas tourné a la
# meme frequence d'echantillonnage compare du pretraitement, pas de la croissance.
from regime_guard import assert_paired  # noqa: E402

assert_paired(PAIRS, bench_root=os.path.dirname(os.path.abspath(ROOT)) or ".")

# metric per dataset (MOABB: roc_auc for LeftRightImagery, accuracy otherwise)
AUC_DS = {"bnci2014_004", "cho2017", "lee2019_mi", "physionetmi", "shin2017a",
          "weibo2014"}


def metric(ds):
    return "AUC" if ds in AUC_DS else "acc"


records = []
for g, b in PAIRS:
    G = df[df["model"] == g][KEY + ["score"]].rename(columns={"score": "g"})
    B = df[df["model"] == b][KEY + ["score"]].rename(columns={"score": "b"})
    M = G.merge(B, on=KEY, how="inner")
    if M.empty:
        continue
    M["pair"] = f"{g} vs {b}"
    records.append(M)

M = pd.concat(records, ignore_index=True)
M["delta"] = M["g"] - M["b"]
print(f"observations appariees: {len(M)}\n")


def test(sub):
    """One observation per (subject, session): mean over seeds. Then Wilcoxon."""
    o = sub.groupby(["subject", "session"])[["g", "b"]].mean()
    if len(o) < 6:
        return len(o), np.nan, np.nan, np.nan
    d = o["g"].values - o["b"].values
    if np.allclose(d, 0):
        return len(o), 0.0, 1.0, 0.0
    p = wilcoxon(o["g"].values, o["b"].values).pvalue
    return len(o), float(d.mean()), float(p), float((d > 0).mean())


print("=" * 100)
print("A) VUE D'ENSEMBLE : chaque paire x protocole (moyenne des deltas par dataset)")
print("=" * 100)
for pair in M["pair"].unique():
    print(f"\n{pair}")
    for ev in ["within_session", "cross_session", "cross_subject"]:
        s = M[(M["pair"] == pair) & (M["eval"] == ev)]
        if s.empty:
            continue
        # dataset-balanced: mean delta within dataset, then across datasets
        per_ds = s.groupby("dataset")["delta"].mean()
        wins = (per_ds > 0).sum()
        print(f"  {ev:15s} delta={per_ds.mean():+.4f}   "
              f"datasets ou grow>bd : {wins}/{len(per_ds)}")

print("\n" + "=" * 100)
print("B) DETAIL PAR CELLULE (paire x protocole x dataset), Wilcoxon apparie par sujet")
print("=" * 100)
out = []
for (pair, ev, ds), sub in M.groupby(["pair", "eval", "dataset"]):
    n, d, p, frac = test(sub)
    out.append(dict(pair=pair, eval=ev, dataset=ds, n=n, delta=d, p=p,
                    frac_pos=frac, metric=metric(ds),
                    n_seeds=sub["seed"].nunique()))
res = pd.DataFrame(out)

sig = res[(res["p"] < 0.05) & (res["delta"] > 0)].sort_values("delta", ascending=False)
print(f"\n>>> GROW GAGNE SIGNIFICATIVEMENT (p<0.05) : {len(sig)} cellules sur {len(res)}")
if len(sig):
    print(sig.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

sigl = res[(res["p"] < 0.05) & (res["delta"] < 0)].sort_values("delta")
print(f"\n>>> GROW PERD SIGNIFICATIVEMENT (p<0.05) : {len(sigl)} cellules")
if len(sigl):
    print(sigl.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

ns = res[res["p"] >= 0.05]
print(f"\n>>> PAS DE DIFFERENCE DETECTABLE : {len(ns)} cellules")

print("\n" + "=" * 100)
print("C) TABLE COMPLETE (triee par delta)")
print("=" * 100)
print(res.sort_values("delta", ascending=False).to_string(
    index=False, float_format=lambda v: f"{v:+.4f}"))

print("\n" + "=" * 100)
print("D) EFFET DE LA TAILLE DU DATASET (la croissance aide-t-elle en petit regime ?)")
print("=" * 100)
n_subj = df.groupby("dataset")["subject"].nunique().rename("n_subjects")
agg = res.groupby("dataset")["delta"].mean().to_frame("delta_moyen").join(n_subj)
print(agg.sort_values("n_subjects").to_string(float_format=lambda v: f"{v:+.4f}"))
