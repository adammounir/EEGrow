"""One tidy table for the whole grid, plus the summaries a reader actually wants.

Everything downstream reads these three files, so the aggregation rules live here once:

* one observation is one (protocol, dataset, subject, session, model, seed) score, which
  is what MOABB writes; nothing is averaged before it reaches the tidy file;
* the five seeds are replicates of one measurement, not five independent samples, so
  every summary averages them *within* a (subject, session) before any statistic --
  otherwise the effective n is inflated fivefold and every p-value is wrong;
* the metric differs by dataset (MOABB gives roc_auc for two-class LeftRightImagery and
  accuracy otherwise), so scores are never pooled across datasets without saying which.

The alignment arm is kept in the tidy file, tagged, and excluded from the summaries:
it answers a different question and lives in the Euclidean-alignment PR.
"""
import glob
import os

import pandas as pd

AUC_DS = {"bnci2014_004", "cho2017", "lee2019_mi", "physionetmi", "shin2017a",
          "weibo2014"}
PAIRS = [("grow_shallow", "bd_shallow"), ("grow_sccnet", "bd_sccnet"),
         ("grow_eegnex", "bd_eegnex"), ("grow_deep", "bd_deep4")]
ML = ["csp_lda", "csp_svm", "fgmdm", "mdm", "ts_lr", "ts_svm"]
KEY = ["eval", "dataset", "subject", "session", "seed"]

rows = []
for p in sorted(glob.glob("benchmarks/results/*/*/*.csv")):
    ev, ds, stem = os.path.abspath(p).split(os.sep)[-3:]
    parts = stem[:-4].split("__")
    if not parts[-1].startswith("seed"):
        continue
    d = pd.read_csv(p)
    d["eval"], d["dataset"] = ev, ds
    d["model"] = parts[0]
    d["seed"] = parts[-1][4:]
    d["align"] = "euclidean" if len(parts) > 2 else "none"
    rows.append(d)

df = pd.concat(rows, ignore_index=True)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])
df["metric"] = df["dataset"].map(lambda d: "roc_auc" if d in AUC_DS else "accuracy")
df["family"] = df["model"].map(
    lambda m: "growing" if m.startswith("grow") else
              ("braindecode" if m.startswith("bd_") else "riemann/csp"))
cols = ["eval", "dataset", "subject", "session", "seed", "model", "family", "align",
        "score", "metric", "time", "samples", "samples_test", "n_classes", "channels"]
df = df[[c for c in cols if c in df.columns]]
df.to_csv("eegrow_benchmark_all_scores.csv.gz", index=False, compression="gzip")
print(f"eegrow_benchmark_all_scores.csv.gz : {len(df)} lignes, "
      f"{df['model'].nunique()} modeles, {df['dataset'].nunique()} datasets")

# `df.align` is a DataFrame method, so the column has to be indexed by name
raw = df[df["align"] == "none"].copy()

# --- per model x protocol x dataset, seeds averaged within (subject, session) first
obs = (raw.groupby(["eval", "dataset", "model", "family", "metric",
                    "subject", "session"], as_index=False)["score"].mean())
lvl = (obs.groupby(["eval", "dataset", "model", "family", "metric"], as_index=False)
       .agg(score_mean=("score", "mean"), score_sd=("score", "std"),
            n_obs=("score", "size")))
lvl.to_csv("eegrow_benchmark_levels.csv", index=False)
print(f"eegrow_benchmark_levels.csv        : {len(lvl)} lignes "
      f"(modele x protocole x dataset, graines moyennees par sujet/session)")

# --- the paired contrast the PR is about, one row per (pair, protocol, dataset)
from scipy.stats import wilcoxon  # noqa: E402

recs = []
for grow, fixed in PAIRS:
    a = obs[obs.model == grow].drop(columns=["model", "family"])
    b = obs[obs.model == fixed].drop(columns=["model", "family"])
    m = a.merge(b, on=["eval", "dataset", "metric", "subject", "session"],
                suffixes=("_grow", "_fixed"))
    for (ev, ds, me), g in m.groupby(["eval", "dataset", "metric"]):
        d = g.score_grow - g.score_fixed
        try:
            p = wilcoxon(g.score_grow, g.score_fixed).pvalue
        except ValueError:
            p = float("nan")
        recs.append(dict(pair=f"{grow} vs {fixed}", eval=ev, dataset=ds, metric=me,
                         n_obs=len(g), grow=g.score_grow.mean(),
                         fixed=g.score_fixed.mean(), delta=d.mean(),
                         delta_sd=d.std(), frac_positive=(d > 0).mean(),
                         wilcoxon_p=p))
pairs = pd.DataFrame(recs).sort_values(["pair", "eval", "dataset"])
pairs.to_csv("eegrow_benchmark_paired.csv", index=False)
print(f"eegrow_benchmark_paired.csv        : {len(pairs)} contrastes apparies")

print("\n=== meilleure famille par protocole x dataset (niveau absolu) ===")
best = (lvl.sort_values("score_mean", ascending=False)
        .groupby(["eval", "dataset"]).head(1)
        .sort_values(["eval", "dataset"]))
print(best[["eval", "dataset", "model", "family", "metric", "score_mean"]]
      .to_string(index=False))

print("\n=== moyenne par famille et protocole (datasets AUC seulement, comparables) ===")
auc = lvl[lvl.metric == "roc_auc"]
print(auc.pivot_table(index="family", columns="eval", values="score_mean",
                      aggfunc="mean").round(4).to_string())

print("\n=== le meilleur de chaque famille, par protocole (datasets AUC) ===")
top = (auc.groupby(["eval", "family", "model"], as_index=False)["score_mean"].mean()
       .sort_values("score_mean", ascending=False)
       .groupby(["eval", "family"]).head(1).sort_values(["eval", "score_mean"],
                                                       ascending=[True, False]))
print(top.to_string(index=False))
