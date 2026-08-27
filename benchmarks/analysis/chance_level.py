"""How much of the v5 benchmark is actually distinguishable from chance?

THE PROBLEM
-----------
A benchmark cell reports something like "0.62 accuracy" and we read it as decoding.
But an accuracy is a proportion measured on a finite number of test trials, and with
few trials the *chance* level is not 1/n_classes -- it is whatever value a coin-flip
classifier exceeds 5% of the time. Combrisson & Jerbi (2015) make the point for
accuracy: at 20 test trials and two classes a random classifier reaches 70% often
enough that 70% means nothing.

Several cells here are far past that warning. `shin2017a` within-session has 20 trials
per session; `physionetmi` has 45. Whatever those rows say, they cannot support a
per-subject claim.

WHAT THIS COMPUTES
------------------
Three things, all from the published v5 scores and no new compute.

1. The exact one-sided 95% chance threshold for every score, under the null that the
   classifier is at chance. Two nulls, because MOABB uses two metrics:

     accuracy  -> Binomial(n, 1/n_classes). Combrisson & Jerbi's case.
     roc_auc   -> the Mann-Whitney U null. AUC = U/(n1*n2), and under H0 U has a known
                  exact distribution. The normal approximation sd is
                  sqrt((n1+n2+1)/(12*n1*n2)), which at n=20 balanced is 0.132 -- so the
                  95% threshold sits at AUC 0.72, not 0.5. The AUC cells are *not*
                  exempt from the problem; they hide it better.

2. How much of the between-subject spread is pure sampling noise. This is the part that
   matters for the paper's error bars. If subject scores in a cell have sd 0.11 and the
   binomial sd at that trial count is also 0.11, then the "between-subject variability"
   we plot is the measurement, not the subjects, and no amount of extra seeds removes it
   -- seeds re-run the same trials, so they average away initialisation noise and
   nothing else.

3. Whether the cell mean survives a group-level test across subjects. This is the honest
   counterweight: a small per-subject n does NOT invalidate the cell mean. It inflates
   each subject's error bar and costs power, but the across-subject test remains valid.
   Cells can therefore be jointly (a) significant as a group and (b) incapable of
   supporting any statement about an individual subject. Saying so precisely is the
   contribution.

EFFECTIVE n
-----------
Not `samples_test` in every case. The v5 rows are already averaged over CV folds:

  within_session   MOABB runs 5-fold CV inside the session, so every trial is predicted
                   exactly once and the score is over `samples` trials, not the
                   `samples_test` of one fold.
  cross_session    one held-out session      -> `samples_test`
  cross_subject    one held-out subject      -> `samples_test`

Getting this wrong by a factor of five moves every threshold, so it is asserted below
rather than assumed.

Usage: python chance_level.py [results_dir] [out_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else
               Path(__file__).resolve().parents[1] / "results_v5_published")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           Path(__file__).resolve().parent / "chance")


# --------------------------------------------------------------- null distributions
def binom_threshold(n: int, p: float, alpha: float = ALPHA) -> float:
    """Smallest accuracy a random classifier exceeds with probability <= alpha.

    Exact and one-sided: the smallest k with P(X >= k) <= alpha, over n. Found by an
    explicit scan rather than `isf`, whose off-by-one convention on discrete supports
    is exactly the kind of silent half-percent that would shift every verdict here.
    Returns inf when no attainable accuracy is significant -- which is a real outcome
    at these trial counts, not an error.
    """
    k = np.arange(n + 1)
    sf = stats.binom.sf(k - 1, n, p)           # P(X >= k)
    ok = k[sf <= alpha]
    return float(ok.min()) / n if len(ok) else np.inf


def _u_null_counts(n1: int, n2: int) -> np.ndarray:
    """Exact null counts of the Mann-Whitney U statistic, U = 0 .. n1*n2.

    Classic recurrence c(a, b, u) = c(a-1, b, u-b) + c(a, b-1, u): the largest-ranked
    observation belongs either to sample 1 (contributing b) or to sample 2. Base cases
    c(0, b, .) = c(a, 0, .) = [1] at u = 0.
    """
    prev = [np.array([1.0]) for _ in range(n2 + 1)]        # a = 0
    for a in range(1, n1 + 1):
        cur = [np.array([1.0])]                            # b = 0
        for b in range(1, n2 + 1):
            out = np.zeros(a * b + 1)
            src = prev[b]                                  # c(a-1, b, u-b)
            out[b:b + len(src)] += src
            src2 = cur[b - 1]                              # c(a, b-1, u)
            out[:len(src2)] += src2
            cur.append(out)
        prev = cur
    return prev[n2]


def auc_threshold(n1: int, n2: int, alpha: float = ALPHA, exact_max: int = 80) -> float:
    """Smallest ROC-AUC a random classifier exceeds with probability <= alpha.

    Exact via the U null for small samples, normal approximation with a continuity
    correction beyond `exact_max` per group. Returns inf when nothing is attainable.
    """
    if n1 <= exact_max and n2 <= exact_max:
        counts = _u_null_counts(n1, n2)
        sf = counts[::-1].cumsum()[::-1] / counts.sum()    # sf[u] = P(U >= u)
        u = np.flatnonzero(sf <= alpha)
        return float(u.min()) / (n1 * n2) if len(u) else np.inf
    sd = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    return float(0.5 + stats.norm.isf(alpha) * sd + 0.5 / (n1 * n2))


def sampling_sd(score: float, n: int, metric: str) -> float:
    """Sd of the score from trial sampling alone, at the observed level.

    accuracy: binomial. roc_auc: Hanley & McNeil (1982), the standard estimate under
    the alternative rather than the (smaller) null variance -- using the null variance
    here would understate the noise for the good cells.
    """
    if metric == "accuracy":
        return float(np.sqrt(max(score * (1 - score), 0.0) / n))
    n1 = n2 = n / 2.0
    a = min(max(score, 1e-6), 1 - 1e-6)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)
    return float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------- load
def load_scores() -> pd.DataFrame:
    s = pd.read_csv(RESULTS / "eegrow_benchmark_all_scores.csv.gz")

    # Assert the fold geometry the effective-n rule depends on, rather than trusting it.
    w = s[s["eval"] == "within_session"]
    # Folds are unequal when the session size is not divisible by 5, so the ratio drifts
    # to ~5.4; the guard is against a different fold COUNT (3, 10), which would move
    # every threshold by a factor of two or more.
    ratio = (w["samples"] / w["samples_test"]).round(1)
    assert ratio.between(4.5, 5.6).all(), f"within_session not 5-fold: {ratio.unique()}"

    s["n_eff"] = np.where(s["eval"] == "within_session", s["samples"], s["samples_test"])
    s["n_eff"] = s["n_eff"].astype(int)
    s["chance"] = np.where(s["metric"] == "accuracy", 1.0 / s["n_classes"], 0.5)
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_scores()

    # Thresholds depend only on (metric, n_eff, n_classes) -- a few dozen combinations
    # for 78k rows, so compute each once.
    keys = s[["metric", "n_eff", "n_classes"]].drop_duplicates()
    thr = {}
    for _, r in keys.iterrows():
        n = int(r.n_eff)
        if r.metric == "accuracy":
            thr[(r.metric, n, r.n_classes)] = binom_threshold(n, 1.0 / r.n_classes)
        else:
            thr[(r.metric, n, r.n_classes)] = auc_threshold(n // 2, n - n // 2)
    s["threshold"] = [thr[(m, n, c)] for m, n, c in
                      zip(s["metric"], s["n_eff"], s["n_classes"])]
    s["above_chance"] = s["score"] >= s["threshold"]
    s["sampling_sd"] = [sampling_sd(v, n, m) for v, n, m in
                        zip(s["score"], s["n_eff"], s["metric"])]

    s.to_csv(OUT / "scores_with_chance.csv.gz", index=False)

    # ---------------------------------------------------------- 1. geometry per cell
    geom = (s.groupby(["dataset", "eval", "metric"])
             .agg(n_classes=("n_classes", "first"), n_eff=("n_eff", "median"),
                  chance=("chance", "first"), threshold=("threshold", "median"))
             .reset_index())
    geom["excess_needed"] = geom["threshold"] - geom["chance"]
    geom = geom.sort_values("excess_needed", ascending=False)
    geom.to_csv(OUT / "chance_geometry.csv", index=False)

    print("=" * 78)
    print("EXACT 95% CHANCE THRESHOLD PER CELL  (one-sided, alpha=0.05)")
    print("=" * 78)
    print(geom.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------- 2. how much spread is sampling noise only
    # Unit = one (subject, session) evaluated by one model: average over seeds, which
    # re-run the same trials and so cannot reduce trial-sampling noise.
    unit = (s.groupby(["eval", "dataset", "model", "family", "metric", "subject",
                       "session"], dropna=False)
             .agg(score=("score", "mean"), n_eff=("n_eff", "first"),
                  chance=("chance", "first"), threshold=("threshold", "first"),
                  sampling_sd=("sampling_sd", "mean"))
             .reset_index())

    rows = []
    for (ev, ds, mo, fam, me), g in unit.groupby(
            ["eval", "dataset", "model", "family", "metric"]):
        if len(g) < 3:
            continue
        obs_sd = float(g["score"].std(ddof=1))
        samp_sd = float(np.sqrt((g["sampling_sd"] ** 2).mean()))
        true_var = obs_sd ** 2 - samp_sd ** 2
        # Group-level test across units: does the cell mean beat chance at all?
        try:
            p_group = float(stats.wilcoxon(g["score"] - g["chance"],
                                           alternative="greater").pvalue)
        except ValueError:
            p_group = np.nan
        rows.append(dict(
            eval=ev, dataset=ds, model=mo, family=fam, metric=me, n_units=len(g),
            n_eff=int(g["n_eff"].median()), chance=float(g["chance"].iloc[0]),
            threshold=float(g["threshold"].iloc[0]), score_mean=float(g["score"].mean()),
            obs_sd=obs_sd, sampling_sd=samp_sd,
            true_sd=float(np.sqrt(max(true_var, 0.0))),
            noise_share=float(min(samp_sd ** 2 / obs_sd ** 2, 1.0)) if obs_sd > 0 else np.nan,
            frac_units_above=float(g["score"].ge(g["threshold"]).mean()),
            p_group=p_group))
    cells = pd.DataFrame(rows)
    cells["group_significant"] = cells["p_group"] < ALPHA
    cells.to_csv(OUT / "cell_chance_verdicts.csv", index=False)

    print()
    print("=" * 78)
    print("HOW MUCH OF THE BETWEEN-SUBJECT SPREAD IS TRIAL-SAMPLING NOISE?")
    print("(averaged over models; noise_share = 1 means the error bars are measurement)")
    print("=" * 78)
    per_cell = (cells.groupby(["dataset", "eval"])
                     .agg(n_eff=("n_eff", "median"), obs_sd=("obs_sd", "mean"),
                          samp_sd=("sampling_sd", "mean"), true_sd=("true_sd", "mean"),
                          noise_share=("noise_share", "mean"),
                          frac_above=("frac_units_above", "mean"),
                          frac_group_sig=("group_significant", "mean"))
                     .reset_index().sort_values("noise_share", ascending=False))
    print(per_cell.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -------------------------------------------- 3. cells that lose their claim
    print()
    print("=" * 78)
    print("CELLS SIGNIFICANT AS A GROUP BUT WITH <25% OF SUBJECTS INDIVIDUALLY ABOVE")
    print("(the cell mean stands; any per-subject statement in the paper does not)")
    print("=" * 78)
    bad = cells[(cells["group_significant"]) & (cells["frac_units_above"] < 0.25)]
    print(f"{len(bad)} of {len(cells)} cells "
          f"({100 * len(bad) / max(len(cells), 1):.0f}%)")
    print(bad.groupby(["dataset", "eval"]).size().to_string())

    print()
    print("CELLS WHERE EVEN THE GROUP TEST FAILS (nothing to report at all):")
    dead = cells[~cells["group_significant"]]
    print(f"{len(dead)} of {len(cells)} cells "
          f"({100 * len(dead) / max(len(cells), 1):.0f}%)")
    if len(dead):
        print(dead.groupby(["dataset", "eval"]).size().to_string())

    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
