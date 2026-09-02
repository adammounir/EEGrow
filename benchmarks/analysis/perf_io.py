"""Load the campaign's score CSVs into one frame, and the statistics the figures test.

`growth_io` reads the JSONL *histories* -- what happened inside a fit. This reads the
CSV *scores* -- what came out of it. They are two different exports of the same
campaign and neither is derivable from the other, which is why both exist.

WHAT THE LOADER DERIVES, AND WHY EACH ONE IS NOT OPTIONAL

``metric`` and ``chance``. The CSVs record a bare ``score`` column and nothing that
says what it measures. MOABB reports ROC-AUC for a ``LeftRightImagery`` paradigm and
accuracy for ``MotorImagery``, so a frame that pools them is pooling two scales with
two different chance levels (0.5 against 1/n_classes) -- and every "growth wins by
+0.04" statement silently averages across both. The paradigm is a per-dataset config
fact, not a per-row one, so it is reconstructed here from
:data:`AUC_DATASETS` rather than guessed from the value.

``family`` and ``arch``. The three arms of a comparison do not share a substring:
``grow_deep``'s controls are ``bd_deep4`` and ``fix_deepeeg``. Any figure that wants
the decomposition ``grow - bd = (grow - fix) + (fix - bd)`` needs the triples spelled
out, and spelling them out once here is the difference between one wrong pairing and
twelve.

``above_chance``. A per-row exact test that this cell learned anything at all. The v5
audit found the headline ``grow_deep`` gain was an artifact of a *baseline sitting
below chance*: the delta was real, the claim it supported was not. Carrying the flag
on every row makes that failure mode visible in a figure instead of discoverable in a
post-mortem.

THE UNIT OF ANALYSIS IS THE SUBJECT.

Rows are (subject, session, seed) triples, and within a subject they are correlated:
the same person, the same electrodes, the same day. Testing at row level treats
n = 3 seeds x 2 sessions x 9 subjects = 54 as 54 independent observations when it is 9,
which inflates every p-value's significance by roughly the square root of that ratio --
measured at 10^4 on this grid. :func:`by_subject` collapses to one number per
(dataset, subject) and every test in this module runs on its output. There is no path
through here that tests at row level; that is deliberate.

Usage::

    import perf_io
    sc = perf_io.load("benchmarks/analysis/perf_final/scores")
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- grid

#: Datasets whose paradigm is ``LeftRightImagery``, for which MOABB reports ROC-AUC.
#: Read off ``benchmarks/config/dataset/*.yaml``; hard-coded rather than re-parsed
#: because the analysis must be runnable against an archived score tree whose configs
#: have since moved.
AUC_DATASETS = {"bnci2014_004", "cho2017", "lee2019_mi", "physionetmi", "shin2017a",
                "weibo2014"}

#: MOABB's dataset display names, back to the config keys the rest of the campaign
#: uses. The CSVs carry the display name; every path, claim and TSV row carries the key.
DATASET_KEY = {
    "AlexandreMotorImagery": "alexmi", "BNCI2014-001": "bnci2014_001",
    "BNCI2014-002": "bnci2014_002", "BNCI2014-004": "bnci2014_004",
    "BNCI2015-001": "bnci2015_001", "Cho2017": "cho2017",
    "Lee2019-MI": "lee2019_mi", "PhysionetMotorImagery": "physionetmi",
    "Schirrmeister2017": "schirrmeister2017", "Shin2017A": "shin2017a",
    "Weibo2014": "weibo2014", "Zhou2016": "zhou2016",
}

#: The (braindecode, fixed control, growing) triples. Same table as
#: ``growth_dynamics.TRIPLES``, duplicated for the same reason it is duplicated there:
#: importing would couple a score-frame module to a history-frame module for four lines.
TRIPLES = {"shallow": ("bd_shallow", "fix_shallow", "grow_shallow"),
           "deep": ("bd_deep4", "fix_deepeeg", "grow_deep"),
           "sccnet": ("bd_sccnet", "fix_sccnet", "grow_sccnet"),
           "eegnex": ("bd_eegnex", "fix_eegnex", "grow_eegnex")}

MODEL_ORDER = ["bd_shallow", "fix_shallow", "grow_shallow",
               "bd_deep4", "fix_deepeeg", "grow_deep",
               "bd_sccnet", "fix_sccnet", "grow_sccnet",
               "bd_eegnex", "fix_eegnex", "grow_eegnex"]

EVAL_ORDER = ["within_session", "cross_session", "cross_subject"]

#: One number per subject is the unit; these are the columns collapsed into it.
UNIT = ["eval", "align_tag", "model", "dataset", "subject"]

ALPHA = 0.05
BOOT = 20000
RNG = np.random.default_rng(0)


def family_of(model: str) -> str:
    if model.startswith("grow"):
        return "growing"
    if model.startswith("bd_"):
        return "braindecode"
    if model.startswith("fix_"):
        return "fixed control"
    return "other"


def arch_of(model: str) -> str:
    for arch, arms in TRIPLES.items():
        if model in arms:
            return arch
    return "other"


# ---------------------------------------------------------------------- null levels

def binom_threshold(n: int, p: float, alpha: float = ALPHA) -> float:
    """Smallest accuracy a random classifier exceeds with probability <= alpha.

    Exact and one-sided: the smallest k with P(X >= k) <= alpha. Lifted from
    ``chance_level.binom_threshold``, which established the convention -- an explicit
    scan rather than ``isf``, whose off-by-one on a discrete support is exactly the
    silent half-percent that would move a verdict here. ``inf`` when nothing attainable
    is significant, which is a real outcome at small trial counts and not an error.
    """
    k = np.arange(n + 1)
    ok = k[stats.binom.sf(k - 1, n, p) <= alpha]
    return float(ok.min()) / n if len(ok) else np.inf


def auc_threshold(n: int, alpha: float = ALPHA) -> float:
    """Smallest ROC-AUC a random classifier exceeds with probability <= alpha.

    Normal approximation to the Mann-Whitney null with a continuity correction, at
    balanced groups of n/2. ``chance_level`` computes the exact null below 80 per
    group; the approximation is used throughout here because this runs per (dataset,
    eval) rather than per row, the smallest group on the grid is 20, and the two agree
    to under 0.005 there -- well inside the width of every interval drawn against it.
    """
    n1 = n2 = max(n / 2.0, 1.0)
    sd = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    return float(0.5 + stats.norm.isf(alpha) * sd + 0.5 / (n1 * n2))


# --------------------------------------------------------------------------- load

def load(root) -> pd.DataFrame:
    """Every score CSV under ``root``, as one frame with the derived columns.

    ``root`` is a ``<eval>/<dataset>/<model>__seed<k>.csv`` tree. Files are read
    individually rather than concatenated on disk so a truncated one -- the normal
    state while a campaign is still writing -- is skipped with a name rather than
    poisoning the frame.
    """
    frames, skipped = [], []
    for path in sorted(glob.glob(str(Path(root) / "*" / "*" / "*.csv"))):
        try:
            d = pd.read_csv(path)
        except Exception:                                    # truncated mid-write
            skipped.append(path)
            continue
        if d.empty or "score" not in d.columns:
            skipped.append(path)
            continue
        frames.append(d)
    if not frames:
        raise SystemExit(f"no readable score CSV under {root}")
    sc = pd.concat(frames, ignore_index=True)
    if skipped:
        print(f"  [perf_io] skipped {len(skipped)} unreadable/empty CSV")

    sc["dataset"] = sc["dataset"].map(DATASET_KEY).fillna(sc["dataset"])
    # `align` is the method and `align_level` the scope; one tag is what every figure
    # groups by, and "easubject" is the tag the claim directories already use.
    sc["align_tag"] = np.where(sc["align"].eq("none"), "none",
                               "ea" + sc["align_level"].fillna(""))
    sc["family"] = sc["model"].map(family_of)
    sc["arch"] = sc["model"].map(arch_of)
    sc["metric"] = np.where(sc["dataset"].isin(AUC_DATASETS), "roc_auc", "accuracy")
    sc["chance"] = np.where(sc["metric"].eq("roc_auc"), 0.5, 1.0 / sc["n_classes"])

    # The trial count the score is estimated on. Within-session is 5-fold CV over the
    # whole session, so every trial is a test trial exactly once and the effective n is
    # `samples`; the two cross-* evaluations score one held-out block, so it is
    # `samples_test`. Getting this backwards would move every chance threshold by sqrt5.
    sc["n_eff"] = np.where(sc["eval"].eq("within_session"),
                           sc["samples"], sc["samples_test"]).astype(int)
    sc["above_chance"] = [
        s > (auc_threshold(n) if m == "roc_auc" else binom_threshold(n, c))
        for s, n, m, c in zip(sc.score, sc.n_eff, sc.metric, sc.chance)]
    # Distance above the null in units of the null's own spread, so datasets with
    # different chance levels and different trial counts are on one axis.
    sc["excess"] = sc["score"] - sc["chance"]
    return sc


def attach_params(sc: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Add the parameter count of the model that produced each score.

    The size of a *growing* arm is a per-fold outcome, not a config constant -- that is
    the whole point of the arm -- so it cannot be looked up from a table and has to
    come from the history export. Joined on the full fold key and then averaged per
    cell: a Pareto point for an arm whose folds ended at different widths is a mean of
    real widths, and the figure that draws it says so.
    """
    keys = ["eval", "dataset", "model", "align_tag", "seed", "subject"]
    # The history export leaves `align_tag` NaN on the unaligned arm where the score
    # CSVs write "none". Joining on a NaN key silently drops exactly half the grid --
    # and drops it *uniformly*, so the loss looks like a plausible coverage gap rather
    # than the key mismatch it is. Normalised here, before the join, not after.
    fits = fits.copy()
    fits["align_tag"] = fits["align_tag"].fillna("none")
    p = (fits.groupby(keys, as_index=False)
             .agg(n_params=("params_end", "mean"), width_end=("width_end", "mean"),
                  target_width=("target_width", "mean"),
                  epochs=("epochs", "mean"), seconds=("seconds", "mean")))
    p["subject"] = pd.to_numeric(p["subject"], errors="coerce")
    out = sc.copy()
    out["subject"] = pd.to_numeric(out["subject"], errors="coerce")
    return out.merge(p, on=keys, how="left")


# ----------------------------------------------------------------- unit of analysis

def by_subject(sc: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    """One row per (eval, align, model, dataset, subject): the independent unit.

    Sessions and seeds are averaged, not stacked. Two sessions of one subject are the
    same electrodes on the same head and three seeds are the same data; treating them
    as independent is what inflated the p-values on this grid by four orders of
    magnitude. Every test below consumes this and only this.
    """
    agg = {value: "mean", "above_chance": "mean", "chance": "first",
           "metric": "first", "n_eff": "sum", "samples": "mean"}
    for col in ("n_params", "width_end", "target_width", "epochs", "seconds"):
        if col in sc.columns:
            agg[col] = "mean"
    out = (sc.groupby(UNIT, as_index=False)
             .agg(n_rows=(value, "size"), **{k: (k, v) for k, v in agg.items()}))
    out["family"] = out["model"].map(family_of)
    out["arch"] = out["model"].map(arch_of)
    return out


def paired(subj: pd.DataFrame, a: str, b: str, value: str = "score") -> pd.DataFrame:
    """``a - b`` on the subjects both arms scored, within (eval, align, dataset).

    Inner join, never a reindex: an arm that has not been run on a dataset must drop
    out of the comparison, not enter it as a zero. On a campaign that is still writing
    -- and on ``cross_subject``, where the fixed controls were never scheduled at all
    -- the un-joined difference is silently a comparison against nothing.
    """
    keys = ["eval", "align_tag", "dataset", "subject"]
    left = subj[subj.model == a].set_index(keys)
    right = subj[subj.model == b].set_index(keys)
    common = left.index.intersection(right.index)
    if len(common) == 0:
        return pd.DataFrame(columns=keys + ["delta"])
    d = (left.loc[common, value] - right.loc[common, value]).rename("delta")
    out = d.reset_index()
    out["a_score"] = left.loc[common, value].to_numpy()
    out["b_score"] = right.loc[common, value].to_numpy()
    out["a_above"] = left.loc[common, "above_chance"].to_numpy()
    out["b_above"] = right.loc[common, "above_chance"].to_numpy()
    return out


def boot_ci(x: np.ndarray, boot: int = BOOT) -> tuple[float, float]:
    """Percentile bootstrap of the mean, resampling *subjects*."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mde(x: np.ndarray, power: float = 0.80, alpha: float = ALPHA) -> float:
    """Smallest true effect this many subjects would detect, at the observed spread.

    The number that separates "we measured no effect" from "we could not have seen
    one". A null reported without it is not a null: on the 9-subject datasets the
    detectable effect is several points, which is larger than most of the effects being
    argued about. Two-sided, normal approximation, paired.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    sd = float(np.std(x, ddof=1))
    z = stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power)
    return z * sd / np.sqrt(len(x))


def test(delta: np.ndarray) -> dict:
    """Effect, interval, sign test and power for one paired contrast at subject level."""
    x = np.asarray(delta, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(n=len(x), mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                    win=np.nan, n_win=0, mde=np.nan, sd=np.nan)
    lo, hi = boot_ci(x)
    p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    return dict(n=len(x), mean=float(x.mean()), lo=lo, hi=hi, p=p,
                win=float((x > 0).mean()), n_win=int((x > 0).sum()),
                mde=mde(x), sd=float(np.std(x, ddof=1)))


def holm(ps) -> np.ndarray:
    """Holm-Bonferroni within a family of contrasts tested together."""
    ps = np.asarray(ps, dtype=float)
    ok = np.isfinite(ps)
    out = np.full(len(ps), np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(ps[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


def decompose(subj: pd.DataFrame, arch: str, evaluation: str,
              align_tag: str = "none") -> dict | None:
    """``grow - bd = (grow - fix) + (fix - bd)`` for one architecture and protocol.

    Returns None when the fixed control is missing, which is the *planned* state on
    ``cross_subject`` and not an incomplete one. Reporting ``grow - bd`` there without
    saying the decomposition is unavailable is what credits growth with a codebase
    effect; measured on ``bnci2014_001/within_session``, that mis-attribution is
    +5.06 points of the +4.83 total.
    """
    bd, fix, grow = TRIPLES[arch]
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    total = paired(sel, grow, bd)
    if total.empty:
        return None
    out = {"arch": arch, "eval": evaluation, "align_tag": align_tag,
           "total": test(total.delta.to_numpy()),
           "n_datasets": total.dataset.nunique(), "has_control": False}
    growth, codebase = paired(sel, grow, fix), paired(sel, fix, bd)
    if not growth.empty and not codebase.empty:
        out["growth"] = test(growth.delta.to_numpy())
        out["codebase"] = test(codebase.delta.to_numpy())
        out["has_control"] = True
        # The share of the total the control accounts for. Undefined and left NaN when
        # the total is near zero, where a ratio is noise amplified rather than a number.
        t, c = out["total"]["mean"], out["codebase"]["mean"]
        out["codebase_share"] = c / t if abs(t) > 1e-6 else np.nan
    return out
