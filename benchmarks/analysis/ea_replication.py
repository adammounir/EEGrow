"""Stage 1 gate: does our Euclidean Alignment reproduce the published effect?

THE TARGET, fixed before the run so it cannot be fitted after it. Junqueira,
Aristimunha, Chevallier & de Camargo (arXiv 2401.10746), Table 2 -- BNCI2014_001, LOSO,
2-class left/right hand:

    No-EA 68.93 +/- 12.61  ->  Offline-EA 73.98 +/- 11.21     delta = +5.05 pp

WHAT IS AND IS NOT COMPARED. The absolute level is NOT the test. Their harness is not
ours -- different early stopping, different selection rule, different seeds -- so a
level that lands 4 points either side of 68.93 says nothing. The paired per-subject
DELTA is the test, because everything that separates the two harnesses is shared
between our raw and aligned arms and cancels in the difference.

UNIT OF ANALYSIS: the held-out subject. A cross_subject cell writes 18 rows, which are
9 held-out subjects x 2 SESSIONS of each -- and the seeds are a third layer of
replication on top. Sessions and seeds are both averaged INSIDE a subject before
anything is tested. n = 9. At n=9 a two-sided Wilcoxon floors at p=0.0039
(attained iff all 9 subjects agree), so the readout is the effect size and its
bootstrap CI; the p-value is there to be reported, not to be the verdict.

WHY THIS IS A GATE AND NOT A RESULT. The question the project cares about is the
INTERACTION (does alignment pay a growing net more than a fixed one -- see
`ea_interaction.py`). That is a difference of differences, and it is only interpretable
if each difference is real. This script's only job is to say whether ours is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Table 2 of arXiv 2401.10746, BNCI2014_001 LOSO, in accuracy points.
PAPER = {"no_ea": 0.6893, "ea": 0.7398, "delta": 0.0505}
MODELS = ["bd_eegnet", "bd_shallow", "bd_deep4"]
RNG = np.random.default_rng(0)
BOOT = 20000


def load(root: Path) -> pd.DataFrame:
    """Every cross_subject CSV under `root`, raw and aligned arms together.

    The arm is read off the `align` COLUMN, not off the filename: the column is written
    by the run itself and survives a concat, which a naming convention does not.
    """
    files = sorted(root.rglob("cross_subject/bnci2014_001/*.csv"))
    if not files:
        raise SystemExit(f"no cross_subject results under {root}")
    d = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if "align" not in d.columns:
        raise SystemExit("results carry no `align` column: these are pre-EA runs")
    return d


def boot_ci(a: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(a), size=(BOOT, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def wilcoxon(a: np.ndarray) -> float:
    return float(stats.wilcoxon(a).pvalue) if np.abs(a).sum() > 0 else 1.0


def paired_delta(d: pd.DataFrame, models: list[str]) -> pd.Series:
    """EA minus raw, per held-out subject, sessions and seeds averaged inside it.

    Restricted to the points BOTH arms scored: a cell whose aligned run failed must drop
    out of the raw arm too, or the delta silently becomes a comparison of two different
    subject sets.

    THE KEY MUST INCLUDE `session`. A cross_subject cell writes 18 rows -- 9 held-out
    subjects x 2 sessions -- so a (model, subject, seed) key is duplicated twice over.
    Aligning two Series on a duplicated index is not a pairing: `.loc[common]` returns
    the cartesian product of the matching rows on each side, which silently doubles the
    row count and pairs session 0 of one arm against session 1 of the other. It also
    made the completeness check report 27 phantom half-paired points on a tree where
    every cell was complete. The key is the full unit; the collapse to the subject
    happens afterwards, deliberately.
    """
    # `d["align"]`, never `d.align`: DataFrame.align is a pandas METHOD, so the
    # attribute form compares a bound method to a string, yields False, and pandas then
    # reads False as a column label. Same trap as `d.eval`, and it fails loudly here
    # only by luck -- with a boolean column in the frame it would filter silently.
    d = d[d["model"].isin(models)]
    key = ["model", "subject", "session", "seed"]
    ea = d[d["align"] == "euclidean"].set_index(key)["score"]
    raw = d[d["align"] == "none"].set_index(key)["score"]
    for name, s in (("euclidean", ea), ("none", raw)):
        if s.index.duplicated().any():
            raise SystemExit(
                f"the {name} arm has duplicate {tuple(key)} rows: pairing on a "
                "duplicated index would silently take a cartesian product")
    common = ea.index.intersection(raw.index)
    if len(common) == 0:
        raise SystemExit(f"no {tuple(key)} point has both arms")
    dropped = len(ea.index.union(raw.index)) - len(common)
    if dropped:
        print(f"  [warn] {dropped} {tuple(key)} points have only one arm "
              "and are excluded from every number below")
    delta = (ea.loc[common] - raw.loc[common]).sort_index()
    # sessions and seeds are both replication of the same held-out subject
    return delta.groupby(level=["model", "subject"]).mean()


def line(name: str, a: np.ndarray) -> str:
    lo, hi = boot_ci(a)
    return (f"  {name:<26}{a.mean() * 100:+6.2f} pp  [{lo * 100:+5.2f}, {hi * 100:+5.2f}]"
            f"  p={wilcoxon(a):.2g}  win={np.mean(a > 0):.0%}  n={len(a)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    d = load(args.root)

    print("=" * 84)
    print("LEVELS  (mean over subjects and seeds; NOT the test -- harnesses differ)")
    print("=" * 84)
    lvl = d.pivot_table(index="model", columns="align", values="score", aggfunc="mean")
    print((lvl * 100).round(2).to_string())
    pooled = d[d["model"].isin(MODELS)].groupby("align")["score"].mean()
    print(f"\n  pooled over the paper's 3 nets:  no-EA {pooled.get('none', np.nan)*100:.2f}"
          f"  ->  EA {pooled.get('euclidean', np.nan)*100:.2f}")
    print(f"  paper (Table 2):                 no-EA {PAPER['no_ea']*100:.2f}"
          f"  ->  EA {PAPER['ea']*100:.2f}")

    print("\n" + "=" * 84)
    print("THE GATE: paired EA - raw, per held-out subject (seeds averaged within)")
    print("=" * 84)
    per_sub = paired_delta(d, MODELS)

    # Pooled over the three nets, which is the quantity Table 2's delta is: one number
    # per subject, so the nets are averaged inside a subject rather than stacked -- three
    # nets on the same subject are not three independent observations.
    pooled_sub = per_sub.groupby(level="subject").mean().to_numpy()
    print(line("POOLED (3 nets)", pooled_sub))
    print(f"  {'paper target':<26}{PAPER['delta'] * 100:+6.2f} pp\n")

    for m in MODELS:
        if m in per_sub.index.get_level_values("model"):
            print(line(m, per_sub.xs(m, level="model").to_numpy()))

    lo, hi = boot_ci(pooled_sub)
    print("\n" + "=" * 84)
    verdict = ("PASS" if lo > 0 else
               "FAIL -- CI includes 0" if hi > 0 else
               "FAIL -- alignment HURTS")
    print(f"VERDICT: {verdict}")
    if lo > 0:
        # A positive CI is necessary but not sufficient: an effect an order of magnitude
        # off the published one is a different phenomenon, not a replication.
        ratio = pooled_sub.mean() / PAPER["delta"]
        print(f"  effect is {ratio:.2f}x the published +5.05 pp"
              + ("  (same order -- replication holds)" if 0.4 <= ratio <= 2.5 else
                 "  (WRONG ORDER OF MAGNITUDE -- investigate before building on it)"))
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
