"""Read out the budget x selection 2x2 across the eight non-ML arms (grid 500573).

The claim under test is NOT about any one model. `patience` and `selection_monitor` are
global knobs of the benchmark, so the question is how many arms the shipped default
(`patience=20` + selection on `valid_loss`) was undertraining. bd_deep4 is analysed
separately by deep4_budget.py; this script covers the other eight and is the one that
decides whether v5's deep cells have to be re-run.

Every contrast is PAIRED on (dataset, subject, session, seed): each arm scores the same
subject-sessions, and the unpaired spread across subjects on bnci2014_001 is several
times the effect size. Unpaired means would hide the effect in the between-subject
variance.

The square is read as a factorial, never as three contrasts against one baseline. On
bd_deep4 the simple effects did not add up -- budget alone +0.057, selection alone
-0.011, together +0.132, interaction +0.085 (p=1.1e-07) -- so a main effect quoted on
its own misstates the finding. The interaction column below is what says whether that
carries over per arm.

For the growing arms there is a second readout that the fixed arms do not have: the
width at the epoch that got restored. Probe 500569 measured grow_shallow reaching its
target 40/40 in BOTH arms while `valid_loss` selection handed back a 25-30 filter
model. When that happens the score is not the score of the architecture the benchmark
claims to measure, and accuracy alone does not reveal it.

Usage: python budget_models.py --root /scratch/amounir/eegrow_budget/grid_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# (budget, selection) -> arm directory name. Same four cells as the bd_deep4 square.
CELLS = {("p20", "loss"): "p20_loss", ("p20", "acc"): "p20_acc",
         ("full", "loss"): "full_loss", ("full", "acc"): "full_acc"}
ARMS = list(CELLS.values())
# The cell the benchmark actually shipped, and therefore the reference every headline
# number is quoted against.
SHIPPED = "p20_loss"
UNIT = ["dataset", "subject", "session", "seed"]
# From the chance audit: 288 test trials, 4 classes, exact binomial upper 95% bound.
CHANCE_95 = 0.295


def load_scores(root: Path) -> pd.DataFrame:
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for arm in ARMS:
            for p in sorted((mdir / arm).rglob(f"{mdir.name}__seed*.csv")):
                d = pd.read_csv(p)
                d["arm"], d["model"] = arm, mdir.name
                rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_fits(root: Path) -> pd.DataFrame:
    """One row per fit, with the growth readout the accuracy CSVs cannot carry."""
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for arm in ARMS:
            for p in sorted((mdir / arm).rglob(f"{mdir.name}__seed*__fits.jsonl")):
                seed = int(p.stem.split("seed")[1].split("__")[0])
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    h = r.get("history") or []
                    if not h:
                        continue
                    rest = r.get("restored_epoch")
                    rows.append(dict(
                        model=mdir.name, arm=arm, seed=seed, epochs=len(h),
                        restored_epoch=rest,
                        final_tr=h[-1].get("train_loss"),
                        width_final=_width_upto(h, len(h)),
                        width_at_restored=_width_upto(h, rest) if rest else None,
                        growth_events=sum(1 for e in h if e.get("grow_applied"))))
    return pd.DataFrame(rows)


def _width_upto(hist: list[dict], n: int | None) -> int | None:
    """Width in effect at epoch ``n``.

    ``grow_width_after`` is recorded ONLY on growth epochs (see the docstring of
    ``skorch_integration._record_growth``), so reading the key straight off an
    arbitrary epoch returns None four times out of five. The width in effect is the
    last value recorded at or before that epoch.
    """
    if not n:
        return None
    seen = None
    for e in hist[:n]:
        v = e.get("grow_width_after")
        if v is not None:
            seen = v
    return seen


def contrast(d: pd.Series) -> tuple[float, float | None]:
    if d.abs().sum() == 0:
        return 0.0, None
    return float(d.mean()), float(stats.wilcoxon(d).pvalue)


def square(sc: pd.DataFrame, subject_level: bool = False) -> dict | None:
    """The 2x2 for one model, paired on the units all four cells share.

    ``subject_level`` collapses the two sessions and the two seeds of a subject into one
    number before testing. The 36 rows of a cell are NOT 36 independent observations --
    the seeds are internal replication of the same subject-session and the sessions share
    a subject -- so the row-level Wilcoxon overstates its own significance, by four orders
    of magnitude on some contrasts (see analysis/growth_contrast.py). The subject-level
    test (n=9) is the one that holds for a paper. Note its floor: a two-sided Wilcoxon on
    9 pairs cannot go below p=0.0039, so read the effect, not the star.
    """
    s = {k: sc[sc.arm == a].set_index(UNIT)["score"] for k, a in CELLS.items()}
    if any(v.empty for v in s.values()):
        return None
    common = s[("p20", "loss")].index
    for v in s.values():
        common = common.intersection(v.index)
    if len(common) < 5:
        return None
    s = {k: v.loc[common] for k, v in s.items()}
    if subject_level:
        s = {k: v.groupby(level=["dataset", "subject"]).mean() for k, v in s.items()}
    out = {"n": len(next(iter(s.values())))}
    for k, v in s.items():
        out[f"cell_{k[0]}_{k[1]}"] = float(v.mean())
    out["budget_at_loss"] = contrast(s[("full", "loss")] - s[("p20", "loss")])
    out["budget_at_acc"] = contrast(s[("full", "acc")] - s[("p20", "acc")])
    out["sel_at_p20"] = contrast(s[("p20", "acc")] - s[("p20", "loss")])
    out["sel_at_full"] = contrast(s[("full", "acc")] - s[("full", "loss")])
    out["interaction"] = contrast((s[("full", "acc")] - s[("full", "loss")])
                                  - (s[("p20", "acc")] - s[("p20", "loss")]))
    out["both_vs_shipped"] = contrast(s[("full", "acc")] - s[("p20", "loss")])
    return out


def fmt(c: tuple[float, float | None]) -> str:
    v, p = c
    return f"{v:+.4f}" + (f" (p={p:.1g})" if p is not None else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--subject-level", action="store_true",
                    help="collapse sessions and seeds per subject before testing (n=9)")
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    sc = load_scores(args.root)
    if sc.empty:
        print("no score CSVs found under", args.root)
        return 1

    print("=" * 100)
    print("COVERAGE -- cells finished so far")
    print("=" * 100)
    cov = (sc.groupby(["model", "arm"]).size().unstack(fill_value=0)
             .reindex(columns=ARMS, fill_value=0))
    print(cov.to_string())
    print("\n(a complete cell is 36 rows: 9 subjects x 2 sessions x 2 seeds)")

    rows = []
    for model, g in sc.groupby("model"):
        r = square(g, subject_level=args.subject_level)
        if r is None:
            print(f"\n{model}: square incomplete, skipped")
            continue
        rows.append((model, r))

    if not rows:
        print("\nno complete square yet")
        return 0

    print("\n" + "=" * 100)
    print("THE SQUARE, per model (mean held-out accuracy, paired)")
    print("=" * 100)
    print(f"chance 95% for this cell = {CHANCE_95}\n")
    print(f"{'model':<14}{'n':>4}{'p20/loss':>11}{'p20/acc':>10}"
          f"{'full/loss':>11}{'full/acc':>10}   {'shipped cell below chance?':<26}")
    for model, r in rows:
        shipped = r["cell_p20_loss"]
        print(f"{model:<14}{r['n']:>4}{shipped:>11.4f}{r['cell_p20_acc']:>10.4f}"
              f"{r['cell_full_loss']:>11.4f}{r['cell_full_acc']:>10.4f}   "
              f"{'YES' if shipped < CHANCE_95 else '':<26}")

    print("\n" + "=" * 100)
    print("CONTRASTS (paired, Wilcoxon)")
    print("=" * 100)
    for model, r in rows:
        print(f"\n{model}  (n={r['n']})")
        print(f"    budget      @sel=valid_loss  {fmt(r['budget_at_loss'])}")
        print(f"    budget      @sel=valid_acc   {fmt(r['budget_at_acc'])}")
        print(f"    selection   @patience 20     {fmt(r['sel_at_p20'])}")
        print(f"    selection   @patience 200    {fmt(r['sel_at_full'])}")
        print(f"    INTERACTION                  {fmt(r['interaction'])}")
        print(f"    both vs shipped default      {fmt(r['both_vs_shipped'])}")

    print("\n" + "=" * 100)
    print("HEADLINE -- how many arms did the shipped default undertrain?")
    print("=" * 100)
    print(f"{'model':<14}{'both vs shipped':>20}{'interaction':>20}"
          f"{'additive?':>12}")
    for model, r in rows:
        v, p = r["both_vs_shipped"]
        iv, ip = r["interaction"]
        # "Additive" = the two knobs can be reported separately. If not, neither main
        # effect may be quoted alone -- the bd_deep4 lesson.
        add = "no" if (ip is not None and ip < 0.05) else "yes"
        print(f"{model:<14}{fmt((v, p)):>20}{fmt((iv, ip)):>20}{add:>12}")

    # Empty when only the score CSVs were copied off the cluster (the fit JSONLs are
    # ~250 MB and are usually left there), so guard rather than index into nothing.
    fits = load_fits(args.root)
    grew = fits if fits.empty else fits[fits.width_final.notna()
                                        & (fits.growth_events > 0)]
    if not grew.empty:
        print("\n" + "=" * 100)
        print("GROWING ARMS -- was the SCORED model the one that grew?")
        print("=" * 100)
        # Per fit, not per group: the median of the widths hides the effect, because
        # most fits restore at full width and a minority lose a lot. On the subject-1
        # probe width_lost reached 15 of 40; the median over all subjects is 1. Both
        # are true and only the distribution says which matters.
        grew = grew.assign(width_lost=grew.width_final - grew.width_at_restored)
        w = (grew.groupby(["model", "arm"])
                 .agg(n=("width_final", "size"),
                      epochs=("epochs", "median"),
                      restored=("restored_epoch", "median"),
                      width_final=("width_final", "median"),
                      events=("growth_events", "median"),
                      lost_median=("width_lost", "median"),
                      lost_p90=("width_lost", lambda s: s.quantile(0.9)),
                      lost_max=("width_lost", "max"),
                      frac_narrowed=("width_lost", lambda s: (s > 0).mean())))
        print(w.to_string(float_format=lambda v: f"{v:.2f}"))
        print("\n  width_lost = width_final - width_at_the_scored_epoch."
              "\n  frac_narrowed = fraction of fits where selection handed back a"
              "\n  NARROWER net than training produced. When it does, the score is not"
              "\n  the score of the architecture the benchmark claims to measure --"
              "\n  and accuracy alone never reveals it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
