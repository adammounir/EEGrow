"""Growth vs its width-matched fixed control, paired, and whether the protocol moves it.

WHAT THIS ANSWERS THAT budget_models.py DOES NOT. That script reads each model's own
2x2 and says how much the shipped default undertrained it. It never compares two models,
so the numbers it prints cannot support the claim the benchmark exists to make -- that
growing to width W is better (or not) than building at width W. Reading that off two
columns of means is an unpaired comparison across subjects whose spread is several times
the effect; it also cannot say whether the budget fix CHANGES the verdict, which is the
finding: the growth advantage on the deep pair looked like +0.007 under the shipped
protocol and is +0.001 once both arms are trained properly.

TWO FAMILIES OF PAIRS, and they answer different questions. Taken from the configs.

  GROWTH (the claim): grow_X vs fix_X. The SAME class built frozen at the geometry
  growth ends on -- same file, same init path, same callbacks; only `_can_grow`
  differs. This is the only contrast in which growth is the sole difference.

  REFERENCE (a different question): grow_X vs bd_X, i.e. our implementation against
  braindecode's at the same nominal width. A difference here is growth PLUS everything
  that separates the two codebases -- measured, braindecode's stock Xavier init starts
  bd_shallow 0.34 nats above ln(k) at epoch 0 against +0.08 for ours. Worth reporting,
  never as evidence about growth.

The benchmark originally had only ONE same-class control (fix_deepeeg, for the deep
arm) and read the other three off the braindecode references. That is what conflated
the two families. Note also that grow_eegnex was never width-matched to bd_eegnex at
all: `filter_1` sizes both the growable junction and the fixed tail conv, growth widens
only the junction, so the grown net ends with 70 classifier inputs where bd_eegnex has
280 (see fix_eegnex.yaml). Its -0.089 belongs in the reference family, not the growth
one.

bd_deep4 is deliberately in NEITHER family for grow_deep: it is a 4-stage 25/50/100/200
net against a 2-stage w1=8 one, which measures the architecture and nothing else.

UNIT OF ANALYSIS. The 36 rows per cell are 9 subjects x 2 sessions x 2 seeds. They are
not 36 independent observations: the two seeds are internal replication of the same
subject-session, and the two sessions share a subject. A Wilcoxon on 36 correlated rows
overstates its own significance. The subject-level test (n=9, averaging sessions and
seeds within a subject) is the one that holds for a paper; the row-level test is printed
next to it only to show how much of the significance was coming from the correlation.
With n=9 the two-sided Wilcoxon cannot go below p=0.0039, so read effect sizes and CIs,
not stars.

Usage: python growth_contrast.py --root <grid_models dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# See the module docstring: two families, tested and Holm-corrected separately because
# they are two questions, not one question asked twice.
FAMILIES = {
    "GROWTH -- growing vs the SAME class frozen at the end geometry": [
        ("grow_shallow", "fix_shallow"), ("grow_deep", "fix_deepeeg"),
        ("grow_eegnex", "fix_eegnex"), ("grow_sccnet", "fix_sccnet")],
    "REFERENCE -- our implementation vs braindecode's (NOT evidence about growth)": [
        ("grow_shallow", "bd_shallow"), ("grow_eegnex", "bd_eegnex"),
        ("grow_sccnet", "bd_sccnet")],
}
ARMS = ["p20_loss", "p20_acc", "full_loss", "full_acc"]
SHIPPED, FIXED_PROTOCOL = "p20_loss", "full_acc"
UNIT = ["dataset", "subject", "session", "seed"]
RNG = np.random.default_rng(0)
BOOT = 20000


def load(root: Path) -> pd.DataFrame:
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for arm in ARMS:
            for p in sorted((mdir / arm).rglob(f"{mdir.name}__seed*.csv")):
                d = pd.read_csv(p)
                d["arm"], d["model"] = arm, mdir.name
                rows.append(d)
    return pd.concat(rows, ignore_index=True)


def boot_ci(d: np.ndarray) -> tuple[float, float]:
    """Paired bootstrap over units. Resamples units, not rows within a unit."""
    idx = RNG.integers(0, len(d), size=(BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def test(d: pd.Series) -> dict:
    a = d.to_numpy(dtype=float)
    lo, hi = boot_ci(a)
    p = float(stats.wilcoxon(a).pvalue) if np.abs(a).sum() > 0 else 1.0
    return dict(n=len(a), mean=float(a.mean()), lo=lo, hi=hi, p=p,
                win=float((a > 0).mean()))


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni within a family. The family here is the four pairs: one
    decision is made per pair, and they are tested together."""
    order = np.argsort(ps)
    out = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out.tolist()


def by_subject(s: pd.Series) -> pd.Series:
    """Collapse sessions and seeds into one number per subject: the independent unit."""
    return s.groupby(level=["dataset", "subject"]).mean()


def fmt(r: dict) -> str:
    return (f"{r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
            f"p={r['p']:.2g}  win={r['win']:.0%}  n={r['n']}")


def build_deltas(sc: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict:
    """delta[pair][arm] = growing - control, paired on the units both models scored.

    A pair whose control has not been run yet is skipped rather than reported empty:
    the fixed controls land as a separate SLURM array, so a partially-populated tree is
    the normal state and must not look like a null result.
    """
    out: dict[tuple[str, str], dict[str, pd.Series]] = {}
    for g, f in pairs:
        per_arm = {}
        for arm in ARMS:
            sg = sc[(sc.model == g) & (sc.arm == arm)].set_index(UNIT)["score"]
            sf = sc[(sc.model == f) & (sc.arm == arm)].set_index(UNIT)["score"]
            common = sg.index.intersection(sf.index)
            if len(common) < 5:
                per_arm = {}
                break
            per_arm[arm] = (sg.loc[common] - sf.loc[common]).sort_index()
        if per_arm:
            out[(g, f)] = per_arm
        else:
            print(f"  [skipped] {g} vs {f}: no complete square for the control yet")
    return out


def report(title: str, deltas: dict) -> None:
    for level, collapse in (("SUBJECT-LEVEL (n=9, the independent unit)", by_subject),
                            ("ROW-LEVEL (n=36, correlated -- shown for contrast)",
                             lambda s: s)):
        print("=" * 100)
        print(f"{title} -- {level}")
        print("=" * 100)
        print("mean delta [bootstrap 95% CI]  p(Wilcoxon)  win = share of units where "
              "growing wins\n")
        for arm in ARMS:
            res = {pair: test(collapse(d[arm])) for pair, d in deltas.items()}
            hp = holm([r["p"] for r in res.values()])
            tag = {SHIPPED: "  <- the shipped protocol",
                   FIXED_PROTOCOL: "  <- the corrected protocol"}.get(arm, "")
            print(f"  {arm}{tag}")
            for (pair, r), ph in zip(res.items(), hp):
                print(f"    {pair[0]:<13} vs {pair[1]:<12} {fmt(r)}  holm={ph:.2g}")
            print()

        # Does fixing the protocol change the verdict? A difference in differences on
        # the SAME units, so it is a paired test too, not a comparison of two p-values.
        print("  DID THE PROTOCOL MOVE THE VERDICT?  "
              f"(delta@{FIXED_PROTOCOL} - delta@{SHIPPED}, paired)")
        res = {pair: test(collapse(d[FIXED_PROTOCOL]) - collapse(d[SHIPPED]))
               for pair, d in deltas.items()}
        hp = holm([r["p"] for r in res.values()])
        for (pair, r), ph in zip(res.items(), hp):
            print(f"    {pair[0]:<13} vs {pair[1]:<12} {fmt(r)}  holm={ph:.2g}")
        print()

    print("=" * 100)
    print(f"VERDICT TABLE -- {title.split(' --')[0]} (subject-level means)")
    print("=" * 100)
    print(f"{'pair':<32}{'shipped':>12}{'corrected':>12}{'shift':>12}   sign flip?")
    for pair, d in deltas.items():
        a = float(by_subject(d[SHIPPED]).mean())
        b = float(by_subject(d[FIXED_PROTOCOL]).mean())
        flip = "YES" if np.sign(a) != np.sign(b) else ""
        print(f"{pair[0] + ' vs ' + pair[1]:<32}{a:>12.4f}{b:>12.4f}"
              f"{b - a:>+12.4f}   {flip}")
    print("\nA sign flip means the published direction for that pair was an artefact of"
          "\nthe training protocol, not a property of the models.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    sc = load(args.root)

    # Holm runs within a family, never across the two: the growth question and the
    # implementation question are answered separately, so correcting them together
    # would penalise each for the other's comparisons.
    for title, pairs in FAMILIES.items():
        deltas = build_deltas(sc, pairs)
        if not deltas:
            print(f"\n{title}: nothing to report yet\n")
            continue
        report(title, deltas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
