"""Read out the bd_deep4 budget/selection experiment (see exp_deep4_budget.py).

Primary: held-out accuracy, paired on (dataset, subject, session, seed) against the
`p20_loss` baseline. Paired because every arm scores the same subject-sessions, and the
unpaired spread across subjects on this dataset is several times the effect size.

Secondary, and the reason this generalises: `restored_epoch` and the epoch the fit ends
on. `selection_monitor` is a global knob, so a selection effect here is a claim about
the benchmark, not about one model.

Also computes the fourth, unrun cell of the 2x2 -- patience 200 with selection on
valid_loss. It is bit-identical to `p20_loss` iff argmin(valid_loss) < 25 on every
full-budget fit, since the trajectories coincide for the first 24 epochs. This checks
that rather than assuming it.

Usage: python deep4_budget.py --root <exp output dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ["p20_loss", "p20_acc", "full_acc", "full_cos", "full_loss"]
BASE = "p20_loss"
# From the chance audit: 288 test trials, 4 classes, exact binomial upper 95% bound.
CHANCE_95 = 0.295


def load_fits(root: Path) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for p in sorted((root / arm).rglob("bd_deep4__seed*__fits.jsonl")):
            seed = int(p.stem.split("seed")[1].split("__")[0])
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                h = r.get("history") or []
                if not h:
                    continue
                tr = [e.get("train_loss") for e in h]
                vl = [e.get("valid_loss") for e in h]
                va = [e.get("valid_acc") for e in h]
                vl_ok = [v for v in vl if v is not None]
                rows.append(dict(
                    arm=arm, seed=seed, fit=r.get("fit"), epochs=len(h),
                    restored_epoch=r.get("restored_epoch"),
                    stop_reason=r.get("stop_reason"),
                    final_tr=tr[-1], min_tr=float(np.nanmin(np.array(tr, float))),
                    # 1-indexed epoch of the valid_loss minimum: the fourth cell of
                    # the 2x2 is free exactly when this stays under 25.
                    ep_min_vloss=int(np.nanargmin(np.array(vl, float))) + 1,
                    ep_max_vacc=int(np.nanargmax(np.array(va, float))) + 1,
                    min_vloss=float(np.nanmin(np.array(vl, float))),
                    max_vloss=float(np.nanmax(np.array(vl, float))),
                    best_vacc=float(np.nanmax(np.array(va, float))),
                    last_vacc=float(va[-1]),
                    vloss_blowup=bool(len(vl_ok) > 1
                                      and max(vl_ok) > 2 * min(vl_ok)),
                    # Is the train loss still descending when the run ends? If yes the
                    # budget bound; if no, the net converged and the budget was enough.
                    still_descending=bool(len(tr) > 5 and tr[-1] < tr[-5]),
                    lr_final=h[-1].get("lr")))
    return pd.DataFrame(rows)


def load_scores(root: Path) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for p in sorted((root / arm).rglob("bd_deep4__seed*.csv")):
            d = pd.read_csv(p)
            d["arm"] = arm
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 220)

    fits = load_fits(args.root)
    if fits.empty:
        print("no fit records found"); return 1
    present = [a for a in ARMS if a in set(fits.arm)]

    print("=" * 78)
    print("TRAINING: how long did it run, and which epoch got scored?")
    print("=" * 78)
    g = (fits.groupby("arm")
             .agg(n=("epochs", "size"), epochs=("epochs", "median"),
                  restored=("restored_epoch", "median"),
                  ep_min_vloss=("ep_min_vloss", "median"),
                  ep_max_vacc=("ep_max_vacc", "median"),
                  final_tr=("final_tr", "median"),
                  best_vacc=("best_vacc", "median"),
                  blowup=("vloss_blowup", "mean"),
                  descending=("still_descending", "mean"),
                  lr_final=("lr_final", "median"))
             .reindex(present))
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  blowup = fraction of fits where max(valid_loss) > 2 x min(valid_loss)")
    print("  descending = fraction still improving train loss over the last 5 epochs")

    full = fits[fits.arm.isin(["full_acc", "full_cos"])]
    if not full.empty:
        late = full[full.ep_min_vloss >= 25]
        print("\n" + "-" * 78)
        print("The unrun 4th cell (patience 200 + selection on valid_loss)")
        print("-" * 78)
        if late.empty:
            print(f"  argmin(valid_loss) < 25 on all {len(full)} full-budget fits "
                  f"(max {int(full.ep_min_vloss.max())}) -> that cell is bit-identical "
                  f"to {BASE} and did not need running.")
        else:
            print(f"  {len(late)}/{len(full)} full-budget fits have their valid_loss "
                  f"minimum at epoch >= 25 (max {int(full.ep_min_vloss.max())}). "
                  "The cell is NOT free; run it before claiming the 2x2.")

    sc = load_scores(args.root)
    if sc.empty:
        print("\n(no score CSVs yet)"); return 0
    print("\n" + "=" * 78)
    print("PRIMARY: held-out accuracy")
    print("=" * 78)
    print(f"(exact 95% chance threshold for this cell: {CHANCE_95}; v5 scored 0.272)")
    print(sc.groupby("arm")["score"].agg(["size", "mean", "std"])
            .reindex([a for a in ARMS if a in set(sc.arm)])
            .to_string(float_format=lambda v: f"{v:.4f}"))

    unit = ["dataset", "subject", "session", "seed"]
    b = sc[sc.arm == BASE].set_index(unit)["score"]
    for arm in [a for a in ARMS if a != BASE and a in set(sc.arm)]:
        s = sc[sc.arm == arm].set_index(unit)["score"]
        common = b.index.intersection(s.index)
        if len(common) < 5:
            continue
        d = s.loc[common] - b.loc[common]
        w = stats.wilcoxon(d) if d.abs().sum() > 0 else None
        print(f"\n  {arm} - {BASE}, paired on {len(common)} units: "
              f"mean {d.mean():+.4f}, median {d.median():+.4f}, "
              f"wins {int((d > 0).sum())}/{len(d)}"
              + (f", Wilcoxon p={w.pvalue:.2g}" if w else ""))

    factorial(sc, unit)
    return 0


# The 2x2 read as a factorial rather than as three contrasts against one baseline.
# It has to be read this way: the simple effects do NOT add up. Budget alone is
# +0.057, selection alone is -0.011, and together they are +0.132 -- so quoting
# either main effect on its own misstates the finding.
CELLS = {("p20", "loss"): "p20_loss", ("p20", "acc"): "p20_acc",
         ("full", "loss"): "full_loss", ("full", "acc"): "full_acc"}


def factorial(sc: pd.DataFrame, unit: list[str]) -> None:
    have = set(sc.arm)
    if not all(a in have for a in CELLS.values()):
        print("\n(2x2 incomplete -- factorial readout skipped)")
        return
    s = {k: sc[sc.arm == a].set_index(unit)["score"] for k, a in CELLS.items()}
    common = s[("p20", "loss")].index
    for v in s.values():
        common = common.intersection(v.index)
    s = {k: v.loc[common] for k, v in s.items()}

    print("\n" + "=" * 78)
    print(f"FACTORIAL: budget x selection, paired on {len(common)} units")
    print("=" * 78)
    print(f"{'':>14}{'sel=valid_loss':>16}{'sel=valid_acc':>16}")
    for b, label in (("p20", "patience  20"), ("full", "patience 200")):
        print(f"{label:>14}{s[(b, 'loss')].mean():>16.4f}"
              f"{s[(b, 'acc')].mean():>16.4f}")

    def contrast(name: str, d: pd.Series) -> None:
        w = stats.wilcoxon(d) if d.abs().sum() > 0 else None
        print(f"  {name:<46} {d.mean():+.4f}"
              + (f"   p={w.pvalue:.2g}" if w else ""))

    print("\nSimple effects")
    contrast("budget, at selection=valid_loss",
             s[("full", "loss")] - s[("p20", "loss")])
    contrast("budget, at selection=valid_acc",
             s[("full", "acc")] - s[("p20", "acc")])
    contrast("selection, at patience 20",
             s[("p20", "acc")] - s[("p20", "loss")])
    contrast("selection, at patience 200",
             s[("full", "acc")] - s[("full", "loss")])

    print("\nInteraction (difference of differences)")
    # Non-zero => the two knobs are not separable and neither may be reported alone.
    contrast("(full_acc - full_loss) - (p20_acc - p20_loss)",
             (s[("full", "acc")] - s[("full", "loss")])
             - (s[("p20", "acc")] - s[("p20", "loss")]))
    contrast("full_acc - p20_loss  (both knobs vs the shipped default)",
             s[("full", "acc")] - s[("p20", "loss")])


if __name__ == "__main__":
    raise SystemExit(main())
