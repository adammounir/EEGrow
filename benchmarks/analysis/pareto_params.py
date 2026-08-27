"""What does a point of accuracy cost in parameters?

THE ARGUMENT
------------
The benchmark's growing arms do not uniformly beat the fixed arms on accuracy, and
chasing that comparison alone concedes the interesting question. A grown network reaches
its accuracy at a size it *discovered*; a fixed network reaches its accuracy at a size a
human chose in advance. If the grown one lands on a better accuracy-per-parameter
frontier, that is a statement about learning, and it holds even where the accuracy
column is a tie.

`params_end` is recorded for every fit in v5, so this costs no compute.

TWO COST AXES, because they answer different questions:

  params_end    what the deployed model costs. The number to quote for an implant or a
                headset.
  param_epochs  the integral of parameter count over training epochs -- what the model
                cost to TRAIN. Growing arms start small, so they can win here even when
                they end at the same size. This is the axis a fixed net cannot fake by
                being pruned afterwards.

CREDIBILITY GATE
----------------
A model that sits at chance has a wonderful accuracy-per-parameter ratio in the sense
that it wastes none, and including it would produce a frontier made of broken models.
So the frontier is computed only over cells where the arm is above its own exact chance
threshold (see chance_level.py). This is the join that makes the plot honest.

Usage: python pareto_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results_v5_published"
CHANCE = HERE / "chance" / "cell_chance_verdicts.csv"
OUT = HERE / "pareto"

# Each growing arm and the fixed arm it is meant to be compared against. `fix_deepeeg`
# is the size-matched control for `grow_deep`; `bd_deep4` is the off-the-shelf net.
COUNTERPART = {"grow_shallow": "bd_shallow", "grow_deep": "bd_deep4",
               "grow_eegnex": "bd_eegnex", "grow_sccnet": "bd_sccnet"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CHANCE.exists():
        sys.exit(f"run chance_level.py first (missing {CHANCE})")

    cells = pd.read_csv(CHANCE)
    fits = pd.read_csv(RESULTS / "eegrow_v5_fits.csv.gz")
    budget = pd.read_csv(RESULTS / "eegrow_v5_budget.csv.gz")

    cost = (fits.groupby(["eval", "dataset", "model"])
                .agg(params_start=("params_start", "median"),
                     params_end=("params_end", "median"),
                     width_end=("width_end", "median"),
                     reached_target=("reached_target", "mean"))
                .reset_index())
    train_cost = (budget.groupby(["eval", "dataset", "model"])
                        .agg(param_epochs=("param_epochs", "median"),
                             epochs=("epochs", "median"))
                        .reset_index())

    df = (cells.merge(cost, on=["eval", "dataset", "model"], how="inner")
               .merge(train_cost, on=["eval", "dataset", "model"], how="left"))
    df["credible"] = df["score_mean"] >= df["threshold"]
    df.to_csv(OUT / "cells_with_cost.csv", index=False)
    print(f"{len(df)} deep-learning cells with a parameter count "
          f"({df['credible'].sum()} above their own chance threshold)")

    # ------------------------------------------------- 1. head-to-head vs counterpart
    idx = df.set_index(["eval", "dataset", "model"])
    rows = []
    for (ev, ds, mo), r in idx.iterrows():
        if mo not in COUNTERPART:
            continue
        try:
            f = idx.loc[(ev, ds, COUNTERPART[mo])]
        except KeyError:
            continue
        rows.append(dict(
            eval=ev, dataset=ds, grow=mo, fixed=COUNTERPART[mo],
            grow_score=r.score_mean, fixed_score=f.score_mean,
            d_score=r.score_mean - f.score_mean,
            grow_params=r.params_end, fixed_params=f.params_end,
            param_ratio=f.params_end / r.params_end,
            grow_pe=r.param_epochs, fixed_pe=f.param_epochs,
            pe_ratio=f.param_epochs / r.param_epochs if r.param_epochs else np.nan,
            both_credible=bool(r.credible and f.credible)))
    h2h = pd.DataFrame(rows)
    h2h.to_csv(OUT / "head_to_head.csv", index=False)

    ok = h2h[h2h["both_credible"]]
    print()
    print("=" * 78)
    print("ACCURACY vs PARAMETERS, growing arm against its fixed counterpart")
    print(f"(credible cells only: {len(ok)} of {len(h2h)})")
    print("=" * 78)
    summ = (ok.groupby("grow")
              .agg(n=("d_score", "size"), d_score=("d_score", "mean"),
                   param_ratio=("param_ratio", "median"),
                   pe_ratio=("pe_ratio", "median"),
                   n_smaller=("param_ratio", lambda x: int((x > 1).sum())),
                   n_better=("d_score", lambda x: int((x > 0).sum())))
              .reset_index())
    summ["cheaper_and_not_worse"] = [
        int(((ok[ok.grow == g].param_ratio > 1) &
             (ok[ok.grow == g].d_score > -0.01)).sum()) for g in summ["grow"]]
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------- 2. Pareto frontier
    def frontier(g: pd.DataFrame) -> pd.DataFrame:
        """Non-dominated points: nothing is both smaller AND at least as accurate."""
        g = g.sort_values("params_end")
        keep, best = [], -np.inf
        for _, r in g.iterrows():
            if r.score_mean > best:
                keep.append(r.name)
                best = r.score_mean
        return g.loc[keep]

    cred = df[df["credible"]]
    fronts = []
    for (ev, ds), g in cred.groupby(["eval", "dataset"]):
        f = frontier(g)
        f = f.assign(on_front=True)
        fronts.append(f[["eval", "dataset", "model", "family", "score_mean",
                         "params_end", "param_epochs", "on_front"]])
    front = pd.concat(fronts)
    front.to_csv(OUT / "pareto_front.csv", index=False)

    print()
    print("=" * 78)
    print("HOW OFTEN IS EACH ARM ON THE ACCURACY/PARAMETER PARETO FRONT?")
    print("(per dataset x eval; credible cells only)")
    print("=" * 78)
    n_cells = cred.groupby("model").size().rename("cells")
    n_front = front.groupby("model").size().rename("on_front")
    tab = pd.concat([n_cells, n_front], axis=1).fillna(0)
    tab["rate"] = tab["on_front"] / tab["cells"]
    tab["median_params"] = cred.groupby("model")["params_end"].median()
    print(tab.sort_values("rate", ascending=False)
             .to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------- 3. the headline: cheaper at no cost in accuracy
    print()
    print("=" * 78)
    print("CELLS WHERE A GROWING ARM IS SMALLER AND NOT WORSE (delta > -0.01)")
    print("=" * 78)
    win = ok[(ok["param_ratio"] > 1) & (ok["d_score"] > -0.01)]
    print(f"{len(win)} of {len(ok)} credible comparisons "
          f"({100 * len(win) / max(len(ok), 1):.0f}%)")
    if len(win):
        print(win[["eval", "dataset", "grow", "fixed", "grow_score", "fixed_score",
                   "d_score", "grow_params", "fixed_params", "param_ratio"]]
              .sort_values("param_ratio", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
