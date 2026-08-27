"""Figures for the chance-level and accuracy-per-parameter analyses.

Same convention as `v5_figures`: functions take already-loaded frames and return the
Figure, so a path change breaks in one place and each figure stays reviewable.

Two claims, two figures.

  chance_geometry     what a score has to exceed before it means anything, and how much
                      of the between-subject spread we plot is trial-sampling noise. The
                      point is that these are properties of the DATASET, fixed before
                      any model is trained -- so they say which cells could ever have
                      carried a result.

  accuracy_per_param  the growing arms against their fixed counterparts on two axes at
                      once. Anything in the upper-right quadrant is smaller AND better,
                      which is the claim that does not depend on winning the accuracy
                      column outright.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_LABEL = {"within_session": "within-session", "cross_session": "cross-session",
              "cross_subject": "cross-subject (LOSO)"}
GROW_LABEL = {"grow_shallow": "ShallowFBCSPNet", "grow_sccnet": "SCCNet",
              "grow_eegnex": "EEGNeX", "grow_deep": "DeepEEGNet"}
GROW_COLOR = {"grow_shallow": "C0", "grow_sccnet": "C1",
              "grow_eegnex": "C2", "grow_deep": "C3"}


def chance_geometry(geom: pd.DataFrame, cells: pd.DataFrame):
    """Left: how far above nominal chance a single score must sit. Right: noise share."""
    f, (a0, a1) = plt.subplots(1, 2, figsize=(13, 6.5))

    g = geom.sort_values("excess_needed")
    lab = [f"{r.dataset} · {EVAL_LABEL.get(r['eval'], r['eval'])}  (n={int(r.n_eff)})"
           for _, r in g.iterrows()]
    y = np.arange(len(g))
    a0.barh(y, g["excess_needed"], color=np.where(g["metric"] == "accuracy",
                                                  "C0", "C1"))
    a0.set_yticks(y, lab, fontsize=7)
    a0.set_xlabel("how far above nominal chance the 95% threshold sits")
    a0.set_title("A single score has to clear this before it means anything\n"
                 "(exact one-sided null: binomial for accuracy, Mann-Whitney for AUC)",
                 fontsize=10)
    a0.axvline(0, color="k", lw=0.8)
    for i, (_, r) in enumerate(g.iterrows()):
        a0.text(r.excess_needed + 0.004, i, f"{r.chance:.2f}→{r.threshold:.2f}",
                va="center", fontsize=6)
    a0.set_xlim(0, g["excess_needed"].max() * 1.35)
    a0.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l)
                       for c, l in [("C0", "accuracy"), ("C1", "roc_auc")]],
              loc="lower right", fontsize=8)

    ns = (cells.groupby(["dataset", "eval"])
               .agg(n_eff=("n_eff", "median"), noise=("noise_share", "mean"),
                    obs=("obs_sd", "mean"), samp=("sampling_sd", "mean"))
               .reset_index().sort_values("noise"))
    y = np.arange(len(ns))
    a1.barh(y, ns["noise"], color=np.where(ns["noise"] > 0.4, "C3", "C7"))
    a1.set_yticks(y, [f"{r.dataset} · {EVAL_LABEL.get(r['eval'], r['eval'])}"
                      f"  (n={int(r.n_eff)})" for _, r in ns.iterrows()], fontsize=7)
    a1.set_xlabel("share of between-subject variance that is trial-sampling noise")
    a1.set_title("How much of the error bar is the measurement, not the subject?\n"
                 "(red: above 40% — more seeds cannot reduce this)", fontsize=10)
    a1.axvline(0.4, color="C3", ls=":", lw=1)
    a1.set_xlim(0, 1)
    f.tight_layout()
    return f


def accuracy_per_param(h2h: pd.DataFrame):
    """Parameter ratio against accuracy delta, one point per credible cell."""
    ok = h2h[h2h["both_credible"]]
    f, (a0, a1) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, xcol, xlab in [
            (a0, "param_ratio", "parameters of the fixed net / parameters of the grown net"),
            (a1, "pe_ratio", "parameter-epochs of fixed / parameter-epochs of grown")]:
        for g, sub in ok.groupby("grow"):
            ax.scatter(sub[xcol], sub["d_score"], s=46, alpha=0.85,
                       color=GROW_COLOR.get(g, "C7"), label=GROW_LABEL.get(g, g),
                       edgecolor="k", linewidth=0.4, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        # Shade in DATA coordinates: the quadrant starts at ratio 1, not at some
        # fraction of the axis, and on a log axis those are nowhere near each other.
        x0, x1 = ax.get_xlim()
        y1 = ok["d_score"].max() * 1.15
        ax.fill_between([1, x1], 0, y1, color="C2", alpha=0.07, zorder=0)
        ax.set_xlim(x0, x1)
        ax.axhline(0, color="k", lw=0.8, zorder=2)
        ax.axvline(1, color="k", lw=0.8, zorder=2)
    a0.set_ylabel("accuracy of grown − accuracy of fixed")
    a0.set_title("Deployed cost", fontsize=11)
    a1.set_title("Training cost", fontsize=11)
    a0.legend(fontsize=9, loc="lower right")
    n_ur = int(((ok["param_ratio"] > 1) & (ok["d_score"] > 0)).sum())
    f.suptitle(f"Smaller and no worse: {n_ur} of {len(ok)} credible cells sit in the "
               f"upper-right quadrant\n(cells where both arms clear their own exact "
               f"chance threshold)", fontsize=11)
    f.tight_layout()
    return f


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).resolve().parent
    out = here / "figures" / "chance_pareto"
    out.mkdir(parents=True, exist_ok=True)

    geom = pd.read_csv(here / "chance" / "chance_geometry.csv")
    cells = pd.read_csv(here / "chance" / "cell_chance_verdicts.csv")
    h2h = pd.read_csv(here / "pareto" / "head_to_head.csv")

    for name, fig in [("chance_geometry", chance_geometry(geom, cells)),
                      ("accuracy_per_param", accuracy_per_param(h2h))]:
        p = out / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"wrote {p}")
