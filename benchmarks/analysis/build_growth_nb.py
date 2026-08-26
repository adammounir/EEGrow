"""Generate benchmarks/analysis/eegrow_growth.ipynb (training dynamics / growth)."""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/Users/adammounir/Desktop/Inria/Exploration/eegrow/benchmarks/analysis/"
           "eegrow_growth.ipynb")
ROOT = sys.argv[2] if len(sys.argv) > 2 else "../results_smoke"

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Training dynamics of the growing arms

The score tables say *whether* a growing model won. They cannot say **what it did** --
how wide it actually grew, whether it reached the width it was compared against, or
what capacity it paid for along the way. None of that was observable until the fit
records existed: growth width was passed to `logger.info` and the benchmark built the
callback with `verbose=False`, so no run of the published grid recorded it.

That blind spot had a cost. `GrowingShallowFBCSPNet` and `GrowingDeepEEGNet` did not
declare `target_width`, the attribute the growth callback reads, so their growth was
never capped -- measured 8 -> 77 against a target of 32. A whole campaign ran without
it being visible, because there was no observable that would have shown it. **Figure 2
is the standing check that it stays fixed.**

Three choices this notebook makes rather than assumes:

1. **The unit is a fit, not a cell.** One cross-validation fold trains one model with
   its own trajectory. Averaging trajectories across folds before looking at them
   would hide exactly the variability worth seeing.
2. **Width is a step function.** It changes at growth events and is constant between
   them, so it is drawn with `steps-post`. A straight line between two growths would
   draw widths the model never had.
3. **The budget axis is parameter-epochs, not final parameters.** A growing model
   spends most of its epochs narrower than it ends, so its final width overstates what
   it cost. Summing `n_params` over epochs is what the efficiency claim is actually
   about.
""")

code(r"""
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "..")                    # benchmarks/ -> analysis/growth_io.py
sys.path.insert(0, ".")
import growth_io

ROOT = Path("%s")          # <- the only path to change for another campaign
FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GROW_C, FIX_C = "#C2255C", "#4C6EF5"

fits, curves = growth_io.load(ROOT)
print(f"{len(fits)} fits, {len(curves)} epoch records")
print(f"cells: {fits.groupby(['eval','dataset','model']).ngroups}")
fits[["eval","dataset","model","seed","fit","width_start","width_end",
      "target_width","epochs","max_epochs","params_start","params_end"]].head(8)
""" % ROOT)

md(r"""
## 1. Did every growable model reach the width it is compared against?

The paired comparison only means "growing to width W versus starting at W" if growth
actually lands on W. Two ways it fails: the cap is not connected (the model overshoots)
or early stopping cuts the fit before enough growth events have fired (it undershoots).
The second is not a bug but it changes what the comparison measures, so it has to be
reported either way.
""")

code(r"""
g = fits[fits["target_width"].notna() & (fits["target_width"] > fits["width_start"])]
if g.empty:
    print("no growable fits in this campaign (frozen arms only)")
else:
    summary = (g.assign(over=g.width_end > g.target_width,
                        under=g.width_end < g.target_width)
                .groupby(["model"])
                .agg(fits=("fit","size"),
                     reached=("reached_target","mean"),
                     overshot=("over","mean"),
                     undershot=("under","mean"),
                     width_end_med=("width_end","median"),
                     target=("target_width","max"),
                     stopped_early=("stopped_early","mean")))
    display(summary.round(3))
    bad = g[g.width_end > g.target_width]
    assert bad.empty, (
        f"{len(bad)} fits grew past target_width -- the cap is not connected. "
        "This is the regression that AUDIT_PRE_RERUN.md A1 describes.")
    print("OK: no fit exceeded its target width.")
""")

md(r"""
## Figure 1 -- width over training

One line per fit. The dashed rule is the target width, i.e. the width of the fixed
model this arm is paired against. A trajectory that flattens below the rule is a model
that stopped before it finished growing; one that crosses it is a broken cap.
""")

code(r"""
g = fits[fits["target_width"].notna() & (fits["target_width"] > fits["width_start"])]
models = sorted(g["model"].unique())
if not models:
    print("nothing growable to draw")
else:
    cw = curves.merge(g[["eval","dataset","model","seed","fit","target_width"]],
                      on=["eval","dataset","model","seed","fit"])
    panels = sorted(cw.groupby(["model","dataset"]).groups)
    n = len(panels)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5*ncol, 2.7*nrow),
                             squeeze=False, sharex=True)
    for ax, (model, ds) in zip(axes.ravel(), panels):
        sub = cw[(cw.model == model) & (cw.dataset == ds)]
        for _, one in sub.groupby(["seed","fit"]):
            one = one.sort_values("epoch")
            ax.plot(one["epoch"], one["width"], drawstyle="steps-post",
                    color=GROW_C, alpha=.45, lw=1.2)
        tgt = sub["target_width"].iloc[0]
        ax.axhline(tgt, ls="--", lw=1, color="0.35")
        ax.annotate(f"target {int(tgt)}", xy=(1, tgt), xycoords=("axes fraction","data"),
                    ha="right", va="bottom", fontsize=7, color="0.35")
        ax.set_title(f"{model} · {ds}", fontsize=8.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("growable width")
        ax.set_ylim(0, max(tgt, sub["width"].max()) * 1.18)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Growth trajectory — one line per cross-validation fold", y=1.0)
    fig.tight_layout(); fig.savefig(FIGS/"05_width_trajectory.png", bbox_inches="tight")
    plt.show()
""")

md(r"""
## Figure 2 -- final width against target

The compact form of the same check, and the one to look at first after any change to
the growth code. Every point on the diagonal means the cap holds and every fit finished
growing. Points below it are early stops; a point above it is the bug returning.
""")

code(r"""
g = fits[fits["target_width"].notna() & (fits["target_width"] > fits["width_start"])]
if g.empty:
    print("no growable fits")
else:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    rng = np.random.default_rng(0)
    for model, sub in g.groupby("model"):
        jitter = rng.uniform(-.18, .18, len(sub))
        ax.scatter(sub["target_width"] + jitter, sub["width_end"], s=26, alpha=.6,
                   label=f"{model} (n={len(sub)})")
    lo = 0
    hi = max(g["target_width"].max(), g["width_end"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.4", lw=1, zorder=0)
    ax.annotate("reached target", xy=(hi*.62, hi*.66), fontsize=8, color="0.4",
                rotation=38)
    ax.fill_between([lo, hi], [lo, hi], hi, color="#C92A2A", alpha=.055, zorder=0)
    ax.annotate("cap broken", xy=(hi*.28, hi*.9), fontsize=8, color="#C92A2A")
    ax.set_xlabel("target width (= width of the paired fixed model)")
    ax.set_ylabel("width reached")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Where growth actually stopped")
    fig.tight_layout(); fig.savefig(FIGS/"06_width_vs_target.png", bbox_inches="tight")
    plt.show()
""")

md(r"""
## Figure 3 -- capacity paid for, not capacity ended with

Left: parameter count per epoch, growing versus its fixed control. Right: the running
sum, which is the budget the efficiency argument is about. The gap between the two
curves at the end of training is the compute a growing model did **not** spend.

Both arms are drawn from the same records: the fixed arm is recorded too, and its
constant width is what makes the comparison legible.
""")

code(r"""
pairs = {"grow_shallow": "bd_shallow", "grow_sccnet": "bd_sccnet",
         "grow_deep": "fix_deepeeg", "grow_eegnex": "bd_eegnex"}
have = {g: f for g, f in pairs.items()
        if {g, f} <= set(curves["model"].unique())}
if not have:
    print("no complete (growing, fixed) pair in this campaign:",
          sorted(curves["model"].unique()))
else:
    fig, axes = plt.subplots(len(have), 2, figsize=(9, 3.0*len(have)), squeeze=False)
    for row, (grow, fixed) in enumerate(have.items()):
        for ax_i, (col, ylab) in enumerate(
                [("n_params", "parameters"), ("cum", "cumulative parameter-epochs")]):
            ax = axes[row][ax_i]
            for model, colour, lab in ((fixed, FIX_C, "fixed"), (grow, GROW_C, "growing")):
                sub = curves[curves.model == model].sort_values("epoch")
                if sub.empty:
                    continue
                sub = sub.assign(cum=sub.groupby(["eval","dataset","seed","fit"])
                                 ["n_params"].cumsum())
                m = sub.groupby("epoch")[col].mean()
                q1 = sub.groupby("epoch")[col].quantile(.1)
                q9 = sub.groupby("epoch")[col].quantile(.9)
                ax.plot(m.index, m.values, color=colour, lw=1.8, label=f"{lab} ({model})")
                ax.fill_between(m.index, q1.values, q9.values, color=colour, alpha=.15)
            ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
            if ax_i == 0:
                ax.set_title(f"{grow} vs {fixed}", fontsize=9, loc="left")
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Capacity over training: instantaneous (left) and cumulative (right)",
                 y=1.0)
    fig.tight_layout(); fig.savefig(FIGS/"07_capacity.png", bbox_inches="tight")
    plt.show()
""")

md(r"""
## Figure 4 -- learning curves, with the growth events marked

The question a reviewer asks about any growth method: does adding neurons mid-training
disturb the optimisation? The vertical ticks are the epochs where a growth actually
changed the width. If validation accuracy dipped systematically right after them, the
optimizer-state transfer would be failing to do its job.
""")

code(r"""
if not have:
    print("no pair to draw")
else:
    fig, axes = plt.subplots(1, len(have), figsize=(4.2*len(have), 3.4), squeeze=False)
    for ax, (grow, fixed) in zip(axes[0], have.items()):
        for model, colour, lab in ((fixed, FIX_C, "fixed"), (grow, GROW_C, "growing")):
            sub = curves[(curves.model == model) & curves["valid_acc"].notna()]
            if sub.empty:
                continue
            m = sub.groupby("epoch")["valid_acc"].mean()
            sd = sub.groupby("epoch")["valid_acc"].std()
            ax.plot(m.index, m.values, color=colour, lw=1.8, label=lab)
            ax.fill_between(m.index, m - sd, m + sd, color=colour, alpha=.15)
        gs = curves[curves.model == grow].sort_values("epoch")
        if not gs.empty:
            gs = gs.assign(jump=gs.groupby(["eval","dataset","seed","fit"])["width"].diff())
            events = sorted(gs.loc[gs["jump"] > 0, "epoch"].unique())
            for e in events:
                ax.axvline(e, color=GROW_C, alpha=.20, lw=1, zorder=0)
            ax.annotate("growth events", xy=(.02, .04), xycoords="axes fraction",
                        color=GROW_C, fontsize=7.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
        ax.set_title(f"{grow} vs {fixed}", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS/"08_learning_curves.png", bbox_inches="tight")
    plt.show()
""")

md(r"""
## Figure 5 -- accuracy against the budget that bought it

Each point is one fit: what it cost on the x-axis (parameter-epochs, log scale), what
it reached on the y-axis. The claim a growing method has to support is not "higher
accuracy" but **"the same accuracy further left"** -- and if the growing cloud sits
left of the fixed cloud at equal height, that is visible here and nowhere in the score
tables.
""")

code(r"""
budget = growth_io.parameter_epochs(curves)
if budget["best_valid_acc"].isna().all():
    print("no validation accuracy recorded (train_split=None?) -- nothing to plot")
else:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model, sub in budget.groupby("model"):
        grow = model.startswith("grow")
        ax.scatter(sub["param_epochs"], sub["best_valid_acc"], s=30, alpha=.65,
                   color=GROW_C if grow else FIX_C,
                   marker="o" if grow else "s", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("parameter-epochs (log)"); ax.set_ylabel("best validation accuracy")
    ax.set_title("Same accuracy for a smaller budget is the claim; further left is better")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS/"09_budget_pareto.png", bbox_inches="tight")
    plt.show()
    display(budget.groupby("model")[["param_epochs","epochs","params_end",
                                     "best_valid_acc"]].median().round(4))
""")

md(r"""
## 6. Early stopping versus growth

`grow_every=5` with `max_epochs=200` allows up to 39 growth events, but `EarlyStopping`
has patience 20 on validation accuracy. If fits routinely stop at epoch 30, a model
that grows one filter per event never approaches its target -- and the paired contrast
silently becomes "a narrow model versus a wide one" rather than a test of growth.

This is a property of the training schedule, not a defect, but it decides how the
paired result must be worded. The table below is what the wording has to respect.
""")

code(r"""
g = fits[fits["target_width"].notna() & (fits["target_width"] > fits["width_start"])]
if g.empty:
    print("no growable fits")
else:
    tbl = (g.assign(gap=g.target_width - g.width_end,
                    frac=(g.width_end - g.width_start) /
                         (g.target_width - g.width_start))
             .groupby("model")
             .agg(fits=("fit","size"),
                  median_epochs=("epochs","median"),
                  max_epochs=("max_epochs","max"),
                  stopped_early=("stopped_early","mean"),
                  frac_of_growth_done=("frac","median"),
                  median_gap_to_target=("gap","median")))
    display(tbl.round(3))
    if (tbl["frac_of_growth_done"] < 0.99).any():
        print("At least one arm does not finish growing before training ends.")
        print("The paired claim must then be stated as 'grown to width W_reached',")
        print("not 'grown to the reference width'.")
""")

md(r"""
## 7. Provenance of these records

The score rows now carry their own regime (`sfreq`, resampling target, band-pass,
package versions, eegrow commit). That is what makes a campaign auditable after the
fact without the Hydra run directories -- whose deletion by an `rsync --delete` is what
left 1170 cells of the published grid untraceable. Data that carries its regime cannot
lose it.
""")

code(r"""
csvs = sorted(ROOT.glob("*/*/*__seed*.csv"))
csvs = [p for p in csvs if "__fits" not in p.name]
if not csvs:
    print("no score CSV next to the fit records")
else:
    d = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    cols = [c for c in ("sfreq","resample_cfg","fmin","fmax","n_chans_in","n_times_in",
                        "device","v_moabb","v_braindecode","v_torch","v_gromo",
                        "eegrow_sha") if c in d.columns]
    if not cols:
        print("these CSVs predate row-level provenance:", list(d.columns))
    else:
        display(d[cols].drop_duplicates().T)
        rates = d["sfreq"].unique()
        assert len(rates) == 1, f"mixed sampling rates in one campaign: {rates}"
        print(f"single sampling rate across {len(d)} rows: {rates[0]} Hz")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
