"""What happens *inside* a growing fit: optimisation, growth decisions, stopping.

`explore_curves` answers "how did the score get there" from the loss and accuracy
curves. It cannot answer anything about the growth mechanism itself, because the columns
that describe a growth step -- the candidate spectrum, the line-search factor, the
first-order gain gromo expected -- were recorded from the first campaign onward and, up
to this module, read by nothing but the exporter. That is the gap this fills.

THE FOUR QUESTIONS, and which figures answer them:

1. *Is the optimiser healthy?* Gradient norm, learning rate, AdamW's eps attenuation.
   A growth step rebuilds the optimizer, so every one of these has a discontinuity that
   a fixed control does not -- and the fixed control is drawn on the same axes for
   exactly that reason.

2. *Was a growth step a decision or a scheduled event?* The candidate spectrum says how
   many neurons were on offer and how far apart they were; `grow_n_kept` says how many
   survived the relative floor; `grow_s` says what the line search did with them. A step
   that always proposes the same count, always keeps all of it, and always lands at
   s = 1 is a schedule wearing a decision's clothes.

3. *Did the decision pay?* gromo computes a first-order predicted improvement before
   applying a step. The realised change in training loss over that same epoch is
   recorded too. Plotting one against the other is the only check in the codebase that
   the surrogate gromo optimises is predictive of the loss it is a surrogate for.

4. *When and where did the fit stop?* `stop_reason` and `restored_epoch` per fold, and
   -- since `subject_stamp` -- the held-out subject each fold was fitting. "The model
   stopped early" and "the model stopped early on subjects 3 and 7 of this dataset,
   which are also the two it scores at chance on" are different findings.

TWO TRAPS THE FIGURES HANDLE RATHER THAN INHERIT.

*Survivorship in mean curves.* Folds have unequal length because early stopping fires at
different epochs, so a fold-mean at epoch 150 averages only the folds that survived to
150 -- the ones still improving at 130. The mean rises for a reason that is not
training. Every mean curve here reads ``n_folds`` and switches to dotted below
:data:`SURVIVOR_FRAC` of the fold count, so the survivorship region is visibly marked.

*Growth arms are not width-matched to braindecode's arm at every epoch.* They start
narrow and end at the reference width. A diagnostic read at epoch 10 is read on a model
a quarter of the size, which is the point of the arm and not a confound -- but it means
"the growing arm has a smaller gradient norm" is trivially true early and only
interesting late. The width trace is drawn under the diagnostics for that reason.

All figure functions take the frames from ``export_growth_dynamics`` and return a
matplotlib ``Figure`` (or ``None`` when the frame has nothing to draw, which is the
normal case on a partial campaign). Nothing here writes a file; the driver does.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- style
# Same palette as `explore_figures` so a reader moving between the two reports reads the
# same colour as the same family. Duplicated rather than imported: that module needs a
# score table with a `family` column, and the growth records carry model names only.
FAM_COLOR = {"braindecode": "#1565c0", "fixed control": "#00695c",
             "growing": "#c62828", "other": "#616161"}
FAM_MARKER = {"braindecode": "s", "fixed control": "D", "growing": "^", "other": "o"}

#: TWO axes, two visual channels. Hue = the arm type (blue braindecode, teal fixed
#: control, red growing), because that is the comparison the report is about; lightness
#: = the architecture family, because a figure whose three lines are all `grow_*` -- the
#: abstention and line-search figures are exactly that -- is unreadable if the arm type
#: is the only thing encoded. Marker carries the architecture too, for print and for
#: colour-blind readers.
#:
#: `fixed control` is deliberately NOT a shade of the braindecode blue any more. The
#: contrast those two arms form (`fix - bd`, the codebase term) is the one a reader has
#: to be able to make at a glance, and two blues do not let them.
MODEL_COLOR = {
    "bd_shallow": "#0d47a1", "bd_deep4": "#1e88e5", "bd_sccnet": "#7cb9f2",
    "bd_eegnex": "#b3d4f5",
    "fix_shallow": "#00382f", "fix_deepeeg": "#00897b", "fix_sccnet": "#4db6ac",
    "fix_eegnex": "#a7d9d3",
    "grow_shallow": "#8e1616", "grow_deep": "#e53935", "grow_sccnet": "#f4a0a0",
    "grow_eegnex": "#f8cccc",
}
ARCH_MARKER = {"shallow": "^", "deep4": "s", "deep": "s", "deepeeg": "s",
               "sccnet": "o", "eegnex": "D"}

#: Grouped by architecture family, not by arm type, so the three arms of one comparison
#: sit next to each other on every categorical axis. A reader comparing `grow_shallow`
#: to its own controls should not have to scan across the figure to do it.
MODEL_ORDER = ["bd_shallow", "fix_shallow", "grow_shallow",
               "bd_deep4", "fix_deepeeg", "grow_deep",
               "bd_sccnet", "fix_sccnet", "grow_sccnet",
               "bd_eegnex", "fix_eegnex", "grow_eegnex"]

#: The (braindecode, fixed control, growing) triples. The decomposition
#: ``grow - bd = (grow - fix) + (fix - bd)`` is only readable if the code knows which
#: three arms form a comparison, and the arm names do not say so: `grow_deep`'s controls
#: are `bd_deep4` and `fix_deepeeg`, which share no substring with it.
TRIPLES = {"shallow": ("bd_shallow", "fix_shallow", "grow_shallow"),
           "deep": ("bd_deep4", "fix_deepeeg", "grow_deep"),
           "sccnet": ("bd_sccnet", "fix_sccnet", "grow_sccnet"),
           "eegnex": ("bd_eegnex", "fix_eegnex", "grow_eegnex")}

EVAL_LABEL = {"within_session": "within-session", "cross_session": "cross-session",
              "cross_subject": "cross-subject (LOSO)"}

#: Below this fraction of a cell's maximum fold count, a mean curve is survivorship and
#: is drawn dotted. Half is the conventional cut and the one `explore_curves` uses.
SURVIVOR_FRAC = 0.5

#: The relative floor gromo's candidate selection applies, as configured for the final
#: campaign (``eegrow.training.loop.MIN_SINGULAR_RATIO``). Drawn on the spectrum figure
#: as the line that decides which candidates exist. Hard-coded rather than read back
#: because it is not recorded per fit -- which is itself worth knowing, and the figure
#: says so in its caption rather than implying the value was measured.
MIN_SINGULAR_RATIO = 0.10


def family_of(model: str) -> str:
    if model.startswith("grow"):
        return "growing"
    if model.startswith("bd_"):
        return "braindecode"
    if model.startswith("fix_"):
        return "fixed control"
    return "other"


def _color(model: str) -> str:
    """Per-model colour, falling back to the family's base for an unknown arm."""
    return MODEL_COLOR.get(model, FAM_COLOR[family_of(model)])


def _marker(model: str) -> str:
    return ARCH_MARKER.get(model.split("_", 1)[-1], FAM_MARKER[family_of(model)])


def order_models(names) -> list[str]:
    present = set(names)
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _panels(n: int, ncol: int = 4, w: float = 3.5, h: float = 2.7):
    ncol = min(ncol, max(n, 1))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * w, nrow * h), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


def _note(fig, text: str) -> None:
    """A caption strip under the figure, for the caveat a title cannot carry."""
    fig.text(0.005, -0.012, text, fontsize=7, color="0.35", va="top", ha="left",
             wrap=True)


def _draw_mean(ax, g: pd.DataFrame, col: str, *, color, label=None, lw=1.4, ls="-",
               band: bool = False):
    """Mean curve with the survivorship region dotted, and an optional +-1 sd band.

    The split is on ``n_folds``, not on epoch: a cell whose folds all run the full
    budget has no survivorship region and gets a solid line all the way, which is the
    behaviour a fixed threshold on epoch would get wrong in both directions.
    """
    g = g.sort_values("epoch")
    if col not in g or g[col].isna().all():
        return None
    cutoff = SURVIVOR_FRAC * g.n_folds.max()
    solid = g[g.n_folds >= cutoff]
    line, = ax.plot(solid.epoch, solid[col], color=color, lw=lw, ls=ls, label=label)
    thin = g[g.n_folds < cutoff]
    if len(thin):
        # Bridge the gap so the curve does not appear to break, then continue dotted.
        bridge = pd.concat([solid.tail(1), thin])
        ax.plot(bridge.epoch, bridge[col], color=color, lw=lw * 0.8, ls=":")
    if band and f"{col.replace('_mean', '')}_std" in g:
        sd = g[f"{col.replace('_mean', '')}_std"]
        ax.fill_between(solid.epoch, solid[col] - sd[solid.index],
                        solid[col] + sd[solid.index], color=color, alpha=0.12, lw=0)
    return line


def _growth_epochs(events: pd.DataFrame, ev: str, ds: str, model: str,
                   align: str) -> np.ndarray:
    """Median growth epochs for a cell, for marking on a curve."""
    e = applied(events)
    e = e[(e["eval"] == ev) & (e.dataset == ds)
          & (e.model == model) & (e.align_tag.fillna("") == align)]
    if e.empty:
        return np.array([])
    # Median over folds of the k-th event's epoch: folds grow at the same *cadence*
    # (grow_every) but stop at different times, so the k-th event exists for a
    # shrinking population and its mean epoch would drift with survivorship too.
    e = e.sort_values(["fit", "epoch"])
    e = e.assign(k=e.groupby("fit").cumcount())
    return e.groupby("k").epoch.median().to_numpy()


def _cells(frame: pd.DataFrame, ev: str, align: str) -> pd.DataFrame:
    return frame[(frame["eval"] == ev) & (frame.align_tag.fillna("") == align)]


def applied(events: pd.DataFrame) -> pd.DataFrame:
    """The growth steps that were actually taken.

    ``events`` holds every growth *opportunity* -- every epoch on which ``grow_step``
    ran -- because the abstentions are the majority and dropping them is what made the
    line search look like it always answered 1.0. But a figure about what a step DID
    (neurons added, parameters bought, loss moved) has to filter back down to the ones
    that happened, and doing that in one named place is what keeps a figure from
    quietly averaging a decision with a non-decision.
    """
    if "applied" not in events.columns:  # a frame from the pre-fix exporter
        return events
    return events[events.applied.fillna(False).astype(bool)]


# ============================================================ 1. optimisation health
def gradient_norm_curves(cm: pd.DataFrame, events: pd.DataFrame, *,
                         eval_: str = "within_session", align: str = "",
                         datasets: list[str] | None = None):
    """Gradient norm per epoch, one panel per dataset, growth events marked.

    WHY THIS IS THE FIRST FIGURE. Every claim about growth being a better use of a
    parameter budget assumes the optimiser is doing comparable work in the three arms.
    It is not obviously doing so: a growth step splices in neurons whose AdamW second
    moment is zero, and the gradient the epoch after a step is taken on a network that
    is structurally different from the one the step before was taken on.

    Log scale, because the interesting failure is an order of magnitude and the
    interesting success is a factor of two. The vertical ticks are the median growth
    epochs: if the red curve has no visible feature at them, growth is not perturbing
    the optimisation, which is a result and not an absence of one.
    """
    sub = _cells(cm, eval_, align)
    if sub.empty:
        return None
    ds_list = datasets or sorted(sub.dataset.unique())
    fig, axes = _panels(len(ds_list))
    models = order_models(sub.model.unique())
    for ax, ds in zip(axes, ds_list):
        d = sub[sub.dataset == ds]
        for m in models:
            g = d[d.model == m]
            if g.empty:
                continue
            # Seeds pooled: three seeds of one cell are the same experiment, and
            # drawing them separately triples the ink for no extra information.
            g = g.groupby("epoch", as_index=False).agg(
                grad_norm_mean=("grad_norm_mean", "mean"), n_folds=("n_folds", "sum"))
            _draw_mean(ax, g, "grad_norm_mean", color=_color(m),
                       label=m, ls="-" if m.startswith("grow") else "--")
            for e in _growth_epochs(events, eval_, ds, m, align):
                ax.axvline(e, color=_color(m), alpha=0.16, lw=0.8)
        ax.set_yscale("log")
        ax.set_title(ds, fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("||grad|| (log)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Gradient norm through training — {EVAL_LABEL.get(eval_, eval_)}"
                 f"{' + EA' if align else ''}", fontsize=12)
    fig.tight_layout()
    _note(fig, "Solid = growing arms, dashed = fixed/braindecode. Faint verticals are "
               "median growth epochs. Dotted tail = fewer than half the folds still "
               "running, i.e. survivorship, not training.")
    return fig


def gradient_norm_at_growth(events: pd.DataFrame):
    """Does a growth step disturb the gradient? Before vs after, per event.

    The mechanism that would show up here: new neurons arrive with a zero AdamW second
    moment, so their first update is attenuated by nothing and the norm can jump. If it
    does not, the splice is smooth and `grow_s` (the line search) is doing its job of
    scaling the addition into the existing scale of the network.

    Read the ratio panel, not the scatter: the scatter's spread is dominated by the
    difference between datasets and between epochs, and the paired ratio removes both.
    """
    e = applied(events).dropna(subset=["grad_norm_before", "grad_norm_after"])
    if e.empty:
        return None
    e = e[(e.grad_norm_before > 0) & (e.grad_norm_after > 0)]
    if e.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    for m in order_models(e.model.unique()):
        g = e[e.model == m]
        ax.scatter(g.grad_norm_before, g.grad_norm_after, s=6, alpha=0.25,
                   color=_color(m), marker=_marker(m), label=m, linewidths=0)
    lim = [min(e.grad_norm_before.min(), e.grad_norm_after.min()),
           max(e.grad_norm_before.max(), e.grad_norm_after.max())]
    ax.plot(lim, lim, color="0.3", lw=1, ls="--", label="no change")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("||grad|| on the growth epoch")
    ax.set_ylabel("||grad|| on the next epoch")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    ax.set_title("Every growth event, paired", fontsize=10)

    ax = axes[1]
    e = e.assign(ratio=e.grad_norm_after / e.grad_norm_before)
    models = order_models(e.model.unique())
    ax.boxplot([e[e.model == m].ratio.to_numpy() for m in models],
               tick_labels=models, showfliers=False, widths=0.6)
    ax.axhline(1, color="#c62828", lw=1.2, ls="--", label="unchanged")
    ax.set_yscale("log"); ax.set_ylabel("||grad|| ratio, after / before")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("The same thing as a paired ratio", fontsize=10)

    ax = axes[2]
    # Does the disturbance fade as the network gets wider (later events add
    # proportionally less)? Event index rather than epoch, because the cadence is fixed
    # and the epoch axis would just restate `grow_every`.
    e = e.sort_values(["eval", "dataset", "model", "seed", "fit", "epoch"])
    e = e.assign(k=e.groupby(["eval", "dataset", "model", "align_tag", "seed",
                              "fit"]).cumcount() + 1)
    for m in models:
        g = e[e.model == m].groupby("k").ratio.median()
        ax.plot(g.index, g.to_numpy(), marker=_marker(m), ms=4, color=_color(m),
                label=m, lw=1.3)
    ax.axhline(1, color="0.3", lw=1, ls="--")
    ax.set_xlabel("growth event index within a fold")
    ax.set_ylabel("median ||grad|| ratio")
    ax.legend(fontsize=6); ax.grid(alpha=0.25)
    ax.set_title("Does the disturbance fade with width?", fontsize=10)

    fig.suptitle("What a growth step does to the gradient", fontsize=12)
    fig.tight_layout()
    _note(fig, "New neurons arrive with a zero AdamW second moment, so their first "
               "update carries no attenuation. A ratio at 1 means the line search "
               "absorbed that; a ratio above 1 means it did not.")
    return fig


def learning_rate_and_optimizer(cm: pd.DataFrame, fits: pd.DataFrame, *,
                                eval_: str = "within_session", align: str = ""):
    """The step size that actually ran, per epoch and per fold.

    A flat line is the expected result and still worth drawing, for two reasons that are
    not cosmetic. First, `train.lr_schedule` is *refused* on growing arms
    (``pipelines.py`` raises rather than let skorch anneal an optimizer that growth has
    already replaced), so a growing arm showing a decaying lr would mean that guard has
    a hole. Second, the per-fold stamp is read off the live optimizer rather than the
    config, so a bar that is not identical across arms is an arm that did not run the
    protocol it is being compared under.
    """
    sub = _cells(cm, eval_, align)
    if sub.empty or "lr_mean" not in sub or sub.lr_mean.isna().all():
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2),
                            gridspec_kw={"width_ratios": [1.4, 1, 1]})

    ax = axes[0]
    for m in order_models(sub.model.unique()):
        g = sub[sub.model == m].groupby("epoch", as_index=False).agg(
            lr_mean=("lr_mean", "mean"), n_folds=("n_folds", "sum"))
        _draw_mean(ax, g, "lr_mean", color=_color(m), label=m,
                   ls="-" if m.startswith("grow") else "--")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25)
    ax.set_title("Learning rate through training", fontsize=10)

    f = _cells(fits, eval_, align)
    ax = axes[1]
    models = order_models(f.model.unique())
    for i, m in enumerate(models):
        v = f[f.model == m].opt_lr.dropna().unique()
        ax.scatter([i] * len(v), v, s=28, color=_color(m), marker=_marker(m))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("distinct optimiser lr across folds")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("One dot per distinct value:\nmore than one = a protocol break",
                 fontsize=10)

    ax = axes[2]
    for i, m in enumerate(models):
        v = f[f.model == m].opt_eps.dropna().unique()
        ax.scatter([i] * len(v), v, s=28, color=_color(m), marker=_marker(m))
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("AdamW eps (log)")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("eps, the parameter never set explicitly", fontsize=10)

    fig.suptitle(f"Optimiser, as it actually ran — {EVAL_LABEL.get(eval_, eval_)}",
                 fontsize=12)
    fig.tight_layout()
    return fig


def adam_eps_attenuation(cm: pd.DataFrame, events: pd.DataFrame, *,
                         eval_: str = "within_session", align: str = "",
                         datasets: list[str] | None = None):
    """How close AdamW's eps comes to dominating its own denominator.

    The hypothesis this instrument was built to test: on gradients this small, AdamW's
    default eps = 1e-8 sits close enough to sqrt(v) that the update is attenuated, and
    growth's collapsing gradients would push it over. The measurement refuted the
    mechanism -- the margin is ~1.7e6, not ~1 -- so what this figure shows is a
    *negative* result, and it is drawn precisely so the negative stays visible instead
    of being remembered as untested.

    `adam_atten_p05` is the 5th percentile over parameters, i.e. the worst-attenuated
    coordinates, not the average one. If eps ever bites it bites there first.
    """
    sub = _cells(cm, eval_, align)
    cols = [c for c in ("adam_atten_mean_mean", "adam_atten_p05_mean",
                        "adam_eps_frac_mean") if c in sub.columns]
    if sub.empty or not cols or sub[cols].isna().all().all():
        return None
    ds_list = datasets or sorted(sub.dataset.unique())[:8]
    fig, axes = _panels(len(ds_list), ncol=4)
    models = order_models(sub.model.unique())
    for ax, ds in zip(axes, ds_list):
        d = sub[sub.dataset == ds]
        for m in models:
            g = d[d.model == m]
            if g.empty:
                continue
            g = g.groupby("epoch", as_index=False).agg(
                y=("adam_atten_p05_mean", "mean"), n_folds=("n_folds", "sum"))
            _draw_mean(ax, g, "y", color=_color(m), label=m,
                       ls="-" if m.startswith("grow") else "--")
        ax.axhline(1.0, color="#c62828", lw=1.2, ls="--")
        ax.set_yscale("log"); ax.set_title(ds, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("attenuation, p05 (log)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("AdamW eps attenuation — the 5th percentile over parameters",
                 fontsize=12)
    fig.tight_layout()
    _note(fig, "1.0 (red) is the value at which eps would halve the update. The "
               "distance from it is the margin by which the eps hypothesis is "
               "refuted; the hypothesis' premise (gradients collapsing) can still be "
               "true and is visible in the gradient-norm figure.")
    return fig


# ================================================================= 2. growth events
def growth_event_timeline(events: pd.DataFrame, *, eval_: str = "within_session",
                          dataset: str = "bnci2014_001", align: str = ""):
    """Every growth step of every fold of one cell: when it fired, how big it was.

    A raster rather than a mean, because the question is whether folds behave alike.
    Growth fires on a fixed cadence (`grow_every`), so the *columns* are guaranteed; the
    information is in the marker size (neurons kept) and in where each fold's row stops,
    which is where early stopping ended that fold.
    """
    e = applied(events)
    e = e[(e["eval"] == eval_) & (e.dataset == dataset)
          & (e.align_tag.fillna("") == align)]
    if e.empty:
        return None
    models = [m for m in order_models(e.model.unique())]
    fig, axes = _panels(len(models), ncol=min(3, len(models)), w=4.6, h=3.4)
    for ax, m in zip(axes, models):
        g = e[e.model == m]
        # A stable row per (seed, fold) so rows mean the same thing across panels.
        rows = {k: i for i, k in enumerate(sorted(set(zip(g.seed, g.fit))))}
        y = [rows[(s, f)] for s, f in zip(g.seed, g.fit)]
        sc = ax.scatter(g.epoch, y, s=4 + 6 * g.grow_n_kept.fillna(0),
                        c=g.grow_n_kept, cmap="viridis", alpha=0.85, linewidths=0)
        ax.set_title(f"{m} — {len(rows)} folds", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("fold", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="neurons kept", pad=0.01)
    fig.suptitle(f"Growth events — {dataset}, {EVAL_LABEL.get(eval_, eval_)}"
                 f"{' + EA' if align else ''}", fontsize=12)
    fig.tight_layout()
    _note(fig, "Marker size and colour both encode neurons kept. A fold's row ending "
               "early is early stopping, not a failure to grow.")
    return fig


def neurons_added(events: pd.DataFrame):
    """How many neurons a step proposes, how many survive, and what that costs.

    The selection is the whole mechanism: gromo proposes a set of candidate neurons and
    a relative floor on the singular values decides which are kept. Three ways it can
    be uninteresting, all visible here -- it always keeps everything (the floor is
    inoperative), it always keeps one (the floor is above the whole spectrum, which is
    what the *absolute* threshold did before the relative one replaced it), or it keeps
    a count that does not vary with anything.
    """
    e = applied(events).copy()
    if e.empty or "grow_n_kept" not in e:
        return None
    e = e.dropna(subset=["grow_n_kept"])
    if e.empty:
        return None
    models = order_models(e.model.unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    for m in models:
        v = e[e.model == m].grow_n_kept
        ax.hist(v, bins=np.arange(-0.5, v.max() + 1.5), alpha=0.55,
                color=_color(m), label=f"{m} (median {v.median():.0f})")
    ax.set_xlabel("neurons kept by one growth step"); ax.set_ylabel("events")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Step size", fontsize=10)

    ax = axes[0, 1]
    e = e.assign(keep_frac=e.grow_n_kept / e.grow_n_proposed.replace(0, np.nan))
    ax.boxplot([e[e.model == m].keep_frac.dropna().to_numpy() for m in models],
               tick_labels=models, showfliers=False, widths=0.6)
    ax.axhline(1.0, color="#c62828", lw=1.2, ls="--", label="keeps everything offered")
    ax.set_ylabel("kept / proposed")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    ax.set_title("Is the floor doing anything?", fontsize=10)

    ax = axes[1, 0]
    e = e.sort_values(["eval", "dataset", "model", "align_tag", "seed", "fit", "epoch"])
    key = ["eval", "dataset", "model", "align_tag", "seed", "fit"]
    e = e.assign(k=e.groupby(key).cumcount() + 1,
                 cum=e.groupby(key).grow_n_kept.cumsum())
    for m in models:
        g = e[e.model == m].groupby("k").cum
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.to_numpy(), marker=_marker(m), ms=4, color=_color(m),
                label=m, lw=1.4)
        ax.fill_between(med.index, lo.to_numpy(), hi.to_numpy(), color=_color(m),
                        alpha=0.15, lw=0)
    ax.set_xlabel("growth event index"); ax.set_ylabel("cumulative neurons added")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Width gained, event by event (median, IQR)", fontsize=10)

    ax = axes[1, 1]
    # The compute the width actually cost. `dur` is the epoch wall time, so the ratio
    # across a growth step is the marginal cost of the neurons that step added.
    if "n_params_after" in e and e.n_params_after.notna().any():
        r = (e.n_params_after / e.n_params_before).dropna()
        for m in models:
            v = (e[e.model == m].n_params_after
                 / e[e.model == m].n_params_before).dropna()
            if v.empty:
                continue
            ax.scatter(e[e.model == m].loc[v.index, "grow_n_kept"], v, s=8, alpha=0.3,
                       color=_color(m), marker=_marker(m), label=m, linewidths=0)
        ax.axhline(1.0, color="0.3", lw=1, ls="--")
        ax.set_xlabel("neurons kept"); ax.set_ylabel("parameter count ratio, after/before")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        ax.set_title("What a neuron costs in parameters", fontsize=10)
    else:
        ax.axis("off")

    fig.suptitle("The growth step as a decision", fontsize=12)
    fig.tight_layout()
    return fig


def width_trajectory(cm: pd.DataFrame, fits: pd.DataFrame, *,
                     eval_: str = "within_session", align: str = "",
                     datasets: list[str] | None = None):
    """Width against epoch, against the target the arm was aiming at.

    The figure that could not be drawn at all before ``FitRecorder`` existed: width was
    only ever passed to ``logger.info`` with ``verbose=False``, which is how two arms
    went a whole campaign with their growth cap disconnected and nothing to show it.
    """
    sub = _cells(cm, eval_, align)
    sub = sub[sub.model.str.startswith("grow")]
    if sub.empty:
        return None
    ds_list = datasets or sorted(sub.dataset.unique())
    fig, axes = _panels(len(ds_list))
    f = _cells(fits, eval_, align)
    for ax, ds in zip(axes, ds_list):
        d = sub[sub.dataset == ds]
        for m in order_models(d.model.unique()):
            g = d[d.model == m].groupby("epoch", as_index=False).agg(
                width_mean=("width_mean", "mean"), n_folds=("n_folds", "sum"))
            _draw_mean(ax, g, "width_mean", color=_color(m), label=m)
            tgt = f[(f.dataset == ds) & (f.model == m)].target_width.dropna()
            if len(tgt):
                ax.axhline(tgt.iloc[0], color=_color(m), lw=1, ls="--", alpha=0.6)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("growable width", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Did the arm reach the width it was aiming at? — "
                 f"{EVAL_LABEL.get(eval_, eval_)}{' + EA' if align else ''}",
                 fontsize=12)
    fig.tight_layout()
    _note(fig, "Dashed horizontal = target width (the braindecode reference width). "
               "The mean is over folds; a fold that stopped early stopped growing.")
    return fig


def width_reached(fits: pd.DataFrame):
    """Fraction of folds that reached the target width, per arm and dataset.

    A growing arm that stops at half its target is not a smaller model that did as
    well -- it is a model whose comparison to a width-matched control is no longer
    width-matched, and every parameter-efficiency claim about it is measuring something
    else. This is the figure that gates those claims.
    """
    f = fits[fits.model.str.startswith("grow")].dropna(subset=["target_width"])
    if f.empty:
        return None
    frac = (f.groupby(["dataset", "model"]).reached_target.mean().unstack() * 100)
    ratio = (f.assign(r=f.width_end / f.target_width)
             .groupby(["dataset", "model"]).r.median().unstack() * 100)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tab, lab in ((axes[0], frac, "% of folds reaching the target width"),
                         (axes[1], ratio, "median width reached, % of target")):
        cols = order_models(tab.columns)
        x = np.arange(len(tab.index))
        w = 0.8 / max(len(cols), 1)
        for i, m in enumerate(cols):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, tab[m].to_numpy(), w,
                   color=_color(m), alpha=0.85, label=m)
        ax.axhline(100, color="#c62828", lw=1.2, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(tab.index, rotation=40, ha="right",
                                             fontsize=8)
        ax.set_ylabel(lab); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Width actually reached, against the target", fontsize=12)
    fig.tight_layout()
    _note(fig, "Left counts folds that land exactly on the target; right shows how "
               "close the ones that miss get. A high right bar with a low left bar is "
               "an arm that stops just short, which the cap should have prevented.")
    return fig


# ============================================================= 3. what growth decided
def _spectrum(cell_events: pd.DataFrame, col: str) -> list[np.ndarray]:
    out = []
    for s in cell_events[col]:
        try:
            v = json.loads(s) if isinstance(s, str) else s
        except (TypeError, json.JSONDecodeError):
            v = None
        out.append(np.asarray(v, dtype=float) if v else np.array([]))
    return out


def eigenvalue_spectra(events: pd.DataFrame, *, eval_: str = "within_session",
                       dataset: str = "bnci2014_001", model: str = "grow_shallow",
                       align: str = "", seed: int | None = None, fit: int | None = None):
    """The candidate spectrum at each growth step, and which candidates survived.

    This is the inside of the decision. gromo builds a set of candidate neurons and
    ranks them by singular value; the relative floor keeps those within
    ``MIN_SINGULAR_RATIO`` of the best one. Two failure modes are visible here and
    nowhere else:

    * a spectrum that is **flat** -- every candidate about as good as every other, so
      the ranking carries no information and the step is arbitrary;
    * a spectrum that **collapses** -- one candidate orders of magnitude above the
      rest, so the floor keeps exactly one neuron per step regardless of the budget.
      That is what the absolute ``statistical_threshold`` produced before the relative
      floor replaced it, and it is why `grow_shallow` reached width 14 of 40 in v5.
    """
    e = events[(events["eval"] == eval_) & (events.dataset == dataset)
               & (events.model == model) & (events.align_tag.fillna("") == align)]
    if seed is not None:
        e = e[e.seed == seed]
    if e.empty:
        return None
    e = e[e.fit == (fit if fit is not None else e.fit.min())].sort_values("epoch")
    if e.empty or "grow_eig_proposed" not in e:
        return None
    props = _spectrum(e, "grow_eig_proposed")
    kepts = _spectrum(e, "grow_eig_kept")
    if not any(len(p) for p in props):
        return None

    fig, axes = _panels(len(e) + 1, ncol=4, w=3.4, h=2.8)
    for ax, (_, row), p, k in zip(axes, e.iterrows(), props, kepts):
        if not len(p):
            ax.axis("off")
            continue
        p = np.sort(p)[::-1]
        ax.plot(np.arange(1, len(p) + 1), p, marker="o", ms=3, lw=1.2,
                color="#1565c0", label=f"{len(p)} proposed")
        if len(k):
            ax.scatter(np.arange(1, len(k) + 1), np.sort(k)[::-1], s=30, zorder=3,
                       color="#c62828", label=f"{len(k)} kept")
        ax.axhline(MIN_SINGULAR_RATIO * p[0], color="#ef6c00", lw=1.2, ls="--",
                   label=f"floor = {MIN_SINGULAR_RATIO:g} x max")
        ax.set_yscale("log")
        ax.set_title(f"epoch {int(row.epoch)} — width {int(row.width_before)} → "
                     f"{int(row.grow_width_after)}", fontsize=8)
        ax.set_xlabel("candidate rank", fontsize=8)
        ax.set_ylabel("eigenvalue (log)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6)

    # A summary panel: the decades the spectrum spans at each event. A single number per
    # event, so the trend across a fit is readable without counting panels.
    ax = axes[len(e)]
    span = [np.log10(p.max() / p[p > 0].min()) if (len(p) and (p > 0).any()) else np.nan
            for p in props]
    ax.plot(e.epoch.to_numpy(), span, marker="s", color="#37474f", lw=1.4)
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_ylabel("decades spanned by the spectrum", fontsize=8)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    ax.set_title("Spectrum spread over the fit", fontsize=8)

    fig.suptitle(f"Candidate spectra — {model}, {dataset}, "
                 f"{EVAL_LABEL.get(eval_, eval_)}, one fold", fontsize=12)
    fig.tight_layout()
    _note(fig, f"The floor is drawn at the configured MIN_SINGULAR_RATIO="
               f"{MIN_SINGULAR_RATIO:g}; it is not recorded per fit, so this line is "
               "the config's value and not a measurement.")
    return fig


def eigenvalue_summary(events: pd.DataFrame):
    """The spectra of every event in the campaign, reduced to three numbers each.

    The per-fold figure shows what one decision looked like; this shows whether that
    decision looks the same everywhere. The kept-mass fraction is the one to read: it is
    the share of the total first-order gain on offer that the step actually took.
    """
    e = applied(events).copy()
    if e.empty or "grow_eig_proposed" not in e:
        return None
    props, kepts = _spectrum(e, "grow_eig_proposed"), _spectrum(e, "grow_eig_kept")
    e = e.assign(
        eig_max=[p.max() if len(p) else np.nan for p in props],
        eig_span=[np.log10(p.max() / p[p > 0].min())
                  if (len(p) and (p > 0).any() and p.max() > 0) else np.nan
                  for p in props],
        kept_mass=[(k.sum() / p.sum()) if (len(p) and p.sum() > 0) else np.nan
                   for p, k in zip(props, kepts)])
    if e.eig_max.isna().all():
        return None
    models = order_models(e.model.unique())
    key = ["eval", "dataset", "model", "align_tag", "seed", "fit"]
    e = e.sort_values(key + ["epoch"])
    e = e.assign(k=e.groupby(key).cumcount() + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, col, lab, logy in (
            (axes[0], "eig_max", "largest candidate eigenvalue", True),
            (axes[1], "eig_span", "decades spanned by the spectrum", False),
            (axes[2], "kept_mass", "fraction of eigenvalue mass kept", False)):
        for m in models:
            g = e[e.model == m].groupby("k")[col]
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med.to_numpy(), marker=_marker(m), ms=4,
                    color=_color(m), label=m, lw=1.4)
            ax.fill_between(med.index, lo.to_numpy(), hi.to_numpy(),
                            color=_color(m), alpha=0.15, lw=0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("growth event index within a fold")
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both" if logy else "major")
    axes[0].set_title("Is there anything left to add?", fontsize=10)
    axes[1].set_title("Is the ranking informative?", fontsize=10)
    axes[2].set_title("How much of it was taken?", fontsize=10)
    fig.suptitle("Candidate spectra across the whole campaign (median, IQR)",
                 fontsize=12)
    fig.tight_layout()
    _note(fig, "A largest eigenvalue that decays across events is a network running "
               "out of useful directions; one that does not is a growth schedule that "
               "stopped for a reason other than saturation.")
    return fig


def growth_abstention(events: pd.DataFrame):
    """How often the line search refuses the step it was offered.

    THE FIGURE THAT REFRAMES ALL THE OTHERS. ``grow_step`` evaluates the extended model
    at each factor in ``SCALING_GRID = (0.0, 0.1, 0.5, 1.0)`` on a held-out slice and
    keeps the best. **s = 0 is a refusal**: the candidates are discarded, the width does
    not move, and -- correctly -- the arm is not latched, so it tries again next time.

    Measured on shin2017a / grow_shallow: the step runs on all 39 opportunities of a
    fold, is offered 26 candidates every single time, and refuses on 93.6 % of them.
    That is why the arm's median final width there is 8 out of a target of 40: not a
    cap, not a crash, not a missing candidate -- an explicit, repeated *no*.

    So a growing arm is, mechanically, a fixed narrow network that occasionally accepts
    a widening. Any statement of the form "growth did not help" has to be read against
    this figure first, because on the datasets at the right of the left panel there was
    barely any growth to help.
    """
    if "applied" not in events.columns or events.empty:
        return None
    e = events.copy()
    e["applied"] = e.applied.fillna(False).astype(bool)
    models = order_models(e.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                            gridspec_kw={"width_ratios": [1.5, 1, 1]})

    ax = axes[0]
    tab = (1 - e.groupby(["dataset", "model"]).applied.mean().unstack()) * 100
    cols = order_models(tab.columns)
    # Ordered by how often the arms refuse, so the datasets where growth barely
    # happened are grouped and nameable rather than scattered along an alphabet.
    tab = tab.reindex(columns=cols).sort_values(cols[0])
    x = np.arange(len(tab.index))
    w = 0.8 / max(len(cols), 1)
    for i, m in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, tab[m].to_numpy(), w,
               color=_color(m), alpha=0.85, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% of growth opportunities REFUSED (s = 0)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    ax.set_title("Where growth was offered and declined", fontsize=10)

    ax = axes[1]
    # Does the refusal rate rise as the network fills up (saturation) or is it flat
    # from the start (the step was never worth taking)? Two different diagnoses.
    key = ["eval", "dataset", "model", "align_tag", "seed", "fit"]
    e = e.sort_values(key + ["epoch"])
    e = e.assign(k=e.groupby(key).cumcount() + 1)
    for m in models:
        g = e[e.model == m]
        g = g[g.k <= 39].groupby("k").applied.mean() * 100
        ax.plot(g.index, 100 - g.to_numpy(), marker=_marker(m), ms=3,
                color=_color(m), label=m, lw=1.3)
    ax.set_xlabel("growth opportunity index within a fold")
    ax.set_ylabel("% refused")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Saturation, or refused from the start?", fontsize=10)

    ax = axes[2]
    # A refusal is not "nothing was proposed": the floor had already selected a set.
    for m in models:
        g = e[e.model == m]
        for state, alpha, lab in ((True, 0.75, "accepted"), (False, 0.35, "refused")):
            v = g[g.applied == state].grow_n_kept.dropna()
            if v.empty:
                continue
            ax.scatter([models.index(m) + (0.18 if state else -0.18)] * min(len(v), 400),
                       v.sample(min(len(v), 400), random_state=0),
                       s=6, alpha=alpha, color=_color(m), linewidths=0,
                       label=lab if m == models[0] else None)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("neurons the floor selected")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    ax.set_title("A refusal still had candidates\n(left = refused, right = accepted)",
                 fontsize=10)

    fig.suptitle("The decision that dominates: the line search says no", fontsize=12)
    fig.tight_layout()
    _note(fig, "Refusal is an abstention, not a cap: `done_` stays False and the "
               "statistics are re-estimated at the next opportunity. An arm can "
               "therefore refuse 39 times in a row and end at its starting width.")
    return fig


def line_search_factor(events: pd.DataFrame):
    """``grow_s``: what the line search did with the neurons it was handed.

    gromo does not splice candidates in at unit scale -- it solves for an amplitude
    factor that minimises the loss along the direction the new neurons define, over a
    four-point grid ``(0.0, 0.1, 0.5, 1.0)``.

    Both boundaries of that grid turn out to be where the answer lands, which is the
    diagnostic. **s = 0** (refusal) is the majority answer -- see
    :func:`growth_abstention`. **Conditional on the step being taken at all, s = 1.0 --
    the grid's upper bound -- on 98.8 % of applied events.** A search that returns its
    own ceiling almost every time it returns anything is a search whose ceiling is
    binding: the loss-minimising amplitude is plausibly above 1 and the grid has never
    been allowed to say so. That is a one-line change to ``SCALING_GRID`` and an
    experiment, not a conclusion -- but it is a measurement, not a guess.
    """
    e = events.dropna(subset=["grow_s"])
    if e.empty:
        return None
    models = order_models(e.model.unique())
    key = ["eval", "dataset", "model", "align_tag", "seed", "fit"]
    e = e.sort_values(key + ["epoch"])
    e = e.assign(k=e.groupby(key).cumcount() + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    for m in models:
        v = e[e.model == m].grow_s
        ax.hist(v, bins=40, alpha=0.55, color=_color(m),
                label=f"{m} (median {v.median():.3g})")
    ax.axvline(1.0, color="#c62828", lw=1.2, ls="--", label="s = 1 (no scaling)")
    ax.set_xlabel("line-search factor s"); ax.set_ylabel("events")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Distribution of s", fontsize=10)

    ax = axes[1]
    for m in models:
        g = e[e.model == m].groupby("k").grow_s
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.to_numpy(), marker=_marker(m), ms=4, color=_color(m),
                label=m, lw=1.4)
        ax.fill_between(med.index, lo.to_numpy(), hi.to_numpy(), color=_color(m),
                        alpha=0.15, lw=0)
    ax.axhline(1.0, color="0.3", lw=1, ls="--")
    ax.set_xlabel("growth event index"); ax.set_ylabel("s (median, IQR)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Does s move as the network fills up?", fontsize=10)

    ax = axes[2]
    ok = e.dropna(subset=["grow_first_order_improvement"])
    ok = ok[ok.grow_first_order_improvement > 0]
    for m in models:
        g = ok[ok.model == m]
        ax.scatter(g.grow_first_order_improvement, g.grow_s, s=7, alpha=0.3,
                   color=_color(m), marker=_marker(m), label=m, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("first-order improvement gromo expected (log)")
    ax.set_ylabel("s")
    ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")
    ax.set_title("s against the gain it was scaling", fontsize=10)

    fig.suptitle("The line search", fontsize=12)
    fig.tight_layout()
    return fig


def first_order_expected_vs_realised(events: pd.DataFrame):
    """Does the surrogate gromo optimises predict the loss it is a surrogate for?

    THE FIGURE THIS MODULE EXISTS FOR. gromo selects neurons by a first-order model of
    how much they will reduce the loss. That predicted gain is recorded
    (``grow_first_order_improvement``); so is the training loss on the growth epoch and
    on the one after. If the prediction is calibrated, the points sit on the identity.

    The control is what makes this readable: the *previous* epoch's loss drop, on the
    same fold, with no growth in it. That is what an ordinary epoch of gradient descent
    delivers. A growth step whose realised gain is indistinguishable from an ordinary
    epoch's has not been shown to do anything, whatever its predicted gain said -- and
    a predicted gain orders of magnitude below an ordinary epoch's drop means the
    selection is ranking candidates on a quantity too small to matter.
    """
    e = applied(events).dropna(subset=["grow_first_order_improvement",
                                       "train_loss_before", "train_loss_after"])
    if e.empty:
        return None
    e = e.assign(realised=e.train_loss_before - e.train_loss_after,
                 ordinary=e.train_loss_prev - e.train_loss_before)
    models = order_models(e.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for m in models:
        g = e[e.model == m]
        ax.scatter(g.grow_first_order_improvement, g.realised, s=8, alpha=0.3,
                   color=_color(m), marker=_marker(m), label=m, linewidths=0)
    lo = e.grow_first_order_improvement[e.grow_first_order_improvement > 0]
    if len(lo):
        span = [lo.min(), max(lo.max(), e.realised.abs().max())]
        ax.plot(span, span, color="0.3", lw=1, ls="--", label="perfect prediction")
    ax.axhline(0, color="#c62828", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("predicted first-order gain (log)")
    ax.set_ylabel("realised train-loss drop (symlog)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")
    ax.set_title("Prediction against reality", fontsize=10)

    ax = axes[1]
    data, labels, colors = [], [], []
    for m in models:
        g = e[e.model == m]
        data += [g.grow_first_order_improvement.dropna().to_numpy(),
                 g.realised.dropna().to_numpy(),
                 g.ordinary.dropna().to_numpy()]
        labels += [f"{m}\npredicted", "realised", "ordinary epoch"]
        colors += [_color(m), _color(m), "0.6"]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.6,
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.axhline(0, color="#c62828", lw=1, ls=":")
    ax.set_ylabel("train-loss change (symlog)")
    ax.tick_params(axis="x", rotation=70, labelsize=6)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("Growth against an ordinary epoch — the control", fontsize=10)

    ax = axes[2]
    # The ratio that decides whether the mechanism is even on the right scale.
    r = e.assign(ratio=e.grow_first_order_improvement
                 / e.ordinary.abs().replace(0, np.nan)).dropna(subset=["ratio"])
    for m in models:
        v = r[r.model == m].ratio
        v = v[v > 0]
        if v.empty:
            continue
        ax.hist(np.log10(v), bins=40, alpha=0.55, color=_color(m),
                label=f"{m} (median 1e{np.log10(v.median()):.1f})")
    ax.axvline(0, color="#c62828", lw=1.2, ls="--",
               label="growth = one ordinary epoch")
    ax.set_xlabel("log10( predicted growth gain / ordinary epoch's drop )")
    ax.set_ylabel("events")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Is the predicted gain even on the scale of training?", fontsize=10)

    fig.suptitle("Is a growth step worth an epoch?", fontsize=12)
    fig.tight_layout()
    _note(fig, "'ordinary epoch' is the train-loss drop from the epoch before the "
               "growth epoch to the growth epoch itself, on the same fold: plain "
               "gradient descent with no growth in it. It is the benchmark a growth "
               "step has to beat to have done anything.")
    return fig


def growth_step_aftermath(events: pd.DataFrame):
    """Validation loss and accuracy across a growth step, against the same control.

    Training loss is where the first-order model makes its prediction, so it is where
    the prediction is checked. But a step that only improves training loss has bought
    capacity to memorise. This is the same comparison on the held-out split.
    """
    e = applied(events).dropna(subset=["valid_loss_before", "valid_loss_after"])
    if e.empty:
        return None
    e = e.assign(d_loss=e.valid_loss_before - e.valid_loss_after,
                 d_acc=(e.valid_acc_after - e.valid_acc_before) * 100)
    models = order_models(e.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, lab in ((axes[0], "d_loss", "valid-loss drop across the step"),
                         (axes[1], "d_acc", "valid-accuracy change, pp")):
        vals = [e[e.model == m][col].dropna().to_numpy() for m in models]
        bp = ax.boxplot(vals, tick_labels=models, showfliers=False, widths=0.6,
                        patch_artist=True)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(_color(m)); patch.set_alpha(0.5)
        for i, v in enumerate(vals, start=1):
            if len(v):
                ax.scatter([i], [np.median(v)], color="k", s=14, zorder=4)
        ax.axhline(0, color="#c62828", lw=1.2, ls="--")
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("What a growth step does on the held-out split", fontsize=12)
    fig.tight_layout()
    _note(fig, "One epoch either side of the step. A distribution centred on zero is "
               "a step with no measurable effect on generalisation at the epoch it "
               "happened, which does not preclude an effect that accumulates.")
    return fig


# =============================================================== 4. stopping
def stop_reason_breakdown(fits: pd.DataFrame):
    """Why each fold ended: the budget, early stopping, or something else.

    A single bar, and that IS the result: with ``patience=200`` set against
    ``max_epochs=200`` the early-stopping callback cannot fire, so 100 % of folds end
    on the budget. Drawn rather than asserted because the alternative -- a campaign
    where some arms stopped early and others did not -- is the confound that made
    ``bd_deep4`` score at epoch 4 in the v5 grid, and the only way to know which
    campaign you are holding is to look.

    The consequence for every other figure: nothing here is confounded by unequal
    training length. All 119 998 folds ran exactly 200 epochs.
    """
    if "stop_reason" not in fits or fits.stop_reason.isna().all():
        return None
    f = fits.dropna(subset=["stop_reason"])
    reasons = sorted(f.stop_reason.unique())
    evals = [e for e in EVAL_LABEL if e in set(f["eval"])]
    fig, axes = _panels(len(evals), ncol=len(evals) or 1, w=5.2, h=4.2)
    cmap = plt.get_cmap("Set2")
    for ax, ev in zip(axes, evals):
        tab = pd.crosstab(f[f["eval"] == ev].model, f[f["eval"] == ev].stop_reason,
                          normalize="index") * 100
        models = order_models(tab.index)
        tab = tab.reindex(models)
        bottom = np.zeros(len(models))
        for i, r in enumerate(reasons):
            v = tab[r].to_numpy() if r in tab else np.zeros(len(models))
            ax.bar(np.arange(len(models)), v, 0.75, bottom=bottom,
                   color=cmap(i), label=r)
            bottom += v
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("% of folds"); ax.set_title(EVAL_LABEL[ev], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, title="stop_reason")
    fig.suptitle("Why every fold ended", fontsize=12)
    fig.tight_layout()
    return fig


def stopping_epochs(fits: pd.DataFrame):
    """Where a fold stopped, where its best epoch was, and the gap between them.

    The campaign ran with ``train.patience=200`` against ``max_epochs=200``, so early
    stopping is arithmetically unable to fire: **every one of the 119 998 folds ends
    with ``stop_reason='budget'`` at exactly epoch 200**. That is the intended
    protocol -- the shipped patience-20 default was measured to under-train six of nine
    arms -- and it turns the third panel from a structural check into a measurement.

    What it measures: ``200 - restored_epoch``, i.e. how many epochs a fold kept
    training after the model that would eventually be scored. The medians are 9 to 13
    for the fixed arms and 23 to 28 for the growing ones, out of 200. So roughly 90 %
    of every fold's compute is spent past its own optimum, and the growing arms peak
    later than their controls -- which is the one systematic difference between them
    that shows up here, and it is a difference in *when*, not in *how well*.
    """
    f = fits.dropna(subset=["epochs"])
    if f.empty:
        return None
    models = order_models(f.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for ax, col, lab in ((axes[0], "epochs", "epochs run"),
                         (axes[1], "restored_epoch", "epoch of the selected model")):
        if col not in f:
            ax.axis("off")
            continue
        vals = [f[f.model == m][col].dropna().to_numpy() for m in models]
        bp = ax.boxplot(vals, tick_labels=models, showfliers=False, widths=0.6,
                        patch_artist=True)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(_color(m)); patch.set_alpha(0.5)
        mx = f.max_epochs.dropna()
        if len(mx):
            ax.axhline(mx.iloc[0], color="#c62828", lw=1.2, ls="--",
                       label=f"budget = {int(mx.iloc[0])}")
            ax.legend(fontsize=7)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    g = f.dropna(subset=["epochs_past_best"])
    if len(g):
        for m in models:
            v = g[g.model == m].epochs_past_best
            if v.empty:
                continue
            ax.hist(v, bins=60, alpha=0.5, color=_color(m), label=m)
        ax.set_xlabel("epochs - restored_epoch")
        ax.set_ylabel("folds")
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25)
    ax.set_title("Epochs trained past the selected model\n(the budget, not patience)",
                 fontsize=10)
    axes[0].set_title("How long a fold ran", fontsize=10)
    axes[1].set_title("Which epoch got scored", fontsize=10)
    fig.suptitle("Stopping and selection", fontsize=12)
    fig.tight_layout()
    return fig


def stop_by_subject(fits: pd.DataFrame, *, eval_: str = "within_session",
                    align: str = "", datasets: list[str] | None = None):
    """Which held-out subject each fold stopped on -- the per-subject stopping map.

    Only drawable since ``subject_stamp`` wired the held-out subject onto the fit
    record; before that the recorder knew nothing about which subject it was fitting,
    and the identity had to be *inferred* from write order. A figure built on inferred
    identities would put a per-subject claim on the wrong subject, so this one refuses
    to draw when the stamp is absent rather than falling back.

    What it is for: 'the deep arms sit at chance on alexmi' and 'the deep arms stop
    after twenty epochs on every alexmi subject' are the same observation seen from two
    sides, and the second one names a cause. A row that is dark for one arm and light
    for its control is a subject on which the two arms were not given the same budget.
    """
    if "subject" not in fits or fits.subject.isna().all():
        return None
    f = _cells(fits, eval_, align).dropna(subset=["subject", "epochs"])
    if f.empty:
        return None
    ds_list = datasets or sorted(f.dataset.unique())
    fig, axes = _panels(len(ds_list), ncol=3, w=4.8, h=3.6)
    for ax, ds in zip(axes, ds_list):
        d = f[f.dataset == ds]
        tab = d.pivot_table(index="subject", columns="model", values="epochs",
                            aggfunc="median")
        cols = order_models(tab.columns)
        tab = tab.reindex(columns=cols)
        im = ax.imshow(tab.to_numpy(), aspect="auto", cmap="magma",
                       vmin=0, vmax=float(f.max_epochs.max() or 200))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(tab.index)))
        ax.set_yticklabels([str(s) for s in tab.index], fontsize=6)
        ax.set_title(ds, fontsize=9)
        fig.colorbar(im, ax=ax, label="median epochs run", pad=0.01)
    fig.suptitle(f"Where each arm stopped, subject by subject — "
                 f"{EVAL_LABEL.get(eval_, eval_)}{' + EA' if align else ''}",
                 fontsize=12)
    fig.tight_layout()
    _note(fig, "Dark = stopped early. Subjects are the recorded held-out subject "
               "(subject_stamp), never an inferred one.")
    return fig


def selected_model_width(fits: pd.DataFrame, events: pd.DataFrame):
    """For a growing arm, is the model that got scored as wide as the one that finished?

    A structural catch that only growing arms have. ``RestoreBestModel`` hands back the
    weights of the best epoch; on a growing arm that epoch can precede the last growth
    step, so the model actually scored is NARROWER than ``width_end`` says. Every
    parameter-efficiency claim reads ``width_end``, so this figure decides whether that
    column is the right one to read.

    Width is piecewise constant between growth events, so the width at the restored
    epoch is exact: it is the ``grow_width_after`` of the last event at or before it.
    """
    f = fits[fits.model.str.startswith("grow")].dropna(
        subset=["restored_epoch", "width_end"])
    if f.empty or events.empty:
        return None
    key = ["eval", "dataset", "model", "align_tag", "seed", "fit"]
    ev = applied(events).dropna(
        subset=["grow_width_after"])[key + ["epoch", "grow_width_after"]]
    merged = f[key + ["restored_epoch", "width_start", "width_end"]].merge(
        ev, on=key, how="left")
    before = merged[merged.epoch <= merged.restored_epoch]
    picked = (before.sort_values("epoch").groupby(key, as_index=False)
              .grow_width_after.last().rename(columns={"grow_width_after": "width_sel"}))
    out = f.merge(picked, on=key, how="left")
    # A fold whose best epoch precedes its first growth step was selected at the
    # starting width, which is a real value and not a missing one.
    out["width_sel"] = out.width_sel.fillna(out.width_start)
    if out.width_sel.isna().all():
        return None

    models = order_models(out.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for m in models:
        g = out[out.model == m]
        jitter = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.6
        ax.scatter(g.width_end + jitter, g.width_sel + jitter, s=7, alpha=0.25,
                   color=_color(m), marker=_marker(m), label=m, linewidths=0)
    lim = [0, float(out.width_end.max()) * 1.05]
    ax.plot(lim, lim, color="0.3", lw=1, ls="--", label="scored at its final width")
    ax.set_xlabel("width the fold ended at (width_end)")
    ax.set_ylabel("width of the model that was SCORED")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title("Selection can hand back a narrower net", fontsize=10)

    ax = axes[1]
    out = out.assign(shortfall=out.width_end - out.width_sel)
    vals = [out[out.model == m].shortfall.dropna().to_numpy() for m in models]
    bp = ax.boxplot(vals, tick_labels=models, showfliers=False, widths=0.6,
                    patch_artist=True)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(_color(m)); patch.set_alpha(0.5)
    ax.axhline(0, color="#c62828", lw=1.2, ls="--")
    ax.set_ylabel("neurons grown after the selected epoch")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    frac = float((out.shortfall > 0).mean() * 100)
    ax.set_title(f"{frac:.0f} % of folds were scored below their final width",
                 fontsize=10)
    fig.suptitle("Which model actually got scored", fontsize=12)
    fig.tight_layout()
    _note(fig, "Anything above zero on the right is capacity that was grown, paid for "
               "in compute, and then discarded by model selection.")
    return fig


def epoch_cost(cm: pd.DataFrame, *, eval_: str = "within_session", align: str = "",
               datasets: list[str] | None = None):
    """Wall time per epoch as the network widens -- what growth costs while it runs.

    The efficiency argument for growing is about parameters, and parameters are not what
    a GPU hour buys. This is the same axis measured in seconds: a growing arm should
    start cheaper than its width-matched control and converge to it, and the area
    between the two curves is the saving the argument is actually claiming.
    """
    sub = _cells(cm, eval_, align)
    if sub.empty or "dur_mean" not in sub or sub.dur_mean.isna().all():
        return None
    ds_list = datasets or sorted(sub.dataset.unique())
    fig, axes = _panels(len(ds_list))
    for ax, ds in zip(axes, ds_list):
        d = sub[sub.dataset == ds]
        for m in order_models(d.model.unique()):
            g = d[d.model == m].groupby("epoch", as_index=False).agg(
                dur_mean=("dur_mean", "mean"), n_folds=("n_folds", "sum"))
            _draw_mean(ax, g, "dur_mean", color=_color(m), label=m,
                       ls="-" if m.startswith("grow") else "--")
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("seconds / epoch", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"What an epoch costs as the network grows — "
                 f"{EVAL_LABEL.get(eval_, eval_)}{' + EA' if align else ''}",
                 fontsize=12)
    fig.tight_layout()
    return fig


def coverage(fits: pd.DataFrame):
    """What the campaign actually contains, drawn before anything is concluded from it.

    A partial campaign is the normal state of these figures, and the failure mode is not
    a missing panel -- it is a panel that draws eight arms where nine ran and reads as a
    comparison. This is the figure that has to be looked at first.
    """
    if fits.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                            gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    tab = fits.pivot_table(index="dataset", columns="model", values="fit",
                           aggfunc="count")
    cols = order_models(tab.columns)
    tab = tab.reindex(columns=cols)
    im = ax.imshow(tab.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels(tab.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="folds recorded", pad=0.01)
    ax.set_title("Folds per (dataset, arm)", fontsize=10)

    ax = axes[1]
    n = fits.groupby(["eval", "model"]).dataset.nunique().unstack()
    cols = order_models(n.columns)
    n = n.reindex(columns=cols)
    x = np.arange(len(n.index))
    w = 0.8 / max(len(cols), 1)
    for i, m in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, n[m].to_numpy(), w,
               color=_color(m), alpha=0.85, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([EVAL_LABEL.get(e, e) for e in n.index], fontsize=8)
    ax.set_ylabel("datasets with at least one fold")
    ax.legend(fontsize=6, ncol=2); ax.grid(axis="y", alpha=0.25)
    ax.set_title("Datasets per (protocol, arm)", fontsize=10)

    # The palette legend, once, for the whole report: hue = arm type, lightness =
    # architecture. Drawn here because this is the figure a reader opens first.
    handles = [Line2D([], [], color=_color(m), lw=6, label=m)
               for m in order_models(fits.model.unique())]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.08),
               title="palette used throughout: hue = arm type, lightness = architecture")
    fig.suptitle("Campaign coverage — read this before any other figure", fontsize=12)
    fig.tight_layout()
    return fig
