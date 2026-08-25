"""Training-dynamics figures for the v5 campaign: what happens *during* the fit.

Everything in `explore_figures` is drawn from one number per fold -- the test score --
and so can only answer "who won". None of it can say why: whether a model that scores
at chance never learned or learned and overfitted, whether the 200-epoch budget or the
patience-20 early stopping is the binding constraint, whether a growing arm reached the
width it was aiming at, or what any of that cost. Those questions live in the per-epoch
records, which is what this module reads.

THE ONE TRAP IN AVERAGING TRAINING CURVES. Every fold stops when early stopping fires,
so folds have different lengths. Take the fold-mean at epoch 120 and you are averaging
only the folds that survived to 120 -- and those are not a random subset, they are the
ones that were still improving at 100. The mean therefore rises for a reason that has
nothing to do with training. Every figure here that draws a mean curve also carries the
surviving fold count and switches the line to dotted once it drops below half the folds,
so the part of the curve that is survivorship is visibly marked as such rather than
read as a result.

The frames come from `export_v5_tidy.py`:

    curves_mean    (eval, dataset, model, seed, epoch) -> fold-mean curves + n_folds
    fold_summary   one row per fold: where its best epoch was, losses there
    fits           one row per fold: widths, params, epochs, seconds
    budget         one row per fold: parameter-epochs, the honest cost axis
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_figures import (
    EVAL_LABEL,
    FAM_COLOR,
    FAM_MARKER,
    family_of,
    order_models,
)

# Below this fraction of a cell's folds the mean curve is survivorship, not training.
SURVIVING = 0.5


def _color(model: str) -> str:
    return FAM_COLOR[family_of(model)]


def _marker(model: str) -> str:
    return FAM_MARKER[family_of(model)]


def _panels(n: int, ncol: int, w: float = 3.6, h: float = 2.6):
    nrow = int(np.ceil(n / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    return f, axes.ravel()


def _draw_curve(ax, g: pd.DataFrame, col: str, model: str, *, ls="-", lw=1.4,
                label=None, band: str | None = None):
    """One mean curve, solid while most folds survive and dotted after.

    `g` is one (cell, seed) block or a seed-pooled block, sorted by epoch, carrying
    `n_folds`. The split point is the last epoch at which at least SURVIVING of the
    cell's maximum fold count is still running.
    """
    g = g.sort_values("epoch")
    if g.empty:
        return
    keep = g.n_folds >= SURVIVING * g.n_folds.max()
    cut = int(g.epoch[keep].max()) if keep.any() else int(g.epoch.min())
    solid, thin = g[g.epoch <= cut], g[g.epoch >= cut]
    ax.plot(solid.epoch, solid[col], ls=ls, lw=lw, color=_color(model), label=label)
    if len(thin) > 1:
        ax.plot(thin.epoch, thin[col], ls=":", lw=lw * 0.8, color=_color(model),
                alpha=0.7)
    if band and f"{band}" in g:
        ax.fill_between(solid.epoch, solid[col] - solid[band], solid[col] + solid[band],
                        color=_color(model), alpha=0.13, lw=0)


def _pool_seeds(cm: pd.DataFrame) -> pd.DataFrame:
    """Average the per-seed fold-means into one curve per (eval, dataset, model, epoch).

    Weighted by `n_folds` so a seed whose folds mostly died does not carry the same
    weight as one still running five, and `n_folds` is summed so the survivorship rule
    keeps working after pooling.
    """
    out = []
    cols = ["train_loss_mean", "valid_loss_mean", "valid_acc_mean", "width_mean",
            "n_params_mean"]
    for key, g in cm.groupby(["eval", "dataset", "model", "epoch"], sort=False):
        w = g.n_folds.to_numpy(dtype=float)
        rec = dict(zip(["eval", "dataset", "model", "epoch"], key))
        rec["n_folds"] = float(w.sum())
        for c in cols:
            v = g[c].to_numpy(dtype=float)
            ok = np.isfinite(v)
            rec[c] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        # Spread across seeds, which is the band a reader wants on a pooled curve.
        rec["valid_acc_seed_sd"] = float(g.valid_acc_mean.std())
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------- learning curves
def learning_curves(cm: pd.DataFrame, order: list[str], eval_: str = "within_session",
                    chance: dict[str, float] | None = None):
    """Validation accuracy against epoch, one panel per dataset, one line per model.

    The figure Stella asked for, drawn for every dataset rather than the showcase one.
    Read it against the chance line, which is per-dataset: a flat line on chance is a
    model that never learned, and it looks identical to a model that learned nothing
    *useful* only until you compare its train loss in the next figure.

    `chance` is 1/n_classes per dataset. Without it a panel whose y range spans 0.49 to
    0.51 looks like a curve with structure; with the chance rule drawn on it, it reads
    as what it is -- a monitor that never left chance for the whole run.
    """
    pooled = _pool_seeds(cm[cm["eval"] == eval_])
    f, axes = _panels(len(order), 4)
    seen = {}
    for ax, ds in zip(axes, order):
        blk = pooled[pooled.dataset == ds]
        for mod in order_models(blk.model.unique()):
            g = blk[blk.model == mod]
            if g.empty:
                continue
            _draw_curve(ax, g, "valid_acc_mean", mod, band="valid_acc_seed_sd")
            seen[mod] = True
        if chance and ds in chance:
            ax.axhline(chance[ds], color="0.25", lw=1.0, ls="--")
        ax.set_title(ds, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes[len(order):]:
        ax.axis("off")
    for ax in axes[::4]:
        ax.set_ylabel("internal valid accuracy", fontsize=8)
    handles = [plt.Line2D([], [], color=_color(m), lw=1.6, label=m) for m in seen]
    axes[0].legend(handles=handles, fontsize=5.5, ncol=2)
    f.suptitle(f"Training curves — {EVAL_LABEL[eval_]}, fold-mean over seeds. "
               "Dotted = fewer than half the folds still running (survivorship)",
               fontsize=12)
    f.tight_layout()
    return f


def loss_curves(cm: pd.DataFrame, order: list[str], eval_: str = "within_session"):
    """Train loss (dashed) against validation loss (solid), per dataset.

    The pair is what separates the two ways of scoring at chance. A train loss that
    falls to zero while the valid loss turns up is a model that memorised the session;
    a train loss that never falls is a model that could not fit even the training
    trials, and the fix for those two is opposite.
    """
    pooled = _pool_seeds(cm[cm["eval"] == eval_])
    f, axes = _panels(len(order), 4)
    for ax, ds in zip(axes, order):
        blk = pooled[pooled.dataset == ds]
        for mod in order_models(blk.model.unique()):
            g = blk[blk.model == mod]
            _draw_curve(ax, g, "valid_loss_mean", mod, lw=1.3)
            _draw_curve(ax, g, "train_loss_mean", mod, ls="--", lw=0.9)
        ax.set_title(ds, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes[len(order):]:
        ax.axis("off")
    for ax in axes[::4]:
        ax.set_ylabel("loss", fontsize=8)
    axes[0].legend(handles=[plt.Line2D([], [], color="0.3", lw=1.3, label="valid"),
                            plt.Line2D([], [], color="0.3", lw=0.9, ls="--",
                                       label="train")], fontsize=6)
    f.suptitle(f"Fit or memorise? train (dashed) vs valid (solid) loss — "
               f"{EVAL_LABEL[eval_]}", fontsize=12)
    f.tight_layout()
    return f


# ------------------------------------------------------------- growth, in the curve
def growth_epochs(g: pd.DataFrame) -> list[int]:
    """Epochs of one fit at whose END the module grew, read off the width trajectory.

    Derived from `width` rather than from the `grow_applied` flag on purpose: the flag
    only exists on records written after the instrumentation landed, and the campaign
    Stella wants to look at predates it. A width that increases between consecutive
    epochs *is* a growth event, exactly and without inference.

    Note the off-by-one that matters for reading these figures. `GromoGrowth` runs in
    `on_epoch_end`, so epoch e's own loss and accuracy were computed BEFORE the neurons
    were added: the first epoch that reflects a wider model is e+1. The marker is drawn
    at e, and the effect, if any, is to its right.
    """
    g = g.sort_values("epoch")
    w = g.width.to_numpy(dtype=float)
    ep = g.epoch.to_numpy()
    grew = np.flatnonzero(np.diff(w) > 0)
    return [int(ep[i]) for i in grew]


def growth_annotated_curves(curves: pd.DataFrame, model: str, dataset: str,
                            eval_: str = "within_session", seed: int = 0,
                            n_folds: int = 6):
    """Per-fold training curves with a rule at every growth event -- Stella's ask.

    Drawn PER FOLD and not as a mean, which is the whole point. Fold A grows at epoch 5
    and fold B at epoch 7; average them and the discontinuity that the figure exists to
    show is smoothed into a ramp. A pooled version of this figure would answer a
    different question than the one asked, so there is not one.

    Accuracy (left axis, solid) and validation loss (right axis, dashed) share a panel
    because the interesting failure is when they disagree -- loss jumping while accuracy
    holds is a model whose confidence broke and whose predictions have not yet, and that
    is precisely the shape suspected at the growth events on the diverging folds.
    """
    blk = curves[(curves["eval"] == eval_) & (curves.dataset == dataset)
                 & (curves.model == model) & (curves.seed == seed)]
    fits = sorted(blk.fit.unique())[:n_folds]
    f, axes = _panels(len(fits), 3, w=3.8, h=2.8)
    for ax, fit in zip(axes, fits):
        g = blk[blk.fit == fit].sort_values("epoch")
        ax.plot(g.epoch, g.valid_acc, color=_color(model), lw=1.4)
        ax.set_ylabel("valid acc", fontsize=7, color=_color(model))
        ax2 = ax.twinx()
        ax2.plot(g.epoch, g.valid_loss, color="0.35", lw=1.0, ls="--")
        ax2.plot(g.epoch, g.train_loss, color="0.65", lw=0.8, ls=":")
        ax2.set_ylabel("loss", fontsize=7, color="0.35")
        for e in growth_epochs(g):
            ax.axvline(e, color="0.15", lw=0.8, alpha=0.55)
        w0, w1 = g.width.iloc[0], g.width.iloc[-1]
        ax.set_title(f"fold {fit} — width {w0:.0f} → {w1:.0f}", fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes[len(fits):]:
        ax.axis("off")
    f.suptitle(f"{model} on {dataset} ({EVAL_LABEL[eval_]}, seed {seed}) — vertical "
               "rules mark growth events. Solid: valid acc. Dashed: valid loss. "
               "Dotted: train loss", fontsize=11)
    f.tight_layout()
    return f


def growth_event_response(curves: pd.DataFrame, model: str, control: str,
                          eval_: str = "within_session", window: int = 5,
                          col: str = "valid_acc"):
    """Peri-growth average: what the metric does around a growth event, over all folds.

    "How does the accuracy change exactly at the moment you grow" is a question about an
    average effect, and the per-fold figure above cannot answer it -- one fold is one
    sample. So align every growth event at lag 0 and average across events, the same
    construction as an evoked response, with the value at lag 0 subtracted so each event
    contributes a *change* and not its level.

    THE CONFOUND, AND THE CONTROL. Later epochs are better epochs: a net improves over
    training whether or not anything was added to it, so a peri-event average of a
    growing arm slopes upward for reasons that have nothing to do with growth. Reading
    that slope as the effect of growth would be the whole error. Hence `control` -- a
    FIXED arm, sampled at the same epoch numbers, which experiences the passage of
    training and nothing else. The growth effect is the gap between the two curves, not
    the height of either.

    Shaded bands are 95% CI over events. Where the bands overlap, the data does not
    separate growing from merely continuing.
    """
    def response(mod: str, at_growth: bool) -> pd.DataFrame:
        blk = curves[(curves["eval"] == eval_) & (curves.model == mod)]
        lags = range(-window, window + 1)
        acc = {k: [] for k in lags}
        for _, g in blk.groupby(["dataset", "seed", "fit"], sort=False):
            g = g.sort_values("epoch")
            series = dict(zip(g.epoch.to_numpy(), g[col].to_numpy(dtype=float)))
            # The control has no growth events of its own, so it is sampled at the
            # epochs where the growing arm would have had them: multiples of the same
            # `grow_every`, inferred from the growing arm's own spacing below.
            events = growth_epochs(g) if at_growth else _control_epochs(g, curves,
                                                                        model, eval_)
            for e in events:
                base = series.get(e)
                if base is None or not np.isfinite(base):
                    continue
                for k in lags:
                    v = series.get(e + k)
                    if v is not None and np.isfinite(v):
                        acc[k].append(v - base)
        rows = []
        for k, vals in acc.items():
            if len(vals) < 2:
                continue
            a = np.asarray(vals)
            rows.append(dict(lag=k, mean=a.mean(), ci=1.96 * a.std(ddof=1) / len(a) ** .5,
                             n=len(a)))
        return pd.DataFrame(rows).sort_values("lag")

    grow, ctrl = response(model, True), response(control, False)
    f, ax = plt.subplots(figsize=(6.4, 4.0))
    for df, lbl, c in ((grow, f"{model} (at its growth events)", _color(model)),
                       (ctrl, f"{control} (fixed, same epochs)", "0.45")):
        if df.empty:
            continue
        ax.plot(df.lag, df["mean"], color=c, lw=1.8, marker="o", ms=3.5, label=lbl)
        ax.fill_between(df.lag, df["mean"] - df.ci, df["mean"] + df.ci, color=c,
                        alpha=0.18, lw=0)
    ax.axvline(0, color="0.15", lw=0.9)
    ax.axhline(0, color="0.15", lw=0.7, ls=":")
    ax.set_xlabel("epochs relative to the growth event (0 = last epoch before it)")
    ax.set_ylabel(f"Δ {col} vs lag 0")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    n = int(grow.n.max()) if not grow.empty else 0
    ax.set_title(f"Peri-growth response — {EVAL_LABEL[eval_]}, {n} events pooled over "
                 "datasets/seeds/folds.\nThe effect of growth is the GAP to the fixed "
                 "control, not the slope of either curve.", fontsize=9)
    f.tight_layout()
    return f


def _control_epochs(g: pd.DataFrame, curves: pd.DataFrame, model: str,
                    eval_: str) -> list[int]:
    """Epochs at which to sample a fixed arm, matching the growing arm's cadence.

    The growing arm grows every `grow_every` epochs; the control has to be read at the
    same points in training or the comparison is between different parts of the curve.
    The cadence is recovered from the data rather than hard-coded, so the control stays
    matched if a config changes it.
    """
    sub = curves[(curves["eval"] == eval_) & (curves.model == model)]
    spacings = []
    for _, blk in sub.groupby(["dataset", "seed", "fit"], sort=False):
        ev = growth_epochs(blk)
        spacings += list(np.diff(ev))
    every = int(np.median(spacings)) if spacings else 5
    last = int(g.epoch.max())
    return list(range(every, last + 1, max(1, every)))


def curve_seed_band(cm: pd.DataFrame, dataset: str, eval_: str = "within_session"):
    """The same cell's curve for each of the five seeds, one panel per model.

    Every other curve figure pools seeds, which hides whether the pooled shape is a
    shape at all. If five seeds trace five different curves, the mean curve is an
    artefact and no claim about "the training dynamics" of that arm survives.
    """
    sub = cm[(cm["eval"] == eval_) & (cm.dataset == dataset)]
    models = order_models(sub.model.unique())
    f, axes = _panels(len(models), 3, w=3.8, h=2.7)
    for ax, mod in zip(axes, models):
        g = sub[sub.model == mod]
        for seed, h in g.groupby("seed"):
            h = h.sort_values("epoch")
            ax.plot(h.epoch, h.valid_acc_mean, lw=1.1, alpha=0.85,
                    color=plt.cm.viridis(seed / max(g.seed.max(), 1)),
                    label=f"seed {seed}")
        ax.set_title(mod, fontsize=9, color=_color(mod))
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes[len(models):]:
        ax.axis("off")
    for ax in axes[::3]:
        ax.set_ylabel("valid accuracy", fontsize=8)
    axes[0].legend(fontsize=6)
    f.suptitle(f"Is the mean curve a curve? five seeds on {dataset}, "
               f"{EVAL_LABEL[eval_]}", fontsize=12)
    f.tight_layout()
    return f


def paired_curves(cm: pd.DataFrame, pairs: list[tuple[str, str]], order: list[str],
                  eval_: str = "within_session"):
    """Each growing arm against the fixed arm it is matched with, per dataset.

    The contrast the campaign exists for, drawn in the only place it is visible as a
    process rather than a final number: if growing helps, the red curve should be at or
    below the blue one early (fewer parameters, slower start) and above it late.
    """
    pooled = _pool_seeds(cm[cm["eval"] == eval_])
    f, axes = _panels(len(order) * len(pairs), len(pairs), w=3.4, h=2.4)
    i = 0
    for ds in order:
        for grow, fixed in pairs:
            ax = axes[i]; i += 1
            blk = pooled[pooled.dataset == ds]
            for mod in (fixed, grow):
                g = blk[blk.model == mod]
                if not g.empty:
                    _draw_curve(ax, g, "valid_acc_mean", mod, label=mod)
            ax.set_title(f"{ds}\n{grow} vs {fixed}", fontsize=7)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=6)
            if i <= len(pairs):
                ax.legend(fontsize=5.5)
    for ax in axes[i:]:
        ax.axis("off")
    f.suptitle(f"Matched pairs, epoch by epoch — {EVAL_LABEL[eval_]} "
               "(red growing, blue/navy fixed)", fontsize=12)
    f.tight_layout()
    return f


# ------------------------------------------------------------------ where the fit ends
def stopping_epoch(fs: pd.DataFrame, fits: pd.DataFrame, order: list[str],
                   eval_: str = "within_session"):
    """Which constraint actually binds: the epoch budget or the patience.

    Left: the epoch the model is *selected* at (best valid accuracy). Right: the epoch
    training stopped at. A selected epoch far below the stop epoch means patience-20 is
    spending 20 epochs after every improvement for nothing; a stop epoch pinned at
    max_epochs means the budget is binding and the arm is being cut off mid-learning.
    Those are opposite problems and the campaign has both, in different arms.
    """
    a = fs[fs["eval"] == eval_]
    b = fits[fits["eval"] == eval_]
    models = order_models(set(a.model) & set(b.model))
    f, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    y = np.arange(len(order))
    w = 0.8 / max(len(models), 1)
    for ax, (frame, col, title) in zip(axes, [
            (a, "epoch_of_best", "epoch of best valid accuracy (the model kept)"),
            (b, "epochs", "epoch training actually stopped at")]):
        med = frame.groupby(["dataset", "model"])[col].median()
        for i, mod in enumerate(models):
            vals = [med.get((d, mod), np.nan) for d in order]
            ax.barh(y + (i - (len(models) - 1) / 2) * w, vals, w,
                    color=_color(mod), alpha=0.85,
                    label=mod if ax is axes[0] else None)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("epochs (median over folds and seeds)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(title, fontsize=10)
    mx = float(b.max_epochs.max()) if "max_epochs" in b else np.nan
    if np.isfinite(mx):
        axes[1].axvline(mx, color="#c62828", lw=1.2, ls="--",
                        label=f"max_epochs = {int(mx)}")
        axes[1].legend(fontsize=7, loc="lower right")
    axes[0].legend(fontsize=6, ncol=2, loc="lower right")
    f.suptitle(f"Budget or patience — which one ends the fit? {EVAL_LABEL[eval_]}",
               fontsize=12)
    f.tight_layout()
    return f


def overfit_gap(fs: pd.DataFrame, order: list[str], eval_: str = "within_session"):
    """Validation loss minus train loss at the selected epoch, per dataset and model.

    Measured at the epoch the model is kept, not at the end: the gap after early
    stopping has already fired describes a model nobody uses. A large positive gap on a
    dataset where the score is at chance is the signature of memorising a handful of
    trials, and it is the one diagnosis that says "this needs more data, not a better
    architecture".
    """
    sub = fs[fs["eval"] == eval_]
    models = order_models(sub.model.unique())
    m = sub.groupby(["dataset", "model"]).gap_at_best.agg(["mean", "std", "size"])
    m["ci"] = 1.96 * m["std"] / np.sqrt(m["size"].clip(lower=1))
    f, ax = plt.subplots(figsize=(13, 5.4))
    x = np.arange(len(order))
    w = 0.8 / max(len(models), 1)
    for i, mod in enumerate(models):
        vals = [m["mean"].get((d, mod), np.nan) for d in order]
        err = [m["ci"].get((d, mod), np.nan) for d in order]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, yerr=err, capsize=1.5,
               color=_color(mod), alpha=0.85, label=mod, error_kw={"lw": 0.6})
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("valid loss $-$ train loss, at the selected epoch")
    ax.legend(fontsize=6.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Generalisation gap where the model is actually kept — "
                 f"{EVAL_LABEL[eval_]}", fontsize=12)
    f.tight_layout()
    return f


def valid_vs_test(fs: pd.DataFrame, tidy: pd.DataFrame, eval_: str = "within_session"):
    """The internal validation accuracy against the MOABB test score it selected for.

    Early stopping monitors skorch's internal 20 % split; the number that gets reported
    is MOABB's held-out fold. If those two disagree, the arm is being early-stopped on a
    signal unrelated to what is measured -- which would make every "this model is worse"
    statement partly a statement about model selection. One point per (dataset, model),
    both axes in their own units, so only the rank correlation is meaningful.
    """
    from scipy.stats import spearmanr

    v = fs[fs["eval"] == eval_].groupby(["dataset", "model"]).best_valid_acc.mean()
    t = (tidy[tidy["eval"] == eval_].groupby(["dataset", "model"]).score.mean())
    i = v.index.intersection(t.index)
    v, t = v[i], t[i]
    rho, p = spearmanr(v, t)
    f, ax = plt.subplots(figsize=(7.0, 6.2))
    for mod in order_models({k[1] for k in i}):
        sel = [k for k in i if k[1] == mod]
        ax.scatter([v[k] for k in sel], [t[k] for k in sel], s=42, alpha=0.85,
                   color=_color(mod), marker=_marker(mod), label=mod,
                   edgecolors="white", linewidths=0.5)
    lo = float(min(v.min(), t.min())) - 0.03
    hi = float(max(v.max(), t.max())) + 0.03
    ax.plot([lo, hi], [lo, hi], color="0.45", lw=1, ls="--")
    ax.set_xlabel("internal valid accuracy (what early stopping watches)")
    ax.set_ylabel("MOABB test score (what gets reported)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title(rf"Does the stopping criterion track the score? $\rho$ = {rho:+.2f} "
                 rf"(p = {p:.1g}, n = {len(i)})", fontsize=11)
    f.tight_layout()
    return f


# ------------------------------------------------------------------------ growth itself
def width_trajectory(cm: pd.DataFrame, order: list[str],
                     eval_: str = "within_session"):
    """Width against epoch for the growable arms, every dataset.

    The v5 report drew this for one dataset. Across twelve it answers a different
    question: whether the plateau is a property of the architecture (same ceiling
    everywhere) or of the data (ceiling scales with trials). The fixed control is drawn
    too, as the flat line the growing arms are supposed to leave behind.
    """
    pooled = _pool_seeds(cm[cm["eval"] == eval_])
    pooled = pooled[pooled.width_mean.notna()]
    if pooled.empty:
        return None
    f, axes = _panels(len(order), 4)
    seen = {}
    for ax, ds in zip(axes, order):
        blk = pooled[pooled.dataset == ds]
        for mod in order_models(blk.model.unique()):
            g = blk[blk.model == mod].sort_values("epoch")
            if g.width_mean.isna().all():
                continue
            _draw_curve(ax, g, "width_mean", mod)
            seen[mod] = True
        ax.set_title(ds, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch", fontsize=7)
    for ax in axes[len(order):]:
        ax.axis("off")
    for ax in axes[::4]:
        ax.set_ylabel("width", fontsize=8)
    axes[0].legend(handles=[plt.Line2D([], [], color=_color(m), lw=1.6, label=m)
                            for m in seen], fontsize=6)
    f.suptitle(f"Where growth stops, dataset by dataset — {EVAL_LABEL[eval_]}",
               fontsize=12)
    f.tight_layout()
    return f


def width_reached(fits: pd.DataFrame, order: list[str]):
    """Final width against the target, and how often the target is reached.

    `reached_target` is the campaign's own definition of "the growth schedule ran to
    completion". A growing arm that reaches it everywhere is being throttled by the
    schedule; one that reaches it nowhere is being throttled by early stopping, and the
    two call for different fixes. Only arms with a target are plotted -- a fixed control
    has none, and drawing it at zero would read as a failure to grow.
    """
    g = fits[fits.target_width.notna()]
    if g.empty:
        return None
    f, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    models = order_models(g.model.unique())
    y = np.arange(len(order))
    w = 0.8 / max(len(models), 1)
    end = g.groupby(["dataset", "model"]).width_end.median()
    tgt = g.groupby(["dataset", "model"]).target_width.median()
    reach = g.groupby(["dataset", "model", "eval"]).reached_target.mean() \
             .groupby(["dataset", "model"]).mean()
    for i, mod in enumerate(models):
        off = (i - (len(models) - 1) / 2) * w
        axes[0].barh(y + off, [end.get((d, mod), np.nan) for d in order], w,
                     color=_color(mod), alpha=0.85, label=mod)
        axes[0].scatter([tgt.get((d, mod), np.nan) for d in order], y + off,
                        marker="|", s=90, color="0.15", zorder=3)
        axes[1].barh(y + off, [reach.get((d, mod), np.nan) for d in order], w,
                     color=_color(mod), alpha=0.85)
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=8)
    axes[0].set_xlabel("final width (bar) vs target width (tick)")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[1].set_xlabel("fraction of folds reaching the target width")
    axes[1].set_xlim(0, 1)
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    f.suptitle("Did the growth schedule run to completion? (all evaluation modes)",
               fontsize=12)
    f.tight_layout()
    return f


def budget_pareto(budget: pd.DataFrame, tidy: pd.DataFrame,
                  eval_: str = "within_session"):
    """Test score against parameter-epochs, the capacity a fold actually paid for.

    Final parameter count flatters a growing model, which spent most of its epochs
    narrower than it ended; summing parameters over epochs is the budget it really
    spent. A model up and to the left is doing more with less, which is the entire
    thesis of the growing arm and the only axis on which it can win while losing on
    score.
    """
    b = (budget[budget["eval"] == eval_]
         .groupby(["dataset", "model"]).param_epochs.mean())
    t = tidy[tidy["eval"] == eval_].groupby(["dataset", "model"]).above.mean()
    i = b.index.intersection(t.index)
    if not len(i):
        return None
    f, ax = plt.subplots(figsize=(9.5, 6.2))
    for mod in order_models({k[1] for k in i}):
        sel = sorted([k for k in i if k[1] == mod], key=lambda k: b[k])
        ax.plot([b[k] for k in sel], [t[k] for k in sel], marker=_marker(mod), ms=5,
                lw=0.9, alpha=0.85, color=_color(mod), label=mod)
    ax.set_xscale("log")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xlabel("parameter-epochs per fold (log)")
    ax.set_ylabel("score $-$ chance")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Capacity actually paid for, against what it bought — "
                 f"{EVAL_LABEL[eval_]}", fontsize=12)
    f.tight_layout()
    return f


def cost_per_epoch(fits: pd.DataFrame, order: list[str]):
    """Seconds per epoch against final width, one point per (dataset, model, eval).

    The price of growth in wall clock. A growing arm that ends at the same width as its
    fixed control but costs more per epoch is paying for the growth statistics
    themselves (the einsum in `compute_s_update`), not for the extra parameters -- and
    that overhead is what decides whether the method is usable at scale.
    """
    g = fits[(fits.epochs > 0) & fits.seconds.notna()].copy()
    g["s_per_epoch"] = g.seconds / g.epochs
    m = g.groupby(["dataset", "model", "eval"]).agg(
        s=("s_per_epoch", "median"), w=("width_end", "median"),
        p=("params_end", "median")).reset_index()
    f, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, xcol, xlabel in [(axes[0], "w", "final width"),
                             (axes[1], "p", "final parameters (log)")]:
        for mod in order_models(m.model.unique()):
            h = m[m.model == mod]
            ax.scatter(h[xcol], h.s, s=34, alpha=0.8, color=_color(mod),
                       marker=_marker(mod), label=mod if ax is axes[0] else None,
                       edgecolors="white", linewidths=0.4)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("seconds per epoch (log)")
        ax.grid(alpha=0.3, which="both")
    axes[1].set_xscale("log")
    axes[0].legend(fontsize=6, ncol=2)
    f.suptitle("What an epoch costs, and what sets the cost", fontsize=12)
    f.tight_layout()
    return f


def selected_epoch_vs_data(fs: pd.DataFrame, fits: pd.DataFrame):
    """The epoch a fold is selected at, against how many trials it trained on.

    Written for what the fold summary turned up: the median selected epoch is 1 on the
    smallest folds and 35 on the largest, monotonically, and every fold stops exactly 20
    epochs later -- which is the patience. So on the data-poor cells the internal
    validation accuracy never improves after the first epoch or two and training is over
    before it began. With ``grow_every=5``, a fold selected at epoch 1 has not taken a
    single growth step, which makes "growing does not help on small datasets" a statement
    about the schedule rather than about growth.

    Both axes are per fold. The right panel is the same thing as a distribution, because
    a median hides that the small-data cells are not merely early on average -- they are
    concentrated on epoch 1.
    """
    m = fs.merge(fits[["eval", "dataset", "model", "seed", "fit", "n_train",
                       "max_epochs"]],
                 on=["eval", "dataset", "model", "seed", "fit"], suffixes=("", "_f"))
    f, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    g = m.groupby(["eval", "dataset", "model"]).agg(
        n=("n_train", "median"), best=("epoch_of_best", "median"),
        stop=("epochs", "median")).reset_index()
    for mod in order_models(g.model.unique()):
        h = g[g.model == mod].sort_values("n")
        ax.scatter(h.n, h.best, marker=_marker(mod), s=32, alpha=0.8,
                   color=_color(mod), label=mod, edgecolors="white", linewidths=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trials the fold trained on (`n_train`, log)")
    ax.set_ylabel("epoch of best internal valid accuracy (log)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Training stops earlier the less data there is", fontsize=11)

    ax = axes[1]
    bins = pd.qcut(m.n_train, 6, duplicates="drop")
    labels, data = [], []
    for b, h in m.groupby(bins, observed=True):
        labels.append(f"{int(b.left)}–{int(b.right)}")
        data.append(h.epoch_of_best.to_numpy())
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_xlabel("trials the fold trained on")
    ax.set_ylabel("epoch of best (log)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("...and on the smallest folds it is epoch 1", fontsize=11)
    f.suptitle("Why the small datasets never train: patience-20 on a signal that "
               "stops moving", fontsize=12)
    f.tight_layout()
    return f


def selection_degeneracy(cm: pd.DataFrame, fits: pd.DataFrame, order: list[str],
                         eval_: str = "within_session"):
    """Does the internal validation accuracy ever move at all?

    Early stopping monitors skorch's internal 20 % split. On the smallest datasets that
    split is a handful of trials, and the measured consequence is worse than coarse: on
    alexmi, physionetmi and shin2017a within-session the fold-mean valid accuracy takes
    **one** distinct value over the entire run. A constant monitor means early stopping
    fires at a fixed epoch on no information and the selected model is whichever epoch
    the argmax tie-break happens to return.

    Those are exactly the datasets on which the deep arms sit at chance. So "deep
    learning fails there" has (at least) two candidate causes -- not enough data to
    learn, and a model-selection criterion that carries no signal -- and this figure is
    what separates them from the rest of the report rather than letting them be read as
    the same finding.
    """
    sub = cm[cm["eval"] == eval_]
    if sub.empty:
        return None
    nuniq = (sub.groupby(["dataset", "model", "seed"]).valid_acc_mean.nunique()
             .groupby(["dataset", "model"]).median())
    vsize = (fits[fits["eval"] == eval_].groupby("dataset").n_train.median() * 0.2)
    ds = [d for d in order if d in set(sub.dataset.unique())]
    models = order_models(sub.model.unique())
    f, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(ds))
    w = 0.8 / max(len(models), 1)
    for i, mod in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w,
               [nuniq.get((d, mod), np.nan) for d in ds], w,
               color=_color(mod), alpha=0.85, label=mod)
    ax.axhline(1, color="#c62828", lw=1.4, ls="--",
               label="1 = the monitor never moves")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("distinct valid-accuracy values along a run (log)")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_title("How much signal early stopping has to work with", fontsize=11)

    ax = axes[1]
    ax.barh(np.arange(len(ds)), [vsize.get(d, np.nan) for d in ds], color="0.55")
    ax.set_yticks(np.arange(len(ds))); ax.set_yticklabels(ds, fontsize=8)
    ax.axvline(10, color="#c62828", lw=1.2, ls="--", label="10 trials")
    ax.set_xscale("log")
    ax.set_xlabel("trials in skorch's internal validation split (20 % of n_train, log)")
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.set_title("...and how many trials it is measured on", fontsize=11)
    f.suptitle(f"Is the stopping criterion measurable? — {EVAL_LABEL[eval_]}",
               fontsize=12)
    f.tight_layout()
    return f
