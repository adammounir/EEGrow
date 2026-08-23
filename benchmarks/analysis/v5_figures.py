"""Figures for the v5 interim report.

One module rather than code inside notebook cells, for the reason `growth_io` exists:
a figure that encodes a claim is a thing to review, and a cell nobody re-reads is not
reviewable. The notebook imports these and calls them; `build_v5_report.py` calls the
same functions to write PNGs for the LaTeX version. There is exactly one definition of
each figure.

Every function takes already-loaded frames and returns the Figure. None of them read a
file: loading belongs to `growth_io`, so a path convention change breaks in one place.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVALS = ["within_session", "cross_session", "cross_subject"]
EVAL_LABEL = {"within_session": "within-session", "cross_session": "cross-session",
              "cross_subject": "cross-subject (LOSO)"}
PAIRS = [("grow_shallow", "bd_shallow"), ("grow_sccnet", "bd_sccnet"),
         ("grow_eegnex", "bd_eegnex"), ("grow_deep", "fix_deepeeg")]
PAIR_LABEL = {"grow_shallow vs bd_shallow": "ShallowFBCSPNet",
              "grow_sccnet vs bd_sccnet": "SCCNet",
              "grow_eegnex vs bd_eegnex": "EEGNeX",
              "grow_deep vs fix_deepeeg": "DeepEEGNet"}
GROW = ["grow_shallow", "grow_sccnet", "grow_eegnex", "grow_deep"]
EV_STYLE = {"within_session": ("o", "C0"), "cross_session": ("s", "C1"),
            "cross_subject": ("^", "C2")}


# ---------------------------------------------------------------- score-level figures
def delta_by_dataset(paired: pd.DataFrame):
    """Forest plot of the paired delta, one panel per architecture.

    A forest plot, not bars: the question is "is the effect on the same side of zero
    everywhere", and a shared zero line answers it directly. Filled marker = the
    growing arm won that dataset.
    """
    inv = {v: k for k, v in PAIR_LABEL.items()}
    archs = [a for a in PAIR_LABEL.values() if inv[a] in set(paired.pair)]
    f, axes = plt.subplots(1, len(archs), figsize=(4.2 * len(archs), 5.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, arch in zip(axes, archs):
        g = paired[paired.pair == inv[arch]]
        ds = sorted(g.dataset.unique())
        ypos = {d: i for i, d in enumerate(ds)}
        for ev, (m, c) in EV_STYLE.items():
            h = g[g["eval"] == ev]
            if not len(h):
                continue
            ax.scatter(h.delta, [ypos[d] for d in h.dataset], marker=m, s=46,
                       facecolors=[c if d > 0 else "none" for d in h.delta],
                       edgecolors=c, linewidths=1.4, label=EVAL_LABEL[ev])
        ax.axvline(0, color="0.3", lw=1)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(ds if ax is axes[0] else [""] * len(ds), fontsize=8)
        ax.set_title(arch, fontsize=11)
        ax.set_xlabel(r"$\Delta$ (grow $-$ fixed)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower left")
    f.suptitle("Filled marker = the growing arm won that dataset", fontsize=11)
    f.tight_layout()
    return f


def family_levels(levels: pd.DataFrame):
    """Absolute ROC-AUC by family. The paired delta cannot say whether either arm is
    any good; this can. AUC datasets only, so chance is 0.5 and the bars are
    comparable."""
    fam = (levels[levels.metric == "roc_auc"]
           .groupby(["family", "eval"], as_index=False).score_mean.mean()
           .pivot(index="family", columns="eval", values="score_mean")
           .reindex(columns=EVALS))
    f, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(EVALS))
    w = 0.26
    for i, family in enumerate(["braindecode", "growing", "fixed control"]):
        if family not in fam.index:
            continue
        vals = [fam.loc[family, e] for e in EVALS]
        ax.bar(x + (i - 1) * w, vals, w, label=family)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([EVAL_LABEL[e] for e in EVALS])
    ax.set_ylabel("ROC-AUC (AUC datasets only)")
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.5, color="0.4", lw=1, ls=":")
    ax.legend(fontsize=9)
    ax.set_title("Absolute level by family — chance is 0.5")
    f.tight_layout()
    return f


def win_matrix(paired: pd.DataFrame):
    """Fraction of datasets won, architecture x protocol.

    Deliberately the *unweighted* count over datasets: "growth helps on this dataset"
    is the unit the claim is about, and subjects inside one dataset are not independent
    draws of the population of datasets.
    """
    rows = []
    for pair, g in paired.groupby("pair"):
        for ev, h in g.groupby("eval"):
            rows.append(dict(arch=PAIR_LABEL.get(pair, pair), eval=ev,
                             frac=(h.delta > 0).mean(), n=len(h),
                             won=int((h.delta > 0).sum())))
    m = pd.DataFrame(rows)
    piv = m.pivot(index="arch", columns="eval", values="frac").reindex(columns=EVALS)
    lab = m.pivot(index="arch", columns="eval", values="won").reindex(columns=EVALS)
    tot = m.pivot(index="arch", columns="eval", values="n").reindex(columns=EVALS)
    f, ax = plt.subplots(figsize=(6.6, 3.6))
    im = ax.imshow(piv.values, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(EVALS)))
    ax.set_xticklabels([EVAL_LABEL[e] for e in EVALS], fontsize=9)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v != v:
                continue
            ax.text(j, i, f"{int(lab.values[i, j])}/{int(tot.values[i, j])}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(v - 0.5) > 0.32 else "black")
    f.colorbar(im, ax=ax, label="fraction of datasets won")
    ax.set_title("Datasets where the growing arm beat its matched control")
    f.tight_layout()
    return f


def noise_floor(scores: pd.DataFrame, rng_seed: int = 0):
    """Negative control: every model against ITSELF, seeds split in half.

    The true effect is zero by construction, so the spread returned here is the
    resolution of the whole procedure. A paired delta smaller than this is not a small
    effect, it is nothing -- which is the only honest way to read the near-zero
    DeepEEGNet and SCCNet contrasts.
    """
    rng = np.random.default_rng(rng_seed)
    seeds = sorted(scores.seed.unique())
    recs = []
    for _ in range(200):
        perm = rng.permutation(seeds)
        a, b = set(perm[: len(seeds) // 2]), set(perm[len(seeds) // 2:])
        for model, g in scores.groupby("model"):
            ga = (g[g.seed.isin(a)].groupby(["eval", "dataset", "subject", "session"])
                  .score.mean())
            gb = (g[g.seed.isin(b)].groupby(["eval", "dataset", "subject", "session"])
                  .score.mean())
            j = ga.to_frame("a").join(gb.to_frame("b"), how="inner").dropna()
            if len(j):
                recs.append(dict(model=model, delta=float((j.a - j.b).mean())))
    nf = pd.DataFrame(recs)
    f, ax = plt.subplots(figsize=(7.8, 4.2))
    order = sorted(nf.model.unique())
    ax.boxplot([nf[nf.model == m].delta for m in order], tick_labels=order,
               showfliers=False, widths=0.6)
    ax.axhline(0, color="0.3", lw=1)
    # Dashed lines, NOT a filled span. A shaded band spanning the whole plot reads as
    # "this is the noise region", which is the opposite of what it marks: the noise is
    # the boxes, and the lines are the effect being claimed above them.
    real = 0.0138  # the largest effect claimed anywhere in this report
    for sign in (-1, 1):
        ax.axhline(sign * real, color="C3", ls="--", lw=1.3,
                   label=(f"±{real:.4f} = largest effect claimed" if sign > 0
                          else None))
    floor = nf.delta.abs().quantile(0.95)
    ax.set_ylabel(r"$\Delta$ of a model against itself")
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Noise floor: 200 random half-splits of the seeds\n"
                 f"95th percentile of $|\\Delta|$ = {floor:.4f}, "
                 f"{real / floor:.1f}× below the largest effect claimed", fontsize=10)
    f.tight_layout()
    return f, nf


def seed_stability(scores: pd.DataFrame):
    """Spread across seeds, per model. A model whose seed spread exceeds the effect
    being claimed cannot support that claim from one seed."""
    obs = (scores.groupby(["eval", "dataset", "subject", "session", "model"])
           .score.std().reset_index(name="sd"))
    f, ax = plt.subplots(figsize=(7.4, 4.0))
    order = sorted(obs.model.unique())
    ax.boxplot([obs[obs.model == m].sd.dropna() for m in order], tick_labels=order,
               showfliers=False, widths=0.6)
    ax.set_ylabel("SD across the 5 seeds\n(within one subject/session)")
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.axhline(0.0138, color="C3", ls="--", lw=1,
               label="0.0138 = the largest effect this report claims")
    ax.legend(fontsize=8)
    ax.set_title("Seed-to-seed spread dwarfs every effect measured here")
    f.tight_layout()
    return f


# --------------------------------------------------------------- growth-level figures
def width_vs_target(fits: pd.DataFrame):
    """Where growth actually stopped, against the width it was compared to.

    This is the figure that reframes the whole report. The paired contrast is written
    as "growing to a target width versus training at that width directly", but the
    records say the growing arms mostly never arrive: the bar is the width reached, the
    marker is the target. A growing arm that wins from a third of the target width is
    making a *stronger* claim than the one the table states, not a weaker one.
    """
    g = fits[fits.model.isin(GROW)]
    agg = g.groupby("model").agg(
        w_start=("width_start", "mean"), w_end=("width_end", "mean"),
        w_max=("width_end", "max"), target=("target_width", "mean"),
        reached=("reached_target", "mean"), grew=("grew", "mean")).reindex(GROW)
    agg = agg.dropna(how="all")
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    y = np.arange(len(agg))
    ax.barh(y, agg.w_end - agg.w_start, left=agg.w_start, height=0.5,
            color="C0", alpha=0.85, label="mean width reached")
    ax.scatter(agg.w_start, y, marker="|", s=260, color="0.25", zorder=3,
               label="start width")
    ax.scatter(agg.target, y, marker="D", s=52, color="C3", zorder=3,
               label="target width (what it is compared to)")
    ax.scatter(agg.w_max, y, marker=">", s=52, facecolors="none", edgecolors="C0",
               zorder=3, label="best fold")
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index)
    ax.set_xlabel("growable width (filters)")
    # Room on the right for the legend, so it never sits on top of a target marker --
    # which is the one thing in this panel a reader must be able to see.
    ax.set_xlim(0, agg.target.max() * 1.42)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Growth stops well short of the target")
    ax.grid(axis="x", alpha=0.25)

    ax2.barh(y - 0.18, agg.grew, 0.34, color="C2", label="folds that grew at all")
    ax2.barh(y + 0.18, agg.reached, 0.34, color="C3",
             label="folds that reached the target")
    for i, (gr, re) in enumerate(zip(agg.grew, agg.reached)):
        ax2.text(gr + 0.012, i - 0.18, f"{gr:.0%}", va="center", fontsize=8)
        ax2.text(re + 0.012, i + 0.18, f"{re:.1%}", va="center", fontsize=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(agg))
    ax2.set_xlim(0, 1.22)
    ax2.set_xlabel("fraction of folds")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("A third of folds never grow at all")
    ax2.grid(axis="x", alpha=0.25)
    f.suptitle("What the growing arms actually did — v5, all folds", fontsize=11)
    f.tight_layout()
    return f, agg


def width_trajectory(curves: pd.DataFrame, dataset: str, eval_: str = "within_session"):
    """Width against epoch, one line per fold, drawn as a step function.

    Width is constant between growth events, so `steps-post` is the only honest
    interpolation: a straight line between two growths draws widths the model never
    had.
    """
    sel = curves[(curves.dataset == dataset) & (curves["eval"] == eval_)
                 & curves.model.isin(GROW)]
    models = [m for m in GROW if m in set(sel.model)]
    f, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.8),
                           sharey=False)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        g = sel[sel.model == model]
        for (seed, fit), h in g.groupby(["seed", "fit"]):
            h = h.sort_values("epoch")
            ax.step(h.epoch, h.width, where="post", lw=0.7, alpha=0.28, color="C0")
        tgt = g.groupby(["seed", "fit"]).width.max()
        ax.set_title(f"{model}\nmax reached {tgt.max():.0f}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("growable width")
    f.suptitle(f"Growth trajectory — one line per fold, {dataset} / {eval_}",
               fontsize=11)
    f.tight_layout()
    return f


def learning_curves(curves: pd.DataFrame, dataset: str, eval_: str = "within_session"):
    """Train loss, valid loss and valid accuracy against epoch, growable vs fixed.

    This is the figure that did not exist when the S2 anomaly was reported: the script
    behind the published table attached no recorder, so there were no curves to look at.
    """
    sel = curves[(curves.dataset == dataset) & (curves["eval"] == eval_)]
    show = [m for m in ["grow_shallow", "bd_shallow", "grow_sccnet", "bd_sccnet"]
            if m in set(sel.model)]
    f, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for k, (col, name) in enumerate([("train_loss", "train loss"),
                                     ("valid_loss", "valid loss"),
                                     ("valid_acc", "valid accuracy")]):
        ax = axes[k]
        if col not in sel.columns:
            ax.text(0.5, 0.5, f"{col} not recorded", ha="center", va="center")
            ax.set_axis_off()
            continue
        for i, model in enumerate(show):
            g = sel[sel.model == model].groupby("epoch")[col]
            mean, sd = g.mean(), g.std()
            ax.plot(mean.index, mean.values, lw=1.6, color=f"C{i}", label=model,
                    ls="-" if model.startswith("grow") else "--")
            ax.fill_between(mean.index, mean - sd, mean + sd, color=f"C{i}",
                            alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    f.suptitle(f"Learning curves, mean ± SD over folds — {dataset} / {eval_}",
               fontsize=11)
    f.tight_layout()
    return f


def capacity(curves: pd.DataFrame, dataset: str, eval_: str = "within_session"):
    """Parameter count over training: instantaneous, and cumulative.

    The cumulative panel is the budget the run actually paid. Final parameter count
    flatters a growing model, which spent most of its epochs narrower than it ended.
    """
    sel = curves[(curves.dataset == dataset) & (curves["eval"] == eval_)]
    show = [m for m in ["grow_shallow", "bd_shallow", "grow_sccnet", "bd_sccnet"]
            if m in set(sel.model)]
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.0))
    for i, model in enumerate(show):
        g = sel[sel.model == model].groupby("epoch").n_params.mean()
        ax.step(g.index, g.values, where="post", lw=1.6, color=f"C{i}", label=model,
                ls="-" if model.startswith("grow") else "--")
        ax2.plot(g.index, g.cumsum().values, lw=1.6, color=f"C{i}", label=model,
                 ls="-" if model.startswith("grow") else "--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("parameters")
    ax.set_title("Instantaneous capacity", fontsize=10)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cumulative parameter-epochs")
    ax2.set_title("Budget actually paid", fontsize=10)
    for a in (ax, ax2):
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    f.suptitle(f"Capacity over training — {dataset} / {eval_}", fontsize=11)
    f.tight_layout()
    return f


def budget_pareto(budget: pd.DataFrame, dataset: str, eval_: str = "within_session"):
    """Accuracy against parameter-epochs, one dataset at a time.

    "Same accuracy for a smaller budget" is the claim growth is actually making, so the
    x axis is cost and further LEFT at equal height is better. Log x because the
    architectures span two orders of magnitude.

    **One dataset, not the grid.** Pooling would average validation accuracy over
    datasets with two classes and datasets with four, whose chance levels are 0.5 and
    0.25 -- a mean over both is a mean over two different quantities, the same mistake
    the metric rule exists to prevent. So the caller names the dataset and the axis
    means one thing.
    """
    sel = budget[(budget.dataset == dataset) & (budget["eval"] == eval_)]
    b = sel.groupby("model").agg(
        pe=("param_epochs", "median"), acc=("best_valid_acc", "mean"),
        acc_sd=("best_valid_acc", "std"), n=("param_epochs", "size"))
    b = b[b.pe > 0].dropna(subset=["acc"]).sort_values("pe")
    f, ax = plt.subplots(figsize=(8.4, 5.0))
    # Alternate the label side so neighbouring points on the log axis do not collide.
    for i, (model, r) in enumerate(b.iterrows()):
        fam = ("C0" if model.startswith("grow") else
               "C1" if model.startswith("bd_") else "C2")
        ax.errorbar(r.pe, r.acc, yerr=r.acc_sd, fmt="o", ms=9, color=fam,
                    capsize=3, alpha=0.9)
        dy = 11 if i % 2 == 0 else -17
        ax.annotate(model, (r.pe, r.acc), textcoords="offset points",
                    xytext=(0, dy), fontsize=8, ha="center", color=fam)
    ax.set_xscale("log")
    ax.set_xlabel("median parameter-epochs per fold  (cost — further left is cheaper)")
    ax.set_ylabel("mean best validation accuracy")
    ax.grid(alpha=0.25)
    ax.set_title(f"Budget/accuracy trade-off — {dataset} / {eval_}\n"
                 "up and to the left is better; blue = growing, "
                 "orange = braindecode, green = fixed control", fontsize=10)
    f.tight_layout()
    return f, b


def subject_curves(curves: pd.DataFrame, subject, pair=("grow_sccnet", "bd_sccnet"),
                   eval_: str = "within_session"):
    """One subject's curves against the rest of the cohort, growable vs fixed.

    Needs ``curves`` passed through ``growth_io.attach_subjects`` -- the subject label
    is inferred from write order, not recorded, and that function documents what the
    inference rests on.

    The cohort mean is on every panel in grey for a reason: the question asked of a
    single subject is never "what did its curve do" but "did its curve do something the
    others did not". Without the reference, any curve looks anomalous.
    """
    sel = curves[(curves["eval"] == eval_) & curves.subject.notna()]

    def trace(sub: pd.DataFrame, col: str):
        """Mean and SD over fits, **truncated where half the fits have stopped**.

        Every fit early-stops at its own epoch, so a plain mean over epoch is a mean
        over a shrinking, self-selected set: the folds still alive at epoch 120 are the
        ones that were still improving, which makes accuracy appear to keep climbing
        long after most fits ended. Cutting at 50% survival keeps a comparison between
        two arms a comparison over the same epochs.
        """
        g = sub.groupby("epoch")[col]
        n = g.size()
        keep = n.index[n >= 0.5 * n.max()]
        return g.mean().loc[keep], g.std().loc[keep]

    f, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    panels = [("train_loss", "train loss"), ("valid_acc", "valid accuracy"),
              ("width", "growable width")]
    for k, (col, name) in enumerate(panels):
        ax = axes[k]
        for i, model in enumerate(pair):
            m = sel[sel.model == model]
            if col == "width" and m[col].isna().all():
                continue  # fixed arm has no growable width: leave the panel to the other
            others, _ = trace(m[m.subject != subject], col)
            ax.plot(others.index, others.values, color="0.65", lw=1.2,
                    ls="-" if model.startswith("grow") else "--",
                    label=f"{model}, other 8 subjects")
            mean, sd = trace(m[m.subject == subject], col)
            ax.plot(mean.index, mean.values, lw=1.9, color=f"C{i}",
                    ls="-" if model.startswith("grow") else "--",
                    label=f"{model}, S{subject}")
            ax.fill_between(mean.index, mean - sd, mean + sd, color=f"C{i}", alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="lower right")
    f.suptitle(f"S{subject} vs the rest of the cohort — {eval_}, mean ± SD over "
               f"5 seeds x 2 sessions x 5 folds (cut at 50% fold survival)",
               fontsize=11)
    f.tight_layout()
    return f


def stopping_budget(fits: pd.DataFrame, *, max_epochs: int = 200,
                    grow_every: int = 5):
    """Why the growing arms never reach their target: they run out of epochs.

    The target width is not a property of the growth mechanism alone -- it is a race
    between the growth schedule (one opportunity every ``grow_every`` epochs) and the
    stopping rule. This panel puts the two on the same axis, which is the only way the
    race is visible.

    Left: how long fits actually run against the 200-epoch budget they were given.
    Right: neurons needed to reach target, against neurons the arm gets per opportunity
    at the median fit length. A bar above the line cannot reach its target no matter
    how well gromo picks its neurons.
    """
    g = fits[fits.model.isin(GROW)]
    need = {"grow_shallow": 32, "grow_sccnet": 18, "grow_deep": 24, "grow_eegnex": 6}
    order = [m for m in GROW if m in set(g.model)]

    f, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ax.boxplot([g[g.model == m].epochs.values for m in order], vert=False,
               tick_labels=[m.replace("grow_", "") for m in order], showfliers=False,
               widths=0.55)
    ax.axvline(max_epochs, color="C3", ls="--", lw=1.4,
               label=f"max_epochs = {max_epochs}")
    ax.set_xlabel("epochs actually run")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    frac = (g.epochs < max_epochs).mean()
    ax.set_title(f"Every fit stops early: {100 * frac:.1f}% of "
                 f"{len(g):,} folds end before the budget", fontsize=10)

    # Both rates are computed PER FOLD and then medianed, so the two bars share a
    # denominator. Mixing a median fit length into one and a pooled sum into the other
    # compares two different quantities, which is the whole point of the panel.
    x = np.arange(len(order))
    req, got = [], []
    for m in order:
        h = g[g.model == m]
        opp = ((h.epochs - 1) // grow_every).clip(lower=1)
        req.append(float((need[m] / opp).median()))
        got.append(float(((h.width_end - h.width_start) / opp).median()))
    ax2.bar(x - 0.19, req, 0.36, color="C3", label="needed per opportunity")
    ax2.bar(x + 0.19, got, 0.36, color="C0", label="actually added per opportunity")
    for i, (r, o) in enumerate(zip(req, got)):
        ax2.annotate(f"{r:.1f}", (i - 0.19, r), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8, color="C3")
        ax2.annotate(f"{o:.2f}", (i + 0.19, o), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8, color="C0")
    ax2.set_xticks(x, [m.replace("grow_", "") for m in order])
    ax2.set_ylabel("neurons per growth opportunity")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_title(f"Growth rate vs the rate the target demands\n"
                  f"(grow_every={grow_every}, median fit length)", fontsize=10)
    f.tight_layout()
    return f


def growth_events(curves: pd.DataFrame, dataset: str, eval_: str = "within_session"):
    """When growth actually happens, and how long the fit trains after it stops.

    Two things end growth before the target: reaching the cap, and the latch in
    ``GromoGrowth.on_epoch_end`` that sets ``done_`` after a single step which adds no
    neurons, disabling growth for the rest of that fit. Either way the gap between the
    last growth and the last epoch is capacity the fit never got to use, so it is
    plotted directly.
    """
    sel = curves[(curves.dataset == dataset) & (curves["eval"] == eval_)
                 & curves.model.isin(GROW)]
    rows = []
    for (model, _, _), d in sel.groupby(["model", "seed", "fit"]):
        d = d.sort_values("epoch")
        rose = d.width.diff().fillna(0) > 0
        rows.append({"model": model, "n_growths": int(rose.sum()),
                     "last_epoch": int(d.epoch.max()),
                     "last_growth": int(d.epoch[rose].max()) if rose.any() else 0})
    r = pd.DataFrame(rows)
    r["idle"] = r.last_epoch - r.last_growth
    order = [m for m in GROW if m in set(r.model)]

    f, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ax.boxplot([r[r.model == m].n_growths.values for m in order], vert=False,
               tick_labels=[m.replace("grow_", "") for m in order], showfliers=False,
               widths=0.55)
    ax.set_xlabel("growth events per fit (width strictly increased)")
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("How many times a fit actually grew", fontsize=10)
    ax2.boxplot([r[r.model == m].idle.values for m in order], vert=False,
                tick_labels=[m.replace("grow_", "") for m in order], showfliers=False,
                widths=0.55)
    ax2.set_xlabel("epochs trained after the last growth")
    ax2.grid(alpha=0.25, axis="x")
    ax2.set_title("Epochs spent frozen at the width it stopped at", fontsize=10)
    f.suptitle(f"Growth timing — {dataset} / {eval_}, {len(r)} fits", fontsize=11)
    f.tight_layout()
    return f, r


def subject_delta(scores: pd.DataFrame, dataset: str, pair=("grow_sccnet", "bd_sccnet"),
                  eval_: str = "within_session"):
    """Per-subject paired delta with its per-seed sign, for one dataset and pair.

    The dataset-level delta is a mean over subjects, and a mean hides a split: on
    bnci2014_001 the SCCNet contrast is near zero overall while two subjects lose
    consistently and the rest gain slightly. Reporting only the mean answers a question
    nobody asked about a single subject.

    Colour encodes how many of the five seeds agree on the sign, which is what
    separates a real per-subject effect from one seed's noise.
    """
    s = scores[(scores.dataset == dataset) & (scores["eval"] == eval_)
               & scores.model.isin(pair)]
    p = (s.pivot_table(index=["subject", "seed"], columns="model", values="score")
         .assign(d=lambda x: x[pair[0]] - x[pair[1]]).reset_index())
    agg = p.groupby("subject").d.agg(["mean", "std"])
    n_neg = p.assign(neg=p.d < 0).groupby("subject").neg.sum()

    f, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(agg))
    cmap = plt.get_cmap("coolwarm")
    ax.barh(y, agg["mean"], xerr=agg["std"], height=0.6,
            color=[cmap(n / 5) for n in n_neg], edgecolor="0.3", lw=0.6,
            error_kw={"lw": 1.0, "ecolor": "0.35"})
    for i, (subj, n) in enumerate(n_neg.items()):
        ax.annotate(f"{int(n)}/5", (agg["mean"].iloc[i], i), fontsize=8,
                    textcoords="offset points", ha="left" if agg["mean"].iloc[i] >= 0
                    else "right", xytext=(6 if agg["mean"].iloc[i] >= 0 else -6, -3))
    ax.axvline(0, color="0.2", lw=1.0)
    ax.set_yticks(y, [f"S{s}" for s in agg.index])
    ax.set_xlabel(f"{pair[0]} − {pair[1]}   (mean over 5 seeds, ± SD across seeds)")
    ax.grid(alpha=0.25, axis="x")
    ax.set_title(f"Per-subject delta — {dataset} / {eval_}\n"
                 "label = seeds out of 5 that agree the delta is negative; "
                 "red = all five", fontsize=10)
    f.tight_layout()
    return f, agg.assign(n_neg=n_neg)
