"""Exploration figures for the 14-model grid: who wins, where, and how hard is what.

These answer a different question from `v5_figures`, on different data. `v5_figures`
asks "did growing help", on the v5 campaign, which has no classical baseline at all.
This module asks the questions Sylvain asked for -- which model is the *champion* of its
family on each dataset, which datasets are easy and which are hard, and how a model's
behaviour changes across the three evaluation modes -- and those can only be asked on
`results_published`, the campaign that still carries the six riemann/csp pipelines.

Two conventions run through every figure here, and both exist because the naive version
of the plot is misleading:

**Score minus chance, never raw score.** The grid mixes metrics: six datasets are scored
with ROC-AUC (chance 0.5) and six with accuracy (chance 1/n_classes, so 0.25 on the
four-class ones). Putting raw scores on one axis makes bnci2014_001 look catastrophic
next to shin2017a when in fact it is the easier of the two. Subtracting chance is the
minimum needed to make a cross-dataset axis mean anything, and it is still not a
perfect normalisation -- a ceiling at 1.0 compresses the easy end.

**One fixed dataset order everywhere.** Ordered by measured difficulty, hardest on the
left. The point of a fixed order is that flipping between two figures becomes a
comparison: a dataset that sits low in every panel is a property of the dataset, not of
the model being plotted. `ORDER_XSESS` is the single exception, used only by the
evaluation-mode figure, where the six datasets that have a cross-session split have to
come first for the truncated line to be readable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Canonical family order and colours. A superset: the two campaigns declare different
# families (`results_published` has riemann/csp and no fixed control, v5 the reverse),
# and every figure filters this down to what its frame actually contains rather than
# assuming three. Green / blue / red as asked, with the fixed control a darker blue so
# it reads as a member of the fixed side rather than a fourth thing.
FAMILIES = ["riemann/csp", "braindecode", "fixed control", "growing"]
FAM_COLOR = {"riemann/csp": "#2e7d32", "braindecode": "#1565c0",
             "fixed control": "#0b3c73", "growing": "#c62828"}
FAM_MARKER = {"riemann/csp": "o", "braindecode": "s", "fixed control": "D",
              "growing": "^"}

EVALS = ["within_session", "cross_session", "cross_subject"]
EVAL_LABEL = {"within_session": "within-session", "cross_session": "cross-session",
              "cross_subject": "cross-subject (LOSO)"}
EVAL_COLOR = {"within_session": "#0b6e99", "cross_session": "#c77700",
              "cross_subject": "#7b2d8b"}

# Model order inside a family, fixed so the x axis never reshuffles between figures.
MODEL_ORDER = ["csp_lda", "csp_svm", "mdm", "fgmdm", "ts_lr", "ts_svm",
               "bd_shallow", "bd_sccnet", "bd_eegnex", "bd_deep4", "fix_deepeeg",
               "grow_shallow", "grow_sccnet", "grow_eegnex", "grow_deep"]


def families_in(tidy: pd.DataFrame) -> list[str]:
    """The families this frame actually has, in canonical order.

    Called instead of reading `FAMILIES` directly so the same figure code serves both
    campaigns. A family the frame does not contain must not occupy an x-offset slot:
    an empty green bar next to every blue one reads as "the classical pipeline scored
    zero here", which is the opposite of "it was not run".
    """
    present = set(tidy.family.unique())
    return [f for f in FAMILIES if f in present] + sorted(present - set(FAMILIES))


def order_models(names) -> list[str]:
    """Canonical order for an arbitrary set of model names, unknown ones appended."""
    present = set(names)
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def models_in(tidy: pd.DataFrame) -> list[str]:
    """Present models in canonical order, unknown names appended alphabetically."""
    return order_models(tidy.model.unique())


def family_of(model: str) -> str:
    """Family from the model name, same rule as `aggregate_published._family`.

    Duplicated deliberately rather than imported: the curve frames carry no `family`
    column (the growth records key on the model name only), and a figure module that
    needed the score table just to colour a line would not be usable on its own.
    """
    if model.startswith("grow"):
        return "growing"
    if model.startswith("bd_"):
        return "braindecode"
    if model.startswith("fix_"):
        return "fixed control"
    return "riemann/csp"


def prepare(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (eval, dataset, model, subject): the analysis unit for every figure.

    Two reductions, both deliberate:

    *Chance is subtracted* -- see the module docstring.

    *Seeds and sessions are averaged away first, subjects are not.* The error bars in
    these figures are over subjects, because that is the population a claim like "this
    dataset is hard" generalises over. Leaving seed and session as separate rows would
    triple the n and shrink every interval by sqrt(3) without adding an independent
    observation of anything.
    """
    s = scores.copy()
    s["chance"] = np.where(s.metric == "roc_auc", 0.5, 1.0 / s.n_classes)
    s["above"] = s.score - s.chance
    return (s.groupby(["eval", "dataset", "model", "family", "subject"], as_index=False)
            .agg(above=("above", "mean"), score=("score", "mean"),
                 chance=("chance", "first"), metric=("metric", "first"),
                 n_classes=("n_classes", "first"), channels=("channels", "first"),
                 samples=("samples", "first")))


def dataset_order(tidy: pd.DataFrame, eval_: str = "within_session") -> list[str]:
    """Datasets hardest-first, by mean above-chance over all 14 models.

    Computed from `within_session` and then reused for the other two evaluation modes
    rather than recomputed per panel: an order that changes between panels destroys the
    only thing a shared order buys, which is that vertical position means the same
    thing everywhere.
    """
    g = tidy[tidy["eval"] == eval_].groupby("dataset").above.mean()
    return list(g.sort_values().index)


def _cell(tidy: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and 95% normal CI over subjects, for the given grouping."""
    g = tidy.groupby(keys).above
    out = g.agg(mean="mean", sd="std", n="size").reset_index()
    out["ci"] = 1.96 * out.sd / np.sqrt(out.n.clip(lower=1))
    return out


# ------------------------------------------------------------------- family champions
def champion_vs_mean(tidy: pd.DataFrame, order: list[str]):
    """The figure that replaces a family average: champion against family mean.

    A family mean answers "how good is the average member", which is a question nobody
    has. If riemann/csp holds one pipeline that wins and five that are mediocre, the
    mean reports mediocre and the finding is gone. Here the bar is the family mean and
    the marker is the family's best model on that dataset, labelled -- so the gap
    between the two *is* the amount the mean was hiding.
    """
    fams = families_in(tidy)
    f, axes = plt.subplots(len(EVALS), 1, figsize=(13, 12), sharex=True)
    x = np.arange(len(order))
    width = 0.78 / len(fams)
    for ax, ev in zip(axes, EVALS):
        sub = tidy[tidy["eval"] == ev]
        m = _cell(sub, ["dataset", "family", "model"])
        for i, fam in enumerate(fams):
            h = m[m.family == fam].set_index(["dataset", "model"])
            means, champs, names = [], [], []
            for d in order:
                if d not in h.index.get_level_values(0):
                    means.append(np.nan); champs.append(np.nan); names.append("")
                    continue
                blk = h.loc[d]
                means.append(blk["mean"].mean())
                champs.append(blk["mean"].max())
                names.append(blk["mean"].idxmax())
            off = (i - (len(fams) - 1) / 2) * width
            ax.bar(x + off, means, width, color=FAM_COLOR[fam], alpha=0.32,
                   edgecolor=FAM_COLOR[fam], label=f"{fam} (family mean)")
            ax.scatter(x + off, champs, marker=FAM_MARKER[fam], s=44, zorder=3,
                       color=FAM_COLOR[fam], label=f"{fam} (champion)")
            for xi, (c, nm) in enumerate(zip(champs, names)):
                if np.isfinite(c) and nm:
                    ax.annotate(nm.replace("bd_", "").replace("grow_", "").replace("csp_", ""),
                                (xi + off, c), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=6,
                                color=FAM_COLOR[fam], rotation=90)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_ylabel("score $-$ chance")
        ax.set_title(EVAL_LABEL[ev], fontsize=11, loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    axes[-1].set_xlabel("dataset (hardest to easiest, same order in every figure)")
    f.suptitle("Family champion vs family mean — the gap is what an average hides",
               fontsize=12)
    f.tight_layout()
    return f


def champion_lines(tidy: pd.DataFrame, order: list[str]):
    """Champions only, as three lines: which family is on top, at a glance.

    Same data as the previous figure with the family means dropped. Worth having both:
    the bars answer "is the mean lying", the lines answer "who wins where", and reading
    the second off the first means visually subtracting six overlapping bars.
    """
    f, axes = plt.subplots(1, len(EVALS), figsize=(15, 4.6), sharey=True)
    x = np.arange(len(order))
    for ax, ev in zip(axes, EVALS):
        sub = tidy[tidy["eval"] == ev]
        m = _cell(sub, ["dataset", "family", "model"])
        for fam in families_in(tidy):
            h = m[m.family == fam]
            best = h.loc[h.groupby("dataset")["mean"].idxmax()].set_index("dataset")
            y = [best["mean"].get(d, np.nan) for d in order]
            e = [best["ci"].get(d, np.nan) for d in order]
            ax.errorbar(x, y, yerr=e, marker=FAM_MARKER[fam], color=FAM_COLOR[fam],
                        lw=1.6, ms=5, capsize=2, label=fam)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
        ax.set_title(EVAL_LABEL[ev], fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("champion's score $-$ chance")
    axes[0].legend(fontsize=8)
    f.suptitle("Best model of each family, per dataset (95% CI over subjects)",
               fontsize=12)
    f.tight_layout()
    return f


def champion_share(tidy: pd.DataFrame):
    """How often each model is its family's champion, over the 12x3 dataset-eval cells.

    A champion that changes with every dataset is not a champion, it is noise picked by
    a maximum over 14 models. This counts how concentrated the wins are, which is what
    licenses (or refuses) the sentence "ts_lr is the classical baseline to beat".
    """
    m = _cell(tidy, ["eval", "dataset", "family", "model"])
    best = m.loc[m.groupby(["eval", "dataset", "family"])["mean"].idxmax()]
    cnt = best.groupby(["family", "model"]).size().rename("wins").reset_index()
    n_cells = best.groupby("family").size()
    f, ax = plt.subplots(figsize=(9, 4.6))
    pos, labels, colors = [], [], []
    y = 0
    for fam in families_in(tidy):
        h = cnt[cnt.family == fam].sort_values("wins")
        for _, r in h.iterrows():
            pos.append((y, r.wins)); labels.append(r.model); colors.append(FAM_COLOR[fam])
            y += 1
        y += 0.8
    ax.barh([p for p, _ in pos], [w for _, w in pos], color=colors, alpha=0.85)
    ax.set_yticks([p for p, _ in pos])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"dataset x eval cells won inside its own family "
                  f"(max {int(n_cells.max())})")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Is the champion stable? Wins per model within its family", fontsize=12)
    f.tight_layout()
    return f


# ------------------------------------------------------------------- the crossed plots
def per_dataset_all_models(tidy: pd.DataFrame, order: list[str],
                           eval_: str = "within_session"):
    """One panel per dataset, all 14 models, coloured by family.

    This is the "for one dataset, how do all the algorithms behave" half of the crossed
    pair. Model order on the x axis is canonical and identical in every panel, so a
    shape is comparable between panels: a panel where the green block sits above the
    blue block is a dataset where the classical pipelines win outright.
    """
    sub = tidy[tidy["eval"] == eval_]
    m = _cell(sub, ["dataset", "model", "family"]).set_index(["dataset", "model"])
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    models = models_in(tidy)
    x = np.arange(len(models))
    for ax, d in zip(axes, order):
        for xi, mod in enumerate(models):
            if (d, mod) not in m.index:
                continue
            r = m.loc[(d, mod)]
            ax.errorbar([xi], [r["mean"]], yerr=[r["ci"]], marker="o", ms=4,
                        color=FAM_COLOR[r["family"]], capsize=2)
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_title(d, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=90, fontsize=5.5)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(order):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("score $-$ chance", fontsize=8)
    f.suptitle(f"Every model, one panel per dataset — {EVAL_LABEL[eval_]} "
               f"(green riemann/csp, blue braindecode, red growing)", fontsize=12)
    f.tight_layout()
    return f


def per_model_all_evals(tidy: pd.DataFrame, order_xsess: list[str], n_xsess: int):
    """One panel per model, the 12 within / 6 cross-session / 12 cross-subject values.

    Built to the shape asked for: the six datasets that *have* a cross-session split
    come first on the x axis, so the cross-session line stops halfway across instead of
    breaking into six disconnected segments. The dashed rule marks where it must stop --
    without it a reader reads the truncation as a model that failed on the right half.
    """
    models = models_in(tidy)
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow),
                           sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(order_xsess))
    fam_of = dict(zip(tidy.model, tidy.family))
    for ax, mod in zip(axes, models):
        for ev in EVALS:
            h = _cell(tidy[(tidy["eval"] == ev) & (tidy.model == mod)],
                      ["dataset"]).set_index("dataset")
            y = [h["mean"].get(d, np.nan) for d in order_xsess]
            ax.plot(x, y, marker="o", ms=3.5, lw=1.3, color=EVAL_COLOR[ev],
                    label=EVAL_LABEL[ev])
        ax.axvline(n_xsess - 0.5, color="0.5", ls="--", lw=0.9)
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_title(mod, fontsize=9, color=FAM_COLOR[fam_of.get(mod, "growing")])
        ax.set_xticks(x)
        ax.set_xticklabels(order_xsess, rotation=90, fontsize=5.5)
        ax.grid(alpha=0.3)
    for ax in axes[len(models):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("score $-$ chance", fontsize=8)
    axes[0].legend(fontsize=6)
    f.suptitle("Every dataset, one panel per model — the six cross-session datasets "
               "come first, which is where the orange line stops", fontsize=12)
    f.tight_layout()
    return f


# ----------------------------------------------------------------- dataset properties
def difficulty(tidy: pd.DataFrame, order: list[str]):
    """How hard is each dataset, and what is it made of.

    The question this exists to answer is the operational one: "name me an easy dataset
    and a hard one". Left panel is the answer, as the best score any of the 14 models
    reached (a dataset is as easy as its easiest win, not as its average entrant) next
    to the all-model mean. Right panel is the material the answer is made of -- trials,
    channels, classes, metric -- because a difficulty ranking with no mechanism beside
    it is a list to memorise rather than something to reason with.
    """
    f, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                           gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    for ev in EVALS:
        m = _cell(tidy[tidy["eval"] == ev], ["dataset"]).set_index("dataset")
        best = (tidy[tidy["eval"] == ev].groupby(["dataset", "model"]).above.mean()
                .groupby("dataset").max())
        ax.plot([m["mean"].get(d, np.nan) for d in order], y, marker="o", ms=4,
                lw=1.4, color=EVAL_COLOR[ev], label=f"{EVAL_LABEL[ev]} (mean)")
        ax.plot([best.get(d, np.nan) for d in order], y, marker="*", ms=9, ls=":",
                lw=1.0, color=EVAL_COLOR[ev], label=f"{EVAL_LABEL[ev]} (best model)")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("score $-$ chance")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Difficulty: mean over 14 models, and the best of them", fontsize=10)

    ax = axes[1]
    props = (tidy.groupby("dataset")
             .agg(trials=("samples", "median"), channels=("channels", "median"),
                  classes=("n_classes", "first"), metric=("metric", "first"),
                  subjects=("subject", "nunique")))
    cell = props.loc[order]
    ax.axis("off")
    tbl = ax.table(cellText=[[f"{int(r.trials)}", f"{int(r.channels)}",
                              f"{int(r.classes)}", r.metric, f"{int(r.subjects)}"]
                             for _, r in cell.iterrows()],
                   rowLabels=order,
                   colLabels=["trials/subj", "chan", "cls", "metric", "subj"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.35)
    ax.set_title("What each dataset is made of", fontsize=10)
    f.suptitle("Easy and hard, in the same dataset order as every other figure",
               fontsize=12)
    f.tight_layout()
    return f


def rank_heatmap(tidy: pd.DataFrame, order: list[str], eval_: str = "within_session"):
    """Rank of each model within each dataset, 1 = best of 14.

    Ranks, not scores, because the scores are not on one scale even after subtracting
    chance: a 0.02 gap is decisive on shin2017a and noise on schirrmeister2017. Ranking
    inside a dataset removes the dataset's difficulty entirely, which is what makes the
    row means comparable -- and the row means are the answer to "if I only get to beat
    one model, which one".
    """
    sub = tidy[tidy["eval"] == eval_]
    m = _cell(sub, ["dataset", "model"]).pivot(index="model", columns="dataset",
                                               values="mean")
    m = m.reindex(index=[x for x in models_in(tidy) if x in m.index], columns=order)
    ranks = m.rank(axis=0, ascending=False)
    row_order = ranks.mean(axis=1).sort_values().index
    ranks = ranks.loc[row_order]
    f, ax = plt.subplots(figsize=(11, 5.4))
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="RdYlGn_r", vmin=1,
                vmax=len(ranks), cbar_kws={"label": "rank within dataset"},
                annot_kws={"size": 7}, ax=ax, linewidths=0.4, linecolor="white")
    ax.set_xlabel("dataset (hardest to easiest)")
    ax.set_ylabel("")
    ax.set_yticklabels([f"{t.get_text()}  (mean rank "
                        f"{ranks.mean(axis=1)[t.get_text()]:.1f})"
                        for t in ax.get_yticklabels()], fontsize=8, rotation=0)
    ax.set_title(f"Rank within dataset — {EVAL_LABEL[eval_]}, rows sorted by mean rank",
                 fontsize=12)
    f.tight_layout()
    return f


def eval_penalty(tidy: pd.DataFrame, order: list[str]):
    """What each evaluation mode costs, per dataset and per family.

    Generalising across sessions and across subjects are two different asks, and the
    interesting question is whether the cost of each is a property of the dataset or of
    the model family. Plotted as a delta against that dataset's own within-session
    result so the dataset's difficulty cancels out.
    """
    base = (tidy[tidy["eval"] == "within_session"]
            .groupby(["dataset", "family"]).above.mean())
    f, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, ev in zip(axes, ["cross_session", "cross_subject"]):
        cur = tidy[tidy["eval"] == ev].groupby(["dataset", "family"]).above.mean()
        d = (cur - base).dropna()
        x = np.arange(len(order))
        fams = families_in(tidy)
        w = 0.78 / len(fams)
        for i, fam in enumerate(fams):
            y = [d.get((ds, fam), np.nan) for ds in order]
            ax.bar(x + (i - (len(fams) - 1) / 2) * w, y, w, color=FAM_COLOR[fam],
                   alpha=0.85, label=fam)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{EVAL_LABEL[ev]} $-$ within-session", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"$\Delta$ (score $-$ chance)")
    axes[0].legend(fontsize=8)
    f.suptitle("The cost of generalising, by dataset and family "
               "(negative = harder than within-session)", fontsize=12)
    f.tight_layout()
    return f


def best_of(tidy: pd.DataFrame, fams: list[str]) -> pd.Series:
    """Best single model of `fams` on each dataset, as a Series indexed by dataset."""
    m = (tidy[tidy.family.isin(fams)]
         .groupby(["dataset", "model"]).above.mean())
    return m.groupby("dataset").max()


def delta_champion(a: list[str], b: list[str]):
    """Delta between the champion of `a` and the champion of `b`, per dataset.

    Carries a selection bias worth naming: a maximum over more candidates is larger in
    expectation, so this favours whichever side has more models. It is fine when the
    gap is large (riemann/csp beats the deep models by 0.1-0.3, far past what a max over
    six versus eight can manufacture) and misleading when it is small -- which is why
    the growing-versus-fixed contrast uses `delta_pairs` instead.
    """
    def fn(sub: pd.DataFrame) -> pd.Series:
        return (best_of(sub, a) - best_of(sub, b)).dropna()
    return fn


def delta_pairs(pairs: list[tuple[str, str]]):
    """Mean architecture-matched delta per dataset: mean over pairs of (first - second).

    The unbiased version of the above when the two sides are the same architectures
    trained two ways. Each pair contributes one number, so neither side benefits from
    having more entrants, and a pair missing on a dataset drops out instead of being
    silently replaced by a different architecture.
    """
    def fn(sub: pd.DataFrame) -> pd.Series:
        m = sub.groupby(["dataset", "model"]).above.mean()
        out = {}
        for d in sub.dataset.unique():
            blk = m.loc[d] if d in m.index.get_level_values(0) else None
            if blk is None:
                continue
            vals = [blk[x] - blk[y] for x, y in pairs if x in blk.index and y in blk.index]
            if vals:
                out[d] = float(np.mean(vals))
        return pd.Series(out)
    return fn


def group_scatter(tidy: pd.DataFrame, order: list[str], a: list[str], b: list[str],
                  *, label_a: str, label_b: str):
    """Champion of one group against champion of another, one point per dataset x mode.

    The diagonal is parity; a point below it is a dataset where nothing in group A
    reached what group B reached. Marker shape is the evaluation mode, so a cloud that
    crosses the diagonal only for one mode is visible as such -- which is exactly what
    happens on the 14-model grid, and is the whole reason `train_size_crossover` exists.
    """
    f, ax = plt.subplots(figsize=(6.6, 6.4))
    mk = {"within_session": "o", "cross_session": "s", "cross_subject": "^"}
    lo, hi = np.inf, -np.inf
    for ev in EVALS:
        sub = tidy[tidy["eval"] == ev]
        if sub.empty:
            continue
        xs, ys = best_of(sub, b), best_of(sub, a)
        common = [d for d in order if d in xs.index and d in ys.index]
        if not common:
            continue
        lo = min(lo, xs[common].min(), ys[common].min())
        hi = max(hi, xs[common].max(), ys[common].max())
        ax.scatter(xs[common], ys[common], marker=mk[ev], s=52, alpha=0.85,
                   color=EVAL_COLOR[ev], label=EVAL_LABEL[ev],
                   edgecolors="white", linewidths=0.6)
        if ev == "within_session":
            for d in common:
                ax.annotate(d, (xs[d], ys[d]), textcoords="offset points",
                            xytext=(4, -8), fontsize=6, color="0.35")
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.4", lw=1, ls="--")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"best {label_b} (score $-$ chance)")
    ax.set_ylabel(f"best {label_a} (score $-$ chance)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title(f"Below the diagonal = {label_b} wins that dataset", fontsize=12)
    f.tight_layout()
    return f


def train_size_crossover(tidy: pd.DataFrame, raw: pd.DataFrame, delta_fn,
                         *, ylabel: str, title: str):
    """A per-dataset delta against how many trials the fit had.

    Written for the finding it first exposed on the 14-model grid: the classical
    pipelines win within-session and lose cross-subject, which reads as an
    evaluation-mode effect until you notice the three modes differ by roughly thirty
    times in training data -- within-session gives a fit one session of one subject
    (20 trials on shin2017a) while cross-subject gives it every other subject (12 444 on
    schirrmeister2017).

    The x axis is MOABB's ``samples``, the pool the fit drew from; within-session that is
    the session total, of which about four fifths trains. Log scale, since the range is
    three decades and the expected shape of a sample-size effect is a diminishing return,
    not a line. The reported statistic is a rank correlation, which assumes neither.

    What keeps a positive result here from being a restatement of the mode label is the
    per-mode correlation printed alongside: if it survives inside within-session alone,
    the mode is a proxy for sample size rather than the cause.
    """
    from scipy.stats import spearmanr

    rows = []
    for ev in EVALS:
        sub = tidy[tidy["eval"] == ev]
        if sub.empty:
            continue
        d = delta_fn(sub)
        n = raw[raw["eval"] == ev].groupby("dataset").samples.median()
        for ds, val in d.items():
            if ds in n.index:
                rows.append(dict(ev=ev, dataset=ds, delta=val, n=n[ds]))
    r = pd.DataFrame(rows)
    rho, p = spearmanr(np.log10(r.n), r.delta)

    f, ax = plt.subplots(figsize=(9.5, 6.0))
    mk = {"within_session": "o", "cross_session": "s", "cross_subject": "^"}
    for ev in EVALS:
        h = r[r.ev == ev]
        if h.empty:
            continue
        ax.scatter(h.n, h.delta, s=62, alpha=0.9, color=EVAL_COLOR[ev], marker=mk[ev],
                   label=EVAL_LABEL[ev], edgecolors="white", linewidths=0.6)
        for _, q in h.iterrows():
            ax.annotate(q.dataset, (q.n, q.delta), textcoords="offset points",
                        xytext=(5, 4), fontsize=6, color="0.35")
    # Log-linear trend, drawn only to carry the eye: the claim is the rank correlation,
    # which does not assume this shape.
    b_, a_ = np.polyfit(np.log10(r.n), r.delta, 1)
    xs = np.logspace(np.log10(r.n.min()), np.log10(r.n.max()), 50)
    ax.plot(xs, a_ + b_ * np.log10(xs), color="0.35", lw=1.2, ls="--", zorder=0)
    ax.axhline(0, color="0.2", lw=1)
    if (r.delta > 0).any() and (r.delta < 0).any():
        lo, hi = r[r.delta > 0].n.min(), r[r.delta < 0].n.max()
        if lo < hi:
            ax.axvspan(lo, hi, color="0.85", alpha=0.45, zorder=-1)
            ax.annotate(f"both signs occur\n{int(lo)}-{int(hi)} trials",
                        (np.sqrt(lo * hi), ax.get_ylim()[0]), ha="center", va="bottom",
                        fontsize=7, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("trials available to the fit (MOABB `samples`; within-session this is "
                  "the session total, ~80% of it trains)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"{title} — Spearman $\\rho$ = {rho:+.2f} "
                 f"(p = {p:.1g}, n = {len(r)})", fontsize=12)
    f.tight_layout()
    return f, r


def campaign_agreement(tidy_a: pd.DataFrame, tidy_b: pd.DataFrame,
                       *, label_a: str, label_b: str, eval_: str = "within_session"):
    """The same cells measured by two campaigns, plotted against each other.

    This exists because of a specific doubt: the older grid was interrupted when the
    cluster went down and some cells were re-run without the resample argument fully
    threaded, so its sampling rate is not verifiable cell by cell. The v5 grid ships a
    provenance row pinning 250 Hz. Where the two campaigns ran the same model on the
    same dataset, agreement bounds how much that doubt can actually be worth.

    Only models present in both are plotted, so v5's missing classical pipelines cannot
    masquerade as disagreement. Read it with the independence caveat printed in the
    report: a fraction of the underlying rows are bit-identical between the two exports,
    so this is agreement between two overlapping campaigns, not two independent replicates.
    """
    from scipy.stats import spearmanr

    common = sorted(set(tidy_a.model) & set(tidy_b.model))
    ga = (tidy_a[(tidy_a["eval"] == eval_) & tidy_a.model.isin(common)]
          .groupby(["dataset", "model"]).above.mean())
    gb = (tidy_b[(tidy_b["eval"] == eval_) & tidy_b.model.isin(common)]
          .groupby(["dataset", "model"]).above.mean())
    i = ga.index.intersection(gb.index)
    ga, gb = ga[i], gb[i]
    rho, p = spearmanr(ga, gb)
    resid = (gb - ga).abs()

    f, axes = plt.subplots(1, 2, figsize=(12.5, 5.4),
                           gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    fam = tidy_b.drop_duplicates("model").set_index("model").family
    for mod in common:
        sel = [k for k in i if k[1] == mod]
        if not sel:
            continue
        ax.scatter([ga[k] for k in sel], [gb[k] for k in sel], s=34, alpha=0.85,
                   color=FAM_COLOR.get(fam.get(mod, "growing"), "0.4"),
                   marker=FAM_MARKER.get(fam.get(mod, "growing"), "o"))
    lo = float(min(ga.min(), gb.min())) - 0.02
    hi = float(max(ga.max(), gb.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{label_a} (score $-$ chance)")
    ax.set_ylabel(f"{label_b} (score $-$ chance)")
    ax.grid(alpha=0.3)
    ax.set_title(rf"{len(i)} shared cells, $\rho$ = {rho:+.3f} (p = {p:.1g})",
                 fontsize=11)

    ax = axes[1]
    ax.hist(resid, bins=24, color="0.55", edgecolor="white")
    ax.axvline(float(resid.median()), color="#c62828", lw=1.4,
               label=f"median {resid.median():.4f}")
    ax.axvline(float(resid.quantile(0.95)), color="#c62828", lw=1.0, ls="--",
               label=f"p95 {resid.quantile(0.95):.4f}")
    ax.set_xlabel("|difference| between campaigns, per (dataset, model)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("How far apart the two campaigns land", fontsize=11)
    f.suptitle(f"Does the resampling doubt move the numbers? — {EVAL_LABEL[eval_]}, "
               f"models common to both campaigns", fontsize=12)
    f.tight_layout()
    return f, resid


# ------------------------------------------------------- what the means are hiding
def subject_spread(tidy: pd.DataFrame, order: list[str],
                   eval_: str = "within_session"):
    """Every subject as a point, for each family's champion on each dataset.

    Every other figure in this module reduces subjects to a mean and a CI. That is the
    right summary and it hides the shape: a dataset where one family wins by 0.05 on
    average can be one where it wins on every subject, or one where it wins hugely on
    three and loses on the rest. Those license different sentences, and only the second
    is a reason to look for subject-level structure.
    """
    sub = tidy[tidy["eval"] == eval_]
    fams = families_in(sub)
    f, ax = plt.subplots(figsize=(14, 5.6))
    width = 0.78 / len(fams)
    rng = np.random.default_rng(0)
    for i, fam in enumerate(fams):
        h = sub[sub.family == fam]
        champ = (h.groupby(["dataset", "model"]).above.mean()
                 .groupby("dataset").idxmax())
        off = (i - (len(fams) - 1) / 2) * width
        for xi, d in enumerate(order):
            if d not in champ.index:
                continue
            mod = champ[d][1]
            pts = h[(h.dataset == d) & (h.model == mod)].above.to_numpy()
            if not len(pts):
                continue
            # Jitter is cosmetic; the x position carries no information beyond the
            # dataset and the family, so spreading the points is free.
            ax.scatter(xi + off + rng.uniform(-0.28, 0.28, len(pts)) * width,
                       pts, s=13, alpha=0.55, color=FAM_COLOR[fam],
                       edgecolors="none")
            ax.plot([xi + off - width * 0.4, xi + off + width * 0.4],
                    [pts.mean()] * 2, color=FAM_COLOR[fam], lw=2.0)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("score $-$ chance, one point per subject")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=FAM_COLOR[x],
                                  label=f"{x} champion") for x in fams], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"How the win is distributed over subjects — {EVAL_LABEL[eval_]}",
                 fontsize=12)
    f.tight_layout()
    return f


def subject_delta(tidy: pd.DataFrame, order: list[str], a: list[str], b: list[str],
                  *, label: str, eval_: str = "within_session"):
    """Per-subject paired delta between two groups' champions, as a distribution.

    Paired *within subject*, which is the only comparison the design supports: the same
    subject is run through both arms, so the subject's own difficulty cancels and the
    remaining spread is the arms disagreeing. The fraction above zero, printed per
    dataset, is the quantity a Wilcoxon would test -- reported here as a picture because
    12 p-values invite exactly the reading they should not get.
    """
    sub = tidy[tidy["eval"] == eval_]
    ga = sub[sub.family.isin(a)]
    gb = sub[sub.family.isin(b)]
    f, ax = plt.subplots(figsize=(13, 5.6))
    rng = np.random.default_rng(1)
    pct: list = []
    for xi, d in enumerate(order):
        ha, hb = ga[ga.dataset == d], gb[gb.dataset == d]
        if ha.empty or hb.empty:
            continue
        ca = ha.groupby("model").above.mean().idxmax()
        cb = hb.groupby("model").above.mean().idxmax()
        pa = ha[ha.model == ca].set_index("subject").above
        pb = hb[hb.model == cb].set_index("subject").above
        common = pa.index.intersection(pb.index)
        if not len(common):
            continue
        d_ = (pa[common] - pb[common]).to_numpy()
        color = "#1565c0" if d_.mean() > 0 else "#c62828"
        ax.scatter(xi + rng.uniform(-0.16, 0.16, len(d_)), d_, s=15, alpha=0.6,
                   color=color, edgecolors="none")
        ax.plot([xi - 0.3, xi + 0.3], [d_.mean()] * 2, color="0.15", lw=2.0)
        pct.append((xi, (d_ > 0).mean()))
    # Annotated after the loop, in axes coordinates: reading `get_ylim()` inside the
    # loop pins the first label to the limit as it stood before the later datasets
    # widened it, which puts one label at a different height from all the others.
    for xi, v in pct:
        ax.annotate(f"{v:.0%}", (xi, 0.985), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6.5, color="0.35")
    ax.axhline(0, color="0.3", lw=1.2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Subject by subject, not on average — {EVAL_LABEL[eval_]} "
                 "(% = subjects above zero)", fontsize=12)
    f.tight_layout()
    return f


def win_matrix(tidy: pd.DataFrame, eval_: str = "within_session"):
    """Pairwise: how often does the row model beat the column model, over subjects.

    Independent of every scale question in this module -- no chance subtraction, no
    metric mixing, because a within-(dataset, subject) comparison never leaves the pair.
    That makes it the check on the champion figures: if the champion story is an
    artefact of averaging above-chance scores, it will not survive here.
    """
    sub = tidy[tidy["eval"] == eval_]
    models = models_in(sub)
    piv = sub.pivot_table(index=["dataset", "subject"], columns="model",
                          values="score")
    n = len(models)
    m = np.full((n, n), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j or a not in piv or b not in piv:
                continue
            d = (piv[a] - piv[b]).dropna()
            if len(d):
                m[i, j] = (d > 0).mean()
    f, ax = plt.subplots(figsize=(1.0 + 0.62 * n, 0.9 + 0.55 * n))
    sns.heatmap(pd.DataFrame(m, index=models, columns=models), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0.5, vmin=0, vmax=1, ax=ax,
                annot_kws={"size": 6.5}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "P(row beats column) over (dataset, subject)"})
    ax.set_title(f"Head to head — {EVAL_LABEL[eval_]}", fontsize=12)
    ax.tick_params(labelsize=7)
    f.tight_layout()
    return f


# Studentised range / sqrt(2) at alpha = 0.05, the constant in Nemenyi's critical
# difference. Tabulated because scipy has no direct entry point for it.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def mean_rank_cd(tidy: pd.DataFrame, eval_: str = "within_session",
                 min_cov: float = 0.8):
    """Mean rank over datasets with Nemenyi's critical difference.

    The standard way this field compares many methods over many datasets, and the one
    figure here that says which differences are *not* resolvable: any two models whose
    mean ranks differ by less than the critical difference are, on this many datasets,
    indistinguishable. Ranks per dataset, so the mixed metric never enters, and the
    blocks are datasets, so the n that sets the bar is 12 -- which is why the bar is
    wide and why "SCCNet wins" needs the head-to-head figure behind it.
    """
    sub = tidy[tidy["eval"] == eval_]
    m = sub.groupby(["dataset", "model"]).above.mean().unstack()
    # A rank needs a complete block, so something has to go. Dropping models is the
    # naive move and it is the wrong one on a running campaign: the four Riemannian
    # pipelines are missing a single dataset each, and dropping them would delete the
    # classical arm from the one figure that says which gaps are resolvable. So drop
    # thin models first (under `min_cov` of the datasets -- an arm that has barely
    # started), then drop the few datasets the survivors still disagree on.
    m = m.loc[:, m.notna().mean() >= min_cov]
    m = m.dropna(axis=0, how="any")
    if m.shape[1] < 2 or m.shape[0] < 3:
        return None
    ranks = m.rank(axis=1, ascending=False)
    mean_rank = ranks.mean().sort_values()
    k, N = m.shape[1], m.shape[0]
    cd = _Q05.get(k, 3.4) * np.sqrt(k * (k + 1) / (6 * N))
    f, ax = plt.subplots(figsize=(9.5, 0.42 * k + 2.2))
    y = np.arange(len(mean_rank))
    ax.barh(y, mean_rank.values,
            color=[FAM_COLOR[family_of(x)] for x in mean_rank.index], alpha=0.85)
    ax.errorbar(mean_rank.values, y, xerr=cd / 2, fmt="none", ecolor="0.2",
                elinewidth=1.1, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(mean_rank.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean rank over {N} datasets (1 = best); "
                  f"bars are $\\pm$CD/2, CD = {cd:.2f}")
    ax.grid(axis="x", alpha=0.3)
    dropped = sorted(set(tidy.model.unique()) - set(m.columns))
    ax.set_title(f"Overlapping bars = not distinguishable on {N} datasets — "
                 f"{EVAL_LABEL[eval_]}"
                 + (f"\n({k} models; too thin to rank: {', '.join(dropped)})"
                    if dropped else ""), fontsize=11)
    f.tight_layout()
    return f


def chance_map(tidy: pd.DataFrame, order: list[str]):
    """Which (model, dataset) cells are above chance at all, per evaluation mode.

    Cells whose 95 % CI over subjects covers zero are cells where nothing was learned,
    and comparing two of them is comparing two noise draws. Drawn as its own map because
    it is the precondition for reading every other figure: a family gap on a dataset
    where both families are in the grey band is not a gap.
    """
    f, axes = plt.subplots(1, len(EVALS), figsize=(5.6 * len(EVALS), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ev in zip(axes, EVALS):
        sub = tidy[tidy["eval"] == ev]
        if sub.empty:
            ax.axis("off"); continue
        c = _cell(sub, ["dataset", "model"])
        c["lo"] = c["mean"] - c["ci"]
        piv = c.pivot(index="model", columns="dataset", values="lo")
        piv = piv.reindex(index=[m for m in models_in(tidy) if m in piv.index],
                          columns=[d for d in order if d in piv.columns])
        sns.heatmap(piv, cmap="RdYlGn", center=0, ax=ax, annot=True, fmt=".2f",
                    annot_kws={"size": 5.5}, linewidths=0.4, linecolor="white",
                    cbar=False)
        ax.set_title(EVAL_LABEL[ev], fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(labelsize=6.5)
    f.suptitle("Lower bound of the 95 % CI over subjects — red = not above chance",
               fontsize=12)
    f.tight_layout()
    return f


def coverage_map(scores: pd.DataFrame, order: list[str]):
    """How many seeds each (model, dataset, mode) cell actually has.

    A campaign that is still running has an uneven grid, and an uneven grid biases every
    model-averaged figure toward whichever arms finished. This is the figure that makes
    that visible instead of leaving it in a caption -- and it is the first one to look at
    when a number in this report changes between builds.
    """
    f, axes = plt.subplots(1, len(EVALS), figsize=(5.6 * len(EVALS), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    mx = int(scores.seed.nunique())
    for ax, ev in zip(axes, EVALS):
        sub = scores[scores["eval"] == ev]
        if sub.empty:
            ax.axis("off"); continue
        piv = sub.pivot_table(index="model", columns="dataset", values="seed",
                              aggfunc="nunique")
        piv = piv.reindex(index=[m for m in order_models(scores.model.unique())
                                 if m in piv.index],
                          columns=[d for d in order if d in piv.columns])
        sns.heatmap(piv, cmap="YlGnBu", vmin=0, vmax=mx, ax=ax, annot=True, fmt=".0f",
                    annot_kws={"size": 6.5}, linewidths=0.4, linecolor="white",
                    cbar=False)
        ax.set_title(EVAL_LABEL[ev], fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(labelsize=6.5)
    f.suptitle(f"Seeds per cell (blank = not run yet, full = {mx})", fontsize=12)
    f.tight_layout()
    return f


def seed_noise(scores: pd.DataFrame, order: list[str],
               eval_: str = "within_session"):
    """Seed-to-seed spread, which is the floor every delta in this report sits on.

    Computed within a (dataset, model, subject, session) cell, so it isolates the one
    thing the seed controls -- initialisation, batch order, and the within-session split
    -- from every other source of variation. A family gap smaller than this is not a
    gap, and the classical pipelines sitting at exactly zero here is not a bug: none of
    them takes a random_state.
    """
    sub = scores[scores["eval"] == eval_]
    sd = (sub.groupby(["dataset", "model", "subject", "session"]).score
          .agg(["std", "size"]).reset_index())
    sd = sd[sd["size"] > 1]
    if sd.empty:
        return None
    models = [m for m in order_models(sub.model.unique()) if m in set(sd.model)]
    f, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(order))
    w = 0.8 / max(len(models), 1)
    med = sd.groupby(["dataset", "model"])["std"].median()
    for i, mod in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w,
               [med.get((d, mod), np.nan) for d in order], w,
               color=FAM_COLOR[family_of(mod)], alpha=0.85, label=mod)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("median seed-to-seed sd within a (subject, session)")
    ax.legend(fontsize=6.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"The noise floor — {EVAL_LABEL[eval_]}", fontsize=12)
    f.tight_layout()
    return f


def per_model_train_size(tidy: pd.DataFrame, raw: pd.DataFrame):
    """Every model's above-chance score against the trials its fits had.

    The per-model version of `train_size_crossover`, which reduces each mode to one
    number per family. Here each line is one model over all 30 (dataset, mode) points,
    so the question becomes whether the sample-size dependence is a family property or
    an architecture property -- steeper lines need more data to be worth using, and that
    is a statement about a specific network, not about deep learning.
    """
    n = raw.groupby(["eval", "dataset"]).samples.median()
    m = tidy.groupby(["eval", "dataset", "model"]).above.mean()
    f, ax = plt.subplots(figsize=(10.5, 6.2))
    for mod in models_in(tidy):
        pts = [(n[k[:2]], v) for k, v in m.items()
               if k[2] == mod and k[:2] in n.index]
        if len(pts) < 3:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=FAM_MARKER[family_of(mod)], ms=4, lw=1.0, alpha=0.8,
                color=FAM_COLOR[family_of(mod)], label=mod)
    ax.set_xscale("log")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xlabel("trials available to the fit (MOABB `samples`, log)")
    ax.set_ylabel("score $-$ chance")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Every model against the amount of data it was given", fontsize=12)
    f.tight_layout()
    return f


def cost_vs_score(scores: pd.DataFrame, tidy: pd.DataFrame,
                  eval_: str = "within_session"):
    """Wall-clock seconds per fold against what the fold scored.

    The axis a supervisor asks about after the accuracy one. It also separates the two
    families on a dimension that has nothing to do with EEG: the classical pipelines are
    seconds, the deep arms are minutes to hours, and a 0.02 advantage bought with three
    orders of magnitude of compute is a different result from the same advantage bought
    for free.
    """
    t = (scores[scores["eval"] == eval_]
         .groupby(["dataset", "model"]).time.median())
    a = tidy[tidy["eval"] == eval_].groupby(["dataset", "model"]).above.mean()
    i = t.index.intersection(a.index)
    if not len(i):
        return None
    f, ax = plt.subplots(figsize=(9.8, 6.2))
    for mod in models_in(tidy):
        sel = sorted([k for k in i if k[1] == mod], key=lambda k: t[k])
        if not sel:
            continue
        ax.plot([t[k] for k in sel], [a[k] for k in sel],
                marker=FAM_MARKER[family_of(mod)], ms=4.5, lw=0.8, alpha=0.85,
                color=FAM_COLOR[family_of(mod)], label=mod)
    ax.set_xscale("log")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xlabel("median seconds per fold (log)")
    ax.set_ylabel("score $-$ chance")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"What the score cost — {EVAL_LABEL[eval_]}", fontsize=12)
    f.tight_layout()
    return f
