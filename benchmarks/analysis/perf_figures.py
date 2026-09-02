"""Where growth wins, where it loses, and how much of either is growth at all.

The companion to `growth_dynamics`, which answers *how* the mechanism behaves. This
answers *whether it pays* -- and the two are separate reports because they are separate
exports on separate cadences, not because the questions are unrelated.

THE FIGURE THIS MODULE EXISTS FOR is :func:`decomposition`. Everything else supports,
qualifies or audits it.

``grow - bd`` is the number a reader wants and the number that cannot be reported
alone. The growing arm and the braindecode arm differ in two ways at once: growth is
on, *and* the class is eegrow's re-implementation rather than braindecode's. The fixed
control ``fix_*`` -- the same eegrow class, same file, same init path, growth disabled
-- splits them::

    grow - bd  =  (grow - fix)  +  (fix - bd)
                   growth          codebase

On ``bnci2014_001/within_session`` those two terms are +0.005 and +0.019: three
quarters of the headline is the re-implementation, and the fixed control is the only
instrument that says so. This is why the control is not redundant with braindecode's
own arm even though the two are width-matched to the parameter.

WHAT THE SUPPORTING FIGURES ARE FOR

*The chance audit.* A delta measured over a baseline that never left chance is
arithmetic, not a result. :func:`chance_map` and the hatching on
:func:`decomposition_by_dataset` mark those cells so they cannot be read as wins.

*Power.* :func:`power` draws each contrast's minimum detectable effect beside its
interval. A contrast whose MDE is wider than the effect under discussion has not
measured a null; it has measured nothing, and the two get written up identically if
nobody plots the MDE.

*Cost.* :func:`pareto` puts accuracy against parameter count. A growing arm that
matches a fixed one at a fifth of the size is a different claim from one that matches
it -- and it is the claim growth is actually for.

THE UNIT OF ANALYSIS IS THE SUBJECT throughout; see `perf_io.by_subject`. Every
interval here is a bootstrap over subjects, every sign test counts subjects, and no
figure in this module consumes a row-level frame.

All functions take frames from `perf_io` and return a matplotlib ``Figure``, or None
when there is nothing to draw. Nothing here writes a file; the driver does.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from scipy import stats

import perf_io
from perf_io import EVAL_ORDER, MODEL_ORDER, TRIPLES

# --------------------------------------------------------------------------- style
# Same palette as `growth_dynamics`, so a reader moving between the two reports reads
# the same colour as the same arm.
FAM_COLOR = {"braindecode": "#1565c0", "fixed control": "#00695c",
             "growing": "#c62828", "other": "#616161"}
MODEL_COLOR = {
    "bd_shallow": "#0d47a1", "bd_deep4": "#1e88e5", "bd_sccnet": "#7cb9f2",
    "bd_eegnex": "#b3d4f5",
    "fix_shallow": "#00382f", "fix_deepeeg": "#00897b", "fix_sccnet": "#4db6ac",
    "fix_eegnex": "#a7d9d3",
    "grow_shallow": "#8e1616", "grow_deep": "#e53935", "grow_sccnet": "#f4a0a0",
    "grow_eegnex": "#f8cccc",
}
ARCH_MARKER = {"shallow": "^", "deep": "s", "sccnet": "o", "eegnex": "D",
               "other": "."}
EVAL_LABEL = {"within_session": "within-session", "cross_session": "cross-session",
              "cross_subject": "cross-subject (LOSO)"}
ALIGN_LABEL = {"none": "raw", "easubject": "Euclidean alignment"}

#: The two terms of the decomposition, and the total. Colours are deliberately not the
#: arm colours: these are *contrasts*, not arms, and a reader who reads the growth term
#: in the growing arm's red will read it as the growing arm's score.
TERM_COLOR = {"total": "#37474f", "growth": "#a51f18", "codebase": "#00695c"}
TERM_LABEL = {"total": "grow − bd  (the headline)",
              "growth": "grow − fix  (growth)",
              "codebase": "fix − bd  (codebase)"}

DATASET_ORDER = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004",
                 "bnci2015_001", "cho2017", "lee2019_mi", "physionetmi",
                 "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016"]


def _present(frame: pd.DataFrame, col: str, order: list[str]) -> list[str]:
    have = set(frame[col].dropna().unique())
    return [x for x in order if x in have] + sorted(have - set(order))


def _finish(fig, title: str, subtitle: str = "") -> plt.Figure:
    """Title block above the axes, reserving *inches* rather than a figure fraction.

    A fraction reserves a band that shrinks with the figure, so the same rect that
    clears the title on a 7-inch panel puts it through the subtitle on a 4-inch one.
    These figures range from 4 to 9 inches tall, so the header is sized in inches and
    converted, and the subtitle is wrapped to the figure's own width.
    """
    h, w = fig.get_figheight(), fig.get_figwidth()
    sub = textwrap.fill(subtitle, width=int(w * 15)) if subtitle else ""
    lines = sub.count("\n") + 1 if sub else 0
    head = 0.34 + 0.17 * lines
    fig.tight_layout(rect=(0, 0, 1, max(1 - head / h, 0.5)))
    fig.text(0.008, 1 - 0.06 / h, title, fontsize=13, fontweight="bold",
             ha="left", va="top")
    if sub:
        fig.text(0.008, 1 - 0.30 / h, sub, fontsize=9, color="#546e7a",
                 ha="left", va="top", linespacing=1.35)
    return fig


# =========================================================================== 0. grid

def coverage(sc: pd.DataFrame) -> plt.Figure:
    """What has actually been scored, per protocol -- including what never will be.

    Cells are datasets scored, not folds: a cell that ran on 12 datasets and one that
    ran on 1 are both "done" by any completion count, and only one of them supports a
    cross-dataset claim.

    The three empty columns under cross-subject are the finding. The ``fix_*`` arms were
    never scheduled on LOSO -- zero datasets, zero claims, zero rows in the grid plan --
    so on the protocol that matters most for a deployed decoder, ``grow − bd`` can never
    be split into its growth and codebase terms. That is a gap in the design, not in the
    run, and no amount of waiting closes it.
    """
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(13, 4.4), sharey=True)
    models = _present(sc, "model", MODEL_ORDER)
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        sub = sc[sc["eval"] == ev]
        for i, m in enumerate(models):
            for j, tag in enumerate(["none", "easubject"]):
                cell = sub[(sub.model == m) & (sub.align_tag == tag)]
                n = cell.dataset.nunique()
                ax.barh(i + (0.18 if j else -0.18), n, height=0.34,
                        color=MODEL_COLOR.get(m, "#616161"),
                        alpha=1.0 if j == 0 else 0.45,
                        edgecolor="white", linewidth=0.5)
                if n:
                    ax.text(n + 0.15, i + (0.18 if j else -0.18), str(n),
                            va="center", fontsize=7, color="#37474f")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8, family="monospace")
        ax.set_xlim(0, 13)
        ax.set_title(EVAL_LABEL[ev], fontsize=10)
        ax.set_xlabel("datasets scored")
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
        if sub.empty:
            continue
        missing = [m for m in models if sub[sub.model == m].empty]
        if missing:
            ax.text(0.5, 0.02, f"{len(missing)} arm(s) absent by design",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color="#a51f18", style="italic")
    np.atleast_1d(axes)[0].legend(
        handles=[Patch(facecolor="#616161", label="raw"),
                 Patch(facecolor="#616161", alpha=0.45, label="EA")],
        fontsize=7, loc="lower right", framealpha=0.9)
    return _finish(fig, "What the campaign measured",
                   "datasets with at least one scored fold, per arm, protocol and "
                   "alignment")


# ================================================================ 1. THE decomposition

def decomposition(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """The headline split into growth and codebase, per architecture and protocol.

    Read the black interval first -- that is ``grow − bd``, the number a paper would
    report. Then read the two coloured ones under it: they sum to it, and they say how
    much of it is the growth mechanism versus how much is eegrow's class differing from
    braindecode's at initialisation.

    An interval crossing zero is a contrast this grid cannot resolve; the MDE annotation
    on :func:`power` says how large an effect would have had to be to show up. Where the
    codebase bar is missing, the fixed control was never run and the headline is
    **not decomposable** -- the figure says so rather than drawing the total alone.
    """
    archs = [a for a in TRIPLES if a in set(subj.arch)]
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(13.5, 4.6), sharex=True)
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        rows, y = [], 0.0
        ticks, labels = [], []
        for arch in archs:
            d = perf_io.decompose(subj, arch, ev, align_tag)
            if d is None:
                continue
            terms = ["total", "growth", "codebase"] if d["has_control"] else ["total"]
            base = y
            for term in terms:
                r = d[term]
                ax.errorbar(r["mean"], y,
                            xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                            fmt=ARCH_MARKER.get(arch, "o"), ms=6, capsize=3,
                            color=TERM_COLOR[term], lw=1.6,
                            mfc=TERM_COLOR[term] if term == "total" else "white",
                            mew=1.6, zorder=3)
                ax.text(r["hi"] + 0.004, y, f"{r['n_win']}/{r['n']}", va="center",
                        fontsize=6.5, color=TERM_COLOR[term])
                y -= 0.75
            if not d["has_control"]:
                ax.text(0.02, y + 0.4, "not decomposable — no fixed control",
                        transform=ax.get_yaxis_transform(), fontsize=7,
                        color="#a51f18", style="italic", va="center")
                y -= 0.75
            ticks.append((base + y + 0.75) / 2)
            labels.append(arch)
            y -= 0.6
            rows.append(d)
        if not rows:
            ax.text(0.5, 0.5, "nothing scored", transform=ax.transAxes, ha="center",
                    color="#90a4ae")
            ax.set_axis_off()
            continue
        ax.axvline(0, color="#455a64", lw=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9, family="monospace")
        # Half a row of slack top and bottom: the "not decomposable" note sits on its
        # own line and would otherwise be drawn onto the axis frame.
        ax.set_ylim(y + 0.4, 0.6)
        ax.set_title(EVAL_LABEL[ev], fontsize=10)
        ax.set_xlabel("Δ score (subject-level mean, 95 % bootstrap)")
        ax.grid(axis="x", alpha=0.25)
    np.atleast_1d(axes)[0].legend(
        handles=[Line2D([], [], color=TERM_COLOR[t], marker="o", ls="-",
                        mfc=TERM_COLOR[t] if t == "total" else "white",
                        label=TERM_LABEL[t]) for t in TERM_COLOR],
        fontsize=7.5, loc="lower left", framealpha=0.92)
    return _finish(
        fig, "grow − bd = (grow − fix) + (fix − bd)",
        f"{ALIGN_LABEL.get(align_tag, align_tag)} · n = subjects · "
        "the fraction beside each interval is subjects on which the term is positive")


def decomposition_by_dataset(subj: pd.DataFrame, arch: str, evaluation: str,
                             align_tag: str = "none") -> plt.Figure | None:
    """The same three terms, one row per dataset: is the average a summary or a mixture?

    A pooled mean is only a description if the datasets agree. Where the growth term
    changes sign across datasets, "growth helps by +0.005" is a statement about no
    dataset in particular.

    A dataset whose *baseline arm never beat chance* is hatched. A delta over a baseline
    at chance is arithmetic on noise, and the v5 audit found the single largest reported
    growth gain was exactly that. Hatched rows are excluded from the pooled interval
    printed in the corner.
    """
    bd, fix, grow = TRIPLES[arch]
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    pairs = {"total": (grow, bd), "growth": (grow, fix), "codebase": (fix, bd)}
    got = {k: perf_io.paired(sel, a, b) for k, (a, b) in pairs.items()}
    got = {k: v for k, v in got.items() if not v.empty}
    if "total" not in got:
        return None
    datasets = [d for d in DATASET_ORDER if d in set(got["total"].dataset)]
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(datasets) + 2.4))
    suspect = set()
    for i, ds in enumerate(datasets):
        for k, off in (("total", 0.24), ("growth", 0.0), ("codebase", -0.24)):
            if k not in got:
                continue
            g = got[k][got[k].dataset == ds]
            if g.empty:
                continue
            r = perf_io.test(g.delta.to_numpy())
            # "The reference this delta is measured against never learned." Judged on
            # the control arm of the pair, per subject, and only flagged when it fails
            # on the majority of them -- one unlucky subject is not a broken baseline.
            bad = float(np.mean(g.b_above < 0.5)) > 0.5
            if bad:
                suspect.add(ds)
            ax.errorbar(r["mean"], i + off,
                        xerr=[[max(r["mean"] - r["lo"], 0)],
                              [max(r["hi"] - r["mean"], 0)]],
                        fmt=ARCH_MARKER.get(arch, "o"), ms=5, capsize=2.5, lw=1.4,
                        color=TERM_COLOR[k], alpha=0.35 if bad else 1.0,
                        mfc=TERM_COLOR[k] if k == "total" else "white", mew=1.4)
        if ds in suspect:
            ax.axhspan(i - 0.45, i + 0.45, facecolor="#a51f18", alpha=0.07, zorder=0,
                       hatch="///", edgecolor="#a51f18", lw=0)
    ax.axvline(0, color="#455a64", lw=1)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Δ score (subject-level mean, 95 % bootstrap)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[Line2D([], [], color=TERM_COLOR[t], marker="o", ls="-",
                              mfc=TERM_COLOR[t] if t == "total" else "white",
                              label=TERM_LABEL[t]) for t in TERM_COLOR if t in got],
              fontsize=7.5, loc="best", framealpha=0.92)
    clean = got["total"][~got["total"].dataset.isin(suspect)]
    if len(clean) > 2:
        r = perf_io.test(clean.delta.to_numpy())
        ax.text(0.99, 0.01,
                f"pooled over sound datasets: {r['mean']:+.4f} "
                f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  n={r['n']} subjects",
                transform=ax.transAxes, ha="right", fontsize=7.5, color="#37474f")
    note = (f" · {len(suspect)} dataset(s) hatched: the reference arm did not beat "
            "chance") if suspect else ""
    return _finish(fig, f"{arch} — the decomposition, dataset by dataset",
                   f"{EVAL_LABEL[evaluation]} · "
                   f"{ALIGN_LABEL.get(align_tag, align_tag)}{note}")


def subject_delta(subj: pd.DataFrame, arch: str, evaluation: str,
                  align_tag: str = "none") -> plt.Figure | None:
    """Every subject's growth term, sorted -- the spread the mean hides.

    A mean of +0.005 is compatible with "growth helps every subject a little" and with
    "growth helps four subjects a lot and hurts five", and those are different findings
    with different next experiments. Each point is one subject on one dataset, so the
    figure also shows whether the sign is a property of subjects or of datasets.
    """
    bd, fix, grow = TRIPLES[arch]
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    g = perf_io.paired(sel, grow, fix)
    label = TERM_LABEL["growth"]
    if g.empty:                      # no fixed control: fall back to the headline
        g = perf_io.paired(sel, grow, bd)
        label = TERM_LABEL["total"]
    if g.empty:
        return None
    g = g.sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    colors = [MODEL_COLOR.get(grow) if v > 0 else "#546e7a" for v in g.delta]
    ax.bar(range(len(g)), g.delta, color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="#263238", lw=1)
    r = perf_io.test(g.delta.to_numpy())
    ax.axhline(r["mean"], color="#a51f18", lw=1.2, ls="--",
               label=f"mean {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    ax.axhspan(r["lo"], r["hi"], color="#a51f18", alpha=0.10, lw=0)
    ax.set_xlim(-0.5, len(g) - 0.5)
    ax.set_xlabel(f"{len(g)} subject × dataset units, sorted")
    ax.set_ylabel("Δ score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.text(0.99, 0.03, f"positive on {r['n_win']}/{r['n']} "
                        f"({r['win']:.0%}) · Wilcoxon p = {r['p']:.2g}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#37474f")
    return _finish(fig, f"{arch} — {label}, subject by subject",
                   f"{EVAL_LABEL[evaluation]} · {ALIGN_LABEL.get(align_tag, align_tag)}")


def _panel_grid(n: int, ncols: int = 3, w: float = 4.6,
                h: float = 3.4) -> tuple[plt.Figure, list[plt.Axes]]:
    """A dataset-per-panel grid, with the unused cells switched off, not left blank.

    An empty axes still draws its spines, and a reader counts twelve frames and asks
    which two datasets are missing. There are none: the grid simply did not divide.
    """
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_axis_off()
    return fig, list(axes[:n])


def _label_edges(axes: list[plt.Axes], xlabel: str, ylabel: str,
                 ncols: int = 3) -> None:
    """Axis labels on the left column and on the *last used* panel of each column.

    ``axes[-ncols:]`` is the wrong bottom row whenever the last row is partial: it
    labels whichever panels happen to end the list and leaves a column's real bottom
    panel unlabelled.
    """
    for i, ax in enumerate(axes):
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if i + ncols >= len(axes):
            ax.set_xlabel(xlabel, fontsize=8)


def dumbbell(subj: pd.DataFrame, arch: str, evaluation: str,
             align_tag: str = "none") -> plt.Figure | None:
    """Both arms' absolute score on every subject, joined by the delta between them.

    The figure Sylvain drew on paper at the 01/09 review, and the one the aggregate
    cannot replace. ``subject_delta`` plots the same contrast, but only its *difference*
    -- and a difference has no scale of its own: +0.02 on a subject at 0.55 and +0.02 on
    a subject at 0.92 are the same bar there and are not the same result. Here the
    triangle is braindecode, the circle the growing arm, and both sit on the score axis
    beside their dataset's chance line, so a row where both markers are on that line is
    visibly a row whose delta means nothing.

    Subjects are sorted by the braindecode arm, which makes the second question legible
    without a second figure: whether the segments lengthen, shorten or flip sign along
    the axis -- whether growth pays where the subject is hard, where it is easy, or
    indifferently.

    The pair drawn is ``grow − bd``, the headline, and the headline is the one number
    this report exists to say cannot be read alone: it adds the growth term to the
    codebase term, and :func:`decomposition` finds the second larger than the first
    almost everywhere. A red segment here says the eegrow growing arm ended above
    braindecode's on that subject; it does not say growth is what put it there.
    """
    bd, _fix, grow = TRIPLES[arch]
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    g = perf_io.paired(sel, grow, bd)
    if g.empty:
        return None
    chance = sel.groupby("dataset").chance.first()
    datasets = _present(g, "dataset", DATASET_ORDER)
    fig, axes = _panel_grid(len(datasets))
    for ax, ds in zip(axes, datasets):
        d = g[g.dataset == ds].sort_values("b_score").reset_index(drop=True)
        y = np.arange(len(d))
        # Marker and line weight follow the subject count: the same 6-point marker that
        # reads as a dumbbell on zhou2016 (4 subjects) is a solid block on physionetmi
        # (109), where the panel has to read as a band instead.
        ms = float(np.clip(48 / max(len(d), 1), 2.2, 7.0))
        lw = float(np.clip(60 / max(len(d), 1), 0.5, 1.6))
        for yi, row in zip(y, d.itertuples()):
            ax.plot([row.b_score, row.a_score], [yi, yi], lw=lw,
                    color=MODEL_COLOR[grow] if row.delta > 0 else "#78909c",
                    alpha=0.85, zorder=1, solid_capstyle="round")
        ax.plot(d.b_score, y, ARCH_MARKER.get(arch, "^"), ms=ms,
                color=MODEL_COLOR[bd], lw=0, zorder=2, label=bd)
        ax.plot(d.a_score, y, "o", ms=ms, color=MODEL_COLOR[grow], lw=0, zorder=3,
                label=grow)
        ch = float(chance.get(ds, np.nan))
        if np.isfinite(ch):
            ax.axvline(ch, color="#263238", lw=1.0, ls=":", zorder=0)
        r = perf_io.test(d.delta.to_numpy())
        ax.set_title(f"{ds}   n={len(d)}", fontsize=9, family="monospace")
        ax.text(0.02, 0.98, f"Δ {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]\n"
                            f"grow ahead on {r['n_win']}/{r['n']}",
                transform=ax.transAxes, va="top", fontsize=7.2, color="#37474f",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        ax.set_ylim(-1, len(d))
        ax.set_yticks([])
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", alpha=0.22)
    _label_edges(axes, "score", "subjects, sorted by braindecode")
    axes[0].legend(fontsize=7, loc="best", framealpha=0.92, numpoints=1)
    return _finish(fig, f"{arch} — every subject, both arms, on the score axis",
                   f"{EVAL_LABEL[evaluation]} · {ALIGN_LABEL.get(align_tag, align_tag)}"
                   f" · triangle {bd}, circle {grow}, segment red where growth is "
                   "ahead · dotted line is chance")


def power(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """Effect against minimum detectable effect: which nulls are nulls.

    A contrast whose interval crosses zero has two possible readings -- there is no
    effect, or there was not enough grid to see one -- and only the MDE separates them.
    The grey bar is the smallest true effect this many subjects would detect at 80 %
    power; a contrast whose |effect| sits well inside its own MDE has *not* measured a
    null, and writing it up as one is the error this figure exists to prevent.
    """
    rows = []
    for ev in EVAL_ORDER:
        for arch in TRIPLES:
            d = perf_io.decompose(subj, arch, ev, align_tag)
            if d is None:
                continue
            for term in ("total", "growth", "codebase"):
                if term in d:
                    rows.append({"eval": ev, "arch": arch, "term": term, **d[term]})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 0.34 * len(r) + 2.0))
    ypos = np.arange(len(r))[::-1]
    ax.barh(ypos, r["mde"], height=0.72, color="#cfd8dc", zorder=1,
            label="MDE at 80 % power (either sign)")
    ax.barh(ypos, -r["mde"], height=0.72, color="#cfd8dc", zorder=1)
    for y, row in zip(ypos, r.itertuples()):
        ax.errorbar(row.mean, y, xerr=[[max(row.mean - row.lo, 0)],
                                       [max(row.hi - row.mean, 0)]],
                    fmt="o", ms=5, capsize=2.5, lw=1.5, zorder=3,
                    color=TERM_COLOR[row.term])
    ax.axvline(0, color="#455a64", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{x.eval.replace('_', '-'):14s} {x.arch:8s} {x.term}"
                        for x in r.itertuples()], fontsize=7.5, family="monospace")
    ax.set_xlabel("Δ score")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    underpowered = int(((r["mean"].abs() < r["mde"]) & (r["lo"] * r["hi"] < 0)).sum())
    return _finish(fig, "Is a null a null, or an empty measurement?",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · {underpowered} of "
                   f"{len(r)} contrasts have an effect smaller than their own MDE and "
                   "an interval crossing zero")


# ================================================================== 2. absolute level

def per_dataset_levels(subj: pd.DataFrame, evaluation: str,
                       align_tag: str = "none") -> plt.Figure | None:
    """Absolute score of every arm on every dataset, against that dataset's chance line.

    Deltas are unreadable without levels. +0.03 over a decoder at 0.55 and +0.03 over
    one at 0.85 are not the same result, and a delta over one at chance is not a result.
    The dashed line is chance for the dataset's own paradigm -- 1/n_classes for the
    accuracy datasets, 0.5 for the AUC ones -- so the reader can see which arms cleared
    it before reading any comparison between them.
    """
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    if sel.empty:
        return None
    datasets = [d for d in DATASET_ORDER if d in set(sel.dataset)]
    models = _present(sel, "model", MODEL_ORDER)
    ncol = 4
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow),
                             squeeze=False)
    for ax, ds in zip(axes.ravel(), datasets):
        d = sel[sel.dataset == ds]
        present = [m for m in models if m in set(d.model)]
        for i, m in enumerate(present):
            v = d[d.model == m].score.to_numpy()
            lo, hi = perf_io.boot_ci(v)
            ax.errorbar(i, v.mean(), yerr=[[max(v.mean() - lo, 0)],
                                           [max(hi - v.mean(), 0)]],
                        fmt=ARCH_MARKER.get(perf_io.arch_of(m), "o"), ms=5, capsize=2,
                        lw=1.4, color=MODEL_COLOR.get(m, "#616161"))
        ch = float(d.chance.iloc[0])
        ax.axhline(ch, color="#a51f18", ls="--", lw=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([m.replace("_", "\n") for m in present], fontsize=6,
                           family="monospace")
        ax.set_title(f"{ds}  ({d.metric.iloc[0]}, n={d.subject.nunique()})",
                     fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(x=0.08)
    for ax in axes.ravel()[len(datasets):]:
        ax.set_axis_off()
    return _finish(fig, "Absolute level, and how far it is above chance",
                   f"{EVAL_LABEL[evaluation]} · {ALIGN_LABEL.get(align_tag, align_tag)}"
                   " · subject mean ± 95 % bootstrap · dashed line = chance")


def chance_map(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """Fraction of subjects on which each arm beat chance, per dataset and protocol.

    The audit that has to run before any delta is read. A cell near zero is a cell where
    the arm learned nothing on most subjects, and every comparison involving it is a
    comparison of two noise draws. The v5 grid's largest reported growth gain came from
    such a cell, which is why this is a headline figure and not an appendix.
    """
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(15, 5.0), sharey=True)
    models = _present(subj, "model", MODEL_ORDER)
    datasets = [d for d in DATASET_ORDER if d in set(subj.dataset)]
    im = None
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        sel = subj[(subj["eval"] == ev) & (subj.align_tag == align_tag)]
        grid = np.full((len(datasets), len(models)), np.nan)
        for i, ds in enumerate(datasets):
            for j, m in enumerate(models):
                c = sel[(sel.dataset == ds) & (sel.model == m)]
                if len(c):
                    grid[i, j] = c.above_chance.mean()
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for i in range(len(datasets)):
            for j in range(len(models)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                            fontsize=6,
                            color="#212121" if grid[i, j] > 0.25 else "white")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=90, fontsize=7, family="monospace")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=7.5, family="monospace")
        ax.set_title(EVAL_LABEL[ev], fontsize=10)
    # Order matters and is the bug this comment exists to stop coming back: a
    # `subplots_adjust` AFTER `fig.colorbar(ax=[...])` overwrites the positions the
    # colorbar just shrank the axes into, and the bar is redrawn on top of the last
    # panel. Reserve the header first, attach the colorbar last, never the reverse.
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 0.62 / h, bottom=0.16)
    if im is not None:
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.7, pad=0.02,
                     label="subjects above chance (α = 0.05, one-sided exact)")
    fig.text(0.008, 1 - 0.06 / h, "Did this arm learn anything here?", fontsize=13,
             fontweight="bold", ha="left", va="top")
    fig.text(0.008, 1 - 0.30 / h,
             f"{ALIGN_LABEL.get(align_tag, align_tag)} · a delta measured against a red "
             "cell is arithmetic on noise", fontsize=9, color="#546e7a", va="top")
    return fig


# ==================================================================== 3. head to head

def win_matrix(subj: pd.DataFrame, evaluation: str,
               align_tag: str = "none") -> plt.Figure | None:
    """Row beats column on what fraction of the subjects both scored.

    Paired and per-cell: each entry is computed on the units the two arms share, so an
    arm that ran on fewer datasets is not penalised for the datasets it is missing. The
    diagonal is blank. This is the figure that answers "who wins" without committing to
    a single scalar ranking -- and the asymmetries in it are why a single scalar is a
    poor summary.
    """
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    models = _present(sel, "model", MODEL_ORDER)
    if len(models) < 2:
        return None
    grid = np.full((len(models), len(models)), np.nan)
    counts = np.zeros_like(grid)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                continue
            d = perf_io.paired(sel, a, b)
            if len(d) >= 5:
                grid[i, j] = float((d.delta > 0).mean())
                counts[i, j] = len(d)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
    for i in range(len(models)):
        for j in range(len(models)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0%}\n{int(counts[i, j])}",
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(grid[i, j] - 0.5) > 0.28 else "#212121")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90, fontsize=8, family="monospace")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8, family="monospace")
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of shared subjects row > column")
    return _finish(fig, "Head to head, paired on shared subjects",
                   f"{EVAL_LABEL[evaluation]} · {ALIGN_LABEL.get(align_tag, align_tag)}"
                   " · second line is the number of subjects compared")


def mean_rank(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """Mean rank across subjects, per protocol, with the critical difference.

    Ranks make datasets with different metrics and different difficulties commensurate,
    which averaging raw scores does not. The bar is Nemenyi's critical difference at
    α = 0.05: arms whose intervals overlap are not separated by this grid. Computed only
    on the complete square -- subjects every arm scored -- because a rank over a varying
    set of competitors is not a rank.
    """
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(13.5, 4.0))
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        sel = subj[(subj["eval"] == ev) & (subj.align_tag == align_tag)]
        if sel.empty:
            ax.set_axis_off()
            continue
        wide = sel.pivot_table(index=["dataset", "subject"], columns="model",
                               values="score").dropna()
        if wide.empty or wide.shape[1] < 2:
            ax.text(0.5, 0.5, "no complete square", transform=ax.transAxes,
                    ha="center", color="#90a4ae")
            ax.set_axis_off()
            continue
        ranks = wide.rank(axis=1, ascending=False)
        k, n = wide.shape[1], len(wide)
        mean = ranks.mean().sort_values()
        # Nemenyi q at alpha=0.05 for k = 2..12 (studentised range / sqrt2), the table
        # every CD diagram uses; indexed by k so the constant is never silently reused
        # at the wrong arm count.
        q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
        cd = q05.get(k, 3.3) * np.sqrt(k * (k + 1) / (6.0 * n))
        ax.barh(range(len(mean)), mean.to_numpy(),
                color=[MODEL_COLOR.get(m, "#616161") for m in mean.index], height=0.65)
        ax.errorbar(mean.to_numpy(), range(len(mean)), xerr=cd / 2, fmt="none",
                    ecolor="#37474f", capsize=3, lw=1.2)
        ax.set_yticks(range(len(mean)))
        ax.set_yticklabels(mean.index, fontsize=8, family="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("mean rank (1 = best)")
        ax.set_title(f"{EVAL_LABEL[ev]}  (n={n} subjects, k={k})", fontsize=9.5)
        ax.grid(axis="x", alpha=0.25)
        ax.text(0.98, 0.03, f"CD = {cd:.2f}", transform=ax.transAxes, ha="right",
                fontsize=8, color="#37474f")
    return _finish(fig, "Mean rank on the complete square",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · error bar is half the "
                   "Nemenyi critical difference: overlapping arms are unseparated")


def champion_share(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """How often each arm is the best on a subject -- and how thin the win is.

    A mean-rank table hands the reader one winner. This says whether that winner is a
    consensus or a plurality: an arm that is best on 22 % of subjects in a 9-arm field
    is genuinely ahead, and one that is best on 12 % is a coin-flip away from any other.
    The runner-up gap on the right says how much is at stake in each of those wins.
    """
    fig, axes = plt.subplots(2, len(EVAL_ORDER), figsize=(13.5, 6.4),
                             gridspec_kw={"height_ratios": [1.4, 1]})
    axes = np.atleast_2d(axes)
    for col, ev in enumerate(EVAL_ORDER):
        top, bot = axes[0, col], axes[1, col]
        sel = subj[(subj["eval"] == ev) & (subj.align_tag == align_tag)]
        wide = (sel.pivot_table(index=["dataset", "subject"], columns="model",
                                values="score").dropna() if not sel.empty
                else pd.DataFrame())
        if wide.empty or wide.shape[1] < 2:
            top.set_axis_off()
            bot.set_axis_off()
            continue
        best = wide.idxmax(axis=1)
        share = best.value_counts(normalize=True)
        order = [m for m in MODEL_ORDER if m in share.index]
        top.bar(range(len(order)), [share[m] for m in order],
                color=[MODEL_COLOR.get(m, "#616161") for m in order])
        top.axhline(1.0 / wide.shape[1], color="#455a64", ls=":", lw=1.2)
        top.set_xticks(range(len(order)))
        top.set_xticklabels(order, rotation=90, fontsize=7, family="monospace")
        top.set_ylabel("share of subjects won" if col == 0 else "")
        top.set_title(f"{EVAL_LABEL[ev]}  (n={len(wide)})", fontsize=9.5)
        top.grid(axis="y", alpha=0.25)
        # How much the winner actually won by, per subject.
        srt = np.sort(wide.to_numpy(), axis=1)
        gap = srt[:, -1] - srt[:, -2]
        bot.hist(gap, bins=40, color="#455a64")
        bot.axvline(float(np.median(gap)), color="#a51f18", lw=1.3,
                    label=f"median {np.median(gap):.3f}")
        bot.set_xlabel("winner − runner-up, per subject")
        bot.set_ylabel("subjects" if col == 0 else "")
        bot.legend(fontsize=7.5)
        bot.grid(axis="y", alpha=0.25)
    return _finish(fig, "Who wins a subject, and by how much",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · dotted line on the top "
                   "row is the share expected if every arm were equivalent")


# ========================================================================== 4. cost

def pareto(subj: pd.DataFrame, evaluation: str,
           align_tag: str = "none") -> plt.Figure | None:
    """Score against parameter count -- the trade growth is actually for.

    Growth's claim was never "higher accuracy"; it was "the same accuracy from a model
    that sized itself". This is the axis that claim lives on. Parameter counts for the
    growing arms are *measured per fold* and averaged, not looked up, because their
    size is an outcome of the run; the horizontal bar is the interquartile range of the
    sizes the folds actually ended at, and an arm with a wide bar did not converge on
    one architecture.
    """
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    sel = sel[sel.n_params.notna()] if "n_params" in sel.columns else pd.DataFrame()
    if sel.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    # Scores are centred per dataset before pooling: raw scores across twelve datasets
    # of two metrics have no common axis, and a mean over them is a mean over dataset
    # difficulty as much as over models. The reference mean is taken over ALL arms on
    # the dataset -- centring within (model, dataset) would subtract each arm's own
    # mean and collapse every point onto zero, which is a silently empty figure rather
    # than a wrong one.
    ref = sel.groupby("dataset").score.mean()
    sel = sel.assign(z=sel.score - sel.dataset.map(ref))
    for k, m in enumerate(_present(sel, "model", MODEL_ORDER)):
        d = sel[sel.model == m]
        z, p = d.z.to_numpy(), d.n_params.to_numpy()
        lo, hi = perf_io.boot_ci(z)
        ax.errorbar(np.median(p), z.mean(),
                    xerr=[[max(np.median(p) - np.percentile(p, 25), 0)],
                          [max(np.percentile(p, 75) - np.median(p), 0)]],
                    yerr=[[max(z.mean() - lo, 0)], [max(hi - z.mean(), 0)]],
                    fmt=ARCH_MARKER.get(perf_io.arch_of(m), "o"), ms=9, capsize=3,
                    lw=1.5, color=MODEL_COLOR.get(m, "#616161"))
        # Arms of one architecture land at nearly the same size, so a fixed offset
        # stacks their labels on top of each other; alternating the vertical offset
        # separates them without a layout solver.
        ax.annotate(m, (np.median(p), z.mean()), textcoords="offset points",
                    xytext=(9, (12, -16, 4)[k % 3]), fontsize=7.5, family="monospace",
                    color=MODEL_COLOR.get(m, "#616161"))
    ax.set_xscale("log")
    ax.axhline(0, color="#455a64", lw=0.8, ls=":")
    ax.set_xlabel("parameters at the end of the fit (median, IQR bar)")
    ax.set_ylabel("score, centred within dataset")
    ax.grid(alpha=0.25)
    return _finish(fig, "Accuracy against size",
                   f"{EVAL_LABEL[evaluation]} · {ALIGN_LABEL.get(align_tag, align_tag)}"
                   " · up and left is better")


def pareto_subjects(subj: pd.DataFrame, evaluation: str,
                    align_tag: str = "none") -> plt.Figure | None:
    """The same trade, before the median: one point per subject, one panel per dataset.

    :func:`pareto` is a summary of twelve datasets in nine points, and it is the right
    summary for the question it answers -- does growth buy accuracy per parameter. It is
    the wrong one for everything the size does *within* a dataset. A growing arm whose
    subjects stop between 3k and 12k parameters and a fixed arm frozen at 7k are one
    point each there, indistinguishable; here the first is a cloud and the second a
    vertical line, and that difference is the whole question of whether the final size
    is a property of the subject or of the configuration.

    The number in each panel is Spearman's ρ between size and score *across subjects*
    for the growing arm. It is the premise of the donor-receiver protocol: if ρ were
    near 1, "size predicts better than accuracy" would be a sentence about one quantity
    wearing two names. Scores are raw here, not centred, because a panel is one dataset
    and one metric -- there is nothing to put on a common axis.
    """
    sel = subj[(subj["eval"] == evaluation) & (subj.align_tag == align_tag)]
    sel = sel[sel.n_params.notna()] if "n_params" in sel.columns else pd.DataFrame()
    if sel.empty:
        return None
    datasets = _present(sel, "dataset", DATASET_ORDER)
    fig, axes = _panel_grid(len(datasets))
    for ax, ds in zip(axes, datasets):
        d = sel[sel.dataset == ds]
        for m in _present(d, "model", MODEL_ORDER):
            v = d[d.model == m]
            ax.plot(v.n_params, v.score, ARCH_MARKER.get(perf_io.arch_of(m), "o"),
                    ms=4.0, alpha=0.65, lw=0, color=MODEL_COLOR.get(m, "#616161"),
                    label=m)
        ch = float(d.chance.iloc[0])
        ax.axhline(ch, color="#263238", lw=1.0, ls=":", zorder=0)
        notes = []
        for m in ("grow_shallow", "grow_deep", "grow_sccnet"):
            v = d[d.model == m]
            # A growing arm that stopped at the same width on every subject has no rank
            # to correlate; reporting rho on a constant would be a number, not a fact.
            if len(v) >= 8 and np.ptp(v.n_params.to_numpy()) > 0:
                rho = stats.spearmanr(v.n_params, v.score)[0]
                spread = v.n_params.max() / max(v.n_params.min(), 1)
                notes.append(f"{m}: ρ={rho:+.2f}  ×{spread:.1f}")
        if notes:
            ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes, va="bottom",
                    fontsize=6.8, family="monospace", color="#37474f",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        # A dataset whose arms all sit inside one decade gets a log axis whose minor
        # ticks are all labelled and overprint each other into a solid bar. Below a
        # third of a decade the log scale buys nothing anyway, so drop to linear.
        p = d.n_params.to_numpy(float)
        if p.max() / max(p.min(), 1) > 2.0:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title(f"{ds}   {d.subject.nunique()} subjects", fontsize=9,
                     family="monospace")
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.22)
    _label_edges(axes, "parameters at end of fit", "score")
    axes[0].legend(fontsize=6.2, loc="best", framealpha=0.92, numpoints=1, ncol=2)
    return _finish(fig, "Accuracy against size, subject by subject",
                   f"{EVAL_LABEL[evaluation]} · {ALIGN_LABEL.get(align_tag, align_tag)}"
                   " · one marker per subject · ρ is the rank correlation between a "
                   "subject's final size and its score, ×n the ratio of largest to "
                   "smallest size within the arm")


def cost(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure | None:
    """Wall-clock per fold against the score it bought.

    A growth step rebuilds the optimizer and runs a line search over a held-out slice,
    so a growing arm's epoch is not a fixed arm's epoch. Whether that overhead is
    material is a measurement, not an assumption, and the campaign records it per fold.
    """
    if "seconds" not in subj.columns or subj.seconds.isna().all():
        return None
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(13.5, 4.0), sharey=True)
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        sel = subj[(subj["eval"] == ev) & (subj.align_tag == align_tag)
                   & subj.seconds.notna()]
        if sel.empty:
            ax.set_axis_off()
            continue
        models = _present(sel, "model", MODEL_ORDER)
        for i, m in enumerate(models):
            v = sel[sel.model == m].seconds.to_numpy()
            ax.boxplot([v], positions=[i], widths=0.6, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=MODEL_COLOR.get(m, "#616161"),
                                     alpha=0.75, lw=0.8),
                       medianprops=dict(color="white", lw=1.4),
                       whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=90, fontsize=7, family="monospace")
        ax.set_yscale("log")
        ax.set_title(EVAL_LABEL[ev], fontsize=9.5)
        ax.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("seconds per fold")
    return _finish(fig, "What a fold cost",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · same 200-epoch budget "
                   "for every arm, so this is per-epoch overhead, not more training")


def width_reached(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure | None:
    """Did the growing arms reach the width their fixed controls were frozen at?

    The decomposition assumes the three arms of a triple are comparable. Two of them
    are, to the parameter. The third is only comparable *if it grew all the way* -- and
    where it did not, ``grow − fix`` is confounded with a size difference the figure
    has to show rather than the reader assume away.
    """
    if "width_end" not in subj.columns:
        return None
    g = subj[subj.model.str.startswith("grow") & subj.width_end.notna()
             & (subj.align_tag == align_tag)]
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    labels, pos = [], 0
    for ev in EVAL_ORDER:
        for m in _present(g, "model", MODEL_ORDER):
            d = g[(g["eval"] == ev) & (g.model == m)]
            if d.empty:
                continue
            frac = (d.width_end / d.target_width).to_numpy()
            frac = frac[np.isfinite(frac)]
            if not len(frac):
                continue
            ax.boxplot([frac], positions=[pos], widths=0.6, showfliers=False,
                       patch_artist=True, zorder=3,
                       boxprops=dict(facecolor=MODEL_COLOR.get(m, "#616161"),
                                     alpha=0.8, lw=0.8, zorder=3),
                       medianprops=dict(color="white", lw=1.4, zorder=4))
            reached = float((frac >= 0.999).mean())
            ax.text(pos, 1.06, f"{reached:.0%}", ha="center", fontsize=7,
                    color="#37474f")
            # An arm that reached the target on nearly every fold collapses to a box of
            # zero height sitting exactly under the target line, which reads as a
            # missing bar. Marked explicitly so "always at target" and "not run" cannot
            # look the same.
            if reached > 0.75:
                ax.plot([pos], [1.0], marker="_", ms=16, mew=3, zorder=5,
                        color=MODEL_COLOR.get(m, "#616161"))
            labels.append(f"{ev.replace('_', '-')}\n{m}")
            pos += 1
    # Behind the boxes, for the same reason.
    ax.axhline(1.0, color="#a51f18", ls="--", lw=1.2, zorder=1, label="target width")
    ax.set_xticks(range(pos))
    ax.set_xticklabels(labels, fontsize=6.5, family="monospace")
    ax.set_ylabel("final width / target width")
    ax.set_ylim(0, 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    return _finish(fig, "Width matching: did the growing arm get there?",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · the percentage above "
                   "each box is the share of subjects that reached the target exactly")


# ================================================================= 5. protocol effects

def alignment_effect(subj: pd.DataFrame) -> plt.Figure:
    """What Euclidean alignment is worth, per arm and protocol.

    Half the grid is EA and half is raw, and every other figure fixes one of them.
    This is the one place the choice itself is the variable. Paired on the same
    subjects, so it is the alignment effect and not a difference in coverage.
    """
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(13.5, 4.2), sharex=True)
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        sel = subj[subj["eval"] == ev]
        models = _present(sel, "model", MODEL_ORDER)
        drawn = 0
        for i, m in enumerate(models):
            keys = ["dataset", "subject"]
            raw = sel[(sel.model == m) & (sel.align_tag == "none")].set_index(keys)
            ea = sel[(sel.model == m) & (sel.align_tag == "easubject")].set_index(keys)
            common = raw.index.intersection(ea.index)
            if len(common) < 5:
                continue
            d = (ea.loc[common, "score"] - raw.loc[common, "score"]).to_numpy()
            r = perf_io.test(d)
            ax.errorbar(r["mean"], i, xerr=[[max(r["mean"] - r["lo"], 0)],
                                            [max(r["hi"] - r["mean"], 0)]],
                        fmt=ARCH_MARKER.get(perf_io.arch_of(m), "o"), ms=6, capsize=3,
                        lw=1.5, color=MODEL_COLOR.get(m, "#616161"))
            ax.text(r["hi"] + 0.002, i, f"{r['n_win']}/{r['n']}", va="center",
                    fontsize=6.5, color="#546e7a")
            drawn += 1
        ax.axvline(0, color="#455a64", lw=1)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8, family="monospace")
        ax.invert_yaxis()
        ax.set_title(EVAL_LABEL[ev], fontsize=10)
        ax.set_xlabel("EA − raw")
        ax.grid(axis="x", alpha=0.25)
        if not drawn:
            ax.text(0.5, 0.5, "no paired coverage", transform=ax.transAxes,
                    ha="center", color="#90a4ae")
    return _finish(fig, "Euclidean alignment, paired on the same subjects",
                   "n = subjects scored under both alignments · 95 % bootstrap")


def protocol_penalty(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """The same arm, the same subjects, three protocols -- the generalisation cost.

    Within-session, cross-session and cross-subject are three different questions asked
    of one decoder, and the drop between them is the quantity a deployed system cares
    about. Restricted to subjects present in all three so the lines are a protocol
    effect and not a change of population.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    sel = subj[subj.align_tag == align_tag]
    for m in _present(sel, "model", MODEL_ORDER):
        d = sel[sel.model == m]
        wide = d.pivot_table(index=["dataset", "subject"], columns="eval",
                             values="score")
        evs = [e for e in EVAL_ORDER if e in wide.columns]
        wide = wide[evs].dropna()
        if wide.empty or len(evs) < 2:
            continue
        means = [wide[e].mean() for e in evs]
        ax.plot(range(len(evs)), means, marker=ARCH_MARKER.get(perf_io.arch_of(m), "o"),
                color=MODEL_COLOR.get(m, "#616161"), lw=1.6, ms=7, label=m)
        for x, e in enumerate(evs):
            lo, hi = perf_io.boot_ci(wide[e].to_numpy())
            ax.plot([x, x], [lo, hi], color=MODEL_COLOR.get(m, "#616161"), lw=1,
                    alpha=0.6)
    ax.set_xticks(range(len(EVAL_ORDER)))
    ax.set_xticklabels([EVAL_LABEL[e] for e in EVAL_ORDER], fontsize=9)
    ax.set_ylabel("score, subjects common to all three protocols")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=3, framealpha=0.92)
    return _finish(fig, "What generalisation costs",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · each line is one arm on "
                   "the subjects it scored under every protocol")


def seed_noise(sc: pd.DataFrame, align_tag: str = "none") -> plt.Figure:
    """Seed-to-seed spread against the effects being argued about.

    The scale bar for every other figure. Three seeds per cell means the spread across
    them is measurable, and an effect smaller than it is an effect smaller than
    re-running the same configuration. The red lines are the growth terms from
    :func:`decomposition`, drawn on the same axis.
    """
    sel = sc[sc.align_tag == align_tag]
    spread = (sel.groupby(["eval", "dataset", "model", "subject", "session"])
                 .score.agg(["std", "size"]))
    spread = spread[spread["size"] >= 2]["std"].dropna()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    if spread.empty:
        ax.text(0.5, 0.5, "fewer than two seeds anywhere", transform=ax.transAxes,
                ha="center", color="#90a4ae")
        return _finish(fig, "Seed noise", "")
    ax.hist(spread, bins=60, color="#455a64")
    med = float(spread.median())
    ax.axvline(med, color="#263238", lw=1.4, ls="--",
               label=f"median seed sd {med:.4f}")
    subj = perf_io.by_subject(sel)
    for arch in TRIPLES:
        d = perf_io.decompose(subj, arch, "within_session", align_tag)
        if d and d["has_control"]:
            ax.axvline(abs(d["growth"]["mean"]), color=TERM_COLOR["growth"], lw=1.2,
                       alpha=0.85)
            ax.text(abs(d["growth"]["mean"]), ax.get_ylim()[1] * 0.92,
                    f" |growth| {arch}", rotation=90, fontsize=7,
                    color=TERM_COLOR["growth"], va="top")
    ax.set_xlabel("sd across seeds, within one (dataset, subject, session, arm)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, "How big is the noise the effects have to clear?",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · red lines are the "
                   "within-session growth terms, on the same scale")


def train_size(subj: pd.DataFrame, align_tag: str = "none") -> plt.Figure | None:
    """Score against training-set size, per family.

    The hypothesis growth is supposed to serve: a model that sizes itself should be
    ahead where data is scarce and level off where it is plentiful. If the growing and
    fixed families' curves are parallel, growth is not doing the thing it was proposed
    for, whatever the pooled delta says.
    """
    if "samples" not in subj.columns:
        return None
    sel = subj[(subj.align_tag == align_tag) & subj.samples.notna()
               & (subj.samples > 0)]
    if sel.empty:
        return None
    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(13.5, 4.2), sharey=True)
    for ax, ev in zip(np.atleast_1d(axes), EVAL_ORDER):
        d = sel[sel["eval"] == ev]
        if d.empty:
            ax.set_axis_off()
            continue
        # Centred within dataset for the same reason as `pareto`: otherwise this plots
        # dataset difficulty against dataset size and calls it a learning curve.
        d = d.assign(z=d.groupby("dataset").score.transform(lambda s: s - s.mean()))
        # ONE set of bin edges for all three families, taken over the pooled column.
        # Per-family quantiles put each family's points at different x, so the three
        # lines are read against each other at sizes they were never compared at --
        # and where a family has few distinct sizes the quantiles collapse and the
        # line breaks into disconnected segments, which is what this figure did.
        edges = np.unique(np.quantile(d.samples, np.linspace(0, 1, 7)))
        if len(edges) < 3:
            ax.text(0.5, 0.5, "training size is constant here",
                    transform=ax.transAxes, ha="center", color="#90a4ae")
            ax.set_xticks([])
            continue
        idx = np.clip(np.digitize(d.samples, edges[1:-1]), 0, len(edges) - 2)
        for fam in ("braindecode", "fixed control", "growing"):
            f, fi = d[d.family == fam], idx[(d.family == fam).to_numpy()]
            if len(f) < 10:
                continue
            pts = [(f.samples[fi == b].median(), f.z[fi == b].mean(),
                    int((fi == b).sum())) for b in range(len(edges) - 1)]
            pts = [(x, y, n) for x, y, n in pts if n >= 5]
            if len(pts) < 2:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                    lw=1.7, color=FAM_COLOR[fam], label=fam)
        ax.set_xscale("log")
        ax.axhline(0, color="#455a64", lw=0.8, ls=":")
        ax.set_xlabel("training trials")
        ax.set_title(EVAL_LABEL[ev], fontsize=9.5)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("score, centred within dataset")
    np.atleast_1d(axes)[0].legend(fontsize=8, framealpha=0.92)
    return _finish(fig, "Does growth pay off where data is scarce?",
                   f"{ALIGN_LABEL.get(align_tag, align_tag)} · shared sextiles of "
                   "training-set size, bins with fewer than 5 units dropped · training "
                   "size is largely a dataset property here, so read this as a trend "
                   "across datasets, not a learning curve within one")
