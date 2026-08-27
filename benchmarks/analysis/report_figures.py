"""One figure set for the whole project, so a reader sees the state in one page.

Same convention as `v5_figures` and `chance_pareto_figures`: every function takes
already-loaded frames and returns the Figure, and nothing here reads a file outside
`main`. Figures that already exist are IMPORTED rather than re-drawn -- a claim gets one
definition, and a figure that disagrees with its script is worse than no figure.

What is new here is the EA panel. `ea_replication.py` prints that analysis; these three
figures show it, and they call its own `load`/`paired_delta`/`boot_ci` so the picture
cannot drift from the numbers in the text.

The set is deliberately organised by CLAIM, not by campaign:

  family_levels        which family wins, and in which data regime -- the one place the
                       absolute level is the subject rather than the delta
  paired_forest        the growing - fixed contrast per cell, with the cells whose arms
                       do not clear their own chance threshold drawn hollow. This is the
                       correction v5 did not have.
  accuracy_per_param   the same contrast on two axes, so "smaller AND no worse" is
                       readable without winning the accuracy column outright
  ea_gate              stage 1: does our Euclidean Alignment reproduce +5.05 pp
  ea_where             where the gain lives -- EA pays the weak subjects, which is why
                       selecting subjects would destroy the effect it is meant to show
  ea_power             why the gate moved to Cho2017: the MDE at n=9 is above the target
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from chance_pareto_figures import accuracy_per_param, chance_geometry  # noqa: F401
from ea_replication import MODELS, PAPER, boot_ci, load, paired_delta

EVALS = ["within_session", "cross_session", "cross_subject"]
EVAL_LABEL = {"within_session": "within-session", "cross_session": "cross-session",
              "cross_subject": "cross-subject (LOSO)"}
EV_STYLE = {"within_session": ("o", "C0"), "cross_session": ("s", "C1"),
            "cross_subject": ("^", "C2")}
PAIR_LABEL = {"grow_shallow vs bd_shallow": "ShallowFBCSPNet",
              "grow_sccnet vs bd_sccnet": "SCCNet",
              "grow_eegnex vs bd_eegnex": "EEGNeX  (not width-matched)",
              "grow_deep vs bd_deep4": "Deep4Net"}
FAMILY_LABEL = {"riemann/csp": "Riemann / CSP", "braindecode": "braindecode (fixed)",
                "growing": "growing"}
NET_LABEL = {"bd_eegnet": "EEGNet", "bd_shallow": "ShallowFBCSPNet",
             "bd_deep4": "Deep4Net"}


# --------------------------------------------------------------- v5 grid, corrected
def family_levels(levels: pd.DataFrame):
    """Absolute ROC-AUC per family and protocol.

    AUC datasets only: pooling accuracy and AUC would average two scales whose chance
    levels differ, and the bar heights would then mean nothing. This is the figure that
    carries the regime story -- Riemannian ahead where training is within a subject,
    networks ahead only once they see dozens of them.
    """
    fam = (levels[levels.metric == "roc_auc"]
           .groupby(["family", "eval"], as_index=False).score_mean.mean()
           .pivot(index="family", columns="eval", values="score_mean")
           .reindex(columns=EVALS))
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(EVALS))
    w = 0.26
    for i, family in enumerate(["riemann/csp", "braindecode", "growing"]):
        vals = [fam.loc[family, e] for e in EVALS]
        ax.bar(x + (i - 1) * w, vals, w, label=FAMILY_LABEL[family],
               color=f"C{i}", edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [EVAL_LABEL[e] for e in EVALS])
    ax.set_ylabel("ROC-AUC, mean over the 6 AUC datasets")
    ax.set_ylim(0.5, 0.86)
    ax.axhline(0.5, color="0.4", lw=1, ls=":")
    ax.text(-0.42, 0.506, "chance", fontsize=7, color="0.4", ha="left")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Deep learning only pays once the training set is many subjects",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    f.tight_layout()
    return f


def paired_forest(chance: pd.DataFrame):
    """growing − fixed per (dataset, protocol), one panel per architecture.

    Hollow marker = at least one of the two arms does not clear its own exact chance
    threshold in that cell, so the delta is a difference between two coin flips. 27 of
    the 120 comparisons are in that state, and they include the largest positive deltas
    in the whole grid -- which is exactly why they are drawn differently rather than
    dropped silently.
    """
    pairs = [p for p in PAIR_LABEL if p in set(chance.pair)]
    f, axes = plt.subplots(1, len(pairs), figsize=(3.9 * len(pairs), 5.6), sharex=True)
    axes = np.atleast_1d(axes)
    ds = sorted(chance.dataset.unique())
    ypos = {d: i for i, d in enumerate(ds)}
    for ax, pair in zip(axes, pairs):
        g = chance[chance.pair == pair]
        cred = g.grow_ok & g.fixed_ok
        for ev, (m, c) in EV_STYLE.items():
            h = g[g["eval"] == ev]
            ok = (h.grow_ok & h.fixed_ok).to_numpy()
            ax.scatter(h.delta, [ypos[d] for d in h.dataset], marker=m, s=48,
                       facecolors=np.where(ok, c, "none"), edgecolors=c,
                       linewidths=1.3, label=EVAL_LABEL[ev], zorder=3)
        ax.axvline(0, color="0.3", lw=1)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(ds if ax is axes[0] else [""] * len(ds), fontsize=8)
        n_pos = int((g.delta[cred] > 0).sum())
        ax.set_title(f"{PAIR_LABEL[pair]}\n{n_pos}/{int(cred.sum())} credible cells won",
                     fontsize=10)
        ax.set_xlabel(r"$\Delta$  (growing $-$ fixed)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="lower left")
    f.suptitle("Filled = both arms beat their own chance threshold;  hollow = at least "
               "one arm is at chance, so the delta is uninterpretable", fontsize=10.5)
    f.tight_layout()
    return f


# ------------------------------------------------------------------ stage 1: the EA gate
def ea_gate(delta: pd.Series, d: pd.DataFrame, dataset: str, n_sub: int):
    """Left: the paired delta per net, with its bootstrap CI, against the paper's target.
    Right: every held-out subject's raw → aligned pair, pooled over the three nets.

    The absolute level is NOT the test and the right panel is not a claim about it: our
    harness is not theirs. Everything that separates the two harnesses is shared between
    our own two arms and cancels in the paired delta, which is what the left panel shows.
    """
    f, (a0, a1) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})

    rows = [("pooled (3 nets)", delta.groupby(level="subject").mean().to_numpy())]
    rows += [(NET_LABEL[m], delta.xs(m, level="model").to_numpy()) for m in MODELS]
    y = np.arange(len(rows))[::-1]
    for (name, a), yi in zip(rows, y):
        lo, hi = boot_ci(a)
        c = "C3" if name.startswith("pooled") else "C0"
        a0.plot([lo * 100, hi * 100], [yi, yi], color=c, lw=2.6, solid_capstyle="round",
                zorder=2)
        a0.scatter(a.mean() * 100, yi, s=90, color=c, zorder=3, edgecolor="white",
                   linewidth=1.1)
        a0.scatter(a * 100, np.full(len(a), yi) + 0.20, s=13, color=c, alpha=0.45,
                   zorder=1)
        a0.text(hi * 100 + 0.5, yi, f"{a.mean() * 100:+.2f} pp", va="center", fontsize=9,
                color=c)
    a0.axvline(0, color="0.3", lw=1)
    a0.axvline(PAPER["delta"] * 100, color="C2", lw=1.6, ls="--", zorder=1)
    a0.text(PAPER["delta"] * 100 + 0.15, y[0] + 0.55, "published +5.05 pp",
            color="C2", fontsize=9)
    a0.set_yticks(y, [r[0] for r in rows], fontsize=10)
    a0.set_xlabel("accuracy of aligned − accuracy of raw, per held-out subject (pp)")
    a0.set_ylim(y[-1] - 0.7, y[0] + 0.95)
    a0.set_title(f"{dataset}, LOSO, n = {n_sub} subjects\n"
                 "bar = 95 % bootstrap CI of the mean, dots = the subjects", fontsize=10)
    a0.grid(axis="x", alpha=0.25)
    a0.set_axisbelow(True)

    sub = (d[d["model"].isin(MODELS)]
           .groupby(["align", "subject"]).score.mean().unstack("align") * 100)
    # Label the subjects only when there are few enough to read. At n=52 fifty-two
    # labels are a texture, not information, and the individual identities stop being
    # the point once the panel is about the distribution.
    label_subjects = len(sub) <= 12
    for s, r in sub.iterrows():
        up = r["euclidean"] >= r["none"]
        a1.plot([0, 1], [r["none"], r["euclidean"]], color="C0" if up else "C3",
                lw=1.5 if label_subjects else 0.9,
                alpha=0.85 if label_subjects else 0.45,
                marker="o", ms=5 if label_subjects else 3, zorder=2)
        if label_subjects:
            a1.text(1.04, r["euclidean"], f"S{s}", fontsize=8, va="center",
                    color="C0" if up else "C3")
    a1.plot([0, 1], [sub["none"].mean(), sub["euclidean"].mean()], color="k", lw=3,
            marker="o", ms=8, zorder=3, label="mean")
    a1.set_xticks([0, 1], ["raw", "Euclidean\nalignment"], fontsize=10)
    a1.set_xlim(-0.22, 1.22)
    a1.set_ylabel("accuracy (%), pooled over the 3 nets")
    a1.margins(y=0.08)
    n_up = int((sub["euclidean"] > sub["none"]).sum())
    a1.set_title(f"{n_up} of {len(sub)} subjects gain\n"
                 f"{sub['none'].mean():.2f} % → {sub['euclidean'].mean():.2f} %",
                 fontsize=10)
    a1.legend(fontsize=9, loc="lower left")
    a1.grid(axis="y", alpha=0.25)
    a1.set_axisbelow(True)
    f.tight_layout()
    return f


def ea_where(sets: list[tuple[str, pd.Series, pd.DataFrame]]):
    """The EA gain against the raw baseline, one point per held-out subject.

    This panel exists because it was WRONG the first time. On BNCI2014-001 the nine
    subjects gave rho = -0.467 and the reading was "alignment pays the subjects a BCI
    works worst on". At n=52 the same measurement gives rho = +0.038: the pattern was a
    small-sample artefact. Both are drawn, with their n, rather than the second quietly
    replacing the first -- an n=9 correlation is exactly the kind of claim that survives
    into a paper if the refutation is not shown next to it.
    """
    f, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, delta, d) in zip(axes, sets):
        raw = (d[(d["align"] == "none") & d["model"].isin(MODELS)]
               .groupby("subject").score.mean() * 100)
        gain = delta.groupby(level="subject").mean() * 100
        both = pd.concat([raw.rename("raw"), gain.rename("gain")], axis=1).dropna()
        rho, p = stats.spearmanr(both["raw"], both["gain"])

        ax.axhline(0, color="0.3", lw=1)
        ax.scatter(both["raw"], both["gain"], s=95 if len(both) <= 12 else 42, zorder=3,
                   color=np.where(both["gain"] > 0, "C0", "C3"), edgecolor="white",
                   linewidth=1.1)
        if len(both) <= 12:
            for s, r in both.iterrows():
                ax.annotate(f"S{s}", (r["raw"], r["gain"]), textcoords="offset points",
                            xytext=(8, 4), fontsize=9, color="0.35")
        b, a = np.polyfit(both["raw"], both["gain"], 1)
        xs = np.linspace(both["raw"].min() - 2, both["raw"].max() + 2, 50)
        ax.plot(xs, a + b * xs, color="0.5", lw=1.2, ls="--", zorder=1)
        ax.set_xlabel("raw accuracy of the held-out subject (%)")
        ax.set_title(f"{label}   n = {len(both)}\n"
                     f"Spearman $\\rho$ = {rho:+.3f}  (p = {p:.2f})", fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("gain from alignment (pp)")
    f.suptitle("Does alignment pay the subjects the decoder fails on? "
               "Only at n = 9.", fontsize=11)
    f.tight_layout()
    return f


def ea_datasets(d_bnci: pd.Series, d_cho: pd.Series):
    """The same gate on 9 subjects and on 52, per network and pooled.

    This is the figure the campaign was run to produce. Two things are visible at once
    that no single-dataset panel can show.

    The CIs SHRINK: same estimator, same code, four times the subjects. That is the
    whole return on moving datasets, and it is why "no effect" was never available from
    the n=9 panel.

    The per-network ORDER changes: ShallowFBCSPNet carried the clearest effect at n=9
    (+3.98, p=0.012) and carries none at n=52 (+0.59, p=0.24), while Deep4Net does the
    opposite. Per-network structure read off nine subjects was noise. The pooled row is
    the estimand the published +5.05 pp refers to, and it is the only row to quote.
    """
    f, ax = plt.subplots(figsize=(8.6, 4.6))
    rows = [("pooled (3 nets)", "pooled")] + [(NET_LABEL[m], m) for m in MODELS]
    for i, (name, key) in enumerate(rows):
        yi = len(rows) - 1 - i
        for delta, off, c, lab in [(d_bnci, +0.17, "C1", "BNCI2014-001, n = 9"),
                                   (d_cho, -0.17, "C0", "Cho2017, n = 52")]:
            a = (delta.groupby(level="subject").mean().to_numpy() if key == "pooled"
                 else delta.xs(key, level="model").to_numpy())
            lo, hi = boot_ci(a)
            ax.plot([lo * 100, hi * 100], [yi + off] * 2, color=c, lw=3,
                    solid_capstyle="round", zorder=2,
                    label=lab if i == 0 else None)
            ax.scatter(a.mean() * 100, yi + off, s=64, color=c, zorder=3,
                       edgecolor="white", linewidth=1.1)
    ax.axvline(0, color="0.3", lw=1)
    ax.axvline(PAPER["delta"] * 100, color="C2", lw=1.6, ls="--", zorder=1)
    # Opaque backing: the n=9 intervals run straight through this label, and a legend
    # you have to read a bar through is not a legend.
    ax.text(PAPER["delta"] * 100 + 0.15, len(rows) - 0.55, "published\n+5.05 pp",
            color="C2", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.axhline(len(rows) - 1.5, color=(0, 0, 0, 0.18), lw=0.9)
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows], fontsize=10)
    ax.set_xlabel("aligned − raw, per held-out subject (pp), 95 % bootstrap CI")
    ax.set_ylim(-0.7, len(rows) - 0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Four times the subjects: the interval shrinks, and the "
                 "per-network story does not survive", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    f.tight_layout()
    return f


def ea_power(delta: pd.Series):
    """Minimum detectable effect against the number of held-out subjects.

    The limit here is the between-SUBJECT sd, not the number of seeds: seeds are
    replication inside a subject and shrink a variance component that is already small.
    That is the whole argument for moving the gate to a 52-subject dataset rather than
    adding seeds on a 9-subject one.
    """
    a = delta.groupby(level="subject").mean().to_numpy()
    sd = float(a.std(ddof=1))

    # t quantiles at n-1 df, NOT normal ones -- the same formula `ea_replication.py`
    # prints. At n=9 the two differ by 0.8 pp, which is a sixth of the effect being
    # tested, so a figure drawn with z would contradict the text it illustrates.
    def mde_at(k):
        return ((stats.t.ppf(0.975, k - 1) + stats.t.ppf(0.80, k - 1))
                * sd / np.sqrt(k)) * 100

    n = np.arange(5, 60)
    mde = np.array([mde_at(k) for k in n])

    f, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(n, mde, color="C0", lw=2.2, zorder=3)
    ax.axhline(PAPER["delta"] * 100, color="C2", lw=1.6, ls="--", zorder=2)
    ax.text(57, PAPER["delta"] * 100 + 0.25, "target +5.05 pp", color="C2", fontsize=9,
            ha="right")
    ax.fill_between(n, PAPER["delta"] * 100, mde, where=mde > PAPER["delta"] * 100,
                    color="C3", alpha=0.10, zorder=1)
    for xi, lab, c in [(9, "BNCI2014-001\nn = 9", "C3"), (52, "Cho2017\nn = 52", "C0")]:
        yi = mde_at(xi)
        ax.scatter([xi], [yi], s=80, color=c, zorder=4, edgecolor="white", linewidth=1.1)
        ax.annotate(f"{lab}\nMDE {yi:.2f} pp", (xi, yi), textcoords="offset points",
                    xytext=(10, 12), fontsize=9, color=c)
    ax.set_xlabel("held-out subjects")
    ax.set_ylabel("minimum detectable effect, power .80 (pp)")
    ax.set_title(f"What the design can resolve, at the measured between-subject "
                 f"sd of {sd * 100:.2f} pp\n"
                 "red: the region where the target is smaller than the noise floor",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(5, 59)
    f.tight_layout()
    return f


def main() -> None:
    here = Path(__file__).resolve().parent
    bench = here.parent
    out = here / "figures" / "report"
    out.mkdir(parents=True, exist_ok=True)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea-root", type=Path, required=True,
                    help="results tree holding the bnci2014_001 cross_subject EA cells")
    ap.add_argument("--cho-root", type=Path, required=True,
                    help="results tree holding the cho2017 cross_subject EA cells")
    args = ap.parse_args()

    levels = pd.read_csv(bench / "results_published" / "eegrow_benchmark_levels.csv")
    chance = pd.read_csv(here / "chance" / "growing_vs_fixed_chance.csv")
    h2h = pd.read_csv(here / "pareto" / "head_to_head.csv")
    d_b = load(args.ea_root, "bnci2014_001")
    d_c = load(args.cho_root, "cho2017")
    delta_b = paired_delta(d_b, MODELS)
    delta_c = paired_delta(d_c, MODELS)
    n_b = delta_b.index.get_level_values("subject").nunique()
    n_c = delta_c.index.get_level_values("subject").nunique()

    figs = [("family_levels", family_levels(levels)),
            ("paired_forest", paired_forest(chance)),
            ("accuracy_per_param", accuracy_per_param(h2h)),
            ("ea_gate", ea_gate(delta_b, d_b, "BNCI2014-001", n_b)),
            ("ea_gate_cho", ea_gate(delta_c, d_c, "Cho2017", n_c)),
            ("ea_datasets", ea_datasets(delta_b, delta_c)),
            ("ea_where", ea_where([("BNCI2014-001", delta_b, d_b),
                                   ("Cho2017", delta_c, d_c)])),
            ("ea_power", ea_power(delta_b))]
    for name, fig in figs:
        # Per-figure hashsalt: matplotlib derives clip-path ids from it, and two SVGs
        # inlined in the same HTML page with colliding ids clip each other.
        plt.rcParams["svg.hashsalt"] = name
        for ext in ("svg", "png"):
            p = out / f"{name}.{ext}"
            fig.savefig(p, dpi=150, bbox_inches="tight",
                        facecolor="white" if ext == "png" else "none")
            print(f"wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
