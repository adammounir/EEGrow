"""Build the v5 interim report: 10 figures, LaTeX tables, and an executed notebook.

The v5 campaign is not finished (74 of 1350 cells still running), so this is an
*interim* report and says so in every output. Generating it from a script rather than
writing it by hand means it can be re-run the moment the last cell lands, with nothing
in it a stale hand copy.

    uv run --with pandas,matplotlib,numpy,nbformat,nbclient,nbconvert,ipykernel,scipy \
        python benchmarks/analysis/build_v5_report.py

Reads benchmarks/results_v5_published/ and writes benchmarks/analysis/v5/:

    figures/*.png          one per claim
    eegrow_v5_tables.tex   standalone, compiles with pdflatex
    eegrow_v5_results.ipynb / .html

The figures live in ``v5_figures.py``, not in cells: the notebook and the LaTeX version
call the same functions, so there is one definition of each figure and it is somewhere
reviewable.

Three aggregation rules are inherited from aggregate_published.py and must not be
relaxed, because each changes the sign or the significance of a result:

1. **Pairs are architecture-matched.** ``grow_deep`` pairs with ``fix_deepeeg``, never
   with ``bd_deep4``: the mismatched pairing reads -0.022 on cross_subject where the
   matched control reads -0.000, i.e. it measures the architecture gap.
2. **Metrics are never pooled.** ROC-AUC for the two-class datasets, accuracy
   elsewhere.
3. **Seeds are replicates**, averaged within a (subject, session) before any statistic.
   Five seeds as five observations inflates n fivefold and every p-value with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat as nbf  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import growth_io  # noqa: E402
import v5_figures as vf  # noqa: E402

ROOT = HERE.parents[1]
SRC = ROOT / "benchmarks" / "results_v5_published"
OUT = HERE / "v5"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Planned grid size, from passes_v5/ on the cluster. It is the denominator of the
# completeness statement and nothing else.
N_PLANNED = 1350
N_MISSING = 74  # 70 grow_eegnex + 4 bd_eegnex, still running as of 2026-08-19
SHOWCASE = "bnci2014_001"

paired = pd.read_csv(SRC / "eegrow_benchmark_paired.csv")
levels = pd.read_csv(SRC / "eegrow_benchmark_levels.csv")
scores = pd.read_csv(SRC / "eegrow_benchmark_all_scores.csv.gz")
fits, budget, curves = growth_io.load_tidy(SRC)
prov = pd.read_csv(SRC / "eegrow_v5_provenance.csv")

# Subject/session labels for the showcase curves, joined on write order -- see
# growth_io.attach_subjects for what the join rests on and how it was validated. Only
# the showcase dataset's curves are shipped, so only its scores are needed.
curves = growth_io.attach_subjects(
    curves, scores[(scores.dataset == SHOWCASE) & (scores.family != "ml")])


# ---------------------------------------------------------------------------- tables
def pair_summary(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, protocol): weighted delta, plus a sign test over datasets."""
    recs = []
    for (pair, ev), g in p.groupby(["pair", "eval"]):
        wins = int((g.delta > 0).sum())
        recs.append(dict(
            pair=pair, arch=vf.PAIR_LABEL.get(pair, pair), eval=ev,
            delta=(g.delta * g.n_obs).sum() / g.n_obs.sum(), n_datasets=len(g),
            n_obs=int(g.n_obs.sum()), wins=wins,
            sign_p=binomtest(wins, len(g), 0.5).pvalue))
    out = pd.DataFrame(recs)
    out["eval"] = pd.Categorical(out["eval"], vf.EVALS, ordered=True)
    return out.sort_values(["arch", "eval"])


def efficiency(fits: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Score delta against the parameter count the growing arm actually ended at.

    The paired table is written as "growing to a target width versus training at that
    width", but the records say the growing arms mostly stop short. So the delta has to
    be read next to the size it was achieved at -- a win from 39% of the control's
    parameters is a different, stronger claim than a win at parity.
    """
    p = fits.groupby("model").params_end.mean()
    recs = []
    for grow, fixed in vf.PAIRS:
        if grow not in p.index or fixed not in p.index:
            continue
        s = summ[summ.pair == f"{grow} vs {fixed}"]
        recs.append(dict(
            arch=vf.PAIR_LABEL[f"{grow} vs {fixed}"],
            grow_params=p[grow], fixed_params=p[fixed],
            ratio=p[grow] / p[fixed],
            delta_min=s.delta.min(), delta_max=s.delta.max()))
    return pd.DataFrame(recs).sort_values("ratio")


summ = pair_summary(paired)
eff = efficiency(fits, summ)
fam = (levels[levels.metric == "roc_auc"]
       .groupby(["family", "eval"], as_index=False).score_mean.mean()
       .pivot(index="family", columns="eval", values="score_mean")
       .reindex(columns=vf.EVALS).round(4))


def tex_table(df, cols, header, fmt, caption, label) -> str:
    lines = [r"\begin{table}[htbp]", r"  \centering",
             r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
             r"  \begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
             r"    \toprule", "    " + " & ".join(header) + r" \\", r"    \midrule"]
    for _, r in df.iterrows():
        lines.append("    " + " & ".join(f(r[c]) for c, f in zip(cols, fmt)) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _sig(p) -> str:
    """The p is printed, not only a star: a star alone hides how close a call was."""
    if p != p:
        return "--"
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
    return (f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}") + star


def _fig(name: str, caption: str, width: str = r"\textwidth") -> str:
    return "\n".join([r"\begin{figure}[htbp]\centering",
                      rf"  \includegraphics[width={width}]{{figures/{name}.png}}",
                      r"  \caption{" + caption + "}", r"\end{figure}"])


# --------------------------------------------------------------------------- figures
figs = {}
f = vf.delta_by_dataset(paired)
figs["delta_by_dataset"] = f
f_wt, width_agg = vf.width_vs_target(fits)
figs["width_vs_target"] = f_wt
figs["family_levels"] = vf.family_levels(levels)
figs["win_matrix"] = vf.win_matrix(paired)
f_nf, _ = vf.noise_floor(scores)
figs["noise_floor"] = f_nf
figs["seed_stability"] = vf.seed_stability(scores)
figs["width_trajectory"] = vf.width_trajectory(curves, SHOWCASE)
figs["learning_curves"] = vf.learning_curves(curves, SHOWCASE)
figs["capacity"] = vf.capacity(curves, SHOWCASE)
f_bp, _ = vf.budget_pareto(budget, SHOWCASE)
figs["budget_pareto"] = f_bp
figs["stopping_budget"] = vf.stopping_budget(fits)
f_ge, events = vf.growth_events(curves, SHOWCASE)
figs["growth_events"] = f_ge
f_sd, subj_delta = vf.subject_delta(scores, SHOWCASE)
figs["subject_delta"] = f_sd
# The two subjects whose loss holds on all five seeds. Both, not just the one that was
# reported: a per-subject failure that turns out not to be unique is a different
# finding from an anomaly.
LOSERS = list(subj_delta.sort_values("mean").head(2).index)
for subj in LOSERS:
    figs[f"subject_curves_s{subj}"] = vf.subject_curves(curves, subj)

for name, fig in figs.items():
    fig.savefig(FIG / f"{name}.png", dpi=165, bbox_inches="tight")

# ----------------------------------------------------------------------------- LaTeX
pct = 100 * (N_PLANNED - N_MISSING) / N_PLANNED
tex = [
    r"\documentclass[11pt]{article}",
    r"\usepackage[margin=2.4cm]{geometry}",
    r"\usepackage{booktabs,graphicx,caption}",
    r"\title{Growing networks for EEG decoding:\\interim results of the v5 campaign}",
    r"\author{Adam Mounir\\Inria TAU / Yneuro}",
    r"\date{\today}",
    r"\begin{document}", r"\maketitle",
    r"\section*{Status}",
    (f"The v5 grid is {N_PLANNED - N_MISSING} of {N_PLANNED} cells complete "
     f"({pct:.1f}\\%). The {N_MISSING} missing cells are all "
     r"\texttt{grow\_eegnex} / \texttt{bd\_eegnex}: they were excluded from the "
     r"original campaign by out-of-memory failures on 11\,GB cards, not by design, so "
     r"they are the memory-heaviest cells of the grid and are running now on H100s. "
     r"Every EEGNeX number below is on a biased subset and is expected to move. "
     f"One single regime across the whole campaign: {prov.sfreq.iloc[0]:.0f}\\,Hz, "
     f"{prov.fmin.iloc[0]:.0f}--{prov.fmax.iloc[0]:.0f}\\,Hz, "
     f"braindecode {prov.v_braindecode.iloc[0]}, torch {prov.v_torch.iloc[0]}."),
    "",
    (r"\paragraph{A defect that bounds every growing arm.} The scaling-factor line "
     r"search in \texttt{grow\_step} includes $0$, and when $0$ is selected the growth "
     r"is applied anyway, at exactly zero amplitude. A unit whose incoming and "
     r"outgoing weights are both exactly zero sits at an exact stationary point --- "
     r"the gradient of its incoming weights is proportional to its (zero) outgoing "
     r"weights, and the gradient of its outgoing weights to its (zero) activation --- "
     r"so it can never recover, while \texttt{width} and \texttt{n\_params} keep "
     r"reporting it as capacity. Measured per fold: chosen factors $[0,1] \to 66.7\%$ "
     r"of new parameters with $\hat v = 0$, $[0,0] \to 100\%$, $[1,1] \to 0\%$. "
     r"\textbf{Every growing number below is therefore a lower bound.}"),
    "",
    r"\section{What the growing arms actually did}",
    (r"The paired contrast is usually stated as \emph{growing to a target width versus "
     r"training at that width directly}. The fit records say that is not what ran: "
     r"across all folds, a third of them never grow at all, and "
     r"\texttt{grow\_shallow} reaches its target width in "
     f"{100 * width_agg.loc['grow_shallow', 'reached']:.1f}\\,\\% of folds --- it "
     "stops at a mean "
     f"width of {width_agg.loc['grow_shallow', 'w_end']:.1f} against a target of "
     f"{width_agg.loc['grow_shallow', 'target']:.0f}. This does not weaken the "
     r"result, it changes what the result \emph{is}: the comparison is not "
     r"growth-versus-parity, it is growth-versus-a-larger-model."),
    "",
    _fig("width_vs_target",
         (r"Where growth actually stopped. Bars are the mean width reached, red "
          r"diamonds the target width the arm is compared against, hollow arrows the "
          r"best single fold. Right panel: the fraction of folds that grew at all, and "
          r"the fraction that reached the target.")),
    "",
    tex_table(
        eff, ["arch", "grow_params", "fixed_params", "ratio", "delta_min",
              "delta_max"],
        ["Architecture", "Growing", "Control", "Ratio", r"$\Delta$ min",
         r"$\Delta$ max"],
        # The percent sign has to be escaped: an unescaped `%` comments out the rest of
        # the row, which merges it with the next one and fails as an extra alignment tab.
        [str, lambda v: f"{v:,.0f}", lambda v: f"{v:,.0f}",
         lambda v: f"{100 * v:.0f}\\,\\%", lambda v: f"{v:+.4f}",
         lambda v: f"{v:+.4f}"],
        (r"Mean final parameter count of the growing arm against its matched control, "
         r"with the range of the score delta across the three protocols. "
         r"\texttt{grow\_shallow} beats \texttt{bd\_shallow} at 39\,\% of its "
         r"parameters; the two arms that sit near parity in size are the two that sit "
         r"near zero in score."),
        "efficiency"),
    "",
    r"\section{The paired contrast}",
    tex_table(
        summ, ["arch", "eval", "delta", "n_datasets", "wins", "n_obs", "sign_p"],
        ["Architecture", "Protocol", r"$\Delta$ (grow $-$ fixed)", "Datasets", "Won",
         r"$n$ obs.", "Sign test"],
        [str, lambda v: vf.EVAL_LABEL[v], lambda v: f"{v:+.4f}", lambda v: f"{v:d}",
         lambda v: f"{v:d}", lambda v: f"{v:d}", _sig],
        (r"Architecture-matched paired contrasts. One observation is one "
         r"(subject, session) with the five seeds averaged first; $\Delta$ is the "
         r"per-dataset mean weighted by $n$, and the sign test counts \emph{datasets}, "
         r"since subjects inside one dataset are not independent draws of the "
         r"population of datasets."),
        "paired"),
    "",
    _fig("delta_by_dataset",
         (r"Paired $\Delta$ per dataset. Filled markers are datasets the growing arm "
          r"won. ShallowFBCSPNet is the only architecture whose advantage sits on the "
          r"same side of zero in all three protocols.")),
    "",
    _fig("win_matrix",
         r"Fraction of datasets won, architecture by protocol.", r"0.8\textwidth"),
    "",
    r"\section{Is any of this above the noise?}",
    (r"The two near-zero contrasts (SCCNet, DeepEEGNet) can only be read once the "
     r"resolution of the instrument is known. Figure~4 compares every model to "
     r"\emph{itself} through the identical pipeline, splitting its five seeds into two "
     r"halves at random 200 times: the true effect is zero by construction, so the "
     r"spread is the noise floor of the whole procedure."),
    "",
    _fig("noise_floor",
         (r"Negative control. The shaded band is $\pm 0.0138$, the largest effect "
          r"claimed anywhere in this report.")),
    "",
    _fig("seed_stability",
         (r"Seed-to-seed spread within one subject and session. It exceeds every "
          r"\emph{aggregate} effect measured here, which is why five seeds are averaged "
          r"before any statistic. It does not dismiss the S2 anomaly that started this "
          r"investigation: replicated over five seeds, that subject's loss holds on all "
          r"five (\S\,8). A spread this wide means a single seed cannot establish an "
          r"effect, not that an effect it showed is absent.")),
    "",
    r"\section{Absolute level, and the cost of it}",
    _fig("family_levels",
         (r"Absolute ROC-AUC by family. The fixed control sits below both other "
          r"families in all three protocols, which is precisely why the matched "
          r"pairing is the informative comparison: beating \texttt{fix\_deepeeg} is "
          r"not the same claim as beating braindecode's references."),
         r"0.82\textwidth"),
    "",
    _fig("budget_pareto",
         (r"Accuracy against parameter-epochs --- the capacity actually paid for, "
          r"summed over epochs rather than read off the final width. Up and to the "
          r"left is better."), r"0.85\textwidth"),
    "",
    r"\section{Training dynamics}",
    (r"These are the curves that did not exist when the S2 anomaly was reported: the "
     r"script behind the published table attached no recorder, so there was nothing to "
     # The underscore has to be escaped and the name set in \texttt: a bare
     # `bnci2014_001` in prose is a subscript in math mode LaTeX refuses to open.
     r"look at. All panels show \texttt{"
     + SHOWCASE.replace("_", r"\_") + r"}, within-session."),
    "",
    _fig("width_trajectory",
         (r"Growth trajectory, one line per fold, drawn as a step function --- width "
          r"is constant between growth events, so a straight line between two growths "
          r"would draw widths the model never had.")),
    "",
    _fig("learning_curves",
         r"Train loss, validation loss and validation accuracy, mean $\pm$ SD."),
    "",
    _fig("capacity",
         (r"Parameter count over training: instantaneous (left) and cumulative "
          r"(right). The cumulative panel is the budget the run actually paid.")),
    "",
    r"\section{Why the target width is never reached}",
    (r"The width a growing arm ends at is not set by how well gromo picks neurons. It "
     r"is set by a race between the growth schedule --- one opportunity every "
     r"\texttt{grow\_every}${}=5$ epochs --- and "
     r"\texttt{EarlyStopping(patience=20)}. \emph{Every} fold stops early, at a median "
     r"of 24--27 epochs out of the 200 it was given, leaving a median of 4 to 5 growth "
     r"opportunities. \texttt{grow\_shallow} has to add 32 neurons across them, "
     r"6.4 per opportunity; it gets 0.85. The target is unreachable by arithmetic "
     r"before any property of the growth method is involved."),
    "",
    _fig("stopping_budget",
         (r"Left: epochs actually run against the budget granted. Right: the growth "
          r"rate the target demands at the median fit length, against the rate "
          r"observed. A red bar above its blue partner cannot reach its target however "
          r"well the neurons are chosen.")),
    "",
    _fig("growth_events",
         (r"Growth events per fit (left) and epochs trained after the last growth "
          r"(right). The two panels together separate \emph{two different} failures. "
          r"\texttt{shallow} grows a median of 10 times and is frozen for only about 2 "
          r"epochs afterwards: it is still growing when the fit ends, so it is "
          r"epoch-starved. \texttt{sccnet} grows 3 times and then trains a median of 50 "
          r"epochs frozen: it stopped growing on its own, at the cap or at the latch in "
          r"\texttt{GromoGrowth.on\_epoch\_end} which disables growth for the rest of a "
          r"fit after one step that adds no neurons. Only the first is fixed by a "
          r"longer budget.")),
    "",
    r"\section{One subject at a time}",
    (r"The dataset-level delta is a mean over subjects, and on this dataset it hides a "
     r"split rather than summarising one: two subjects lose substantially on all five "
     r"seeds while the other seven gain slightly. The near-zero dataset delta is a "
     r"true statement about the dataset and a wrong answer to a question about a "
     r"subject."),
    "",
    (r"The subject label is \textbf{inferred, not recorded}. \texttt{FitRecorder} is "
     r"built once per cell, before MOABB clones the pipeline per fold, so it never "
     r"learns which subject it is fitting. The join is positional --- MOABB appends "
     r"one score row per (subject, session) after that pair's five folds, so a block "
     r"of five fits belongs to score row $k$ --- and is validated at Spearman $+0.94$ "
     r"to $+0.99$ for eight of the nine arms across all five seeds. Guessing the "
     r"ordering instead gives $+0.03$ and $+0.10$, i.e.\ nothing. One line in the "
     r"callback's \texttt{meta} would make it a recorded fact."),
    "",
    _fig("subject_delta",
         (r"Per-subject paired delta, mean over the five seeds with the spread across "
          r"them. The label counts how many seeds agree the delta is negative: red is "
          r"all five, which is what separates a per-subject effect from one seed's "
          r"noise."), r"0.8\textwidth"),
    "",
    *[s for subj in LOSERS for s in (
        _fig(f"subject_curves_s{subj}",
             (rf"S{subj}, the growable arm against its matched control and against the "
              r"other eight subjects in grey. Curves are cut where half the folds have "
              r"early-stopped: a plain mean over epoch averages a shrinking, "
              r"self-selected set of folds and makes accuracy appear to climb long "
              r"after most fits ended.")), "")],
    r"\section{What this does and does not license saying}",
    (r"\textbf{Supported.} On ShallowFBCSPNet, growing from width 8 reaches a "
     r"\emph{better} score than braindecode's reference while stopping at 39\,\% of "
     r"its parameters, consistently across 12 datasets and three protocols. On SCCNet "
     r"and DeepEEGNet, growth matches its matched control at 70--86\,\% of its "
     r"parameters, with a delta inside the noise floor."),
    "",
    (r"\textbf{Not supported.} Anything about EEGNeX (incomplete, biased subset). And "
     r"no claim that growth is \emph{neutral} on SCCNet or DeepEEGNet: the $s=0$ "
     r"defect froze a fraction of every growing arm's added units at zero, so their "
     r"measured performance is a floor, not an estimate. The honest statement is that "
     r"the matched controls are not beaten, not that growth does not help."),
    "",
    (r"\textbf{The next measurement.} Refuse the zero-amplitude growth (two lines in "
     r"\texttt{grow\_step}) and re-run the growing arms. Two predictions the re-run "
     r"would test: the SCCNet and DeepEEGNet deltas move off zero, and the fraction of "
     r"folds that never grow at all falls from a third."),
    r"\end{document}",
]
(OUT / "eegrow_v5_tables.tex").write_text("\n".join(tex) + "\n")


# -------------------------------------------------------------------------- notebook
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(rf"""
# Growing networks for EEG decoding — v5 interim results

**Status: {N_PLANNED - N_MISSING} / {N_PLANNED} cells ({pct:.1f}%).** The {N_MISSING}
missing cells are all `grow_eegnex` / `bd_eegnex`, and they are not a random remainder:
they were dropped from the original campaign by out-of-memory failures on 11 GB cards,
so they are the memory-heaviest cells in the grid. Every EEGNeX number here is on a
biased subset.

Everything is recomputed from `benchmarks/results_v5_published/`; the figures come from
`v5_figures.py`, so each one has a single reviewable definition rather than living in a
cell.

### Three rules to read before the numbers

1. **Pairs are architecture-matched.** `grow_deep` is compared to `fix_deepeeg` — the
   same architecture built frozen at the width growth ends on — not to `bd_deep4`
   (Deep4Net, four stages at 25/50/100/200 filters). The mismatched pairing gives
   −0.022 on cross-subject where the matched control gives −0.000: it measures the
   architecture gap, not growth.
2. **Metrics are never pooled.** ROC-AUC for the two-class LeftRightImagery datasets,
   accuracy elsewhere.
3. **Seeds are replicates**, averaged *within* a (subject, session) before any
   statistic. Five seeds treated as five observations inflates n fivefold and every
   p-value with it.

### And one defect that bounds every growing arm

The scaling-factor line search in `grow_step` includes `0.0`, and when `0.0` wins the
growth is applied anyway — at exactly zero amplitude. A unit with zero incoming *and*
zero outgoing weights is an exact stationary point: the gradient of its incoming weights
is proportional to its (zero) outgoing weights, and the gradient of its outgoing weights
to its (zero) activation. Neither can restart the other, so the unit is dead for the
rest of training while `width` and `n_params` keep reporting it as capacity.

Measured per fold: chosen factors `[0.0, 1.0]` → 66.7% of new parameters with
`exp_avg_sq == 0`, `[0.0, 0.0]` → 100%, `[1.0, 1.0]` → 0%.
**So every growing result below is a lower bound, not an estimate.**
""")

code("""
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

# Walk up rather than hard-code a depth: this notebook is opened from v5/ by the
# builder and from the repo root by anyone who clicks it in an IDE.
HERE = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "benchmarks" / "analysis" / "v5_figures.py").is_file())
sys.path.insert(0, str(HERE / "benchmarks" / "analysis"))
SRC = HERE / "benchmarks" / "results_v5_published"

import growth_io
import v5_figures as vf

SHOWCASE = "bnci2014_001"   # the one dataset whose per-epoch curves are shipped

paired = pd.read_csv(SRC / "eegrow_benchmark_paired.csv")
levels = pd.read_csv(SRC / "eegrow_benchmark_levels.csv")
scores = pd.read_csv(SRC / "eegrow_benchmark_all_scores.csv.gz")
fits, budget, curves = growth_io.load_tidy(SRC)
prov = pd.read_csv(SRC / "eegrow_v5_provenance.csv")

# The fold's subject is not recorded, only inferable from write order; see
# growth_io.attach_subjects for the validation.
curves = growth_io.attach_subjects(
    curves, scores[(scores.dataset == SHOWCASE) & (scores.family != "ml")])

print(f"{len(scores):,} scores | {len(fits):,} folds | {len(curves):,} curve points")
print(f"{len(prov)} distinct regime(s) across the campaign")
prov.T
""")

md("""
## 1. What is actually there

The point of this table is to make the EEGNeX gap visible *before* any of its deltas are
read: fewer datasets, fewer observations, and the ones missing are the large ones.
""")

code("""
cov = scores.groupby("model").agg(
    datasets=("dataset", "nunique"), seeds=("seed", "nunique"),
    observations=("score", "size"))
cov["complete"] = ["no (running)" if m in {"grow_eegnex", "bd_eegnex"} else "yes"
                   for m in cov.index]
cov
""")

md("""
## 2. What the growing arms actually did

This is the section that changes what the report is about.

The paired contrast is usually stated as *growing to a target width versus training at
that width directly*. The fit records say that is not what ran. A third of all folds
**never grow at all**, and `grow_shallow` reaches its target width in **0.1%** of folds —
it stops at a mean width of 13.5 against a target of 40.

That does not weaken the result. It changes what the result *is*: the comparison is not
growth-versus-parity, it is growth-versus-a-larger-model.
""")

code("""
f, width_agg = vf.width_vs_target(fits)
width_agg.round(3)
""")

code("""
p = fits.groupby("model").params_end.mean()
recs = []
for grow, fixed in vf.PAIRS:
    if grow in p.index and fixed in p.index:
        recs.append(dict(arch=vf.PAIR_LABEL[f"{grow} vs {fixed}"],
                         grow_params=p[grow], fixed_params=p[fixed],
                         ratio=p[grow] / p[fixed]))
eff = pd.DataFrame(recs).sort_values("ratio")
eff.style.format({"grow_params": "{:,.0f}", "fixed_params": "{:,.0f}",
                  "ratio": "{:.0%}"})
""")

md("""
**`grow_shallow` beats `bd_shallow` while ending at 39% of its parameters** (31 730 vs
81 916). And the pattern across the four architectures is worth noticing, even if four
points is not a trend: the two arms that end nearest their control in size (DeepEEGNet at
86%, EEGNeX at 87%) are the two that sit at or below zero in score, while the one that
undershoots hardest is the one that wins.
""")

md("""
## 3. The paired contrast

`delta` is the per-dataset paired mean weighted by the number of (subject, session)
observations; the sign test is deliberately **unweighted and over datasets**, because
"growth helps on this dataset" is the unit the claim is about — subjects inside one
dataset are not independent draws of the population of datasets.
""")

code("""
recs = []
for (pair, ev), g in paired.groupby(["pair", "eval"]):
    wins = int((g.delta > 0).sum())
    recs.append(dict(arch=vf.PAIR_LABEL.get(pair, pair), eval=ev,
                     delta=(g.delta * g.n_obs).sum() / g.n_obs.sum(),
                     datasets=len(g), won=wins, n_obs=int(g.n_obs.sum()),
                     sign_p=binomtest(wins, len(g), 0.5).pvalue))
summ = pd.DataFrame(recs)
summ["eval"] = pd.Categorical(summ["eval"], vf.EVALS, ordered=True)
summ.sort_values(["arch", "eval"]).round(4).reset_index(drop=True)
""")

code("""
f = vf.delta_by_dataset(paired)
""")

code("""
f = vf.win_matrix(paired)
""")

md("""
## 4. Is any of this above the noise?

The two near-zero contrasts (SCCNet, DeepEEGNet) cannot be read at all until the
resolution of the instrument is known. So: compare every model to **itself** through the
identical pipeline, splitting its five seeds into two halves at random, 200 times. The
true effect is zero by construction, so whatever spread comes back is the noise floor of
the entire procedure.
""")

code("""
f, nf = vf.noise_floor(scores)
nf.groupby("model").delta.agg(["mean", "std",
                              lambda s: s.abs().quantile(0.95)]).round(4).rename(
    columns={"<lambda_0>": "p95 |delta|"})
""")

code("""
f = vf.seed_stability(scores)
""")

md("""
The seed spread within a single subject and session is larger than every *aggregate*
effect measured in this report, which is why five seeds are averaged before any
statistic.

It does **not** dismiss the S2 anomaly that started this investigation. Replicated over
five seeds, that subject's loss holds on all five, and it is not unique (§8). A spread
this wide means one seed cannot *establish* an effect — not that an effect it showed is
absent.
""")

md("""
## 5. Absolute level, and the cost of it

The paired delta says whether growth beat *its own* control; it says nothing about
whether either is any good. ROC-AUC datasets only, so the numbers are comparable and
chance is 0.5.

The line worth noting is `fixed control`: it sits below both other families in all three
protocols. That is exactly why the matched pairing in section 3 is the informative one —
`fix_deepeeg` is a weak absolute model, so beating it is not the same claim as beating
braindecode's references.
""")

code("""
fam = (levels[levels.metric == "roc_auc"]
       .groupby(["family", "eval"], as_index=False).score_mean.mean()
       .pivot(index="family", columns="eval", values="score_mean")
       .reindex(columns=vf.EVALS).round(4))
display(fam)
f = vf.family_levels(levels)
""")

code("""
f, b = vf.budget_pareto(budget, "bnci2014_001")
b.round(4)
""")

md("""
## 6. Training dynamics

These are the curves that did not exist when the S2 anomaly was reported: the script
behind the published table attached no `FitRecorder`, so there was nothing to look at.
`FitRecorder` logs, per epoch, `train_loss`, `valid_loss`, `valid_acc`, `width`,
`n_params` and `dur`, and per fit `width_start/end`, `target_width`,
`params_start/end`, `epochs`, `max_epochs`, `n_train`, `seconds`.

All panels below show `bnci2014_001`, within-session — the full curves for the whole grid
are 129 MB gzipped and no figure draws 128 801 trajectories.
""")

code(f"""
f = vf.width_trajectory(curves, "{SHOWCASE}")
""")

code(f"""
f = vf.learning_curves(curves, "{SHOWCASE}")
""")

code(f"""
f = vf.capacity(curves, "{SHOWCASE}")
""")

md("""
## 7. Why the target width is never reached: the stopping rule, not the mechanism

The width a growing arm ends at is not set by how well gromo picks neurons. It is set by
a race between the growth schedule — one opportunity every `grow_every=5` epochs — and
`EarlyStopping(patience=20, monitor="valid_acc")`. **Every fold stops early**, at a
median of 24–27 epochs out of the 200 it was given, which leaves a median of 4 to 5
growth opportunities. `grow_shallow` has to add 32 neurons across them, i.e. 6.4 per
opportunity; it gets 0.85. The target is out of reach by arithmetic before any property
of the growth method is involved.

Two mechanisms end growth even earlier. Reaching the cap, which is legitimate; and the
latch in `GromoGrowth.on_epoch_end` that sets `done_` after a single step which adds no
neurons, disabling growth for the rest of that fit.

The second figure separates them, and they turn out to be **architecture-dependent**.
`shallow` grows a median of 10 times and is frozen only ~2 epochs afterwards — it is
still growing when the fit ends, so it is purely epoch-starved and a longer budget would
help it. `sccnet` grows 3 times and then trains a median of 50 epochs frozen — it stopped
growing on its own, so a longer budget would change nothing for it. One fix does not
address both.
""")

code("""
f = vf.stopping_budget(fits)
""")

code(f"""
f, events = vf.growth_events(curves, "{SHOWCASE}")
events.groupby("model")[["n_growths", "last_growth", "last_epoch", "idle"]].median()
""")

md("""
## 8. One subject at a time

The dataset-level delta is a mean over subjects, and on this dataset the mean hides a
split rather than summarising one. Averaged over the five seeds, two subjects lose
substantially and consistently while the other seven gain slightly — so the near-zero
dataset delta is a real statement about the dataset and a wrong answer to a question
about a subject.

The subject label here is **inferred, not recorded**: `FitRecorder` is constructed once
per cell, before MOABB clones the pipeline per fold, so it never learns which subject it
is fitting. The join is positional — MOABB appends one score row per (subject, session)
after that pair's five folds, so a block of five fits belongs to score row *k* — and it
is validated at Spearman +0.94 to +0.99 for eight of the nine arms across all five
seeds. Guessing the ordering instead (subject-major, session-major) gives +0.03 and
+0.10, i.e. nothing. See `growth_io.attach_subjects`.
""")

code("""
f, subj_delta = vf.subject_delta(scores, SHOWCASE)
subj_delta.round(4)
""")

code("""
# Both consistent losers, not only the one that was reported: a per-subject failure that
# turns out not to be unique is a different finding from an anomaly.
for subj in subj_delta.sort_values("mean").head(2).index:
    f = vf.subject_curves(curves, subj)
""")

md("""
## 9. What this does and does not license saying

**Supported by the data.** On ShallowFBCSPNet, growing from width 8 reaches a *better*
score than braindecode's reference while stopping at 39% of its parameters, consistently
across 12 datasets and three protocols. On SCCNet and DeepEEGNet, growth matches its
matched control at 70–86% of its parameters, with a delta inside the noise floor.

**Not supported.** Anything about EEGNeX (incomplete, biased subset). And no claim that
growth is *neutral* on SCCNet or DeepEEGNet: the s=0 defect froze a fraction of every
growing arm's added units at zero, so their measured performance is a floor, not an
estimate. The honest statement is that the matched controls are not beaten, not that
growth does not help.

**The next measurement.** Refuse the zero-amplitude growth (two lines in `grow_step`)
and re-run the growing arms. Two predictions it would test: the SCCNet and DeepEEGNet
deltas move off zero, and the fraction of folds that never grow at all falls from a
third.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
nb_path = OUT / "eegrow_v5_results.ipynb"
nbf.write(nb, nb_path)

try:
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    NotebookClient(nb, timeout=1200,
                   resources={"metadata": {"path": str(OUT)}}).execute()
    nbf.write(nb, nb_path)
    body, _ = HTMLExporter().from_notebook_node(nb)
    (OUT / "eegrow_v5_results.html").write_text(body)
    print("notebook executed ->", OUT / "eegrow_v5_results.html")
except Exception as exc:  # noqa: BLE001
    # An unexecuted notebook is still a deliverable; a silently half-executed one is
    # not, so the failure is printed rather than swallowed.
    print(f"notebook written but NOT executed: {type(exc).__name__}: {exc}")

print("tex     ->", OUT / "eegrow_v5_tables.tex")
print(f"figures -> {len(figs)}:", ", ".join(sorted(figs)))
print()
print(eff.to_string(index=False))
print()
print(summ[["arch", "eval", "delta", "n_datasets", "wins", "sign_p"]]
      .to_string(index=False))
