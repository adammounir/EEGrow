"""Build the exploration notebook Sylvain asked for: champions, difficulty, crossed plots.

    uv run --with pandas,numpy,scipy,matplotlib,seaborn,nbformat,nbclient,nbconvert,ipykernel \
        python benchmarks/analysis/build_explore_report.py

Reads ``benchmarks/results_published/`` -- *not* ``results_v5_published/``. That choice
is the whole reason this script is separate from ``build_v5_report.py`` and it needs
stating plainly: the v5 campaign has no classical baseline. Its grid declares the six
``ml_*`` pipelines but none of them have run, so on v5 the words "family champion" can
only mean "best of four braindecode models", and the comparison that matters -- deep
against Riemannian geometry -- is not expressible. ``results_published`` carries all
fourteen models, so every question here is asked there.

The price of that choice, stated once and repeated in the report: this is the campaign
whose resampling is not uniformly verifiable. Some cells were killed when Margaret went
down and were re-run without the resample argument fully threaded, and unlike v5 this
directory ships no provenance row to check the sampling rate cell by cell. Everything
below is therefore a map of where to look, not a result to publish. Re-running the six
CPU-only ``ml_*`` configs under v5 is what turns it into one.

Writes benchmarks/analysis/explore/:

    figures/*.png                   one per question
    eegrow_exploration.ipynb/.html  the shareable lab notebook
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat as nbf  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore_figures as ef  # noqa: E402

ROOT = HERE.parents[1]
SRC = ROOT / "benchmarks" / "results_published"
OUT = HERE / "explore"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

scores = pd.read_csv(SRC / "eegrow_benchmark_all_scores.csv.gz")
tidy = ef.prepare(scores)
ORDER = ef.dataset_order(tidy)

# The evaluation-mode figure needs the six datasets that have a cross-session split
# first, each block kept in difficulty order so the two orders agree wherever they can.
XS = set(tidy[tidy["eval"] == "cross_session"].dataset.unique())
ORDER_XS = [d for d in ORDER if d in XS] + [d for d in ORDER if d not in XS]
N_XS = len(XS)

figs = {
    "champion_vs_mean": ef.champion_vs_mean(tidy, ORDER),
    "champion_lines": ef.champion_lines(tidy, ORDER),
    "champion_share": ef.champion_share(tidy),
    "per_dataset_all_models": ef.per_dataset_all_models(tidy, ORDER),
    "per_model_all_evals": ef.per_model_all_evals(tidy, ORDER_XS, N_XS),
    "difficulty": ef.difficulty(tidy, ORDER),
    "rank_heatmap": ef.rank_heatmap(tidy, ORDER),
    "eval_penalty": ef.eval_penalty(tidy, ORDER),
    "deep_vs_classical": ef.group_scatter(
        tidy, ORDER, a=["braindecode", "growing"], b=["riemann/csp"],
        label_a="deep model (braindecode or growing)", label_b="riemann/csp pipeline"),
}
figs["train_size_crossover"], _xr = ef.train_size_crossover(
    tidy, scores, ef.delta_champion(["braindecode", "growing"], ["riemann/csp"]),
    ylabel="best deep $-$ best riemann/csp (score $-$ chance)",
    title="Deep learning's deficit is a sample-size deficit")

# The second pass of figures, added once the v5 report needed them. They apply here
# unchanged -- this campaign is the one with a complete classical arm, so the
# subject-level and head-to-head views are if anything better supported here. What
# cannot be drawn on this grid is anything from the training records: `results_published`
# predates `FitRecorder`, so it has no per-epoch history at all and no curve figure of
# any kind can be reconstructed from its score CSVs.
figs["coverage_map"] = ef.coverage_map(scores, ORDER)
figs["per_dataset_all_models_xsubj"] = ef.per_dataset_all_models(
    tidy, ORDER, eval_="cross_subject")
figs["rank_heatmap_xsubj"] = ef.rank_heatmap(tidy, ORDER, eval_="cross_subject")
figs["mean_rank_cd"] = ef.mean_rank_cd(tidy)
figs["win_matrix"] = ef.win_matrix(tidy)
figs["chance_map"] = ef.chance_map(tidy, ORDER)
figs["seed_noise"] = ef.seed_noise(scores, ORDER)
figs["subject_spread"] = ef.subject_spread(tidy, ORDER)
figs["subject_delta_deep"] = ef.subject_delta(
    tidy, ORDER, a=["braindecode", "growing"], b=["riemann/csp"],
    label=r"per-subject $\Delta$ (deep champion $-$ classical champion)")
figs["per_model_train_size"] = ef.per_model_train_size(tidy, scores)
figs["cost_vs_score"] = ef.cost_vs_score(scores, tidy)

figs = {k: v for k, v in figs.items() if v is not None}
for name, fig in figs.items():
    fig.savefig(FIG / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- key numbers
def champion_table(t: pd.DataFrame) -> pd.DataFrame:
    m = t.groupby(["eval", "dataset", "family", "model"]).above.mean().reset_index()
    best = m.loc[m.groupby(["eval", "dataset", "family"]).above.idxmax()]
    return best.pivot_table(index=["eval", "dataset"], columns="family",
                            values=["model", "above"], aggfunc="first")


champs = champion_table(tidy)
w = tidy[tidy["eval"] == "within_session"]
per_fam = w.groupby(["dataset", "family"]).above.mean().unstack()
best_per_fam = (w.groupby(["dataset", "family", "model"]).above.mean()
                .groupby(["dataset", "family"]).max().unstack())
gap = (best_per_fam - per_fam)  # what the family mean hides
cls_wins = int((best_per_fam["riemann/csp"] >
                best_per_fam[["braindecode", "growing"]].max(axis=1)).sum())
n_ds = len(best_per_fam)
# "At chance" = the family's best model, averaged over subjects, within one CI of zero.
ci = (w.groupby(["dataset", "family", "model"]).above
      .agg(["mean", "std", "size"]).reset_index())
ci["ci"] = 1.96 * ci["std"] / np.sqrt(ci["size"])
bd_best = ci[ci.family != "riemann/csp"].loc[
    ci[ci.family != "riemann/csp"].groupby("dataset")["mean"].idxmax()]
at_chance = bd_best[bd_best["mean"] - bd_best["ci"] <= 0]

print(f"riemann/csp champion beats every deep model on {cls_wins}/{n_ds} datasets "
      f"(within_session)")
print(f"largest champion-minus-family-mean gap: "
      f"{gap.max().max():.3f} ({gap.stack().idxmax()})")
print("deep at chance (best deep model's CI covers 0):",
      ", ".join(at_chance.dataset) or "none")


# ------------------------------------------------------------------------- notebook
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Reading the 14-model grid — champions, difficulty, and the three evaluation modes

**What this is.** The exploration pass: for each dataset, which model actually wins
inside each family, how hard each dataset is, and how the three evaluation modes differ.
Nine figures, all drawn from one tidy frame, all sharing one dataset order.

**Which campaign, and why it matters.** This reads `results_published`, the 14-model
grid — six `riemann/csp` pipelines, four `braindecode`, four `growing`, 12 datasets,
3 evaluation modes, 5 seeds. Not the v5 campaign: v5 declares the six `ml_*` configs in
its grid but has run none of them, so on v5 "family champion" can only mean "best of
four braindecode models" and the comparison that carries the most information — deep
against Riemannian geometry — cannot be asked at all.

**The caveat that comes with that.** `results_published` is the campaign whose
resampling is not uniformly verifiable: cells killed when Margaret went down were
re-run without the resample argument fully threaded, and this directory ships no
provenance row to audit the sampling rate cell by cell. So read every number below as
*where to look*, not as a result. The six `ml_*` configs are CPU-only and cheap; running
them under v5 is what would make this publishable.

**Two conventions, both to stop a misleading plot.**

1. **Score minus chance, everywhere.** The grid mixes metrics — six datasets scored with
   ROC-AUC (chance 0.5), six with accuracy (chance 1/n_classes, so 0.25 on the
   four-class ones). On a raw-score axis bnci2014_001 looks worse than shin2017a when it
   is in fact far easier. Subtracting chance is the minimum that makes a cross-dataset
   axis mean anything; it still compresses the easy end against the 1.0 ceiling.
2. **One dataset order, hardest first, in every figure.** So that flipping between two
   panels is a comparison. A dataset low in every panel is a fact about the dataset.
""")

code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path.cwd()
sys.path.insert(0, str(HERE.parent))
import explore_figures as ef

SRC = HERE.parents[1] / "results_published"   # notebook runs with cwd = analysis/explore
scores = pd.read_csv(SRC / "eegrow_benchmark_all_scores.csv.gz")
tidy = ef.prepare(scores)
ORDER = ef.dataset_order(tidy)
XS = set(tidy[tidy["eval"] == "cross_session"].dataset.unique())
ORDER_XS = [d for d in ORDER if d in XS] + [d for d in ORDER if d not in XS]
N_XS = len(XS)

print(f"{len(scores):,} score rows | {tidy.model.nunique()} models | "
      f"{tidy.dataset.nunique()} datasets | {tidy.subject.nunique()} subject ids")
print("families:", tidy.groupby("family").model.unique().to_dict())
print("dataset order (hardest first):", ORDER)
""")

md(r"""
## 1. What a family average hides

The first figure is the one that replaces a per-family bar chart. A family mean answers
"how good is the average member", which is a question nobody has: if `riemann/csp` holds
one pipeline that wins and five that are mediocre, the mean reports mediocre and the
finding disappears. Bar = family mean, marker = that family's best model on that
dataset, labelled. **The gap between the two is the amount the mean was hiding.**
""")

code("""
f = ef.champion_vs_mean(tidy, ORDER); plt.show()
""")

code("""
# The gap, as a number: best member minus family mean, within_session.
w = tidy[tidy["eval"] == "within_session"]
fam_mean = w.groupby(["dataset","family"]).above.mean().unstack()
fam_best = (w.groupby(["dataset","family","model"]).above.mean()
            .groupby(["dataset","family"]).max().unstack())
gap = (fam_best - fam_mean).loc[ORDER]
print("champion minus family mean (within_session):")
print(gap.round(3).to_string())
print()
print("worst case:", gap.stack().idxmax(), "->", round(gap.stack().max(), 3),
      "-- reporting the mean there understates the family by that much")
""")

md(r"""
## 2. Who wins where

Same data, family means dropped, so the three champions read as three lines. Then: is
the champion the *same model* everywhere? A champion that changes with every dataset is
not a champion, it is a maximum over 14 noisy numbers.
""")

code("""
f = ef.champion_lines(tidy, ORDER); plt.show()
""")

code("""
f = ef.champion_share(tidy); plt.show()
""")

code("""
m = tidy.groupby(["eval","dataset","family","model"]).above.mean().reset_index()
best = m.loc[m.groupby(["eval","dataset","family"]).above.idxmax()]
print("how often each model is its family's champion (36 dataset x eval cells / family):")
print(best.groupby(["family","model"]).size().sort_values(ascending=False).to_string())
""")

md(r"""
## 3. The crossed pair

Two ways through the same table, which is the point: one panel per dataset showing all
14 models, then one panel per model showing all 12 datasets. The second is built to the
shape asked for — the six datasets that *have* a cross-session split come first on the
x axis, so the cross-session line stops halfway across instead of breaking into six
disconnected segments, and the dashed rule marks where it must stop.
""")

code("""
f = ef.per_dataset_all_models(tidy, ORDER); plt.show()
""")

code("""
f = ef.per_model_all_evals(tidy, ORDER_XS, N_XS); plt.show()
""")

md(r"""
## 4. Easy and hard

The operational question: *name me an easy dataset and a hard one*. Left panel answers
it two ways — the mean over all 14 models, and the best single model, because a dataset
is as easy as its easiest win and not as its average entrant. Right panel is the
material the answer is made of, since a difficulty ranking with no mechanism next to it
is a list to memorise rather than something to reason with.
""")

code("""
f = ef.difficulty(tidy, ORDER); plt.show()
""")

code("""
# Where does deep learning fail outright? "At chance" = the best of the eight deep
# models, averaged over subjects, with a 95% CI that still covers zero.
ci = (w.groupby(["dataset","family","model"]).above.agg(["mean","std","size"])
      .reset_index())
ci["ci"] = 1.96 * ci["std"] / np.sqrt(ci["size"])
deep = ci[ci.family != "riemann/csp"]
bd_best = deep.loc[deep.groupby("dataset")["mean"].idxmax()].set_index("dataset")
cls = ci[ci.family == "riemann/csp"]
cls_best = cls.loc[cls.groupby("dataset")["mean"].idxmax()].set_index("dataset")
cmp = pd.DataFrame({
    "best_deep": bd_best["model"], "deep": bd_best["mean"].round(3),
    "deep_ci": bd_best["ci"].round(3),
    "best_classical": cls_best["model"], "classical": cls_best["mean"].round(3),
}).loc[ORDER]
cmp["deep_at_chance"] = cmp.deep - cmp.deep_ci <= 0
print(cmp.to_string())
""")

md(r"""
## 5. Ranks, so the datasets stop confounding the models

Scores are not on one scale even after subtracting chance: a 0.02 gap is decisive on
shin2017a and noise on schirrmeister2017. Ranking *inside* a dataset removes that
dataset's difficulty entirely, which is what makes the row means comparable — and the
row means are the answer to "if I only get to beat one model, which one".
""")

code("""
f = ef.rank_heatmap(tidy, ORDER); plt.show()
""")

md(r"""
## 6. What generalising costs

Within-session, cross-session and cross-subject are three different asks. The question
worth asking is whether the cost of each is a property of the dataset or of the model
family — plotted as a delta against each dataset's own within-session result, so the
dataset's difficulty cancels.
""")

code("""
f = ef.eval_penalty(tidy, ORDER); plt.show()
""")

code("""
pen = (tidy.groupby(["eval","family"]).above.mean().unstack(0))
print("mean above-chance by family and eval:"); print(pen.round(3).to_string())
""")

md(r"""
## 7. The whole thing as one scatter

Best deep model against best classical pipeline, one point per dataset and evaluation
mode. The diagonal is parity; below it, no braindecode or growing model reached what a
Riemann pipeline reached on that dataset.
""")

code("""
f = ef.group_scatter(tidy, ORDER, a=["braindecode","growing"], b=["riemann/csp"],
                     label_a="deep model (braindecode or growing)",
                     label_b="riemann/csp pipeline"); plt.show()
""")

code("""
for ev in ef.EVALS:
    s = tidy[tidy["eval"] == ev]
    m = s.groupby(["dataset","family","model"]).above.mean().reset_index()
    cls = m[m.family == "riemann/csp"].groupby("dataset").above.max()
    dl  = m[m.family != "riemann/csp"].groupby("dataset").above.max()
    c = cls.index.intersection(dl.index)
    print(f"{ev:15s} deep champion beats classical on {int((dl[c]>cls[c]).sum())}"
          f"/{len(c)} datasets | mean delta {float((dl[c]-cls[c]).mean()):+.3f}")
""")

md(r"""
## 8. Why the sign flips — and it is not the evaluation mode

The scatter above flips sign between the modes: the classical champion wins 10 of 12
datasets within-session and loses 9 of 12 cross-subject. Read as an evaluation-mode
effect that is a puzzle. Read against how many trials each fit actually got, it is not
one — the three modes differ by roughly **thirty times** in training data, because
within-session gives a fit one session of one subject (20 trials on shin2017a) while
cross-subject gives it every other subject (12 444 on schirrmeister2017).

The one thing that stops this from being a restatement of the mode label: **the
correlation survives inside within-session alone**, where the mode is held constant.
""")

code("""
f, r = ef.train_size_crossover(
    tidy, scores, ef.delta_champion(["braindecode","growing"], ["riemann/csp"]),
    ylabel="best deep $-$ best riemann/csp (score $-$ chance)",
    title="Deep learning's deficit is a sample-size deficit")
plt.show()
""")

code("""
from scipy.stats import spearmanr
rows = []
for ev in ef.EVALS:
    s = tidy[tidy["eval"] == ev]
    m = s.groupby(["dataset","family","model"]).above.mean().reset_index()
    cls = m[m.family == "riemann/csp"].groupby("dataset").above.max()
    dl  = m[m.family != "riemann/csp"].groupby("dataset").above.max()
    n = scores[scores["eval"] == ev].groupby("dataset").samples.median()
    for d in cls.index.intersection(dl.index):
        rows.append(dict(ev=ev, dataset=d, delta=dl[d]-cls[d], n=n[d]))
r = pd.DataFrame(rows)
print("all 30 dataset x mode cells: rho = %+.2f, p = %.2g" %
      spearmanr(np.log10(r.n), r.delta))
for ev in ef.EVALS:
    h = r[r.ev == ev]
    if len(h) > 3:
        print("  %-15s rho = %+.2f, p = %.3f  (mode held constant, n=%d)" %
              ((ev,) + tuple(spearmanr(np.log10(h.n), h.delta)) + (len(h),)))
""")

md(r"""
## 9. The second pass — subjects, head to head, and what it cost

Everything above compares means. These are the views that can contradict a mean: the
per-subject spread behind each family champion, the pairwise win rate (which never
leaves a (dataset, subject) pair, so it touches neither the chance subtraction nor the
mixed metric), the critical difference that says which of these gaps twelve datasets can
actually resolve, the seed noise floor every delta sits on, and the compute the scores
were bought with.

Note what is missing and why: this campaign predates `FitRecorder`, so it has **no
per-epoch training records**. Learning curves, stopping-epoch diagnostics and growth
trajectories exist only in the v5 report (`analysis/explore_v5/`) and cannot be
reconstructed here from the score CSVs.
""")

code("""
f = ef.coverage_map(scores, ORDER); plt.show()
""")

code("""
f = ef.subject_spread(tidy, ORDER); plt.show()
""")

code("""
f = ef.subject_delta(tidy, ORDER, a=["braindecode","growing"], b=["riemann/csp"],
                     label=r"per-subject $\\Delta$ (deep $-$ classical champion)")
plt.show()
""")

code("""
f = ef.win_matrix(tidy); plt.show()
""")

code("""
f = ef.mean_rank_cd(tidy); plt.show()
""")

code("""
f = ef.chance_map(tidy, ORDER); plt.show()
""")

code("""
f = ef.seed_noise(scores, ORDER); plt.show()
""")

code("""
f = ef.rank_heatmap(tidy, ORDER, eval_="cross_subject"); plt.show()
""")

code("""
f = ef.per_dataset_all_models(tidy, ORDER, eval_="cross_subject"); plt.show()
""")

code("""
f = ef.per_model_train_size(tidy, scores); plt.show()
""")

code("""
f = ef.cost_vs_score(scores, tidy); plt.show()
""")

md(r"""
## What I take from this, and what I refuse to take

Written as claims with the measurement attached, so each one can be attacked.

**The family average was hiding a real gap.** Champion minus family mean reaches ~0.17
above chance on zhou2016 — where the growing family's mean is roughly half its
champion. Any sentence of the form "family X scores Y" in earlier notes should be read
as a statement about the average member, which is not a thing anybody deploys.

**The champion is stable enough to name.** `ts_lr` is the classical champion on most
dataset-eval cells and `bd_sccnet` the braindecode champion on most, which is what
licenses "the baseline to beat is `ts_lr`" as an instruction rather than an artefact of
a maximum over 14 numbers.

**Within-session, deep learning does not merely lose, it fails outright on two
datasets.** On physionetmi and shin2017a the best of the eight deep models has a
confidence interval that still covers chance, while a Riemann pipeline sits +0.18 clear
of it. That is a different statement from "deep is a bit behind". Those two datasets give
a fit **45 and 20 trials** respectively.

**The sign flips with the evaluation mode, and the reason is sample size, not the mode.**
The classical champion wins 10 of 12 datasets within-session (mean −0.095 for deep) and
*loses* 9 of 12 cross-subject (+0.037). Plotted against trials available per fit, the
deep-minus-classical delta correlates at Spearman ρ = +0.82 (p = 2e-8, 30 dataset-mode
cells), with both signs occurring only in a band around 160–1700 trials. The correlation
holds **inside within-session alone** (ρ = +0.73, p = 0.007, 12 datasets), where the mode
is constant — so it is a data-volume effect that the mode happens to be a proxy for, not
the mode itself. This is the sentence I would actually defend out of this whole pass, and
it reframes the alignment work: alignment buys effective sample size, and the figure says
exactly which regime has room for that to show.

**Difficulty is a property of the dataset and it is stable across evaluation modes.**
The order barely moves between within-session, cross-session and cross-subject, which
is what makes a single fixed order legitimate — and gives a defensible answer to "give
me one easy dataset and one hard one": schirrmeister2017 easy, shin2017a hard, and
bnci2014_001 as the mid-range default it is already being used as.

**What I refuse to conclude.** Nothing here is a publishable comparison. The resampling
of this campaign is not verifiable cell by cell, and a sampling-rate difference between
arms is exactly the kind of confound that would produce "deep fails on the hard
datasets". The next measurement is not another plot: it is the six CPU-only `ml_*`
configs run under v5, whose provenance row pins sfreq at 250 Hz for every cell. If the
picture survives that, it is a paper section; if it does not, this notebook is the
record of why the earlier campaign could not be trusted.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
nb_path = OUT / "eegrow_exploration.ipynb"
nbf.write(nb, nb_path)

try:
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    NotebookClient(nb, timeout=1800,
                   resources={"metadata": {"path": str(OUT)}}).execute()
    nbf.write(nb, nb_path)
    body, _ = HTMLExporter().from_notebook_node(nb)
    (OUT / "eegrow_exploration.html").write_text(body)
    print("notebook executed ->", OUT / "eegrow_exploration.html")
except Exception as exc:  # noqa: BLE001
    print(f"notebook written but NOT executed: {type(exc).__name__}: {exc}")

print(f"figures -> {len(figs)}:", ", ".join(sorted(figs)))
