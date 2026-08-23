"""The full exploration of the v5 campaign — complete: 450 cells, 1848 runs.

    uv run --with pandas,numpy,scipy,matplotlib,seaborn,nbformat,nbclient,nbconvert,ipykernel \
        python benchmarks/analysis/build_explore_v5_report.py

**What changed on 2026-08-19, and it changes the whole report.** v5 used to declare the
six ``ml_*`` classical pipelines and have run none of them, which made the one
comparison the exploration turns on -- deep learning against Riemannian geometry --
inexpressible on the only campaign with a provenance row. That arm has since run
(``slurm/submit_ml_v5.sh``, 498 cells), so this report asks the deep-versus-classical
question on v5 itself rather than borrowing the older grid for it.

Three things still have to be read carefully, and every figure that is affected says so:

* **Coverage is uniform against the plan, and the plan is not a rectangle.** The nine
  deep arms carry five seeds everywhere; the six classical pipelines carry five on
  within-session and one on the two deterministic modes, because none of them takes a
  ``random_state`` and leave-one-session-out / leave-one-subject-out enumerate the data
  instead of drawing a split -- verified on 516 control groups, seed 1 against seed 0,
  max |delta| = 0.0. `coverage_map` is still the first figure, but it colours against
  that plan, so a single seed there reads as complete rather than as a hole.
* **`fixed control` is a family of one.** Its family mean equals its champion by
  construction; the bar and the marker coincide, and that is an artefact of the grid.
* **A champion of a larger family wins for free.** A maximum over more candidates is
  larger in expectation. That bias is negligible against the deep-versus-classical gap
  (0.1-0.3) and decisive against the growing-versus-fixed one (~0.01), so the first uses
  champions and the second uses the four architecture-matched pairs.

Writes benchmarks/analysis/explore_v5/ -- 39 figures, one tidy frame, one dataset order.
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
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore_curves as ec  # noqa: E402
import explore_figures as ef  # noqa: E402

ROOT = HERE.parents[1]
SRC = ROOT / "benchmarks" / "results_v5_published"
OLD = ROOT / "benchmarks" / "results_published"
OUT = HERE / "explore_v5"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# `grow_deep` pairs with `fix_deepeeg`, never with `bd_deep4`: that mismatched pairing
# measures the architecture gap, not the growth.
PAIRS = [("grow_shallow", "bd_shallow"), ("grow_sccnet", "bd_sccnet"),
         ("grow_eegnex", "bd_eegnex"), ("grow_deep", "fix_deepeeg")]
FIXED = ["braindecode", "fixed control"]
GROWING = ["growing"]
DEEP = ["braindecode", "fixed control", "growing"]
CLASSICAL = ["riemann/csp"]
SHOWCASE = "bnci2014_001"

scores = pd.read_csv(SRC / "eegrow_benchmark_all_scores.csv.gz")
tidy = ef.prepare(scores)
ORDER = ef.dataset_order(tidy)
XS = set(tidy[tidy["eval"] == "cross_session"].dataset.unique())
ORDER_XS = [d for d in ORDER if d in XS] + [d for d in ORDER if d not in XS]
N_XS = len(XS)

old_tidy = ef.prepare(pd.read_csv(OLD / "eegrow_benchmark_all_scores.csv.gz"))

fits = pd.read_csv(SRC / "eegrow_v5_fits.csv.gz")
budget = pd.read_csv(SRC / "eegrow_v5_budget.csv.gz")
cmean = pd.read_csv(SRC / "eegrow_v5_curves_mean.csv.gz")
fsum = pd.read_csv(SRC / "eegrow_v5_fold_summary.csv.gz")
# The curve frames only exist for the deep arms -- the classical pipelines have neither
# epochs nor a width, so `FitRecorder` is never attached to them. Ordering the curve
# panels by the score-table difficulty keeps them flippable against the score figures
# anyway; datasets with no curves simply have no lines.
ORDER_C = [d for d in ORDER if d in set(cmean.dataset.unique())]
# Chance for the *internal* validation accuracy the curves plot is always 1/n_classes,
# never 0.5: that split is scored by accuracy even on the datasets MOABB reports with
# ROC-AUC, so reusing the score table's chance column here would put the rule in the
# wrong place on the four-class datasets.
CHANCE = (1.0 / scores.groupby("dataset").n_classes.first()).to_dict()

HAS_ML = "riemann/csp" in set(tidy.family)

figs: dict = {}

# ------------------------------------------------------------------ 0. what exists yet
figs["coverage_map"] = ef.coverage_map(scores, ORDER)

# ---------------------------------------------------------------- 1. family champions
figs["champion_vs_mean"] = ef.champion_vs_mean(tidy, ORDER)
figs["champion_lines"] = ef.champion_lines(tidy, ORDER)
figs["champion_share"] = ef.champion_share(tidy)

# ------------------------------------------------------------------- 2. crossed plots
figs["per_dataset_all_models"] = ef.per_dataset_all_models(tidy, ORDER)
figs["per_dataset_all_models_xsubj"] = ef.per_dataset_all_models(
    tidy, ORDER, eval_="cross_subject")
figs["per_model_all_evals"] = ef.per_model_all_evals(tidy, ORDER_XS, N_XS)

# ------------------------------------------------------- 3. difficulty and reliability
figs["difficulty"] = ef.difficulty(tidy, ORDER)
figs["rank_heatmap"] = ef.rank_heatmap(tidy, ORDER)
figs["rank_heatmap_xsubj"] = ef.rank_heatmap(tidy, ORDER, eval_="cross_subject")
figs["mean_rank_cd"] = ef.mean_rank_cd(tidy)
figs["win_matrix"] = ef.win_matrix(tidy)
figs["chance_map"] = ef.chance_map(tidy, ORDER)
figs["seed_noise"] = ef.seed_noise(scores, ORDER)
figs["eval_penalty"] = ef.eval_penalty(tidy, ORDER)

# --------------------------------------------------------------- 4. the two contrasts
figs["subject_spread"] = ef.subject_spread(tidy, ORDER)
figs["growing_vs_fixed"] = ef.group_scatter(
    tidy, ORDER, a=GROWING, b=FIXED,
    label_a="growing model", label_b="fixed model")
figs["subject_delta_growing"] = ef.subject_delta(
    tidy, ORDER, a=GROWING, b=FIXED,
    label=r"per-subject $\Delta$ (growing champion $-$ fixed champion)")
figs["train_size_growing"], XR = ef.train_size_crossover(
    tidy, scores, ef.delta_pairs(PAIRS),
    ylabel=r"mean matched-pair $\Delta$ (grow $-$ fixed)",
    title="Does growing pay off more when data is scarce?")

if HAS_ML:
    figs["deep_vs_classical"] = ef.group_scatter(
        tidy, ORDER, a=DEEP, b=CLASSICAL,
        label_a="deep model (braindecode, fixed or growing)",
        label_b="riemann/csp pipeline")
    figs["subject_delta_deep"] = ef.subject_delta(
        tidy, ORDER, a=DEEP, b=CLASSICAL,
        label=r"per-subject $\Delta$ (deep champion $-$ classical champion)")
    figs["train_size_deep"], XD = ef.train_size_crossover(
        tidy, scores, ef.delta_champion(DEEP, CLASSICAL),
        ylabel="best deep $-$ best riemann/csp (score $-$ chance)",
        title="Is deep learning's deficit a sample-size deficit here too?")
else:
    XD = None

figs["per_model_train_size"] = ef.per_model_train_size(tidy, scores)
figs["cost_vs_score"] = ef.cost_vs_score(scores, tidy)
figs["campaign_agreement"], RESID = ef.campaign_agreement(
    old_tidy, tidy, label_a="results_published", label_b="results_v5")

# ------------------------------------------------------------- 5. training dynamics
figs["learning_curves"] = ec.learning_curves(cmean, ORDER_C, chance=CHANCE)
figs["learning_curves_xsubj"] = ec.learning_curves(
    cmean, ORDER_C, eval_="cross_subject", chance=CHANCE)
figs["loss_curves"] = ec.loss_curves(cmean, ORDER_C)
figs["curve_seed_band"] = ec.curve_seed_band(cmean, SHOWCASE)
figs["paired_curves"] = ec.paired_curves(cmean, PAIRS, ORDER_C[:6])
figs["stopping_epoch"] = ec.stopping_epoch(fsum, fits, ORDER_C)
figs["selected_epoch_vs_data"] = ec.selected_epoch_vs_data(fsum, fits)
figs["selection_degeneracy"] = ec.selection_degeneracy(cmean, fits, ORDER_C)
figs["overfit_gap"] = ec.overfit_gap(fsum, ORDER_C)
figs["valid_vs_test"] = ec.valid_vs_test(fsum, tidy)
figs["width_trajectory"] = ec.width_trajectory(cmean, ORDER_C)
figs["width_reached"] = ec.width_reached(fits, ORDER_C)
figs["budget_pareto"] = ec.budget_pareto(budget, tidy)
figs["cost_per_epoch"] = ec.cost_per_epoch(fits, ORDER_C)

figs = {k: v for k, v in figs.items() if v is not None}
for name, fig in figs.items():
    fig.savefig(FIG / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------- key numbers
cov = tidy.groupby(["model", "eval"]).dataset.nunique().unstack()
w = tidy[tidy["eval"] == "within_session"]
gap = ((w.groupby(["dataset", "family", "model"]).above.mean()
        .groupby(["dataset", "family"]).max().unstack())
       - w.groupby(["dataset", "family"]).above.mean().unstack())
rho_g, p_g = spearmanr(np.log10(XR.n), XR.delta)
# Minimum effect this test could have caught, so a null is reported as a bound and not
# as an absence. Fisher-z, two-sided alpha .05, power .80.
mde = float(np.tanh((1.959964 + 0.8416212) / np.sqrt(len(XR) - 3)))

print("coverage (datasets per model x eval):")
print(cov.to_string())
print(f"\nlargest champion-minus-family-mean gap: {gap.max().max():.3f} "
      f"({gap.stack().idxmax()})")
print(f"paired grow-fixed delta vs train size: rho={rho_g:+.2f} p={p_g:.2g} "
      f"n={len(XR)} | MDE at 80% power = {mde:.2f}")
if XD is not None:
    rho_d, p_d = spearmanr(np.log10(XD.n), XD.delta)
    print(f"deep-minus-classical delta vs train size: rho={rho_d:+.2f} p={p_d:.2g} "
          f"n={len(XD)}")
print(f"campaign agreement: median |diff| {RESID.median():.4f}, "
      f"p95 {RESID.quantile(0.95):.4f}")

# --------------------------------------------------------------------------- notebook
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Reading the v5 grid — who wins, where, why, and what it cost

**What this is.** The full exploration pass on **the completed v5 campaign**:
`results_v5_published`. Fifteen models in four families — six `riemann/csp` classical
pipelines, four `braindecode` reference nets, one `fixed control` (`fix_deepeeg`), four
`growing` — over 12 datasets and 3 evaluation modes, 450 cells and 1848 runs, plus the
per-epoch training records for all 140 490 deep folds.

**Two things that got the campaign to this state.** v5 used to have no classical arm at
all: the six `ml_*` configs were declared in the grid and none had run, so the comparison
every one of these figures turns on was inexpressible on the only campaign with a
provenance row; that arm has since run (498 cells). And `EEGClassifier` defaults to
`iterator_train__drop_last=True`, which at `batch_size=64` gave **zero** training batches
on any cell whose internal 80 % split held fewer than 64 trials — 180 cells trained for
exactly no steps and published their initialisation, silently, with a well-formed CSV.
Those cells were re-run after the fix. No `train_loss` is NaN in these frames any more,
and the growing arms grow on 100 % of folds instead of two thirds.

**Three things that change how the figures must be read.**

1. **Coverage is complete, but the plan is not a rectangle.** Every one of the 450 cells
   is at its planned seed count. The nine deep arms carry five seeds in all three modes;
   the six classical pipelines carry five within-session and one on the two deterministic
   modes, where they take no `random_state` and the folds are an enumeration rather than
   a draw. Figure 0 colours against that plan, so a single seed there reads as complete.
2. **`fixed control` is a family of one.** Its family mean equals its champion by
   construction — the bar and the marker coincide, and that coincidence is an artefact.
3. **A champion of a larger family wins for free.** A maximum over more candidates is
   larger in expectation. Negligible against the deep-versus-classical gap (0.1–0.3),
   decisive against the growing-versus-fixed one (~0.01) — so the first is drawn with
   champions and the second with the four architecture-matched pairs.

**Two conventions, both to stop a misleading plot.** Score minus chance everywhere — the
grid mixes metrics, six datasets on ROC-AUC (chance 0.5) and six on accuracy (chance
1/n_classes) — and one fixed dataset order, hardest first, in every figure, so that
flipping between two of them is a comparison.
""")

code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path.cwd()                      # notebook runs with cwd = analysis/explore_v5
sys.path.insert(0, str(HERE.parent))
import explore_figures as ef
import explore_curves as ec

SRC = HERE.parents[1] / "results_v5_published"
OLD = HERE.parents[1] / "results_published"
PAIRS = [("grow_shallow","bd_shallow"), ("grow_sccnet","bd_sccnet"),
         ("grow_eegnex","bd_eegnex"), ("grow_deep","fix_deepeeg")]
FIXED, GROWING = ["braindecode","fixed control"], ["growing"]
DEEP, CLASSICAL = ["braindecode","fixed control","growing"], ["riemann/csp"]
SHOWCASE = "bnci2014_001"

scores = pd.read_csv(SRC / "eegrow_benchmark_all_scores.csv.gz")
tidy = ef.prepare(scores)
old_tidy = ef.prepare(pd.read_csv(OLD / "eegrow_benchmark_all_scores.csv.gz"))
ORDER = ef.dataset_order(tidy)
XS = set(tidy[tidy["eval"] == "cross_session"].dataset.unique())
ORDER_XS = [d for d in ORDER if d in XS] + [d for d in ORDER if d not in XS]
N_XS = len(XS)

fits   = pd.read_csv(SRC / "eegrow_v5_fits.csv.gz")
budget = pd.read_csv(SRC / "eegrow_v5_budget.csv.gz")
cmean  = pd.read_csv(SRC / "eegrow_v5_curves_mean.csv.gz")
fsum   = pd.read_csv(SRC / "eegrow_v5_fold_summary.csv.gz")
ORDER_C = [d for d in ORDER if d in set(cmean.dataset.unique())]

print(f"{len(scores):,} score rows | {tidy.model.nunique()} models | "
      f"{tidy.dataset.nunique()} datasets")
print("families:", ef.families_in(tidy))
print("dataset order (hardest first):", ORDER)
print()
print("provenance (the regimes every cell actually ran under):")
print(pd.read_csv(SRC / "eegrow_v5_provenance.csv").to_string(index=False))
""")

md(r"""
## 0. What has actually been measured

The campaign is complete: 1848 cells, every one of them at its planned seed count. The
plan is not a rectangle, and the figure now says so in colour rather than leaving it to
this caption. The nine deep arms carry five seeds in all three modes. The six classical
pipelines carry five on within-session and **one** on cross-session and cross-subject,
and that single seed is the whole plan, not a hole: not one of `CSP`+LDA, `SVC(rbf)`,
`TangentSpace`+LR, `MDM` or `FgMDM` takes a `random_state`, and leave-one-session-out
and leave-one-subject-out enumerate the data rather than drawing a split, so the seed
has nothing left to move. Running them five times would be the same number five times.

That is a claim about the code, so the grid carries a control: seed 1 on five cheap
(mode, dataset) combinations of the deterministic modes -- the cells reading **2**
below. Over the 516 (mode, dataset, model, subject, session) groups they cover, seed 1
reproduces seed 0 with a maximum absolute deviation of **0.0**. The argument holds under
moabb 1.5.0, and the 80 % of that arm's compute it saves bought `schirrmeister2017`
cross-subject, which is 50 h of CPU per cell.

This is still the first figure on purpose: it is where you look when a model-averaged
number changes between builds.
""")

code("""
f = ef.coverage_map(scores, ORDER); plt.show()
""")

code("""
print("datasets per model x eval (12 within, 12 cross-subject, 6 cross-session):")
print(tidy.groupby(["model","eval"]).dataset.nunique().unstack().fillna(0)
      .astype(int).to_string())
print()
# Completeness against the plan, not against a flat 5 -- see above. Counting cells that
# hold five seeds would report the classical arm's deterministic modes as 80% missing.
n_seed = scores.groupby(["eval","dataset","model"]).seed.nunique()
ml = set(scores.loc[scores.family == "riemann/csp", "model"])
ix = n_seed.index.to_frame(index=False)
plan = np.where(ix["eval"].isin(["cross_session","cross_subject"]) & ix.model.isin(ml),
                1, ef.planned_seeds(scores))
print(f"cells: {len(n_seed)}, runs: {int(n_seed.sum())}")
print(f"at or above plan: {int((n_seed.values >= plan).sum())}/{len(n_seed)}, "
      f"short of plan: {int((n_seed.values < plan).sum())}")
""")

code("""
# The classical arm's seed policy, stated and checked rather than assumed.
# Nothing in CSP+LDA, SVC, MDM, FgMDM or tangent-space+LR takes a random_state, and
# leave-one-subject-out / leave-one-session-out are enumerations of the data -- so
# `seed` can only reach the within-session KFold. The grid therefore runs 5 seeds
# within-session and 1 on the other two modes, with a few second-seed control cells.
ml = scores[scores.family == "riemann/csp"]
g = (ml.groupby(["eval","dataset","model","subject","session"]).score
       .agg(["std","count"]).reset_index())
g = g[g["count"] > 1]
print("classical arm, seed-to-seed sd where a control seed exists:")
print(g.groupby("eval")["std"].agg(["max","mean","size"]).to_string())
print()
print("max = 0 on the deterministic modes -> the single-seed cells lose nothing.")
""")

md(r"""
## 1. What a family average hides

The figure Sylvain asked for. Bar = family mean, marker = that family's best model on
that dataset, labelled. The gap between them **is** the amount the mean was hiding.

`fixed control` has one member, so its bar and marker coincide — ignore that column for
this question.
""")

code("""
f = ef.champion_vs_mean(tidy, ORDER); plt.show()
""")

code("""
w = tidy[tidy["eval"] == "within_session"]
fam_mean = w.groupby(["dataset","family"]).above.mean().unstack()
fam_best = (w.groupby(["dataset","family","model"]).above.mean()
            .groupby(["dataset","family"]).max().unstack())
gap = (fam_best - fam_mean).loc[ORDER]
print("champion minus family mean (within_session):")
print(gap.round(3).to_string())
print()
print("worst case:", gap.stack().idxmax(), "->", round(gap.stack().max(), 3))
print()
print("NOTE: on the hardest datasets the gap is ~0 because every model sits at chance")
print("there. A small gap can mean 'the family is uniform' or 'nothing works'.")
""")

md(r"""
## 2. Who wins where, and is the champion stable

If the champion changed with every dataset it would be a maximum over fifteen noisy
numbers rather than a model worth naming.
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
print("how often each model is its family's champion:")
print(best.groupby(["family","model"]).size().sort_values(ascending=False).to_string())
""")

md(r"""
## 3. The crossed pair

One panel per dataset showing every model, then one panel per model showing every
dataset. Built to the requested shape: in the second, the six datasets that *have* a
cross-session split come first on the x axis, so the cross-session line stops halfway
instead of breaking into six disconnected segments, and the dashed rule marks where it
must stop.
""")

code("""
f = ef.per_dataset_all_models(tidy, ORDER); plt.show()
""")

code("""
f = ef.per_dataset_all_models(tidy, ORDER, eval_="cross_subject"); plt.show()
""")

code("""
f = ef.per_model_all_evals(tidy, ORDER_XS, N_XS); plt.show()
""")

md(r"""
## 4. Easy and hard, and where nothing was learned at all

The operational question — "if you can only make one model win, which dataset do you
pick" — plus its precondition. A family gap on a dataset where both families sit inside
the grey band is not a gap, and the chance map is where that is visible.
""")

code("""
f = ef.difficulty(tidy, ORDER); plt.show()
""")

code("""
f = ef.chance_map(tidy, ORDER); plt.show()
""")

code("""
ci = (w.groupby(["dataset","model"]).above.agg(["mean","std","size"]).reset_index())
ci["ci"] = 1.96 * ci["std"] / np.sqrt(ci["size"])
best = ci.loc[ci.groupby("dataset")["mean"].idxmax()].set_index("dataset").loc[ORDER]
best["at_chance"] = best["mean"] - best["ci"] <= 0
print("best model per dataset, within_session, and whether its CI still covers chance:")
print(best[["model","mean","ci","at_chance"]].round(3).to_string())
""")

md(r"""
## 5. Ranks, head to head, and what is actually resolvable

Three views of the same ordering, each removing a different objection. Ranks remove the
dataset's difficulty; the head-to-head matrix never leaves a (dataset, subject) pair so
it touches neither chance subtraction nor the mixed metric; the critical-difference plot
says which of the differences twelve datasets can support at all.
""")

code("""
f = ef.rank_heatmap(tidy, ORDER); plt.show()
""")

code("""
f = ef.rank_heatmap(tidy, ORDER, eval_="cross_subject"); plt.show()
""")

code("""
f = ef.mean_rank_cd(tidy); plt.show()
""")

code("""
f = ef.win_matrix(tidy); plt.show()
""")

code("""
f = ef.seed_noise(scores, ORDER); plt.show()
""")

code("""
f = ef.eval_penalty(tidy, ORDER); plt.show()
""")

md(r"""
## 6. Deep learning against Riemannian geometry, on v5

The comparison v5 could not make until its classical arm ran. Three views: who wins the
dataset, whether the win holds subject by subject, and whether the answer depends on how
much data the fit had.

On the 14-model grid this last figure was the decisive one — the deep-minus-classical
delta tracked trials per fit at Spearman ρ = +0.82, which reframed an apparent
evaluation-mode effect as a sample-size effect. Asked again here, on a campaign whose
sampling rate is pinned cell by cell and whose classical arm ran under that same pinned
regime, it comes back at **ρ = +0.81 (p = 6e-08, n = 30)**. The reframing survives the
change of campaign, which is the point of asking it twice.
""")

code("""
f = ef.subject_spread(tidy, ORDER); plt.show()
""")

code("""
f = ef.group_scatter(tidy, ORDER, a=DEEP, b=CLASSICAL,
                     label_a="deep model (braindecode, fixed or growing)",
                     label_b="riemann/csp pipeline"); plt.show()
""")

code("""
f = ef.subject_delta(tidy, ORDER, a=DEEP, b=CLASSICAL,
                     label=r"per-subject $\\Delta$ (deep $-$ classical champion)")
plt.show()
""")

code("""
f, XD = ef.train_size_crossover(
    tidy, scores, ef.delta_champion(DEEP, CLASSICAL),
    ylabel="best deep $-$ best riemann/csp (score $-$ chance)",
    title="Is deep learning's deficit a sample-size deficit here too?")
plt.show()
""")

code("""
rho, p = spearmanr(np.log10(XD.n), XD.delta)
print(f"all modes: rho = {rho:+.2f}, p = {p:.2g}, n = {len(XD)}")
for ev in ef.EVALS:
    h = XD[XD.ev == ev]
    if len(h) > 3:
        r_, p_ = spearmanr(np.log10(h.n), h.delta)
        print(f"  {ev:15s} rho = {r_:+.2f}, p = {p_:.3f}, n = {len(h):2d}, "
              f"mean delta = {h.delta.mean():+.4f}, "
              f"deep wins {int((h.delta > 0).sum())}/{len(h)}")
""")

code("""
f = ef.per_model_train_size(tidy, scores); plt.show()
""")

code("""
f = ef.cost_vs_score(scores, tidy); plt.show()
""")

md(r"""
## 7. Growing against fixed

The campaign's own question, three ways: as champions, subject by subject, and as the
four architecture-matched pairs against training-set size. The pairs are the ones to
trust — a maximum over five fixed models beats a maximum over four growing ones for
free, and the effect being measured is about 0.01.
""")

code("""
f = ef.group_scatter(tidy, ORDER, a=GROWING, b=FIXED,
                     label_a="growing model", label_b="fixed model"); plt.show()
""")

code("""
f = ef.subject_delta(tidy, ORDER, a=GROWING, b=FIXED,
                     label=r"per-subject $\\Delta$ (growing $-$ fixed champion)")
plt.show()
""")

code("""
f, XR = ef.train_size_crossover(
    tidy, scores, ef.delta_pairs(PAIRS),
    ylabel=r"mean matched-pair $\\Delta$ (grow $-$ fixed)",
    title="Does growing pay off more when data is scarce?")
plt.show()
""")

code("""
rho, p = spearmanr(np.log10(XR.n), XR.delta)
mde = float(np.tanh((1.959964 + 0.8416212) / np.sqrt(len(XR) - 3)))
print(f"all modes: rho = {rho:+.2f}, p = {p:.2g}, n = {len(XR)}")
for ev in ef.EVALS:
    h = XR[XR.ev == ev]
    if len(h) > 3:
        r_, p_ = spearmanr(np.log10(h.n), h.delta)
        print(f"  {ev:15s} rho = {r_:+.2f}, p = {p_:.2f}, n = {len(h):2d}, "
              f"mean delta = {h.delta.mean():+.4f}")
print()
print(f"Minimum effect this test could catch at 80% power: rho >= {mde:.2f}.")
print("So a small value here is a BOUND, not an absence.")
""")

md(r"""
## 8. Training curves — what happens during the fit

Everything above is drawn from one number per fold, so none of it can say *why*. A model
at chance may have failed to fit its training trials or fitted them perfectly and
generalised nothing, and those call for opposite fixes.

**One trap, marked in every figure.** Folds have different lengths because early stopping
fires at different times, so the fold-mean at a late epoch is taken over only the folds
that survived — which are the ones that were still improving. Each mean curve is drawn
solid while at least half the folds are still running and **dotted afterwards**: the
dotted part is survivorship, not training.
""")

code("""
CHANCE = (1.0 / scores.groupby("dataset").n_classes.first()).to_dict()
f = ec.learning_curves(cmean, ORDER_C, chance=CHANCE); plt.show()
""")

code("""
f = ec.learning_curves(cmean, ORDER_C, eval_="cross_subject", chance=CHANCE); plt.show()
""")

code("""
f = ec.loss_curves(cmean, ORDER_C); plt.show()
""")

code("""
f = ec.curve_seed_band(cmean, SHOWCASE); plt.show()
""")

code("""
f = ec.paired_curves(cmean, PAIRS, ORDER_C[:6]); plt.show()
""")

md(r"""
## 9. Where the fit ends, and whether it ended for the right reason

`max_epochs=200`, `grow_every=5`, `EarlyStopping(patience=20, monitor="valid_acc")` on
skorch's internal 20 % split. Two failure shapes are possible; the campaign has one of
them, and overwhelmingly. Budget truncation is essentially absent — **2 folds out of
140 490 reach 200 epochs**, so the 200-epoch ceiling is not what ends these fits. What
does end them is the patience: the median fold runs 29 epochs, and **40 % of folds pick
their best epoch at epoch 5 or earlier**, then spend 20 more waiting. Read the rest of
this section as being about that shape, not about a budget that ran out.

The last figure of this section is the one that could invalidate the others: early
stopping watches an internal split, but the number reported is MOABB's held-out fold. If
those two do not track each other, "this model is worse" is partly a statement about
model selection.
""")

code("""
f = ec.stopping_epoch(fsum, fits, ORDER_C); plt.show()
""")

code("""
f = ec.selected_epoch_vs_data(fsum, fits); plt.show()
""")

code("""
m = fsum.merge(fits[["eval","dataset","model","seed","fit","n_train"]],
               on=["eval","dataset","model","seed","fit"], suffixes=("","_f"))
m["bin"] = pd.qcut(m.n_train, 6, duplicates="drop")
print("selected epoch against training-set size, per fold:")
print(m.groupby("bin", observed=True).agg(best=("epoch_of_best","median"),
      stop=("epochs","median"), folds=("epoch_of_best","size")).to_string())
print()
print(f"folds selected at epoch <= 5: {(m.epoch_of_best <= 5).mean():.1%}")
print(f"folds that ever reach max_epochs: {(m.epochs >= 200).mean():.2%}")
""")

code("""
f = ec.selection_degeneracy(cmean, fits, ORDER_C); plt.show()
""")

code("""
q = (cmean[cmean["eval"]=="within_session"]
     .groupby(["dataset","model","seed"]).valid_acc_mean.nunique()
     .groupby("dataset").median())
flat = q[q <= 1]
print("datasets where the early-stopping monitor never moves (within_session):")
print(flat.to_string() if len(flat) else "  none")
print()
print("distinct valid-accuracy values along a run, median per dataset:")
print(q.sort_values().to_string())
""")

code("""
print("fraction of folds that stopped before max_epochs, by model:")
print(fits.groupby("model").stopped_early.mean().round(3).to_string())
print()
print("median epoch of best valid acc vs median epoch stopped, by model:")
a = fsum.groupby("model").epoch_of_best.median()
b = fits.groupby("model").epochs.median()
print(pd.DataFrame({"best_at": a, "stopped_at": b,
                    "wasted": (b - a)}).round(1).to_string())
""")

code("""
f = ec.overfit_gap(fsum, ORDER_C); plt.show()
""")

code("""
f = ec.valid_vs_test(fsum, tidy); plt.show()
""")

md(r"""
## 10. Growth itself, and what it cost

Width against epoch across every dataset, whether the schedule ran to completion, and
the two cost axes: parameter-epochs (the capacity a fold actually paid for, which is the
only axis on which a growing model can win while losing on score) and seconds per epoch.
""")

code("""
f = ec.width_trajectory(cmean, ORDER_C); plt.show()
""")

code("""
f = ec.width_reached(fits, ORDER_C); plt.show()
""")

code("""
g = fits[fits.target_width.notna()]
print("growth outcome per model (all evaluation modes):")
print(g.groupby("model").agg(folds=("fit","size"), grew=("grew","mean"),
                             reached=("reached_target","mean"),
                             width_end=("width_end","median"),
                             target=("target_width","median")).round(3).to_string())
""")

code("""
f = ec.budget_pareto(budget, tidy); plt.show()
""")

code("""
f = ec.cost_per_epoch(fits, ORDER_C); plt.show()
""")

md(r"""
## 11. Does the resampling doubt actually move the numbers?

The older grid was interrupted when the cluster went down and some cells were re-run
without the resample argument fully threaded, so its sampling rate is not verifiable cell
by cell. v5 ships a provenance row pinning 250 Hz for every cell — now including the
classical arm, which ran under the same pinned regime and differs only in `device=cpu`.
Where both campaigns ran the same model on the same dataset, their agreement bounds how
much that doubt is worth.

**Read with one caveat**: a fraction of the underlying score rows are bit-identical
between the two exports, so these are two overlapping campaigns, not two independent
replicates. The informative part is the rows that differ.
""")

code("""
f, resid = ef.campaign_agreement(old_tidy, tidy,
                                 label_a="results_published", label_b="results_v5")
plt.show()
""")

code("""
k = ["eval","dataset","model","subject","session","seed"]
a = pd.read_csv(OLD / "eegrow_benchmark_all_scores.csv.gz")
common = sorted(set(a.model) & set(scores.model))
A = a[a.model.isin(common)].set_index(k).score
B = scores[scores.model.isin(common)].set_index(k).score
A, B = A[~A.index.duplicated()], B[~B.index.duplicated()]
i = A.index.intersection(B.index)
d = (A[i] - B[i]).abs()
print(f"overlapping fold-level rows: {len(i):,} over {len(common)} shared models")
print(f"bit-identical: {100*(d < 1e-12).mean():.1f}%  <- NOT independent replicates")
print(f"of the rest: median |diff| {d[d >= 1e-12].median():.4f}, "
      f"p95 {d[d >= 1e-12].quantile(.95):.4f}")
print(f"campaign means on the overlap: published {A[i].mean():.4f}  v5 {B[i].mean():.4f}")
""")

md(r"""
## What I take from this

Filled in from the printed numbers above rather than written in advance, so this section
cannot drift from the frames it is built on. The campaign is complete, so these numbers
are final for this grid — what would move them is a change of grid, not another cell.
""")

code("""
w = tidy[tidy["eval"] == "within_session"]
fam_best = (w.groupby(["dataset","family","model"]).above.mean()
            .groupby(["dataset","family"]).max().unstack())
gap = (fam_best - w.groupby(["dataset","family"]).above.mean().unstack())
print(f"1. A family average hides up to {gap.stack().max():.3f} "
      f"(worst: {gap.stack().idxmax()}).")
ch = (w.groupby(["dataset","family","model"]).above.mean()
      .groupby(["dataset","family"]).idxmax())
print("2. Champions, within_session:")
for fam in ef.families_in(tidy):
    sel = [v[2] for k, v in ch.items() if k[1] == fam]
    if sel:
        top = pd.Series(sel).value_counts()
        print(f"     {fam:14s} {top.index[0]} on {top.iloc[0]}/{len(sel)} datasets")
if "riemann/csp" in set(tidy.family):
    for ev in ef.EVALS:
        s = tidy[tidy["eval"] == ev]
        d = (ef.best_of(s, DEEP) - ef.best_of(s, CLASSICAL)).dropna()
        if len(d):
            print(f"3. {ev:15s} deep beats classical on {int((d>0).sum())}/{len(d)} "
                  f"datasets, mean delta {d.mean():+.3f}")
print(f"4. Campaign agreement: median |diff| {resid.median():.4f}, "
      f"p95 {resid.quantile(0.95):.4f}.")
print(f"5. Growth: {fits[fits.target_width.notna()].reached_target.mean():.1%} of "
      f"growable folds reach their target width; "
      f"{fits.stopped_early.mean():.1%} of all folds stop before max_epochs.")
""")

md(r"""
### What v5 still cannot answer

These are limits of the **grid**, not of its completeness — no additional cell of this
campaign would lift any of them. Each needs a different run or a code change.

* **Nothing about alignment.** `align` is `'none'` on all 78 585 score rows of both
  campaigns, so the Euclidean-alignment question is not expressible from either. It needs
  its own run, not more of this one.
* **The per-fold curves carry no subject label.** `FitRecorder` stamps
  (eval, dataset, model, seed) but not the fold's subject — it is constructed once per
  cell, before MOABB clones the pipeline per fold — so every per-subject claim drawn from
  a curve figure rests on the write-order join in `growth_io.attach_subjects` (re-checked
  on these frames: Spearman +0.84 to +0.99 on eight of nine arms, `bd_deep4` at +0.33 to
  +0.59, mean +0.89 over 45 cells). One line in the callback's `meta` would make the
  label recorded instead of inferred.
* **Every growing number is a lower bound, not an estimate.** The scaling-factor line
  search in `grow_step` includes `0.0`, and when `0.0` wins the growth is applied anyway
  at exactly zero amplitude. A unit with zero incoming *and* zero outgoing weights sits
  at an exact stationary point and can never recover, while `width` and `n_params` keep
  counting it as capacity. The defect is not fixed, so "growth does not beat its matched
  control" is what the data support — not "growth does not help".
* **The winning scale factor is never logged**, so the size of that bound cannot be read
  off this campaign; it can only be re-measured by a run that records it.
* **Growth is not measured against a stopping rule it can finish under.** 40 % of folds
  select their best epoch by epoch 5, and one growth opportunity arrives every 5 epochs,
  so `grow_shallow` reaches its target width on 0.1 % of folds. What is measured here is
  a race between the growth schedule and early stopping, not growth at parity.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
nb_path = OUT / "eegrow_v5_exploration.ipynb"
nbf.write(nb, nb_path)

try:
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    NotebookClient(nb, timeout=3600,
                   resources={"metadata": {"path": str(OUT)}}).execute()
    nbf.write(nb, nb_path)
    body, _ = HTMLExporter().from_notebook_node(nb)
    (OUT / "eegrow_v5_exploration.html").write_text(body)
    print("notebook executed ->", OUT / "eegrow_v5_exploration.html")
except Exception as exc:  # noqa: BLE001
    print(f"notebook written but NOT executed: {type(exc).__name__}: {exc}")

print(f"figures -> {len(figs)}:", ", ".join(sorted(figs)))
