"""Emit the FINAL benchmark grid: the table the paper is written from.

This is not another sweep. It replaces ``results_v5_published/`` outright, because
three independent defects make every v5 score unusable in a figure:

  * ``drop_last`` -- 4 cells (shin2017a / physionetmi / alexmi within_session,
    shin2017a cross_session) were scored on networks that took ZERO gradient steps.
  * s=0 growth -- ``growable_width`` and ``n_params`` count permanently dead
    neurons, so the parameter axis of the efficiency figures -- our central claim --
    is inflated on exactly the arms the claim is about.
  * the undertrained protocol -- worth up to +0.13 accuracy and it REORDERS arms
    (``grow_shallow`` reads +0.021 or -0.006 depending on budget and control).

WHY THE GRID IS SMALLER THAN v5, AND WHERE THE CUTS FALL
--------------------------------------------------------
The corrected protocol is ``patience=200`` with ``max_epochs=200``, i.e. early
stopping never fires and every fit runs its full budget. v5's fits stopped at a mean
of 36.2 epochs (median 29), so the SAME grid now costs 200 / 36.2 = **5.52x** more.
v5 spent 2762 GPU-h on its deep arms; re-running it as-is is ~15 200 GPU-h. The three
cuts below buy that back, and none of them costs a result.

That 5.52x is the GLOBAL mean and it is the wrong number to plan any single protocol
with, because how early v5 stopped depended on the protocol -- cross_subject fits ran
to 61.0 epochs against 33.0 for within_session, i.e. the expensive protocol was
already the closest to full budget and inflates least. The factor is therefore
measured per eval from ``eegrow_v5_budget.csv.gz`` (200 / mean epochs of these six
arms) rather than assumed:

    within_session  33.0 epochs -> x6.06      cross_session  35.8 -> x5.59
    cross_subject   61.0 epochs -> x3.28

A caveat that keeps the number honest in one direction only: this scales wall-clock by
epochs at constant cost per epoch, and for the ``grow_*`` arms neither holds -- more
epochs means more growth steps at ``grow_every``, and each step widens the net, so a
later epoch costs more than an early one. The estimate is a LOWER bound on the growing
arms and roughly right on the ``bd_*`` ones.

``eegnex`` is dropped -- 1422 GPU-h, 51 % of v5's entire deep budget, spent on the one
comparison that is a MEASURED artefact: ``filter_1`` sizes the growable junction AND
the fixed tail, so ``grow_eegnex`` ends with 70 classifier inputs against braindecode's
280. Its -0.089 collapses to -0.0098 ns against the width-matched twin. Re-running it
unchanged would spend half the campaign re-publishing an artefact. (Restoring it means
fixing the geometry first, via ``filter_1_in``; that is a code change, not a re-run.)

The ``fix_*`` arms are back on the two cheap protocols, and that reversal is worth
stating. They were dropped because the growth contrast looked settled (SLURM 500952:
0/4 pairs survive Holm, no sign flip) -- but that experiment is **n=9 subjects on
bnci2014_001 alone**, on hopper cards this grid cannot be paired with, and its best
pair sits at p=0.055. "0/4 survive Holm at n=9" is the textbook underpowered null, not
a null, and the paper's whole framing rests on it: growth is presented as architecture
search at the price of one training run, which makes ``fix_X`` -- the same class frozen
at the target geometry -- the ablation that isolates the contribution. An ablation that
carries the framing cannot be the one measurement made on a single dataset.

They go on ``within_session`` and ``cross_session`` only. That is ~57 GPU-h, whose worst
cell is 3.7 h -- invisible under a 73.9 h critical path, so it costs no wall clock at
all -- and it takes the ablation from 9 subjects on 1 dataset to ~250 across 12.
``bd_X`` remains the reference level throughout, which is the contrast a reviewer reads.

THE ALIGNMENT AXIS, AND WHY IT IS NOT THE MAIN TABLE
----------------------------------------------------
Euclidean alignment estimates the held-out subject's whitening matrix on that subject's
own (unlabelled) trials. That is unsupervised domain adaptation, not label leakage, but
it is **transductive**: it assumes a batch of test trials exists. Published braindecode
and MOABB numbers are inductive, so an aligned headline table would no longer be
checkable against any of them -- and the external anchor is what makes "growing beats
braindecode" credible rather than self-reported. ``align=none`` therefore carries the
claim; ``align=euclidean`` is a second table that reports the higher absolute scores and
answers whether the advantage survives the strongest preprocessing in the field.

The scientific target is the growth x alignment interaction, measured so far on cho2017
alone (n=52): four estimates of the same sign, +0.72 to +1.06 pp, but the effect sits
under its own MDE (~1.0-1.3) and one survives Holm. It is a signal, not a result. The
aligned arm takes it to 12 datasets.

COST OF THE ALIGNED ARM -- THE TRAP THAT ALMOST SET THIS WRONG
---------------------------------------------------------------
Aligned cells were measured 3-5x cheaper than raw ones on cho2017 (grow euclidean
3h19-3h43 against grow none ~16h50). That number does NOT transfer here, and planning
on it would have under-budgeted the campaign by a factor of three.

The saving was early stopping. Those cells ran the *shipped* protocol (``patience:
null`` -> 20, verified in ``eegrow_xds``'s deployed ``config.yaml``): whitened data
converges, patience fires, and the fit ends around epoch 42, while the raw arm never
stops improving and burns all 200. Under THIS grid's protocol ``patience=200`` equals
``max_epochs``, so early stopping can never fire and both arms run the full budget.
Whitening is a one-off transform of the data, not a per-epoch cost.

So an aligned cell costs what its raw twin costs: the arm is priced at 1.00x, and the
aligned axis doubles whatever evals it is switched on for. ``--align-evals`` is the knob;
the default is the two cheap protocols, where the axis is ~113 GPU-h with a 3.7 h worst
cell and again costs no wall clock. Adding ``cross_subject`` is a real +704 GPU-h and a
real extension of the campaign -- it is EA's natural home (subject transfer) but it is a
new claim, not a generalisation of one we already hold.

Seeds fall from 5. THE UNIT OF ANALYSIS IS THE HELD-OUT SUBJECT: seeds are averaged
INSIDE the subject before any statistic, so a seed only shrinks the within-subject
noise component while the paired test is dominated by between-subject variance. That
is why the seed count here is a function of how many subjects the protocol yields:

    within_session / cross_session -> 3 seeds   (few units, seeds still buy precision)
    cross_subject                  -> 1 seed    (9-109 held-out subjects per dataset;
                                                 the between-subject term already
                                                 dominates and seeds are redundant)

and it is also where the money is: cross_subject is **97.1 %** of the cost of these
six arms in v5 (1073.6 h of 1105.8). At 3 seeds it would be ~2110 GPU-h; at 1 it is
~704. within_session + cross_session together are ~113 GPU-h at full budget -- free.

Total: **~817 GPU-h**, against ~15 200 for v5 re-run verbatim.

LONGEST-FIRST, AND WHY THE ORDER IS PART OF THE ARTEFACT
--------------------------------------------------------
``pack_run.sh`` claims cells in file order. Cell costs here span four orders of
magnitude -- alexmi within_session is seconds, ``lee2019_mi`` cross_subject
``bd_deep4`` is 23.6 h in v5 and therefore ~78 h at full budget. A cell cannot be
split across allocations, so a 78 h cell claimed late in the campaign is a cell that
never finishes. Emitting shortest-first would look like progress and lose exactly the
cells that cost the most to redo.

So cells are ordered by MEASURED v5 cost, descending (longest-processing-time first,
the standard makespan heuristic). The second property matters as much: if the fleet
runs out of time, what is missing is the CHEAP tail, which can be finished in hours on
any node at any moment -- rather than a monster cell that needs another 5 days.

Cost comes from v5's own ``time`` column, so a cell with no v5 ancestor (there are
none in this grid) sorts last rather than crashing.

USAGE
    python benchmarks/slurm/final_grid.py --out benchmarks/slurm/final_grid.tsv

The TSV is committed. That is deliberate and it is the same reason ``make_grid.py``
writes one: the exact set of cells a campaign covers has to be a diffable artefact,
not a shell expansion nobody can reconstruct when a reviewer asks what ran.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

BENCH = Path(__file__).resolve().parent.parent
CONFIG = BENCH / "config"
sys.path.insert(0, str(BENCH))
from utils import align_tag, cell_stem  # noqa: E402
V5_SCORES = BENCH / "results_v5_published" / "eegrow_benchmark_all_scores.csv.gz"
V5_BUDGET = BENCH / "results_v5_published" / "eegrow_v5_budget.csv.gz"

# The corrected protocol. Written here, next to the cost model that assumes it, because
# these two overrides ARE the reason this grid exists and `config.yaml` deliberately
# ships the losing values (flipping the defaults would re-date results_v5_published/).
# `final_grid.sbatch` passes exactly this string; if it ever diverges from this line,
# the cost estimate below is describing a campaign nobody ran.
PROTOCOL = "train.patience=200 train.selection_monitor=valid_acc"
MAX_EPOCHS = 200

# Three architecture families, each as the pair the paper actually contrasts:
# our growing net against braindecode's reference at the target width.
#
# `grow_deep` is paired with `bd_deep4` here as a REFERENCE level, not as a control:
# bd_deep4 is a different network (4 stages, 25/50/100/200) and the same-class frozen
# twin is `fix_deepeeg`. That contrast is settled and lives in its own experiment; what
# this grid measures is our arms against the library everyone else runs.
MODELS = ["bd_shallow", "grow_shallow",
          "bd_sccnet", "grow_sccnet",
          "bd_deep4", "grow_deep"]

# The width-matched controls: same class as the growing arm, frozen at the target
# geometry, so the pair differs by growth alone. Restricted to the two cheap protocols
# -- see the docstring: this buys the ablation 12 datasets for ~7 % of the campaign and
# no wall clock, whereas putting it on cross_subject would cost more than the rest of
# the grid combined to answer a question the cheap protocols already answer.
FIX_MODELS = ["fix_shallow", "fix_sccnet", "fix_deepeeg"]
FIX_EVALS = ("within_session", "cross_session")

# Evals that also run an `align=euclidean` twin. Same default and the same reasoning:
# the interaction is the target, cho2017 already showed it does not depend on the arm
# (pooled - within = -0.08 pp, p=0.89), and these two protocols are free in wall clock.
ALIGN_EVALS = ("within_session", "cross_session")
ALIGNS = ("none", "euclidean")

# Only these carry more than one session, so cross_session is defined on them alone --
# MOABB answers "Only one session available" elsewhere, which surfaces as an empty
# result rather than an error. Same list as make_grid.py; kept in sync by hand because
# importing it would drag that module's argparse in.
CROSS_SESSION_DATASETS = ["bnci2014_001", "bnci2014_004", "bnci2015_001",
                          "lee2019_mi", "shin2017a", "zhou2016"]

# See the module docstring: seeds compensate for a scarcity of units of analysis, and
# cross_subject has no such scarcity.
SEEDS = {"within_session": [0, 1, 2],
         "cross_session": [0, 1, 2],
         "cross_subject": [0]}


def group(name: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG / name).glob("*.yaml"))


def measured_cost() -> dict[tuple[str, str, str], float]:
    """Mean v5 wall-clock per (eval, dataset, model) cell, in hours.

    Averaged over v5's seeds rather than taken per seed: the seed is not a property
    of how expensive a cell is, and this grid does not carry v5's seed numbering.
    """
    d = pd.read_csv(V5_SCORES)
    per_cell = (d.groupby(["eval", "dataset", "model", "seed"])["time"].sum() / 3600)
    mean = per_cell.groupby(level=[0, 1, 2]).mean()
    return {k: float(v) for k, v in mean.items()}


def budget_factor(models: list[str]) -> tuple[dict[tuple[str, str], float],
                                              dict[str, float]]:
    """``MAX_EPOCHS`` / mean epochs v5 actually ran: per (eval, model), then per eval.

    Per (eval, model) rather than per eval alone, because the arms of this grid do not
    stop at the same point and the grid is no longer homogeneous: adding the ``fix_*``
    controls to a single per-eval mean would move the factor for ``bd_deep4`` because a
    frozen shallow net stopped somewhere else. The per-eval mean survives only as the
    fallback for a (eval, model) v5 never ran.

    Restricted to the arms of THIS grid: eegnex is dropped from the campaign and
    averaging it in would describe a campaign nobody is planning.
    """
    b = pd.read_csv(V5_BUDGET)
    b = b[b.model.isin(models)]
    per_cell = {(ev, m): MAX_EPOCHS / float(e)
                for (ev, m), e in b.groupby(["eval", "model"])["epochs"].mean().items()}
    per_eval = {ev: MAX_EPOCHS / float(e)
                for ev, e in b.groupby("eval")["epochs"].mean().items()}
    return per_cell, per_eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="-")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--fix-models", nargs="+", default=FIX_MODELS,
                    help="width-matched controls; [] to drop the growth ablation")
    ap.add_argument("--fix-evals", nargs="+", default=list(FIX_EVALS))
    ap.add_argument("--align-evals", nargs="+", default=list(ALIGN_EVALS),
                    help="evals that also get an align=euclidean twin; "
                         "adding cross_subject roughly doubles the campaign")
    a = ap.parse_args()
    # `none` spells the empty list: argparse's nargs="+" cannot take zero values, and
    # "price this grid without the arm I just added" is the first question anyone asks
    # of a cost model. Without it the only way to ask is to edit the file.
    drop = lambda v: [] if v == ["none"] else v
    a.fix_models, a.fix_evals = drop(a.fix_models), drop(a.fix_evals)
    a.align_evals = drop(a.align_evals)

    known_ds, known_m = group("dataset"), group("model")
    for m in [*a.models, *a.fix_models]:
        if m not in known_m:
            raise SystemExit(f"unknown model config: {m}")
    for ev in [*a.fix_evals, *a.align_evals]:
        if ev not in SEEDS:
            raise SystemExit(f"unknown eval: {ev}")

    cost = measured_cost()
    cells: list[tuple[float, str, str, str, str, int]] = []
    for ev, seeds in SEEDS.items():
        datasets = CROSS_SESSION_DATASETS if ev == "cross_session" else known_ds
        models = list(a.models) + (list(a.fix_models) if ev in a.fix_evals else [])
        aligns = ALIGNS if ev in a.align_evals else ("none",)
        for ds in datasets:
            for m in models:
                # -1.0 for a cell v5 never ran: sorts last, which is the safe end.
                c = cost.get((ev, ds, m), -1.0)
                for al in aligns:
                    for s in seeds:
                        cells.append((c, ev, ds, m, al, s))

    cells.sort(key=lambda t: -t[0])
    # The expected output basename travels WITH the cell rather than being rebuilt in
    # shell. `pack_run.sh` decides a cell is done by testing for `<stem>.csv`, and the
    # runner derives that name from the align config; a shell reimplementation that
    # drifted by one character would make the packer re-run the entire campaign on top
    # of itself without a single error. See utils.cell_stem.
    tags = {al: align_tag(yaml.safe_load((CONFIG / "align" / f"{al}.yaml").read_text()))
            for al in ALIGNS}
    text = "".join(f"{ev}\t{ds}\t{m}\t{al}\t{s}\t{cell_stem(m, tags[al], s)}\n"
                   for _, ev, ds, m, al, s in cells)

    if a.out == "-":
        print(text, end="")
    else:
        Path(a.out).write_text(text)

    per_cell, per_eval = budget_factor([*a.models, *a.fix_models])
    # An aligned cell is priced at 1.00x its raw twin. See the docstring: the 3-5x
    # saving measured on cho2017 was early stopping, which `patience=200` disables.
    by_eval: dict[str, list[float]] = {}
    by_arm: dict[str, list[float]] = {}
    for c, ev, _ds, m, al, _s in cells:
        h = max(c, 0.0) * per_cell.get((ev, m), per_eval[ev])
        by_eval.setdefault(ev, []).append(h)
        arm = "fix (growth ablation)" if m in a.fix_models else "grow / bd"
        by_arm.setdefault(f"{arm}, align={al}", []).append(h)

    full = sum(sum(v) for v in by_eval.values())
    worst = max(max(v) for v in by_eval.values())
    report = [
        f"{len(cells)} cells -> {a.out}",
        f"  protocol           {PROTOCOL}",
        f"  full-budget cost   {full:8.1f} GPU-h",
        f"  worst single cell  {worst:8.1f} h     <- CELL_TIMEOUT and the partition "
        f"wall must both exceed this",
        "  by eval:",
    ]
    for ev, v in sorted(by_eval.items(), key=lambda kv: -sum(kv[1])):
        report.append(f"    {ev:16s} {sum(v):8.1f} GPU-h  {len(v):4d} cells  "
                      f"worst {max(v):6.1f} h")
    report.append("  by arm:")
    for arm, v in sorted(by_arm.items(), key=lambda kv: -sum(kv[1])):
        report.append(f"    {arm:28s} {sum(v):8.1f} GPU-h  {len(v):4d} cells")
    print("\n".join(report), file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
