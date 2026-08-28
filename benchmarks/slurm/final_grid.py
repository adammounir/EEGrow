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

The ``fix_*`` arms are dropped -- the GROWTH contrast is already settled at full budget
(SLURM 500952, 24 cells): 0/4 pairs survive Holm, no sign flip. ``bd_X`` stays as the
reference level, which is the contrast a reviewer reads.

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
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parent.parent
CONFIG = BENCH / "config"
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


def budget_factor(models: list[str]) -> dict[str, float]:
    """``MAX_EPOCHS`` / mean epochs v5 actually ran, per eval.

    Restricted to the arms of THIS grid: eegnex and the fixed controls stopped at
    different points and averaging them in would describe a campaign that is not
    the one being planned.
    """
    b = pd.read_csv(V5_BUDGET)
    b = b[b.model.isin(models)]
    return {ev: MAX_EPOCHS / float(e) for ev, e in b.groupby("eval")["epochs"].mean().items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="-")
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()

    known_ds, known_m = group("dataset"), group("model")
    for m in a.models:
        if m not in known_m:
            raise SystemExit(f"unknown model config: {m}")

    cost = measured_cost()
    cells: list[tuple[float, str, str, str, int]] = []
    for ev, seeds in SEEDS.items():
        datasets = CROSS_SESSION_DATASETS if ev == "cross_session" else known_ds
        for ds in datasets:
            for m in a.models:
                # -1.0 for a cell v5 never ran: sorts last, which is the safe end.
                c = cost.get((ev, ds, m), -1.0)
                for s in seeds:
                    cells.append((c, ev, ds, m, s))

    cells.sort(key=lambda t: -t[0])
    text = "".join(f"{ev}\t{ds}\t{m}\t{s}\n" for _, ev, ds, m, s in cells)

    if a.out == "-":
        print(text, end="")
    else:
        Path(a.out).write_text(text)

    fac = budget_factor(a.models)
    by_eval: dict[str, list[float]] = {}
    for c, ev, *_ in cells:
        by_eval.setdefault(ev, []).append(max(c, 0.0) * fac[ev])

    full = sum(sum(v) for v in by_eval.values())
    worst = max(max(v) for v in by_eval.values())
    report = [
        f"{len(cells)} cells -> {a.out}",
        f"  protocol           {PROTOCOL}",
        f"  full-budget cost   {full:8.1f} GPU-h",
        f"  worst single cell  {worst:8.1f} h     <- CELL_TIMEOUT and the partition "
        f"wall must both exceed this",
    ]
    for ev, v in sorted(by_eval.items(), key=lambda kv: -sum(kv[1])):
        report.append(f"    {ev:16s} {sum(v):8.1f} GPU-h  {len(v):4d} cells  "
                      f"x{fac[ev]:.2f}  worst {max(v):6.1f} h")
    print("\n".join(report), file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
