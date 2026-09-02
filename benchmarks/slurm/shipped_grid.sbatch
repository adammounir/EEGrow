#!/bin/bash
#SBATCH --job-name=eegrow_shipped
#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --exclude=margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/logs/shipped_%j.log
#
# THE SHIPPED-PROTOCOL TWIN of the final benchmark. Same code, same cells, same cards,
# same seeds; the ONLY difference from `final_grid.sbatch` is the two knobs on the
# PROTOCOL line. Read that file first -- every decision here except the protocol and the
# timeout is inherited from it, and its comments are not repeated.
#
# WHY THIS CAMPAIGN EXISTS
#
# `final_grid.py`, `final_grid.sbatch`, `CAMPAIGNS.md` and `config.yaml` all assert that
# the shipped protocol "REORDERS arms". That assertion is doing real work -- it is why
# v5 was thrown away and why 817 GPU-h were spent re-running it -- and the evidence
# under it is SLURM 500573: bnci2014_001, n=9 subjects, 8 arms, 2 seeds, within_session
# only. That is enough to justify overriding the defaults for our own campaign. It is
# not enough to tell the field that its published rankings are protocol artefacts,
# which is the claim the paper wants to make.
#
# This grid turns the assertion into a measurement: two complete 12-dataset rankings
# whose only difference is `patience` and `selection_monitor`. The readout is not an
# accuracy delta (we already have that) -- it is whether the ORDER of the arms changes,
# per dataset and per eval, and by how much (Kendall tau, sign flips, top-1 changes).
#
# WHY align=none ONLY
#
# The claim is about the ranking of ARCHITECTURES, so it needs one preprocessing arm,
# and `align=none` is the one that carries the headline table (aligned numbers are
# transductive and not checkable against published braindecode/MOABB scores). The
# emitted grid is exactly the 558 `align=none` cells of `final_grid.tsv` -- verified as
# a set equality, not by regenerating with the same flags -- so every cell here has a
# corrected-protocol twin at the same (eval, dataset, model, seed).
#
# gpu:turing:3 for the SAME reason as the final grid, applied one level up: there the
# paired difference was grow_X - bd_X, here it is shipped - corrected on the same cell.
# A card class split across that difference puts hardware inside the protocol effect.
set -uo pipefail
export ROOT=/scratch/amounir/eegrow_budget
#
# PROTOCOL: the values `config.yaml` ships (`patience: null` -> 20, `valid_loss`),
# passed EXPLICITLY rather than through pack_run.sh's `PROTOCOL=shipped` shorthand.
# The shorthand runs the identical protocol -- it just empties the override string --
# but then the two numbers that define this campaign appear in no log and in no run
# record, and a year from now the only evidence of which protocol produced these CSVs
# would be this comment. Written out, `hydra`'s override list carries them per cell.
export PROTOCOL="train.patience=20 train.selection_monitor=valid_loss"
#
# CELL_TIMEOUT: 48 h against a worst cell of 22.5 h (lee2019_mi cross_subject bd_deep4).
#
# That 22.5 h is a MEASUREMENT, not a projection: v5 ran this exact protocol, so
# `final_grid.py --protocol shipped` prices each cell at its own measured v5 wall clock
# with no budget factor applied. The final grid needed 47 % margin because its worst
# cell was an extrapolation known to be a lower bound on the growing arms; here the
# extrapolation is gone, and 113 % of margin is generous rather than thin.
#
# It is also why `--time` is 2 days and not 7: 238 GPU-h total, and early stopping DOES
# fire under this protocol, which is the entire reason the twin is affordable.
export CELL_TIMEOUT=172800
bash "$ROOT/benchmarks/slurm/pack_run.sh"
