#!/bin/bash
#SBATCH --job-name=eegrow_final
#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --exclude=margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/logs/final_%j.log
#
# THE FINAL BENCHMARK. This allocation wrapper is the file to read before submitting,
# because it is where the two decisions that can silently ruin the campaign are made:
# WHICH CHECKOUT runs (ROOT) and WHICH TRAINING PROTOCOL it runs under (PROTOCOL).
# Everything else -- G, K, the device ceiling, the cell list -- is derived by
# plan_campaign.py from measurements and arrives on the sbatch line.
#
# Submit through the generated script, never by hand:
#     python benchmarks/slurm/final_grid.py --out benchmarks/slurm/final_grid.tsv
#     bash benchmarks/slurm/deploy_final.sh          # ships HEAD + stamps .eegrow_sha
#     python benchmarks/slurm/plan_campaign.py \
#         --grid /scratch/amounir/final_grid.tsv --outdir /scratch/amounir/passes_final \
#         --root /scratch/amounir/eegrow_budget --tag final --gres-type turing \
#         --wrapper benchmarks/slurm/final_grid.sbatch
#     bash /scratch/amounir/passes_final/submit.sh
#
# --gres-type turing is NOT optional, and it is the easiest line to leave out because
# the campaign runs perfectly well without it. plan_campaign emits `--gres=gpu:N` when
# it is absent, which overrides this file's `gpu:turing:3` and lets SLURM answer with
# whatever card class is free. Every headline number is a subject-paired grow_X minus
# bd_X; a pair split across two card classes has a hardware difference inside the
# difference, and no amount of later analysis can take it back out. See the gres block
# further down for why pascal nodes are excluded outright rather than merely unasked.
#
# ---------------------------------------------------------------------------
# ROOT: the budget checkout, NOT /scratch/amounir/eegrow.
#
# The August tree carries the drop_last fix (3cefa3a) but NOT the s=0 abstention
# (5337c56). Running the growing arms there would re-add permanently dead neurons and
# re-inflate `growable_width` -- i.e. re-break the parameter axis that carries this
# paper's central claim. pack_run.sh asserts the imported package both lives under
# ROOT and contains the abstention, so this line cannot be wrong silently.
export ROOT=/scratch/amounir/eegrow_budget
#
# PROTOCOL: the corrected training protocol, passed explicitly because config.yaml
# deliberately ships the losing values (flipping them would re-date every archived v5
# figure). Measured on a 36-unit paired square, bnci2014_001 within_session, bd_deep4:
#
#                       sel=valid_loss   sel=valid_acc
#     patience  20          0.3501           0.3390
#     patience 200          0.4074           0.4818
#
# The interaction is +0.0854 (p=1.1e-07): the two knobs do not add, so passing one
# alone gives a third wrong protocol rather than half a fix. Submitting without them
# reproduces the undertrained campaign with no warning and well-formed CSVs. This
# string must stay byte-identical to PROTOCOL in final_grid.py, whose cost model
# assumes it.
export PROTOCOL="train.patience=200 train.selection_monitor=valid_acc"
#
# CELL_TIMEOUT: 6 days. The ceiling exists to stop a hung cell from holding a slot
# forever, so it belongs just above the worst HONEST cell, not near the median.
#
# It was 120 h against a projected worst cell of 74 h. Both numbers moved, and the
# margin moved the wrong way. That 74 h came from a budget factor averaged over a whole
# eval, which is only defensible while every arm of that eval stops at the same point;
# the grid is no longer homogeneous (the fixed controls are back), so final_grid.py
# projects per (eval, model) now. Under that factor the worst cell is lee2019_mi
# cross_subject grow_deep at 98.1 h -- and it is a GROWING arm, i.e. exactly the case
# the projection under-estimates, because more epochs means more growth steps and each
# step widens the net. 120 h would have left 22 % of margin on a number known to be a
# lower bound, which is how a four-day cell gets killed at hour 120 and returns nothing.
#
# 144 h leaves 47 % and still fits under the 7-day wall with 24 h for the sweep that
# follows. (fix_deepeeg on that same cell projects at 112.9 h, which is one more reason
# the fixed controls are restricted to the two cheap protocols.)
export CELL_TIMEOUT=518400
#
# gpu:turing:3, not gpu:3. Not for memory -- these nets are 30k-283k parameters -- but
# because every headline number is a SUBJECT-PAIRED difference between grow_X and
# bd_X. If one arm of a pair lands on ampere and the other on turing, a hardware
# difference sits inside the paired difference and there is no way to remove it
# afterwards. tau also carries pascal nodes (sm_60), on which torch 2.13+cu130 fails
# outright with "no kernel image is available"; they are excluded rather than merely
# not requested. margpu021 advertises gpu:turing:3 and answers "No devices were found"
# -- it does not fail, it runs every cell it claims against a device that is not there.
# margpu023 is drained. Usable: margpu[017-020,022,028], 20 turing GPUs.
#
# --exclusive because --mem is NOT enforced on this cluster: the turing nodes report
# AllocMem=0, SLURM co-scheduled two 120G allocations on margpu020 during v5, and the
# arbiter became the kernel OOM killer, which uses SIGKILL, which no trap catches. 85
# cells died that way. A packing runner already decides its own co-tenancy through K;
# letting the scheduler add a second tenant set on top is over-subscription by accident.
#
# 7 days is the partition maximum and it is needed: this grid's cost is 97 %
# cross_subject and its heaviest pass runs K=1 (schirrmeister2017 needs 27 GB of host
# RAM per tenant, lee2019_mi 11.5 GB), so those cells serialise 3 at a time.
#
# Several of these can be submitted at once and SHOULD be -- plan_campaign.py emits 3
# allocations for the K=1 pass. They cooperate through the atomic claim directory
# rather than partitioning the grid, so an extra allocation adds throughput with no
# split to get wrong, and a node that finishes early picks up what a slower one left.
set -uo pipefail
bash "$ROOT/benchmarks/slurm/pack_run.sh"
