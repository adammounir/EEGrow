#!/bin/bash
# Stage 1: does OUR Euclidean Alignment reproduce the published effect?
# 3 nets x {raw, EA} x 3 seeds = 18 array tasks, each running the full 9-fold LOSO.
#
# WHY THIS RUNS BEFORE ANYTHING ELSE. The question we actually care about is the
# INTERACTION -- whether removing inter-subject nuisance pays a growing net more than it
# pays a fixed one (`ea_interaction.py`). An interaction is a difference of differences,
# so it is only interpretable if each difference is itself real. If our EA does not
# reproduce the effect that is already published, an interaction computed on top of it
# measures our bug, not the hypothesis. So: replicate first, gate on it, interact after.
#
# THE GUARD-RAIL, fixed in advance. Junqueira, Aristimunha, Chevallier & de Camargo
# (arXiv 2401.10746), Table 2, BNCI2014_001 LOSO, 2-class left/right hand:
#     No-EA 68.93 +/- 12.61  ->  Offline-EA 73.98 +/- 11.21   (+5.05 pp)
#     overall across datasets 70.48 -> 76.07
# The absolute level is allowed to drift -- their harness is not ours -- but the paired
# per-subject DELTA has to land positive and in that order of magnitude. It does not, we
# debug our EA instead of building the next stage on it.
#
# WHY `dataset.paradigm=LeftRightImagery`. Our dataset config ships MotorImagery, which
# on BNCI2014_001 is the 4-class task. The paper's numbers are 2-class left/right. A
# 4-class replication would be measuring a different task and could not be compared to
# the table above at all.
#
# WHY THE PROTOCOL IS OVERRIDDEN. `train.patience=200 train.selection_monitor=valid_acc`
# is the CORRECTED protocol, not the shipped default. Under the default every score comes
# from the model 20 epochs past its own best, restored by a validation loss that is a
# broken proxy on a split this small -- worth +0.13 on bd_deep4 alone. Replicating a
# paper under a protocol we have already measured to be broken would tell us nothing
# about our EA.
#
# WHY 3 SEEDS AND NOT 1. The verdict is a +5.05 pp effect read off 9 held-out subjects.
# Seed noise on a LOSO fold is worth points, so a single seed cannot separate +3 from +7
# -- which is exactly the distinction the gate is made of. The seed is shared between the
# raw and aligned arms of a pair, so it cancels in the paired delta and only tightens it.
#
# NO LABEL LEAKAGE, and it is worth stating because LOSO + a data-dependent transform is
# where leakage usually hides. EA whitens each subject by ITS OWN reference covariance,
# computed from that subject's trials without ever touching y. The held-out subject is
# aligned with its own unlabelled recording, which is precisely the paper's "offline"
# setting. Nothing crosses the train/test boundary.
#
# COST. Extrapolated from v5's cross_subject medians on this dataset (per LOSO fit:
# bd_shallow 330 s, bd_deep4 189 s, bd_sccnet 135 s as the EEGNet-sized proxy) x 3.0 for
# the full budget -- the measured full/p20 ratio from job 500952 (fix_shallow 2:25->7:24
# = x3.06, fix_sccnet 2:03->4:54 = x2.39). That is ~1965 s per (subject x arm), so ~30
# GPU-h for all 18 tasks. It is an UPPER bound by roughly 2x: LeftRightImagery halves the
# trials against the 4-class paradigm those medians were measured on.
#
# gpu:hopper:1 is pinned for the same reason as the fixed-control array: every comparison
# here is paired, and a mixed-card array puts a hardware difference inside the pairing.
#SBATCH --job-name=ea-repro
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-17%6
#SBATCH --output=/scratch/amounir/eegrow_ea/logs/%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

EA=/scratch/amounir/eegrow_ea
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench

export PYTHONPATH=$EA/src
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EA/benchmarks/slurm/ea_repro_cells.txt")
read -r MODEL ALIGN SEED <<< "$LINE"
if [ -z "${MODEL:-}" ]; then echo "no cell for task $SLURM_ARRAY_TASK_ID"; exit 1; fi
OUT=$EA/results_ea
export EEGROW_BENCH_ROOT=$OUT

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname)"
echo "  model=$MODEL align=$ALIGN seed=$SEED"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# The `bench` env carries an editable .pth pointing at a DIFFERENT checkout
# (/scratch/amounir/eegrow). PYTHONPATH wins, but assert it rather than assume: a silent
# fall-through would run a tree that has no alignment code at all, and `align=euclidean`
# would then quietly produce a second copy of the raw arm.
python - <<'PY' || exit 1
import eegrow
import torch
from eegrow.alignment import euclidean_align  # noqa: F401

assert eegrow.__file__.startswith("/scratch/amounir/eegrow_ea/"), eegrow.__file__
assert torch.cuda.is_available(), "no CUDA on a gpu partition job"
import braindecode.models as bdm
assert hasattr(bdm, "EEGNet"), "braindecode too old: bd_eegnet resolves EEGNet"
print("guard ok", eegrow.__file__, torch.cuda.get_device_name(0))
PY

# Resumable under --requeue: the per-cell CSV is written last, so its presence means the
# cell finished. The `ea` tag is in the stem, so the raw and aligned arms of the same
# (model, seed) cannot collide here or in the fit records.
TAG=""
if [ "$ALIGN" = "euclidean" ]; then TAG="__easubject"; fi
DONE=$OUT/results/cross_subject/bnci2014_001/${MODEL}${TAG}__seed${SEED}.csv
if [ -f "$DONE" ]; then echo "already done: $DONE"; exit 0; fi

cd $EA/benchmarks
t0=$SECONDS
python run_moabb_hydra.py \
    "model=$MODEL" "align=$ALIGN" \
    dataset=bnci2014_001 dataset.paradigm=LeftRightImagery \
    eval=cross_subject "seed=$SEED" \
    train.patience=200 train.selection_monitor=valid_acc \
    "results_dir=$OUT/results"
rc=$?
echo "[$(date)] $MODEL $ALIGN seed$SEED rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
