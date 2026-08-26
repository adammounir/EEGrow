#!/bin/bash
# The EA replication, moved to a dataset that can actually answer it: Cho2017, 52
# subjects. 3 nets x {raw, EA} x 1 seed = 6 array tasks, each a full 52-fold LOSO.
#
# WHY NOT bnci2014_001. It was run first (jobs 501098/501101) and the answer was that
# nine subjects cannot decide this. Measured there: the between-SUBJECT sd of the paired
# delta is 6.14 pp, so the minimum detectable effect at n=9 and power .80 is +6.54 pp --
# larger than the +5.05 pp the gate is checking for. Detecting +5.05 needs 12 subjects.
# The limit is not the seeds: going from one seed to two moved the CI width only from
# 8.2 to 7.5 pp. At n=52 the MDE drops to ~2.4 pp and the question becomes decidable,
# which is also why ONE seed is enough here where three were needed there.
#
# Cho2017 is natively 2-class left/right hand, so no paradigm override: the dataset
# config already says LeftRightImagery. All 52 subjects are in the shared MNE cache
# (MNE-gigadb-data, 10 GB), so nothing downloads inside a GPU job.
#
# COST SCALES AS n^2, WHICH IS THE WHOLE REASON THIS IS TIMED FIRST. A LOSO campaign is
# n folds, each training on n-1 subjects, so going from 9 to 52 subjects is not 5.8x but
# ~30x -- before counting Cho2017's 64 channels against bnci's 22. Run the probe below
# and read the extrapolation off it rather than trusting that arithmetic:
#
#     EA_SUBJECTS=10 sbatch --array=0 benchmarks/slurm/ea_cho_gpu.sbatch
#
# gpu:hopper:1. This started as `ampere` on the reasoning that the partition holds ~18
# ampere cards against 3 hopper, so the wide pool would drain in one wave. Measured on
# 2026-08-26, that was wrong: all 18 ampere cards were allocated, one hopper card was
# free, and slurm's own estimate for the ampere probe was a 37-hour wait. The pending
# queue explains it -- nine array tasks asking for a generic `gres/gpu:1` with a 7-day
# wall, which land on ampere and hold it. What matters is what is FREE, not what exists.
#
# Nothing here pairs against another campaign, so the card only has to be constant across
# these 6 cells; hopper also happens to be what the bnci arm of this same replication ran
# on, which makes stage 1 internally consistent rather than less so.
#SBATCH --job-name=ea-cho
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
# %3, because the partition has exactly 3 hopper cards. A higher throttle would not buy
# any parallelism -- slurm caps at the hardware either way -- it would only queue the
# extra tasks under our own name and make the run look stuck.
#SBATCH --array=0-5%3
#SBATCH --output=/scratch/amounir/eegrow_ea/logs/%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

EA=/scratch/amounir/eegrow_ea
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench

export PYTHONPATH=$EA/src
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EA/benchmarks/slurm/ea_cho_cells.txt")
read -r MODEL ALIGN SEED <<< "$LINE"
if [ -z "${MODEL:-}" ]; then echo "no cell for task $SLURM_ARRAY_TASK_ID"; exit 1; fi

# EA_SUBJECTS caps the subject list for the timing probe. It changes the ESTIMAND -- a
# LOSO over 10 subjects trains on 9, not 51 -- so probe results go to a separate tree
# and can never be mistaken for the real campaign.
SUBJ_OVERRIDE=""
OUT=$EA/results_cho
if [ -n "${EA_SUBJECTS:-}" ]; then
    SUBJ_OVERRIDE="dataset.subjects=[$(seq -s, 1 "$EA_SUBJECTS")]"
    OUT=$EA/results_cho_probe${EA_SUBJECTS}
fi
export EEGROW_BENCH_ROOT=$OUT

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname)"
echo "  model=$MODEL align=$ALIGN seed=$SEED subjects=${EA_SUBJECTS:-all} out=$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# The `bench` env carries an editable .pth pointing at a DIFFERENT checkout. PYTHONPATH
# wins, but assert it: a silent fall-through would run a tree with no alignment code,
# and `align=euclidean` would then quietly produce a second copy of the raw arm.
python - <<'PY' || exit 1
import eegrow
import torch
from eegrow.alignment import euclidean_align  # noqa: F401

assert eegrow.__file__.startswith("/scratch/amounir/eegrow_ea/"), eegrow.__file__
assert torch.cuda.is_available(), "no CUDA on a gpu partition job"
print("guard ok", eegrow.__file__, torch.cuda.get_device_name(0))
PY

TAG=""
if [ "$ALIGN" = "euclidean" ]; then TAG="__easubject"; fi
DONE=$OUT/results/cross_subject/cho2017/${MODEL}${TAG}__seed${SEED}.csv
if [ -f "$DONE" ]; then echo "already done: $DONE"; exit 0; fi

cd $EA/benchmarks
t0=$SECONDS
python run_moabb_hydra.py \
    "model=$MODEL" "align=$ALIGN" \
    dataset=cho2017 $SUBJ_OVERRIDE \
    eval=cross_subject "seed=$SEED" \
    train.patience=200 train.selection_monitor=valid_acc \
    "results_dir=$OUT/results"
rc=$?
echo "[$(date)] $MODEL $ALIGN seed$SEED rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
