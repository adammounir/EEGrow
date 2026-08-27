#!/bin/bash
# Grid A: give `pooled - within` enough subjects to answer.
#
# Why this grid exists
# --------------------
# The pilot's verdict on pooling was "+0.036, p = 0.29" over the 9 subjects of
# BNCI2014_001. That is not a null result, it is an absence of power: with a per-subject sd
# of 0.094 the design's minimum detectable effect was 0.101 -- a ten-point gain -- and the
# 95% CI ran from -0.037 to +0.109. Reporting that as "pooling does not help" would have
# been a claim the data cannot support in either direction.
#
# Cho2017 has 52 subjects and records all 22 target electrodes natively, so it is a `core`
# dataset and this grid is independent of the interpolation question. 52 + the pilot's 9
# clears the n = 56 the observed effect needs at 80% power.
#
# The interesting quantity is not the mean
# ----------------------------------------
# The pilot's per-subject deltas ran from -0.060 to +0.226 while seed-to-seed sd was 0.021.
# The spread is four times the noise: pooling helps some subjects a great deal and hurts
# others, and the group mean averages that away. With 52 subjects the regression of the
# delta on the within-subject score becomes testable, which is the result worth having --
# "pooling rescues the subjects a single dataset cannot decode" is a mechanism, "+0.036" is
# not.
#
# Ordering is the stopping rule
# -----------------------------
# align=euclidean on both arms comes first (12 points): that is the power question and it
# answers in roughly a day. Then pooled/scale, replicating the one effect the pilot did
# establish at adequate power (euclidean - scale = +0.037, 8/9 subjects, p = 0.009), then
# pooled/none to complete the ladder. Cancelling after the first block still answers the
# question this grid was launched for.
#
# lodo is absent: it was +0.012 (p = 0.71), it is not the claim, and it is cheap to add
# later on whichever cells turn out to matter.
#
# Cost: 24 points. A `pooled` cell is 52 LOSO folds x ~300 s = ~4.5 h, a `within` cell ~1 h,
# so 18 pooled + 6 within is around 87 GPU-h -- of which the first 12 points (the power
# question: within and pooled at align=euclidean) are ~33 h. The remaining 12 replicate the
# alignment ladder at n = 52, which is worth it only if the first block is clean; points are
# ordered so that cancelling after task 11 loses nothing this grid was launched for.
#SBATCH --job-name=xds2-cho
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-23%6
#SBATCH --output=/scratch/amounir/eegrow/xds_logs/cho_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/benchmarks/slurm/xds2_cho.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r TARGET ARM MODEL ALIGN SEED TIER <<< "$LINE"
echo "[$(date)] xds2-cho task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> target=$TARGET arm=$ARM model=$MODEL align=$ALIGN seed=$SEED tier=$TIER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks:/scratch/amounir/eegrow/src
export MNE_DATA=/scratch/amounir/mne_data
# outside the repo on purpose: benchmarks/pool/ is inside the rsync --delete mirror and a
# sync from the laptop wiped the whole cache once already
export EEGROW_POOL_ROOT=/scratch/amounir/pool_cache

cd /scratch/amounir/eegrow
srun python benchmarks/cross_dataset.py \
  --target "$TARGET" --arm "$ARM" --model "$MODEL" \
  --align "$ALIGN" --seed "$SEED" --pool-tier "$TIER"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
