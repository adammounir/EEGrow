#!/bin/bash
# Grid B: make the interpolation axis falsifiable, starting with the negative control.
#
# Why the pilot's interpolation null says nothing
# ----------------------------------------------
# `core+interp` minus `core` came out at -0.010 (bd) and +0.001 (grow). The tier adds one
# dataset, Shin2017A: +1740 trials on 36864 (+4.7%) and +29 subjects on 247 (+11.7%). And
# 9 of the 22 target electrodes are recorded natively in the core pool, gap 0.00 cm, so no
# interpolation happens on them at all. What was measured is "does 5% more data move the
# score", not "does interpolation enable cross-montage transfer". A zero there was the
# expected outcome under both hypotheses, which is the definition of an uninformative test.
#
# The geometric guard is also not what it claims
# ----------------------------------------------
# It thresholds the distance from each reconstructed electrode to the nearest recorded one
# at 4.5 cm. Against the measured reconstruction fidelity (Cho2017 cut down to Zhou2016's
# 14-electrode cap, 13 reconstructed electrodes with ground truth) that distance has
# Spearman rho = -0.011 with the achieved correlation. It is not weakly predictive; it is
# not predictive. POz passes at 3.30 cm and reconstructs at corr 0.50, rel_error 2.28,
# because its support is all anterior -- the parietal region of that cap is empty and the
# spline smooths across the hole. Directional measures do better (second-nearest distance
# rho = -0.48, worst-of-three-sectors rho = -0.51) but n = 13 electrodes from one emulation
# cannot calibrate a threshold, and inventing one would repeat exactly the mistake this
# project just spent a day retracting. So the guard is left as it is, honestly labelled as
# a distance filter, and the admission question is answered by measurement instead.
#
# The two arms
# ------------
# core+extrap   BNCI2014_004: 3 electrodes, guard switched off, 19 of 22 channels invented
#               from 3 spatial degrees of freedom, the worst 10.1 cm from anything
#               recorded. This is the negative control and it runs FIRST, because it
#               qualifies every other number on this axis: if the pooled arm cannot tell
#               *this* apart from `core`, then it is insensitive to what the pool contains
#               and no interpolation result here is interpretable -- including the pilot's
#               null. A grid that cannot fail its own negative control is not measuring.
# core+lowrank  Zhou2016: 14 electrodes, geometrically admitted (4.44 cm worst) but the
#               projection is rank 14 in 22 columns, so the covariance is singular. Deep
#               nets invert nothing and should be indifferent; this asks whether
#               rank-deficient interpolated data is usable data.
#
# Both against `core` at align=euclidean, the arm where pooling is least handicapped, with
# the pilot's `core` cells as the matched baseline -- same target, same folds, same seeds,
# so the contrast is paired per subject.
#
# Cost: 12 points, target BNCI2014_001 so 9 folds each. Around 8 GPU-h.
#SBATCH --job-name=xds2-interp
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-11%4
#SBATCH --output=/scratch/amounir/eegrow/xds_logs/interp_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/benchmarks/slurm/xds2_interp.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r TARGET ARM MODEL ALIGN SEED TIER <<< "$LINE"
echo "[$(date)] xds2-interp task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> target=$TARGET arm=$ARM model=$MODEL align=$ALIGN seed=$SEED tier=$TIER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks:/scratch/amounir/eegrow/src
export MNE_DATA=/scratch/amounir/mne_data
export EEGROW_POOL_ROOT=/scratch/amounir/pool_cache

cd /scratch/amounir/eegrow
srun python benchmarks/cross_dataset.py \
  --target "$TARGET" --arm "$ARM" --model "$MODEL" \
  --align "$ALIGN" --seed "$SEED" --pool-tier "$TIER"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
