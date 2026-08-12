#!/bin/bash
# Stage 1 of the cross-dataset experiment: build the montage-harmonised trial cache.
#
# What this does, and why it is a separate stage
# ---------------------------------------------
# Every cross-dataset run needs the same epochs: 8-32 Hz, 250 Hz, a 3.0 s window
# anchored on each dataset's own interval[0], projected onto the 22 sensorimotor
# electrodes of BCI IV-2a. Recomputing that inside each training job would redo the
# band-pass and the spline solve hundreds of times, and -- worse for the science -- the
# raw and Euclidean-aligned arms could then drift apart on a preprocessing detail. One
# cache, read by every arm, makes the arms differ only where we intend them to.
#
# CPU only: this is IO plus a band-pass plus one 22x22 spline solve per subject. Asking
# for a GPU would queue behind the training jobs for nothing.
#
# Idempotent: build_subject() returns the cached file untouched unless --overwrite, so a
# requeued or re-submitted task costs nothing and a partial dataset can be topped up.
# One task per dataset (not per subject): the per-subject work is minutes, and MOABB's
# download/cache layer is happier with one process per dataset.
#
# The cache lives OUTSIDE the repository, under $EEGROW_POOL_ROOT. It used to default to
# benchmarks/pool/, i.e. inside the tree that is rsync --delete-ed from the laptop, and a
# sync with no local benchmarks/pool/ duly wiped all 276 cached subjects here. The raw
# MOABB data survived so it cost a rebuild, not a re-download -- but hours of derived data
# must not sit in a mirrored directory.
#
# Array indices follow `sorted(tier("core+interp+lowrank"))`:
#   0 bnci2014_001  1 cho2017  2 lee2019_mi  3 physionetmi
#   4 schirrmeister2017  5 shin2017a  6 weibo2014  7 zhou2016
# plus one hand-added index for the negative control:
#   8 bnci2014_004, built with --max-gap-cm 0 (guard disabled)
# physionetmi (109 subjects) and schirrmeister2017 (128 ch at 500 Hz) set the time
# budget; the rest finish in well under an hour.
#
# zhou2016 (`lowrank`) and bnci2014_004 (`extrap`) are built but NOT in the default pool.
# zhou2016's 14 electrodes make the projection rank-14, which breaks CSP and makes
# Euclidean alignment whiten inside a subspace; bnci2014_004's 3 electrodes put 19 of the
# 22 targets beyond the 4.5 cm geometric guard, the worst at 10.1 cm, so it is refused
# unless the guard is switched off on purpose. Both exist to be run as their own arms,
# where the question is exactly "what happens when the projection is not supported".
#SBATCH --job-name=build-pool
#SBATCH --partition=normal,normal-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-8%5
#SBATCH --output=/scratch/amounir/eegrow/pool_logs/build_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

echo "[$(date)] build-pool task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks:/scratch/amounir/eegrow/src
export MNE_DATA=/scratch/amounir/mne_data
export EEGROW_POOL_ROOT=/scratch/amounir/pool_cache

cd /scratch/amounir/eegrow
if [ "${SLURM_ARRAY_TASK_ID}" -eq 8 ]; then
  # the negative control: 19 of 22 channels invented from 3 electrodes, guard off
  srun python benchmarks/pool.py build --dataset bnci2014_004 --max-gap-cm 0
else
  srun python benchmarks/pool.py build --tier core+interp+lowrank \
    --index "${SLURM_ARRAY_TASK_ID}"
fi
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
