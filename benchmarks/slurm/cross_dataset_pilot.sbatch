#!/bin/bash
# Cross-dataset pilot: does pooling datasets beat training on one?
#
# The question, stated so it can come out negative
# -----------------------------------------------
# Target BNCI2014_001, always scored on the same held-out subjects (leave-one-subject-out,
# 9 folds, matching MOABB's CrossSubjectEvaluation so the baseline arm is protocol-
# compatible with the published grid). Three training sets:
#   within  the target's other 8 subjects            <- the matched baseline
#   lodo    the 5 other pool datasets, target unseen <- zero-shot transfer
#   pooled  those 5 datasets + the target's 8        <- the only arm with MORE data
# `pooled - within` is the answer. `lodo - within` says whether transfer works at all.
#
# LOSO is kept for every arm rather than a cheaper 3-fold. A 3-fold would train `within`
# on 6 subjects instead of 8, weakening the baseline -- and the baseline is precisely the
# thing that must not be handicapped, or the pooled gain is manufactured.
#
# Why three alignment levels
# --------------------------
# none / scale / euclidean. Pooling raw datasets is confounded by amplitude: different
# amplifiers and references put them orders of magnitude apart, so one dataset can
# dominate the loss for non-neural reasons. Whitening removes that as a side effect, so
# without `scale` (per-subject std, nothing else) a gain under `euclidean` could be either
# whitening or plain rescaling. Only euclidean - scale is evidence about whitening.
#
# Why two pool tiers
# ------------------
# core        6 datasets, 220 subjects, every target electrode recorded natively.
# core+interp adds Shin2017A: 29 subjects reachable ONLY through interpolation, whose
#             30-electrode 10-05 cap keeps the projection full rank (22). core vs
#             core+interp is therefore a clean test of what interpolation buys, with rank
#             held constant. Zhou2016 is deliberately absent: 14 electrodes make the
#             projection rank-14, which breaks CSP and makes Euclidean alignment whiten
#             inside a subspace -- it would confound the two axes this grid separates.
#             The `within` arm never reads the pool, so it is not repeated per tier.
#
# Interpolation is already validated (benchmarks/interp_fidelity.py): reconstructing the
# 22 target channels from Zhou2016's 14-electrode cap, on Cho2017 where the truth is
# recorded, costs -0.0032 accuracy (p = 0.88) cross-subject. So a null result here means
# "no transferable information", not "the signal was destroyed".
#
# Grid: 90 points. 54 core (3 arms x 3 align x 2 models x 3 seeds) + 36 core+interp
# (2 arms x 3 align x 2 models x 3 seeds). Models are the growth pair grow_shallow /
# bd_shallow, so every cell also measures whether growth exploits a bigger, more
# heterogeneous pool better than a fixed net -- the eegrow question, asked where extra
# data actually exists.
#SBATCH --job-name=xds-pilot
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-89%8
#SBATCH --output=/scratch/amounir/eegrow/xds_logs/pilot_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/benchmarks/slurm/cross_dataset_pilot.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r TARGET ARM MODEL ALIGN SEED TIER <<< "$LINE"
echo "[$(date)] xds task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> target=$TARGET arm=$ARM model=$MODEL align=$ALIGN seed=$SEED tier=$TIER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks:/scratch/amounir/eegrow/src
export MNE_DATA=/scratch/amounir/mne_data

cd /scratch/amounir/eegrow
srun python benchmarks/cross_dataset.py \
  --target "$TARGET" --arm "$ARM" --model "$MODEL" \
  --align "$ALIGN" --seed "$SEED" --pool-tier "$TIER"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
