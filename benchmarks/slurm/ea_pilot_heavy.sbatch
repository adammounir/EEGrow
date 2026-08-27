#!/bin/bash
# Pilot 1, heavy half: the ALIGNED arm (align=euclidean) of cho2017 + physionetmi.
# Same experiment as ea_pilot_light.sbatch -- read that header for the hypothesis, the
# dose-response design, the model pairing, and why the raw arm is never recomputed.
# These are the two high-subject-count points of the dose-response:
#     cho2017     52 subjects, 64 ch
#     physionetmi 109 subjects, 64 ch
#
# Why 64G and not the 48G of fix250_heavy
# ---------------------------------------
# CrossSubjectEvaluation asks the paradigm for every subject at once, and
# euclidean_align writes into a fresh float64 array of the same shape -- so the peak
# RSS is about TWICE the epoch tensor during alignment, then falls back.
# Worst point of this array is cho2017: ~10.5k trials x 64 ch x 751 samples x 8 B
# ~= 4.0 GB per copy, so ~8 GB live at the peak instead of ~4 GB.
# (physionetmi is smaller in trials: ~4.9k x 64 x 751 ~= 1.9 GB per copy.)
# 48G would very likely hold, but an OOM here costs a whole re-run of a 20 h cell for
# the sake of 16 GB of RAM on a node that has plenty. This is CPU RAM, not GPU: it
# does not change the training regime, so the aligned and raw arms stay comparable.
#SBATCH --job-name=ea-heavy
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-39%6
#SBATCH --output=/scratch/amounir/eegrow/ea_logs/heavy_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/ea_pilot_heavy.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED <<< "$LINE"
echo "[$(date)] ea-heavy task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL dataset=$DS model=$MODEL seed=$SEED align=euclidean"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=/scratch/amounir/eegrow/moabb_results

cd /scratch/amounir/eegrow/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" \
  align=euclidean \
  dataset.resample=250 overwrite=true suffix="ea_${MODEL}_s${SEED}"
