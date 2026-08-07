#!/bin/bash
# Re-run, at the benchmark's canonical 250 Hz, every cell whose latest run had landed
# on the NATIVE sampling rate. Cause: the eegrow_retry_*/eegrow_tail scripts never
# passed dataset.resample, so relaunched cells silently used the dataset's native rate
# while the production grid ran at 250 Hz. A grow_X/bd_X pair split across two rates
# compares preprocessing, not growth, so those pairs were unusable.
# dataset.resample is now pinned in the dataset configs; it is also passed explicitly
# here, and overwrite=true + a fresh suffix force MOABB to recompute rather than read
# back the native-rate result from its cache.
#SBATCH --job-name=fix250-normal
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-69%8
#SBATCH --output=/scratch/amounir/eegrow/fix250_logs/normal_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/fix250_normal.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED <<< "$LINE"
echo "[$(date)] fix250-normal task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL dataset=$DS model=$MODEL seed=$SEED"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=/scratch/amounir/eegrow/moabb_results

cd /scratch/amounir/eegrow/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" \
  dataset.resample=250 overwrite=true suffix="fix250_${MODEL}_s${SEED}"
