#!/bin/bash
# The 8 cells of the fix250 grid that OOMed on 442404 (fix250-heavy). All 8 are the
# same configuration: schirrmeister2017 (128 channels, the heaviest dataset) x
# grow_eegnex (largest growth tensor S) -- cross_subject seeds 0-4, within_session
# seeds 1,2,4.
#
# The OOM is NOT in training: it is in gromo growth statistics,
#   loop.py:121 grow_step -> compute_statistics
#     -> conv2d_growing_module.py:1019 compute_s_update -> torch.einsum
#   "Tried to allocate 11.70 GiB. GPU 0 has a total capacity of 39.52 GiB
#    of which 8.42 GiB is free; this process has 31.09 GiB in use"
# => needs ~43 GB, so a 40 GB ampere card cannot hold it. expandable_segments was
# already set on 442404 and did not help: this is a genuine allocation peak, not
# fragmentation. The 8 failures all landed on margpu008 while the same config kept
# running on margpu007 -- failure depends on which card the task lands on.
#
# Fix: pin gpu:hopper:1 (H100 80 GB -- margpu012 x2, margpu013 x1, partition gpu-best)
# so batch_size stays 64, same regime as every other cell of the grid. No code change.
#
# IMPORTANT vs eegrow_retry_heavy_h100.sbatch: that older script passed
# "overwrite=false" and NO dataset.resample, which is precisely how those cells ended
# up at the dataset native rate and contaminated the grid. Here resample is pinned to
# 250 explicitly and overwrite=true forces MOABB to recompute instead of reading the
# native-rate result back from its cache.
#SBATCH --job-name=fix250-h100
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-7%2
#SBATCH --output=/scratch/amounir/eegrow/fix250_logs/h100_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/fix250_h100.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED <<< "$LINE"
echo "[$(date)] fix250-h100 task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL dataset=$DS model=$MODEL seed=$SEED"
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
  dataset.resample=250 overwrite=true suffix="fix250_${MODEL}_s${SEED}"
