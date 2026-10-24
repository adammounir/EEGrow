#!/bin/bash
# Timing probe for campaign A (24/10/2026): one line of ${CELLS} per array task,
#
#     EVAL ARM DATASET SEED [hydra overrides ...]
#
# Trains fold 0 only, for EPOCHS epochs (timing_probe.py), and writes one JSON under
# $ROOT/probe/. The card type is chosen at submission (--constraint=<type> on the sbatch
# command line); the script records what it actually got, it does not gate on it.
# All 12 datasets are read from the preprocessed MOABB cache (several raw folders are
# gone from MNE_DATA).
#SBATCH --job-name=probe
#SBATCH --partition=tau,gpu-best,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=turing|rtx|ampere|hopper
#SBATCH --exclude=margpu020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append

set -uo pipefail
set -f

ROOT=${ROOT:?ROOT must be passed with --export}
CELLS=${CELLS:?CELLS must be passed with --export}
TAG=${TAG:?TAG must be passed with --export}
EPOCHS=${EPOCHS:-10}
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/benchmarks/$CELLS")
read -r EVAL ARM DS SEED OVERRIDES <<< "$LINE"
if [ -z "${SEED:-}" ]; then echo "REFUSED: malformed line '$LINE'"; exit 1; fi
OUT=$ROOT/probe/$TAG/${EVAL}__${DS}__${ARM}__seed${SEED}.json
if [ -f "$OUT" ]; then echo "already done: $OUT"; exit 0; fi
echo "[$(date)] task=$SLURM_ARRAY_TASK_ID node=$(hostname) $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed -n 1p

cd $ROOT/benchmarks
# shellcheck disable=SC2086
python timing_probe.py --eval "$EVAL" --epochs "$EPOCHS" --out "$OUT" -- \
    model=$ARM dataset=$DS "seed=$SEED" $OVERRIDES \
    cache.enabled=true cache.path=/scratch/amounir/moabb_cache
echo "[$(date)] rc=$?"
