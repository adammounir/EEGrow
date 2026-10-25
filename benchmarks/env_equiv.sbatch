#!/bin/bash
# Equivalence test of the bench126 env (torch 2.13.0+cu126, same version as bench's cu130
# build, but with sm_60/sm_70 kernels): may campaign A also use the volta and pascal cards?
# PREREG: analysis/PREREG_env_equivalence.md. One line of ${CELLS} per array task:
#
#     LABEL ENV ARM SEED [hydra overrides ...]
#
# The card type is chosen at submission (--constraint / --partition on the command line);
# EXPECT, when set, refuses any other card name. Full within-session cell, validation
# protocol (200 epochs), results under $ROOT/equiv/LABEL/.
#SBATCH --job-name=equiv
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append

set -uo pipefail
set -f

ROOT=${ROOT:?ROOT must be passed with --export}
CELLS=${CELLS:?CELLS must be passed with --export}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/benchmarks/$CELLS")
read -r LABEL ENV ARM SEED OVERRIDES <<< "$LINE"
if [ -z "${SEED:-}" ]; then echo "REFUSED: malformed line '$LINE'"; exit 1; fi
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV" || { echo "REFUSED: no env $ENV"; exit 1; }
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1
export EEGROW_SHA="$(tr -d '[:space:]' < $ROOT/.eegrow_sha)"
OUT=$ROOT/equiv/$LABEL

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID node=$(hostname) $LINE"
GOT="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"
echo "card: $GOT"
if [ -n "${EXPECT:-}" ] && [ "$GOT" != "$EXPECT" ]; then echo "REFUSED: card '$GOT' != '$EXPECT'"; exit 1; fi
ROOT="$ROOT" python - <<'PY' || exit 1
import os, eegrow, torch
assert eegrow.__file__.startswith(os.environ["ROOT"] + "/"), eegrow.__file__
assert torch.cuda.is_available(), "no CUDA on a gpu job"
x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()   # a kernel actually runs
print("guard ok", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

DONE=$OUT/results/within_session/bnci2014_001/${ARM}__seed${SEED}.csv
if [ -f "$DONE" ]; then echo "already done: $DONE"; exit 0; fi
cd $ROOT/benchmarks
t0=$SECONDS
# shellcheck disable=SC2086
python run_cell.py \
    model=$ARM dataset=bnci2014_001 eval=within_session "seed=$SEED" \
    train.max_epochs=200 train.patience=200 \
    train.selection_monitor=valid_acc train.smooth_window=5 train.tie_break=earliest \
    train.restore_selected=false train.save_selected=false \
    train.growth_holdout=0.1 train.lr_schedule=null \
    $OVERRIDES \
    cache.enabled=true cache.path=/scratch/amounir/moabb_cache \
    "results_dir=$OUT/results"
rc=$?
echo "[$(date)] $LABEL rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
