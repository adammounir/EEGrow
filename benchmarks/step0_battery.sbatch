#!/bin/bash
# Step-0 training battery (PREREG_step0_training_battery.md § 3): 12 cells, one per
# array task, one line of step0_cells.txt each (`arm candidate lr wd seed`).
#
# One card model for every cell (PREREG § 4, effect of the card): RTX 2080 Ti, the 7
# turing cards of margpu017/018 (probed 14/10 with nvidia-smi from srun). `gres=turing`
# is not a card model by itself, so the job also REFUSES any other card by name.
#
# Fresh checkout ROOT, never overwritten while a job runs. PYTHONPATH beats the `bench`
# env's editable .pth that points at /scratch/amounir/eegrow (the trap of 10/09).
#SBATCH --job-name=s0-battery
#SBATCH --partition=tau,gpu-best
#SBATCH --gres=gpu:turing:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --output=/scratch/amounir/eegrow_step0/logs/battery_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

ROOT=/scratch/amounir/eegrow_step0
CARD="NVIDIA GeForce RTX 2080 Ti"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1
export EEGROW_SHA="$(tr -d '[:space:]' < $ROOT/.eegrow_sha)"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/benchmarks/step0_cells.txt")
read -r ARM CAND LR WD SEED <<< "$LINE"
OUT=$ROOT/step0_results/$CAND
export EEGROW_BENCH_ROOT=$OUT

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) sha=$EEGROW_SHA"
echo "  arm=$ARM candidate=$CAND lr=$LR wd=$WD seed=$SEED"
GOT="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"
if [ "$GOT" != "$CARD" ]; then echo "REFUSED: card '$GOT' != '$CARD'"; exit 1; fi

python - <<'PY' || exit 1
import eegrow, torch
assert eegrow.__file__.startswith("/scratch/amounir/eegrow_step0/"), eegrow.__file__
assert torch.cuda.is_available(), "no CUDA on a gpu job"
import pipelines, inspect
assert "weight_decay" in inspect.getsource(pipelines._build_dl), "stale pipelines.py"
print("guard ok", eegrow.__file__, torch.cuda.get_device_name(0))
PY

DONE=$OUT/results/within_session/bnci2014_001/${ARM}__seed${SEED}.csv
FITS=$OUT/results/within_session/bnci2014_001/${ARM}__seed${SEED}__fits.jsonl
if [ -f "$DONE" ]; then echo "already done: $DONE"; exit 0; fi
# A requeued or relaunched cell must not append to a half-written JSONL (phantom fits):
# move it aside, never delete it.
if [ -f "$FITS" ]; then mv "$FITS" "$FITS.aborted.$(date +%s)"; fi

cd $ROOT/benchmarks
t0=$SECONDS
python run_cell.py \
    model=$ARM dataset=bnci2014_001 eval=within_session "seed=$SEED" \
    train.max_epochs=200 train.patience=200 \
    train.selection_monitor=valid_acc train.smooth_window=5 train.tie_break=earliest \
    train.restore_selected=false train.save_selected=true \
    train.growth_holdout=0.1 "train.scaling_grid=[0.0,0.25,0.5,1.0,2.0,4.0]" \
    train.lr_schedule=null "train.lr=$LR" "train.weight_decay=$WD" \
    "results_dir=$OUT/results"
rc=$?
echo "[$(date)] $ARM $CAND seed$SEED rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
