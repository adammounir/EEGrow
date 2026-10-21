#!/bin/bash
# Validation campaign of the growth rule (PREREG_validation_growth_rule.md, 21/10/2026):
# one cell per array task, one line of ${CELLS} each:
#
#     LABEL ARM DATASET SEED [hydra overrides ...]
#
# LABEL = model__rule (results under valid_results/LABEL/), ARM = the model config. The
# overrides carry the model's recipe (CATALOGUE § 5) and the growth rule, so a line is
# the WHOLE specification of its cell: nothing is defaulted from the environment.
#
# Same card model as step 0 (RTX 2080 Ti pool, the non-Ti and phantom cards excluded),
# same common protocol as step0_battery.sbatch. Fresh checkout ROOT, never overwritten
# while a job runs; PYTHONPATH beats the `bench` env's editable .pth.
#SBATCH --job-name=valid
#SBATCH --partition=tau,gpu-best
#SBATCH --gres=gpu:turing:1
#SBATCH --exclude=margpu020,margpu021,margpu023,margpu028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
set -f   # the overrides carry [a,b] lists: never let the shell glob them

ROOT=${ROOT:?ROOT must be passed with --export}
CELLS=${CELLS:?CELLS must be passed with --export}
CARD="NVIDIA GeForce RTX 2080 Ti"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1
export EEGROW_DEBUG_GROWTH=1
export EEGROW_SHA="$(tr -d '[:space:]' < $ROOT/.eegrow_sha)"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/benchmarks/$CELLS")
read -r LABEL ARM DS SEED OVERRIDES <<< "$LINE"
if [ -z "${SEED:-}" ]; then echo "REFUSED: malformed line '$LINE'"; exit 1; fi
OUT=$ROOT/valid_results/$LABEL
export EEGROW_BENCH_ROOT=$OUT
# A growing arm states its rule explicitly (never the historical defaults by accident,
# the trap of 18/10): the line must name growth_normalization AND growth_fixed_s.
case "$ARM" in grow_*)
  case "$OVERRIDES" in *train.growth_normalization=*train.growth_fixed_s=*) ;;
    *) echo "REFUSED: growing arm without an explicit rule: $LINE"; exit 1;; esac;;
esac
# The two large datasets exist only as preprocessed epochs in the MOABB cache (their raw
# files are gone from MNE_DATA): read them from there, never re-derive them.
CACHE=""
case "$DS" in cho2017|lee2019_mi) CACHE="cache.enabled=true cache.path=/scratch/amounir/moabb_cache";; esac

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) sha=$EEGROW_SHA"
echo "  label=$LABEL arm=$ARM dataset=$DS seed=$SEED"
echo "  overrides=$OVERRIDES ${SMOKE:-} $CACHE"
GOT="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"
if [ "$GOT" != "$CARD" ]; then echo "REFUSED: card '$GOT' != '$CARD'"; exit 1; fi

ROOT="$ROOT" python - <<'PY' || exit 1
import eegrow, torch
import os; assert eegrow.__file__.startswith(os.environ["ROOT"] + "/"), eegrow.__file__
assert torch.cuda.is_available(), "no CUDA on a gpu job"
import pipelines, inspect
assert "opt_extra" in inspect.getsource(pipelines._build_dl), "stale pipelines.py"
from eegrow.training import loop
assert hasattr(loop, "TINY_PAPER_SQ_NORM"), "stale loop.py"
print("guard ok", eegrow.__file__, torch.cuda.get_device_name(0))
PY

DONE=$OUT/results/within_session/$DS/${ARM}__seed${SEED}.csv
FITS=$OUT/results/within_session/$DS/${ARM}__seed${SEED}__fits.jsonl
if [ -f "$DONE" ]; then echo "already done: $DONE"; exit 0; fi
# A requeued cell must not append to a half-written JSONL (phantom fits): move it aside.
if [ -f "$FITS" ]; then mv "$FITS" "$FITS.aborted.$(date +%s)"; fi

cd $ROOT/benchmarks
t0=$SECONDS
# shellcheck disable=SC2086  # OVERRIDES / SMOKE / CACHE are word lists on purpose
python run_cell.py \
    model=$ARM dataset=$DS eval=within_session "seed=$SEED" \
    "train.max_epochs=${MAX_EPOCHS:-200}" "train.patience=${MAX_EPOCHS:-200}" \
    train.selection_monitor=valid_acc train.smooth_window=5 train.tie_break=earliest \
    train.restore_selected=false train.save_selected=true \
    train.growth_holdout=0.1 train.lr_schedule=null \
    $OVERRIDES ${SMOKE:-} $CACHE \
    "results_dir=$OUT/results"
rc=$?
echo "[$(date)] $LABEL $ARM $DS seed$SEED rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
