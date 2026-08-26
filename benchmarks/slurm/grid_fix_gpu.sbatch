#!/bin/bash
# The three MISSING fixed controls: fix_shallow, fix_eegnex, fix_sccnet.
# 3 models x 4 cells (the same budget x selection square) x 2 seeds = 24 array tasks.
#
# WHY. Three of the benchmark's four growth pairs were `grow_*` against BRAINDECODE's
# reference at the target width. Those two are not the same code, so the paired delta
# contains growth PLUS everything that separates the codebases -- measured, braindecode's
# stock Xavier init starts bd_shallow 0.34 nats above ln(k) at epoch 0 against +0.08 for
# ours. The strongest result in the table (grow_eegnex -0.089, losing on 9 subjects of 9)
# is exactly the kind of number that confound can manufacture. Only the deep pair had a
# same-class frozen control (fix_deepeeg), and it is the pair that says nothing.
#
# These arms are that control for the other three: same class, same file, same init path,
# same callbacks, built directly at the geometry growth ends on; only `_can_grow` differs.
#
# WIDTH-MATCHING IS VERIFIED, NOT ASSUMED. tests/test_models.py::
# test_fixed_control_matches_grown_geometry grows each arm to its target through the
# benchmark's own callback and diffs the state_dict shapes against the frozen twin.
# That test exists because eegnex failed it: `filter_1` sizes the growable junction AND
# the fixed tail conv block_5 (hence the classifier), and growth widens only the
# junction, so grow_eegnex ends at junction 8 / tail 2 with 70 classifier inputs while
# braindecode's filter_1=8 has 280. The naive `filter_1: 8` control would have been a
# different, larger network -- and the eegnex pair was never width-matched to begin with.
# `filter_1_in` (new, mirroring `w2_in` on the deep arm) decouples the two.
#
# WHAT DOES NOT NEED RE-RUNNING. grow_eegnex is untouched: `filter_1_in` defaults to
# `filter_1`, so its config builds the same network as before, bit for bit. The 64 cells
# of grid 500573 stand.
#
# COST. No growth here, so these are cheaper than the grow_* arms (31.5 s/fold-row at
# full budget, measured on probe 500569); expect bd_shallow's ~24 s. A cell is 18
# fold-rows: ~8 min full budget, ~3 min at patience 20, so ~2 h 15 of compute and ~45 min
# of wall clock at 3 concurrent tasks.
#
# gpu:hopper:1 is pinned, and NOT for memory (these nets are ~50k params). The whole
# point is a fit-by-fit paired comparison against the grow_* cells of grid 500573, which
# ran on hopper; a pascal cell here would put a hardware difference inside the paired
# difference.
#SBATCH --job-name=fix-controls
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-23%3
#SBATCH --output=/scratch/amounir/eegrow_budget/grid_models_logs/%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

BUD=/scratch/amounir/eegrow_budget
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench

export PYTHONPATH=$BUD/src
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BUD/grid_fix_cells.txt")
read -r MODEL ARM PATIENCE SEL SCHED SEED <<< "$LINE"
if [ -z "${MODEL:-}" ]; then echo "no cell for task $SLURM_ARRAY_TASK_ID"; exit 1; fi
OUT=$BUD/grid_models/$MODEL/$ARM
export EEGROW_BENCH_ROOT=$OUT

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname)"
echo "  model=$MODEL arm=$ARM patience=$PATIENCE sel=$SEL sched=$SCHED seed=$SEED"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# The `bench` env carries an editable .pth pointing at the OTHER checkout
# (/scratch/amounir/eegrow), which holds uncommitted v5 work. PYTHONPATH wins, but that
# is worth asserting rather than assuming: a silent fall-through would run the wrong
# code, and the frozen-control arms are exactly the ones that depend on this checkout's
# `filter_1_in`.
python - <<'PY' || exit 1
import inspect

import eegrow
import torch
from eegrow.models.growing_eegnex import GrowingEEGNeX

assert eegrow.__file__.startswith("/scratch/amounir/eegrow_budget/"), eegrow.__file__
assert torch.cuda.is_available(), "no CUDA on a gpu partition job"
assert "filter_1_in" in inspect.signature(GrowingEEGNeX.__init__).parameters, \
    "this checkout predates filter_1_in: fix_eegnex would build the wrong geometry"
print("guard ok", eegrow.__file__, torch.cuda.get_device_name(0))
PY

# Resumable under --requeue: the per-cell CSV is written last, so its presence means the
# cell finished and a requeued task must not redo the fits.
DONE=$OUT/results/within_session/bnci2014_001/${MODEL}__seed${SEED}.csv
if [ -f "$DONE" ]; then echo "already done: $DONE"; exit 0; fi

cd $BUD/benchmarks
t0=$SECONDS
python run_cell.py \
    "model=$MODEL" dataset=bnci2014_001 eval=within_session "seed=$SEED" \
    "train.patience=$PATIENCE" "train.selection_monitor=$SEL" \
    "train.lr_schedule=$SCHED" "results_dir=$OUT/results"
rc=$?
echo "[$(date)] $MODEL $ARM seed$SEED rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
