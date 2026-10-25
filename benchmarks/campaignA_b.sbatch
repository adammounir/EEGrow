#!/bin/bash
# Campaign A, amendment 2 (analysis/PREREG_campaign_A.md § 6): the pending tasks of the
# main wave resubmitted with this gate; identical to campaignA.sbatch but for the card
# gate and the exclusion list (diff the two files). One array task = one line of ${ARRAY}, a
# comma-separated list of cell ids; each id is a line of ${CELLS}
# (campaignA_make_cells.py), run in sequence:
#
#     CELL_ID PROT DS MODEL ARM SEED SUBJ RESULTS_DIR CSV [hydra overrides ...]
#
# A cell whose CSV exists is done and skipped, so a preempted/requeued task (tau has
# PriorityTier 100 over gpu-best/gpu, PreemptMode=REQUEUE) resumes at its first
# unfinished cell; a half-written JSONL is moved aside first (phantom fits otherwise).
#
# Card -> env: sm >= 7.5 -> `bench` (cu130); Tesla V100-PCIE-32GB / P100-PCIE-16GB and,
# since amendment 1 of PREREG_env_equivalence.md, GTX 1080 Ti / V100-SXM2-32GB ->
# `bench126` (cu126); anything else is REFUSED. Excluded nodes: margpu004 (V100S,
# unreachable 25/10) and margpu028 (RTX 2080 8 GB: GPU OOM on 128-channel cells).
# margpu020 is back: its dead card fails the CUDA probe below, so a task landing on it is
# held 60 s and requeued (cuda-ordinal-vs-nvidia-smi), at most 8 times. Tasks of the
# 80 GB arrays are also given --cpus-per-task=9 and --exclude margpu032 at submission: the
# cluster does not reserve memory (CR_CORE), cores are what caps tasks per node.
#
# --mem / --time / --array are given at submission, per array file (its name says them).
#SBATCH --job-name=campA
#SBATCH --partition=tau,gpu-best,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu004,margpu028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
set -f   # the overrides carry [a,b] lists: never let the shell glob them

ROOT=${ROOT:?ROOT must be passed with --export}
CELLS=${CELLS:?CELLS must be passed with --export}
ARRAY=${ARRAY:?ARRAY must be passed with --export}
TID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
IDS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/benchmarks/$ARRAY")
if [ -z "$IDS" ]; then echo "REFUSED: no line $SLURM_ARRAY_TASK_ID in $ARRAY"; exit 1; fi

NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n 1p)"
case "$NAME" in
  # exactly the two cards of the equivalence test (read/env_equiv.md, 25/10)
  "Tesla V100-PCIE-32GB"|"Tesla P100-PCIE-16GB") ENV=bench126 ;;
  # amendment 1 of the equivalence test (25/10 evening): both arms ÉQUIVALENT
  "NVIDIA GeForce GTX 1080 Ti"|"Tesla V100-SXM2-32GB") ENV=bench126 ;;
  *V100*|*P100*|*1080*) echo "REFUSED: $NAME was never tested (PREREG § 2)"; exit 1 ;;
  *) if awk -v c="$CAP" 'BEGIN{exit !(c+0 >= 7.5)}'; then ENV=bench
     else echo "REFUSED: $NAME (sm $CAP) has no tested env"; exit 1; fi ;;
esac
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate $ENV
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export HYDRA_FULL_ERROR=1
export EEGROW_SHA="$(tr -d '[:space:]' < $ROOT/.eegrow_sha)"
echo "[$(date)] task=$TID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) gpus=${SLURM_JOB_GPUS:-?} card='$NAME' sm=$CAP env=$ENV sha=$EEGROW_SHA"
echo "  cells=$IDS"

ROOT="$ROOT" python - <<'PY'
import os, torch
import eegrow
assert eegrow.__file__.startswith(os.environ["ROOT"] + "/"), eegrow.__file__
free, tot = torch.cuda.mem_get_info(0)
a = torch.randn(64, 64, device="cuda"); x = torch.randn(2, 1, 32, 32, device="cuda")
w = torch.randn(4, 1, 3, 3, device="cuda")
float((a @ a).sum()); float(torch.nn.functional.conv2d(x, w).sum())
from eegrow.training import loop
assert hasattr(loop, "_randomize_candidates"), "stale loop.py"
print("guard ok", torch.__version__, torch.cuda.get_device_name(0), f"{free / 2**30:.1f} GiB free")
PY
rc=$?
if [ $rc -ne 0 ]; then
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 8 ]; then
    echo "CARD PROBE FAILED (rc=$rc) on $(hostname) gpus=${SLURM_JOB_GPUS:-?}: holding 60 s, requeue"
    sleep 60; scontrol requeue "$TID"; sleep 120; exit 0
  fi
  echo "REFUSED: card probe failed $SLURM_RESTART_COUNT times"; exit 1
fi

cd $ROOT/benchmarks
FAILED=0
for ID in ${IDS//,/ }; do
  LINE=$(sed -n "$((ID + 1))p" "$ROOT/benchmarks/$CELLS")
  read -r CID PROT DS MODEL ARM SEED SUBJ RDIR CSV OVERRIDES <<< "$LINE"
  if [ "$CID" != "$ID" ] || [ -z "${OVERRIDES:-}" ]; then
    echo "REFUSED: line $((ID + 1)) of $CELLS is not cell $ID: '$LINE'"; FAILED=$((FAILED + 1)); continue
  fi
  if [ -f "$CSV" ]; then echo "CELL $ID already done: $CSV"; continue; fi
  FITS="${CSV%.csv}__fits.jsonl"
  # A requeued cell must not append to a half-written JSONL: move it aside. Its side
  # files (weights, logits) are keyed by fit_uid and only reachable from a JSONL line.
  if [ -f "$FITS" ]; then mv "$FITS" "$FITS.aborted.$(date +%s)"; fi
  echo "[$(date)] CELL $ID start $PROT $DS $MODEL $ARM seed$SEED subj$SUBJ"
  t0=$SECONDS
  # shellcheck disable=SC2086  # OVERRIDES is a word list on purpose
  python run_cell.py dataset=$DS eval=$PROT "seed=$SEED" "only_subject=$SUBJ" \
      "results_dir=$RDIR" $OVERRIDES ${EXTRA:-}
  rc=$?
  if [ $rc -eq 0 ] && [ ! -f "$CSV" ]; then echo "CELL $ID rc=0 but no $CSV"; rc=99; fi
  echo "[$(date)] CELL $ID rc=$rc wall=$((SECONDS - t0))s card='$NAME' env=$ENV"
  [ $rc -ne 0 ] && FAILED=$((FAILED + 1))
done
echo "[$(date)] task=$TID done, failed=$FAILED"
[ $FAILED -eq 0 ]
