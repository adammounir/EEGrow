#!/bin/bash
# Reading 3 (analysis/PREREG_growth_adaptation_B.md), GPU part: MODE=tune (one line of
# tune_tasks.txt per array task) or MODE=run (one line of run_tasks.txt). A subject whose
# json exists is skipped, so a preempted/requeued task resumes where it stopped.
#SBATCH --partition=gpu-best,tau,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=turing|ampere|hopper
#SBATCH --exclude=margpu020,margpu028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
DIR=${DIR:?DIR must be passed with --export}
MODE=${MODE:?MODE must be passed with --export}
TID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh && conda activate bench
export MNE_DATA=/scratch/amounir/mne_data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$DIR"
G=../grids/A/campaignA_cells_planned_av5.txt
NAME="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed -n 1p)"
echo "[$(date)] task=$TID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) card='$NAME'"
python - <<'PY'
import torch
free, tot = torch.cuda.mem_get_info(0)
x = torch.randn(2, 1, 32, 32, device="cuda"); w = torch.randn(4, 1, 3, 3, device="cuda")
float(torch.nn.functional.conv2d(x, w).sum())
assert tot >= 10 * 2**30, f"card too small: {tot / 2**30:.1f} GiB"
print("guard ok", torch.cuda.get_device_name(0), f"{tot / 2**30:.1f} GiB")
PY
rc=$?
if [ $rc -ne 0 ]; then
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 8 ]; then
    echo "CARD PROBE FAILED on $(hostname): holding 60 s, requeue"; sleep 60; scontrol requeue "$TID"; sleep 120; exit 0
  fi
  echo "REFUSED: card probe failed ${SLURM_RESTART_COUNT} times"; exit 1
fi
case $MODE in
  tune) read -r M S <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" out/tune_tasks.txt)"
        python -u adapt_B.py tune --cells $G --model "$M" --subject "$S" --out out ;;
  run)  python -u adapt_B.py run --cells $G --tasks out/run_tasks.txt --line "$SLURM_ARRAY_TASK_ID" --out out ;;
  *) echo "REFUSED: MODE=$MODE"; exit 1 ;;
esac
rc=$?
echo "[$(date)] task=$TID rc=$rc"
exit $rc
