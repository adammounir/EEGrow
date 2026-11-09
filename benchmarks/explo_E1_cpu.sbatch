#!/bin/bash
# E1 (analysis/EXPLO_internal_E1.md), exploratory, CPU only, read-only on the campaign A
# archive. MODE=extract (array: one task per model) or read (after every extract).
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
set -uo pipefail
DIR=${DIR:?DIR must be passed with --export}
MODE=${MODE:?MODE must be passed with --export}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd "$DIR"
MODELS=(shallow eegnex conformer ctnet msvtnet)
case $MODE in
  extract) python -u explo_internal_E1.py extract --model "${MODELS[$SLURM_ARRAY_TASK_ID]}" \
             --cells grids/A/campaignA_cells_planned_av5.txt grids/S/campaignA_cells_planned_av5.txt --out out ;;
  read)    python -u explo_internal_E1.py read --out out ;;
  *) echo "REFUSED: MODE=$MODE"; exit 1 ;;
esac
rc=$?
echo "rc=$rc"
exit $rc
