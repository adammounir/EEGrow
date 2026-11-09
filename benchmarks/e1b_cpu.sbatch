#!/bin/bash
# E1b (analysis/PREREG_E1b_ffn_ablation.md), CPU only, read-only on the campaign A archive.
# MODE=selftest | tasks | run (array: one task = one (dataset, subject) line of out/tasks.txt) | read
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
set -uo pipefail
DIR=${DIR:?DIR must be passed with --export}
MODE=${MODE:?MODE must be passed with --export}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd "$DIR"
CELLS=grids/A/campaignA_cells_planned_av5.txt
case $MODE in
  selftest|tasks) python -u e1b_ablation.py $MODE --cells $CELLS --out out ;;
  run)  python -u e1b_ablation.py run --cells $CELLS --tasks out/tasks.txt --line "$SLURM_ARRAY_TASK_ID" --out out ;;
  read) python -u e1b_ablation.py read --cells $CELLS --tasks out/tasks.txt --out out ;;
  *) echo "REFUSED: MODE=$MODE"; exit 1 ;;
esac
rc=$?
echo "rc=$rc"
exit $rc
