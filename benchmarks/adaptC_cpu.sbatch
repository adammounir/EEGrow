#!/bin/bash
# Reading 4 (analysis/PREREG_growth_adaptive_size.md + PREREG_growth_signature.md), CPU part:
# MODE=tasks, merge (tune.json) or read (adapt_C.md, signature_C.md). No GPU.
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
set -uo pipefail
DIR=${DIR:?DIR must be passed with --export}
MODE=${MODE:?MODE must be passed with --export}
export CUDA_VISIBLE_DEVICES=
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd "$DIR"
case $MODE in
  tasks) python -u adapt_C.py tasks --cells ../grids/A/campaignA_cells_planned_av5.txt --out out ;;
  merge) python -u adapt_C.py merge --out out ;;
  read)  python -u adapt_C.py read --out out --adapt-b adapt_B_reading3.json ;;
  *) echo "REFUSED: MODE=$MODE"; exit 1 ;;
esac
rc=$?
echo "rc=$rc"
exit $rc
