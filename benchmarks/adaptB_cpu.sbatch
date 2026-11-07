#!/bin/bash
# Reading 3 (analysis/PREREG_growth_adaptation_B.md), CPU part: MODE=merge (tune.json) or
# MODE=read (adapt_B.md / .json). No GPU.
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
set -uo pipefail
DIR=${DIR:?DIR must be passed with --export}
MODE=${MODE:?MODE must be passed with --export}
export CUDA_VISIBLE_DEVICES=
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd "$DIR"
case $MODE in
  merge|read) python -u adapt_B.py $MODE --out out ;;
  *) echo "REFUSED: MODE=$MODE"; exit 1 ;;
esac
rc=$?
echo "rc=$rc"
exit $rc
