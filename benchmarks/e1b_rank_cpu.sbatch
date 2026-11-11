#!/bin/bash
# E1b-rank (analysis/EXPLO_E1b_rank.md), CPU only, read-only on campaign A and the frozen E1b / E1b-bias copies.
# MODE=selftest | run (array: one task = one line of $E1B/out/tasks.txt) | read
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
set -uo pipefail
DIR=${DIR:?DIR must be passed with --export}
E1B=${E1B:?E1B must be passed with --export}
BIAS=${BIAS:?BIAS must be passed with --export}
MODE=${MODE:?MODE must be passed with --export}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd "$E1B"
case $MODE in
  selftest) python -u "$DIR/e1b_rank.py" selftest --e1b "$E1B" --bias "$BIAS" --out "$DIR/selftest" ;;
  run)  python -u "$DIR/e1b_rank.py" run --e1b "$E1B" --bias "$BIAS" --line "$SLURM_ARRAY_TASK_ID" --out "$DIR/out" ;;
  read) python -u "$DIR/e1b_rank.py" read --e1b "$E1B" --bias "$BIAS" --out "$DIR/out" ;;
  *) echo "REFUSED: MODE=$MODE"; exit 1 ;;
esac
rc=$?
echo "rc=$rc"
exit $rc
