#!/bin/bash
# One array task = one cell of the classical-baseline arm, written into the *v5*
# results tree so the deep-vs-classical contrast becomes expressible on the campaign
# that has a provenance row. Point files come from make_ml_grid.py.
#
# Parameterised rather than copied four times: partition, memory, wall clock and array
# throttle are what differ between the four passes (26 minutes / 2 h / 27 h / 95 h per
# cell), and everything else -- environment, cache, suffix scheme, results tree -- must
# be identical across them or the passes are not one campaign. submit_ml_v5.sh holds
# the four sizings.
#
#   POINTS   path to the TSV (eval, dataset, model, seed) point file   [required]
#   OVERWRITE  true|false, passed to MOABB                             [default true]
#
#SBATCH --job-name=ml_v5
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/amounir/logs/ml_v5/%x_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

: "${POINTS:?set POINTS=/path/to/points.txt}"
OVERWRITE="${OVERWRITE:-true}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED <<< "$LINE"
[ -z "${MODEL:-}" ] && { echo "no point at index $SLURM_ARRAY_TASK_ID of $POINTS"; exit 1; }

RESULTS_DIR=/scratch/amounir/results_v5
CACHE=/scratch/amounir/moabb_cache
# cfg.model.label, not the config name: run_moabb_hydra writes <label>__seed<n>.csv
# and every analysis keys on the label (csp_lda, not ml_csp_lda).
LABEL="${MODEL#ml_}"
OUT="$RESULTS_DIR/$EVAL/$DS/${LABEL}__seed${SEED}.csv"

echo "[$(date -Is)] ml_v5 task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0}" \
     "node=$(hostname) -> eval=$EVAL dataset=$DS model=$MODEL seed=$SEED"

# Idempotent: a requeued task, or a resubmitted array, must not recompute a cell that
# already landed. The runner checks the file the analysis actually reads, not a claim.
if [ -s "$OUT" ]; then
  echo "already done: $OUT"
  exit 0
fi

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=/scratch/amounir/eegrow/moabb_results
# Opposite of pack_run.sh, and deliberately: that script pins one BLAS thread because
# it packs K co-tenants onto a node and they would otherwise fight for cores. Here one
# task owns its 8 cores and the work *is* BLAS -- covariance estimation, tangent-space
# projection, the SVM kernel -- so the threads are the only parallelism available.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

cd /scratch/amounir/eegrow/benchmarks
# The suffix keys MOABB's HDF5 store, and the store is per (eval, dataset): a suffix
# shared by two cells of the same dataset means two processes opening one file for
# writing. Per (model, seed) makes that impossible.
# No dataset.resample override: the 12 dataset YAMLs are pinned to 250.0, and
# check_cache.py confirmed the warm cache is keyed on exactly that paradigm, so the
# whole arm is a cache hit. Passing it again would be harmless but the pin is the
# record.
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" \
  cache.enabled=true cache.path="$CACHE" \
  results_dir="$RESULTS_DIR" \
  overwrite="$OVERWRITE" suffix="mlv5_${LABEL}_s${SEED}"
