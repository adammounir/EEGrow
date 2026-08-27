#!/bin/bash
#SBATCH --job-name=eegrow_memgrid
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclude=margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/logs/memgrid_%j.log
# Peak GPU memory for all (model x dataset) cells, so K can be set per memory class
# instead of globally. See benchmarks/profile_grid_memory.py for the method and for
# why the previously measured figures are stale (they predate the growth cap).
#
# One GPU is enough and more would be wrong: co-tenants would contaminate the
# high-water mark this is trying to measure.
#
#   sbatch benchmarks/slurm/profile_grid_memory.sbatch
#
# --exclude: pascal nodes are Tesla P100 (sm_60) and this torch build ships
# sm_75..sm_120 only -- any kernel launch there dies with "no kernel image is
# available for execution on the device". Measured on margpu029.
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
cd /scratch/amounir/eegrow/benchmarks
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export EEGROW_BENCH_ROOT=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset EEGROW_CUDA_FRACTION
module load cuda/13.1 2>/dev/null || true

# PROFILE_OUT is overridable because the numbers in profile_mem/grid_memory.json were
# taken before the 2026-08-16 fix to GromoGrowth.on_epoch_end, which stopped the growth
# step materialising the whole epoch on the card. Every growing arm's figure in that
# file therefore carries a training fold that the code no longer allocates, and the
# driver resumes from whatever the file already contains -- so re-measuring into it
# would silently skip all 108 cells. A fresh path is what makes the re-measurement
# happen at all.
echo "=== node=$(hostname) out=${PROFILE_OUT:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python profile_grid_memory.py \
    --out ${PROFILE_OUT:-/scratch/amounir/eegrow/benchmarks/profile_mem/grid_memory.json} \
    --shapes /scratch/amounir/eegrow/benchmarks/profile_mem/shapes.json \
    --cache /scratch/amounir/moabb_cache

echo "=== DONE $(date -Is)"
