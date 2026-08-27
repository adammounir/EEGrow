#!/bin/bash
#SBATCH --job-name=eegrow_profile
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/eegrow/benchmarks/slurm/logs/profile_%j.log
# Measure what one benchmark cell actually costs, cold vs warm.
#
# The campaign's cost model was never measured, only assumed. The recorded fit
# time says a cross_session cell trains in single-digit minutes, yet those cells
# held a whole GPU for hours. This job accounts for the difference: it runs the
# same cell twice, once with an empty epoch cache and once with a warm one, so
# the preprocessing share stops being a guess.
#
#   sbatch benchmarks/slurm/profile_cache.sbatch
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
cd /scratch/amounir/eegrow/benchmarks
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export EEGROW_BENCH_ROOT=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
module load cuda/13.1 2>/dev/null || true

# The cache lives OUTSIDE the repo on purpose. An `rsync --delete` of the repo
# once erased the epoch cache along with 276 subjects of pooled data; anything
# expensive to rebuild does not belong under a directory that gets mirrored.
CACHE=/scratch/amounir/moabb_cache
mkdir -p "$CACHE" slurm/logs profile

DS=${DS:-bnci2014_001}
EV=${EV:-cross_session}
MODEL=${MODEL:-grow_shallow}

echo "=== node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=== cell: $EV / $DS / $MODEL"

# COLD: the cache directory for this dataset is cleared first, so the run pays
# the full preprocessing exactly as every job of the production campaign did.
echo "=== [1/3] COLD (cache written but empty on entry)"
rm -rf "${CACHE:?}"/*"${DS}"* 2>/dev/null || true
python profile_cell.py \
    model=$MODEL dataset=$DS eval=$EV seed=0 \
    cache.enabled=true cache.path=$CACHE 2>&1 | tail -40

# WARM: identical command. Everything it reads was produced by the run above,
# which is the situation of jobs 2..70 on this dataset in a real grid.
echo "=== [2/3] WARM (same command, cache now populated)"
python profile_cell.py \
    model=$MODEL dataset=$DS eval=$EV seed=1 \
    cache.enabled=true cache.path=$CACHE 2>&1 | tail -40

# NO CACHE: the production behaviour, as a control. Without it the warm/cold gap
# could be credited to the OS page cache rather than to `cache_config`.
echo "=== [3/3] NO CACHE (production behaviour, control)"
python profile_cell.py \
    model=$MODEL dataset=$DS eval=$EV seed=2 \
    cache.enabled=false 2>&1 | tail -40

echo "=== cache size:"; du -sh "$CACHE"
echo "=== DONE"
