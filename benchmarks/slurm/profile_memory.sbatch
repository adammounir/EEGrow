#!/bin/bash
#SBATCH --job-name=eegrow_mem
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/eegrow/benchmarks/slurm/logs/mem_%j.log
# How many co-tenant processes fit on one 11 GB card?
#
# The first profiling run answered the wrong half of that question. A cell's
# model needs 778-942 MiB *allocated* -- Bruno's "less than a gigabyte" is
# correct -- but the process holds 3876-6620 MiB *reserved*, because PyTorch's
# caching allocator keeps every block it has ever taken. The device only sees
# the reserved figure, so at 6.5 GB exactly one process fits on an RTX 2080 Ti
# and the packing plan collapses to K=1. On the 80 GB A100 the reference script
# targets, 6.5 GB x 10 still fits, which is why K=10 worked there and cannot be
# copied here.
#
# So the question is whether the reserved figure can be brought back down to the
# allocated one. This runs the same cell under three allocator settings and
# reports what each holds.
#
#   sbatch benchmarks/slurm/profile_memory.sbatch
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
cd /scratch/amounir/eegrow/benchmarks
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export EEGROW_BENCH_ROOT=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
module load cuda/13.1 2>/dev/null || true

CACHE=/scratch/amounir/moabb_cache
DS=${DS:-bnci2014_001}
EV=${EV:-cross_session}
MODEL=${MODEL:-grow_shallow}
OUT=/scratch/amounir/eegrow/benchmarks/profile_mem
mkdir -p "$OUT"

echo "=== node=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

run_one () {
  local tag="$1"; shift
  echo "=== $tag"
  env "$@" python profile_cell.py \
      model=$MODEL dataset=$DS eval=$EV seed=7 \
      cache.enabled=true cache.path=$CACHE \
      profile.out_dir="$OUT/$tag" 2>&1 | grep -E "torch_peak|util_|evaluate_s|samples" || true
}

# Baseline: the allocator's default behaviour, i.e. what the campaign ran.
run_one default PYTORCH_CUDA_ALLOC_CONF=

# expandable_segments lets the allocator grow and release one virtual segment
# instead of hoarding fixed blocks, which is the documented fix for exactly this
# reserved-vs-allocated gap.
run_one expandable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A hard ceiling on the fraction of the card one process may reserve. If the
# allocator simply never had a reason to give memory back, capping it should
# cost nothing; if the cell genuinely needs the headroom, this run OOMs, and
# that is the answer too.
run_one capped_15pct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        EEGROW_CUDA_FRACTION=0.15

echo "=== DONE"
