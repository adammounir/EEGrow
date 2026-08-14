#!/bin/bash
# Packed work-queue runner: many cells per GPU, one allocation, no requeue.
#
# WHY THIS EXISTS
# ---------------
# The Hydra launcher (config/hydra/launcher/tau.yaml) asks SLURM for one whole
# GPU per (eval x dataset x model x seed) point, with array_parallelism=16. The
# recorded fit time of a cross_session cell is 1-4 minutes and these networks
# hold well under a gigabyte. So each cell spent minutes computing and hours
# queueing, on a device it used a few percent of. The whole cross_session grid --
# 6 datasets x 8 deep arms x 5 seeds -- is 6.5 GPU-hours of actual training.
#
# This script inverts that. It takes ONE allocation of G GPUs, keeps it, and
# feeds it from a work queue: K co-tenant processes per GPU, cells claimed
# atomically so several allocations can chew the same grid without duplicating
# work, and no process ever goes back to the queue between cells.
#
# Adapted from Bruno's pack_deep_mi_a100.sh, with two differences that matter on
# Margaret. There is no `-C mps` here: the tau nodes carry turing/ampere GPUs
# with no MPS constraint exposed, so co-tenancy is plain CUDA time-slicing. That
# is adequate precisely because the bottleneck is not compute -- a device idle
# between two small batches yields to another tenant on its own. And the nodes
# have 2-4 GPUs and 32-48 cores, not 8 and 64, so G*K is bounded by cores as
# much as by GPU memory (see K below).
#
# USAGE
#   Inside an interactive allocation (the mode Bruno recommends -- hold the
#   nodes for the day and iterate without re-entering the queue):
#       salloc -p tau --gres=gpu:4 -c 32 --mem=180G -t 12:00:00
#       srun --pty bash
#       GRID=... bash benchmarks/slurm/pack_run.sh
#
#   Or as a batch job: see pack_run.sbatch.
#
# ENVIRONMENT
#   GRID    TSV of `eval<TAB>dataset<TAB>model<TAB>seed`, one cell per line.
#   G       GPUs to round-robin over        (default: from SLURM, else 1)
#   K       co-tenant processes per GPU     (default 4)
#   CACHE   MOABB epoch cache directory     (default /scratch/amounir/moabb_cache)
#   SUFFIX  MOABB result suffix             (default xsess)
set -uo pipefail

ROOT=/scratch/amounir/eegrow
cd "$ROOT/benchmarks"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH="$ROOT/benchmarks"
export EEGROW_BENCH_ROOT="$ROOT/benchmarks"
export MNE_DATA=/scratch/amounir/mne_data
module load cuda/13.1 2>/dev/null || true

# One BLAS thread per process. With K co-tenants per GPU the node is already
# oversubscribed at the process level; letting each of them open 8 OpenMP
# threads turns a CPU-bound preprocessing phase into a thrash.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/scratch/amounir/.mplconfig

GRID="${GRID:?set GRID to a TSV of eval<TAB>dataset<TAB>model<TAB>seed}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
SUFFIX="${SUFFIX:-xsess}"
# SLURM_GPUS_ON_NODE is what the allocation actually granted; asking for 4 and
# getting 2 is normal on a mixed partition, and hard-coding G would then leave
# every fourth process pinned to a device that does not exist.
G="${G:-${SLURM_GPUS_ON_NODE:-1}}"
K="${K:-4}"
NPROC=$((G * K))

CLAIMS="${CLAIMS:-/scratch/amounir/eegrow_claims}"
LOGS="$ROOT/benchmarks/slurm/logs/pack"
mkdir -p "$CLAIMS" "$LOGS" "$CACHE"

echo "PACK node=$(hostname) G=$G K=$K nproc=$NPROC grid=$GRID cells=$(wc -l < "$GRID")"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

mapfile -t ROWS < "$GRID"
p=0
start=$(date +%s)

# Sweep the whole grid, claiming what is free, until a full pass claims nothing.
# Re-sweeping (rather than partitioning the grid up front) is what makes several
# allocations cooperate: a node that finishes early picks up what a slower one
# has not claimed, and a cell whose process died is retried on the next pass
# because the claim is released when no CSV was produced.
while true; do
  claimed=0
  for line in "${ROWS[@]}"; do
    [ -z "$line" ] && continue
    EV=$(echo "$line" | cut -f1)
    DS=$(echo "$line" | cut -f2)
    M=$(echo "$line"  | cut -f3)
    SEED=$(echo "$line" | cut -f4)

    OUT="$ROOT/benchmarks/results/${EV}/${DS}/${M}__seed${SEED}.csv"
    [ -s "$OUT" ] && continue

    CLAIM="$CLAIMS/${EV}__${DS}__${M}__s${SEED}"
    # mkdir is the atomic primitive here: it succeeds for exactly one caller and
    # fails for every other, across processes and across nodes on a shared FS.
    # A lock file with test-then-create would race.
    mkdir "$CLAIM" 2>/dev/null || continue
    claimed=$((claimed + 1))

    gpu=$((p % G))
    (
      # The suffix must be unique per cell. MOABB keys its HDF5 store by
      # (hdf5_path, suffix), and hdf5_path is per (eval, dataset) -- so two
      # co-tenant processes on the same dataset sharing a suffix would open the
      # same file for writing. Sequential jobs got away with it; packing does
      # not. The production fix250 scripts already carried a per-model suffix
      # for exactly this reason.
      CUDA_VISIBLE_DEVICES=$gpu timeout 7200 python run_moabb_hydra.py \
        eval="$EV" dataset="$DS" model="$M" seed="$SEED" \
        cache.enabled=true cache.path="$CACHE" \
        overwrite=true suffix="${SUFFIX}_${M}_s${SEED}" \
        > "$LOGS/${EV}__${DS}__${M}__s${SEED}.log" 2>&1
      # Release the claim if nothing was written, so the next sweep retries this
      # cell instead of recording it as done.
      [ -s "$OUT" ] || rmdir "$CLAIM" 2>/dev/null
    ) &
    p=$((p + 1))

    # Throttle: block until a slot frees. `wait -n` returns on the first child
    # to exit, which keeps every slot busy instead of draining in waves.
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n; done
  done
  wait
  [ "$claimed" -eq 0 ] && break
  echo "PACK sweep done, claimed=$claimed, elapsed=$(( $(date +%s) - start ))s"
done

echo "PACK_DONE launched=$p elapsed=$(( $(date +%s) - start ))s"
