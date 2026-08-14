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
# Adapted from Bruno's pack_deep_mi_a100.sh, with the differences that matter on
# Margaret. There is no `-C mps` here: the tau nodes carry turing/ampere GPUs
# with no MPS constraint exposed, so co-tenancy is plain CUDA time-slicing, and
# the nodes have 2-4 GPUs and 32-48 cores rather than 8 and 64.
#
# WHAT PACKING DOES AND DOES NOT BUY. Profiling (job 463259) put the GPU at
# 61.6 % mean utilisation with only 8.5 % of samples below 20 %. The device is
# NOT idling between batches, so ten tenants do not run ten times faster and it
# would be wrong to promise that. What packing removes is the queue: 240
# cross_session cells at 16-wide array parallelism means 240 queue entries for
# work that totals 6.5 GPU-hours, and sacct records 170,986 job-hours between
# submission and start across the campaign. One allocation, held, with cells fed
# to it is the fix -- co-tenancy is what makes one allocation enough.
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
#   GRID         TSV of `eval<TAB>dataset<TAB>model<TAB>seed`, one per line.
#   G            GPUs to round-robin over    (default: from SLURM, else 1)
#   K            co-tenants per GPU          (default 10, see the bound below)
#   CACHE        MOABB epoch cache           (default /scratch/amounir/moabb_cache)
#   RESULTS_DIR  where the CSVs land         (default benchmarks/results)
#   SUFFIX       MOABB result suffix stem    (default xsess)
#   MAX_SWEEPS   give up after this many passes (default 3, see the loop)
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
export MPLCONFIGDIR=/scratch/amounir/.mplconfig

# THIS LINE IS WHAT MAKES PACKING POSSIBLE AT ALL. Measured on a tau node
# (profile_memory.sbatch, job 463259), same cell three ways:
#
#   allocator default         peak allocated 1045 MiB   peak reserved 10370 MiB
#   expandable_segments       peak allocated  716 MiB   peak reserved   908 MiB
#
# The model needs under a gigabyte either way; what differs is what the process
# holds. PyTorch's default caching allocator keeps every block it has ever taken,
# so one cell squats 10.4 GB of an 11.2 GB card and exactly one process fits --
# which is why one job per GPU was the only arrangement that could work. With
# expandable segments the same cell holds 908 MiB and ten fit, at no cost in
# runtime (105.6 s vs 106.9 s, inside the noise).
#
# This is also why the reference script's K=10 does not transpose by itself: on
# an 80 GB A100, 10.4 GB x 10 still fits and the default allocator never bites.
# On an 11 GB RTX 2080 Ti it decides everything.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A declared ceiling per tenant on top of that. Nothing here should approach it;
# it exists so that one cell with an unexpected batch (the growth statistics on
# schirrmeister2017 once needed 43 GB) fails as its own OOM instead of starving
# the nine processes sharing its card.
#
# 0.20, not the 0.12 this started at. 0.12 was calibrated on the profiled cell
# (grow_shallow / bnci2014_001, 908 MiB reserved) and it killed 15 cells of the
# first packed campaign: 0.12 x 10.58 GiB = 1.27 GiB, while eegnex on zhou2016
# and lee2019 asks for ~1.6 GiB. The logs are unambiguous -- "1.27 GiB allowed"
# with "7.59 GiB is free" on the card. A ceiling set below what the work needs
# does not protect the co-tenants, it just fails the cell for no reason; the
# point of the cap is to catch a runaway, not to ration the normal case.
export EEGROW_CUDA_FRACTION="${EEGROW_CUDA_FRACTION:-0.20}"

GRID="${GRID:?set GRID to a TSV of eval<TAB>dataset<TAB>model<TAB>seed}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
SUFFIX="${SUFFIX:-xsess}"
# SLURM_GPUS_ON_NODE is what the allocation actually granted; asking for 4 and
# getting 2 is normal on a mixed partition, and hard-coding G would then leave
# every fourth process pinned to a device that does not exist.
G="${G:-${SLURM_GPUS_ON_NODE:-1}}"
# K=10: 11264 MiB per card, 908 MiB reserved per tenant with expandable segments,
# 15 % kept back for the CUDA context and fragmentation -> 10 fit. Note this is
# a memory bound, not a throughput promise: the same profiling run measured the
# GPU at 61.6 % utilisation with only 8.5 % of samples below 20 %, so the device
# is not idling and ten tenants will not run ten times faster. The gain packing
# buys is the queue, not the silicon.
#
# AND K IS A PROPERTY OF THE MODEL, NOT OF THE CARD. Measured on one card with
# no cap (job 463377, cross_session/zhou2016, peak reserved):
#
#   grow_shallow    726 MiB   ->  ~10 tenants fit
#   grow_eegnex    6410 MiB   ->  exactly 1 fits (6410 x 2 > 10834 available)
#
# 6084 of those 6410 MiB are genuinely allocated, so this is not allocator slack:
# gromo's growth statistics build a tensor product over the activations of the
# layer being grown, and EEGNeX's conv2 is far wider than ShallowNet's conv_spat.
# Same phenomenon as the 43 GB once seen on schirrmeister2017. A grid mixing the
# two needs either a per-model K -- which this runner does not have, K is global
# -- or one pass per memory class:
#   GRID=grid_eegnex.tsv K=1 EEGROW_CUDA_FRACTION=0.75 bash pack_run.sh
#
# AND K IS NOT ALWAYS A GPU BOUND EITHER. On lee2019_mi the binding constraint is host
# RAM: CrossSessionEvaluation holds the whole dataset (54 subjects x 62 channels,
# 4.4 GB of cached arrays) in the process, and G*K=30 tenants against --mem=120G
# is arithmetically impossible. The first packed campaign lost 30 lee2019 cells
# to the cgroup OOM killer that way -- killed mid-load, no traceback, GPU idle.
# Run a RAM-heavy dataset as its own pass with K=2 rather than raising --mem:
#   GRID=grid_lee.tsv K=2 bash pack_run.sh
K="${K:-10}"
NPROC=$((G * K))

# Writing somewhere other than benchmarks/results makes a campaign a clean
# re-run rather than a patch: the skip-if-done check below would otherwise treat
# every cell of the previous grid as already satisfied.
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmarks/results}"

CLAIMS="${CLAIMS:-/scratch/amounir/eegrow_claims}"
LOGS="$ROOT/benchmarks/slurm/logs/pack"
mkdir -p "$CLAIMS" "$LOGS" "$CACHE" "$RESULTS_DIR"

echo "PACK node=$(hostname) G=$G K=$K nproc=$NPROC grid=$GRID cells=$(wc -l < "$GRID")"
echo "PACK results_dir=$RESULTS_DIR cache=$CACHE start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

mapfile -t ROWS < "$GRID"
p=0
start=$(date +%s)

# Reap the claims of processes that died without releasing them.
#
# The release below (`[ -s "$OUT" ] || rmdir "$CLAIM"`) is the last statement of
# the subshell, so it runs for any failure the python process reports -- and for
# none of the failures that kill it outright. SIGKILL cannot be trapped, and the
# cgroup OOM killer uses SIGKILL. The first packed campaign lost 32 cells exactly
# there: dead, no CSV, claim still standing, therefore skipped by every
# subsequent sweep. The re-sweep is the retry mechanism, and a claim that outlives
# its process silently disables it.
#
# So each claim records the pid and host that took it, and a claim is stale when
# its owner is gone and no CSV exists. Only this host's claims are reaped: pids
# are node-local, and a live process on another node must not have its work
# stolen. A claim with no owner file predates this scheme and is left alone.
reap_stale() {
  local c owner pid host reaped=0
  for c in "$CLAIMS"/*; do
    [ -d "$c" ] || continue
    owner="$c/owner"
    [ -f "$owner" ] || continue
    IFS=' ' read -r host pid < "$owner"
    [ "$host" = "$(hostname)" ] || continue
    kill -0 "$pid" 2>/dev/null && continue
    # The cell key is the directory name: eval__dataset__model__sSEED.
    local key ev ds m sd
    key=$(basename "$c")
    ev=${key%%__*}; sd=${key##*__s}
    m=${key%__s*}; m=${m##*__}
    ds=${key#*__}; ds=${ds%%__*}
    [ -s "$RESULTS_DIR/$ev/$ds/${m}__seed${sd}.csv" ] && continue
    rm -f "$owner"; rmdir "$c" 2>/dev/null && reaped=$((reaped + 1))
  done
  [ "$reaped" -gt 0 ] && echo "PACK reaped $reaped stale claim(s)"
  return 0
}

# Sweep the whole grid, claiming what is free, until a full pass claims nothing.
# Re-sweeping (rather than partitioning the grid up front) is what makes several
# allocations cooperate: a node that finishes early picks up what a slower one
# has not claimed, and a cell whose process died is retried on the next pass
# because the claim is released when no CSV was produced.
#
# MAX_SWEEPS bounds it. A cell that fails deterministically -- a CUDA OOM from a
# ceiling set too low, a dataset that does not have the sessions the protocol
# needs -- releases its claim on every pass and is re-claimed on the next, so
# `claimed` never reaches zero and the allocation burns its walltime retrying
# work that cannot succeed. That is what the first packed campaign did with 15
# cells. Retrying is for transient failures; three passes is generous for that,
# and past it the honest outcome is to stop and report what is missing.
MAX_SWEEPS="${MAX_SWEEPS:-3}"
sweep=0
while true; do
  sweep=$((sweep + 1))
  if [ "$sweep" -gt "$MAX_SWEEPS" ]; then
    echo "PACK giving up after $MAX_SWEEPS sweeps; cells still missing:"
    while IFS=$'\t' read -r EV DS M SEED; do
      [ -s "$RESULTS_DIR/${EV}/${DS}/${M}__seed${SEED}.csv" ] \
        || echo "  MISSING $EV $DS $M seed$SEED"
    done < "$GRID"
    break
  fi
  # At the top of every sweep, not once at startup: a cell killed during sweep N
  # has to become claimable again for sweep N+1, which is the whole point.
  reap_stale
  claimed=0
  for line in "${ROWS[@]}"; do
    [ -z "$line" ] && continue
    EV=$(echo "$line" | cut -f1)
    DS=$(echo "$line" | cut -f2)
    M=$(echo "$line"  | cut -f3)
    SEED=$(echo "$line" | cut -f4)

    OUT="$RESULTS_DIR/${EV}/${DS}/${M}__seed${SEED}.csv"
    [ -s "$OUT" ] && continue

    CLAIM="$CLAIMS/${EV}__${DS}__${M}__s${SEED}"
    # mkdir is the atomic primitive here: it succeeds for exactly one caller and
    # fails for every other, across processes and across nodes on a shared FS.
    # A lock file with test-then-create would race.
    mkdir "$CLAIM" 2>/dev/null || continue
    claimed=$((claimed + 1))

    gpu=$((p % G))
    (
      # Stamp the owner before doing anything, so that a process killed at any
      # later point is recognisable as dead rather than as in-flight. $BASHPID is
      # this subshell's pid; $$ would be the parent's and would look alive
      # forever.
      echo "$(hostname) $BASHPID" > "$CLAIM/owner"
      # The suffix must be unique per cell. MOABB keys its HDF5 store by
      # (hdf5_path, suffix), and hdf5_path is per (eval, dataset) -- so two
      # co-tenant processes on the same dataset sharing a suffix would open the
      # same file for writing. Sequential jobs got away with it; packing does
      # not. The production fix250 scripts already carried a per-model suffix
      # for exactly this reason.
      CUDA_VISIBLE_DEVICES=$gpu timeout 7200 python run_moabb_hydra.py \
        eval="$EV" dataset="$DS" model="$M" seed="$SEED" \
        cache.enabled=true cache.path="$CACHE" \
        results_dir="$RESULTS_DIR" \
        overwrite=true suffix="${SUFFIX}_${M}_s${SEED}" \
        > "$LOGS/${EV}__${DS}__${M}__s${SEED}.log" 2>&1
      # Release the claim if nothing was written, so the next sweep retries this
      # cell instead of recording it as done. The owner file has to go first --
      # rmdir refuses a non-empty directory, and the claim must be *gone*, not
      # merely emptied, for the atomic mkdir to hand it to the next taker.
      if [ -s "$OUT" ]; then
        :
      else
        rm -f "$CLAIM/owner"; rmdir "$CLAIM" 2>/dev/null
      fi
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
