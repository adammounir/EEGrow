#!/bin/bash
# Verify a smoke run, then launch the campaign -- or refuse and say why.
#
# WHY THIS EXISTS
# ---------------
# The decision "the smoke is green, submit the campaign" is a fixed checklist, and a
# checklist that only runs when a human (or an assistant session) happens to be awake
# is a checklist that delays the campaign by however long nobody is looking. Running it
# from a SLURM dependency job makes it fire the moment the last smoke job leaves the
# queue, regardless of who is watching.
#
# It is a GATE, not a launcher: the default outcome is refusal. Every check has to pass
# explicitly, and a check that cannot be evaluated (sacct unreachable, no logs) counts
# as a failure, not as an absence of failure. The campaign costs ~760 GPU-hours; the
# asymmetry between "launched a day late" and "launched on a broken smoke" is enormous.
#
# WHAT IS *NOT* A CRITERION
# -------------------------
# Cells with no CSV. The smoke deliberately sets CELL_TIMEOUT=7200 and grow_every=1, so
# its long cells (cho2017, lee2019_mi, physionetmi, schirrmeister2017 in cross_subject)
# are SIGTERMed mid-training by design -- 45 of 148 on the 2026-08-16 run, on six
# passes whose Elapsed was exactly 02:00:1x. Requiring an empty `cells still missing`
# would therefore refuse a healthy smoke forever. Production is unaffected: its
# CELL_TIMEOUT is 172800 s against a 72 h wall. What matters is that no cell died of
# something the campaign would also hit -- OOM, a cache fault, a traceback -- and those
# are checked.
#
# USAGE
#   gate_launch.sh --jobs <id,id,...> --submit <script> [--dry-run]
#
# --dry-run evaluates every check and prints the verdict without submitting, so the
# gate can be exercised against a live smoke before being trusted with the real one.

set -uo pipefail

JOBS=""
SUBMIT=""
DRY=0
SMOKE_LOGS="/scratch/amounir/logs"
CELL_LOGS="/scratch/amounir/logs/smoke"
RESULTS="/scratch/amounir/results_smoke"
CACHE="/scratch/amounir/moabb_cache"
ROOT="/scratch/amounir/eegrow"

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs)    JOBS="$2"; shift 2 ;;
    --submit)  SUBMIT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "GATE unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$JOBS" ] && [ -n "$SUBMIT" ] || { echo "GATE usage: --jobs a,b,c --submit script" >&2; exit 2; }

echo "GATE start=$(date -Is) host=$(hostname) dry_run=$DRY"
echo "GATE jobs=$JOBS submit=$SUBMIT"

# sacct can lag behind the last job's termination by a few seconds. Cheap insurance
# against reading a job as RUNNING when it has in fact just finished.
[ "$DRY" -eq 1 ] || sleep 90

fail=0
note() { echo "GATE  $1"; }
bad()  { echo "GATE  FAIL: $1"; fail=$((fail + 1)); }

# --- 1. every smoke job terminated cleanly -----------------------------------
# -X: one row per job, not per step, so a job's own state is not confused with its
# batch step's. An empty sacct output is a failure: it means we cannot tell.
SACCT=$(sacct -j "$JOBS" --format=JobID%14,State%14,Elapsed,ExitCode -X -n -P 2>/dev/null)
if [ -z "$SACCT" ]; then
  bad "sacct returned nothing for $JOBS -- cannot establish the smoke's outcome"
else
  n_job=$(wc -l <<<"$SACCT")
  n_ok=$(awk -F'|' '$2 ~ /COMPLETED/ && $4 == "0:0"' <<<"$SACCT" | wc -l)
  note "jobs: $n_ok/$n_job COMPLETED with exit 0:0"
  awk -F'|' '!($2 ~ /COMPLETED/ && $4 == "0:0") {print "GATE    not clean: "$1" "$2" "$3" "$4}' <<<"$SACCT"
  [ "$n_ok" = "$n_job" ] || bad "$((n_job - n_ok)) smoke job(s) did not end COMPLETED 0:0"
fi

# --- 2. no failure of a kind the campaign would also hit ---------------------
# Counted over the per-cell logs *and* the pass logs: an OOM raised inside a cell lands
# in the cell log, one raised by the runner itself lands in the pass log.
#
# THE LOG SET IS DERIVED FROM --jobs, NEVER FROM A GLOB. `smoke_*.log` matched 25 files
# on 2026-08-16 for a 12-job smoke: 13 were left by earlier attempts, two of which
# (464831, 464832) had legitimately aborted on `PACK FATAL: refusing to run G*K tenants
# against an incomplete cache`. A glob therefore made the gate refuse a healthy run
# because of a run it was not asked about. A gate that can read evidence from outside
# the run it is judging is not a gate.
shopt -s nullglob
PASS_LOGS=()
for j in ${JOBS//,/ }; do
  [ -f "$SMOKE_LOGS/smoke_${j}.log" ] && PASS_LOGS+=("$SMOKE_LOGS/smoke_${j}.log")
done
# Cell logs are named after the cell, not the job, so pack_run.sh overwrites them across
# runs by design -- there is no job id to filter on. Bound them by time instead: only
# logs written after the earliest of these jobs started can belong to this run.
SINCE=$(sacct -j "$JOBS" --format=Start -X -n -P 2>/dev/null | grep -v '^Unknown$' | sort | head -1)
if [ -n "$SINCE" ]; then
  mapfile -t CELL < <(find "$CELL_LOGS" -name '*.log' -newermt "$SINCE" 2>/dev/null)
  note "cell logs bounded to those written since $SINCE"
else
  bad "cannot determine the smoke's start time -- refusing to date its logs by guess"
  CELL=()
fi
shopt -u nullglob
if [ "${#CELL[@]}" -eq 0 ]; then
  bad "no per-cell log in $CELL_LOGS -- the smoke produced nothing to check"
else
  note "logs: ${#PASS_LOGS[@]} pass, ${#CELL[@]} cell"
  n_oom=$(grep -lE 'CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED' "${CELL[@]}" "${PASS_LOGS[@]}" 2>/dev/null | wc -l)
  n_cache=$(grep -lE 'Directory not empty|FileExistsError|No data left in file|Did not find any' "${CELL[@]}" "${PASS_LOGS[@]}" 2>/dev/null | wc -l)
  n_trace=$(grep -lE '^Traceback|job FAILED' "${CELL[@]}" "${PASS_LOGS[@]}" 2>/dev/null | wc -l)
  n_fatal=$(grep -hc 'PACK FATAL' "${PASS_LOGS[@]}" 2>/dev/null | paste -sd+ - | bc)
  note "OOM=$n_oom cache_faults=$n_cache tracebacks=$n_trace pack_fatal=${n_fatal:-0}"
  [ "$n_oom" -eq 0 ]   || bad "$n_oom log(s) with a CUDA OOM"
  [ "$n_cache" -eq 0 ] || bad "$n_cache log(s) with a cache fault"
  [ "$n_trace" -eq 0 ] || bad "$n_trace log(s) with a traceback"
  [ "${n_fatal:-0}" -eq 0 ] || bad "${n_fatal} PACK FATAL"

  # Cells that produced a CSV vs cells that were SIGTERMed by the smoke's own 2 h
  # ceiling. Reported, never a criterion -- see the header.
  csv=$(find "$RESULTS" -name '*.csv' -newermt "$SINCE" 2>/dev/null | wc -l)
  note "informational: $csv CSV for ${#CELL[@]} cells started; the shortfall is the smoke's CELL_TIMEOUT=7200 by design"
fi

# --- 3. the cache survived the co-tenancy ------------------------------------
# The whole point of the smoke is 30 tenants hammering one BIDS cache that is not safe
# for concurrent writers. A cache that degraded under 30 must not be handed to 1280.
if [ -x /scratch/amounir/miniforge3/bin/conda ] || [ -f /scratch/amounir/miniforge3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
  conda activate bench 2>/dev/null
fi
CHK=$(cd "$ROOT" && python benchmarks/check_cache.py --cache "$CACHE" 2>&1)
if grep -q 'cache prêt pour la clé de la campagne' <<<"$CHK"; then
  # Sum the MANQUE and MENSONGE columns of the per-dataset table rather than trusting
  # the summary line alone.
  miss=$(awk 'NF==6 && $2 ~ /^[0-9]+$/ {m+=$3; l+=$4} END {print m"/"l}' <<<"$CHK")
  note "cache: verdict OK, manque/mensonge = ${miss:-?}"
  [ "$miss" = "0/0" ] || bad "cache reports $miss missing/lying subjects"
else
  bad "check_cache.py did not certify the cache"
  tail -5 <<<"$CHK" | sed 's/^/GATE    /'
fi

# --- verdict -----------------------------------------------------------------
if [ "$fail" -ne 0 ]; then
  echo "GATE VERDICT=RED ($fail failed check(s)) -- CAMPAIGN NOT SUBMITTED"
  echo "GATE end=$(date -Is)"
  exit 1
fi
echo "GATE VERDICT=GREEN"

if [ "$DRY" -eq 1 ]; then
  echo "GATE dry run: would have run $SUBMIT"
  echo "GATE end=$(date -Is)"
  exit 0
fi

# submit.sh uses --export=ALL, which would otherwise propagate THIS job's SLURM_* and
# CUDA_VISIBLE_DEVICES into all 15 pack jobs. The controller resets the identity
# variables, but the resource ones (SLURM_CPUS_PER_TASK, SLURM_MEM_PER_NODE,
# SLURM_GPUS) are inherited and silently override what submit.sh asks for. Strip them.
for v in $(compgen -v SLURM_) $(compgen -v SBATCH_); do unset "$v"; done
unset CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

echo "GATE submitting $SUBMIT"
bash "$SUBMIT"
rc=$?
echo "GATE submit exit=$rc"
echo "GATE queue after submit:"
squeue -u "$USER" -o '%.10i %.14j %.10T %.6D %R' | sed 's/^/GATE    /'
echo "GATE end=$(date -Is)"
exit $rc
