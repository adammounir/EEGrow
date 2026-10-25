#!/bin/bash
# Campaign A, amendment 2: move the PENDING tasks of running arrays onto another sbatch
# (campaignA_b.sbatch: wider card gate), never a running one.
#
#     bash campaignA_migrate.sh <ROOT> <CELLS> <SBATCH> <jobid>=<array file> ...
#
# Per array job: hold it (nothing new starts), list its pending task ids, cancel exactly
# those, resubmit the same ids of the same array file with the same --mem / --time (the
# file name says them). A task id is a line of the array file, so the same id runs the same
# cells. Tasks that had started keep running under the old job id; a cell whose CSV exists
# is skipped by the sbatch, so no cell runs twice. 80 GB arrays: --cpus-per-task=9 and
# margpu032 excluded (CR_CORE: memory is not reserved, cores cap the tasks per node).
set -euo pipefail
ROOT=$1; CELLS=$2; SB=$3; shift 3
cd "$ROOT/benchmarks"
[ -s "$ROOT/.eegrow_sha" ] && [ -s "$CELLS" ] && [ -s "$SB" ] || { echo "REFUSED: missing input"; exit 1; }
for pair in "$@"; do
  f=${pair#*=}
  [[ $(basename "$f") =~ _mem([0-9]+)_t([0-9]+)\.txt$ ]] && [ -s "$f" ] || { echo "REFUSED: $f"; exit 1; }
done
for pair in "$@"; do
  scontrol hold "${pair%%=*}" 2>/dev/null || true   # running tasks are untouched by hold
done
for pair in "$@"; do
  j=${pair%%=*}; f=${pair#*=}
  [[ $(basename "$f") =~ _mem([0-9]+)_t([0-9]+)\.txt$ ]]
  mem=${BASH_REMATCH[1]}; t=${BASH_REMATCH[2]}
  # pending ids as ranges (a,b-c,...): 1000 single ids would overflow --array
  ids=$(squeue -h -r -j "$j" -t PENDING -o "%K" | sort -n | awk '
    NR == 1 { a = b = $1; next }
    $1 == b + 1 { b = $1; next }
    { out = out (out ? "," : "") (a == b ? a : a "-" b); a = b = $1 }
    END { if (NR) print out (out ? "," : "") (a == b ? a : a "-" b) }')
  if [ -z "$ids" ]; then echo "$j: nothing pending"; continue; fi
  n=$(squeue -h -r -j "$j" -t PENDING -o "%K" | wc -l)
  extra=()
  if [ "$mem" -ge 80 ]; then extra=(--cpus-per-task=9 --exclude=margpu004,margpu028,margpu032); fi
  scancel "${j}_[${ids}]"
  jid=$(sbatch --parsable --array="$ids" --mem=${mem}G --time=${t}:00:00 \
        -o "$ROOT/logs/campA/%A_%a.log" \
        --export=ALL,ROOT=$ROOT,CELLS=$CELLS,ARRAY=$f "${extra[@]}" "$SB")
  echo "migrated $j -> $jid $f ($n pending tasks, ${mem}G, ${t}h ${extra[*]})"
done
# A task requeued (preemption, card probe) after the listing above is pending under the
# old job and was not migrated: release, so it still runs, under the old gate.
for pair in "$@"; do scontrol release "${pair%%=*}" 2>/dev/null || true; done
