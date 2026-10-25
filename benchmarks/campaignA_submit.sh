#!/bin/bash
# Submit campaign-A arrays (campaignA_make_cells.py output), one sbatch per array file,
# with the --mem / --time its name carries (A<nn>_<wave>_mem<G>_t<h>.txt).
#
#     bash campaignA_submit.sh <ROOT> <CELLS> <ARRAY_DIR> <pattern> [extra sbatch args]
#
# e.g. the canary wave only:  bash campaignA_submit.sh /scratch/amounir/eegrow_a \
#          campaignA_cells.txt campaignA_arrays 'A*_canary_*'
# Paths are relative to ROOT/benchmarks. Prints one "submitted <jobid> <file>" per array
# and refuses (exit 1) before submitting anything if a file name does not parse or the
# checkout's sha stamp is missing.
set -euo pipefail
ROOT=$1; CELLS=$2; ADIR=$3; PAT=$4; shift 4
cd "$ROOT/benchmarks"
[ -s "$ROOT/.eegrow_sha" ] || { echo "REFUSED: no $ROOT/.eegrow_sha"; exit 1; }
[ -s "$CELLS" ] || { echo "REFUSED: no $CELLS"; exit 1; }
mkdir -p "$ROOT/logs/campA"
files=$(ls $ADIR/$PAT.txt)
for f in $files; do
  [[ $(basename "$f") =~ _mem([0-9]+)_t([0-9]+)\.txt$ ]] || { echo "REFUSED: name $f"; exit 1; }
done
for f in $files; do
  [[ $(basename "$f") =~ _mem([0-9]+)_t([0-9]+)\.txt$ ]]
  mem=${BASH_REMATCH[1]}; t=${BASH_REMATCH[2]}
  n=$(wc -l < "$f")
  jid=$(sbatch --parsable --array=0-$((n - 1)) --mem=${mem}G --time=${t}:00:00 \
        -o "$ROOT/logs/campA/%A_%a.log" \
        --export=ALL,ROOT=$ROOT,CELLS=$CELLS,ARRAY=$f "$@" campaignA.sbatch)
  echo "submitted $jid $f ($n tasks, ${mem}G, ${t}h)"
done
