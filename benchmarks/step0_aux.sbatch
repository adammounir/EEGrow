#!/bin/bash
# Step-0 capacity (PREREG § 5.1) and published-reference (§ 6) runs, one task per line
# of step0_aux_tasks.txt:
#   capacity  <arm> <candidate> <lr> <wd> <seed>
#   reference <shallow|deep> - - - <seed> <subject>
# Same card, checkout and import guard as step0_battery.sbatch.
#SBATCH --job-name=s0-aux
#SBATCH --partition=tau,gpu-best
#SBATCH --gres=gpu:turing:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-65
#SBATCH --output=/scratch/amounir/eegrow_step0/logs/aux_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail

ROOT=/scratch/amounir/eegrow_step0
CARD="NVIDIA GeForce RTX 2080 Ti"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/benchmarks/step0_aux_tasks.txt")
read -r KIND A B C D SEED SUBJ <<< "$LINE"
echo "[$(date)] task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) :: $LINE"
GOT="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)"
if [ "$GOT" != "$CARD" ]; then echo "REFUSED: card '$GOT' != '$CARD'"; exit 1; fi
python -c "import eegrow; assert eegrow.__file__.startswith('$ROOT/'), eegrow.__file__" || exit 1

OUTD=$ROOT/step0_results/aux
mkdir -p $OUTD/done
TAG="${KIND}_${A}_${B}_s${SEED}_${SUBJ:-x}"
if [ -f "$OUTD/done/$TAG" ]; then echo "already done: $TAG"; exit 0; fi

cd $ROOT/benchmarks
t0=$SECONDS
if [ "$KIND" = capacity ]; then
    python step0_capacity.py --arm "$A" --candidate "$B" --lr "$C" --weight-decay "$D" \
        --seed "$SEED" --device cuda --out "$OUTD/capacity.jsonl"
    rc=$?
    # capacity exits 1 on a REFUSED net: that is a result, not a crash.
    [ $rc -le 1 ] && touch "$OUTD/done/$TAG" && rc=0
else
    python step0_reference.py --model "$A" --subject "$SUBJ" --seed "$SEED" \
        --device cuda --out "$OUTD/reference_${A}_S${SUBJ}_seed${SEED}.jsonl"
    rc=$?
    [ $rc -eq 0 ] && touch "$OUTD/done/$TAG"
fi
echo "[$(date)] $TAG rc=$rc WALL $((SECONDS - t0)) s"
exit $rc
