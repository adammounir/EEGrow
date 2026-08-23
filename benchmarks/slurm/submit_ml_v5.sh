#!/bin/bash
# Submit the four passes of the classical-baseline arm into the v5 results tree.
#
# Run once. Every task is idempotent (it exits 0 if its result CSV exists), so a second
# submission is harmless -- but it is also pointless, and four arrays are enough to
# track.
#
# THE SIZINGS, AND WHERE THEY COME FROM. Wall clock is the per-cell total measured on
# the 14-model campaign (sum of fold times, OMP_NUM_THREADS=1, so an upper bound here:
# this arm gets 8 BLAS threads and a fully warm epoch cache).
#
#   pass          cells  worst measured cell   partition     mem    wall   throttle
#   light           444   2.0 h (xsubj shin)   normal         64G     8 h   40
#   mid              18  27.0 h (xsubj cho)    normal-best   160G    3 d     6
#   schirr_within    30   5.7 h                normal         96G    12 h   10
#   schirr_xsubj      6  95.0 h                normal-best   250G     7 d    6
#
# Memory: the July run of this arm put cross_subject on 32 G and SLURM killed it
# (`srun: error: marg026: task 0: Killed`). Leave-one-subject-out holds every subject's
# epochs in RAM at once and MOABB copies them again for the split, so the bound is
# roughly three times the dataset: ~11 GB of epochs for lee2019_mi, ~13 GB for
# schirrmeister2017 at 128 channels. within_session touches one subject, hence 64 G.
#
# Wall clock on the two long passes is 2.7x and 1.8x the worst measured cell, because a
# SLURM timeout is not requeued -- only node failure and preemption are. Those two
# passes also run with overwrite=false so that a requeue resumes from MOABB's
# per-subject store instead of restarting a 95-hour cell; that is safe here only
# because results_v5 has never held an ml_* result, so there is nothing stale to read
# back. The short passes keep overwrite=true.
set -euo pipefail

DIR=/scratch/amounir/ml_v5
SB=/scratch/amounir/eegrow/benchmarks/slurm/ml_v5.sbatch
mkdir -p /scratch/amounir/logs/ml_v5

n() { echo $(( $(wc -l < "$1") - 1 )); }   # highest array index

sbatch --job-name=mlv5-light --partition=normal --mem=64G --time=8:00:00 \
  --array=0-$(n $DIR/ml_light.txt)%40 \
  --export=ALL,POINTS=$DIR/ml_light.txt,OVERWRITE=true "$SB"

sbatch --job-name=mlv5-mid --partition=normal-best --mem=160G --time=3-00:00:00 \
  --array=0-$(n $DIR/ml_mid.txt)%6 \
  --export=ALL,POINTS=$DIR/ml_mid.txt,OVERWRITE=false "$SB"

sbatch --job-name=mlv5-schirrW --partition=normal --mem=96G --time=12:00:00 \
  --array=0-$(n $DIR/ml_schirr_within.txt)%10 \
  --export=ALL,POINTS=$DIR/ml_schirr_within.txt,OVERWRITE=true "$SB"

sbatch --job-name=mlv5-schirrX --partition=normal-best --mem=250G --time=7-00:00:00 \
  --array=0-$(n $DIR/ml_schirr_xsubj.txt)%6 \
  --export=ALL,POINTS=$DIR/ml_schirr_xsubj.txt,OVERWRITE=false "$SB"
