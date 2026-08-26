#!/bin/bash
# Runs gate_launch.sh once the smoke jobs have left the queue, and submits the campaign
# if the gate is green. Submitted with:
#
#   sbatch --dependency=afterany:<a:b:c> \
#          --export=ALL,GATE_JOBS=<a:b:c>,GATE_SUBMIT=<submit.sh> \
#          benchmarks/slurm/gate_launch.sbatch
#
# GATE_JOBS is COLON-separated, and that is not a style choice. --export takes a
# comma-separated list of NAME=VALUE, so a comma-separated job list is split by sbatch
# into GATE_JOBS=<first id> plus a series of exports of variables named after the
# remaining ids -- the gate would then judge one job out of twelve and pass. Colons are
# what Slurm's own dependency lists use; translated to commas for sacct below.
#
# `normal`, not `tau`: this job reads logs and runs sbatch. Putting it on a GPU
# partition would make the campaign's launch queue behind the GPUs it is trying to
# allocate.
#
# Note what a compute node can and cannot do here. `sbatch` works from `normal`
# (verified on marg003). `sacct` does NOT: slurmdbd answers only on the login node,
# so the gate derives the terminal state and the start time from pack_run.sh's own
# `PACK_DONE` and `start=` markers instead. See section 0 of gate_launch.sh -- the
# first version of this wrapper returned RED on a clean smoke for exactly that
# reason, and left the campaign unlaunched overnight.
#
# afterany, not afterok: a smoke job that ends CANCELLED or FAILED must still wake the
# gate, which then refuses out loud. afterok would leave the dependency unsatisfiable
# and the gate would sit in PENDING forever with nobody told.
#SBATCH --job-name=eegrow_gate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/amounir/logs/gate_%j.log

set -uo pipefail
cd /scratch/amounir/eegrow
JOBS="${GATE_JOBS:?GATE_JOBS not set}"
# Refuse a list that arrived truncated rather than judge a fraction of the smoke.
n=$(tr ':' '\n' <<<"${JOBS//,/:}" | grep -c '^[0-9][0-9]*$')
echo "GATE wrapper: ${n} job id(s) received"
[ "$n" -ge 2 ] || { echo "GATE wrapper FATAL: only $n job id parsed from '$JOBS'"; exit 2; }
exec bash benchmarks/slurm/gate_launch.sh \
  --jobs "${JOBS//:/,}" \
  --submit "${GATE_SUBMIT:?GATE_SUBMIT not set}"
