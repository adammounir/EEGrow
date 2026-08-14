#!/bin/bash
#SBATCH --job-name=eegrow_pack
#SBATCH --partition=tau
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --exclude=margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/eegrow/benchmarks/slurm/logs/pack_%j.log
# Batch wrapper around pack_run.sh: one allocation chewing a whole grid.
#
# The excluded nodes are the pascal ones (margpu024-027,029) and the drained
# margpu023, same list as the Hydra launcher. Pascal has no usable tensor cores
# and is slower than the turing/ampere nodes for these networks.
#
# 3 GPUs and 24 cores, not 4 and 32, because that is what a tau node actually
# has free: the turing nodes carry 3-4 GPUs against 32 cores and are rarely
# empty, so a 4-GPU + 32-core request sits in PENDING (Resources) behind a node
# that will never have both at once. Asking for what exists is the difference
# between running now and queueing -- which is the very cost this script exists
# to remove.
#
# Several of these can be submitted at once: they cooperate through the atomic
# claim directory in pack_run.sh rather than each taking a slice of the grid, so
# adding a second allocation adds throughput without any partitioning to get
# wrong.
#
#   python benchmarks/slurm/make_grid.py --eval cross_session \
#       --out /scratch/amounir/grid_xsess.tsv
#   sbatch --export=ALL,GRID=/scratch/amounir/grid_xsess.tsv,K=4 \
#       benchmarks/slurm/pack_run.sbatch
set -uo pipefail
bash /scratch/amounir/eegrow/benchmarks/slurm/pack_run.sh
