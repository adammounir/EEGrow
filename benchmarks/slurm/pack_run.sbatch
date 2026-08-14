#!/bin/bash
#SBATCH --job-name=eegrow_pack
#SBATCH --partition=tau
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --exclude=margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/eegrow/benchmarks/slurm/logs/pack_%j.log
# Batch wrapper around pack_run.sh: one allocation of 4 GPUs chewing a whole grid.
#
# The excluded nodes are the pascal ones (margpu024-027,029) and the drained
# margpu023, same list as the Hydra launcher. Pascal has no usable tensor cores
# and is slower than the turing/ampere nodes for these networks.
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
