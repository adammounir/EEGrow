#!/bin/bash
#SBATCH --job-name=eegrow_ramprofile
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/amounir/eegrow/benchmarks/slurm/logs/hostram_%j.log
# Peak host RAM of one tenant per dataset. See benchmarks/profile_host_ram.py for why
# the GPU profile cannot answer this and why the bound is per-dataset, not per-protocol.
#
# --partition=normal, NOT tau. This probe makes no CUDA call, so asking for a GPU
# would idle a scarce device and queue behind the very jobs it is meant to unblock.
# `normal` had 24 of 32 nodes idle when this was written; tau had 6 of 15.
#
# --mem=120G matches what pack_run.sbatch requests for a whole node, so a dataset
# that does not fit here would not fit a single tenant there either -- the failure
# is then itself the measurement, not an artefact of an under-provisioned probe.
#
# 4 cores: the load is I/O and single-threaded decoding. More would queue for nothing.
#
#   sbatch benchmarks/slurm/profile_host_ram.sbatch
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
cd /scratch/amounir/eegrow/benchmarks
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export EEGROW_BENCH_ROOT=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
# One BLAS thread, as in the packed runner: the figure has to describe a tenant that
# behaves like a production tenant, and thread pools carry per-thread arenas.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== node=$(hostname) mem_limit=$(ulimit -v) start=$(date -Is)"

python profile_host_ram.py \
    --out /scratch/amounir/eegrow/benchmarks/profile_mem/host_ram.json \
    --cache /scratch/amounir/moabb_cache

echo "=== DONE $(date -Is)"
