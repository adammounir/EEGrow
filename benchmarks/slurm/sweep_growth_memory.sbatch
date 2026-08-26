#!/bin/bash
#SBATCH --job-name=eegrow_gsweep
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/logs/gsweep_%j.log
# margpu021 joined the exclude list on 2026-08-16. It advertises `gpu:turing:3` while
# `nvidia-smi --query-gpu` on it answers "No devices were found": SLURM hands out a
# GRES the node cannot serve. A job that lands there does not fail to start, it runs
# every cell it claims against a device that is not there. That is the most expensive
# failure shape available -- work claimed, nothing produced -- so the node is excluded
# rather than merely guarded against (pack_run.sh now also refuses an allocation where
# nvidia-smi sees no card at all).
# Does a growing arm's device peak still depend on the size of the training fold?
#
# THE QUESTION THIS SETTLES
# -------------------------
# profile_grid_memory.py rests on one assumption: device memory is a function of
# (batch, n_chans, n_times, n_outputs) and the architecture, never of the data. That
# is what licenses measuring 108 cells on synthetic tensors. For the fixed arms it
# holds. For the growing arms it did not, because GromoGrowth.on_epoch_end
# materialised the WHOLE epoch on the growth device before the growth step -- so the
# peak carried the training fold with it, and 512 synthetic samples are not a fold.
#
# The consequence was not theoretical: the 2026-08-16 smoke run lost 18 cells to CUDA
# OOM, every one of them a growing arm, every one refused against its own declared
# ceiling while the card sat 6 GiB free. The ceiling was derived from the profile, and
# the profile was measured on 512 samples.
#
# skorch_integration.py now keeps those batches on the host and lets gromo move them
# per batch, which it already did (`x, y = x.to(device), y.to(device)` in both
# compute_statistics and evaluate_model). This sweep is the check that the fix did
# what it claims: same cells, three data volumes, one card.
#
# HOW TO READ THE RESULT
# ----------------------
# For bnci2014_001 the epoched trials are 22 x 1001 float32 = 88 KiB each, so 512 /
# 4096 / 16384 samples are 45 / 361 / 1444 MiB of data. If the peak is flat across the
# three, the peak no longer carries the fold and ONE measurement per cell is again
# sufficient -- the campaign plan can be rebuilt on the existing methodology. If it
# still climbs, the fix is incomplete and no ceiling derived from synthetic data can
# be trusted for a growing arm, whatever margin is put on it.
#
# The 4 growing arms only: the fixed arms never had the dependence (none of them OOMed
# in the smoke), so measuring them here would spend a scarce GPU confirming a null.
#
# Runs entirely on synthetic tensors with the shapes read from the memo
# (profile_mem/shapes.json, all 12 datasets already present), so it makes NO call into
# the MOABB cache and can run while the cache warm-up holds it.
#
#   sbatch benchmarks/slurm/sweep_growth_memory.sbatch
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
cd /scratch/amounir/eegrow/benchmarks
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export EEGROW_BENCH_ROOT=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module load cuda/13.1 2>/dev/null || true

echo "=== node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for N in 512 4096 16384; do
  echo "--- n_samples=$N"
  python profile_grid_memory.py \
      --models grow_shallow,grow_sccnet,grow_deep,grow_eegnex \
      --datasets bnci2014_001 \
      --n-samples "$N" \
      --out "/scratch/amounir/gsweep_n${N}.json" \
      --shapes /scratch/amounir/eegrow/benchmarks/profile_mem/shapes.json
done

echo "=== DONE $(date -Is)"
