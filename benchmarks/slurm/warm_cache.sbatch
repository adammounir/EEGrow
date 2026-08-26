#!/bin/bash
#SBATCH --job-name=eegrow_warmcache
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/amounir/logs/warmcache_%j.log
# Fill the shared MOABB cache serially, one process at a time, before any packed run.
#
# WHY THIS IS A PREREQUISITE AND NOT A CONVENIENCE
# ------------------------------------------------
# The MOABB BIDS cache is not safe for concurrent writers, and its error path is
# destructive: bids_interface.py:2370 calls `interface.erase()` to "remove partial
# cache", deleting the entry from the SHARED directory, so a tenant that stumbles
# destroys the other tenants' work. Measured cost so far: 32 cells of the first packed
# campaign, then 64 of the 2026-08-16 smoke's 128 cells, plus 19 destroyed physionetmi
# sessions.
#
# A cache that is complete FOR THE KEY THE RUN READS removes the failure entirely:
# tenants only read, nothing is partial, and the erase path is never entered.
#
# WHY IT IS NO LONGER profile_host_ram.py -- AND WHAT THAT COST
# ------------------------------------------------------------
# This script used to call the host-RAM probe, on the stated grounds that it "builds the
# paradigm exactly as run_moabb_hydra.py does", said to be "empirically confirmed". That
# was false, and it was the root cause of the whole cache saga:
#
#     profile_host_ram.py : MotorImagery(fmin=8,   fmax=32)     -> ints
#     run_moabb_hydra.py  : MotorImagery(fmin=8.0, fmax=32.0)   -> floats
#
# MOABB keys the cache on `get_digest(process_pipeline)`, a hash of the pipeline's
# *repr*. Those two render as `Band Pass Filter (8–32 Hz)` and
# `Band Pass Filter (8.0–32.0 Hz)`, so they hash differently and key two disjoint caches
# living side by side in the same directory. Every warm-up therefore filled the cache
# the campaign never reads, and every packed run in fact started cold: K tenants missed
# the same entry simultaneously, all derived it, all wrote it, and collided.
#
# It also explains why the warm-up "took 11 minutes instead of 1.5 h" and looked like
# good news -- it was hitting its own already-warm key -- and why a checker that counted
# session directories reported "cache complete" throughout. The two keys are visible on
# disk: physionetmi carries 4 desc- hashes, two per code path.
#
# warm_cache.py imports `check_cache.build_campaign`, the same function the gate checks
# against, so "what we warm" and "what we verify" are one construction and cannot drift.
#
# ONLY THE ARRAY LEVEL IS WARMED, AND THAT IS SUFFICIENT
# -----------------------------------------------------
# The campaign calls `paradigm.get_data`, which returns arrays, and no evaluation config
# sets `return_epochs`, so the epochs entry is never on the read path. It also cannot be
# back-filled: MOABB starts from the deepest cached level, so while the array entry
# exists it returns immediately without regenerating the epochs one. Requiring it would
# demand a warm-up that is impossible as well as pointless.
#
# --partition=normal, no GPU: this is I/O and single-threaded decoding. 8 h because the
# 92 outstanding entries come from raw, including all of cho2017 (52 subjects) and
# schirrmeister2017 (14 subjects at ~27 GiB RSS each).
#
# Order puts the fully-cold datasets first, so a run that goes wrong fails on the
# expensive ones early rather than after everything cheap has succeeded.
#
#   sbatch benchmarks/slurm/warm_cache.sbatch
#   python benchmarks/check_cache.py --cache /scratch/amounir/moabb_cache   # the gate
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
cd /scratch/amounir/eegrow/benchmarks
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export EEGROW_BENCH_ROOT=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== node=$(hostname) start=$(date -Is)"

python warm_cache.py \
    --cache /scratch/amounir/moabb_cache \
    --datasets cho2017,schirrmeister2017,bnci2014_002,physionetmi,lee2019_mi,shin2017a,weibo2014,bnci2014_001,bnci2015_001,zhou2016,bnci2014_004,alexmi

echo "=== contrôle final"
cd /scratch/amounir/eegrow
python benchmarks/check_cache.py --cache /scratch/amounir/moabb_cache

echo "=== DONE $(date -Is)"
