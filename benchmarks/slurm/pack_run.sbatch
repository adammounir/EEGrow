#!/bin/bash
#SBATCH --job-name=eegrow_pack
#SBATCH --partition=tau
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --exclude=margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/logs/pack_%j.log
# --exclusive since 2026-08-18, because --mem is not enforced on this cluster: the
# turing nodes report RealMemory=191000 and AllocMem=0, so SLURM co-scheduled two
# 120G allocations on margpu020 (467745 + 467753) and left the node with 13.7 GB of
# FreeMem. The arbiter then becomes the kernel OOM killer, which uses SIGKILL, which
# no trap can catch: 85 cells died that way in the v5 campaign (73 in 467753 alone),
# each one releasing its claim and being re-claimed until MAX_SWEEPS gave up. A
# packing runner already decides its own co-tenancy through K -- letting the
# scheduler add a second tenant set on top of it is oversubscription by accident.
# The cost is queueing for a whole free node instead of a slice of a busy one.
# margpu021 joined the exclude list on 2026-08-16. It advertises `gpu:turing:3` while
# `nvidia-smi --query-gpu` on it answers "No devices were found": SLURM hands out a
# GRES the node cannot serve. A job that lands there does not fail to start, it runs
# every cell it claims against a device that is not there. That is the most expensive
# failure shape available -- work claimed, nothing produced -- so the node is excluded
# rather than merely guarded against (pack_run.sh now also refuses an allocation where
# nvidia-smi sees no card at all).
# Batch wrapper around pack_run.sh: one allocation chewing a whole grid.
#
# The output path is OUTSIDE the repo, and that is the point. It used to be
# benchmarks/slurm/logs/, inside the tree the deploy rsync mirrors with --delete --
# and no exclude matches `logs/` (`*_logs/` does not). The directory does not exist
# locally, so every deploy deleted the cluster's copy: the log of the host-RAM
# profiling job was destroyed roughly thirty minutes after it was written. A log that
# a routine deploy erases is not a record. /scratch/amounir/logs/ is never synced.
#
# 3 days, not 8 hours. The longest single cell of the grid is 38.2 h (cross_subject
# lee2019_mi bd_eegnex, measured); a cell cannot be split across allocations, so an
# 8 h allocation could never finish one and would kill it at the wall every time. The
# allocation has to outlive the worst cell plus the queue behind it. tau allows 7 days.
#
# The excluded nodes are the pascal ones (margpu024-027,029) and the drained
# margpu023, same list as the Hydra launcher. The reason is not "slower": torch
# 2.13.0+cu130 ships kernels for sm_75..sm_120 and those cards are Tesla P100 at
# sm_60, so CUDA fails outright with "no kernel image is available for execution on
# the device" (measured on margpu029). Same for the volta nodes at sm_70, which is
# why they are not requested either. Usable tau capacity is 28 GPUs on 9 nodes.
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
