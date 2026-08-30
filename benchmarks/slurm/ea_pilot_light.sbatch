#!/bin/bash
# Pilot 1, light half: the ALIGNED arm (align=euclidean) of bnci2014_001.
#
# What this measures
# ------------------
# Not "does Euclidean Alignment help" -- that is settled in the literature. What we
# want is the INTERACTION
#     delta = (grow_EA - grow_raw) - (fixed_EA - fixed_raw)
# i.e. whether architecture growth benefits from alignment MORE than a fixed net does.
# The mechanism: gromo allocates width from the data's second-order statistics (the S
# tensor is a variance), so on multi-subject EEG a growing net spends capacity
# modelling subject nuisance that a fixed net cannot even try to model. Removing the
# nuisance should therefore pay the growing arm more. That is a differential,
# falsifiable prediction, and it explains the currently negative fixed-vs-growing
# result rather than explaining it away.
#
# Why these datasets, why this order
# ----------------------------------
# The three pilot datasets are a DOSE-RESPONSE on subject count, which is what the
# hypothesis is actually about:
#     bnci2014_001 (9 subjects, 22 ch)  <- this script
#     cho2017      (52 subjects, 64 ch) \  ea_pilot_heavy.sbatch
#     physionetmi  (109 subjects, 64 ch) /
# If the interaction is real it must GROW with subject count. A monotone trend over
# three points is far stronger evidence than one significant cell.
# lee2019_mi is excluded on purpose despite its 54 subjects: MOABB 1.5.0 silently
# drops one of its two sessions (off-by-one on session naming), so its raw arm is not
# comparable to the rest of the grid.
#
# Model pairs: grow_shallow/bd_shallow (the only non-losing pair on the raw grid,
# +0.008) and grow_eegnex/bd_eegnex (the worst, -0.032). They bracket the effect, so
# the interaction is measured where growth currently wins AND where it currently
# loses.
#
# The raw arm is NOT recomputed
# -----------------------------
# All 60 raw cells already exist at 250 Hz under
#   results/cross_subject/<ds>/<model>__seed<N>.csv
# align=none is the default and leaves those filenames untouched; the aligned run
# writes <model>__easubject__seed<N>.csv, and MOABB's own hdf5 cache is namespaced
# both by the tagged suffix and by the paradigm class (EAMotorImagery/). So this
# array can never overwrite or silently reuse the control arm -- the accident that
# cost us the native-rate runs cannot repeat here.
#
# resample is pinned in the dataset yaml (250.0); it is passed again on the CLI so the
# rate is visible in the log of every single task.
#SBATCH --job-name=ea-light
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-19%6
#SBATCH --output=/scratch/amounir/eegrow/ea_logs/light_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/ea_pilot_light.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED <<< "$LINE"
echo "[$(date)] ea-light task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL dataset=$DS model=$MODEL seed=$SEED align=euclidean"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=/scratch/amounir/eegrow/moabb_results

cd /scratch/amounir/eegrow/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" \
  align=euclidean \
  dataset.resample=250 overwrite=true suffix="ea_${MODEL}_s${SEED}"
