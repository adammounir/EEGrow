#!/bin/bash
# Re-run the published grid's deep arms at O(1) input amplitude.
#
# Why the grid has to be re-run
# -----------------------------
# MOABB serves epochs in volts, so a motor-imagery trial sits at std ~5e-6 -- and ~3e-6 on
# the low-gain datasets. Handed to a convnet at that scale, ShallowFBCSPNet predicts one
# class for 99.7% of trials. On one BNCI2014_001 fold, 40 epochs, everything else fixed:
#   volts     acc 0.504  auc 0.539   initial train_loss 9.71
#   unit RMS  acc 0.649  auc 0.733   initial train_loss 1.04
# BatchNorm does not rescue it: the forward pass is scale-free, but AdamW's weight decay
# pulls against the ~1e6 first-layer weights that volt-scale inputs require.
#
# The published deep numbers therefore track amplifier gain. Spearman(log10 dataset
# amplitude, mean score) over the 8 pool datasets:
#   deep arms (bd_*, grow_*)      rho = +0.929   p = 0.0009
#   classic arms (csp, ts, fgmdm) rho = +0.119   p = 0.7789
# Both families see the same data and the same folds, so dataset difficulty is common to
# them; the gap between the two rho is what attributes the relation to numerics. Honest
# reservation: n = 8 datasets, and deep-minus-classic is +0.429 (p = 0.29), so the
# per-dataset claim is suggestive rather than established -- which is why this re-run
# exists instead of a footnote.
#
# What is NOT re-run, and why that matters
# ----------------------------------------
# The classic arms. CSP solves a generalised eigenproblem and the Riemannian arms work on
# covariances, so a global scale factor cannot move them; RMSScaler is not in their
# pipeline. Leaving their CSVs byte-identical keeps them as the fixed control against which
# the deep arms' movement is read. If the classic numbers changed too, the re-run would
# have proved nothing about scale.
#
# Results land in benchmarks/results_rms, not benchmarks/results
# --------------------------------------------------------------
# The CSV schema has no column saying whether a row was scaled, so writing into the old
# tree would produce a directory whose rows cannot be told apart -- and the comparison
# old-deep vs new-deep vs classic is the whole point. A separate root keeps all three
# readable at once and leaves the old grid intact as evidence.
#
# 1260 points, derived from the result CSVs by make_dlrescale_points.py rather than typed
# as a cartesian product, so the re-run is a like-for-like replacement of what actually
# ran. MaxArraySize is 1001 here, so this script takes an OFFSET:
#
#     sbatch benchmarks/slurm/dlrescale.sbatch                      # points 0-999
#     sbatch --export=ALL,OFFSET=1000 --array=0-259%8 \
#            benchmarks/slurm/dlrescale.sbatch                      # points 1000-1259
#
# The point file is ordered so the first 420 tasks are the four lowest-amplitude datasets,
# where the collapse should be largest. If the rescale moves nothing there, it moves
# nothing anywhere, and that is worth learning in an hour rather than in five days.
#SBATCH --job-name=dl-rms
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=margpu010
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-999%8
#SBATCH --output=/scratch/amounir/eegrow/dlrms_logs/rms_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POINTS=/scratch/amounir/eegrow/benchmarks/slurm/dlrescale.txt
IDX=$((SLURM_ARRAY_TASK_ID + ${OFFSET:-0}))
LINE=$(sed -n "$((IDX + 1))p" "$POINTS")
if [ -z "$LINE" ]; then
  echo "no point at index $IDX -- nothing to do"; exit 0
fi
read -r EVAL DS MODEL SEED ALIGN <<< "$LINE"
echo "[$(date)] dl-rms task=${SLURM_ARRAY_TASK_ID} idx=${IDX} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL dataset=$DS model=$MODEL seed=$SEED align=$ALIGN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=/scratch/amounir/eegrow/moabb_results_rms

cd /scratch/amounir/eegrow/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" align="$ALIGN" \
  dataset.resample=250 overwrite=true \
  results_dir=/scratch/amounir/eegrow/benchmarks/results_rms \
  suffix="rms_${MODEL}_s${SEED}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} (idx ${IDX}) done"
