#!/bin/bash
#SBATCH --job-name=eegrow_dalloc
#SBATCH --partition=tau,gpu,gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|turing|hopper
#SBATCH --exclude=margpu001,margpu004,margpu020,margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/amounir/logs/dalloc_%A_%a.log
#
# L'AXE ENTRE LA SÉLECTION ET `all_pool`. Voir le docstring de donor_allocate.py : à
# budget d'essais FIXÉ, beta interpole entre « étaler sur tous les donneurs » (beta=0) et
# « tout donner aux meilleurs » (beta grand, qui redevient la sélection top-K). C'est le
# contrôle que l'étage 4 n'avait pas : `all_pool` gagnait avec plus de sujets ET plus
# d'essais, et rien ne séparait les deux.
#
# STATUT : POST-HOC ET DESCRIPTIF, comme le bras `all_pool`. Il n'amende aucun endpoint
# pré-enregistré. Ne pas écrire « confirme » à son sujet.
#
# DIMENSIONNEMENT, MESURÉ ET NON EXTRAPOLÉ. Les cellules de l'étage 3 déjà sur disque
# donnent le repère exact : `f0__r0__k10__acc_top__seed0.csv` sur cho2017 rapporte
# `n_train=2040`, `fit_seconds=66.7` sur A40. Les budgets d'ici sont 600 / 1 000 / 2 000
# essais, soit le même régime ou moins, plus UNE cellule à budget plein par (pli, seed)
# qui vaut 7 800 essais -- c'est-à-dire exactement une cellule `all_pool`, mesurée entre
# 19 et 39 min sur les shards de 523644.
#
#     336 cellules / dataset = 4 plis x 3 seeds x (3 budgets x 9 + 1 budget plein)
#     par shard sur 4 shards : ~84 cellules, dont 3 à budget plein
#     estimation : 84 x ~70 s + 3 x ~5 min ~= 1 h 55  ->  walltime 4 h, marge x2
#
# --mem=24G ET NON 16G : la cellule à budget plein concatène les 7 800 essais du vivier,
# comme `all_pool`, pour qui 16 G avait été jugé trop juste. Le plan mélangeant les deux
# tailles, on dimensionne sur la plus grosse.
#
# --constraint AU LIEU D'UNE LISTE DE NŒUDS. L'env `bench` porte torch cu130 (sm_75+) :
# les pascal (sm_60) et les volta (sm_70) ne peuvent pas exécuter une conv2d, c'est
# mesuré (job 501434_0, `Tesla P100 with CUDA capability sm_60 is not compatible`). La
# contrainte exprime la vraie condition ; l'--exclude ne garde que les cas que la
# contrainte ne couvre pas : margpu023 (DOWN) et margpu020/021 (cartes fantômes, où
# SLURM alloue une carte de plus qu'il n'en existe et le process meurt sur
# `torch.cuda.is_available() is False`). margpu001 et margpu004 sont volta ET dans la
# partition `gpu` : la contrainte les écarte déjà, l'--exclude est une ceinture.
#
# LA PARTITION `gpu` EST DANS LA LISTE, ET C'EST DÉLIBÉRÉ. margpu[010-011] n'appartiennent
# qu'à elle, donc aucune partition de tier supérieur ne peut les préempter, malgré leur
# PriorityTier=1. C'est ce qui a fait démarrer 523645 30 h plus tôt que l'estimation.
#
# APPARIEMENT -- LA RAISON POUR LAQUELLE ON TOURNE DEPUIS CE CHECKOUT-CI. Les cellules
# d'ici se comparent APPARIÉES à celles de dsel2/ et dsel_all/, sujet de test par sujet
# de test. $ROOT est donc /scratch/amounir/eegrow_budget, épinglé au même
# EEGROW_SHA=58c6bee que les cellules déjà écrites, et les plis viennent de
# `donor_select.folds_of` importée -- jamais réimplémentée. Changer l'un des deux casse
# l'appariement en silence, sans qu'aucune colonne ne le signale.
#
#     DATASET=cho2017    N_SHARDS=4 sbatch --array=0-3 slurm/donor_alloc.sbatch
#     DATASET=lee2019_mi N_SHARDS=4 sbatch --array=0-3 slurm/donor_alloc.sbatch
#
# ANALYSE : ni donor_select_analysis.py ni donor_select2_analysis.py. Leurs globs
# attendent les formats v1/v2 et leurs unités de rééchantillonnage n'ont pas de sens ici
# (il n'y a ni règle ni réplicat). Sortie dans un répertoire séparé pour la même raison.
set -uo pipefail

export ROOT=/scratch/amounir/eegrow_budget
cd "$ROOT/benchmarks"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
# $ROOT/src EN PREMIER : l'env `bench` porte un .pth editable vers /scratch/amounir/eegrow
# (l'arbre d'août, sans l'abstention s=0 du 25/08). Sans cette ligne on importerait
# celui-là.
export PYTHONPATH="$ROOT/src:$ROOT/benchmarks"
export EEGROW_BENCH_ROOT="$ROOT/benchmarks"
if [ -z "${EEGROW_SHA:-}" ] && [ -r "$ROOT/.eegrow_sha" ]; then
  export EEGROW_SHA="$(tr -d '[:space:]' < "$ROOT/.eegrow_sha")"
fi
export MNE_DATA=/scratch/amounir/mne_data
module load cuda/13.1 2>/dev/null || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EEGROW_CUDA_FRACTION="${EEGROW_CUDA_FRACTION:-0.45}"

DATASET="${DATASET:-cho2017}"
OUT="${OUT:-/scratch/amounir/dalloc/$DATASET}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
RANKING="${RANKING:-$ROOT/benchmarks/ranking_$DATASET.csv}"
MODEL="${MODEL:-bd_shallow}"
BUDGET_K="${BUDGET_K:-3 5 10}"
BETA="${BETA:-0 1 2 4 8}"
MEASURE="${MEASURE:-acc_probe params_probe}"
SEEDS="${SEEDS:-0 1 2}"
FOLDS="${FOLDS:-4}"
NS="${N_SHARDS:-1}"
SH="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"

python -c "import eegrow, sys; p=eegrow.__file__; \
  sys.exit(0) if p.startswith('$ROOT/src') else sys.exit('FATAL: eegrow imported from '+p)"

set -x
python "$ROOT/benchmarks/donor_allocate.py" \
  --dataset "$DATASET" --model "$MODEL" --ranking "$RANKING" \
  --out "$OUT" --cache "$CACHE" \
  --budget-k $BUDGET_K --budget-full --beta $BETA --measure $MEASURE \
  --seeds $SEEDS --folds "$FOLDS" \
  ${MAX_EPOCHS:+--max-epochs $MAX_EPOCHS} \
  ${THREADS:+--threads $THREADS} \
  --n-shards "$NS" --shard "$SH"
