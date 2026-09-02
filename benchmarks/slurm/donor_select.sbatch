#!/bin/bash
#SBATCH --job-name=eegrow_dsel
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/amounir/logs/dsel_%A_%a.log
#
# Claim 3 : l'intervention. Voir benchmarks/donor_select.py pour le protocole.
#
# CE QUI CHANGE PAR RAPPORT À donor_receiver.sbatch, ET POURQUOI
#
# 1. `--mem=16G`. Le premier jet demandait 64 G en raisonnant que la cellule charge les
#    109 sujets puis en CONCATÈNE 20 dans un seul tenseur (~900 essais x 64 canaux x 752
#    échantillons). MESURÉ par la sonde 514158 : `MaxRSS=2.9G`. Les 109 sujets font
#    ~950 Mo en float32, le pool en fait 175. Demander 64 G ne protégeait de rien et
#    rendait le job plus dur à caser. Même remarque pour `--time` : 2 h suffisent
#    largement pour un shard (mesure ci-dessous), 12 h retardaient le backfill.
#
# 2. `gpu:1` GÉNÉRIQUE, et non `gpu:ampere:1`. Le premier jet demandait ampere pour
#    l'homogénéité matérielle : les règles se comparent APPARIÉES (`params_top` contre
#    `random` sur le même pli, la même seed, les mêmes sujets de test), donc deux
#    cellules appariées sur des cartes différentes embarqueraient une différence de
#    machine dans la différence mesurée. Mesuré le 02/09 : ce choix coûtait la journée.
#    margpu007/008 ont leurs 3+3 GPU alloués et sont en MIXED+PLANNED (réservés par le
#    backfill) pour neuf jobs d'un autre utilisateur à priorité 8438 contre 7203 ici,
#    chacun demandant 24 h. La sonde attendait 45 min sans espoir de démarrage.
#
#    L'argument d'homogénéité tient sur le fond mais pas sur la forme : l'affectation
#    des cellules aux cartes se fait par index de shard, donc INDÉPENDAMMENT de la
#    règle. Un effet carte est alors du bruit réparti sur toutes les règles, pas un
#    biais qui pousse une règle plutôt qu'une autre. On l'enregistre (`node`,
#    `gpu_name` dans chaque CSV) au lieu de l'interdire : l'analyse peut le tester, et
#    si le test ressort, relancer sur une seule classe de carte reste possible.
#
# 3. `--partition=gpu-best` ET NON `tau`. Vérifié le 02/09 : `tau` ne donne accès qu'à
#    15 nœuds (margpu007-008, 017-029) dont tous les GPU utilisables étaient alloués,
#    avec neuf jobs d'un autre utilisateur devant moi à priorité 8438 contre 7203.
#    `gpu-best` couvre margpu[002-009,012-029] -- 26 nœuds -- et `AllowGroups=ALL,
#    AllowQos=ALL` : l'accès était là depuis le début, je ne l'avais jamais regardé.
#    margpu002/003 (Quadro RTX 6000, 2 GPU chacun) y étaient complètement `idle`.
#
#    LE PRIX : `PriorityTier=1` contre 100 pour `tau`, et `PreemptMode=REQUEUE`. Un job
#    de `tau` peut préempter celui-ci, qui repart alors du début. C'est exactement le
#    cas d'usage pour lequel une cellule écrit son CSV EN DERNIER et pour lequel la
#    reprise saute les CSV existants : une préemption coûte au pire la cellule en cours,
#    jamais l'array. D'où `--requeue`. Ne pas mettre un travail non reprenable ici.
#
# 4. `--exclude` sur les nœuds pascal (margpu024-027, 029) : MESURÉ le 02/09 sur
#    margpu024 et margpu026, torch 2.13.0+cu130 lève
#    `AcceleratorError: CUDA error: no kernel image is available` (sm_60 absent des
#    kernels compilés). Ils apparaissent `idle` justement parce que personne ne peut
#    s'en servir -- sans cette ligne le `gpu:1` générique irait s'y planter en premier,
#    ce sont les seuls libres. margpu023 est drained (Not responding).
#    Contre-épreuve sur margpu002 le même jour : Quadro RTX 6000, sm_75, matmul OK.
set -uo pipefail

export ROOT=/scratch/amounir/eegrow_budget
cd "$ROOT/benchmarks"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
# $ROOT/src EN PREMIER : l'env `bench` porte un .pth editable vers /scratch/amounir/eegrow
# (l'arbre d'août, sans l'abstention s=0 du 25/08). Sans cette ligne on importerait
# celui-là. Ici le modèle aval est fixe, donc rien ne grandit -- mais le garde-fou reste,
# parce que le jour où on relancera ce script avec `--model grow_shallow` il comptera.
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
export EEGROW_CUDA_FRACTION="${EEGROW_CUDA_FRACTION:-0.30}"

DATASET="${DATASET:-physionetmi}"
OUT="${OUT:-/scratch/amounir/dsel/$DATASET}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
RANKING="${RANKING:-$ROOT/benchmarks/ranking_$DATASET.csv}"
MODEL="${MODEL:-bd_shallow}"
K="${K:-20}"
FOLDS="${FOLDS:-4}"
SEEDS="${SEEDS:-0 1 2}"
DRAWS="${DRAWS:-6}"
NS="${N_SHARDS:-1}"
SH="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"

python -c "import eegrow, sys; p=eegrow.__file__; \
  sys.exit(0) if p.startswith('$ROOT/src') else sys.exit('FATAL: eegrow imported from '+p)"

set -x
python "$ROOT/benchmarks/donor_select.py" \
  --dataset "$DATASET" --model "$MODEL" --ranking "$RANKING" \
  --out "$OUT" --cache "$CACHE" \
  --k "$K" --folds "$FOLDS" --seeds $SEEDS --random-draws "$DRAWS" \
  ${THREADS:+--threads $THREADS} \
  --n-shards "$NS" --shard "$SH"
