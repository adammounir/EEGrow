#!/bin/bash
#SBATCH --job-name=eegrow_dxr
#SBATCH --partition=tau
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/amounir/logs/dxr_%j.log
#
# La matrice donneur-receveur (claim 2 du chantier ICLR). Voir
# benchmarks/donor_receiver.py pour le protocole ; ce fichier ne décide que deux
# choses, et les deux sont des choix de cohabitation avec la campagne en cours.
#
# gpu:ampere:1, PAS turing. Deux raisons, dans cet ordre :
#
# 1. NE PAS VOLER LA CAMPAGNE. La grille finale demande gpu:turing:3 et six
#    allocations (512163-512168) attendent déjà en PENDING(Priority) sur ces cartes.
#    Une demande turing de plus se met dans la même file et retarde le chemin
#    critique (schirrmeister, ~70 h restantes) pour un travail qui tient en heures.
#    Les nœuds ampere du tau (margpu007-008) sont libres à côté.
#
# 2. LA COMPARAISON EST INTERNE À LA MATRICE. L'argument qui impose `--gres-type
#    turing` à la campagne -- une différence appariée grow_X - bd_X ne doit pas avoir
#    une différence de matériel à l'intérieur -- ne s'applique pas ici : toutes les
#    lignes de cette matrice sont le MÊME bras, et on les compare entre elles. Ce qui
#    compte est que les 156 fits soient sur la même classe de carte, pas laquelle.
#    Une carte par classe est donc suffisant, et c'est pour ça que ce job est
#    schedulable tout de suite alors que la campagne attend.
#
#    Corollaire à ne pas oublier au moment d'écrire le papier : les scores de transfert
#    de cette matrice ne sont PAS comparables case à case aux scores within_session de
#    la campagne (turing). Ils ne le sont pas non plus par construction -- 100 % des
#    essais contre 80 %, pas de fold -- donc rien ne se perd. Ce qui traverse, c'est le
#    RANG des donneurs, et il est mesuré à matériel constant.
#
# Les nœuds pascal (margpu024-027, 029) sont IDLE et inutilisables : sm_60, sur lequel
# torch 2.13+cu130 échoue avec "no kernel image is available". Ne pas les demander en
# voyant `idle` dans sinfo.
set -uo pipefail

export ROOT=/scratch/amounir/eegrow_budget
cd "$ROOT/benchmarks"
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
# $ROOT/src EN PREMIER. L'env `bench` porte un .pth editable qui pointe sur
# /scratch/amounir/eegrow (l'arbre d'août, SANS l'abstention s=0 du 25/08) : sans cette
# ligne le script importerait ce paquet-là, ré-ajouterait des neurones morts et
# gonflerait `params_end` -- c'est-à-dire fausserait exactement la quantité mesurée.
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

# Le garde-fou de la campagne, repris tel quel : ce script partage un nœud avec
# d'autres tenants et un fit qui part en vrille doit échouer chez lui.
export EEGROW_CUDA_FRACTION="${EEGROW_CUDA_FRACTION:-0.30}"

DATASET="${DATASET:-cho2017}"
OUT="${OUT:-/scratch/amounir/dxr/$DATASET}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
SEEDS="${SEEDS:-0 1 2}"
DONORS="${DONORS:-}"
NS="${N_SHARDS:-1}"
# SLURM_ARRAY_TASK_ID quand on tourne en array, sinon la variable explicite. Le shard
# est l'indice DANS LA MATRICE, pas un numéro de machine : les 156 unités sont
# distribuées par `i % n_shards`, donc les shards se partagent le travail sans
# communiquer et un shard tué ne laisse qu'un trou que la reprise rattrape.
SH="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"

python -c "import eegrow, sys; p=eegrow.__file__; \
  sys.exit(0) if p.startswith('$ROOT/src') else sys.exit('FATAL: eegrow imported from '+p)"

run_shard () {
  python "$ROOT/benchmarks/donor_receiver.py" \
    --dataset "$DATASET" \
    --out "$OUT" --cache "$CACHE" --seeds $SEEDS \
    ${DONORS:+--donors $DONORS} \
    ${THREADS:+--threads $THREADS} \
    --n-shards "$NS" --shard "$1"
}

# DEUX MODES DE FAN-OUT. Les fits d'une matrice sont comparés ENTRE EUX -- c'est un
# classement de donneurs -- donc ils doivent tourner sur du matériel homogène, sans quoi
# on remet une différence de machine à l'intérieur de la quantité mesurée (exactement ce
# que `--gres-type turing` empêche du côté de la campagne).
#
# `LOCAL_FANOUT=1` (défaut) : un nœud, N processus. Homogénéité garantie par
# construction, mais plafonnée à un nœud -- 56 CPU logiques, donc ~7 shards à 8 threads.
#
# `LOCAL_FANOUT=0` : job array. C'EST LE MODE RÉELLEMENT UTILISÉ pour cho2017
# (job 513589, 26 tâches). L'homogénéité n'est alors pas gratuite, elle est OBTENUE par
# `--exclude` : marg[003-012] partagent la même classe de CPU (56 cœurs) et les mêmes
# 515 Go, marg[013-032] n'ont que 128 Go et marg[001-002] sont hors-jeu. Toute
# soumission en array DOIT donc porter `--nodes=1` et l'exclusion, sinon
# (a) `--array` + `--nodelist` fait demander TOUS les nœuds listés à CHAQUE tâche
#     (le nodelist fixe le nombre minimum de nœuds) -- c'est ce qui a tué le job 513581 ;
# (b) les tâches se dispersent sur deux classes de mémoire et le classement de donneurs
#     hérite d'une différence de machine.
#
# Les shards se partagent le travail par indice, pas par sujet : chacun recharge les 52
# sujets (2,0 Go, 46 s depuis le cache) et il n'y a rien à synchroniser -- un CSV par
# (donneur, seed), écrit en dernier, donc deux shards ne peuvent pas se marcher dessus
# et une reprise saute ce qui est fait.
set -x
if [ "$NS" -gt 1 ] && [ "${LOCAL_FANOUT:-1}" = "1" ]; then
  pids=()
  for i in $(seq 0 $((NS - 1))); do
    run_shard "$i" &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  exit $rc
fi
run_shard "$SH"
