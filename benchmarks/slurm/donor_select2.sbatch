#!/bin/bash
#SBATCH --job-name=eegrow_dsel2
#SBATCH --partition=tau,gpu-best
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu020,margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/amounir/logs/dsel2_%A_%a.log
#
# Claim 3 v2 : le protocole qui peut trancher (c). Voir benchmarks/donor_select.py,
# section « V2 », et analysis/donor_select2_analysis.py pour la règle de décision.
#
# CE QUI CHANGE PAR RAPPORT À donor_select.sbatch
#
# 1. `--candidates 40 --reps 8`. C'est tout le correctif. En v1 une règle déterministe
#    ne produisait qu'un pool par pli, donc le contraste entre deux règles avait 4
#    réplicats et non 109 -- et un seul pli (le 3) en renversait le signe. Ici on tire 40
#    candidats par (pli, réplicat), PARTAGÉS par les cinq règles : les règles
#    déterministes acquièrent une distribution de pools, et deux règles d'un même
#    réplicat sont comparées sur le même ensemble de sujets disponibles.
#    Unité de rééchantillonnage : 4 plis x 8 réplicats = 32.
#
# 2. `--k 5 10 20`. Les pools sont emboîtés en K (l'ordre est calculé une fois par
#    (pli, réplicat, règle), les K sont des préfixes), y compris pour `random`. Mesuré
#    au --dry-run, le recouvrement resid_top ∩ acc_top vaut 7 % à K=5, 29 % à K=10,
#    56 % à K=20 : K=5 est le seul régime où le contraste a vraiment de la prise, et
#    c'est pour ça qu'il est l'endpoint primaire déclaré. K=20 sert de contrôle de
#    saturation -- il doit reproduire l'indétermination de v1, et si le gain sur random
#    n'y décroît PAS, mon explication de l'échec de v1 est fausse.
#
# 3. `--time=03:00:00` au lieu de 2 h. 480 cellules sur 8 shards = 60 cellules par
#    shard, mais elles ne coûtent pas toutes pareil : le fit est à peu près
#    proportionnel à K, et K=20 (~900 essais) a été mesuré à 39.9 s en v1. Le shard est
#    dimensionné sur la sonde de coût lancée AVANT cet array, pas sur une extrapolation
#    -- c'est l'erreur qui avait fait demander 12 h et 64 G en v1.
#
# 4. `--out /scratch/amounir/dsel2/...`, répertoire SÉPARÉ. Les 72 cellules de v1
#    restent où elles sont : deux protocoles dont l'unité d'analyse diffère ne doivent
#    jamais pouvoir se retrouver dans le même DataFrame. Les noms de fichiers diffèrent
#    aussi (`f0__r3__k5__resid_top__seed3.csv`), donc même un glob distrait ne peut pas
#    les mélanger.
#
# margpu021 EST DANS L'EXCLUSION POUR UNE AUTRE RAISON QUE LES PASCAL (ajouté 02/09).
# `scontrol show node margpu021` annonce CfgTRES gres/gpu=3 pour 2 cartes turing
# réellement utilisables. SLURM place donc des jobs sur une 3ᵉ carte qui n'existe pas,
# et le job meurt en 12-22 s sur `torch.cuda.is_available() is False`. Mesuré : 16
# shards tués comme ça sur 514206 (4,5,6,7), 514207 (2,5,6,7), 514208 (0,7) et 514209
# (0,2,3,4,6,7), soit 960 cellules perdues -- et ZÉRO shard tué ailleurs que sur
# margpu021, pendant que d'autres shards du MÊME array y tournaient normalement (ils
# avaient attrapé une carte réelle : 514207_1 y a fini ses 120 cellules). Ce n'est donc
# pas un nœud en panne, c'est une carte fantôme, et le garde-fou `pick_device` de
# utils.py a fait exactement son travail en refusant le CPU : la perte est propre, aucune
# cellule calculée sur le mauvais matériel n'a été écrite.
#
# margpu020 A LE MÊME DÉFAUT, ajouté à l'exclusion le 02/09 au soir. Gres=gpu:turing:4
# annoncé, AllocTRES gres/gpu=3, et `515639_2` (bras all_pool) y est mort en 12 s sur le
# même `torch.cuda.is_available() is False` pendant que 6 shards frères tournaient
# ailleurs. Aucune cellule dsel2 n'a été perdue dessus -- l'exclusion est préventive.
#
# RATTRAPAGE : ne PAS resoumettre les mêmes indices de shard. `run_cell` saute toute
# cellule dont le CSV existe (donor_select.py:300), donc il suffit de relancer le plan
# COMPLET avec un autre `--n-shards` : le round-robin redistribue les trous sur tous les
# shards au lieu de les laisser en blocs de 60-120.
#
# LE NOUVEAU MODULO DOIT ÊTRE PREMIER AVEC L'ANCIEN. Erreur commise et corrigée le
# 02/09 : relancé à N_SHARDS=12 sur un plan sharde à 8, `514428_0` a fini en 1 min en
# annonçant « progress 80/80 cells, 0.0 min » -- zéro trou. gcd(8, 12) = 4, donc le
# nouveau shard s ne recouvre que les anciens s mod 4 et s mod 4 + 4 : les trous se
# CONCENTRENT (0 pour s≡0, 80 pour s≡2) au lieu de se répartir. Avec N_SHARDS=13
# (premier avec 8) chaque nouveau shard touche les 8 anciens, soit ~20 trous chacun sur
# un plan à 960 cellules, et des jobs assez courts pour passer en backfill.
#
# Avec une dépendance `afterany` si un array du même dataset tourne encore :
# deux processus qui calculent la même cellule écriraient le même chemin en concurrence.
#
# PARTITION `tau,gpu-best` ET NON `gpu-best` SEUL (02/09). Mesuré sur scontrol :
# tau a PriorityTier=100 contre 1 pour gpu-best, PreemptType=preempt/partition_prio,
# et je suis dans le groupe tau. Un job soumis aux deux démarre dans celle qui peut le
# lancer en premier, et une place tau lui revient AVANT toute la file gpu-best. Ce qu'il
# ne faut PAS en attendre : tau ne débloque pas magiquement les nœuds turing. Ils sont
# tenus par jjobard, qui est tau lui aussi -- même tier, donc pas de préemption. Le seul
# nœud tau réellement préemptible aujourd'hui est margpu008 (2 jobs gpu-best d'un autre
# utilisateur). Le gain vient de la priorité, pas de la préemption.
#
# Le reste (exclusion des nœuds pascal sm_60, gpu:1 générique,
# --requeue avec reprise par CSV existant) est justifié dans donor_select.sbatch et n'a
# pas changé.
# RÉPLICATION cho2017 : ce même fichier, paramètres par l'environnement, AUCUN clone.
# Un second .sbatch aurait deux en-têtes à maintenir et divergerait en silence dès la
# première correction appliquée à un seul des deux.
#
#     DATASET=cho2017 K_LIST="3 5 10" CANDIDATES=20 REPS=8 \
#       N_SHARDS=8 sbatch --array=0-7 --time=02:00:00 slurm/donor_select2.sbatch
#
# M=20 et non 40 : le vivier n'y fait que 39 sujets, donc M=40 rendrait TOUT le vivier
# candidat et reproduirait le défaut de v1. Ce qui se conserve d'un dataset à l'autre
# est M/vivier (49 % → 51 %) et K/M (12/25/50 % → 15/25/50 %), pas M ni K.
# L'endpoint primaire reste K=5, celui de l'original : voir
# analysis/PREREG_donor_select2_cho2017.md, écrit avant la première cellule.
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
export EEGROW_CUDA_FRACTION="${EEGROW_CUDA_FRACTION:-0.30}"

DATASET="${DATASET:-physionetmi}"
OUT="${OUT:-/scratch/amounir/dsel2/$DATASET}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
RANKING="${RANKING:-$ROOT/benchmarks/ranking_$DATASET.csv}"
MODEL="${MODEL:-bd_shallow}"
K_LIST="${K_LIST:-5 10 20}"
CANDIDATES="${CANDIDATES:-40}"
REPS="${REPS:-8}"
FOLDS="${FOLDS:-4}"
NS="${N_SHARDS:-1}"
SH="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"

python -c "import eegrow, sys; p=eegrow.__file__; \
  sys.exit(0) if p.startswith('$ROOT/src') else sys.exit('FATAL: eegrow imported from '+p)"

set -x
python "$ROOT/benchmarks/donor_select.py" \
  --dataset "$DATASET" --model "$MODEL" --ranking "$RANKING" \
  --out "$OUT" --cache "$CACHE" \
  --k $K_LIST --candidates "$CANDIDATES" --reps "$REPS" --folds "$FOLDS" \
  ${THREADS:+--threads $THREADS} \
  --n-shards "$NS" --shard "$SH"
