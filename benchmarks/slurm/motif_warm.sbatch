#!/bin/bash
#SBATCH --job-name=motif_warm
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-10%6
#SBATCH --output=/scratch/amounir/logs/motif_warm_%A_%a.log
#
# LE CACHE DES 11 DATASETS AJOUTÉS — un dataset par tâche de tableau.
#
# POURQUOI C'EST PARALLÉLISABLE ALORS QUE `warm_cache.sbatch` EST SÉRIEL. Le sériel de
# l'autre script protège d'une chose précise : deux écrivains sur la MÊME entrée de
# cache MOABB se collisionnent, et le chemin d'erreur de MOABB (`interface.erase()` ->
# `rmtree`) détruit alors l'entrée PARTAGÉE. 32 cellules et 19 sessions physionetmi ont
# été perdues comme ça. Ici les 11 tâches écrivent sous 11 `code` de dataset disjoints
# et sous des digests disjoints : aucune paire de tâches ne peut viser le même chemin.
# La contrainte est donc respectée, pas contournée — un dataset reste chauffé par un
# seul processus, sujet après sujet.
#
# `%6` ET PAS PLUS. La limite n'est pas le CPU (23 nœuds `normal` sont idle) mais le
# disque et le réseau : /scratch est à 91 % et ces datasets n'ont jamais été téléchargés
# ici, donc leur taille est inconnue. Six téléchargements concurrents laissent une marge
# pour voir venir ; onze la suppriment.
#
# LA PORTE DISQUE REFUSE, ELLE N'AVERTIT PAS. Un cache écrit à moitié par manque de
# place est exactement l'état qui déclenche le chemin destructeur ci-dessus : un
# lockfile présent, des données absentes, et tout lecteur qui rapporte un hit, échoue,
# puis efface. Une tâche qui démarre sous le plancher sort en 3 sans rien écrire.
#
# CE JOB NE TOUCHE AUCUN GPU. C'est du téléchargement et du décodage mono-thread, et
# c'est le vrai chemin critique de la campagne : des heures d'E/S avant qu'un seul fit
# soit possible. Il part donc en premier, pendant que les cartes sont occupées ailleurs.
#
#   sbatch benchmarks/slurm/motif_warm.sbatch
set -uo pipefail

# Plancher d'espace libre sur /scratch, en Go. En dessous, la tâche refuse de démarrer.
FLOOR_GB="${FLOOR_GB:-800}"

LIST="${LIST:-/scratch/amounir/eegrow_budget/benchmarks/slurm/motif_datasets.txt}"
# Premier champ des lignes non commentées : le nom du dataset. Le reste de la ligne est
# un commentaire aligné, il ne doit jamais atteindre `--datasets`.
DS=$(grep -v '^[[:space:]]*#' "$LIST" | awk 'NF {print $1}' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$DS" ]; then
  echo "FATAL: pas de dataset a l'indice $SLURM_ARRAY_TASK_ID dans $LIST" >&2
  exit 2
fi

FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc '0-9')
echo "=== $DS  node=$(hostname)  start=$(date -Is)  libre=${FREE_GB}G  plancher=${FLOOR_GB}G"
if [ "$FREE_GB" -lt "$FLOOR_GB" ]; then
  echo "REFUS: ${FREE_GB}G libres < plancher ${FLOOR_GB}G. Un cache ecrit a moitie par" >&2
  echo "manque de place declenche le chemin destructeur de MOABB. Rien n'a ete ecrit." >&2
  exit 3
fi

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench

export ROOT=/scratch/amounir/eegrow_budget
cd "$ROOT/benchmarks"
export PYTHONPATH="$ROOT/src:$ROOT/benchmarks"
export EEGROW_BENCH_ROOT="$ROOT/benchmarks"
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python warm_cache.py --cache /scratch/amounir/moabb_cache --datasets "$DS"
rc=$?

echo "=== $DS  place APRES"
df -h /scratch | tail -1

echo "=== $DS  porte du cache (check_cache, ce dataset seulement)"
python check_cache.py --cache /scratch/amounir/moabb_cache --datasets "$DS"
gate=$?

echo "=== $DS  DONE $(date -Is) warm_rc=$rc gate_rc=$gate"
# Le code de sortie porte la PORTE, pas le warm : c'est lui que `aftercorr` lit pour
# decider si le job GPU de ce dataset a le droit de partir. Un cache incomplet doit
# empecher les fits, pas les laisser echouer un par un dans les logs.
exit $gate
