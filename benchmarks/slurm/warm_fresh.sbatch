#!/bin/bash
#SBATCH --job-name=eegrow_warmfresh
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/amounir/logs/warmfresh_%j.log
#
# LE CACHE DES DEUX DATASETS FRAIS -- dreyer2023a (60 sujets) et liu2024 (50).
#
# POURQUOI C'EST LA PREMIÈRE CHOSE LANCÉE, ET SUR CPU. Rien dans ce job ne touche un
# GPU : c'est du téléchargement et du décodage mono-thread. C'est aussi le chemin
# critique le plus long de tout le programme -- des heures d'E/S avant qu'une seule
# cellule puisse tourner -- donc il part pendant que les campagnes GPU occupent les
# cartes, au lieu d'attendre son tour derrière elles.
#
# CE QUE CES DEUX DATASETS SERVENT À FAIRE, ET CE QU'ILS NE FONT PAS. `restored_epoch`
# (étage C) et la lecture « la taille » de claim 2 ont été formulés SUR les cinq
# datasets déjà mesurés. Les tester à nouveau là-bas ne peut rien prouver : le jardin
# aux sentiers qui bifurquent est déjà parcouru. Ici les chiffres n'existent pas encore,
# donc une prédiction ÉCRITE AVANT le premier fit est falsifiable au sens strict. Ce job
# ne teste rien lui-même -- il rend le test possible.
#
# SÉRIEL ET PAR SUJET, comme warm_cache.sbatch, et pour les mêmes raisons : un seul
# écrivain ne peut pas courser un autre, et le chemin d'erreur de MOABB
# (`interface.erase()` -> rmtree) détruit l'entrée PARTAGÉE quand deux tenants
# collisionnent. 32 cellules et 19 sessions physionetmi ont déjà été perdues ainsi.
#
# LE CHECKOUT EST eegrow_budget ET NON eegrow. `warm_cache.py` importe
# `check_cache.build_campaign` pour construire le paradigme, et les configs des deux
# datasets frais n'existent que dans l'arbre déployé ici. Pointer l'autre arbre
# échouerait sur un fichier manquant -- ou pire, réussirait avec un paradigme différent
# de celui que la campagne lira.
#
# LA PLACE DISQUE EST LA SEULE VRAIE INQUIÉTUDE : /scratch est à 91 % (5,0 To libres) et
# le cache MOABB pèse déjà 79 Go pour douze datasets. On imprime `du` avant et après
# plutôt que de le supposer -- un cache à moitié écrit par manque de place est
# exactement l'état qui déclenche le chemin destructeur ci-dessus.
#
#   sbatch benchmarks/slurm/warm_fresh.sbatch
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench

export ROOT=/scratch/amounir/eegrow_budget
cd "$ROOT/benchmarks"
export PYTHONPATH="$ROOT/src:$ROOT/benchmarks"
export EEGROW_BENCH_ROOT="$ROOT/benchmarks"
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== node=$(hostname) start=$(date -Is)"
echo "=== place disque AVANT"
df -h /scratch | tail -1
du -sh /scratch/amounir/moabb_cache /scratch/amounir/mne_data 2>/dev/null

python warm_cache.py --cache /scratch/amounir/moabb_cache \
    --datasets dreyer2023a,liu2024

echo "=== place disque APRÈS"
df -h /scratch | tail -1
du -sh /scratch/amounir/moabb_cache /scratch/amounir/mne_data 2>/dev/null

echo "=== contrôle : la porte du cache, sur ces deux datasets seulement"
python check_cache.py --cache /scratch/amounir/moabb_cache \
    --datasets dreyer2023a,liu2024

echo "=== DONE $(date -Is)"
