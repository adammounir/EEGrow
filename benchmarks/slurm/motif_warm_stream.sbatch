#!/bin/bash
#SBATCH --job-name=motif_stream
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/amounir/logs/motif_stream_%A_%a.log
#
# CHAUFFE EN FLUX DES DEUX DATASETS QUI NE RENTRENT PAS DANS LE QUOTA.
#
# stieger2021 demande ~345 Go de brut ; le quota /scratch est de 1 To par utilisateur et
# l'arbre pesait deja 878 Go sans lui. Le dataset ne rentre pas, et aucun parallelisme
# n'y change rien -- c'est ce qui a tue 7 des 11 chauffes le 05/09, sur
# `OSError: [Errno 122] Disk quota exceeded`, pendant que `df` affichait 4,2 To libres.
#
# `warm_stream.py` efface le brut de chaque sujet des que son cache est confirme par la
# MEME fonction que la porte, avec un sujet de retard pour qu'aucun fichier reclame par
# un sujet ulterieur ne parte. L'empreinte de pointe tombe de 345 Go a deux sujets.
#
# yang2025 est livre en ARCHIVE UNIQUE de 65,6 Go : la suppression au fil de l'eau y est
# impossible par construction, et le script la refuse tout seul (un fichier reclame par
# 51 sujets n'est jamais efface). Il tient dans le quota une fois les 235 Go liberes, et
# `--purge-raw-at-end` rend l'archive apres coup, sous reserve d'un check_cache vert.
#
#   sbatch --array=0 benchmarks/slurm/motif_warm_stream.sbatch   # stieger2021
#   sbatch --array=1 benchmarks/slurm/motif_warm_stream.sbatch   # yang2025
set -uo pipefail

LIST="${LIST:-/scratch/amounir/eegrow_budget/benchmarks/slurm/motif_datasets.txt}"
DS=$(grep -v '^[[:space:]]*#' "$LIST" | awk 'NF {print $1}' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$DS" ]; then
  echo "FATAL: pas de dataset a l'indice $SLURM_ARRAY_TASK_ID dans $LIST" >&2
  exit 2
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

echo "=== $DS  node=$(hostname)  start=$(date -Is)"
python warm_stream.py --dataset "$DS" --cache /scratch/amounir/moabb_cache ${EXTRA:-}
rc=$?

echo "=== $DS  porte du cache (check_cache, ce dataset seulement)"
python check_cache.py --cache /scratch/amounir/moabb_cache --datasets "$DS"
gate=$?

echo "=== $DS  DONE $(date -Is) stream_rc=$rc gate_rc=$gate"
# Comme motif_warm.sbatch : c'est la PORTE qui porte le code de sortie, parce que c'est
# elle que `aftercorr` lit pour liberer les fits.
exit $gate
