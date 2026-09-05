#!/bin/bash
#SBATCH --job-name=stieger_pull
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/amounir/logs/stieger_pull_%A_%a.log
#
# TIRE stieger2021 EN PARALLELE — additif, ne touche a rien de ce qui tourne.
#
# POURQUOI. Mesure du 05/09 16h35 : stieger2021 = 598 sessions de ~615 Mo, soit
# ~360 Go, telechargees UNE PAR UNE par la tache 539160_0. Debit observe 3,3 Mo/s
# => ~30 h pour ce seul dataset, quand les dix autres finissent en ~3 h. C'est le
# pole serie de toute la campagne, et il est purement reseau.
#
# POURQUOI C'EST SUR. `Stieger2021.data_path(subject)` teste `os.path.exists(fpath)`
# avant chaque telechargement (stieger2021.py:417) : un fichier deja present est
# saute. Les taches ci-dessous prennent des sujets DISJOINTS de ceux que 539160_0
# est en train de parcourir (il monte depuis S1, elles descendent depuis S62), donc
# elles ecrivent des fichiers disjoints. pooch ecrit dans un tmp aleatoire puis
# renomme : meme une collision serait atomique.
#
# CE QUE CE JOB NE FAIT PAS : aucun epoching, aucune ecriture dans le cache MOABB
# traite. C'est le chemin qui detruit une entree partagee quand deux ecrivains se
# croisent (incident des 276 sujets). On ne fait que remplir MNE-Stieger2021-data ;
# 539160_0 gardera l'exclusivite de l'etape de cache, et sa dependance aftercorr
# vers 539161_0 reste intacte parce qu'on ne le touche pas.
set -uo pipefail
source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export MNE_DATA=/scratch/amounir/mne_data
export MPLCONFIGDIR=/scratch/amounir/.mplconfig

python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys, time
from moabb.datasets import Stieger2021

j, n = int(sys.argv[1]), 8
# Sujets 8..62 : on laisse 1..7 a 539160_0, qui les a deja entames.
pool = list(range(62, 7, -1))
mine = pool[j::n]
ds = Stieger2021()
print(f"tache {j} : {len(mine)} sujets {mine}", flush=True)
for s in mine:
    t = time.time()
    try:
        p = ds.data_path(s)
        print(f"OK sujet {s} : {len(p)} fichiers en {time.time()-t:.0f}s", flush=True)
    except Exception as e:  # un sujet qui echoue ne doit pas tuer les suivants
        print(f"ECHEC sujet {s} : {type(e).__name__} {e}", flush=True)
print(f"tache {j} terminee", flush=True)
PY
