#!/bin/bash
#SBATCH --job-name=motif_dxr
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-10
#SBATCH --exclude=margpu001,margpu004,margpu020,margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029,margpu030,margpu031,margpu032
#SBATCH --output=/scratch/amounir/logs/motif_dxr_%A_%a.log
#
# LA MATRICE DONNEUR×RECEVEUR DES 11 DATASETS AJOUTÉS — un dataset par tâche.
#
# SUR GPU, ET C'EST UN FACTEUR 25. La matrice de cho2017 a tourné sur des nœuds CPU
# (`donor_receiver.sbatch`, mode array sur marg[003-012]) : 1 140 s par fit donneur,
# mesuré sur ses CSV. Le même réseau en sonde within_session sur le même dataset coûte
# 40 s sur turing. Rien dans le protocole n'impose le CPU — c'était un choix de
# cohabitation, à un moment où les cartes étaient prises. Elles ne le sont plus.
#
# CE JOB EST AUSSI LA SONDE DE COÛT, ET C'EST POUR ÇA QU'IL PASSE AVANT LA SONDE.
# Le projet a déjà transposé trois fois de suite un coût par cellule et s'est trompé
# les trois fois — une fois d'un ordre de grandeur, une fois d'un facteur 7. La seule
# mesure fiable est prise sur le dataset lui-même. Or un fit donneur voit **100 %** des
# essais d'un sujet, là où un fold de sonde en voit 80 % : le `fit_seconds` écrit ici
# MAJORE donc le coût d'un fold de sonde du même dataset. La table de coûts qui
# dimensionnera `motif_probe` sort de ces logs, gratuitement, sur des fits dont on
# avait besoin de toute façon.
#
# APPARIEMENT PAR `aftercorr`. La tâche i part quand la tâche i de `motif_warm` sort
# en 0 — c'est-à-dire quand la porte `check_cache` de CE dataset est verte, pas quand
# les onze téléchargements sont finis. Aucune barrière globale : le makespan n'est pas
# celui du plus lent des téléchargements plus celui du plus lent des calculs, mais le
# max des deux chaînes par dataset. C'est tout l'intérêt d'`aftercorr` sur `afterok`.
#
# EXCLUSIONS. pascal (sm_60) et volta (sm_70) sont inutilisables avec torch cu130 ("no
# kernel image is available") et apparaissent pourtant `idle` dans sinfo ; margpu020 a
# une carte fantôme (l'ordinal CUDA ne suit pas l'index nvidia-smi, mesuré le 02/09) ;
# margpu023 est drained. Ne pas les redemander en voyant `idle`.
#
# REPRISE GRATUITE. Un CSV par (donneur, seed), écrit en dernier : relancer la même
# commande saute ce qui est fait. Une tâche préemptée ou tuée par le mur de temps ne
# laisse qu'un trou que la relance rattrape, donc il n'y a rien à surveiller ici.
#
#   sbatch --dependency=aftercorr:<jobid_warm> benchmarks/slurm/motif_dxr.sbatch
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
# $ROOT/src EN PREMIER : l'env `bench` porte un .pth editable qui pointe sur
# /scratch/amounir/eegrow, l'arbre d'août SANS l'abstention s=0 du 25/08. Sans cette
# ligne le script importerait ce paquet-la, re-ajouterait des neurones morts et
# gonflerait `params_end` -- c'est-a-dire fausserait la quantite mesuree.
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
# 3 co-tenants sur une carte : le profilage (job 463259) donne 61,6 % d'utilisation
# moyenne avec un seul tenant, donc il reste de la place, mais pas dix fois.
export EEGROW_CUDA_FRACTION="${EEGROW_CUDA_FRACTION:-0.30}"

python -c "import eegrow, sys; p=eegrow.__file__; \
  sys.exit(0) if p.startswith('$ROOT/src') else sys.exit('FATAL: eegrow imported from '+p)"

OUT="/scratch/amounir/dxr_motif/$DS"
NS="${N_SHARDS:-3}"
mkdir -p "$OUT"

echo "=== $DS  node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}  shards=$NS  start=$(date -Is)"

pids=()
for i in $(seq 0 $((NS - 1))); do
  python "$ROOT/benchmarks/donor_receiver.py" \
    --dataset "$DS" --out "$OUT" --cache /scratch/amounir/moabb_cache \
    --seeds 0 1 2 --n-shards "$NS" --shard "$i" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

echo "=== $DS  SONDE DE COUT (majorant d'un fold de sonde : 100 % des essais vs 80 %)"
python - "$OUT" "$DS" <<'PY'
import sys, glob, pandas as pd
out, ds = sys.argv[1], sys.argv[2]
fs = [f for f in glob.glob(f"{out}/*.csv")]
if not fs:
    print(f"COST {ds} AUCUN CSV"); raise SystemExit(0)
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["donor", "seed"])
# Une ligne greppable, volontairement : c'est elle qui dimensionne motif_probe.sbatch.
print(f"COST {ds} n_donneurs={d.donor.nunique()} fits={len(d)} "
      f"fit_s_median={d.fit_seconds.median():.1f} fit_s_p90={d.fit_seconds.quantile(.9):.1f} "
      f"device={d.device.iloc[0]}")
PY

echo "=== $DS  DONE $(date -Is) rc=$rc  csv=$(ls "$OUT"/*.csv 2>/dev/null | wc -l)"
exit $rc
