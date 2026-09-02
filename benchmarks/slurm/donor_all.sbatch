#!/bin/bash
#SBATCH --job-name=eegrow_dall
#SBATCH --partition=tau,gpu-best
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu020,margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/amounir/logs/dall_%A_%a.log
#
# LE BRAS « TOUT LE MONDE DEDANS », demandé par S. Chevallier le 01/09 et absent du
# protocole jusqu'ici. Statut : POST-HOC ET DESCRIPTIF -- lire
# analysis/PREREG_ADDENDUM_all_pool.md AVANT d'en rapporter le moindre chiffre. Il ne
# fait partie d'aucune famille pré-enregistrée et ne peut pas renverser claim 3.
#
# UN FICHIER SÉPARÉ DE donor_select2.sbatch, ET C'EST VOULU. Les deux appellent le même
# donor_select.py, mais avec des dimensionnements incompatibles : --mem=24G au lieu de
# 16G et un temps par cellule 4x supérieur, parce que le pool d'entraînement passe de
# K sujets à |vivier|. Les glisser dans le même en-tête paramétré aurait donné un
# --mem qui convient à un mode et pas à l'autre.
#
# LE CHIFFRE QUI DIMENSIONNE, mesuré au --dry-run et non extrapolé :
#
#     cho2017      vivier 39 sujets x 200 essais = 7 800 essais/fit  (K=10 -> 2 000)
#     physionetmi  vivier 82 sujets x  45 essais = 3 690 essais/fit  (K=20 ->   897)
#
# soit ~4x la plus grosse cellule de règle dans les deux cas. 32 cellules par dataset
# (4 plis x 8 seeds, pas de réplicat : le pool est déterministe étant donné le pli),
# donc 8 cellules par shard sur 4 shards -- largement sous les 3 h même à 4x.
#
# --mem=24G ET NON 16G : le fit tient tout le pool en mémoire (`pool_xy` concatène), et
# 7 800 essais de cho2017 sont ~4x ce qui a été mesuré à 16 G. On demande la marge
# plutôt que de découvrir un OOM à la 30e cellule.
#
# PIÈGE À NE PAS REPRODUIRE, si ce bras devait être rejoué à un autre K : `run_cell`
# nomme la cellule `f{pli}__all_pool__d0__seed{seed}` SANS le K. Deux tailles de pool
# écriraient donc le même chemin et la reprise sauterait la seconde en croyant l'avoir
# faite. Le bras n'a qu'une taille par construction, mais si ça change, changer le nom.
#
# ANALYSE : ne PAS pointer donor_select2_analysis.py ici. Son glob attend le format v2
# et son unité de rééchantillonnage est le réplicat, qui n'existe pas dans ce bras.
#
#     DATASET=cho2017     N_SHARDS=4 sbatch --array=0-3 slurm/donor_all.sbatch
#     DATASET=physionetmi N_SHARDS=4 sbatch --array=0-3 slurm/donor_all.sbatch
#
# margpu020 AJOUTÉ À L'EXCLUSION LE 02/09, MESURÉ ET NON DÉDUIT. `515639_2` y est mort en
# 12 s sur `CUDA_VISIBLE_DEVICES='0' but torch.cuda.is_available() is False`, pendant que
# les 6 autres shards du même array tournaient ailleurs. `scontrol show node margpu020`
# annonce Gres=gpu:turing:4 avec AllocTRES gres/gpu=3 : SLURM nous a donné une 4e carte
# qui ne répond pas, exactement le motif de margpu021 (qui annonce 3 pour 2 réelles).
# La distinction avec [[cuda-ordinal-vs-nvidia-smi]] compte : là-bas la carte fantôme
# avait été DÉDUITE de nvidia-smi et la déduction était fausse (47 cellules perdues) ;
# ici c'est le process qui est mort dessus. Le garde-fou `pick_device` a refusé le CPU,
# donc la perte est propre : aucune cellule calculée sur le mauvais matériel.
#
# Le reste (exclusion des pascal sm_60, margpu021 et sa carte fantôme, tau,gpu-best pour
# la priorité, --requeue avec reprise par CSV existant) est justifié dans
# donor_select2.sbatch et n'a pas changé.
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
OUT="${OUT:-/scratch/amounir/dsel_all/$DATASET}"
CACHE="${CACHE:-/scratch/amounir/moabb_cache}"
RANKING="${RANKING:-$ROOT/benchmarks/ranking_$DATASET.csv}"
MODEL="${MODEL:-bd_shallow}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
FOLDS="${FOLDS:-4}"
NS="${N_SHARDS:-1}"
SH="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"

python -c "import eegrow, sys; p=eegrow.__file__; \
  sys.exit(0) if p.startswith('$ROOT/src') else sys.exit('FATAL: eegrow imported from '+p)"

set -x
python "$ROOT/benchmarks/donor_select.py" \
  --dataset "$DATASET" --model "$MODEL" --ranking "$RANKING" \
  --out "$OUT" --cache "$CACHE" \
  --all-pool --all-seeds $SEEDS --folds "$FOLDS" \
  ${THREADS:+--threads $THREADS} \
  --n-shards "$NS" --shard "$SH"
