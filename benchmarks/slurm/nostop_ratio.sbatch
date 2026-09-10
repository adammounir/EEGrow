#!/bin/bash
# BUDGET PLEIN + PLANCHER SINGULIER ABAISSE -- est-ce que le reseau apprend mieux si
# on ne l'arrete jamais et qu'on laisse passer tous les neurones candidats ?
#
# CE QUE CE PILOTE TESTE, ET CE QU'IL NE TESTE PAS. A LIRE AVANT D'INTERPRETER.
# ----------------------------------------------------------------------------
# Endpoint : accuracy ABSOLUE du meme bras growing, plancher 0.1 contre plancher 1e-6,
# apparie par (bras, graine, alignement, sujet, session). n = 9 sujets, 2 graines,
# 2 bras. C'est un contraste INTRA-BRAS : `grow - fix` n'est PAS teste ici et ne doit
# pas etre lu dans ces resultats -- son MDE sur bnci2014_001 est de +0.0486 (cf.
# deep_bottleneck.sbatch) et rien de ce qui suit ne le franchit.
#
# LES DEUX BOUTONS, ET POURQUOI CE SONT CEUX-LA
# ----------------------------------------------
# (1) SANS EARLY STOPPING. `train.patience >= train.max_epochs` rend l'arret anticipe
#     inatteignable : les 200 epoques sont depensees quoi qu'il arrive. On passe 10000
#     plutot que 200 pour que la condition reste vraie meme si `max_epochs` bouge.
#     `selection_monitor=valid_acc` RESTE : restaurer la meilleure epoque n'est pas de
#     l'early stopping, c'est de la selection de modele, et la retirer re-daterait
#     cette mesure par rapport a toute la campagne (cf. config.yaml, carre 2x2).
#
# (2) PLANCHER SINGULIER A 1e-6. ATTENTION AUX UNITES -- il n'existe AUCUN seuil
#     absolu a 1e-5 dans ce code. Il y a trois seuils et un seul est vivant :
#       * gromo `statistical_threshold` (absolu, defaut 1e-3) -- deja mis a 0.0 par
#         `loop.grow_step`, donc DESACTIVE. Le baisser ne ferait rien.
#       * gromo `numerical_threshold` (absolu, 1e-6) -- garde le pseudo-inverse, pas
#         la selection. Non expose, non touche.
#       * `MIN_SINGULAR_RATIO = 0.1` (RELATIF) -- le seul qui coupe encore. C'est lui
#         qu'on abaisse. 1e-6 en unites relatives revient a "garder tout candidat
#         numeriquement valide", ce que la docstring de `grow_step` decrit comme le
#         comportement de la borne 0.0.
#
# LE PRIOR EST CONTRE, ET C'EST POUR CA QUE LA CELLULE EXISTE
# ------------------------------------------------------------
# `grow_shallow.yaml` porte une sonde a 1 sujet / 1 graine : ratio 0.10 -> acc 0.690,
# ratio 0.01 -> acc 0.552. Baisser le plancher fait ATTEINDRE la cible de largeur plus
# vite (8->32 au lieu de 8->23) et scorer moins bien. n=1, donc rien n'est tranche --
# mais l'hypothese "il apprendra mieux" part avec une sonde contre elle, et si les 8
# cellules la confirment, c'est ce resultat-la qu'on ecrit, pas l'inverse.
#
# POURQUOI UN results_dir PAR RATIO
# ----------------------------------
# `stem = cell_stem(label, align_tag, seed)` (run_moabb_hydra.py:206) n'inclut PAS le
# ratio. Les deux bras d'une paire sont le meme (label, align, seed) : dans un
# repertoire commun le second ecraserait le premier -- CSV, joblib ET fits.jsonl --
# silencieusement. Un sous-repertoire par ratio est la seule separation qui tienne.
#SBATCH --job-name=nostop-ratio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/amounir/eegrow_budget/ns_logs/ns_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/amounir/eegrow_budget

# Le checkout cluster est une COPIE, pas un clone : `git rev-parse` y echoue, donc
# tree_sha() rend None et chaque ligne porterait eegrow_sha=None. deploy_final.sh ecrit
# le stamp, on le relit ici -- meme mecanique qu'a pack_run.sh:89-90.
if [ -z "${EEGROW_SHA:-}" ] && [ -r "$ROOT/.eegrow_sha" ]; then
  export EEGROW_SHA="$(tr -d '[:space:]' < "$ROOT/.eegrow_sha")"
fi

# Les commentaires du fichier de points sont filtres AVANT l'indexation : `sed -n Np`
# sur le fichier brut ferait pointer la tache k sur une ligne de commentaire, et
# `read` rendrait des variables vides -- une cellule qui tourne sur le mauvais point
# sans jamais echouer.
POINTS=$ROOT/benchmarks/slurm/nostop_ratio.txt
LINE=$(grep -v '^[[:space:]]*#' "$POINTS" | grep -v '^[[:space:]]*$' \
       | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r EVAL DS MODEL SEED ALIGN RATIO RTAG <<< "$LINE"
if [ -z "${RTAG:-}" ]; then
  echo "FATAL: ligne de points incomplete pour la tache ${SLURM_ARRAY_TASK_ID}: '$LINE'"
  exit 92
fi
echo "[$(date -Is)] ns task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL ds=$DS model=$MODEL seed=$SEED align=$ALIGN ratio=$RATIO"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
# $ROOT/src EN PREMIER. L'env `bench` porte un editable .pth qui pointe en dur sur
# /scratch/amounir/eegrow/src : sans cette ligne `import eegrow` resout vers l'AUTRE
# checkout. C'est ce qui a tue les 36 cellules de 578248.
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=$ROOT/moabb_results_ns

# La carte fantome : margpu021 annoncait gpu:turing:3 et repondait "No devices were
# found" sans echouer. On sonde `mem_get_info` -- deduire l'ordinal de nvidia-smi est
# precisement ce qui avait fait echouer 47 cellules.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PROBE' || { echo "GPU PROBE FAILED -- abandon de la tache"; exit 90; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() est False"); sys.exit(1)
free, total = torch.cuda.mem_get_info(0)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"PROBE ok: {name} sm_{cap[0]}{cap[1]} free={free/2**30:.1f}G/{total/2**30:.1f}G")
if cap[0] * 10 + cap[1] < 75:
    print("carte sous sm_75 : torch ne peut pas y tourner"); sys.exit(1)
PROBE

# PROVENANCE, ASSERTEE PLUTOT QU'ARRANGEE -- copie de deep_bottleneck.sbatch:117-137.
python - "$ROOT" <<'PY' || { echo "GUARD FAILED -- abandon"; exit 91; }
import inspect, sys
import eegrow
from eegrow.training import loop
root = sys.argv[1].rstrip("/") + "/"
if not eegrow.__file__.startswith(root):
    sys.exit(f"FATAL: ROOT est {root} mais eegrow importe depuis {eegrow.__file__}")
if "best_s == 0.0" not in inspect.getsource(loop.grow_step):
    sys.exit(f"FATAL: {loop.__file__} est anterieur a 5337c56 (abstention s=0 absente)")
sys.path.insert(0, root + "benchmarks")
from utils import tree_sha  # noqa: E402
sha = tree_sha()
if not sha:
    sys.exit("FATAL: ce tree ne sait pas nommer son commit -- eegrow_sha=None partout")
print(f"guard ok  eegrow={eegrow.__file__}  abstention s=0 presente  sha={sha[:12]}")
PY

cd $ROOT/benchmarks

# GARDE-FOU SUR LES DEUX OVERRIDES EUX-MEMES. `min_singular_ratio` n'est declare dans
# AUCUN model/*.yaml : il passe par `+model.` (ajout de cle), et `pipelines.py` le lit
# avec un `.get(..., MIN_SINGULAR_RATIO)` -- donc un override mal ecrit ne leve rien,
# il retombe sur 0.1 et les deux bras de chaque paire deviennent le MEME bras. Le
# resultat serait un nul parfait, bien forme, et faux. On resout la config en mode
# `--cfg job` (aucun entrainement) et on assert sur la valeur resolue : un garde-fou
# refuse, il n'avertit pas.
python - "$EVAL" "$DS" "$MODEL" "$SEED" "$ALIGN" "$RATIO" <<'PY' || { echo "OVERRIDE GUARD FAILED -- abandon"; exit 93; }
import subprocess, sys
ev, ds, model, seed, align, ratio = sys.argv[1:7]
out = subprocess.run(
    [sys.executable, "run_moabb_hydra.py", f"eval={ev}", f"dataset={ds}",
     f"model={model}", f"seed={seed}", f"align={align}",
     f"+model.min_singular_ratio={ratio}",
     "train.patience=10000", "train.selection_monitor=valid_acc",
     "--cfg", "job", "--resolve"],
    capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(f"FATAL: hydra refuse la config\n{out.stderr[-2000:]}")
import re
cfg = out.stdout
m = re.search(r"min_singular_ratio:\s*(\S+)", cfg)
if not m:
    sys.exit("FATAL: min_singular_ratio absent de la config resolue -- "
             "l'override n'a pas atterri dans cfg.model")
got, want = float(m.group(1)), float(ratio)
if got != want:
    sys.exit(f"FATAL: min_singular_ratio resolu a {got}, demande {want}")
p = re.search(r"patience:\s*(\S+)", cfg)
if not p or int(p.group(1)) < 200:
    sys.exit(f"FATAL: patience resolue a {p and p.group(1)} -- l'early stopping "
             "peut encore se declencher")
print(f"override guard ok  min_singular_ratio={got}  patience={p.group(1)}")
PY

# patience=10000 >= max_epochs=200 -> l'early stopping ne peut jamais se declencher.
# selection_monitor=valid_acc reste (selection de modele, pas arret anticipe).
# results_dir par ratio : voir l'en-tete, `stem` ne porte pas le ratio.
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" align="$ALIGN" \
  +model.min_singular_ratio="$RATIO" \
  train.patience=10000 train.selection_monitor=valid_acc \
  dataset.resample=250 \
  results_dir="$ROOT/benchmarks/results_nostop/$RTAG" \
  overwrite=true suffix="ns_${MODEL}_${ALIGN}_s${SEED}_${RTAG}"
