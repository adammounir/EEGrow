#!/bin/bash
# grow_deep CONTRE bd_deep4, A BUDGET PLEIN. La comparaison titre du bras deep, et
# elle n'a jamais tourne sous le protocole corrige.
#
# CE QUE LE CHIFFRE PUBLIE DIT, ET POURQUOI IL EST FAUX
# ------------------------------------------------------
# `results_published/eegrow_benchmark_levels.csv`, bnci2014_001 :
#
#     eval             bd_deep4   grow_deep   delta
#     within_session     0.275      0.408     +0.133
#     cross_session      0.303      0.419     +0.116
#
# 0.275 est SOUS le niveau de chance : 288 essais, 4 classes, seuil exact unilateral
# 95 % = 0.295. Le "gain de la croissance" y est la difference entre un reseau qui
# apprend et un reseau qui n'a pas eu le temps d'apprendre -- Deep4 y etait scoré a
# l'epoque 4, sa perte de validation touchant le fond avant que l'accuracy ne demarre.
# Sur les 30 cellules du workbook, grow_deep - bd_deep4 vaut -0.024 en moyenne et
# grow_deep ne gagne que 7 fois, dont ces deux-la.
#
# Le carre budget x selection (exp_deep4_budget.py, puis SLURM 500573) mesure ce que
# vaut le protocole sur ce meme bras :
#
#                       sel=valid_loss   sel=valid_acc
#     patience  20          0.3501           0.3390
#     patience 200          0.4074           0.4818
#
# bd_deep4 passe de 0.339 a 0.4818 rien qu'en le laissant s'entrainer. Si grow_deep
# reste autour de 0.42 (valeur mesuree a budget plein par 578293), le +0.133 publie
# ne change pas d'ampleur : il CHANGE DE SIGNE. C'est ce que cet array va trancher.
#
# CE QUI EST TESTE ET AVEC QUELLE PUISSANCE
# ------------------------------------------
# n = 9 sujets, 3 graines, 2 alignements, apparie sujet par sujet. L'unite d'analyse
# est le SUJET, pas la ligne : les sessions et les graines d'un meme sujet sont
# correlees et les traiter comme independantes gonfle les p. Effet attendu ici : de
# l'ordre de -0.06, tres au-dessus du MDE de +0.0486 de cette cellule -- contrairement
# a `grow - fix`, que cet array NE PEUT PAS trancher et qu'il ne pretend pas trancher.
#
# POURQUOI `min_singular_ratio` N'EST PAS UN AXE ICI
# ---------------------------------------------------
# Sur `grow_deep` (w2=8) le plancher singulier est INERTE PAR CONSTRUCTION :
# n_proposed = fan-out = 8 pour 8 places, et sur les 1 620 evenements de croissance de
# 578293 il ne coupe rien dans 100.0 % des pas. Le baisser produirait deux bras
# bit-identiques. L'axe ratio vit dans 582215, sur les deux bras ou il mord.
#
# TURING, PAS AMPERE
# -------------------
# margpu020 est idle (4 turing) pendant que la partition `gpu` est saturee. Les 18
# cellules tiennent dessus, donc aucune paire ne traverse de frontiere de classe de
# carte -- la contrainte que deep_bottleneck.sbatch documente. Ne PAS apparier ces
# resultats avec results_deepbottleneck/, qui a tourne sur ampere.
#
# CONCURRENCE : %3, PAS %6. LEÇON DU 10/09
# ----------------------------------------
# La premiere soumission (582231) tournait en %6 sur un noeud qui n'a que 4 turing et
# que d'autres utilisateurs partagent. 12 taches sur 18 ont atterri sans device
# utilisable : `nvidia-smi` listait bien une 2080 Ti mais `torch.cuda.is_available()`
# rendait False. Elles sont mortes en 4 secondes sur le probe (exit 90) -- le garde-fou
# a refuse au lieu de laisser tourner des cellules contre une carte absente, ce qui est
# exactement le mode de defaillance de margpu021 (47 cellules perdues). On plafonne a 3
# pour laisser une carte aux autres jobs du noeud.
#SBATCH --job-name=grow-vs-bd
#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-17%6
#SBATCH --output=/scratch/amounir/eegrow_budget/gvb_logs/gvb_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/amounir/eegrow_budget

# Le checkout cluster est une COPIE, pas un clone : `git rev-parse` y echoue, donc
# tree_sha() rend None et chaque ligne porterait eegrow_sha=None -- la perte de
# tracabilite qui a coute les 1170 cellules v5. deploy_final.sh ecrit le stamp.
if [ -z "${EEGROW_SHA:-}" ] && [ -r "$ROOT/.eegrow_sha" ]; then
  export EEGROW_SHA="$(tr -d '[:space:]' < "$ROOT/.eegrow_sha")"
fi

# Commentaires filtres AVANT l'indexation : `sed -n Np` sur le fichier brut ferait
# pointer la tache k sur une ligne de commentaire et `read` rendrait des variables
# vides -- une cellule qui tourne sur le mauvais point sans jamais echouer.
# POINTS est surchargeable par l'environnement (`sbatch --export=ALL,POINTS=...`)
# pour rejouer un sous-ensemble de cellules sans dupliquer ce template. Le fichier
# d'origine reste le proces-verbal de ce qui a ete soumis le 10/09 ; on ne l'edite
# PAS pour retirer des lignes, ca decalerait l'indexation de toute tache en file.
POINTS=${POINTS:-$ROOT/benchmarks/slurm/grow_vs_bd.txt}
LINE=$(grep -v '^[[:space:]]*#' "$POINTS" | grep -v '^[[:space:]]*$' \
       | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r EVAL DS MODEL SEED ALIGN <<< "$LINE"
if [ -z "${ALIGN:-}" ]; then
  echo "FATAL: ligne de points incomplete pour la tache ${SLURM_ARRAY_TASK_ID}: '$LINE'"
  exit 92
fi
echo "[$(date -Is)] gvb task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL ds=$DS model=$MODEL seed=$SEED align=$ALIGN"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
# $ROOT/src EN PREMIER. L'env `bench` porte un editable .pth qui pointe en dur sur
# /scratch/amounir/eegrow/src : sans cette ligne `import eegrow` resout vers l'AUTRE
# checkout. C'est ce qui a tue les 36 cellules de 578248.
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=$ROOT/moabb_results_gvb

# La carte fantome : margpu021 annoncait gpu:turing:3 et repondait "No devices were
# found" sans echouer, donc chaque cellule tournait contre un device absent et rendait
# des CSV bien formes. On sonde `mem_get_info` -- deduire l'ordinal de nvidia-smi est
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

# GARDE-FOU SUR LE PROTOCOLE LUI-MEME. C'est TOUT l'objet de cet array : le chiffre
# publie est faux parce que le budget etait coupe, donc une cellule qui repartirait
# sur les defauts (patience=20, selection_monitor=valid_loss) reproduirait exactement
# l'artefact qu'on vient mesurer -- avec des CSV bien formes et aucun avertissement.
# On resout la config sans entrainer et on assert. Un garde-fou refuse, il n'avertit pas.
python - "$EVAL" "$DS" "$MODEL" "$SEED" "$ALIGN" <<'PY' || { echo "PROTOCOL GUARD FAILED -- abandon"; exit 93; }
import re, subprocess, sys
ev, ds, model, seed, align = sys.argv[1:6]
out = subprocess.run(
    [sys.executable, "run_moabb_hydra.py", f"eval={ev}", f"dataset={ds}",
     f"model={model}", f"seed={seed}", f"align={align}",
     "train.patience=10000", "train.selection_monitor=valid_acc",
     "--cfg", "job", "--resolve"],
    capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(f"FATAL: hydra refuse la config\n{out.stderr[-2000:]}")
cfg = out.stdout
pat = re.search(r"patience:\s*(\S+)", cfg)
mx = re.search(r"max_epochs:\s*(\S+)", cfg)
sel = re.search(r"selection_monitor:\s*(\S+)", cfg)
if not (pat and mx and sel):
    sys.exit("FATAL: patience / max_epochs / selection_monitor absents de la config resolue")
if int(pat.group(1)) < int(mx.group(1)):
    sys.exit(f"FATAL: patience={pat.group(1)} < max_epochs={mx.group(1)} -- "
             "l'early stopping peut se declencher, c'est l'artefact qu'on mesure")
if sel.group(1) != "valid_acc":
    sys.exit(f"FATAL: selection_monitor={sel.group(1)}, attendu valid_acc")
print(f"protocol guard ok  patience={pat.group(1)} >= max_epochs={mx.group(1)}  "
      f"selection_monitor={sel.group(1)}")
PY

# patience=10000 >= max_epochs=200 -> l'early stopping ne peut jamais se declencher,
# les 200 epoques sont depensees quoi qu'il arrive. selection_monitor=valid_acc reste :
# restaurer la meilleure epoque est de la selection de modele, pas de l'arret anticipe,
# et la retirer re-daterait cette mesure par rapport a toute la campagne.
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" align="$ALIGN" \
  train.patience=10000 train.selection_monitor=valid_acc \
  dataset.resample=250 \
  results_dir="$ROOT/benchmarks/results_growvsbd" \
  overwrite=true suffix="gvb_${MODEL}_${ALIGN}_s${SEED}"
