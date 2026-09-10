#!/bin/bash
# PILOTE w1 + REGULARISATION -- ce que le pilote goulot aurait du faire varier.
#
# CE QUI A CHANGE DEPUIS 578293. A LIRE AVANT D'INTERPRETER.
# ---------------------------------------------------------
# 578293 a fait varier w2 (l'AVAL de la jonction) et n'a rien trouve : issue (b), le
# goulot n'etait pas la contrainte. La mesure sur ses 3240 fits dit pourquoi. Sur les
# 12 bras, la train loss finale tombe a 0.001-0.004 sur 230 essais, la MEILLEURE valid
# loss jamais atteinte reste a 0.1-0.2 nats sous ln(4)=1.3863, la valid loss FINALE est
# au-dessus du prior uniforme sur align=none, et 18 a 28% des fits ne battent jamais ce
# prior. Elargir w2 de 8 a 32 (10036 params en fin de croissance contre 6644 au depart)
# divise la train loss finale par 4 et fait passer l'optimum de validation de l'epoque
# 19 a l'epoque 11.
#
# Autrement dit : le bras n'est PAS limite en capacite, il est limite en
# REGULARISATION. La croissance ajoute de la capacite ; la capacite n'est pas la
# contrainte qui mord ; donc la croissance ne peut rien payer. C'est la version
# *mesuree* de l'issue (b), et c'est le mecanisme a ecrire.
#
# LES DEUX FACTEURS, ET POURQUOI CEUX-LA
# --------------------------------------
#   w1 (8 / 16 / 32) -- conv_time 1->w1 puis conv_spat w1->w1 sur les 22 electrodes.
#     La campagne ET le pilote 578293 l'ont tenu a 8 A TOUS LES NIVEAUX, contre 25/25
#     pour le Deep4Net de reference. Le banc de filtres SPATIAUX est donc 3x plus
#     etroit que la reference, et la jonction growable vit EN AVAL de ce collapse. A
#     comparer a growing_shallow, dont la jonction est conv_time -> conv_spat SUR LE
#     SIGNAL BRUT : y croitre ajoute des bandes de frequence, c'est-a-dire de vraies
#     features d'entree (l'axe FBCSP). C'est l'hypothese de mecanisme du papier.
#
#   reg (off / on) -- drop_prob=0.5 + norm_junction. `growing_deepeeg.py` etait la
#     SEULE archi growing sans aucune regularisation : shallow, sccnet et eegnex sont
#     tous a drop_prob=0.5, et le Deep4Net de braindecode porte 3 modules Dropout.
#
# CE QUE CE PILOTE PEUT ET NE PEUT PAS CONCLURE
# ---------------------------------------------
# Comme 578293 : bnci2014_001 porte 9 sujets, le contraste apparie grow - fix y a un
# MDE de +0.0486 contre un effet vise de +0.0123. CE PILOTE NE TESTE PAS L'ENDPOINT
# grow - fix, et le lire comme un nul serait la faute que `underpowered-not-null`
# documente. Il teste l'accuracy ABSOLUE, dont l'ordre de grandeur de reference est
# l'ecart d'architecture grow_shallow - grow_deep, +0.108.
#
# La question posee est : est-ce que w1 et/ou reg bougent l'accuracy absolue ? Si reg
# la bouge, le diagnostic "regularisation-limited" est confirme et il faudra RELANCER
# grow - fix par-dessus, une fois le bras sorti du regime de memorisation -- c'est la
# seule condition dans laquelle la question de la croissance a un sens.
#
# LIMITE ASSUMEE : `reg` est un facteur GROUPE (dropout ET normalisation ensemble).
# S'il gagne, un second passage devra le decomposer. C'est le prix de ne pas tripler la
# grille, et c'est dit ici plutot que decouvert a la lecture des resultats.
#
# LE NIVEAU w1=8 / reg=off EST RELANCE, PAS REPRIS
# ------------------------------------------------
# Il existe dans results_deepbottleneck aux memes graines et sous le meme PROTOCOL,
# mais il a tourne sur margpu010/011 et celui-ci tourne sur margpu007/008. Meme classe
# de carte, noeuds differents : 12 cellules, c'est le prix de ne pas avoir a s'en
# expliquer. Les six niveaux du plan sont donc sur les deux memes noeuds.
#
# POURQUOI `tau` + turing, ALORS QUE LE PILOTE 578293 ETAIT SUR AMPERE
# ---------------------------------------------------------------------
# Recompte du parc au moment de soumettre, en soustrayant GRES_USED de GRES sur les
# trois partitions accessibles (`tau`, `gpu`, `gpu-best`) : ZERO ampere libre, ZERO
# hopper libre, QUATRE turing libres sur tout le cluster. Les 6 A100 de margpu007/008
# ne sont pas prises par des jobs best-effort mais par des jobs `tau` (jjobard,
# 578320/578324, limite 1 jour, 2h30 ecoulees) -- meme PriorityTier=100, donc la
# preemption ne joue pas et il n'y a rien a prendre. La premiere soumission (580874,
# ampere, annulee sans avoir ecrit une seule ligne) serait restee PENDING jusqu'a 21 h.
#
# Reste turing : margpu021 (3 cartes libres) et margpu020 (1), dans un parc `tau` de
# 23 cartes qui se libere au fil de l'eau. C'est aussi la classe sur laquelle la
# campagne v5 elle-meme a tourne. pascal (sm_60) et volta (sm_70) restent inutilisables :
# torch 2.13+cu130 veut sm_75+.
#
# margpu021 est le noeud de l'incident de la carte fantome (annonçait gpu:turing:3 et
# repondait "No devices were found" en rendant des CSV bien formes). Sonde avant de
# soumettre plutot que supposee : RTX 2080 Ti, sm_75, 10.4G libres, matmul ok. Saine.
#
# Le changement de classe ne casse aucune comparaison : les six niveaux du plan sont
# relances ICI, y compris w1=8/reg=off, donc aucune paire ne traverse de frontiere de
# materiel. C'est exactement pourquoi ce niveau est relance et non repris.
#
# ORDRE DE LA GRILLE
# ------------------
# w1_pilot.txt est trie par GRAINE d'abord : les 24 premieres cellules sont le plan
# COMPLET a la graine 0 (12 modeles x 2 alignements). A %12 cela donne une lecture de
# tout le plan en 2 vagues au lieu de 6, et les graines 1-2 ne font que resserrer les
# intervalles. Reprenable : chaque tache saute sa cellule si le CSV existe.
#
# RESULTS_DIR DEDIE
# -----------------
# Le niveau w1=8/reg=off reutilise les labels `grow_deep` / `fix_deepeeg`, deja
# presents dans results/ ET dans results_deepbottleneck/. Tout part dans results_w1pilot/.
#SBATCH --job-name=w1-pilot
#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-71%12
#SBATCH --output=/scratch/amounir/eegrow_budget/w1_logs/w1_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/amounir/eegrow_budget

# Le checkout cluster est une COPIE, pas un clone : `git rev-parse` y echoue, donc
# tree_sha() rend None et chaque ligne porterait eegrow_sha=None. C'est la perte de
# tracabilite qui a coute les 1170 cellules v5. Le stamp est ecrit au deploiement.
if [ -z "${EEGROW_SHA:-}" ] && [ -r "$ROOT/.eegrow_sha" ]; then
  export EEGROW_SHA="$(tr -d '[:space:]' < "$ROOT/.eegrow_sha")"
fi

POINTS=$ROOT/benchmarks/slurm/w1_pilot.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED ALIGN <<< "$LINE"
echo "[$(date -Is)] w1 task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL ds=$DS model=$MODEL seed=$SEED align=$ALIGN"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
# $ROOT/src EN PREMIER : l'env `bench` porte un editable .pth qui pointe en dur sur
# /scratch/amounir/eegrow/src. Sans cette ligne `import eegrow` resout vers l'AUTRE
# checkout -- c'est ce qui a tue les 36 cellules de 578248.
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=$ROOT/moabb_results_w1

# La carte fantome : margpu021 annonçait gpu:turing:3 et repondait "No devices were
# found" sans echouer, rendant des CSV bien formes contre un device absent. On sonde
# mem_get_info -- deduire l'ordinal de nvidia-smi est ce qui avait tue 47 cellules.
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
# MODELE DE CARTE EPINGLE, PAS SEULEMENT LA CLASSE. `gpu:turing` couvre plusieurs
# modeles ; les comparaisons de ce pilote sont appariees fit par fit, donc une cellule
# sur T4 contre une cellule sur 2080 Ti mettrait une difference de MATERIEL dans la
# difference appariee. On refuse plutot que d'avertir : une tache qui atterrit sur un
# autre modele sort en erreur et se voit dans squeue, au lieu de rendre un CSV bien
# forme qu'on decouvrirait a l'analyse -- ou jamais.
REF = "2080 Ti"
if REF not in name:
    print(f"carte {name} != modele de reference '{REF}' : refus, "
          f"le pilote doit rester homogene"); sys.exit(1)
PROBE

# PROVENANCE ET VERSION DU CODE, ASSERTEES PLUTOT QU'ARRANGEES.
# Un garde-fou refuse, il n'avertit pas : sans le troisieme test ci-dessous, un tree
# anterieur au commit de regularisation accepterait `drop_prob` comme un kwargs inconnu
# ou l'ignorerait, et les six niveaux `_reg` sortiraient bien formes et identiques aux
# niveaux sans reg -- un plan factoriel dont un facteur ne fait rien, sans un mot.
python - "$ROOT" <<'PY' || { echo "GUARD FAILED -- abandon"; exit 91; }
import inspect, sys
import eegrow
from eegrow.training import loop
from eegrow.models.growing_deepeeg import GrowingDeepEEGNet
root = sys.argv[1].rstrip("/") + "/"
if not eegrow.__file__.startswith(root):
    sys.exit(f"FATAL: ROOT est {root} mais eegrow importe depuis {eegrow.__file__}")
if "best_s == 0.0" not in inspect.getsource(loop.grow_step):
    sys.exit(f"FATAL: {loop.__file__} est anterieur a 5337c56 (abstention s=0 absente)")
sig = inspect.signature(GrowingDeepEEGNet.__init__).parameters
for k in ("drop_prob", "norm_junction"):
    if k not in sig:
        sys.exit(f"FATAL: GrowingDeepEEGNet n'a pas de parametre {k} -- tree anterieur "
                 f"a la regularisation, les 36 cellules _reg seraient des doublons")
# et les defauts doivent rester OFF, sinon les 36 cellules sans reg n'en sont pas
if sig["drop_prob"].default != 0.0 or sig["norm_junction"].default is not False:
    sys.exit("FATAL: les defauts de regularisation ne sont plus OFF")
sys.path.insert(0, root + "benchmarks")
from utils import tree_sha  # noqa: E402
sha = tree_sha()
if not sha:
    sys.exit("FATAL: ce tree ne sait pas nommer son commit -- eegrow_sha=None partout")
print(f"guard ok  eegrow={eegrow.__file__}  drop_prob/norm_junction presents  sha={sha[:12]}")
PY

# PROTOCOL byte-identique a final_grid.py et a 578293 : sans ces deux overrides on
# relance la campagne sous-entrainee, avec des CSV bien formes et aucun avertissement.
cd $ROOT/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" align="$ALIGN" \
  train.patience=200 train.selection_monitor=valid_acc \
  dataset.resample=250 \
  results_dir="$ROOT/benchmarks/results_w1pilot" \
  overwrite=true suffix="w1_${MODEL}_${ALIGN}_s${SEED}"
