#!/bin/bash
# ABLATION DE READOUT -- d'ou vient reellement l'ecart de +0.108 grow_shallow/grow_deep ?
#
# CE QUE LE PILOTE w1 (580878) A DEJA ELIMINE
# -------------------------------------------
# Sur le bras deep, deux des trois explications classiques sont mortes, mesurees :
#   * CAPACITE       -- w1 8 -> 32 rapporte +0.0235 (a reg off). Reel et significatif,
#                       mais 4x trop petit pour un ecart de +0.108.
#   * REGULARISATION -- dropout 0.5 + GroupNorm DEGRADE : 10/12 contrastes apparies
#                       negatifs, moyenne -0.014, et les 3 seuls IC95 qui excluent zero
#                       sont TOUS du cote negatif. Aucun contraste positif significatif.
#                       Mon diagnostic "regularisation-limited" lisait bien le symptome
#                       (train loss 0.001, valid a 0.1 nat sous ln 4) et designait le
#                       mauvais organe.
#
# L'EXPLICATION QUI RESTE, ET SA PHYSIQUE
# ---------------------------------------
# shallow : BatchNorm -> Square -> AvgPool -> SafeLog
# deep    : BatchNorm -> ELU    -> MaxPool
#
# `Square -> AvgPool` = moyenne des carres sur une fenetre = LA VARIANCE du signal
# filtre spatialement. Suivie du log, c'est une log-puissance de bande. Or l'imagerie
# motrice est un phenomene ERD/ERS : la quantite discriminante est la PUISSANCE mu/beta,
# pas une forme d'onde phase-lockee -- une fonctionnelle QUADRATIQUE du signal.
# ShallowFBCSPNet est donc du FBCSP ecrit en couches : la statistique qu'il faut estimer
# est DANS l'architecture. Une queue ReLU doit, elle, approximer x^2 par morceaux
# lineaires -- faisable (Deep4Net y arrive avec 25/50/100/200 filtres sur 4 etages) mais
# ca se paie en unites ET en essais. Le bras deep a 8/16 filtres et 230 essais par sujet.
#
# COROLLAIRE, ET C'EST LUI QUI COMPTE POUR LE PAPIER
# --------------------------------------------------
# La jonction growable de deep (conv2b) vit dans l'etage 2, EN AVAL du collapse spatial
# a w1=8. Par inegalite de traitement des donnees, ce que ce collapse a detruit ne se
# recupere pas en aval, quelle que soit la largeur ajoutee. L'axe croissable n'est pas
# l'axe limitant -- ce n'est pas un fait sur la croissance, c'est un fait sur l'endroit
# ou la jonction a ete posee. shallow, lui, fait croitre conv_spat, c'est-a-dire le
# goulot lui-meme : d'ou une croissance qui paie.
#
# POURQUOI ABLATER shallow ET NON AUGMENTER deep
# ----------------------------------------------
# deep differe de shallow sur QUATRE axes a la fois (profondeur, largeur, position de
# jonction, readout). Y ajouter Square/AvgPool/Log ne trancherait rien. On fait donc
# l'inverse : on prend le bras qu'on sait bien mesurer et on lui retire UN facteur, en
# tenant la geometrie du pooling fixe. Resultat verifie au smoke test : les trois
# readouts ont 18372 parametres et final_conv_length=69, a l'identique. C'est une
# ablation de FONCTION, pas un changement de capacite deguise.
#
# LE FACTEUR EST DECOMPOSE D'AVANCE, PAS APRES COUP
# -------------------------------------------------
#   power   : Square -> AvgPool -> SafeLog     (= grow_shallow, la reference)
#   nolog   : Square -> AvgPool -> Identity    (garde la variance, retire le log)
#   generic : ELU    -> MaxPool -> Identity    (la queue de la famille ReLU)
#
# `reg` etait un facteur GROUPE dans 580878 ; il a perdu, donc on n'a pas eu a le
# decomposer, mais on ne refait pas le pari. Le log porte un effet PROPRE : la puissance
# de bande est a queue lourde, et le log transforme l'ERD -- modulation MULTIPLICATIVE
# de la puissance -- en decalage ADDITIF, donc en quelque chose qu'un classifieur
# lineaire sait lire. Sans le barreau intermediaire, un effondrement de `generic` ne
# dirait pas si c'est le carre ou la compression qui porte l'effet.
#
# PREDICTION PREENREGISTREE
# -------------------------
# Echelle attendue :  power > nolog > generic ~ grow_deep (~0.44 sur bnci2014_001).
# Si `generic` TIENT a ~0.54, l'explication du readout est FAUSSE et l'ecart est
# ailleurs -- ce sera a ecrire tel quel, pas a reinterpreter.
#
# CE QUE CETTE CAMPAGNE NE TESTE PAS
# ----------------------------------
# Toujours pas l'endpoint grow - fix : bnci2014_001 porte 9 sujets et son MDE apparie
# reste vers +0.05 contre un effet vise de +0.0123. On teste l'accuracy ABSOLUE, dont
# l'ordre de grandeur de reference est justement le +0.108 qu'on cherche a expliquer.
#
# grow_shallow EST RELANCE, PAS REPRIS
# ------------------------------------
# Ses chiffres v5 viennent d'une AUTRE campagne, sous un protocole different (patience
# et selection_monitor). Apparier deux campagnes est exactement l'erreur documentee par
# `xds-august-grid-obsolete` -- RestoreBestModel valait ~2 pp a lui seul. La reference
# est donc relancee ICI, aux memes graines, alignements, protocole et modele de carte.
# grow_deep, lui, n'est PAS relance : il est deja dans results_w1pilot sous exactement
# ce protocole, ces graines et ces alignements, donc directement appariable.
#
# RESULTS_DIR DEDIE : `grow_shallow` reutilise un label deja present dans results/.
#SBATCH --job-name=readout-abl
#SBATCH --partition=tau
#SBATCH --gres=gpu:turing:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-17%12
#SBATCH --output=/scratch/amounir/eegrow_budget/ro_logs/ro_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/amounir/eegrow_budget

# Le checkout cluster est une COPIE, pas un clone : `git rev-parse` y echoue, donc
# tree_sha() rend None et chaque ligne porterait eegrow_sha=None. Le stamp est ecrit au
# deploiement.
if [ -z "${EEGROW_SHA:-}" ] && [ -r "$ROOT/.eegrow_sha" ]; then
  export EEGROW_SHA="$(tr -d '[:space:]' < "$ROOT/.eegrow_sha")"
fi

POINTS=$ROOT/benchmarks/slurm/readout_ablation.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED ALIGN <<< "$LINE"
echo "[$(date -Is)] readout task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL ds=$DS model=$MODEL seed=$SEED align=$ALIGN"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
# $ROOT/src EN PREMIER : l'env `bench` porte un editable .pth qui pointe en dur sur
# /scratch/amounir/eegrow/src. Sans cette ligne `import eegrow` resout vers l'AUTRE
# checkout -- c'est ce qui a tue les 36 cellules de 578248.
export PYTHONPATH=$ROOT/src:$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=$ROOT/moabb_results_ro

# La carte fantome : on sonde mem_get_info plutot que de deduire l'ordinal de
# nvidia-smi, ce qui avait tue 47 cellules.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PROBE'
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() est False"); sys.exit(1)
free, total = torch.cuda.mem_get_info(0)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"PROBE ok: {name} sm_{cap[0]}{cap[1]} free={free/2**30:.1f}G/{total/2**30:.1f}G")
if cap[0] * 10 + cap[1] < 75:
    print("carte sous sm_75 : torch ne peut pas y tourner"); sys.exit(1)
# MODELE EPINGLE, PAS SEULEMENT LA CLASSE. Deux raisons ici : les contrastes entre
# readouts sont apparies fit par fit, ET l'ancrage grow_deep vient de results_w1pilot,
# qui a tourne sur 2080 Ti. Une cellule sur T4 mettrait du MATERIEL dans les deux
# differences a la fois.
REF = "2080 Ti"
if REF not in name:
    print(f"carte {name} != modele de reference '{REF}' : refus, "
          f"l'ablation doit rester homogene et comparable a results_w1pilot"); sys.exit(1)
PROBE
PROBE_RC=$?

# Une cellule qui tombe sur un creneau fantome revient EN FILE, elle ne meurt pas : le
# defaut est PAR CARTE et pas par noeud, et la sonde rend la main en 5 s -- un creneau
# fantome peut donc "finir" tout un array en quelques minutes, tout en echecs. Plafond
# a 5 remises, au-dela on sort en 90 et ca redevient visible dans sacct.
if [ $PROBE_RC -ne 0 ]; then
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 5 ]; then
    echo "GPU PROBE FAILED (rc=$PROBE_RC) sur $(hostname) -- creneau fantome ou modele" \
         "non conforme, remise en file (essai ${SLURM_RESTART_COUNT:-0}/5)"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 90
  fi
  echo "GPU PROBE FAILED (rc=$PROBE_RC) apres 5 remises en file -- abandon de la tache"
  exit 90
fi

# PROVENANCE ET VERSION, ASSERTEES PLUTOT QU'ARRANGEES.
# Un garde-fou refuse, il n'avertit pas. Sans le test de construction ci-dessous, un
# tree anterieur au commit d'ablation avalerait `readout: generic` comme un kwargs
# inconnu (ou le laisserait filer dans **kwargs) et les 12 cellules ablatees sortiraient
# bien formees et IDENTIQUES a la reference -- une ablation dont le facteur ne fait
# rien, sans un mot. On ne verifie donc pas la signature : on CONSTRUIT les trois et on
# lit les modules obtenus.
python - "$ROOT" <<'PY' || { echo "GUARD FAILED -- abandon"; exit 91; }
import inspect, sys
import torch
from torch import nn
import eegrow
from eegrow.training import loop
from eegrow.models.growing_shallow import GrowingShallowFBCSPNet
from braindecode.modules import SafeLog, Square

root = sys.argv[1].rstrip("/") + "/"
if not eegrow.__file__.startswith(root):
    sys.exit(f"FATAL: ROOT est {root} mais eegrow importe depuis {eegrow.__file__}")
if "best_s == 0.0" not in inspect.getsource(loop.grow_step):
    sys.exit(f"FATAL: {loop.__file__} est anterieur a 5337c56 (abstention s=0 absente)")

sig = inspect.signature(GrowingShallowFBCSPNet.__init__).parameters
if "readout" not in sig:
    sys.exit("FATAL: GrowingShallowFBCSPNet n'a pas de parametre readout -- tree "
             "anterieur a l'ablation, les 12 cellules ablatees seraient des doublons")
if sig["readout"].default != "power":
    sys.exit("FATAL: le defaut de readout n'est plus 'power' -- la reference "
             "grow_shallow ne serait plus la reference")

EXPECTED = {
    "power":   (Square,  nn.AvgPool2d, SafeLog),
    "nolog":   (Square,  nn.AvgPool2d, nn.Identity),
    "generic": (nn.ELU,  nn.MaxPool2d, nn.Identity),
}
ref = None
for r, (c_t, p_t, l_t) in EXPECTED.items():
    m = GrowingShallowFBCSPNet(n_chans=22, n_outputs=4, n_times=1125,
                               n_filters_time=8, n_filters_spat=40,
                               target_n_filters_time=40, readout=r, device="cpu")
    got = (type(m.conv_nonlin), type(m.pool), type(m.pool_nonlin))
    if got != (c_t, p_t, l_t):
        sys.exit(f"FATAL: readout={r} construit {got}, attendu {(c_t, p_t, l_t)}")
    # geometrie tenue fixe : sans ca l'ablation serait un changement de capacite
    geom = (sum(p.numel() for p in m.parameters()), m.final_conv_length)
    if ref is None:
        ref = geom
    elif geom != ref:
        sys.exit(f"FATAL: readout={r} a la geometrie {geom} contre {ref} pour power -- "
                 f"l'ablation changerait la capacite, pas seulement la fonction")

sys.path.insert(0, root + "benchmarks")
from utils import tree_sha  # noqa: E402
sha = tree_sha()
if not sha:
    sys.exit("FATAL: ce tree ne sait pas nommer son commit -- eegrow_sha=None partout")
print(f"guard ok  eegrow={eegrow.__file__}  3 readouts distincts  geom={ref}  sha={sha[:12]}")
PY

# PROTOCOL byte-identique a 580878 : sans ces overrides on relance sous-entraine, avec
# des CSV bien formes et aucun avertissement. `dataset.resample=250` est repris tel quel
# parce que l'ancrage grow_deep de results_w1pilot a tourne dessus.
cd $ROOT/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" align="$ALIGN" \
  train.patience=200 train.selection_monitor=valid_acc \
  dataset.resample=250 \
  results_dir="$ROOT/benchmarks/results_readout" \
  overwrite=true suffix="ro_${MODEL}_${ALIGN}_s${SEED}"
