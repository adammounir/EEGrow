#!/bin/bash
# PILOTE GOULOT -- est-ce que la croissance sur `GrowingDeepEEGNet` ne fait rien parce
# qu'elle grossit derriere un goulot de w2=8 canaux ?
#
# CE QUE CE PILOTE PEUT ET NE PEUT PAS CONCLURE. A LIRE AVANT D'INTERPRETER.
# ------------------------------------------------------------------------
# bnci2014_001 porte 9 sujets. Le contraste apparie grow - fix y a un MDE de +0.0486
# (sd=0.052 mesuree sur la grille complete), contre un effet vise de +0.0123 -- la
# taille de l'effet de croissance sur la paire shallow. Ce pilote NE PEUT DONC PAS
# tester l'endpoint : quel que soit le resultat, grow - fix y sera non concluant, et
# le lire comme un nul serait la faute que `underpowered-not-null` documente.
#
# Ce qu'il teste, c'est l'ETAGE EN AMONT, dont l'effet attendu est un ordre de
# grandeur plus grand : si w2=8 etrangle le reseau, elargir w2 doit bouger l'accuracy
# ABSOLUE des deux bras. L'ordre de grandeur de reference est l'ecart d'architecture
# grow_shallow - grow_deep, +0.108, tres au-dessus du MDE. Trois issues :
#
#   * l'accuracy monte nettement avec w2  -> goulot reel, la geometrie de la campagne
#     est mal specifiee. On fige le niveau, on relance la grille UNE fois, et on
#     declare la selection dans le papier.
#   * l'accuracy ne bouge pas             -> le goulot n'est pas la contrainte. Le nul
#     mesure sur n=324 est le resultat, et il devient un mecanisme a ecrire : la
#     croissance paie quand la jonction growable EST le goulot de capacite (shallow),
#     pas sinon (deep, sccnet).
#   * l'accuracy monte mais grow et fix montent pareil -> le goulot etait reel et
#     n'etait pas ce qui neutralisait la croissance. Il faut alors l'etage 2
#     (cho2017 + physionetmi, n=161, MDE +0.0115) pour trancher l'endpoint.
#
# POURQUOI LES DEUX BRAS BOUGENT A CHAQUE NIVEAU
# -----------------------------------------------
# `fix_deepeeg_wNN` suit `grow_deep_wNN` a chaque niveau. Elargir w2 sur le seul bras
# growing donnerait un plus gros chiffre et un controle casse : la paire ne differerait
# plus par la croissance seule mais par la croissance ET la largeur, ce qui est
# exactement la confusion que `fix_deepeeg` existe pour eliminer.
#
# POURQUOI LE NIVEAU w2=8 EST RELANCE PLUTOT QUE REPRIS DE perf_final/
# --------------------------------------------------------------------
# Il existe deja, aux memes seeds et sous le meme PROTOCOL. Mais il a tourne sur
# turing, et ce pilote tourne sur ampere : reprendre la ligne de base la ferait
# traverser une frontiere de classe de carte a l'interieur meme de la comparaison que
# la figure porte. 12 cellules a ~0.2 h, c'est le prix de ne pas avoir a s'en
# expliquer. Les trois niveaux sont donc sur la meme classe de carte.
#
# POURQUOI `gpu:ampere:1` ET PAS turing
# --------------------------------------
# Constate au moment d'ecrire : les 6 noeuds turing de `tau` sont tous `alloc`, les
# pascal sont inutilisables (torch 2.13+cu130 demande sm_75+, pascal est sm_60) et
# volta (margpu001, sm_70) l'est aussi. Restent margpu010 et margpu011, ampere, idle,
# sur la partition `gpu` -- que les templates de la campagne ne visent jamais. Le
# pilote entier tient dessus, donc aucune paire ne traverse de classe de carte.
#
# RESULTS_DIR DEDIE
# -----------------
# Le niveau w2=8 reutilise les labels `grow_deep` / `fix_deepeeg`. Ecrire dans
# `benchmarks/results` ecraserait la campagne. Tout part dans results_deepbottleneck/.
#SBATCH --job-name=deep-bottleneck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-35%4
#SBATCH --output=/scratch/amounir/eegrow_budget/db_logs/db_%A_%a.out
#SBATCH --open-mode=append

set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/amounir/eegrow_budget
POINTS=$ROOT/benchmarks/slurm/deep_bottleneck.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$POINTS")
read -r EVAL DS MODEL SEED ALIGN <<< "$LINE"
echo "[$(date -Is)] db task=${SLURM_ARRAY_TASK_ID} restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) -> eval=$EVAL ds=$DS model=$MODEL seed=$SEED align=$ALIGN"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
module load cuda/13.1 || true
export PYTHONPATH=$ROOT/benchmarks
export MNE_DATA=/scratch/amounir/mne_data
export MOABB_RESULTS=$ROOT/moabb_results_db

# La carte fantome : margpu021 annonçait gpu:turing:3 et repondait "No devices were
# found" sans echouer, donc chaque cellule tournait contre un device absent et rendait
# des CSV bien formes. On sonde `mem_get_info` -- deduire l'ordinal de nvidia-smi est
# precisement ce qui avait fait echouer 47 cellules. Un noeud qui ne repond pas ici
# doit sortir en erreur MAINTENANT, pas rendre des resultats.
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

# PROTOCOL byte-identique a final_grid.py : sans ces deux overrides on relance la
# campagne sous-entrainee, avec des CSV bien formes et aucun avertissement.
cd $ROOT/benchmarks
srun python run_moabb_hydra.py \
  eval="$EVAL" dataset="$DS" model="$MODEL" seed="$SEED" align="$ALIGN" \
  train.patience=200 train.selection_monitor=valid_acc \
  dataset.resample=250 \
  results_dir="$ROOT/benchmarks/results_deepbottleneck" \
  overwrite=true suffix="db_${MODEL}_${ALIGN}_s${SEED}"
