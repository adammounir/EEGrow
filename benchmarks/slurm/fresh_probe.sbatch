#!/bin/bash
#SBATCH --job-name=eegrow_freshprobe
#SBATCH --partition=tau,gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --exclude=margpu001,margpu004,margpu020,margpu021,margpu023,margpu024,margpu025,margpu026,margpu027,margpu029
#SBATCH --output=/scratch/amounir/logs/freshprobe_%j.log
#
# LA SONDE SUR LES DEUX DATASETS JAMAIS VUS -- 6 cellules, ~24 GPU-h.
#
# Ce que ce job produit, et pourquoi il est le chemin critique du chantier ICLR :
# `params_probe`, `acc_probe` et `restored_epoch` par sujet sur `dreyer2023a` (60) et
# `liu2024` (50). Ce sont les prédicteurs de la pré-registration
# `benchmarks/analysis/PREREG_claim2_fresh_dreyer_liu.md`, écrite AVANT ce job et
# horodatée en tête. Tout ce que le chantier a produit depuis l'étage C est post-hoc ;
# ces 110 sujets sont la seule voie vers du confirmatoire, et elle ne s'emprunte
# qu'une fois. Relire la pré-registration avant de toucher à ce fichier.
#
# LE COÛT EST MESURÉ -- ET LA PREMIÈRE MESURE ÉTAIT LA MAUVAISE. J'avais transposé le
# coût de cho2017 (260 fits x 41 s = 3.0 h/cellule) en supposant « même ordre de
# grandeur ». Faux d'un ordre de grandeur : cho2017 a 64 canaux, dreyer2023a en a 27 et
# liu2024 29, et les logs de 525777/525778 mesurent **15 s par fit de 200 époques** ici.
# Une cellule within_session coûte donc ~1 h 15. Le nombre de sujets (60 et 50) était le
# mauvais axe de comparaison ; c'est n_chans x n_times qui porte le coût.
#
# ET CETTE CORRECTION-LÀ ÉTAIT FAUSSE AUSSI, d'un facteur 7. Les deux axes sont vrais et
# aucun ne suffit : ni le nombre de sujets ni n_chans x n_times ne porte la co-tenance
# (K processus sur une carte) ni le nombre de folds. La seule mesure fiable est prise sur
# le log de la cellule elle-même -- `grep -c "Passing additional parameters"` rapporté à
# son horodatage de départ -- et elle doit être faite dès que la cellule tourne depuis
# 30 min, pas après coup sur un job déjà condamné. Voir le bloc G/K plus bas.
#
# UNE SEULE ALLOCATION, ET C'EST UN CHOIX. Les corrélations de claim 2 sont
# INTRA-dataset : on classe les sujets d'un dataset entre eux, puis on groupe les
# rho en z de Fisher. Aucune comparaison ne traverse une frontière de machine, donc
# l'homogénéité matérielle n'a besoin d'être garantie qu'à l'intérieur d'un dataset
# -- et un seul nœud la donne gratuitement, sans le raisonnement fragile qu'un array
# imposerait. C'est le même argument que `donor_receiver.sbatch`, pour la même raison.
#
# PAS DE CLASSE DE CARTE, corrigé le 03/09 à 17 h 40 après mesure. La version soumise
# demandait `ampere:3` en s'appuyant sur un état du cluster qui n'était déjà plus vrai :
# `sinfo -O gres,gresused` montre que les 6 ampere du tau sont prises (margpu007 pour
# ~20 h par l'array 517347, margpu008 pour 5 j par un `bash` interactif) et qu'AUCUNE
# ampere n'est libre nulle part, pendant que `shipped` finit ses 9 dernières cellules et
# rendra 11 turing dans la soirée. Le pin ampere coûtait donc ~14 h d'attente pour
# obtenir la classe la plus rare. Sans classe, ce job prend le premier bloc de 3 cartes
# utilisables ; le tau et le gpu n'exposent qu'ampere et turing une fois l'exclusion
# appliquée, donc l'homogénéité reste garantie -- une allocation, un nœud, une classe.
#
# L'EXCLUSION DEVIENT PORTANTE. Elle était tenue en creux par le pin ampere : sans lui,
# margpu024-027/029 (pascal sm_60, "no kernel image is available", et pourtant `idle`
# dans sinfo), margpu001/004 (volta), margpu020/021 (cartes fantômes, mesuré le 02/09,
# cf. donor_all.sbatch) et margpu023 (drained) redeviennent atteignables. margpu020
# manquait à la liste d'origine : c'est le nœud qui a tué `515639_2` sans message.
set -uo pipefail

export ROOT=/scratch/amounir/eegrow_budget
#
# PROTOCOL : byte-identique à final_grid.sbatch. config.yaml livre délibérément les
# valeurs perdantes, et passer un seul des deux knobs donne un TROISIÈME protocole
# faux plutôt qu'une demi-correction (l'interaction mesurée vaut +0.0854, p=1.1e-07).
# La sonde des datasets frais doit être la MÊME mesure que celle des 12 datasets de la
# campagne, sinon le classement qu'elle produit n'est pas celui dont claim 2 a estimé
# la rho. Voir [[deep4-instability]].
export PROTOCOL="train.patience=200 train.selection_monitor=valid_acc"
#
# RESULTS_DIR, LOGS et CLAIMS surchargés ENSEMBLE. pack_run.sh l'exige explicitement
# (ligne 57) et CLAIMS est le pire des trois à oublier : `reap_stale` ne moissonne que
# les claims dont il croit connaître le propriétaire, donc partager le répertoire avec
# la campagne `shipped` en cours ferait moissonner ses cellules vivantes par ce job-ci.
export RESULTS_DIR=/scratch/amounir/results_fresh
export CLAIMS=/scratch/amounir/eegrow_claims_fresh
export LOGS=/scratch/amounir/logs/pack_fresh
export GRID="$ROOT/benchmarks/slurm/fresh_probe.tsv"
export SUFFIX=fresh
#
# EEGROW_CUDA_FRACTION : C'EST L'OUBLI QUI A TUÉ LA PREMIÈRE SOUMISSION. `525776` est
# mort en 1 seconde sur `PACK FATAL: EEGROW_CUDA_FRACTION unset` — et ce garde-fou a
# fait exactement son travail : il a REFUSÉ, il n'a pas averti, donc rien n'a tourné
# avec une valeur inventée ([[gates-must-refuse-not-warn]]). Coût : ~6 h de file.
#
# La valeur est DÉRIVÉE DE MESURES, pas choisie. Deux sources concordantes :
#
# 1. `/scratch/amounir/grid_memory_v2.json` donne le pic réservé de `grow_shallow` par
#    dataset sur une 2080 Ti (10 831 MiB), et il est quasi affine en n_chans x n_times :
#    bnci2014_002 18 750 -> 788 MiB, bnci2014_001 22 022 -> 908, shin2017a 75 030 -> 3668,
#    soit 0.041 à 0.049 MiB/unité. dreyer2023a (27 ch x 1250) = 33 750 unités et liu2024
#    (29 x 1000) = 29 000, donc au pire ~1650 et ~1420 MiB : **0.15 et 0.13 de la carte**.
#    Ce sont les deux datasets les plus légers de tout le chantier après les bnci.
# 2. Corroboration indépendante : `525777` et `525778` viennent de faire tourner CE
#    modèle sur CES deux datasets à `EEGROW_CUDA_FRACTION=0.30`, trois processus
#    co-tenants sur une carte, sans un seul OOM.
#
# 0.30 est donc le double du pire cas interpolé, et déjà éprouvé en co-tenance.
export EEGROW_CUDA_FRACTION=0.30
#
# G=3 K=1, APRÈS UNE TROISIÈME MESURE -- ET C'EST LA SEULE QUI VAILLE, parce qu'elle est
# prise sur CE job et pas transposée d'un autre. Historique à ne pas effacer :
#   1ʳᵉ estimation : 4 h/cellule, transposée de cho2017 en comparant le nombre de sujets.
#   2ᵉ : ~1 h 15, transposée des 15 s/fit des matrices D×R. Meilleur axe (n_chans x
#        n_times), mais toujours une transposition -- et fausse d'un facteur 7.
#   3ᵉ, celle-ci : le log de `527597` donne **89 fits en 2 h 34, soit 104 s/fit** en
#        co-tenance K=3, pour ~300 fits attendus (60 sujets x 5 folds, comme les 260 fits
#        mesurés sur les 52 sujets de cho2017). Soit **8,6 h/cellule**, contre un
#        `CELL_TIMEOUT` à 4 h : les 3 cellules allaient être tuées par `timeout` (ligne
#        523 de pack_run.sh) sans écrire une seule ligne.
#
# Ce que la mesure dit du dimensionnement : le travail total est fixe, ~17 GPU-heures.
# Une carte ne peut pas le raccourcir -- K=3 triple le débit apparent et divise par trois
# la vitesse de chaque fit, donc 1 carte = ~17 h de wall clock quel que soit K. Seul le
# nombre de CARTES achète du temps. 3 cartes en K=1 : chaque cellule tourne seule à
# ~35 s/fit = 2 h 55, deux vagues = **~6 h**.
#
# K=1 plutôt que K=2 à nombre de cartes égal : même wall clock, mais les cellules sont
# deux fois plus courtes, donc la première vague est ÉCRITE sur disque au bout de 3 h.
# Avec K=2 les 6 cellules finissent ensemble et un walltime raté les perd toutes les six.
export CELL_TIMEOUT=18000
export G=3 K=1

bash "$ROOT/benchmarks/slurm/pack_run.sh"
