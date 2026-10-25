# PREREG — les cartes volta et pascal peuvent-elles entrer dans la campagne A ?

25/10/2026. Gelé avant le lancement (commit qui contient ce fichier, `env_equiv.sbatch`,
`env_equiv_read.py` et les `equiv_cells_*.txt`).

## 1. Question

L'env `bench` (torch 2.13.0+cu130) n'a pas de noyaux pour sm_60 (pascal) et sm_70 (volta),
soit environ 30 cartes de Margaret, souvent libres. L'env `bench126` est un clone de `bench`
où seul le build de torch change : 2.13.0+cu126, même version, qui a ces noyaux. Le diff
`pip freeze` ne touche que torch, torchaudio, la pile nvidia-* et cuda-bindings/toolkit.

Les deux questions :
1. Est-ce que l'env ou la carte change le résultat plus que ce que change une autre graine ?
2. À quelle vitesse vont ces cartes ?

## 2. Plan

Une cellule complète `within_session` sur bnci2014_001 (9 sujets × 2 sessions × 5 folds),
ShallowFBCSPNet bd et tiny, au protocole de la validation (200 époques).

| Condition | Env | Carte | Graine |
|---|---|---|---|
| ref_s0 | bench | 2080 Ti | 0 |
| ref_s0_rerun | bench | 2080 Ti | 0 |
| ref_s1 | bench | 2080 Ti | 1 |
| b126_turing | bench126 | 2080 Ti | 0 |
| b126_volta | bench126 | V100 | 0 |
| b126_pascal | bench126 | 1080 Ti | 0 |

## 3. Mesures

Unité = le sujet (moyenne de ses folds et sessions), apparié par sujet contre ref_s0.
- `dev(c)` = moyenne sur les sujets de |acc(c) − acc(ref_s0)|.
- `dev(ref_s1)` = l'écart que produit un changement de graine, l'étalon.
- `dev(ref_s0_rerun)` = le non-déterminisme sur la même carte (0 si bit-exact).
- Rapporté aussi : la différence moyenne signée, avec son IC95 bootstrap (10 000, rng 0),
  la largeur finale du bras tiny, et le temps par cellule rapporté à ref_s0.

## 4. Critère

Pour chaque condition b126_* et chaque bras :
- **ÉQUIVALENT** si `dev(c) ≤ dev(ref_s1)` ET si l'IC95 de la différence signée contient 0.
- **REFUSÉ** sinon.

Une carte entre dans la campagne A si ses deux bras sont ÉQUIVALENT (et b126_turing aussi,
sinon c'est l'env qui est refusé). Elle y entre avec `gpu_name` enregistré sur chaque fit,
comme les autres.

Limite connue : 9 sujets, un seul modèle (le moins coûteux). Le test détecte une rupture
(noyaux faux, croissance cassée), pas un biais de 0,5 point.

## 5. Avenant 1 — 25/10/2026 soir, avant tout lancement des nouvelles conditions

Pourquoi : la vague principale de la campagne A (818892-818906) tourne sur ~47 cartes et
Adam demande le maximum de parallélisme. Deux modèles de carte de Margaret, présents et
souvent libres, n'ont jamais été testés : la lecture du 25/10 a eu une **P100** pour
« pascal » (margpu029) et une **V100-PCIE** pour « volta ». Sondés ce soir par `nvidia-smi` :

| Nœuds | Carte | sm | Mémoire |
|---|---|---|---|
| margpu024-027 (16 cartes) | NVIDIA GeForce GTX 1080 Ti | 6.1 | 11 Go |
| margpu032 (4 cartes) | Tesla V100-SXM2-32GB | 7.0 | 32 Go |

(margpu004, V100S : nœud injoignable, allocation révoquée ; hors avenant.)

Ajout de deux conditions, rien d'autre ne change :

| Condition | Env | Carte | Graine |
|---|---|---|---|
| b126_1080ti | bench126 | GTX 1080 Ti (`EXPECT`) | 0 |
| b126_v100sxm2 | bench126 | V100-SXM2-32GB (`EXPECT`) | 0 |

- Mêmes cellules (`equiv_cells_1080ti.txt`, `equiv_cells_v100sxm2.txt` = les lignes de
  `equiv_cells_pascal.txt`, label près), même `env_equiv.sbatch`, **même commit que les
  références (50b56b8)** : seul l'env/la carte change entre la référence et le test.
- Mêmes références : ref_s0 / ref_s1 de `/scratch/amounir/eegrow_v2/equiv/`, copiées
  (lecture seule) dans le checkout de l'avenant. Elles ne sont pas refaites.
- Même critère § 4, même lecteur (deux conditions ajoutées à `CONDS`).
- Conséquence : une carte ÉQUIVALENT entre dans la campagne A par un avenant au PREREG de
  la campagne (porte de carte du sbatch), pour les tâches encore en attente seulement.
