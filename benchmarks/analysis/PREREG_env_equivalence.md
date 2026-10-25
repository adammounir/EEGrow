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
