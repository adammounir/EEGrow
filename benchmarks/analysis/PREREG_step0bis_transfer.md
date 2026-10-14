# PREREG — Étape 0-bis : les critères de comportement tiennent-ils hors de bnci2014_001 ?

**Gelé le 14/10/2026, avant tout lancement.** Décidé par Adam le 14/10 (« oui, maintenant on
enchaîne »), après la ratification de l'avenant 2 de l'étape 0.

## 1. Pourquoi

L'étape 0 a figé son protocole sur `bnci2014_001`. L'avenant 2 a choisi la grille de la line
search sur les sujets 1 à 3 de ce même dataset. Le CADRAGE (§ 5, décision 2) impose de vérifier
les critères de **comportement** aussi sur `shin2017a` et `physionetmi`.

Ces deux datasets sont ceux où la campagne précédente scorait 59 % et 39 % des fits sur une
époque quasi non entraînée. Leurs splits sont minuscules :
- **shin2017a** : 20 essais par session. Le pli garde 16 essais d'entraînement, d'où
  `round(0.1 × 16) = 2` essais de held-out de croissance.
- **physionetmi** : 45 essais. Le pli garde 36 essais d'entraînement, d'où un held-out de 4.

Un critère qui ne tient que là où il a été réglé n'est pas une propriété du protocole.

## 2. Ce qui tourne

- **Bras** : `bd_shallow`, `bd_deep4`, `grow_shallow`, `grow_deep4`.
- **Seeds** : 0, 1, 2.
- **Datasets** : `shin2017a` (29 sujets × 3 sessions × 5 plis = 435 fits par cellule) et
  `physionetmi` (109 × 1 × 5 = 545). Cela fait 24 cellules.
- **Protocole** : celui de l'étape 0, sans aucun changement :
  - config B (Shallow lr 6.25e-4 / wd 0, Deep4 lr 1e-2 / wd 5e-4) ;
  - 200 époques, patience 200 ;
  - sélection `valid_acc` lissée W = 5, sans restauration ;
  - held-out de croissance 10 % ;
  - avenant 2 sur les bras grow : `growth_normalization=match_extending_layer`, grille
    {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64}.
- **Prétraitement** : celui des yaml de dataset, sans changement (2 classes gauche/droite,
  250 Hz ; physionetmi fenêtre [0, 3] s).
- **Matériel** : même pool de cartes RTX 2080 Ti, même garde-fou de carte par nom.
- **Code** : un checkout neuf, figé au commit qui gèle ce PREREG.

## 3. Lecture : `benchmarks/step0bis_criteria.py`, commité avec ce PREREG

Pour chaque couple (bras, dataset), sur les seeds 0 à 2, avec les seuils de l'étape 0 et de
l'avenant 1, sans aucun changement :

| critère | refus si |
|---|---|
| garde-fou `step0_gate.py` | une cellule refusée : elle est relancée et ses critères ne sont pas lus. Le garde-fou vérifie la structure sujets × sessions × plis du dataset, la config, la grille et la normalisation sur 100 % des événements |
| optimum | > 20 % des fits avec `selected_epoch` ≤ 5, ou > 20 % ≥ 195 |
| sélection (bras grow) | tout gardé à chaque pas, ou < 2 candidats proposés par place |
| spectre (bras grow) | < 1 décade (médiane) |
| cible (bras grow) | largeur cible atteinte sur < 80 % des fits |
| calendrier (bras grow) | > 50 % des fits qui ont grandi l'ont fait entièrement avant l'époque 10 |
| censure (bras grow) | s = 64 sur > 20 % des pas appliqués |
| départ | **rapporté seulement** (excès sur ln 2). La limite refusante de l'avenant 1 est relative à une référence 4 classes mesurée sur bnci |

Ne sont **pas** refaits ici :
- la **capacité** et la **référence publiée**, qui sont propres à bnci2014_001 ;
- toute **accuracy** : aucune n'est lue.

## 4. Règle de décision

- **Un couple (bras, dataset) passe** si aucun critère ne le refuse.
- **Un couple refusé** ne peut porter aucun endpoint de l'étape 2 sur ce dataset. Le refus est
  publié tel quel ; aucune correction n'est lancée sans un nouveau PREREG.
- **Si `grow_*` passe sur bnci mais est refusé ici sur la censure**, la grille de l'avenant 2
  est déclarée **spécifique à bnci**. Le résultat de l'avenant 2 reste acquis, mais limité à
  bnci.
- **Si l'optimum refuse un bras `bd_*`**, c'est le choix de l'époque, et non la croissance,
  qui est en cause : le contraste grow contre bd de ce dataset n'est pas lisible à l'étape 2.

## 5. Coût

Il est mesuré en direct (règle `cell-cost-must-be-measured-live`) : les fits sont comptés dans
le log 30 min après le démarrage. `--time` est fixé large (48 h), parce qu'une cellule
requeue repart de zéro : son JSONL partiel est déplacé, jamais complété. physionetmi est lancé
d'abord, car ses données sont en cache. shin2017a suit dès que son téléchargement est vérifié.
