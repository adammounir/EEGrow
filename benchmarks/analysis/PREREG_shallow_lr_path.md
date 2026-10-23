# PREREG — L'avantage de la croissance sur Shallow survit-il au learning rate ?

23/10/2026. Gelé avant le lancement (commit qui contient ce fichier, avec
`lrshallow_make_cells.py` et `lrshallow_read.py`). Aucun résultat de cette campagne
n'est lu avant ce commit.

## 1. Question

Dans la validation (`read/valid.md`), grow_shallow (règle tiny) finit **exactement** à
l'architecture de bd_shallow : 8 → 40 filtres, 100 % des fits à la cible, même nombre
de paramètres. Il la bat pourtant sur les 3 datasets :
- bnci2014_001 : +0.024 [+0.014, +0.033] ;
- cho2017 : +0.033 [+0.022, +0.044] ;
- lee2019_mi : +0.059 [+0.044, +0.075].

Les EEGNeX, TIDNet et MSVTNet qui ont grandi, eux aussi à leur architecture finale
d'origine, ne font pas mieux que leur jumeau.

Adam, 23/10 : le contrôle de largeur doit dire si le **chemin** de croissance apporte
quelque chose au-delà de la taille finale. Pour Shallow, ce contrôle existe déjà, et il
dit oui.

Mais il y a une explication concurrente. bd_shallow pourrait être **sous-optimisé** au
lr 6.25e-4 (la recette braindecode de Shallow, config B de l'étape 0), sous notre
protocole sans schedule. Ce serait le même piège que Deep4 le 14/10, où le lr trop petit
était la cause, et la croissance ne ferait que compenser.

## 2. Plan

- **Ce qui varie** : seulement `train.lr`, sur les **deux** bras (bd_shallow et
  grow_shallow tiny), parmi {2e-4, 2e-3, 6.25e-3, 2e-2}. Avec 6.25e-4, cela donne une
  grille d'environ un demi-ordre de grandeur, de part et d'autre de la recette. La
  grille est un choix eegrow et ne vient d'aucun papier.
- **Ce qui reste identique à la validation**, ligne pour ligne : AdamW, wd 0, batch 64,
  la règle tiny (E.3.1, s = 1, plancher 0.1), le protocole (200 époques, sans arrêt
  anticipé, score = dernière époque, croissance toutes les 5 époques, held-out de 10 %),
  RTX 2080 Ti, checkout 35b6f09.
- **lr 6.25e-4** : on réutilise les cellules de la validation.
- **Datasets** : bnci2014_001 (graines 0-2), cho2017 (graine 0), lee2019_mi (graine 0).
- **40 cellules** = 4 lr × 2 bras × 5 dataset-graines (`lrshallow_cells.txt`).

## 3. Mesures

Unité = le sujet, contrastes appariés par sujet, IC95 par bootstrap percentile
(10 000 tirages, `default_rng(0)`).

## 4. Critère principal

Par dataset, **lr\*** est le lr où bd_shallow a le meilleur score moyen. C'est une
sélection sur le score de test, à l'avantage de bd : le contraste est donc
**conservateur** contre la croissance.

On mesure C = tiny(lr\*) − bd(lr\*), apparié par sujet :
- **SURVIT** si la borne basse de l'IC est > 0 sur les 3 datasets ;
- **DISPARAÎT** si la borne basse est ≤ 0 sur les 3 ;
- **PARTIEL** sinon.

Rapporté, sans valeur décisionnelle :
- bd(lr\*) − bd(6.25e-4), qui dit si bd était sous-optimisé ;
- la table lr × bras ;
- le meilleur tiny contre le meilleur bd ;
- la stabilité et la fraction des fits à la cible.

## 5. Interprétation (avec Adam)

- **SURVIT** : à architecture finale identique, et contre le meilleur lr de bd, la
  croissance donne un meilleur Shallow. C'est un effet du chemin, le cas B.
  L'étape suivante est d'en chercher le mécanisme (initialisation, échelle des poids,
  curriculum de capacité).
- **DISPARAÎT** : l'avantage était un artefact de recette (bd sous-optimisé). Shallow
  rejoint EEGNeX, TIDNet et MSVTNet dans le cas A, et la croissance n'a pas d'effet de
  chemin mesurable.
- **PARTIEL** : rapporté par dataset, sans conclusion générale.

## 6. Ce qui ne sera pas fait

- Aucun autre réglage ne change.
- Aucune relance « pour voir », sauf un échec technique.
- Aucun lr ajouté après lecture.
