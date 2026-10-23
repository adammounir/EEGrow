# PREREG — La croissance des transformers est-elle censurée par le plancher relatif ?

23/10/2026. Gelé avant le lancement (commit qui contient ce fichier, avec
`ratio_make_cells.py` et `ratio_read.py`). Aucun résultat de cette campagne n'est lu
avant ce commit.

## 1. Question

La validation du 21-23/10 (`read/valid.md`) a validé la règle tiny sur les 6 modèles.
Mais avec cette règle :
- **Conformer** n'atteint jamais sa largeur cible : ses FFN passent de 96 à environ 200
  neurones sur 960.
- **CTNet** non plus : environ 608 sur 960.
- **MSVTNet** reste sous la cible dans 44 à 69 % des fits sur cho2017 et lee2019.

La mesure (`growdyn.py`) l'explique :
- Il n'y a pas d'early stopping : patience = 200 = max_epochs.
- Les 39 événements de croissance sont tous appliqués, et la croissance est encore
  active au dernier, à l'époque 195.
- Le plancher relatif `min_singular_ratio = 0.1` ne garde que 2 à 3 neurones sur 40
  par événement sur Conformer, et 11 à 16 sur CTNet.

Adam, 23/10 : « est-ce que les réglages qu'on choisit ne risquent pas de censurer
certains modèles qui auraient pu encore grandir et obtenir de meilleures
performances ? » Plusieurs seuils demandés.

## 2. Plan

- **Ce qui varie** : seulement `model.min_singular_ratio` ∈ {0.03, 0.01, 0.001, 0}.
  À 0, on garde tous les neurones proposés par gromo, dans la limite de la cible.
- **Ce qui reste identique à la cellule tiny de la validation**, octet pour octet :
  la recette, la règle tiny (amplitude E.3.1, s = 1), le protocole (200 époques, sans
  arrêt anticipé, score = dernière époque, croissance toutes les 5 époques, held-out
  de 10 %), RTX 2080 Ti seulement, le même checkout et le même commit (35b6f09 pour le
  code exécuté).
- **Références réutilisées** de la validation : `bd` et `tiny` (r = 0.1). Le code, le
  protocole et la carte sont les mêmes, donc rien n'est à recalculer.
- **Modèles** : Conformer-all, CTNet-all, MSVTNet-all.
- **Datasets** : bnci2014_001 (n = 9, graines 0-2) et lee2019_mi (n = 54, graine 0, là
  où est la puissance).
- **48 cellules** = 3 modèles × 4 seuils × (3 + 1) dataset-graines
  (`ratio_cells.txt`).

## 3. Mesures

Unité = le sujet, contrastes appariés par sujet, IC95 par bootstrap percentile sur les
sujets (10 000 tirages, `default_rng(0)`), comme pour la validation.

Pour chaque (modèle, r, dataset), on rapporte :
- r − r0.1 ;
- r − bd, avec la non-infériorité à 0.02 ;
- la fraction des fits à la cible et la largeur médiane ;
- la stabilité (pertes finies, pic ≤ 1e6).

## 4. Critère principal

Pour chaque modèle, on compare **r = 0 à r = 0.1**. r = 0 est la dose maximale : c'est
elle qui a le plus de chances d'atteindre la cible.
- **GAIN** si la borne basse de l'IC est > 0 sur les 2 datasets (intersection-union,
  sans correction).
- **PERTE** si la borne haute de l'IC est < 0 sur au moins un dataset.
- **PAS DE GAIN DÉTECTÉ** sinon. Ce n'est pas une preuve d'équivalence : on rapporte
  l'IC.

Les seuils 0.03, 0.01 et 0.001 forment une courbe dose-réponse **exploratoire**.

## 5. Décision (avec Adam)

- **GAIN** sur un modèle : le réglage 0.1 censure ce modèle. On change son seuil avant
  l'étape suivante, et le choix du seuil se discute avec la courbe sous les yeux.
- **PAS DE GAIN DÉTECTÉ** ou **PERTE** : on garde 0.1. La croissance partielle n'est
  pas un manque mesurable, on le rapporte, et on passe à l'étape suivante.
- **Contrôle de manipulation** : si r = 0 n'amène pas la majorité des fits à la cible
  sur un modèle, la question n'est pas tranchée pour lui. Le budget d'événements
  (grow_every) est alors l'autre levier, à décider ensemble.

## 6. Ce qui ne sera pas fait

- Aucun autre réglage ne change.
- Aucune relance « pour voir », sauf un échec technique, qui sera rapporté.
- Le seuil n'est pas choisi sur le meilleur score d'une seule cellule.
