# PREREG — 200 époques brident-elles MSVTNet ?

24/10/2026. Gelé avant le lancement (commit qui contient ce fichier, `msvt400_read.py`
et `msvt400_cells.txt`).

## 1. Question

Adam, le 24/10 : est-ce que 200 époques brident l'entraînement ?

Ce que disent les courbes de la validation (`read/seeds_epochs`) :
- **5 modèles sur 6** ont une perte d'entraînement finale ≤ 0.016 et un score plat entre
  l'époque 100 et l'époque 200. Ils ne sont pas bridés.
- **MSVTNet** a une perte finale de 0.14, encore en baisse de 9 à 11 % sur les 20
  dernières époques, et une validation qui monte encore (+0.025 entre l'époque 100 et
  l'époque 200).

## 2. Plan

On reprend les cellules `msvtnet__bd` et `msvtnet__tiny` de la validation, **ligne pour
ligne**. Les seules différences sont `MAX_EPOCHS=400` (avec patience = 400) et le
label `_e400`.

| Élément | Valeur |
|---|---|
| Datasets | bnci2014_001 (graines 0-2), cho2017 (0), lee2019_mi (0) |
| Cellules | 10 |
| Carte | RTX 2080 Ti, comme la validation |
| Checkout | eegrow_valid (35b6f09) |

**Confusion connue pour le bras tiny :** la croissance continue d'avoir lieu toutes les
5 époques, donc on obtient 79 événements au lieu de 39. Le bras tiny à 400 époques peut
donc aussi être plus large. On rapporte sa largeur, et on lit le bras **bd** comme
critère principal.

## 3. Mesures

Unité = le sujet, apparié par sujet, avec un IC95 bootstrap identique à valid_read.

- Δ_bd = bd(400) − bd(200)
- Δ_tiny = tiny(400) − tiny(200)

Rapporté :
- tiny − bd à 400 contre tiny − bd à 200 ;
- la largeur du bras tiny à 400.

## 4. Critère

- **BRIDÉ** si la borne basse de Δ_bd est > 0 sur au moins 2 des 3 datasets. Dans ce
  cas, MSVTNet passe à 400 époques dans la campagne A, sur ses deux bras.
- **NON BRIDÉ** sinon. MSVTNet reste à 200 époques, comme les autres modèles.

Le critère ne porte que sur MSVTNet. Les 5 autres modèles ne sont pas retestés : leur
perte d'entraînement est déjà à environ 0.
