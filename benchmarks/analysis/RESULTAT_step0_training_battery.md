# Résultat — Étape 0, lot 1 : les deux références passent, config gelée B

Lu le 14/10/2026, selon `PREREG_step0_training_battery.md`, gelé au commit `50b68c7`.
- Lecture : `step0_criteria.py`, commité avant la lecture (`38e94a2`).
- Jobs : 754941 (batterie) et 754942 (capacité et référence).
- Cartes : RTX 2080 Ti uniquement.
- Sorties brutes : `step0_read/`.

## Garde-fous

Les **12 cellules sont admises**, avec 90 fits chacune :
- l'estampille de protocole est exacte sur 100 % des records ;
- les `cv_ind` 0…89 sont complets, sans aucun doublon ;
- **aucun pic** (`grad_norm_max_fit` ≤ 1e6 partout), y compris pour Deep4 à lr 1e-2.

Aucune relance n'a été nécessaire.

## Critères

| bras | config | départ (excès, rapporté) | capacité (méd. 3 seeds) | optimum (≤5 / ≥195) | verdict |
|---|---|---|---|---|---|
| bd_shallow | **B** (lr 6.25e-4, wd 0) | +0.460 | 0.0007 | 1.5 % / 11.5 % | **passe** |
| bd_shallow | H (lr 6.25e-4, wd 0.01) | +0.460 | 0.0007 | 1.5 % / 11.5 % | passe |
| bd_deep4 | **B** (lr 1e-2, wd 5e-4) | +0.703 | 0.0066 (0.0008 / 0.0399 / 0.0066) | 0.4 % / 5.9 % | **passe** |
| bd_deep4 | H (lr 6.25e-4, wd 0.01) | +0.703 | **1.773** (1.549 / 2.558 / 1.773) | 3.3 % / 14.1 % | **REFUS** (capacité) |

**Référence publiée** (protocole des auteurs, cropped, 4–38 Hz, 9 sujets × 3 seeds) :

| modèle | notre moyenne | IC bootstrap 95 % | publié | verdict |
|---|---|---|---|---|
| Shallow | **73.8 %** | [63.1, 83.4] | 71.9 % | **passe** |
| Deep | **68.4 %** | [56.0, 79.5] | 70.1 % | **passe** |

Par sujet (moyenne des 3 seeds), Shallow / Deep :

| sujet | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Shallow | .854 | .479 | .889 | .743 | .565 | .538 | .943 | .818 | .808 |
| Deep | .836 | .497 | .856 | .668 | .365 | .476 | .913 | .763 | .782 |

Les deux moyennes tombent à moins de 2 points du chiffre publié. La largeur de l'IC (±10 points)
rappelle que le critère ne détecte qu'un écart grossier, comme annoncé dans le PREREG. Ce qu'il
établit, c'est que les implémentations braindecode 1.5.2 s'entraînent comme en 2017 sous le
protocole d'origine.

## Config gelée (règle § 3.2 : B si B passe tout)

- `bd_shallow` → **B** : AdamW, lr 6.25e-4, weight decay 0.
- `bd_deep4` → **B** : AdamW, lr 1e-2, weight decay 5e-4.

Commun aux deux : batch 64, lr constant, 200 époques, dernière époque scorée.

## Ce que ça dit, au-delà du refus

1. **La config historique de Deep4 est refusée.** Elle a servi à toutes les campagnes, de v5 au
   budget, ainsi qu'à `grow_deep` vs `bd_deep4`. Sous elle, Deep4 ne parvient pas à mémoriser
   32 essais en 200 pas pleins : il finit au-dessus de ln 4 sur les 3 seeds. Avec la recette
   braindecode, il y parvient. Le diagnostic de la docstring de `exp_deep4_lr.py` (« bd_deep4
   cannot fit its own training data ») était donc juste sur le symptôme. En revanche, ses
   remèdes testés, un lr divisé par 10 et un cosinus, partaient dans la mauvaise direction :
   Deep4 a besoin d'un lr **16 fois plus grand**. Tout contraste historique contre `bd_deep4`
   mesurait une référence sous-entraînée.
2. **Shallow est insensible au weight decay** à ce lr. B et H diffèrent à la 4e décimale de la
   perte, et 79/90 fits de la seed 0 choisissent la même époque. L'erreur du brouillon
   (wd « 0 » au lieu de 0.01) n'a donc pas eu d'effet sur les résultats Shallow historiques.
3. **Excès au départ** : +0.46 (Shallow) et +0.70 (Deep4), stables et non refusants. C'est la
   valeur de référence de famille pour les modèles en croissance de l'étape 1 (§ 5.3). Le
   `GrowingDeep4Net` devra donc rester sous +0.85.
4. **Bord fin** : à la dernière époque scorée, 5.9 % (Deep4 B) et 11.5 % (Shallow) des fits ont
   leur meilleure époque lissée à ≥ 195. Ces valeurs restent sous le seuil de 20 %, mais
   montrent que Shallow progresse encore en fin de budget sur une partie des folds.

## Suite

- L'étape 1 se fait sous la config B, par avenant au PREREG.
  - `GrowingDeep4Net` est prêt (`e78fb4d`, parité ≤ 1e-5). Il hérite de lr 1e-2 et wd 5e-4.
    Ce n'est pas sans enjeu pour la croissance : `GromoGrowth` reconstruit l'optimiseur à
    chaque événement, donc la recette est à vérifier dans la batterie.
  - `GrowingShallowFBCSPNet` hérite de lr 6.25e-4 et wd 0.
- Les résultats historiques qui comparent quoi que ce soit à `bd_deep4` sont à requalifier
  (point 1).
