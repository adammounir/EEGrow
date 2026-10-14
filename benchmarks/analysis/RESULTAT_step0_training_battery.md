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

---

# Avenant 1 (lu le 14/10/2026) : les deux bras en croissance sont REFUSÉS, sur la censure seule

Contexte de la lecture :
- PREREG : avenant 1, gelé au commit `aa14de1`.
- Checkout : `/scratch/amounir/eegrow_step0_lot2`.
- Jobs : 755209 (batterie) et 755210 (capacité).
- Sorties : `step0_read/lot2/`.

**Garde-fous** : 6/6 cellules admises, avec 90 fits chacune. `grow_holdout == "disjoint"` sur
100 % des événements, et 0 pic.

| bras | départ (limite) | capacité | optimum ≤5 / ≥195 | sélection : prop./place, tout gardé | spectre | cible | calendrier | **censure (s = 4)** | verdict |
|---|---|---|---|---|---|---|---|---|---|
| grow_shallow | +0.095 (≤ 0.610) | 0.0008 | 0.4 % / 10.7 % | 6.5, 0 % | 4.07 déc. | 100 % | 0 % | **98.8 %** | **REFUS** |
| grow_deep4 | +0.669 (≤ 0.853) | 0.0159 | 1.9 % / 5.6 % | 5.5, 0 % | 3.61 déc. | 98.9 % | 0 % | **81.1 %** | **REFUS** |

Seuil de censure : 20 %. **Tous les autres critères passent** :
- la sélection est réelle (5,5 à 6,5 candidats par place, et jamais tout gardé) ;
- le spectre couvre 3,6 à 4,1 décades ;
- la cible est atteinte sur 99 à 100 % des fits ;
- la croissance est progressive : époque médiane d'un pas appliqué 20 (Shallow) et 25 (Deep4),
  avec 6,4 % et 20,5 % d'abstentions.

Pour la première fois, la croissance de Deep4 est fidèle, sélective et atteint sa cible. Elle
reste refusée parce que la line search ne choisit pas son amplitude.

Conformément au PREREG, **aucun des deux bras ne passe à l'étape 2 sous ce protocole.**

## Pourquoi la censure — analyse EXPLORATOIRE, déclarée après lecture

Elle porte sur les pertes enregistrées à chaque point de la grille (`grow_losses`), pour les
2 037 pas appliqués de Shallow et les 2 229 de Deep4.

| | grow_shallow | grow_deep4 |
|---|---|---|
| perte décroissante sur toute la grille | 96.3 % | 63.7 % |
| gain relatif de perte entre s = 0 et s = 4 (médiane) | **2.1e-4** | **6.8e-6** |
| parabole ajustée sans minimum fini (concave ou linéaire) | 98.8 % | 82.0 % |
| plus grande valeur propre proposée (médiane) | 6.5e-4 | 5.9e-5 |

Sur toute la grille, la perte du held-out est **quasi linéaire en s**, et l'effet est
minuscule. Les neurones ajoutés arrivent avec une amplitude d'ordre √(valeur propre)
(1e-2 à 1e-3). Une échelle de 4 laisse donc le réseau dans le régime linéaire du
développement au premier ordre, où le plus grand `s` gagne par construction. Pour Deep4, 36 %
des pas ne sont pas monotones : à 7e-6 de gain relatif, la line search lit surtout du bruit
de held-out (23 essais).

Ce n'est donc pas une grille « un peu trop courte ». Il manque probablement plusieurs ordres de
grandeur. **Sonde exploratoire lancée** (jobs 755300 et 755301, hors batterie, sans valeur de
critère) :
- les deux bras, sujets 1 à 3, seed 0 ;
- grille {0, 1, 4, 16, 64, 256, 1024, 4096} ;
- même code et mêmes cartes ;
- question posée : un optimum intérieur apparaît-il, et à quelle échelle ?

## Décision à prendre (Adam)

La grille {0 ; 0,25 ; 0,5 ; 1 ; 2 ; 4} a été tranchée par Adam le 13/10. La changer est un
**avenant 2**, sur un nouveau protocole et un nouveau run. Les critères de l'avenant 1 lus ici
restent acquis tels quels. Options, à arbitrer avec le résultat de la sonde :
1. Grille logarithmique sur plusieurs décades, si la sonde montre un optimum intérieur.
2. **Amplitude normalisée** : gromo fournit déjà `normalize_optimal_updates(normalization_type="match_extending_layer")`.
   - Cette fonction remet chaque extension à l'écart-type de la couche qu'elle étend (`growing_module.py`, l. 2920).
   - `eegrow.training.loop.grow_step` ne l'appelle jamais : par défaut, √(valeur singulière) est intégré aux poids (l. 884), d'où l'amplitude ~1e-2.
   - Aucun changement de gromo n'est nécessaire, seulement un appel avant la line search. La grille {0 … 4} prend alors un sens d'échelle relative.
   search, ce qui change gromo et `grow_step`.
3. Accepter `s` au sommet comme une propriété, et retirer le critère de censure. Ce serait
   rejeter un critère après l'avoir lu, ce qui est déconseillé.

## Avenant 2 — LU le 14/10/2026 : les deux bras passent

**Statut de l'avenant : PROPOSÉ par Claude, À RATIFIER PAR ADAM.** Sans ratification, ce qui
suit est exploratoire, et le refus des deux bras sous l'avenant 1 reste acquis.

Jobs 755332 (batterie, 6/6 COMPLETED, 19 à 34 min) et 755333 (capacité), checkout
`/scratch/amounir/eegrow_step0_lot3`, code `e6c836e`. Lecture par `step0_gate.py` puis
`step0_criteria.py` version `35d6234`, figée et poussée avant lecture. Sorties dans
`step0_read/lot3/`.

**Garde-fou** : 6/6 cellules admises, 90 fits chacune, 0 pic. Grille
{0 … 64} et `grow_normalization == "match_extending_layer"` vérifiées sur 100 % des
événements.

| critère | grow_shallow | grow_deep4 |
|---|---|---|
| départ (excès sur ln 4, limite) | +0.103 (≤ 0.610) | +0.646 (≤ 0.853) |
| capacité (médiane de 3 seeds) | 0.0025 | 0.0054 |
| optimum ≤ 5 / ≥ 195 | 1.5 % / 4.8 % | 0.4 % / 4.1 % |
| sélection : proposés par place, tout gardé | 6.5, 0 % | 5.5, 0 % |
| spectre (décades, médiane) | 3.80 | 1.65 |
| cible atteinte | 100 % | 100 % |
| calendrier (tout tôt) | 0 % | 0 % |
| **censure : s au sommet (64), tous sujets** | **0.09 %** | **5.9 %** |
| **censure, sujets 4 à 9 seuls** | **0.14 %** | **5.9 %** |
| verdict | **passe** | **passe** |

Pas appliqués : 2 221 (Shallow) et 2 339 (Deep4).

Le libellé « censure s=4 » de `criteria_growing.md` est un texte resté en dur. Le calcul
utilise bien `grid_top = max(grid) = 64` (`step0_criteria.py`, l. 193).

**Ce que ça dit.**
- La censure tombe de 98.8 % à 0.09 % (Shallow) et de 81.1 % à 5.9 % (Deep4), bien sous le
  seuil de 20 %.
- Elle est la même sur les sujets 4 à 9, qui n'ont pas servi à choisir la grille. Le
  recouvrement avec les sujets 1 à 3 ne porte donc pas le résultat sur ce dataset. La
  reconfirmation sur un autre dataset reste requise avant l'étape 2.
- Aucun autre critère ne se dégrade au-delà de son seuil.
- Seul mouvement notable : le spectre de Deep4 passe de 3.61 à 1.65 décades. Il reste au-dessus
  du seuil d'une décade, mais avec moins de marge. Mécanisme non établi ; pas d'interprétation
  avant une mesure.

**Config gelée, sous réserve de ratification** : config B pour les deux bras, plus
`growth_normalization=match_extending_layer` et la grille {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64}.
