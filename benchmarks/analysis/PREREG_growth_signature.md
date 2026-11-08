# PREREG — La signature du sujet dans l'adaptateur (lecture 4, axe 2)

08/11/2026. **GELÉ** dans le même commit que `PREREG_growth_adaptive_size.md` (axe 1) et
`adapt_C.py`, avant toute adaptation hors du dataset de réglage. Il réutilise les
adaptations de l'axe 1 : seule une passe d'évaluation s'ajoute.

## 1. Question

La lecture 1 a montré que les neurones ajoutés pendant l'entraînement identifient le sujet,
mais autant avec une direction aléatoire, et que la covariance du signal l'identifie trois
fois mieux (top-1 0,95 sur lee2019_mi). À ce plafond, « au-delà de la covariance » ne se
mesure pas par l'identification. On pose donc la question autrement, sans plafond :

**l'adaptateur appris sur une session d'un sujet porte-t-il quelque chose de propre à ce
sujet, qui sert encore sur son autre session, au-delà de ce que recalibre la normalisation
(AdaBN) ?** Et la croissance en porte-t-elle plus que LoRA ?

Test : sur la session B du sujet i, on compare son propre adaptateur (appris sur sa session
A) à l'adaptateur d'un autre sujet j du même dataset (appris sur la session A de j). Dans les
deux cas, AdaBN est recalculée sur la calibration de i : seule l'information des poids de
l'adaptateur diffère. Si son propre adaptateur gagne, l'adaptateur contient une signature du
sujet, stable d'une session à l'autre, que la recalibration des statistiques ne donne pas.

## 2. Données

Les datasets à 2 sessions de l'axe 1 : bnci2014_004, bnci2015_001, lee2019_mi, shin2017a,
zhou2016 (calibration dans une session, test sur l'autre, les deux sens). Mêmes modèles (4,
CTNet exclu), mêmes tirages, k ∈ {10, 50} (shin2017a n'a pas k = 50). Adaptateurs gardés par
l'axe 1 : grow2, grow8, lora2, lora8.

**Partenaires** : pour chaque sujet, les 4 sujets suivants dans un ordre aléatoire fixé avant
tout calcul (`default_rng(0)`), jamais lui-même (zhou2016 : les 3 autres). Le score « autre »
d'un tirage est la moyenne des partenaires, chacun au même (k, sens, tirage).

## 3. Garde-fou G-sig

La passe signature réapplique aussi le propre adaptateur de i. Son score doit redonner celui
de l'axe 1 à 5e-3 près. Si plus de 5 % des réapplications s'en écartent, l'axe 2 est refusé
entier (rien n'est rapporté). Les sujets exclus de l'axe 1 (G0, refus) le sont ici aussi.

## 4. Critères

Différences appariées par sujet, moyennées sur ses tirages puis sur les 4 modèles, groupées
sur les 5 datasets (√n_d, bootstrap stratifié 10 000, `default_rng(0)`, comme l'axe 1).

**Principal**, Holm sur 4 (2 critères × 2 k), α = 0,025 unilatéral :

| critère | contraste |
|---|---|
| **T1** | croissance b = 8 : propre adaptateur − adaptateur d'un autre |
| **T2** | T1 croissance − T1 LoRA, à b = 8 |

b = 8 : le plus grand budget gardé, celui qui a le plus de place pour une signature.
Verdicts : SUPÉRIEUR si Holm rejette ; INFÉRIEUR si borne haute < 0 ; sinon NON SIGNIFICATIF
(pas de marge de non-infériorité : la question est l'existence d'un effet).

Lecture :
- T1 SUP → l'adaptateur par croissance porte une signature du sujet, stable d'une session à
  l'autre, au-delà d'AdaBN ;
- T2 SUP → la croissance en porte plus que LoRA ; T2 INF → LoRA en porte plus ; sinon pas de
  différence mesurable (MDE rapportée).

## 5. Rapporté

T1 à b = 2 et pour LoRA (b = 2, 8) ; T1 croissance − LoRA à b = 2 ; T1 croissance b = 8 par
modèle.

## 6. Ce qui ne sera pas fait

Pas d'identification par la covariance ou par les poids (plafond connu, lecture 1). Aucun
critère, budget, partenaire ou k changé après lecture.

## 7. Limites connues avant lecture

- « Au-delà d'AdaBN » et non « au-delà de la covariance complète » : AdaBN recale moyenne et
  variance de chaque canal des couches normalisées, pas la covariance entre électrodes.
  Shallow n'a qu'une BatchNorm, les transformers en ont dans leur tête convolutive.
- Session et montage du casque restent confondus (lecture 1, § 9) : la « signature » peut
  inclure le placement du casque propre au sujet.
- 5 datasets, dont deux petits (zhou2016 : 4 sujets, bnci2014_004 : 9) ; lee2019_mi
  (54) et shin2017a (29) pèsent le plus.
