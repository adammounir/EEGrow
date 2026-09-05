# Pré-registration — prévalence du motif « la croissance classe mieux les donneurs »

**Écrite le 2026-09-05, AVANT le premier fit des 11 datasets ajoutés.** Commit qui la
porte : voir `git log -- benchmarks/analysis/PREREG_motif_prevalence.md`. Rien de ce
fichier ne doit être modifié après la soumission du job de chauffe ; toute correction
s'écrit en addendum daté, à la suite.

## 1. D'où vient la question

`why_lee_responds.py` (05/09, post-hoc) a établi que sur `cho2017` l'avantage de
`restored_probe` sur `acc_probe` est de la **robustesse** — il disparaît quand on retire
7 sujets à accuracy invraisemblable — alors que sur `lee2019_mi` il **survit et
grandit**. Ces deux lectures sont incompatibles et une seule est une revendication
scientifique intéressante. À n=6 datasets, rien ne permet de dire laquelle est la
règle et laquelle l'exception : un modérateur de |rho| = 0.7 y serait détecté avec une
puissance ~0.3.

Cette campagne ne cherche donc pas un nouvel effet. Elle **mesure la prévalence d'un
effet déjà observé**, sur un nombre de datasets qui rend la mesure interprétable.

## 2. Ce qui est mesuré, exactement

Pour un dataset `d`, sur la matrice donneur×receveur (`donor_receiver.py`, transfert
réel, pas un proxy) :

* `capture(règle)` = fraction de l'oracle top-5 captée par la règle, moyennée sur les
  receveurs. Dénominateur = ce que le **meilleur** top-5 possible gagne sur l'allocation
  uniforme. La normalisation est indispensable : le gisement varie d'un facteur 1.4
  entre datasets et un gain brut confondrait « bonne règle » et « dataset généreux ».
* `gap(d)      = capture(restored_probe) − capture(acc_probe)`
* `gap_clean(d)` = le même après retrait des sujets à `acc_probe` ≥ 0.90.

**K = 5 et le seuil 0.90 sont repris tels quels de l'analyse post-hoc et ne seront pas
ajustés.** Ce sont des constantes héritées, pas des degrés de liberté : les rebalayer
ici transformerait une réplication en une seconde exploration.

### Taxonomie (figée)

| classe | condition |
|---|---|
| `ABSENT` | `gap ≤ 0` |
| `ROBUSTESSE` | `gap > 0`, ≥ 3 suspects, `gap_clean < gap/2` |
| `MOTIF` | `gap > 0`, ≥ 3 suspects, `gap_clean ≥ gap/2` |
| `MOTIF_NON_TESTE` | `gap > 0`, < 3 suspects — le contrôle aberrants ne peut pas tourner |

La quatrième classe existe parce que la définition a un angle mort qu'il serait
malhonnête de cacher : sur un dataset sans sujet suspect, `gap_clean = gap` par
construction et « le motif survit au contrôle » est une phrase vide. Ces datasets
comptent dans l'endpoint primaire (qui ne porte que sur le signe) et **sont exclus** de
l'endpoint secondaire.

## 3. Le prior, calculé avant d'ajouter quoi que ce soit

Sur les 6 datasets déjà en main (`dxr_*` + `ranking_*`, aucun fit nouveau), avec
**k égalisé à 15** comme le fera l'endpoint primaire :

| dataset | n | k | ICC(1,15) | suspects | gap | gap_clean | classe |
|---|---:|---:|---:|---:|---:|---:|---|
| shin2017a | 29 | 45 | 0.174 | 0 | +0.101 | +0.101 | **exclu G2** |
| cho2017 | 52 | 15 | 0.691 | 7 | +0.330 | +0.076 | ROBUSTESSE |
| dreyer2023a | 60 | 15 | 0.779 | 21 | −0.119 | −0.133 | ABSENT |
| lee2019_mi | 54 | 30 | 0.606 | 5 | +0.082 | +0.358 | MOTIF |
| liu2024 | 50 | 15 | 0.453 | 0 | +0.053 | +0.053 | MOTIF_NON_TESTE |
| physionetmi | 109 | 15 | 0.441 | 1 | −0.189 | −0.164 | ABSENT |

**3/5 admis à écart positif (4/6 sans la porte) — test binomial p = 0.50.** À n=6, la
question posée est donc strictement non résolue, et c'est la justification de la
campagne, pas un préliminaire encourageant.

Ce tableau est le point de comparaison, et il est écrit ici pour qu'on ne puisse pas le
reconstruire après coup dans le sens qui arrange. Les `gap` diffèrent légèrement de ceux
de `why_lee_responds.py` (+0.104 sur lee2019_mi) parce que celui-ci n'égalisait pas k :
lee2019_mi a 30 réplicats et en perd la moitié ici. C'est la correction, pas une dérive.

Il corrige au passage une phrase que j'ai dite à l'oral (« 1 gain réel sur 6 ») : celle-là
portait sur l'endpoint end-to-end `dalloc_restored`, qui n'a tourné que sur 3 datasets.
Les deux mesures sont différentes et **ne doivent pas être mélangées** — voir §7.

## 4. Les 11 datasets ajoutés, et pourquoi ceux-là

Critères appliqués *avant* de regarder le moindre résultat : paradigme
`LeftRightImagery` (homogénéité avec 5 des 6 existants), n ≥ 16 sujets (en dessous, un
top-5 n'est plus un choix), instanciation MOABB 1.5.0 vérifiée.

`stieger2021` (62), `yang2025` (51), `hefmiich2025` (37), `guttmannflury2025_mi` (31),
`guttmannflury2025_me` (31), `chang2025` (28), `forenzo2023` (25), `dreyer2023b` (21),
`zhou2020` (20), `kumar2024` (18), `brandl2020` (16).

Deux d'entre eux ne sont pas là pour le nombre :

* **`dreyer2023b`** est le second cohorte du **même protocole** que `dreyer2023a`, l'un
  des deux datasets où la règle échoue. C'est un contrôle conçu : l'échec se réplique-t-il
  dans un protocole, ou `dreyer2023a` était-il un tirage ?
* **`guttmannflury2025_me`** est le jumeau en mouvement **exécuté** de
  `guttmannflury2025_mi` — mêmes 31 sujets, même matériel. C'est le seul contraste
  intra-cohorte de la campagne : il isole ce que l'imagerie coûte.

## 5. Porte d'admission — elle REFUSE, elle n'avertit pas

Un dataset entre dans l'analyse si et seulement si :

* **G1** ≥ 16 sujets ayant ≥ 15 réplicats de sonde complets ;
* **G2** ICC(1,15) de `restored_probe` > 0.30, c'est-à-dire la fiabilité de la **moyenne
  de 15 réplicats** — la quantité que la règle utilise réellement — obtenue par
  Spearman-Brown à partir de l'ICC(1,1). C'est la leçon `schirrmeister2017` : quand tout
  le monde s'arrête à la même taille, il n'y a pas de variance inter-sujets et la règle
  ne peut rien classer ; un `gap ≈ 0` s'y lit comme une absence d'effet alors que c'est
  une absence de mesure ;

  > **Correction datée du 05/09/2026, avant toute soumission.** G2 était d'abord écrite
  > « ICC(1,1) > 0.05 ». Passée sur les 6 datasets existants, cette version excluait 4
  > d'entre eux — dont **les deux seuls où la règle échoue** (`dreyer2023a`,
  > `physionetmi`). Une porte qui retire précisément les négatifs biaise la prévalence
  > vers le haut, et elle l'aurait fait sans rien signaler. Deux défauts s'additionnaient :
  > le seuil portait sur la fiabilité d'un réplicat isolé alors que la règle emploie une
  > moyenne de 15 (`icc-single-vs-mean`), et deux exports de sonde étaient mal routés,
  > rendant un ICC `nan` lu comme un échec. La version corrigée n'exclut que
  > `shin2017a` (ICC(1,15) = 0.174).
  >
  > Je consigne la correction plutôt que de réécrire le seuil en silence, parce qu'elle
  > a été faite **après** avoir vu qui elle excluait. La garantie n'est donc pas ma
  > bonne foi : c'est que **l'endpoint primaire est systématiquement rapporté deux fois,
  > avec et sans la porte**. Si les deux divergent en signe, le script l'écrit en
  > toutes lettres et les deux vont dans le papier. Le seuil ne peut plus régler la
  > réponse. Précédent : `prereg-gate-one-sided`, où une porte unilatérale avait dû
  > être remplacée pour la même raison ;
* **G3** matrice D×R complète : chaque sujet comme donneur × 3 seeds, et chaque case
  (donneur, receveur) présente.

G1 et G2 se lisent **sur la sonde seule**, donc avant d'avoir vu le moindre `gap`. Le
script de lecture lève `SystemExit` si un dataset déclaré échoue G3 : une matrice
partielle doit faire échouer la lecture, pas y entrer avec des receveurs manquants
(mode de défaillance de `gates-must-refuse-not-warn`).

## 6. Endpoints

**Primaire — prévalence du signe.** Nombre de datasets admis avec `gap > 0`, sur
l'ensemble des 6 existants + les nouveaux admis. Test binomial exact unilatéral contre
p₀ = 0.5 (sous « aucune différence systématique entre les deux règles », le signe est
symétrique). IC de Clopper-Pearson à 95 %. **Unité d'analyse = le dataset** — c'est la
question posée ; à l'intérieur d'un dataset, `capture` est déjà moyennée sur les
receveurs.

**Rapporté deux fois, toujours : avec la porte, et sur tous les datasets à matrice
complète sans la porte.** Voir la correction datée en §5 — c'est ce doublon qui empêche
le seuil de G2 de régler la conclusion.

**Secondaire A — magnitude.** Moyenne de `gap` sur les datasets admis, IC bootstrap
20 000 **sur les datasets**.

**Secondaire B — le motif contre la robustesse.** Parmi les seuls datasets ayant ≥ 3
suspects : proportion `MOTIF` / (`MOTIF` + `ROBUSTESSE`). Descriptif et rapporté avec
son n, sans test : le prior y est 1/2 et aucun n atteignable ici ne le tranchera.

**Réplicats égalisés.** La fiabilité pilote la capture (établi : Spearman(ICC, capture)
= +1.00 / +1.00 / +0.886) et les datasets ajoutés ont de 1 à 11 sessions, donc k varie
de 15 à 165. **L'endpoint primaire est calculé sur `restored_probe` sous-échantillonné à
k = 15 réplicats par sujet** (moyenne sur 30 tirages), qui est le k du plus pauvre des
datasets existants. La version à k complet est rapportée en secondaire. Sans cette
égalisation, « les nouveaux datasets répondent mieux » pourrait n'être que « ils ont plus
de fits ». Toujours des ICC(1,1), jamais des ICC(1,k).

## 7. Ce que cette campagne ne mesure PAS

Elle mesure un **classement de donneurs sur du transfert réel**. Elle ne mesure pas le
gain end-to-end d'une allocation de budget : `dalloc_restored` a montré que le lien
entre les deux est faible (3 signes prédits sur 3, mais P(les 3 vrais signes sont ceux
prédits) = 0.751). Aucune phrase du rapport final ne doit convertir une prévalence
mesurée ici en une prévalence de gain end-to-end. La validation end-to-end reste à n=3
et cette campagne ne la déplace pas.

## 8. Modérateurs — liste gelée ici, Holm sur les 11

Corrélation de Spearman entre `gap` et chacun de : n_sujets, n_canaux, longueur
d'époque (s), essais/sujet, `acc_probe` moyen, SD de `acc_probe`, ICC(1,1) de
`restored_probe`, ICC(1,1) de `acc_probe`, Spearman(`acc_probe`, `restored_probe`),
fraction de sujets suspects, gisement de l'oracle. **Aucun autre ne sera testé**, et
aucun ne sera retiré de la liste après coup.

C'est exploratoire et rapporté comme tel. À n = 17 datasets et α corrigé Holm ≈ 0.0045,
la puissance est ~0.70 pour |rho| = 0.7 et ~0.35 pour |rho| = 0.5. Un nul ici ne
prouvera rien ; un positif sera une piste, pas un résultat.

## 9. Ce qui falsifierait

* Une prévalence du signe qui ne se distingue pas de 0.5 → la règle de croissance ne
  classe pas les donneurs mieux que l'accuracy en général, et `lee2019_mi` était un
  tirage. C'est un résultat publiable et il ferme l'angle.
* `dreyer2023b` positif alors que `dreyer2023a` est franchement négatif → la classe
  n'est pas portée par le protocole, et le modérateur, s'il existe, est ailleurs.
* `guttmannflury2025_me` (exécuté) et `_mi` (imaginé) qui divergent → le motif tient à
  la difficulté de la tâche plus qu'au sujet.

## 10. Puissance, dite avant

Sign test à n = 17, α = 0.05 unilatéral : il faut ≥ 13 positifs. Si la vraie prévalence
est 0.75, la puissance est ~0.60 ; si elle est 0.85, ~0.90. **À prévalence vraie 0.65 la
campagne est sous-puissante** et un nul devra être rapporté comme « non résolu », avec
la MDE, jamais comme « pas d'effet » (`underpowered-not-null`).
