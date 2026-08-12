# Résultats de la grille MOABB — ML classique vs braindecode vs croissance

Grille terminée le 2026-08-10. Ce document est la lecture des résultats ; les chiffres
eux-mêmes sont dans `results_published/`, et tout ce qui suit s'en déduit avec
`aggregate_published.py`.

## Ce qui a tourné

| | |
|---|---|
| datasets | 12 (motor imagery MOABB, positions d'électrodes connues) |
| protocoles | `within_session` (12 datasets), `cross_session` (6, ceux qui ont ≥ 2 sessions), `cross_subject` (12) |
| pipelines | 6 Riemann/CSP, 4 braindecode fixes, 4 équivalents croissants |
| graines | 5 par cellule |
| runs | **2100** (14 × 5 × 30 cellules protocole×dataset), aucune cellule manquante |
| scores | **189 062** lignes (une par sujet × session × graine × modèle) |
| échantillonnage | 250 Hz partout, fenêtre imposée par MOABB par dataset |

Les paires comparées sont `grow_shallow`/`bd_shallow`, `grow_sccnet`/`bd_sccnet`,
`grow_eegnex`/`bd_eegnex`, `grow_deep`/`bd_deep4` : même architecture cible, même
prétraitement, la seule différence étant que le bras croissant démarre étroit et
s'élargit pendant l'entraînement.

**Convention statistique.** Les 5 graines sont des répétitions d'une même mesure, pas
5 échantillons indépendants. Elles sont donc moyennées *à l'intérieur* de chaque
(sujet, session) avant toute statistique ; une observation = un sujet/session tenu à
l'écart. Sans cela le n effectif est multiplié par 5 et tous les p sont faux. La
métrique suit MOABB : ROC-AUC sur les datasets à deux classes `LeftRightImagery`
(BNCI2014-004, Cho2017, Lee2019-MI, PhysionetMI, Shin2017A, Weibo2014), accuracy
ailleurs — les scores ne sont jamais mélangés entre les deux.

## 1. Le ML riemannien reste devant, sauf en cross-subject

Moyenne des niveaux absolus par famille, sur les 6 datasets à AUC (les seuls
comparables entre eux) :

| famille | `within_session` | `cross_session` | `cross_subject` |
|---|---|---|---|
| Riemann/CSP | **0.7135** | **0.7603** | 0.7347 |
| braindecode (fixe) | 0.6296 | 0.6487 | **0.7837** |
| croissant | 0.6139 | 0.6438 | 0.7636 |

Le sens de ce résultat est le régime de données. En `within_session` et
`cross_session` l'entraînement porte sur quelques centaines d'essais d'un seul sujet :
un classifieur riemannien, dont le biais inductif encode déjà la structure de
covariance spatiale du signal, y bat un réseau qui doit l'apprendre. En
`cross_subject` le réseau voit des dizaines de sujets, et l'ordre s'inverse — c'est là,
et seulement là, que le deep learning paye. En `within_session`, le meilleur modèle
toutes familles confondues est riemannien sur 10 des 12 datasets, et c'est `ts_lr`
(tangent space + régression logistique) sur 8 d'entre eux.

## 2. La croissance : un seul des quatre couples gagne, et modestement

Test de signe sur les deltas (croissant − fixe) au niveau (protocole, dataset) :

| paire | cellules positives | delta médian | p (signe) |
|---|---|---|---|
| `grow_shallow` vs `bd_shallow` | **24/27** | **+0.0066** | 4.9 × 10⁻⁵ |
| `grow_sccnet` vs `bd_sccnet` | 11/27 | −0.0066 | 0.44 |
| `grow_deep` vs `bd_deep4` | 6/27 | −0.0305 | 5.9 × 10⁻³ |
| `grow_eegnex` vs `bd_eegnex` | 2/27 | −0.0322 | 5.7 × 10⁻⁶ |

(27 et non 30 : voir §3, trois cellules sont au hasard et sont écartées. Avec elles :
27/30, +0.0065, p = 8.4 × 10⁻⁶ — la conclusion ne dépend pas de ce choix.)

Donc l'effet est **réel, systématique, et petit** sur ShallowFBCSPNet : +0.7 point
médian, mais dans la même direction presque partout, ce qui est ce que le test de signe
mesure. Les cellules les plus nettes sont `lee2019_mi/within_session` (+0.0307 AUC,
n = 108, p < 10⁻⁴), `lee2019_mi/cross_session` (+0.0289), `cho2017/within_session`
(+0.0256, n = 52).

Sur les trois autres architectures la référence fixe est meilleure, et pour EEGNeX
c'est franc (2 cellules sur 27). L'interprétation la plus économique : la croissance
aide quand la capacité est le facteur limitant et que le réseau est petit ; sur une
architecture déjà large, démarrer étroit ne fait que perdre des époques.

**Un piège à ne pas rapporter à l'envers.** Les plus gros deltas positifs de la grille
appartiennent à `grow_deep` et ne sont pas des victoires de la croissance :

| cellule | croissant | fixe | hasard |
|---|---|---|---|
| `bnci2014_001` / `within_session` | 0.4080 | 0.2748 | 0.2500 |
| `bnci2014_001` / `cross_session` | 0.4192 | 0.3028 | 0.2500 |

`bd_deep4` y est à 2.5 points du hasard, c'est-à-dire qu'il ne s'entraîne pas. Un delta
de +0.13 contre une référence effondrée dit que Deep4 échoue sur 9 sujets, pas que la
croissance apporte 13 points. C'est pour cette raison que les niveaux absolus et
l'écart au hasard sont dans `eegrow_benchmark_levels.csv` à côté des deltas.

## 3. Trois cellules où aucun réseau n'apprend

Sur `physionetmi/within_session`, `shin2017a/within_session` et
`shin2017a/cross_session`, **les 8 modèles profonds** (4 fixes + 4 croissants) sont à
l'AUC 0.49–0.51, soit exactement le hasard. Les baselines riemanniennes, elles,
décollent (`fgmdm` à 0.681 sur `physionetmi/within_session`). Ce n'est donc pas un
dataset impossible mais un régime où les réseaux n'ont pas assez d'essais par session.
Tout delta apparié calculé dans ces cellules est du bruit entre deux tirages à pile ou
face, et c'est pourquoi le tableau du §2 les écarte.

## 4. Provenance : ce qui est prouvable et ce qui ne l'est plus

Il faut le dire avant que quiconque cite ces chiffres.

Une paire ne mesure la croissance que si ses deux bras ont vu le même prétraitement.
Ce n'était pas garanti : le lancement de production passait `dataset.resample=250` en
ligne de commande, certains scripts de relance l'omettaient, et une cellule relancée
retombait silencieusement sur le taux natif du dataset (500 Hz sur Schirrmeister2017,
1000 Hz sur Lee2019-MI). `regime_guard.py` est le garde-fou contre ça, et la campagne
`fix250` (116 cellules, `slurm/fix250_*.txt`) a été la remédiation.

Le garde-fou lisait sa preuve dans les enregistrements hydra — et ceux-ci **n'existent
plus** : le `rsync --delete` qui a effacé le cache d'épochs a aussi emporté
`benchmarks/multirun/` et `benchmarks/outputs/`. Il ne reste que les logs slurm et les
fichiers de résultats. `provenance_audit.py` fait le compte exact (bras brut,
2100 cellules) :

| | cellules | |
|---|---|---|
| taux natif déjà à 250 Hz → `resample` sans effet, immunes par construction | 630 | 30.0 % |
| certifiées à 250 Hz par un log survivant (dont 27 déjà comptées ci-dessus) | 327 | — |
| **établi, union des deux** | **930** | **44.3 %** |
| aucune trace, dans aucun sens | 1170 | 55.7 % |

Deux points positifs dans ce tableau. Aucune cellule n'est certifiée à un taux autre
que 250 Hz — la preuve, là où elle subsiste, va toujours dans le même sens. Et les
**87** cellules dont on peut établir qu'elles ont tourné à un taux natif à un moment
sont **toutes les 87** certifiées à 250 Hz aujourd'hui : la campagne de relance a bien
réécrit les octets sur le disque. Lee2019-MI, le dataset le plus exposé (facteur 4),
est certifié sur ses 210 cellules.

Ce qui reste est donc une ignorance, pas une contamination connue — mais c'est bien une
ignorance, sur 56 % des cellules, et elle ne se comblera pas par de l'analyse. Le
garde-fou, lui, refuse ces cellules (`UNKNOWN` ≠ 250) : les analyses ci-dessus ont été
produites avec `EEGROW_ALLOW_MIXED=1`, ce qui est une dérogation explicite et non un
oubli.

**Une piste qui ne marche pas**, documentée pour qu'elle ne soit pas retentée : déduire
le taux du temps d'entraînement. Le coût d'un convnet est à peu près linéaire en nombre
d'échantillons d'entrée, donc une graine ayant tourné au taux natif devrait ressortir
d'un facteur natif/250. Le contrôle réfute la mesure :
`lee2019_mi/cross_subject/bd_sccnet` a ses 5 graines certifiées à 250 Hz et son coût
médian par graine varie quand même de 43.5 s à 242.3 s — un facteur 5.6, plus grand que
le facteur 4.0 qu'une vraie contamination à 1000 Hz produirait. L'early stopping et
l'hétérogénéité des GPU dominent le coût ; aucun seuil sur ce ratio ne peut servir de
garde-fou.

**Restaurer la provenance à 100 %** demande de relancer les 1170 cellules sans trace,
avec la colonne `sfreq` désormais écrite dans chaque CSV et `resample: 250.0` épinglé
dans les 12 configs dataset (commit `8812860`) — ce qui rend la panne impossible à
reproduire. Le coût dominant est `schirrmeister2017/cross_subject` ; l'ordre de grandeur
est la centaine de GPU-heures. C'est une décision à prendre avant soumission, pas après.

## Fichiers

| fichier | contenu |
|---|---|
| `results_published/eegrow_benchmark_all_scores.csv.gz` | table longue, 189 062 scores, un par (protocole, dataset, sujet, session, modèle, graine) — rien n'est moyenné |
| `results_published/eegrow_benchmark_levels.csv` | niveaux absolus par (protocole, dataset, modèle), graines moyennées par sujet/session |
| `results_published/eegrow_benchmark_paired.csv` | les 120 contrastes appariés croissant − fixe, avec Wilcoxon et fraction de deltas positifs |
| `aggregate_published.py` | produit les trois fichiers ci-dessus depuis `results/` |
| `provenance_audit.py` | le tableau du §4, reproductible |
| `where_grow_wins.py`, `where_grow_wins2.py` | les analyses appariées détaillées (§2) |
| `regime_guard.py` | le garde-fou de prétraitement |

Les CSV bruts par cellule (2160 fichiers) restent sur le cluster, sous
`benchmarks/results/<protocole>/<dataset>/<modèle>__seed<N>.csv`. `results_published/`
en est l'agrégat versionné, suffisant pour refaire toute l'analyse sans accès à
Margaret.
