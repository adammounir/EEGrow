# Addendum — le bras `all_pool` (« tout le monde dedans »)

**Écrit le 02/09/2026, 18 h 10, APRÈS lecture des résultats de claim 3.**
Ce document existe pour que ce statut soit impossible à oublier au moment de rédiger.

## Pourquoi il est ajouté

Sylvain a nommé trois comparateurs le 01/09, mot pour mot :

> « tu pourrais comparer le cross-subject où tu mets tout le monde dedans… Tu pourrais
> comparer le cross-subject où tu mets que les cinq meilleurs… Et il faudrait le comparer
> à d'autres techniques, genre tu prends des sujets au hasard ou tu prends tous les
> sujets. »

Les deux derniers étaient dans le protocole depuis la v1. **Le premier manquait.**
Cause technique : `draw_candidates` plafonne M strictement sous la taille du vivier
(`donor_select.py`, `if m <= 0 or m >= len(pool)`), et `--k` n'est jamais allé au-delà de
20 sur des viviers de 39 (cho2017) et 82 (physionetmi).

C'est un manque de fond. Sans ce bras, la seule phrase que le protocole autorise est
« parmi les pools de taille K, tel critère vaut mieux que tel autre ». La phrase que veut
un relecteur — **« sélectionner vaut-il mieux que ne pas sélectionner ? »** — reste hors
de portée, alors que « tout mettre » est le comportement par défaut de la littérature.

## Son statut : post-hoc, descriptif, et hors de la famille pré-enregistrée

Il est écrit après avoir lu :

- physionetmi, endpoint primaire **P(K=5) = −0.0377 [−0.0499, −0.0258], p < 1e-4**,
  n = 64 réplicats — significatif et **négatif** (`acc_top` bat `resid_top`) ;
- cho2017, **garde-fou G non tenu** aux trois K, avec (b) `params_bottom` > `random`
  significativement positif à K=3, soit le signe inverse du prédit.

Conséquences, à tenir sans exception :

1. Il **n'amende pas** l'endpoint primaire et n'entre dans **aucune** correction de
   multiplicité de la famille des 2 (`PREREG_donor_select2_cho2017.md`, § Multiplicité).
2. Le verbe **« confirme »** lui est interdit, ainsi que toute formulation qui le
   présenterait comme prévu. La formule à employer est : « bras ajouté après coup, à la
   demande explicite de S. Chevallier, rapporté comme référence descriptive ».
3. Il ne peut pas **renverser** le résultat de claim 3. Il peut le **recadrer** — par
   exemple montrer que les quatre règles perdent toutes contre le vivier entier, ce qui
   rendrait le classement entre elles secondaire.

## Ce qu'il ne mesure pas, et c'est structurel

**K n'est pas apparié.** `all_pool` s'entraîne sur 39 ou 82 sujets, les règles sur 3 à
20. La contrainte « toutes les règles choisissent le même K », qui est la moitié de la
validité du protocole principal, est ici **délibérément violée** : la comparaison
confond « plus de données » et « meilleures données ». C'est légitime pour une référence,
puisque la question posée est exactement « trier bat-il le fait d'avoir tout », et
illégitime pour un contraste entre critères.

**Interdit explicite, mot pour mot :** ne jamais lire un écart `all_pool` − `params_top`
comme un effet de critère de sélection, ni écrire qu'une règle « approche » `all_pool`
avec moins de données sans donner les deux K dans la même phrase.

**Pas de réplicats.** Le pool est déterministe étant donné le pli : il n'y a rien à
tirer. Sa seule variance est l'initialisation, d'où 8 seeds × 4 plis = 32 fits.
L'unité de rééchantillonnage est le **pli** (F = 4), pas le réplicat. C'est assez pour
une référence descriptive et ce ne serait pas assez pour un test — donc on n'en fait pas
un test.

## Comment il sera rapporté

Une seule quantité, par dataset : la moyenne par sujet de test de `all_pool`, et l'écart
à chaque règle à chaque K, **avec un intervalle bootstrap sur les sujets de test** et la
mention des deux K. Aucune valeur p. Le sens de lecture attendu et les trois issues
possibles, écrites maintenant :

| issue | lecture |
|---|---|
| `all_pool` > toutes les règles | la sélection ne paie pas à ces tailles de pool ; c'est le résultat le plus probable et le plus utile à publier |
| `all_pool` ≈ la meilleure règle | K sujets bien choisis valent le vivier entier — intéressant, mais confondu avec la quantité de données |
| `all_pool` < une règle | sélectionner bat tout mettre ; à ne pas annoncer sans vérifier d'abord que le fit sur le vivier entier n'est pas sous-entraîné (lire `epochs` et `stop_reason` dans les CSV) |

La troisième ligne est là parce que c'est l'issue qui m'arrangerait, et c'est précisément
celle qui a un mode de défaillance banal : un pool 4 à 8 fois plus gros avec le même
`patience=200` peut simplement ne pas avoir convergé.

## Isolation des données

Sortie dans `/scratch/amounir/dsel_all/<dataset>`, séparée de `dsel2/`. Noms au format
v1 (`f0__all_pool__d0__seed3.csv`) et nom de règle absent de `RULES` : ni le glob v1 ni
le glob v2 ne peuvent ramasser ces cellules. Deux protocoles dont l'unité d'analyse
diffère ne doivent jamais pouvoir se retrouver dans le même DataFrame.
