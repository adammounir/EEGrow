# Pré-registration — réplication de claim 3 (v2) sur cho2017

**Écrite le 02/09/2026, 11 h 40, avant toute cellule cho2017.**
Elle amende `donor_select2_analysis.py` sans le modifier : ce fichier-là est gelé au
octet près parce que les cellules physionetmi (array 514187) sont en train d'y tomber,
et qu'un fichier de décision qu'on édite pendant que les données arrivent n'est plus
une pré-registration. Tout ce qui n'est pas contredit ici s'applique tel quel.

## Ce qui change, et rien d'autre

| paramètre | physionetmi | cho2017 | pourquoi |
|---|---|---|---|
| n sujets | 109 | 52 | le dataset |
| vivier (4 plis) | 82 | 39 | conséquence |
| `--candidates M` | 40 | **20** | M/vivier = 49 % vs **51 %** : c'est le ratio qui fixe la variance de tirage ajoutée, et c'est lui qu'on conserve, pas M |
| `--k` | 5 10 20 | **3 5 10** | K/M = 15/25/50 % vs 12/25/50 % : le même axe de saturation |
| `--reps` | 8 | 8 | 4 × 8 = 32 unités, inchangé |
| essais par cellule | 225/450/897 | 600/1000/2000 | cho2017 a 200 essais/sujet contre 45 |

**M=20 n'est pas un choix esthétique.** Avec M=40 sur un vivier de 39, tout le vivier
serait candidat et on retomberait *exactement* sur le défaut de v1 : un seul pool par
pli, un contraste à 4 réplicats déguisé en 32. Le recouvrement mesuré des candidats
entre réplicats est de **50 %** (49 % sur physionetmi) — le correctif ajoute bien la
même variance des deux côtés.

## L'endpoint primaire ne change PAS : K=5

C'est le point le plus important du document, et il va contre la tentation.

Sur physionetmi, K=5 avait été choisi parce que c'est le régime de recouvrement le plus
faible. Sur cho2017 le recouvrement le plus faible est à **K=3** (1 %). Prendre K=3
comme primaire ici serait « choisir le K le plus favorable dataset par dataset » : deux
expériences avec deux endpoints différents ne se répliquent pas l'une l'autre, elles
s'additionnent en tant que deux essais indépendants sur la même hypothèse — c'est-à-dire
qu'il faudrait corriger pour les deux, et on perdrait précisément ce qu'une réplication
apporte. **L'endpoint primaire d'une réplication est celui de l'original.** K=3 et K=10
sont secondaires, descriptifs, Holm-corrigés, comme sur physionetmi.

`PRIMARY_K = 5` existe dans les deux plans : `donor_select2_analysis.py --dsel
/scratch/amounir/dsel2/cho2017` tourne **sans modification**.

## Recouvrements mesurés au `--dry-run`, avant tout GPU

| | K=3 | K=5 | K=10 |
|---|---|---|---|
| `resid_top` ∩ `acc_top` | 1 % | **11 %** | 52 % |
| `params_top` ∩ `acc_top` | 2 % | 14 % | 54 % |
| `params_top` ∩ `resid_top` | 92 % | 95 % | 96 % |

Deux lectures à consigner **maintenant**, pour ne pas les découvrir après coup :

1. **cho2017 est le dataset qui discrimine.** ρ(`#params`, accuracy) y vaut +0.042
   contre +0.519 sur physionetmi et +0.647 sur lee2019_mi. Les deux règles y sont donc
   quasi indépendantes, et le contraste (c) y a le levier maximal des quatre datasets.
   C'est pour ça que la réplication se fait ici en premier, et ce choix est antérieur
   aux résultats.

2. **`params_top` ≈ `resid_top` sur cho2017 (95 % à K=5).** Conséquence directe de la
   marginale nulle : résidualiser `#params` sur l'accuracy n'y change presque pas le
   classement. Donc sur ce dataset le contraste ne dépend plus de laquelle des deux
   règles on prend — c'est une simplification réelle, **pas** une licence pour rapporter
   celle qui sort la mieux. Si les deux divergent malgré 95 % de recouvrement, c'est un
   signal de bruit et il faut le dire.

## Ce qui ferait échouer la réplication, écrit à l'avance

- **Garde-fou G** (inchangé) : (a) `resid_top` > `random` et (b) `params_bottom` <
  `random` doivent tenir à chaque K. S'ils ne tiennent pas, le classement ne transporte
  pas jusqu'à cho2017 et **on ne lit pas (c)** — ni pour, ni contre.
- **Prédiction T** (inchangée) : le gain sur `random` doit décroître de K=3 à K=10. S'il
  ne décroît pas, mon explication de l'échec de v1 par la saturation est fausse et ne
  doit plus servir d'argument, y compris sur physionetmi.
- **Équivalence** : uniquement par TOST à ±25 % de l'effet de sélection. Un P non
  significatif tout seul ne donne droit à aucune phrase. Interdits explicites, mot pour
  mot : « la taille gagne », « l'accuracy gagne », « c'est équivalent », «
  indiscernables ».

## Multiplicité entre les deux datasets

Deux datasets, un endpoint primaire chacun. Les deux P sont rapportés **ensemble**,
Holm sur la famille des 2, et le résultat annoncé dans le papier est le pire des deux
après correction. Rapporter physionetmi seul s'il est positif et taire cho2017 s'il ne
l'est pas serait une sélection sur le résultat ; l'ordre de soumission (physionetmi
d'abord, cho2017 ensuite) ne crée aucun droit à choisir après coup.

Si les deux divergent, le compte-rendu est « l'effet dépend du dataset » avec les deux
chiffres, et l'hétérogénéité se lit sur le Q des deux estimations — pas « ça marche sur
physionetmi ».
