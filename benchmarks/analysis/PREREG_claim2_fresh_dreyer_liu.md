# Pré-registration — réplication de claim 2 sur deux datasets jamais vus

**Écrite le 03/09/2026, avant le premier fit sur `dreyer2023a` et `liu2024`.**
Le cache de ces deux datasets a été monté et vérifié (job 525597, 60/60 et 50/50,
zéro manquant zéro mensonge) ; aucune cellule d'entraînement n'a encore tourné sur
eux, ni sonde, ni matrice. C'est la seule fenêtre où ce document peut être écrit.

## Pourquoi ces deux-là, et pourquoi maintenant

Tout ce que le chantier donneur-receveur a produit depuis l'étage C est **post-hoc**.
`restored_epoch` supplante `params_probe` comme prédicteur de qualité de donneur : c'est
mesuré, c'est reproductible, et ça a été trouvé **en regardant les données**. Le plafond
de sélection (36/36) est du même statut. Un relecteur d'ICLR a raison de ne pas les
créditer autrement que comme descriptif.

Les sept autres datasets du cache sont à n ≤ 14 sujets : ils ne peuvent pas servir de
réplication, non pas parce qu'ils sont mauvais mais parce qu'ils sont **sous-puissants**
(§ Puissance). `dreyer2023a` (60 sujets) et `liu2024` (50) sont les deux seuls candidats
2-classes de taille suffisante que MOABB expose et que nous n'avons jamais touchés.
Ils sont donc la seule chance de faire passer quoi que ce soit du descriptif au
confirmatoire, et cette chance ne s'utilise qu'une fois.

## La cible, chiffrée, telle qu'elle est déjà publiée en interne

`donor_meta.py` sur les trois datasets retenus (analyse B, celle rapportée) :

| dataset | n | ρ partielle (`params_probe` | `acc_probe`) |
|---|---|---|
| physionetmi | 109 | +0.385 |
| lee2019_mi | 54 | +0.229 |
| cho2017 | 52 | +0.150 |
| **groupé** | **215** | **+0.288  IC95 [+0.164, +0.414]** |

C'est **+0.288** qui est l'effet à répliquer. Écrit ici pour qu'il ne puisse pas être
redéfini après coup vers une valeur plus facile.

## Puissance — calculée AVANT, pas après un résultat nul

Partielle du premier ordre, `SE = 1/sqrt(n − 4)`, unilatéral α=0.05, puissance 0.80 :

| | df | MDE | contre la cible +0.288 |
|---|---|---|---|
| dreyer2023a seul | 56 | +0.321 | **sous-puissant** |
| liu2024 seul | 46 | +0.351 | **sous-puissant** |
| **les deux groupés** | **102** | **+0.241** | **puissant** |

Conséquence pré-spécifiée, et c'est la décision de conception la plus importante du
document : **l'endpoint primaire est le test GROUPÉ sur les deux datasets.** Les deux
estimations par dataset sont **descriptives uniquement**. Un ρ non significatif sur
`liu2024` seul ne sera pas rapporté comme un échec de réplication — à n=50 il n'a pas la
puissance de le dire, et l'annoncer serait exactement la faute que
[[underpowered-not-null]] consigne.

Réciproquement : si le groupé est positif mais qu'un seul des deux datasets porte tout
l'effet, l'hétérogénéité se lit sur Q et I², et le compte-rendu est « l'effet dépend du
dataset », avec les deux chiffres.

## L'hypothèse primaire

**H1.** Sur `dreyer2023a` ∪ `liu2024`, la ρ de Spearman partielle entre la qualité de
donneur `q_cent` et `restored_epoch` du donneur, contrôlée par `acc_probe`, est
**strictement positive**. Unilatéral, α=0.05, groupage Fisher-z à effets fixes, unité
d'analyse = le **sujet** ([[unit-of-analysis-subject]]).

Le prédicteur est `restored_epoch`, **pas** `params_probe`. C'est le point de la
réplication : l'étage C a trouvé post-hoc que la durée d'apprentissage productif
supplante la taille, et une pré-registration qui testerait `params_probe` re-testerait
ce qui est déjà établi au lieu de ce qui ne l'est pas.

**H2 (secondaire, Holm sur la famille des 2).** `params_probe` contrôlée par
`acc_probe` est également positive — la réplication directe de claim 2 telle que
publiée.

**H3 (descriptive, aucun test).** La partielle de `restored_epoch` sachant
(`acc_probe`, `params_probe`) est supérieure à celle de `params_probe` sachant
(`acc_probe`, `restored_epoch`). C'est la formulation exacte de « supplante ». Elle est
descriptive parce qu'une comparaison de deux partielles corrélées sur les mêmes données
n'a pas de test que je sois prêt à défendre ici.

## Ce qui invalide la lecture — le garde-fou, écrit à l'avance

**G. `acc_probe` doit prédire `q_cent` positivement sur chacun des deux datasets.**
C'est l'effet le plus fort et le mieux établi du chantier (ρ +0.56/+0.67/+0.67 au niveau
du pool). S'il ne transporte pas, ce n'est pas H1 qui est réfutée : c'est la matrice
D×R du dataset qui est cassée, et **on ne lit ni H1 ni H2 sur ce dataset**, ni pour ni
contre. Un garde-fou refuse, il n'avertit pas ([[gates-must-refuse-not-warn]]).

## La porte d'exclusion — remplacée, et voici la mesure qui l'impose

L'ancien critère était `ceil_probe < 0.50`, sur la censure haute de la sonde. Il est
**retiré**, et l'échéance que [[prereg-gate-one-sided]] fixait — le corriger avant
l'arrivée d'un sixième dataset — est celle-ci. Trois candidats de remplacement ont été
mesurés sur les 12 datasets de la campagne, avant tout fit frais ; **deux ont été
réfutés** et c'est consigné ici pour que le troisième ne passe pas pour une évidence.

1. **Censure bilatérale (haut + bas), seuil 0.50 — RÉFUTÉE.** Elle attrape bien
   `shin2017a` (0.002 en haut mais **0.644 en bas** : il passait l'ancienne porte alors
   qu'il est le deuxième plus censuré). Mais elle **exclurait `physionetmi`** (0.514),
   qui est le dataset au plus fort effet, à la plus grande dispersion et au plus grand n.
   La censure par réplicat n'est pas celle de la moyenne des 15.

2. **Sujets épinglés à une borne — RÉFUTÉE, ne discrimine pas.** Un sujet dont les 15
   réplicats collent du même côté porte un majorant, pas une mesure — c'est la bonne
   intuition, mais elle vaut 0 % partout sauf `schirrmeister2017` (21 %),
   `bnci2014_001` (11 %) et `bnci2014_002` (7 %). Le critère est vide.

3. **Fiabilité du classement — RETENUE.** Spearman ne voit que des **rangs** : ni
   l'étendue ni la censure ne l'atteignent directement. Ce qui atténue une ρ de rang est
   la reproductibilité du classement, `ICC(1,k)` de la sonde par sujet, corrigée
   Spearman-Brown pour les k réplicats ([[icc-single-vs-mean]]). Seuil `ICC_FLOOR =
   0.50`, la constante qui existe déjà dans `donor_predictor.py`.

**Et la correction que ça entraîne sur le passé, à écrire maintenant plutôt qu'à
découvrir plus tard.** Sous ce critère `bnci2014_001` **passe** (ICC(1,k) = 0.928 sur
`params`, ses 89 réplicats par sujet rendent son classement très reproductible malgré
une étendue de 41 k–46 k). Il reste néanmoins exclu, mais pour **la vraie raison, qui
est n=9** : son MDE est +0.94, il ne pouvait rien détecter. L'exclusion publiée était le
bon verdict pour le mauvais motif ; l'analyse ne change pas, la justification si.

Le critère porte sur la **sonde seule**, jamais sur `q_cent`. Il est calculable — et
sera calculé — avant que la matrice D×R des datasets frais ne soit lue.

## Ce qui est gelé, au sens propre

- Sonde : `grow_shallow`, `within_session`, `align_tag=none`, seeds 0/1/2,
  protocole `train.patience=200 train.selection_monitor=valid_acc`
  ([[deep4-instability]]). Identique aux 12 datasets de la campagne.
- Qualité de donneur : `dm.donor_quality(d, dm.METRIC)`, `q_cent`, code inchangé.
- Le classement vient de `donor_select_prep.py`, jamais re-dérivé.
- Checkout `eegrow_budget`, `EEGROW_SHA=58c6bee`, le même que les cellules existantes.

## Interdits explicites, mot pour mot

Sur un résultat nul groupé : interdit d'écrire « `restored_epoch` ne prédit pas », «
l'effet ne se réplique pas », « c'est du bruit ». Le seul énoncé autorisé est le
chiffre avec son IC et son MDE. L'équivalence ne s'annonce que par TOST à ±25 % de
+0.288.

Sur un résultat positif : interdit d'écrire que ça « confirme » le plafond de sélection
ou le continuum d'allocation. Ceux-là restent post-hoc et descriptifs quoi qu'il arrive
ici ; H1 ne porte que sur H1.
