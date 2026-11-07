# EXPLO — Quand l'empreinte du sujet apparaît-elle dans les neurones ajoutés ? (campagne A)

07/11/2026. Analyse **exploratoire**, annoncée comme telle : aucun verdict n'en décide
quoi que ce soit dans le papier sans réplication. Plan et script (`embed_growth.py`) gelés
dans le même commit, avant toute lecture des poids de croissance (`__fits__growth`) et des
checkpoints (`__fits__ckpt`). Seules leurs métadonnées (clés, formes, époques) ont été
lues le 07/11.

## 1. Pourquoi

La lecture 1 (`PREREG_growth_embedding_A.md`, lue le 07/11) a trouvé :
- P1 : en fin d'entraînement, les neurones ajoutés identifient le sujet entre deux sessions
  disjointes (lee2019_mi, top-1 0,324) mieux qu'un réseau bd (0,141) ;
- P2 : des neurones insérés dans une direction aléatoire font pareil (0,315).

Si la direction d'insertion ne compte pas, l'empreinte doit être **écrite par
l'entraînement qui suit l'insertion**. C'est la prémisse de la lecture 2 (adapter un
modèle partagé en lui ajoutant des neurones). On la vérifie directement : pour le bras
random, l'empreinte doit être nulle à l'insertion et apparaître ensuite.

Hypothèse de mécanisme (non testée ici) : un neurone ajouté à amplitude quasi nulle reçoit
d'abord un gradient ∝ E[x·δᵀ], ce que TINY calcule en une fois.

## 2. Données

- lee2019_mi, `cross_session`, graine 0, les 54 sujets à 2 sessions, mêmes items que la
  lecture 1 : (sujet, session d'entraînement).
- Bras : `shallow__tiny`, `shallow__random`, `shallow__bd`.
- Instants :
  - **ins** : les tenseurs de la jonction enregistrés juste après chaque insertion
    (`__fits__growth`, 8 événements, époques 5, 10, …, 40) ; chaque neurone est pris à
    l'événement qui l'a créé ;
  - **0** : `__fits__init` ; **10, 25, 50, 100** : `__fits__ckpt` ; **200** : `__fits__final`.
- Ensembles de neurones :
  - **ajoutés** (8-39) aux instants ins, 50, 100, 200 (tous existent à partir de
    l'époque 40) ;
  - **G1** (les neurones créés au 1er événement, époque 5 : indices 8 à w₁−1, soit 8-14
    sur le fit inspecté) aux instants ins, 10, 25, 50, 100, 200 : la trajectoire la plus
    longue ;
  - **départ8** (0-7, tiny et random) et **bd40** (0-39) aux instants 0, 10, 25, 50, 100,
    200 : neurones jamais ajoutés, même horloge.
- Le script vérifie, sans les lire comme résultat, que chaque fit grandit de 8 à 40 par
  événements contigus, le 1er à l'époque 5, et que la largeur de chaque checkpoint est
  celle que ses propres événements donnent à cette époque ; sinon le fit est exclu et
  compté (raison par raison). Un ensemble de neurones de norme nulle (rien à normaliser)
  donne une ligne nulle et il est compté.

## 3. Embedding et mesure

Ceux de la lecture 1, sans changement (spectre + topographie, z-score par bloc,
corrélation, plus proche voisin entre sessions, bootstrap 10 000 tirages sur les sujets,
`default_rng(0)`), à une différence près, nécessaire ici :

- **Ex æquo.** Toutes les cellules ont la graine 0 : à l'init (et, pour random, peut-être à
  l'insertion), les neurones peuvent être identiques d'un sujet à l'autre. `identify`
  prend alors le premier indice, ce qui donne 1/n par accident. `embed_growth.py` compte
  un ex æquo comme 1/(nombre de candidats à égalité) quand le bon sujet en fait partie,
  avec une tolérance de 1e-9 sur la corrélation. Il rapporte aussi, pour chaque ensemble
  et instant, la fraction d'items dont l'embedding est identique (à 1e-9) à celui d'un
  autre sujet.

## 4. Ce qu'on attend, écrit avant de lire

| # | Quantité (top-1, lee2019_mi) | Attendu si « l'entraînement écrit l'empreinte » |
|---|---|---|
| T1 | random, ajoutés, à ins | ≈ hasard (1/54) : la direction ne dépend pas des données |
| T2 | random, ajoutés, 200 − ins | > 0 (borne basse de l'IC) |
| T3 | random, G1, trajectoire ins → 200 | montée, et à quelle époque elle sature |
| T4 | tiny, ajoutés, à ins | rapporté : si > hasard, la règle écrit une empreinte que l'entraînement remplace ou renforce |
| T5 | tiny − random, à chaque instant | rapporté |
| T6 | départ8 et bd40, trajectoire 0 → 200 | à 0, hasard par construction (init identique) ; restent sous les ajoutés |

Lecture :
- T1 ≈ hasard **et** T2 > 0 → l'empreinte est écrite par l'entraînement après
  l'insertion : prémisse de la lecture 2 confirmée (exploratoire).
- T1 > hasard → la direction « random » dépend déjà des données (à comprendre avant
  d'aller plus loin : amplitude, normalisation, ou fuite de l'embedding).
- T2 ≤ 0 → contradiction avec la lecture 1 : chercher l'erreur avant toute
  interprétation.

## 5. Ce qui ne sera pas fait

- Pas d'autre dataset, pas d'autre embedding, pas de sélection d'instants après lecture.
- L'hypothèse de gradient (§ 1) n'est pas testée ici.
- La confusion sujet / montage du casque (PREREG lecture 1, § 9) reste entière.
