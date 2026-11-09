# EXPLO — Les neurones que la croissance ajoute au FFN de Conformer portent-ils une empreinte du sujet ?

09/11/2026. Analyse **exploratoire**. La note et `embed_conformer.py` sont gelés dans le
même commit, avant toute lecture des poids de Conformer à cette fin. Avant le gel, seuls
les noms et les formes des tenseurs d'un fit ont été regardés (`pairs.0.*` : 16 → 25
neurones au 1er événement, bloc 0, époque 5 ; bd : `transformer.0.1.fn.1.{0,3}.weight`).
CPU, lecture seule des poids de la campagne A, aucun entraînement.

## 1. Pourquoi

La lecture 1 (Shallow) a trouvé une empreinte du sujet dans les neurones ajoutés, mais
pas d'écart TINY − random détectable. Ce résultat reste fragile pour deux raisons :
- P2 est sous-puissant (IC [−0,111, +0,130]) ;
- le bras random partage ses tirages entre sujets : avec la graine 0 partout, 2/3 des
  neurones random ont un jumeau exact chez un autre sujet
  (`/scratch/amounir/eegrow_read/randdir_explo/run.log`).

Adam propose (09/11) de trancher avec plusieurs graines, et de ne pas se limiter à
Shallow mais d'inclure un transformer. Avant de payer un réentraînement de Conformer, on
vérifie sur les poids existants que la question a un objet, c'est-à-dire que ses
neurones ajoutés portent une empreinte du sujet.

## 2. Données

- lee2019_mi, `cross_session`, graine 0. Les 54 sujets ont 2 sessions, donc deux fits
  entraînés sur des sessions disjointes. Un item est un (sujet, session d'entraînement).
- Conformer, bras `tiny`, `random` et `bd`. Poids finaux (époque 200), plus les tenseurs
  enregistrés à chaque insertion (`__fits__growth`).
- On ne regarde que le FFN du bloc 0, où E1 place environ 96 % de la croissance. Les
  neurones ajoutés aux autres blocs sont ignorés, et leur part est rapportée
  (`share_j0`).
- Un fit est exclu, et compté par motif, si :
  - il n'a aucun événement au bloc 0 ;
  - sa largeur de départ n'est pas 16 ;
  - ses événements au bloc 0 ne sont pas contigus ;
  - sa largeur finale diffère de celle du dernier événement ;
  - pour bd, sa largeur FFN au bloc 0 n'est pas 160.

## 3. Embedding et mesure

C'est l'analogue, pour un FFN, du « spectre + topographie » de Shallow. Un neurone j du
FFN est décrit par deux profils :
- **lecture** : W1[j]² / ‖W1[j]‖², c'est-à-dire les 40 dimensions du token qu'il lit ;
- **écriture** : W2[:, j]² / ‖W2[:, j]‖², c'est-à-dire les 40 dimensions qu'il écrit.

Pour un ensemble de neurones, on fait la moyenne de chaque profil. La suite est
identique à la lecture 1 :
- chaque bloc est z-scoré sur les items, puis les deux blocs sont concaténés ;
- la distance est la corrélation ;
- on identifie le sujet entre sessions au plus proche voisin, avec les ex æquo de
  `embed_growth.identify_ties` ;
- les IC sont des bootstraps sur les sujets (10 000 tirages, `default_rng(0)`).

Les ensembles :

| nom | source |
|---|---|
| tiny_added / random_added | neurones 16…h−1 du bloc 0, poids finaux |
| tiny_start16 | neurones 0-15 du bloc 0 du fit tiny, poids finaux |
| tiny_added_ins / random_added_ins | chaque neurone ajouté pris à l'insertion qui l'a créé |
| bd_matched | bd, même nombre de neurones que les ajoutés du fit tiny du même item, tirés parmi 160 (20 tirages, `default_rng(t)`) ; moyenne des scores sur les tirages |
| bd_all | les 160 neurones de bd |

Hypothèse de comparabilité, à garder en tête : les axes du token (40 dimensions) sont
ceux d'un réseau entraîné à partir de la même init (graine 0) pour tous les sujets.
Comparer des profils sur ces axes d'un réseau à l'autre suppose que l'entraînement ne
les permute pas.

Diagnostics rapportés :
- `new_over_old_sqnorm_median` : à l'insertion, la norme² moyenne des neurones neufs
  divisée par celle des anciens. Elle doit être ≪ 1 si les neurones neufs sont bien les
  derniers indices.
- Les clones random à l'insertion : même contrôle que pour Shallow.

## 4. Ce qu'on attend, et ce qu'on fera (écrit avant de lire)

- **E0, tiny_added − hasard.**
  - Borne basse > 0 : l'empreinte existe dans Conformer, et Conformer entre dans la
    campagne multi-graines comme second modèle.
  - Sinon : la question TINY contre random n'a pas d'objet dans ce transformer, avec
    cet embedding. Conformer n'entre pas dans la campagne, et on rapporte le nul avec
    sa MDE.
- **E1, tiny_added − bd_matched :** rapporté. Dit si l'empreinte est propre aux neurones
  ajoutés, comme P1 pour Shallow.
- **E2, tiny_added − random_added :** rapporté, ne décide rien. Tirages partagés, graine
  unique : c'est précisément ce que la campagne multi-graines doit trancher.
- Les autres contrastes (insertion, trajectoire 200 − ins, start16) sont rapportés.

## 5. Ce qui ne sera pas fait

- Pas d'autre embedding si E0 échoue : on n'en cherchera pas un qui « marche ».
- Pas de CTNet ni de MSVTNet ici. MSVTNet grandit presque jusqu'à la cible, et la
  croissance de CTNet dépend des données (E1).
