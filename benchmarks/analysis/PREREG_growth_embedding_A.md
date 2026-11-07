# PREREG — Empreinte du sujet dans les neurones ajoutés, sur les poids de la campagne A (lecture 1)

07/11/2026. **GELÉ** dans le commit qui contient aussi
`embed_A.py`, son test synthétique, avant toute exécution sur les poids. Analyse hors ligne
des poids déjà sauvés par la campagne A et par le contrôle d'init de Shallow : aucun
nouvel entraînement, CPU seulement.

Ce qui est déjà connu avant ce gel :
- les scores de la campagne A, lus le 07/11 (`read/campaignA.md`) ;
- le test hors ligne du 24/10 (`PREREG_growth_embedding_offline.md`, lu dans
  `read/embed_offline.md` et `read/embed_controls.md`).

Aucun poids de la campagne A n'a été lu pour cette question. Seules les métadonnées
(clés, formes, largeurs, sessions d'entraînement) ont été inspectées le 07/11 pour écrire
ce plan.

## 1. Question

Le 24/10, sur les fits within-session de la validation :
- les neurones ajoutés par TINY à Shallow identifiaient le sujet d'une session à l'autre
  (lee2019_mi : 0,370 en top-1 contre 0,065 pour bd, hasard 0,019) ;
- c'était non trivial à nombre de neurones égal ;
- mais c'était redondant avec la covariance brute de l'EEG : cov+ajoutés − cov =
  −0,009 [−0,028, +0,000], avec la covariance seule à 0,954, donc près du plafond.

Ce test avait une faiblesse que la campagne A permet de corriger, et une question qu'il
laissait ouverte.
1. **Les données d'entraînement se recouvraient.** Les deux empreintes d'un sujet
   venaient de folds within-session, qui partagent environ 75 % de leurs essais. Une
   partie de l'identification peut venir des essais communs. En cross-session, un sujet
   à 2 sessions a deux fits entraînés sur des sessions **disjointes**.
2. **D'où vient l'empreinte ?** TINY calcule les neurones ajoutés à partir des
   statistiques des données du sujet. Si c'est ce calcul qui écrit le sujet dans les
   neurones, des neurones ajoutés en nombre égal mais dans des directions aléatoires
   (bras `random` de la campagne A) ne devraient pas porter la même empreinte. Le
   24/10, ce bras n'existait pas.

La question est donc : **quand deux réseaux sont entraînés sur deux sessions disjointes
du même sujet, leurs neurones ajoutés par la croissance se ressemblent-ils plus que ceux
de sujets différents, et est-ce la règle TINY qui crée cette ressemblance ?**

La question « au-delà de la covariance » n'est pas un critère ici (§ 9, pourquoi).

## 2. Données (lues dans les fichiers le 07/11)

### 2.1 Fits

- Protocole `cross_session`, graine 0 (avenant 5). Chaque cellule (sujet) contient un fit
  par session de test, entraîné sur toutes les autres sessions du sujet
  (`meta.train_sessions`).
- **Seuls les sujets à exactement 2 sessions** donnent deux fits entraînés sur des
  données disjointes. Au-delà de 2 sessions, les données d'entraînement se recouvrent.
  Exemple lu sur shin2017a : le fit testé sur 0imagery est entraîné sur {2, 4}, celui
  testé sur 4imagery sur {0, 2}.

| dataset | sujets à 2 sessions | électrodes | rôle |
|---|---|---|---|
| lee2019_mi | 54 / 54 | 62 | **critères principaux** |
| bnci2014_001 | 9 / 9 | 22 | rapporté |
| bnci2015_001 | 8 / 12 | 13 | rapporté |

  bnci2014_004 (5 sessions), shin2017a (3 sessions), zhou2016 (3 sessions) et les 4
  sujets à 3 sessions de bnci2015_001 sont exclus, parce que leurs sessions
  d'entraînement se recouvrent.
- **Poids : ceux de la dernière époque** (`__fits__final/<uid>.pt`, clé `state_dict`).
  C'est l'époque qui est scorée dans la campagne A. La largeur finale de Shallow tiny est
  de **40 sur les 154 fits** des trois datasets (lu le 07/11), donc le nombre de neurones
  ajoutés est constant (32).

### 2.2 Les neurones de Shallow

- La jonction qui grandit est `conv_time` → `conv_spat` (`conv_time.layer.weight`
  (40, 1, 25, 1), `conv_spat.layer.weight` (40, 40, 1, n_chans)).
- Dans bd, les mêmes tenseurs s'appellent `conv_time_spat.conv_time.weight` et
  `conv_time_spat.conv_spat.weight`.
- gromo concatène les neurones dans l'ordre (anciens, nouveaux) : les neurones ajoutés
  sont les indices 8 à 39, et les indices 0 à 7 sont ceux de départ. C'est la même
  convention que dans le PREREG du 24/10, § 2. Le script lit les largeurs sur les
  tenseurs (`growable_width` vaut None dans les fichiers de bd). Il exige (départ, fin) =
  (8, 40) pour tiny, random et tiny_bdinit, et (40, 40) pour bd ; sinon, il exclut le
  fit et le compte. Pour EEGNeX, il exige (4, 32) et (32, 32).

### 2.3 Bras (tous en cross_session, sur les mêmes sujets)

| bras | source | neurones |
|---|---|---|
| **ajoutés** | `shallow__tiny` | 8-39 |
| **bd32** | `shallow__bd` | 32 neurones tirés au hasard parmi 40 (20 tirages, `default_rng(t)`, t = 0…19) ; on prend la moyenne des scores des tirages |
| bd40 | `shallow__bd` | 0-39 |
| départ8 | `shallow__tiny` | 0-7 |
| **ajoutés_random** | `shallow__random` | 8-39 (même règle que tiny, direction aléatoire) |
| ajoutés_bdinit | `shallow__tiny_bdinit` (contrôle d'init) | 8-39 |
| **cov** | EEG brut de la session d'entraînement du fit | — |

- **cov** : on prend les données de la session d'entraînement de chaque fit, telles que
  le réseau les a reçues. Elles sont construites par `check_cache.build_campaign(ds)`
  (même dataset, même paradigme et même clé de cache que le runner), lues dans
  `/scratch/amounir/moabb_cache_A` en lecture seule.
- Le calcul est celui de `embed_controls.covariances` :
  1. centrage de chaque essai ;
  2. covariance de chaque essai, normalisée par sa trace ;
  3. moyenne log-euclidienne des essais ;
  4. triangle supérieur de la matrice obtenue.

## 3. Embedding

Il est identique à celui du 24/10 (`embed_offline.describe`). Pour un ensemble S de
neurones :
- **spectre** = moyenne sur S de |rfft(filtre, n = 64)|², normalisé à somme 1 par
  neurone ;
- **topographie** = moyenne sur S de Σ_sorties w², normalisée à somme 1 par neurone ;
- chaque bloc est z-scoré sur les items du dataset, puis les deux blocs sont concaténés ;
- la distance est la distance de corrélation.

Un item est un (sujet, session d'entraînement) : exactement 2 items par sujet.

## 4. Mesure

On identifie le sujet entre sessions au plus proche voisin : on interroge avec les items
de la session A, on cherche parmi ceux de la session B, puis l'inverse.
- Le score d'un sujet vaut 0, 0,5 ou 1, et le hasard vaut 1/n_sujets.
- Pour bnci2015_001, A et B sont la 1re et la 2e session de chaque sujet, quel que soit
  son nom.
- Les contrastes sont appariés par sujet, avec un IC95 par bootstrap percentile sur les
  sujets (10 000 tirages, `default_rng(0)`).

## 5. Critères principaux (lee2019_mi)

Il y a deux critères. On ne conclut que **la règle de croissance écrit le sujet dans ses
neurones** que si les deux passent. Comme la conclusion exige les deux, chaque test se
fait à 95 % sans correction (test intersection-union).

**P1. L'empreinte survit sur des données disjointes.**
D1 = top-1(ajoutés) − top-1(bd32).
- EMPREINTE PROPRE À LA CROISSANCE si la borne basse est > 0 ;
- PAS AU-DELÀ D'UN RÉSEAU ENTRAÎNÉ si l'IC contient 0 ;
- MOINS QU'UN RÉSEAU ENTRAÎNÉ si la borne haute est < 0.

**P2. C'est la règle TINY qui crée l'empreinte.**
D2 = top-1(ajoutés) − top-1(ajoutés_random). Les deux bras ont la même classe, la même
init, le même nombre de neurones ajoutés (32) et la même recette. Seule la direction des
neurones ajoutés diffère.
- CRÉÉE PAR LA RÈGLE si la borne basse est > 0 ;
- PAS PAR LA RÈGLE si l'IC contient 0 : l'empreinte vient de l'entraînement qui suit la
  croissance, quelle que soit la direction ;
- PLUS FORTE SANS LA RÈGLE si la borne haute est < 0.

Lecture :
- P1 ET P2 → la règle TINY écrit le sujet dans les neurones qu'elle ajoute, et cela
  survit à l'entraînement sur des données disjointes.
- P1 sans P2 → l'empreinte est propre aux neurones ajoutés, mais elle vient de
  l'entraînement des neurones nouveaux, pas de la règle.
- Sans P1, le +0,306 du 24/10 venait des données partagées entre folds ; il est rapporté
  comme tel.
- Dans tous les cas, la lecture 1 comme empreinte d'analyse *nouvelle* reste non
  démontrée : la covariance identifie déjà le sujet (§ 6, § 9).

## 6. Rapportés, sans valeur décisionnelle

Tout ce qui suit est rapporté pour lee2019_mi, puis pour bnci2014_001 et bnci2015_001
(n = 9 et 8, sous-puissants : leurs MDE sont rapportés).

- P2 est à lire avec Q2 de la campagne A : en score, tiny ≈ random sur 15/15 endpoints.
- **ajoutés_bdinit − bd32 : l'empreinte dépend-elle de l'init PyTorch ?**
- ajoutés − départ8, ajoutés − bd40.
- Les blocs spectre seul et topographie seule.
- cov seule, cov+ajoutés − cov et cov+bd32 − cov (le contraste du 24/10), sur données
  disjointes cette fois. Descriptif seulement (§ 9).
- L'AUC des distances même sujet contre sujets différents.
- Les fits exclus (largeur ≠ 40 ou départ ≠ 8) et les sujets dont il manque un item.

## 7. Exploratoire, annoncé comme tel

### 7.1 Cross-subject : la signature du sujet laissé de côté

En cross-subject, un fit est entraîné sur tous les sujets sauf celui qu'on teste
(vérifié sur chaque fit par `campaignA_check.py`). Ses neurones ajoutés décrivent donc le
groupe d'entraînement, et non le sujet testé. Mais tous les fits ont la même graine (0) :
deux fits ne diffèrent que par le sujet laissé de côté, plus le bruit du GPU et de la
carte.

- **Données** : lee2019_mi, `shallow__{tiny,bd}`, cross_subject, poids finaux, 54 fits.
- **Signature** de i = e_i − moyenne_j e_j, avec e l'embedding du § 3 (ajoutés 8-39
  pour tiny, 40 neurones pour bd).
- **Cible** de j = −(a_j − moyenne a), avec a_j l'embedding propre du sujet j dans le
  même espace : la moyenne de ses 2 items cross_session (§ 2.3), du même bras (ajoutés
  de tiny pour tiny, bd40 pour bd). Retirer i du groupe déplace les neurones du groupe à
  l'opposé de i, d'où le signe. La covariance n'est pas dans le même espace que
  l'embedding, elle ne sert donc pas de cible.
- **Mesure** : le sujet j dont la cible est la plus corrélée à la signature de i, avec
  top-1 = [j = i]. Le hasard vaut 1/54. On rapporte tiny, bd et tiny − bd, avec un IC
  bootstrap.
- **Rien n'en sera conclu.** Les 54 groupes d'entraînement se recouvrent à 52/53, et les
  fits ont tourné sur des cartes différentes, dont le non-déterminisme peut couvrir le
  signal. Un résultat nul ne dirait rien.

### 7.2 Les autres modèles

Aucun embedding comparable d'un fit à l'autre n'existe dans l'espace des poids pour les
neurones ajoutés des 4 autres modèles. Ce qui a été lu :
- MSVTNet, CTNet et Conformer font grandir les FFN de leurs blocs transformer
  (`GrowingMSVTNetAll`, `GrowingCTNetAll`, `GrowingEEGConformerAll`). Leurs entrées et
  leurs sorties vivent dans le flux résiduel appris, dont la base n'est pas identifiable
  d'un réseau à l'autre.
- EEGNeX (`grow_eegnex_j2`) fait grandir la jonction 2 (`block_4`, 4 → 32), dont les
  entrées sont 64 cartes de caractéristiques apprises.

Seul l'axe du temps est commun d'un fit à l'autre. Pour EEGNeX, on rapporte donc
l'identification sur le **spectre des noyaux ajoutés** (indices 4-31), avec le noyau de
16 points, moyenné sur les canaux d'entrée, contre les noyaux de bd au même endroit.
C'est exploratoire, sur lee2019_mi seulement. Les transformers ne sont pas traités. Un
embedding fonctionnel (activations sur des essais de référence, CKA) demanderait des
passes avant et un autre PREREG.

## 8. Ce qui ne sera pas fait

- Aucun critère, aucun dataset, aucune époque ni aucun embedding ne sera changé après
  lecture. Les poids de l'époque sélectionnée ne sont pas regardés.
- On ne cherchera pas d'autres embeddings si P1 ou P2 échoue.
- On n'annoncera pas l'empreinte comme mécanisme physiologique : les poids spatiaux sont
  des filtres, pas des patterns (Haufe).

## 9. Limites connues avant lecture

- Les deux sessions d'un sujet sont enregistrées à quelques jours d'intervalle avec le
  même casque. Une partie de l'empreinte peut être celle du montage (impédances, position
  du casque) plutôt que celle du cerveau. Ce test ne sépare pas les deux, et la
  covariance non plus.
- **Pourquoi « au-delà de la covariance » n'est pas un critère.** Le premier brouillon
  testait l'identification sur le résidu d'une régression ridge cov → embedding. Son test
  synthétique (`tests/test_embed_A.py`, 07/11, avant toute lecture) l'a réfuté : un
  embedding qui n'est qu'une fonction linéaire de la covariance plus du bruit garde un
  résidu qui identifie 40/40 sujets. Le ridge régularisé ne retire qu'une partie de ce que
  la covariance explique, et le reste suffit à identifier le sujet. Avec 1 953 entrées
  pour 106 items, ce serait pire. Plus généralement, quand la covariance identifie déjà
  presque tous les sujets, toute autre vue des mêmes données (une fonction de la
  covariance pondérée autrement) peut améliorer un classifieur au plus proche voisin sans
  rien contenir de plus. Ce test n'est donc pas faisable avec 54 sujets, et il n'est pas
  fait. cov+ajoutés − cov est seulement décrit.

## 10. Avenant 1 (07/11, avant toute lecture) — cache des covariances

Le § 2 disait les covariances lues dans `/scratch/amounir/moabb_cache_A`. C'est faux pour
lee2019_mi : ses 2 754 cellules de campagne lisaient `/scratch/amounir/moabb_cache`
(`cache.path=` de chaque ligne de cellule, et `cache` du manifest `manifest_A/lee2019_mi.json`) ;
`moabb_cache_A` n'en contient que les sujets 1 et 2. Le premier lancement (job 911992, commit
b419ba2) a donc manqué le cache et reconstruit lee2019 depuis les fichiers bruts retéléchargés.
Il a été annulé au sujet 12/54, avant d'avoir écrit la moindre sortie : aucun poids, aucune
identification, aucun contraste n'a été lu.

Correctif, sans changer ni bras, ni mesure, ni critère, ni seuil :
- le cache de chaque dataset est celui de ses lignes de cellule (`cache_of`), un seul par
  dataset, sinon refus avant tout chargement ;
- un défaut de cache refuse (`REFUSED: cache miss`) au lieu de reconstruire depuis le brut ;
  les covariances sont donc exactement les données que le runner a lues.
