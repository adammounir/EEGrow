# PREREG — Les neurones ajoutés par la croissance portent-ils une empreinte du sujet ?

24/10/2026. Gelé avant toute lecture (commit qui contient ce fichier et
`embed_offline.py`). Analyse hors ligne des poids déjà sauvés par la validation : pas de
nouvel entraînement.

## 1. Question

Piste de Sylvain, relayée par Adam le 24/10 : la croissance des premières couches donne
un « embedding adaptatif » au contexte (sujet ou session). TINY ajoute les directions que
le réseau ne sait pas exprimer et qui feraient baisser la perte. Les neurones ajoutés
décriraient donc ce qui manquait au modèle **pour ce sujet**.

Test le moins cher : ces neurones portent-ils une empreinte du sujet **au-delà** de ce que
porte n'importe quel réseau entraîné sur ce sujet ?

## 2. Données (existantes, lues dans le code)

- `valid_results/shallow__{tiny,bd}`, checkout `eegrow_valid` (35b6f09).
- Évaluation `within_session`.
- `.pt` = `state_dict` à l'**époque sélectionnée** (RestoreBestModel swap=False, médiane
  de l'époque = 48 sur lee), avec la méta subject/session/cv_ind.
- Jonction qui croît : `conv_time` → `conv_spat`.
- gromo concatène `(anciens, nouveaux)`, donc les neurones ajoutés sont les indices ≥ 8.
- Un neurone k, c'est :
  - un filtre temporel `conv_time.weight[k]` (25 points) ;
  - ses poids spatiaux `conv_spat.weight[:, k, 0, :]` (40 × électrodes).
- Datasets avec 2 sessions : **lee2019_mi** (54 sujets, graine 0) et **bnci2014_001**
  (9 sujets, graines 0-2). cho2017 a une seule session, il est exclu.

## 3. Embedding (invariant à l'ordre, au signe et au nombre de neurones)

Pour un ensemble S de neurones d'un fit :
- spectre = moyenne sur S de |rfft(filtre, n=64)|², normalisé à somme 1 par neurone ;
- topographie = moyenne sur S de Σ_out w², normalisée à somme 1 par neurone.

L'empreinte d'un (sujet, session) est la moyenne sur ses folds. Chaque bloc est z-scoré
sur les items du dataset, puis les deux blocs sont concaténés. La distance est la
distance de corrélation.

Bras :
- **ajoutés** : tiny, indices ≥ 8, seulement si la largeur à l'époque sélectionnée est > 8 ;
- **bd** : bd_shallow, ses 40 filtres ;
- **init8** : tiny, indices 0-7.

## 4. Mesure

Identification inter-session au plus proche voisin. On interroge avec la session A et on
cherche dans la session B, puis l'inverse. Le score d'un sujet vaut 0, 0.5 ou 1. Sur
bnci, on fait la moyenne sur les graines. Le hasard vaut 1/n_sujets.

Contraste apparié par sujet, IC95 par bootstrap percentile sur les sujets (10 000 tirages,
`default_rng(0)`).

## 5. Critère principal (lee2019_mi seul, le seul dataset qui a de la puissance)

D = top-1(ajoutés) − top-1(bd).

- **EMPREINTE PROPRE À LA CROISSANCE** si la borne basse de l'IC de D est > 0.
- **PAS AU-DELÀ D'UN RÉSEAU ENTRAÎNÉ** si l'IC de D contient 0. Les ajoutés identifient
  alors le sujet autant que bd, sans plus : l'information vient de l'entraînement sur le
  sujet, pas de la croissance. La lecture 1 (embedding d'analyse) n'a pas de contenu
  propre, et seule la lecture 2 (adaptation) reste ouverte.
- **MOINS QU'UN RÉSEAU ENTRAÎNÉ** si la borne haute est < 0.

Rapportés, sans valeur décisionnelle :
- bnci2014_001 ;
- ajoutés − init8 ;
- les blocs spectre seul et topographie seule ;
- l'AUC des distances même sujet contre sujets différents ;
- le nombre de fits exclus parce que la largeur sélectionnée vaut 8.

## 7. Contrôles (avenant du 24/10, gelé avant leur lecture, `embed_controls.py`)

Le critère principal est passé (+0.306 sur lee). Trois explications banales restent à
exclure, toutes mesurées sur lee2019_mi par des contrastes top-1 appariés par sujet :

- **(a) Nombre de neurones moyennés.** bd réduit, fit par fit, à un sous-ensemble
  aléatoire de même taille que les neurones ajoutés du fit tiny apparié (même sujet,
  même session, même cv_ind ; 20 tirages, `default_rng(seed)`).
- **(b) Largeur à l'époque sélectionnée seule**, qui est un trait du sujet. Plus proche
  voisin sur |Δ largeur|, avec les égalités tranchées au hasard.
- **(c) Covariance de l'EEG brut**, avec le même prétraitement : paradigme, 8-32 Hz,
  250 Hz, le même cache. On prend la moyenne log-euclidienne des covariances d'essais
  normalisées par la trace, puis leur triangle supérieur. On mesure **cov+ajoutés − cov**.

Verdicts :
- (a) et (b) : **EMPREINTE NON TRIVIALE** si la borne basse est > 0 pour ajoutés − bd
  apparié **et** pour ajoutés − largeur seule. Sinon, **EXPLIQUÉE PAR (a) OU (b)**.
- (c) : **AU-DELÀ DE LA COVARIANCE** si la borne basse de cov+ajoutés − cov est > 0.
  Sinon, **REDONDANTE AVEC LA COVARIANCE**.

Lecture : la lecture 1 (une empreinte d'analyse nouvelle) n'a de contenu propre que si
on obtient NON TRIVIALE et AU-DELÀ. Une empreinte redondante avec la covariance ne tue pas
la lecture 2. Elle dit seulement que les neurones ajoutés encodent la statistique de
second ordre des données du sujet, ce qui est cohérent avec la façon dont TINY les
calcule.

## 6. Limites connues avant lecture

- Les poids sont ceux de l'époque sélectionnée, pas de la dernière : la largeur varie
  d'un fit à l'autre.
- Les poids spatiaux sont des filtres, pas des patterns (Haufe) : c'est une empreinte, pas
  une interprétation physiologique.
- Identifier un sujet depuis l'EEG brut est connu pour être facile (covariance). Ce test
  ne dit pas si l'embedding est **utile**, seulement s'il contient plus que le jumeau
  fixe.
- La session n'est pas testable ici : les folds d'une même session partagent environ 75 %
  de leurs données.
