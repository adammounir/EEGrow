# PREREG — E1b : le transformer complet a-t-il besoin des FFN que la croissance laisse étroits ?

09/11/2026. Gelé avec `e1b_ablation.py` et `e1b_cpu.sbatch` dans le même commit, AVANT
qu'aucun score d'ablation ne soit lu. CPU, lecture seule sur l'archive de la campagne A,
aucun entraînement. Fait suite à E1 (`analysis/EXPLO_internal_E1.md`, `read/E1.md`).

Avant le gel, j'ai lu :
- les clés des .pt d'un fit par (modèle, bras) sur bnci2014_001 ;
- la sortie du selftest, qui ne contient que des contrôles synthétiques, des temps, des
  comptes de fits et la porte G0, sans aucun score d'ablation.

Le selftest écrit pourtant des scores de bnci2014_001/1 et lee2019_mi/1 dans un dossier
d'essai. Ils n'ont pas été ouverts, et la lecture repart d'un dossier neuf.

Ce que le selftest a montré (jobs 916989 puis 916997) :
- Le premier essai a refusé une clé inattendue : un modèle grandi sauve aussi
  `pairs.{i}.*`, des alias des mêmes modules (`MultiJunctionGrowingModel.pairs`). Le
  script les compare désormais aux poids du bloc i, refuse s'ils diffèrent, puis les
  écarte.
- Second essai :
  - 17 contrôles synthétiques passés ;
  - 156 fits sur 2 sujets, 0 exclu ;
  - G0 : écart maximal 9e-6 de l'échelle des logits ;
  - 8 à 10 minutes et 2,1 Go de RAM par sujet, sur 4 CPU.

## 1. Pourquoi

E1 montre trois choses :
- dans Conformer, la croissance met 96 % de ses étapes dans le FFN du bloc 0 et finit à
  21-28 % de la largeur FFN de bd, sans perdre d'accuracy (campagne A) ;
- les FFN de bd bougent à peine depuis l'init (3-10 %), le plus au bloc 0 ;
- l'ordre des blocs suit l'allocation de la croissance (Spearman +0,91).

Deux limites :
- bouger peu ne veut pas dire servir à rien : un FFN resté aléatoire peut servir de
  projection fixe ;
- la corrélation peut venir d'une cause commune, le gradient.

Seule une ablation fonctionnelle tranche, c'est-à-dire couper ces FFN au test. Si les FFN
profonds de bd sont superflus, et si la répartition trouvée par la croissance vaut mieux
qu'une répartition uniforme à budget égal, la croissance devient une sonde bon marché
pour dimensionner un transformer. C'est la piste « technique réutilisable » (direction
ICML). Si elle survit, l'étape suivante (E1c, GPU, PREREG à part) entraînera Conformer
depuis zéro à ces largeurs.

## 2. Données

**Fits.** Campagne A (`campaignA_cells_planned_av5.txt`), modèles Conformer, CTNet et
MSVTNet, bras bd et tiny, 3 protocoles, graine 0, poids `__fits__final` (époque 200, ce
que la campagne score). Une tâche = un (dataset, sujet) : 324 tâches.

**Essais de test.** Ce sont ceux que la campagne a scorés (`__fits__logits`), lus dans
le cache en lecture seule de la cellule (`adapt_B.subject_data`).
- within et cross-session : `test_idx` indexe les lignes de `get_data` du sujet seul,
  car la cellule ne charge que lui (`run_moabb_hydra.py`) ;
- cross-subject : `test_idx` indexe le dataset entier, donc on prend les essais de la
  session du sujet, comme la lecture 3.

**Reconstruction.**
- bd : modèle braindecode, défauts publiés.
- tiny : même modèle, FFN redimensionnés aux largeurs grandies. Une GrowingPair calcule
  `second(drop(act(first(x))))`, c'est-à-dire la même fonction.
- Le chargement est strict.

**Porte G0 (règle d'`adapt_B.g0`).** Sur chaque groupe scoré, le modèle reconstruit doit
redonner les logits archivés :
- |Δlogit| ≤ 1e-2 × l'échelle des logits ;
- argmax égal, sauf sur les quasi-égalités ;
- |Δscore| ≤ 5e-3.

**Exclusions.** Un fit est exclu, et compté par motif, si :
- un de ses groupes échoue G0 ;
- ses poids ou ses logits archivés sont non finis ;
- son ensemble de test ne correspond pas (longueur, y) ;
- un test à 2 classes n'en contient qu'une.

**Refus.** Si plus de 5 % des fits d'un (modèle, bras, protocole) échouent G0, ses
endpoints sont REFUSÉS. La lecture refuse tout verdict si une tâche manque.

## 3. Conditions (par fit et par groupe de test)

Chaque FFN est `lin1 → act → dropout → lin2`, et sa sortie revient par le résiduel du
bloc (Conformer `x + FFN(LN(x))`, CTNet `LN(x + FFN(x))`, MSVTNet
`nn.TransformerEncoderLayer`). Un hook sur `lin2` réalise toutes les conditions.

- `full` : rien de coupé.
- `m{j}` : **ablation moyenne** du FFN du bloc j. Sa sortie est remplacée par sa sortie
  moyenne, calculée sur les essais du groupe et toutes les positions dans le passage
  `full`.
- `m_deep` (blocs 1 à B−1) et `m_all` : idem, sur plusieurs blocs.
- `z{j}`, `z_deep`, `z_all` : **ablation zéro**, la branche FFN est supprimée.
- bd seulement, si un fit tiny apparié existe (même protocole, `cv_ind`, session,
  sessions de test et `test_idx`) :
  - `p_tiny` : chaque bloc garde, parmi ses 160 neurones (72 pour MSVTNet), les w_j
    d'importance s_j = ‖W1[j]‖·‖W2[:,j]‖ la plus grande (mesure d'E1). w_j est la
    largeur finale du bloc j dans le fit tiny apparié. Les neurones retirés sont
    remplacés par leur activation moyenne ;
  - `p_unif` : même total Σw_j, réparti également entre les blocs. Le reste va un
    neurone par bloc en partant du DERNIER bloc, pour ne jamais favoriser le bloc 0.

Le score est celui de la campagne (`campaignA_check.score_of`) : ROC-AUC à 2 classes,
accuracy sinon.

## 4. Endpoints primaires (Conformer, bras bd)

Valeur d'un fit : moyenne sur ses groupes de la différence de scores. Valeur d'un sujet :
moyenne de ses fits. Regroupement sur les datasets : poids √n_d, bootstrap stratifié
10 000 tirages, `default_rng(0)`. Holm à α = 0,025 unilatéral sur les 3 protocoles, par
hypothèse.

- **H1 — les FFN profonds de bd sont superflus.**
  - Contraste : `m_deep − full`.
  - p_NI = part des tirages ≤ −0,02 (marge de la campagne A).
  - NON INFÉRIEUR si Holm rejette ; INFÉRIEUR si la borne haute est < 0 et la borne
    basse ≤ −0,02 ; sinon NON CONCLUANT.
- **H2 — le bloc 0 compte plus que les cinq autres ensemble.**
  - Contraste : `m_deep − m0`, test de supériorité.
  - SUPÉRIEUR si Holm rejette ; INFÉRIEUR si la borne haute est < 0 ; sinon NON
    CONCLUANT, avec la MDE.
- **H3 — la répartition de la croissance vaut mieux que l'uniforme, à budget égal.**
  - Contraste : `p_tiny − p_unif`, même règle que H2.

**Attendu d'après E1, écrit avant la lecture :** H1 NI, H2 SUP, H3 SUP. La suite dépend
du résultat :
- **H1 NI et H3 SUP** sur au moins 2 protocoles sur 3 : proposer E1c (Conformer entraîné
  depuis zéro à la largeur de la croissance, contre l'uniforme à budget égal, GPU).
- **H1 INFÉRIEUR** : les FFN profonds de bd servent, même presque aléatoires (hypothèse
  de la projection fixe). La croissance y gagne quand même, puisque tiny ≈ bd.
  « bd n'en a pas besoin » ne tient plus ; E1c reste la seule façon de tester la
  répartition elle-même.
- **H3 non SUP** : la répartition de la croissance n'a rien de spécial à budget égal, et
  la piste « sonde de dimensionnement » s'affaiblit.

## 5. Secondaires (rapportés, ne décident rien)

- Pour chaque (modèle, bras, protocole), contrastes par rapport à `full` de :
  - chaque `m{j}` (profil de coût par bloc) ;
  - `m_deep`, `m_all`, `z_deep`, `z_all`.
- `m_deep − m0` et `z_deep − z0`.
- bd : `p_tiny − full`, `p_unif − full`, `p_tiny − p_unif`.
- Spearman, par fit bd apparié, entre le coût d'ablation par bloc s(full) − s(m_j) et la
  fraction de largeur tiny du bloc ; moyenne par sujet, puis regroupement.
- CTNet et MSVTNet : mêmes mesures. MSVTNet grandit presque jusqu'à la cible, donc
  `p_tiny` y élague à peine.

## 6. Limites connues avant lecture

- La moyenne de l'ablation moyenne est prise sur les essais de test. C'est un vecteur
  constant par bloc, et la référence `full` utilise les mêmes essais.
- Élaguer sans réentraîner sous-estime ce qu'un modèle étroit entraîné ferait : H3
  compare deux élagages, pas deux entraînements (c'est le rôle d'E1c).
- s_j est une importance par les poids. Chez bd, les FFN profonds sont proches de l'init,
  donc ce classement y est presque arbitraire, dans les deux répartitions.
- Graine 0 partout : pas de variance de graine.
- 2 classes = ROC-AUC, sinon accuracy : les Δ mélangent deux échelles, comme dans la
  campagne A.
