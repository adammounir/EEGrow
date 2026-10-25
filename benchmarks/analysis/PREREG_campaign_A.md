# PREREG — Campagne A : la croissance TINY sur 6 modèles × 12 datasets × 3 protocoles

25/10/2026. **GELÉ** dans le commit qui contient aussi le générateur de cellules
(`campaignA_make_cells.py`), le sbatch (`campaignA.sbatch`), le contrôleur d'intégrité
(`campaignA_check.py`), le lecteur (`campaignA_read.py`) et la grille générée
(`benchmarks/campaignA_grid/`). Le gel a eu lieu après l'essai à blanc et avant toute
cellule de la grille. TIDNet est hors de ce gel (§ 2.1).

## 1. Questions

La validation du 21/10 a fait de la règle **tiny** la base commune, VALIDÉE sur 6/6
modèles, sur 3 datasets, en intra-session seulement. La campagne A pose deux questions,
sur les 12 datasets d'ICASSP et sur les 3 protocoles :

- **Q1, la croissance contre la taille fixe.** Un modèle qui grandit avec la règle tiny
  fait-il au moins aussi bien que le même modèle braindecode entraîné à sa largeur
  publiée ? Il s'agit de la non-infériorité, et de la supériorité là où elle existe.
- **Q2, la direction.** Les directions calculées par TINY apportent-elles quelque chose ?
  Le bras **random** garde toutes les décisions de tiny : le moment, la jonction, le
  nombre de neurones et l'amplitude (E.3.1). Il tire seulement les directions au hasard
  (`loop._randomize_candidates`, après la sous-sélection et, en multi-jonction, après le
  choix de la jonction ; la normalisation du papier s'applique ensuite). Le contraste
  tiny − random isole donc la direction.

La campagne produit aussi l'archive des lectures hors ligne : lecture 1 (empreinte du
sujet et de la session) et lecture 2 (adaptation, campagne B). Chacune aura son propre
PREREG, gelé avant d'être lu. Rien de cette archive ne sert à Q1 ou Q2.

## 2. Plan

### 2.1 Bras, modèles, recettes

- **Modèles et recettes** : exactement ceux de la validation (`valid_make_cells.MODELS`,
  importés par le générateur, jamais recopiés) :

  | modèle | bras en croissance | jumeau bd | jonctions |
  |---|---|---|---|
  | Shallow | `grow_shallow` | `bd_shallow` | 1 |
  | EEGNeX | `grow_eegnex_j2` | `bd_eegnex` | 1 |
  | TIDNet | `grow_tidnet` | `bd_tidnet` | 1 |
  | MSVTNet | `grow_msvtnet_all` | `bd_msvtnet` | 2 FFN |
  | CTNet | `grow_ctnet_all` | `bd_ctnet` | 6 FFN |
  | Conformer | `grow_conformer_all` | `bd_conformer` | 6 FFN |

  Les transformers grandissent d'une jonction par pas (`select_best_update`).
- **TIDNet est hors de ce gel** (décision d'Adam, 25/10). Sa recette validée (Adam,
  décroissance L2 0,01) diverge sur les entraînements longs **dans les trois bras**, bd
  compris : 818169 (tiny et random), puis 818399 (bd à l'époque 174, tiny à 147, random
  à 178), sur bnci2014_004 en inter-sujet. Le mécanisme est au § 3.1. Une case dont le bd
  ne dépasse plus le hasard est NON LISIBLE, et les cases inter-sujet et inter-session de
  TIDNet risquaient de l'être en masse. La grille gelée ici porte donc sur les 5 autres
  modèles (`campaignA_make_cells.py --models`). TIDNet y entrera par un avenant, écrit et
  commité avant ses cellules : un correctif de recette identique pour ses trois bras,
  testé d'abord sur les cellules qui divergent. Pour la lecture, il compte comme un
  modèle de la campagne.
- **La lecture** de Q1 et Q2 attend la fin de TIDNet, ou son abandon écrit. Une seule
  lecture, sur la famille complète (§ 3.2).
  `growth_split_half` est désactivé sur MSVTNet, comme dans la validation.
- **Bras** : `bd` (jumeau braindecode), `tiny` (`tiny_paper`, s = 1, grille enregistrée
  {0, .25, .5, 1, 2, 4}) et `random` (= tiny + `train.growth_direction=random`).
- **Protocole d'entraînement**, celui de la validation (`valid_battery.sbatch`) :
  - 200 époques, sans arrêt anticipé (patience 200), sans schedule ;
  - held-out de croissance de 10 %, croissance toutes les 5 époques (époques 5 à 195) ;
  - `selection_monitor=valid_acc`, `smooth_window=5`, `tie_break=earliest`,
    `restore_selected=false` : le réseau scoré est celui de la **dernière époque**.
  - `EEGROW_DEBUG_GROWTH` n'est pas activé : il n'écrit que des traces de diagnostic
    (`skorch_integration._debug_params`), sans effet sur l'entraînement.

### 2.2 Données

- **Datasets** (ICASSP) : alexmi, bnci2014_001, bnci2014_002, bnci2014_004,
  bnci2015_001, cho2017, lee2019_mi, physionetmi, schirrmeister2017, shin2017a,
  weibo2014, zhou2016.
- **Un cache neuf, construit depuis le brut et vérifié.** L'audit du 25/10 a trouvé le
  cache partagé (`/scratch/amounir/moabb_cache`) **tronqué sans erreur** :
  bnci2014_004 sujet 1 servait 1 session sur 5 (160 essais sur 720), physionetmi
  sujet 7 servait 30 essais sur 45. Le cache BIDS de MOABB sert les sessions et les runs
  qu'il trouve sous la clé (`bids_interface.load`), et `check_cache.py` ne voit que les
  sessions entières manquantes. La campagne lit donc son propre cache,
  `/scratch/amounir/moabb_cache_A` :
  - il est construit par `campaignA_manifest.py --build`, un processus par dataset, depuis
    le brut ;
  - chaque sujet relu depuis ce cache doit être identique bit à bit à ce qui vient d'être
    dérivé ;
  - la comparaison avec l'ancien cache (`old_cache` dans le manifeste) liste les sujets
    que les campagnes précédentes ont lus tronqués. C'est une information pour ces
    campagnes, sans effet sur celle-ci.
  - **Exception, lee2019_mi** : son brut a disparu de `MNE_DATA`, et le retélécharger
    coûterait ~65 Go et ~13 h à 1,4 Mo/s. Ce dataset est lu dans l'ancien cache, en
    lecture seule. Le manifeste porte alors `built: false` et le chemin de ce cache, et le
    générateur prend `cache.path` dans le manifeste de chaque dataset. Les preuves :
    - les sujets dérivés du brut avant l'arrêt de la construction sont identiques bit à
      bit à l'ancien cache : **sujets 1 et 2** (818104_6, 200 essais chacun, sessions 0 et 1) ;
    - les 54 sujets ont exactement la structure du dataset : 2 sessions × 100 essais,
      50 par classe ;
    - les troncatures trouvées ailleurs sont des sessions ou des runs manquants, donc
      visibles dans ces comptes.
- **En lecture seule pendant la campagne** (`cache.readonly=true`, `utils.cache_config`) :
  aucun job n'écrit dans le cache. Une entrée manquante serait dérivée en mémoire et non
  sauvée (lu dans `moabb/datasets/base.py`, `_get_single_subject_data_using_cache`). Il
  n'y a donc aucune collision possible entre jobs (incident du 16/08, `check_cache.py`).
- **Le manifeste** (`manifest_A/<dataset>.json`) est la seule source des sujets, des
  sessions, des essais par classe et de la forme d'entrée. Le générateur refuse :
  - une session qui ne peut pas être découpée en 5 plis stratifiés ;
  - un dataset qui mélange des sujets à 1 session et des sujets à plusieurs sessions.
- **Protocoles** :
  - within_session et cross_subject sur les 12 datasets ;
  - cross_session sur les datasets dont tous les sujets ont au moins 2 sessions, d'après
    le manifeste : **bnci2014_001, bnci2014_004, bnci2015_001, lee2019_mi, shin2017a,
    zhou2016**.
- **Graines** : 0-2. Elles vont jusqu'à 4 sur zhou2016, alexmi et bnci2014_001.

### 2.3 Cellules, tâches, cartes

- **Une cellule** = (protocole, dataset, modèle, bras, graine, sujet) = un appel
  `run_cell.py` avec `only_subject` :
  - en intra-session et en inter-session, c'est le sujet évalué (seul ce sujet est
    chargé) ;
  - en inter-sujet, c'est le sujet tenu à l'écart (tous les sujets sont chargés, MOABB
    construit les plis sur la liste complète, un seul pli tourne).
- **Un dossier de résultats par bras** : `<racine>/<modèle>__<bras>/results/<protocole>/<dataset>/`.
  Le nom de cellule ne contient pas la direction ; le 25/10, tiny et random partageant un
  dossier s'étaient mélangés dans un même JSONL. Le générateur refuse deux cellules qui
  écriraient au même endroit, et le lecteur refuse un JSONL dont la direction n'est pas
  celle du bras.
- **Racine** : `/data/tau/iceberg_1/titanic_1/users/amounir/eegrow/campaignA` (iceberg,
  budget eegrow ≤ 1 To décidé le 25/10).
- **Tâches** : une tâche SLURM exécute une liste de cellules en série. Les petites
  cellules d'un même (protocole, dataset, modèle, bras, graine) sont groupées jusqu'à
  ~2 h estimées (MaxArraySize 1001 et MaxJobCount 10 000 sur Margaret). Le modèle de
  coût (sonde de temps 816443, étalonnée sur les temps mesurés de la validation) ne sert
  qu'à grouper les cellules et à fixer la limite de temps : ×2,05 pour la P100, puis une
  marge ×2.
- **Mémoire hôte** : 16 Go par défaut. En inter-sujet, 1,5 × le MaxRSS de la sonde :
  lee2019_mi 40 Go, cho2017 32 Go. Pour schirrmeister2017, la sonde avait buté sur la
  limite de 64 Go, mais elle dérivait et écrivait aussi le cache. L'essai 818344 (les 12
  cellules inter-sujet des 6 modèles × bd/tiny, cache en lecture seule) mesure un MaxRSS
  maximal de 51,1 GiB → **80 Go**.
- **Mémoire GPU** : la plus petite carte admise est la 2080 Ti (10,6 GiB). schirrmeister2017
  a le plus grand C × T des 12 datasets (128 × 1000). La mémoire de croissance s'accumule
  batch par batch (`compute_statistics`) : elle dépend du batch et de la largeur atteinte,
  pas du nombre d'essais. Pics mesurés (`max_memory_reserved`) : inter-sujet, 6 époques
  (818344) : EEGNeX 9,41 GiB, Conformer 8,29, TIDNet 8,06 ; intra-session, 200 époques,
  largeur finale atteinte (818319) : shallow tiny 9,32 GiB sur 2080 Ti.
- **Reprise** : `tau` a la priorité 100 sur gpu-best et gpu, et la préemption remet en
  file (`PreemptMode=REQUEUE`). Une tâche reprise saute les cellules dont le CSV existe
  et recommence la cellule interrompue depuis le début. Un JSONL partiel est mis de côté
  (`.aborted.<ts>`). Les fichiers annexes d'un essai interrompu restent orphelins : ils
  ne sont joignables que par l'identifiant de fit d'une ligne JSONL.
- **Cartes et environnements** (`campaignA.sbatch`) :
  - sm ≥ 7.5 (turing, rtx, ampere, hopper) → `bench` (cu130) ;
  - Tesla V100-PCIE-32GB et Tesla P100-PCIE-16GB → `bench126` (cu126). Ce sont exactement
    les deux cartes du test d'équivalence (`PREREG_env_equivalence.md`, lu le 25/10).
  - Tout le reste est REFUSÉ par la porte, qui teste le nom de la carte. Nœuds exclus :
    - margpu020 (carte morte, 22/10) ;
    - margpu024 à 027 (GTX 1080 Ti seulement) ;
    - margpu004 (V100S) et margpu032 (V100-SXM2), jamais testées sous `bench126`.
  - Avant toute cellule, une sonde CUDA (matmul + conv). Une carte qui échoue est tenue
    60 s puis la tâche est remise en file, 8 fois au plus.
  - Chaque fit enregistre `gpu_name`, chaque ligne de CSV la version de torch.
- **Ordre** : la vague canari passe d'abord (graine 0 de zhou2016 et de bnci2014_004, les
  3 protocoles, les 18 bras). `campaignA_check.py` la vérifie (intégrité seulement, aucun
  contraste n'est lu) avant que le reste soit soumis, les tâches les plus longues en
  premier.
- **Taille de la grille** (`benchmarks/campaignA_grid/campaignA_plan.json`, 5 modèles sans
  TIDNet) : 36 075 cellules. Par bras : 5 070 intra-session, 1 885 inter-session, 5 070
  inter-sujet. Elles sont groupées en 6 601 tâches et 18 arrays, dont 90 tâches canari.
  - Fits : 119 325 intra-session, 14 310 inter-session, 15 210 inter-sujet.
  - Coût estimé : 9 721 GPU-h en équivalent 2080 Ti, dont 8 412 en inter-sujet.
  - Cellule la plus longue : EEGNeX tiny en inter-sujet sur schirrmeister2017, 4,7 h
    estimées.
  - Archive estimée d'après les tailles mesurées pendant l'essai : ~0,6-0,7 To.
- **Essai à blanc** (25/10, avant le gel) :
  - 818169 + 818170 : 54 cellules à 200 époques, `campaignA_check` passe sur 52. Les deux
    autres sont les cellules NaN de TIDNet grandi, d'où la règle du § 3.1.
  - Reprise : 818170_15 a été remise en file de force en plein calcul, et la cellule
    s'est refaite proprement.
  - 818171 / 818172 : `bench126` sur V100 et P100.
  - 818319 / 818344 : mémoire GPU et RAM sur schirrmeister2017.
  - 818399 : les 18 cellules inter-sujet de bnci2014_004, sur le code d'exécution gelé
    (ef63957 ; d1f4535 ne change que le générateur et le lecteur). `campaignA_check`
    passe sur les 18, et les 3 fits TIDNet divergés sont scorés au prédicteur constant.
  - pytest 818397 : 272/272 sur nœud CPU.

### 2.4 Archive par fit (option B, décidée le 25/10)

Sur toutes les cellules :
- les poids à l'init, à l'époque sélectionnée et à la dernière époque ;
- à chaque croissance appliquée, les tenseurs redimensionnés tels qu'ils sont juste
  après la croissance ;
- les logits bruts de chaque essai de test, avec l'identité du pli de test : indices,
  sujet, session, run, empreinte SHA-1 et score ;
- la ligne JSONL du fit (courbes, décisions de croissance, carte, sujets et sessions
  d'entraînement et de test).

En inter-session et en inter-sujet seulement, les poids complets aux époques 10, 25, 50
et 100. Le checkpoint de l'époque e est l'état **à la fin** de l'époque e, donc après la
croissance de cette époque. Ces quatre époques sont des multiples de 5 : un checkpoint
contient les neurones nés à cette époque, avant leur premier pas d'entraînement.

## 3. Critères (figés au gel)

L'unité est le sujet :
- dans un bras, le score d'un sujet est la moyenne de ses lignes CSV (sessions) par
  graine, puis la moyenne sur les graines prévues ;
- les contrastes sont appariés par sujet ;
- les IC à 95 % sont des bootstrap percentile sur les sujets (10 000 tirages,
  `default_rng(0)`).

La métrique est celle de MOABB : accuracy au-delà de 2 classes, ROC-AUC à 2 classes. Le
score principal est celui de la **dernière époque**. Le lecteur est `campaignA_read.py`
et il ne tourne qu'après un `campaignA_check.py` sans problème sur toutes les cellules
terminées.

### 3.1 Par (modèle, dataset, protocole), noté « case »

Pour Δ = moyenne par sujet de (A − B) :

| verdict | condition |
|---|---|
| SUPÉRIEUR | borne basse de l'IC > 0 |
| NON INFÉRIEUR | borne basse > −0.02 |
| INFÉRIEUR | borne haute < 0 et borne basse ≤ −0.02 |
| NON CONCLUANT | sinon, rapporté avec sa MDE (2,80 × σ(Δ)/√n) |

- Q1 : A = tiny, B = bd.
- Q2 : A = tiny, B = random.
- **Lisibilité** : le jumeau bd doit apprendre, c'est-à-dire que la borne basse de
  l'IC de (bd − hasard) doit être > 0. Le hasard vaut 0.5 en AUC et 1/n_classes en
  accuracy ([[roc-auc-chance-vs-nclasses]]). Une case non lisible est rapportée, mais
  sort des comptes.
- **Complétude** : les deux bras comparés ont tous les sujets prévus avec toutes les
  graines prévues. Une case incomplète est rapportée INCOMPLÈTE et sort des comptes.
- **Stabilité et croissance**, comptées par bras et par case, mais elles ne retirent
  aucun score : un fit instable compte avec son score, puisque l'instabilité fait partie
  de la méthode. Un fit dont les sorties de test ne sont pas finies (réseau divergé)
  reçoit le score du prédicteur constant : 0.5 en ROC-AUC, la part de la première classe
  en accuracy. Le fichier de logits le marque `diverged`. Raison : l'essai à blanc
  (818169) a vu TIDNet grandi passer en NaN à l'époque 160-178 en inter-sujet. Le
  ROC-AUC de sklearn refuse alors les probabilités, et la cellule meurt sans CSV ;
  l'accuracy, elle, aurait pris l'argmax des NaN (classe 0) en silence. Le même
  événement doit recevoir le même traitement dans les deux métriques. Mécanisme mesuré
  (818304, hooks sur chaque pas) : ce n'est pas la croissance. Dans les trois bras, bd
  compris, un filtre de la convolution à weight norm `dscnn.temporal.2.net.0.0` s'éteint
  (‖v‖ de 0.2 à l'époque 10, 7e-7 en bd et 2e-8 en tiny à l'époque 100) : la
  décroissance L2 d'Adam ramène v vers 0, et le gradient du weight norm, en g/‖v‖,
  finit par saturer le float32 (3.4e38), puis Adam donne inf/inf = NaN. Le bras grandi
  franchit le seuil plus tôt. La recette de TIDNet (validée, `valid_make_cells`) n'est
  pas changée : la même règle vaut pour les trois bras, et le score secondaire (époque
  sélectionnée, sm5, jamais un instantané non fini) reste défini. On rapporte :
  - les fits divergés (logits non finis), par bras et par case ;
  - les pertes non finies ;
  - `grad_norm_max_fit` > 1e6 ;
  - pour tiny et random, la part des fits dont la largeur finale dépasse la largeur
    initiale.

### 3.2 Endpoints groupés, par (modèle, protocole) : au plus 6 × 3 = 18 tests par question

- Δ̄ est la moyenne des Δ des datasets lisibles et complets, chaque dataset ayant le même
  poids.
- Son IC vient d'un bootstrap stratifié : on rééchantillonne les sujets à l'intérieur de
  chaque dataset, avec 10 000 tirages.
- Le p bootstrap unilatéral est la part des tirages ≤ 0 pour la supériorité, ≤ −0.02 pour
  la non-infériorité.
- La correction de Holm est faite à α = 0,025 unilatéral (la queue de l'IC à 95 %). La
  famille compte, par type de test et par question, les endpoints qui ont au moins un
  dataset lisible et complet : 18 au plus, 15 si TIDNet est abandonné.

Verdicts :
- **Q1** : SUPÉRIEURE si Holm rejette sur la supériorité ; sinon NON INFÉRIEURE si Holm
  rejette sur la non-infériorité ; sinon INFÉRIEURE si la borne haute est < 0 et la borne
  basse ≤ −0.02 ; sinon NON CONCLUANTE.
- **Q2** : « LA DIRECTION COMPTE » si Holm rejette sur la supériorité de tiny − random ;
  « RANDOM MEILLEUR » si la borne haute est < 0 et la borne basse ≤ −0.02 ; sinon NON
  CONCLUANT, rapporté avec la MDE (2,80 × l'écart-type bootstrap) avant de parler de nul
  ([[underpowered-not-null]]).

### 3.3 Secondaires, pré-enregistrés mais qui ne décident rien

- Les mêmes contrastes au score de l'**époque sélectionnée**, re-scoré hors ligne depuis
  les poids `__selected` sur les essais de test archivés (`test_idx`). Le script sera
  commité avant d'être lancé.
- La largeur finale par rapport à la cible, les paramètres, le temps de calcul et
  l'époque sélectionnée, par bras.
- **Contrôle carte** : la répartition de `gpu_name` par bras et par case est rapportée.
  Le contraste Q1 est recalculé sur les seuls sujets dont tous les fits des deux bras ont
  tourné sur une seule famille de cartes. S'il change de verdict, on le rapporte.

## 4. Puissance

- Les petits datasets (zhou2016 avec 4 sujets, alexmi 8, bnci 9) ne portent aucun
  verdict seuls. Ils comptent dans Δ̄ avec leur IC.
- La MDE par case est rapportée avec chaque verdict NON CONCLUANT.

## 5. Ce qui ne sera pas fait

- Aucune cellule ne sera relancée pour « voir ». Seuls les échecs techniques (crash,
  préemption, carte) sont relancés, et ils sont rapportés. Une cellule qui échoue deux
  fois sur la même erreur déterministe reste absente, et sa case est INCOMPLÈTE.
- Aucune recette, aucun seuil et aucune liste de datasets ne sera changé après lecture.
- On n'annoncera pas de gain de la croissance sur un endpoint qui n'est pas listé ici.
