# PREREG pilote EC — les architectures EEG se trompent-elles sur les mêmes essais ?

**Statut : PILOTE PRÉ-ENREGISTRÉ.** Note gelée avec `benchmarks/ec_pilot.py` (sha1 `c6d63a767c06f58557aca8550063d459d4aaad96`) avant toute lecture d'un logit ou d'un score par ce script. Rédigée le 12/11/2026.

**Révision du 12/11/2026, avant le gel, après l'audit du coordinateur. Aucun score n'avait été lu.** Ce qui a changé par rapport à la première version (sha1 `6ccf0894…`) :
- primaire : strates = vraie classe × session (au lieu de la classe seule), pondérées par leur effectif ; la classe seule passe en secondaire ;
- règle de dégénérescence : une strate où l'un des 12 vecteurs est constant est écartée pour tous les couples (au lieu d'écarter la classe paire par paire) ;
- `--read` refuse d'écraser un `EC_pilot.json` existant ;
- G0 : logits finis exigés en `--read` ;
- égalités strictes de la softmax comptées ;
- G1 : un échec sur un npz secondaire exclut ce secondaire au lieu de bloquer la lecture.

Ce pilote décide une seule chose : l'axe continue, ou il s'arrête. Ce n'est pas encore le papier.

## Pourquoi

On compare souvent des architectures EEG par leur accuracy moyenne. Mais deux réseaux peuvent avoir la même accuracy et se tromper sur des essais différents. Ou sur les mêmes.

La question est simple. Deux architectures différentes s'accordent-elles sur leurs erreurs presque autant qu'une architecture avec elle-même sous deux graines ? Si oui, c'est l'essai qui décide, pas le modèle. Le choix d'architecture compte alors peu au niveau de l'essai.

Le plafond naturel est l'accord d'un modèle avec lui-même sous une autre graine. On ne peut pas espérer qu'un autre modèle fasse mieux que ce plafond. On mesure donc un rapport à ce plafond.

## Données

Archive de la campagne A, en lecture seule : `/data/tau/iceberg_1/titanic_1/users/amounir/eegrow/campaignA`.

- **Population.** Bras `bd`, protocole `within_session`, datasets `cho2017` et `lee2019_mi`. Modèles M = {eegnex, msvtnet, ctnet, conformer}. Graines {0, 1, 2}. Cela fait 12 couples (modèle, graine) par sujet.
- **D'où viennent les chemins.** Les cellules sont lues dans `campaignA_grid/campaignA_cells.txt` (une ligne par cellule, champ 8 = chemin du CSV, `campaignA_check.py:54-63`). Le stem est le CSV sans `.csv` (`e1b_ablation.py:256`). Le jsonl est `<stem>__fits.jsonl` et les logits sont `<stem>__fits__logits/<fit_uid>.npz` (`e1b_ablation.py:256-262`, `subject_stamp.py:142`, `subject_stamp.py:185`).
- **Graines 1 et 2.** Elles figurent dans `campaignA_cells.txt` mais pas dans `campaignA_cells_planned_av5.txt` (l'avenant 4 a ramené la campagne à la graine 0). Ce sont donc des cellules partielles, lancées avant l'avenant. Les surcharges de configuration sont identiques entre les 3 graines, sauf la graine elle-même (vérifié sur les lignes des cellules `bd`, `within_session`, des 3 datasets).
- **Ce que la graine change.** Elle fixe l'initialisation (`run_moabb_hydra.py:132`) et elle est passée comme `random_state` à l'évaluation MOABB (`run_moabb_hydra.py:86`). Le découpage en plis en dépend : `WithinSessionSplitter(n_folds=5, shuffle=True, random_state=self.random_state)` (`moabb/evaluations/evaluations.py:90-95`, MOABB 1.5.0 sur Margaret). C'est pourquoi les empreintes de plis diffèrent d'une graine à l'autre.
- **Nombre de plis attendu.** Il vaut 5 × le nombre de sessions du sujet. Le 5 est `n_folds=5` (`moabb/evaluations/evaluations.py:91`), aussi écrit `n_splits: 5` dans `config/eval/within_session.yaml:3`. La règle « 5 × sessions » est celle du contrôle d'intégrité de la campagne (`campaignA_check.py:93`). Les sessions viennent du manifeste de la campagne (`/scratch/amounir/eegrow_read/manifest_A`, copié tel quel dans le dossier de travail, lu comme `campaignA_check.py:207`).

### Ce que contient un npz

`LogitScorer` écrit un npz par groupe de score (`subject_stamp.py:142-146`, `subject_stamp.py:186-190`) :
- `logits` = `net.forward` sur l'entrée du réseau, sans softmax, en float32 (`subject_stamp.py:167-168`) ;
- `y` tel que scoré, `test_idx` = les lignes des métadonnées MOABB, `subject`, `session`, `run` par essai (`subject_stamp.py:270-280`) ;
- `test_fingerprint` = SHA-1 des octets de l'entrée du réseau, puis des octets de `y` (`subject_stamp.py:182-183`) ;
- `score` et `diverged` (`subject_stamp.py:169-177`).

En within_session, un pli tient dans une seule session. Il n'y a donc qu'un groupe par fit, et le fichier s'appelle `<fit_uid>.npz` (`subject_stamp.py:185`).

### Ce qu'est `score`

`score` est `check_scoring(est, scoring=self.scoring)(est, X, y)` (`subject_stamp.py:176-177`), avec le `scoring` du paradigme. Les deux datasets utilisent le paradigme `LeftRightImagery` (`config/dataset/cho2017.yaml:4`, `config/dataset/lee2019_mi.yaml:5`). Pour ce paradigme, MOABB renvoie `"roc_auc"` (`moabb/paradigms/motor_imagery.py:128-131`).

**Sur cho2017 et lee2019_mi, `score` est donc une ROC-AUC, pas une accuracy.**

Le chemin numérique du scorer est celui de `campaignA_check.score_of` (`campaignA_check.py:66-79`) : softmax float32 des logits (la sortie de `predict_proba` de skorch), puis ROC-AUC sur `p[:, 1]`. Ce chemin a été validé sur toute la campagne. Il n'utilise pas `l1 − l0`, car la softmax float32 sature à 1,0 et le scorer classe ces essais comme des ex æquo (`campaignA_check.py:16-21`). Un réseau divergé est scoré comme le prédicteur constant (`subject_stamp.py:194-201`, `campaignA_check.py:69-71`). La porte G1 importe `score_of` tel quel.

### Comment test_idx est relié aux fenêtres

- En within_session, le runner ne charge que le sujet de la cellule : `dataset.subject_list = hit` (`run_moabb_hydra.py:145-156`). Les métadonnées MOABB sont donc celles du `paradigm.get_data` de ce seul sujet, sur toutes ses sessions.
- MOABB score `X[group_idx]` avec `group_idx = test_idx` (`moabb/evaluations/base.py:209-225`). Le npz stocke ce même `test_idx` (`subject_stamp.py:270-275`, `subject_stamp.py:187`).
- L'entrée du réseau est `X` converti en float32 par la première étape du pipeline (`pipelines.py:26-32`, `pipelines.py:347`). La première étape est appliquée par `est[:-1].transform(X)` (`subject_stamp.py:167`).
- Le constructeur de données est importé, jamais réimplémenté : `adapt_B.subject_data` (`adapt_B.py:326-344`). Il construit le jeu de données et le paradigme par `check_cache.build_campaign` (`check_cache.py:69-83`), qui reproduit le runner. Il lit le cache de la cellule (`embed_A.py:171-184`) en lecture seule. Un défaut de cache refuse (`adapt_B.py:338-340`). Il renvoie `X` en float32 dans l'ordre de `get_data`, et c'est la même convention que celle de `e1b_ablation.groups_of` (`e1b_ablation.py:274-295`). E1b a validé cette reconstruction par sa porte G0 (logits reconstruits = logits sauvés).
- `y` du npz est encodé par MOABB pli par pli, avec un `LabelEncoder` (`moabb/evaluations/base.py:144-150`). `subject_data` encode sur le sujet entier, avec les labels triés (`adapt_B.py:342`). Avec 2 classes présentes dans chaque pli, les deux encodages coïncident. La porte G-art le vérifie au lieu de le supposer.

## Mesure

### Correction

c[m, s, i] = 1 si l'essai i est bien classé par le modèle m sous la graine s.

La prédiction suit le chemin du scorer : argmax de la softmax float32 des logits (`campaignA_check.py:74`). C'est aussi ce que fait la prédiction de braindecode : `predict = predict_proba(X).argmax(1)` (`braindecode/classifier.py:196`). La colonne k est la classe k. G0 exige que les classes d'un npz soient exactement 0..C−1. On compte aussi, sans rien en décider (`ec_pilot.py:239-250`) :
- les essais où l'argmax des logits bruts diffère de celui de la softmax ;
- les essais où l'argmax de la softmax n'est pas unique (p0 == p1 à 2 classes). numpy prend alors la première colonne, comme le scorer.

### Error consistency (Geirhos et al., NeurIPS 2020)

Entre deux vecteurs de correction a et b (`ec_pilot.py:62-73`) :
- c_obs = part des essais où a et b sont tous deux corrects ou tous deux faux ;
- c_exp = p_a·p_b + (1−p_a)(1−p_b) ;
- κ = (c_obs − c_exp) / (1 − c_exp).

### Strates (`ec_pilot.py:76-118`)

**Primaire : strates = vraie classe × session.** cho2017 n'a qu'une session : la strate y est la classe seule. lee2019_mi en a deux : 4 strates par sujet.

On calcule κ à l'intérieur de chaque strate, puis on fait la moyenne pondérée par l'effectif de la strate. Les essais d'un sujet sont groupés sur toutes ses sessions.

Pourquoi conditionner :
- sur la classe : deux modèles bons sur la classe 0 et mauvais sur la classe 1 s'accordent déjà par ce seul biais de classe. Ce κ ne dirait rien des essais ;
- sur la session : si une session est plus dure pour tous les modèles, elle agit comme une difficulté partagée. Elle gonfle κ sans rien dire des essais eux-mêmes.

**Strate dégénérée** (`ec_pilot.py:90-103`). Pour un sujet et une strate, si l'UN des 12 vecteurs (modèle, graine) y est constant (tout juste ou tout faux), la strate est écartée pour TOUS les couples. Les mêmes strates entrent donc dans κ_x, κ_seed et κ_same. Dans une strate gardée, aucun vecteur n'est constant, donc c_exp < 1 et κ est fini.

Pourquoi pas l'ancienne règle (écarter la strate paire par paire) : un modèle parfait sur une classe sous ses 3 graines a c_exp = 1 avec lui-même, et la classe sort de son κ_seed. Mais avec un autre modèle, c_exp < 1 et κ vaut 0 sur cette classe. Elle reste dans κ_x. R baisse alors sans aucun effet d'architecture (contrôle 6a du selftest : R = 0,743 sous l'ancienne règle, 0,981 sous la nouvelle).

On compte et on rapporte, par dataset, les strates écartées et la part d'essais perdue, au total et par modèle responsable (`ec_pilot.py:146-157`). Un sujet sans strate valide est écarté et compté. Le refus sous 20 sujets s'applique après cette exclusion.

**Secondaires** : strates = classe seule ; et κ non conditionnel (une seule strate). La même règle de dégénérescence s'y applique.

### Quantités par sujet (`ec_pilot.py:126-143`)

- κ_seed(m) = moyenne de κ(m_s, m_s′) sur les 3 paires de graines non ordonnées. C'est le plafond de graine.
- κ_x(m, n), pour m ≠ n : moyenne de κ(m_s, n_s′) sur les 6 couples ordonnés de graines s ≠ s′. Ces paires inter-architectures traversent le **même** changement de plis que le plafond. C'est ce qui rend la comparaison loyale.
- κ_same(m, n) = moyenne sur s de κ(m_s, n_s). Secondaire, car le pli est partagé.

### Agrégats par dataset (`ec_pilot.py:160-172`)

- K_x = moyenne de κ_x sur les sujets et les 6 paires de modèles ;
- K_seed = moyenne de κ_seed sur les sujets et les 4 modèles ;
- **R = K_x / K_seed** (rapport des moyennes) ;
- IC à 95 % par bootstrap des sujets : 10 000 tirages, `default_rng(0)`, percentiles 2,5 et 97,5, mêmes indices au numérateur et au dénominateur ;
- med_x = médiane sur les sujets de la moyenne par sujet de κ_x.

On rapporte aussi le groupement des deux datasets sur le schéma `sboot` d'E1b (`e1b_ablation.py:479-497` : moyenne des datasets pondérée par √n, sujets ré-échantillonnés dans chaque dataset, `default_rng(0)`). `sboot` est importé tel quel pour K_x et K_seed. Le rapport groupé utilise une copie du même schéma, avec les mêmes tirages au numérateur et au dénominateur (`ec_pilot.py:186-208`). **Ce groupement ne décide rien.**

## Portes (refus, pas avertissement)

Une porte qui échoue arrête le script sans écrire de verdict.

**Pas d'écrasement.** `--read` refuse si `<out>/EC_pilot.json` existe déjà (`ec_pilot.py:575-576`). `--check-mapping` refuse si la sortie d'un dataset existe déjà.

**G0, par sujet** (`ec_pilot.py:286-392`). Le sujet entre seulement si, pour les 12 couples (modèle, graine) :
- nombre de npz = nombre de lignes du jsonl = 5 × sessions ;
- les noms des npz sont exactement les `fit_uid` du jsonl, et chaque ligne du jsonl porte le bon sujet ;
- chaque npz porte un seul sujet (le bon) et une seule session, et ses classes sont exactement 0..C−1 ;
- chaque (session, test_idx) apparaît exactement une fois ;
- même ensemble d'essais et mêmes y pour les 12 couples ;
- aucun `diverged`, et `np.isfinite(logits).all()` pour chaque npz (ces deux contrôles ne sont faits qu'en `--read`) ;
- empreintes identiques entre modèles à graine égale.

Les sujets exclus sont comptés et listés par motif (le premier motif rencontré). **Si un dataset garde moins de 20 sujets, le script refuse la lecture** (`ec_pilot.py:395-398`). Il refuse aussi si moins de 20 sujets gardent au moins une strate valide et un κ_x et un κ_seed finis.

**G1** (`ec_pilot.py:520-538`, `ec_pilot.py:600-608`). Le score recalculé par `campaignA_check.score_of` doit égaler `score` à 1e-6.
- Un seul échec sur un npz du **primaire** refuse la lecture.
- Un échec sur un npz d'un **secondaire** (Shallow, physionetmi) exclut ce secondaire, avec son motif écrit dans le rapport. Il ne bloque pas le primaire.

**G-art, en mode `--check-mapping`** (`ec_pilot.py:403-429`). Pour chaque sujet admis par la partie structurelle de G0, et pour chaque npz de ses 12 couples :
- les indices sont dans [0, nombre de fenêtres) ;
- le label de la fenêtre indexée par test_idx égale le y du npz, sur 100 % des essais ;
- la session et le run de la fenêtre égalent ceux du npz ;
- le SHA-1 de `X[test_idx]` (float32) puis de y égale `test_fingerprint`. C'est la preuve, bit à bit, que la fenêtre est le signal exact que le réseau a vu ;
- le nombre de fenêtres = le `n` du manifeste = le nombre d'essais distincts du sujet.

Si G-art n'est pas établie pour un sujet admis, AUC_art vaut NaN pour son dataset, et ce dataset ne peut pas être GARDE.

`--check-mapping` n'ouvre, dans les vrais npz, que `y`, `test_idx`, `subject`, `session`, `run` et `test_fingerprint` (`np.load` est paresseux : un membre non ouvert n'est jamais décompressé). Il n'ouvre ni `logits`, ni `score`, ni `diverged`.

## Contrôle d'artefact (`ec_pilot.py:432-442`, `ec_pilot.py:483-517`)

- Essais « ratés par tous » : faux pour les 12 couples. Essais « réussis par tous » : corrects pour les 12.
- Pour chaque essai, sur le signal exact que le réseau a vu, on calcule trois caractéristiques :
  - log de l'amplitude crête à crête maximale sur les canaux ;
  - log de la variance moyenne sur les canaux ;
  - kurtosis maximal sur les canaux (Fisher, `scipy.stats.kurtosis`, `bias=True` par défaut).
- On normalise chaque caractéristique par rang à l'intérieur du sujet, sur tous ses essais. Puis on calcule l'AUC « ratés par tous » contre « réussis par tous », sur les essais groupés du dataset.
- AUC_art = max sur les caractéristiques de max(AUC, 1 − AUC).
- Moins de 30 essais ratés par tous dans un dataset : NaN, ce qui compte comme un échec de porte pour GARDE.

Les caractéristiques sont calculées par `--check-mapping` et écrites par sujet (`mapping/<dataset>/<sujet>.npz`, clés (session, test_idx) triées). `--read` vérifie que ces clés sont exactement les essais qu'il lit.

## Règle de décision, écrite avant lecture (`ec_pilot.py:211-229`)

**Par dataset :**
- **GARDE** si med_x ≥ 0,30 ET R ≥ 0,85 ET borne basse de l'IC de R ≥ 0,75 ET AUC_art < 0,70 ;
- **TUE** si med_x ≤ 0,15, OU (R ≤ 0,60 ET borne haute de l'IC de R < 0,75), OU AUC_art ≥ 0,70 ;
- sinon **NON CONCLUANT**.

**Verdict global :**
- GARDE si les 2 datasets sont GARDE ;
- TUE si les 2 sont TUE, ou si AUC_art ≥ 0,70 sur l'un des deux ;
- sinon NON CONCLUANT.

Lecture : GARDE veut dire que l'axe continue. TUE veut dire qu'il s'arrête.

## Secondaires, qui ne décident rien

- κ_same et sa différence avec κ_x, qui estime l'effet du pli partagé ;
- R par paire de modèles : κ_x(m, n) / moyenne de κ_seed(m) et κ_seed(n) ;
- κ conditionnel à la classe seule, et κ non conditionnel (K_x, K_seed, R, med_x, strates écartées) ;
- eegnex sur physionetmi : plafond de graine seul. Strates classe × session, valides pour les 3 vecteurs eegnex ;
- Shallow en graine 0 : κ_same avec les 4 autres en graine 0, contre κ_same entre les 4 en graine 0. Strates classe × session, valides pour les 5 vecteurs (Shallow et les 4, graine 0) ;
- nombre d'argmax non uniques de la softmax, et d'écarts entre l'argmax des logits et celui de la softmax ;
- relation entre κ et l'accuracy du sujet (Spearman) ;
- part des sujets proches du hasard, et leur κ à part.

## Selftest (synthétique, avant toute donnée réelle)

`--selftest` passe 23 contrôles (journal `/scratch/amounir/eegrow_read/EC_dev/selftest_v3.log`) :
- deux copies identiques donnent κ = 1 ; deux vecteurs indépendants donnent κ ≈ 0 ;
- une difficulté par essai partagée par deux modèles donne κ > 0,2 ;
- un κ créé seulement par un biais de classe (κ non conditionnel > 0,2) disparaît sous le conditionnel (|κ| < 0,02) ;
- dégénérescence : un seul vecteur constant dans une strate écarte la strate pour tous les couples, le modèle responsable est nommé, la part d'essais perdue est juste, et un sujet sans strate valide est écarté ;
- une égalité stricte de la softmax (p0 == p1) est comptée ;
- sans effet d'architecture, R = 0,998 [0,979 ; 1,016] ; avec un terme architecture × essai partagé par les graines, R = 0,460 [0,446 ; 0,476] ;
- **scénario de l'auditeur** : un modèle parfait sur une classe sous ses 3 graines, sans autre effet d'architecture, donne R = 0,981 [0,955 ; 1,007]. L'ancienne règle par paire donnait 0,743 ;
- **session plus dure pour tous**, sans difficulté par essai partagée : K_x = 0,304 avec les strates « classe », +0,002 avec le primaire classe × session ;
- la règle de décision, y compris un AUC_art NaN qui bloque GARDE ;
- sur une archive synthétique écrite comme `LogitScorer` l'écrit : G0 admet l'archive propre, puis refuse un npz fantôme, un npz fantôme avec sa ligne de jsonl, un y discordant, un réseau divergé, une empreinte discordante et un logit NaN ;
- G1 refuse un score altéré de 1e-3 sur le primaire ; le même défaut sur un secondaire est compté sans bloquer ;
- `--read` refuse d'écraser un `EC_pilot.json` existant ;
- G-art refuse une permutation des fenêtres à l'intérieur des classes (tous les labels restent justes, l'empreinte la détecte), un label faux et un nombre de fenêtres incohérent ;
- l'AUC d'artefact retrouve une amplitude liée aux ratés (0,993) et reste près de 0,5 au hasard (0,562) ; elle vaut NaN avec moins de 30 ratés par tous, et NaN si un sujet n'a pas G-art.

## Résultat du check-mapping (aucun score lu)

Lancé le 12/11/2026 sur Margaret (`srun -p normal -c 4`, CPU). Journaux : `check_mapping_cho2017.log`, `check_mapping_lee2019_mi.log`. Sorties : `mapping/<dataset>.json` et `mapping/<dataset>/<sujet>.npz`.

Relancé après la révision avec le script final, dans un nouveau dossier `mapping_v2/` (journaux `check_mapping_v2_*.log`). Même résultat, et sorties identiques à `mapping/` : 32 + 29 fichiers de caractéristiques, 0 tableau différent (session, test_idx, caractéristiques, noms), json par dataset identiques. **`--read` lit `mapping_v2/`.**

| dataset | population | G0 structurelle | exclus (motif) | G-art |
|---|---|---|---|---|
| cho2017 | 52 | 32 | jsonl_absent 20 (msvtnet s1 : 3 ; eegnex s1 : 17) | 32/32 |
| lee2019_mi | 54 | 29 | jsonl_absent 19 ; compte 6 | 29/29 |

- Sur les sujets admis : 0 label faux sur 78 240 (cho2017) et 69 600 (lee2019_mi) contrôles essai × couple. 0 empreinte fausse sur 1 920 et 3 480 npz. Nombre de fenêtres = manifeste = essais distincts pour chaque sujet (200 ou 240 sur cho2017, 200 sur lee2019_mi).
- Les 6 exclus « compte » de lee2019_mi sont des fits fantômes ou incomplets : sujets 21, 38 et 50 (eegnex s1 : 12, 17 et 12 npz pour 10 lignes) ; 39 (eegnex **s0** : 12 npz pour 10 lignes) ; 31 (eegnex s1 : 6/6) ; 46 (eegnex s2 : 4/4).
- « G0 structurelle » n'inclut pas le critère `diverged`, que seul `--read` ouvre. `--read` peut donc exclure encore quelques sujets. La marge avant le refus à 20 sujets est de 12 (cho2017) et 9 (lee2019_mi).
- Les sujets admis sont les sujets aux plus petits numéros (cho2017 : 1-31 et 46 ; lee2019_mi : 1-30 sauf 21). Ce n'est pas un tirage au hasard (voir les limites).

## Limites, écrites avant lecture

- within_session seulement, 2 datasets, 4 architectures, imagerie motrice à 2 classes.
- La graine change à la fois l'initialisation et les plis. κ_seed mesure donc l'accord sous ces deux changements ensemble, pas sous la seule initialisation.
- Les sujets proches du hasard ont un κ bruité. Un sujet au hasard a peu d'erreurs « structurées » à partager.
- Les seuils viennent d'un reviewer interne. Ils ne sont pas calibrés.
- Les graines 1 et 2 sont des cellules partielles, antérieures à l'avenant 4. Les sujets qui les ont ne sont pas tirés au hasard. Le pilote porte sur ces sujets-là, pas sur tout le dataset.
- R est un rapport. Si K_seed est proche de 0, R n'est pas interprétable (R = NaN si K_seed ≤ 0, donc jamais GARDE).
- Les strates sont petites : ~100 essais par classe sur cho2017, ~50 par classe et par session sur lee2019_mi. Chaque κ par strate reste bruité ; la moyenne sur les paires et les sujets l'atténue.
- La règle de dégénérescence écarte les strates où un modèle est parfait ou nul. Ce sont souvent les sujets très bons ou très mauvais. La part d'essais perdue est rapportée par modèle responsable ; si elle est grande, le pilote ne porte plus que sur les strates « moyennes ».

## Commandes

Dossier de travail : `/scratch/amounir/eegrow_read/EC_dev/` (copie de `ec_pilot.py` et des modules importés, inchangés : `adapt_A.py`, `adapt_B.py`, `embed_A.py`, `check_cache.py`, `utils.py`, `campaignA_check.py`, `e1b_ablation.py`, `config/`, `grids/A/campaignA_cells.txt`, `manifest/` ; empreintes dans `SHA1SUMS`). `config/` est identique à celle de la copie figée d'E1b.

```
python ec_pilot.py --selftest --out selftest_v3
python ec_pilot.py --check-mapping --cells grids/A/campaignA_cells.txt --manifest manifest --datasets <ds> --out mapping_v2
python ec_pilot.py --read --cells grids/A/campaignA_cells.txt --manifest manifest --mapping mapping_v2 --out read
```

`--read` n'a pas été lancé. Il attend le go d'Adam.
