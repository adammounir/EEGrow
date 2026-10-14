# Cadrage — croissance deep fidèle, sélection inter-datasets, transfert aux modèles de fondation

**Écrit le 03/10/2026. Rien n'est lancé.** Ce document précède les pré-registrations : chaque
étape ci-dessous aura son propre `PREREG_*.md`, écrit et horodaté avant son premier fit. Une
étape ne démarre que si la précédente a passé ses critères. Un critère qui tombe **refuse**, il
n'avertit pas.

Périmètre décidé avec Adam : **MI uniquement**. Un autre paradigme n'apparaît qu'à l'étape 5,
conditionnelle, comme réplication hors échantillon — jamais mélangé à la MI.

---

## 1. Pourquoi relancer : ce que le premier benchmark n'a jamais testé

Le chantier donneur → receveur (point 4, 11-12/09) a conclu que la sonde de croissance ne
choisit pas mieux les donneurs que l'accuracy, y compris pour un modèle de fondation. Cette
conclusion porte sur **une sonde shallow**, et la campagne ne contenait aucun modèle deep à la
fois fidèle et réellement en croissance :

| bras | ce que c'est réellement | source |
|---|---|---|
| `grow_shallow`, `grow_sccnet` | shallow par construction | — |
| `grow_eegnex` | pas apparié en largeur à son jumeau | mémoire `eegnex-width-mismatch` |
| `grow_deep` | réseau maison à 2 étages, largeurs 8 → 32 → 8, sans dropout ; **pas** Deep4Net (4 blocs, 25/50/100/200 filtres, 3 dropouts) | `src/eegrow/models/growing_deepeeg.py` |
| croissance de `grow_deep` | `w2=8` ⇒ 8 candidats pour 8 places, `kept/proposed = 1.0` sur 1 116 cellules : élargissement programmé, **aucune sélection** par gromo | mémoire `gromo-candidate-rank-fanout` |
| `bd_deep4` (référence fixe) | n'ajuste pas ses propres données d'entraînement, scoré à l'époque 4 sous la config par défaut | mémoire `deep4-instability` |
| sonde du point 4 | `grow_shallow` seul (`ranking_<ds>.csv`) | `PREREG_point4_foundation_selection.md` |

**Hypothèse d'Adam.** Sans croissance deep fidèle, on n'a jamais mesuré le signal qui aurait une
chance de transférer à un modèle de fondation, lui-même profond.

**Ce qui la nuance, à garder en tête sans l'écarter.**
- Le contrôle positif shallow (array 593557) a échoué **sur son propre terrain** : une sonde
  shallow qui choisit des donneurs pour un receveur shallow ne bat pas l'accuracy
  (A−B +0.0070 [−0.0078, +0.0212], TOST passe à ±0.026). Une sonde deep qui ferait mieux est
  une hypothèse, pas un acquis : l'étape 2 la teste avant tout GPU de modèle de fondation.
- Le signal n'était pas la taille. À durée d'apprentissage contrôlée (`restored_epoch`), le
  nombre de paramètres finaux meurt (+0.069 [−0.070, +0.204]) ; `restored_epoch` survit
  (mémoire `restored-epoch-supersedes-params`).

## 2. Le verrou structurel : dans un dataset, sélectionner perd par construction

Mémoire `selection-ceiling`, 36/36 : même le meilleur sous-ensemble de K sujets observé fait
moins bien que le vivier entier. Dans un dataset homogène, aucun donneur ne nuit assez pour
qu'il soit payant de l'exclure. Le préreg du point 4 contournait ce verrou en **imposant K de
l'extérieur** (coût d'adaptation) et en ne testant que « lesquels, à K fixé ».

**Le mélange inter-datasets est le seul régime où ce verrou peut céder** : entre datasets
hétérogènes (montages, sessions, consignes), ajouter des données peut faire baisser la
performance (transfert négatif). L'étape 3 mesure si c'est le cas **avant** de construire quoi
que ce soit dessus.

## 3. Règles transverses (valent pour toutes les étapes)

- **Unité d'analyse = le sujet** (receveur), seeds moyennées dedans, IC bootstrap sur les
  sujets, Holm sur les familles déclarées.
- **MDE calculé avant**, cible écrite avant. Un nul sous la MDE se rapporte comme « non
  résolu » avec la borne haute de l'IC.
- **Chaque porte a son calcul de faisabilité** : une porte plus exigeante que l'endpoint
  qu'elle protège est mal posée (G3 du point 4 refusait par construction).
- **Sujets de développement ≠ sujets d'évaluation.** Tout réglage (lr, patience, cadence de
  croissance…) se fait sur un dataset de développement qui ne porte aucun endpoint :
  **`bnci2014_001`** (tranché le 05/10, cf. § 5).
- **Code figé avant soumission**, checkout cluster dédié, jamais écrasé pendant un job.
- **Coût mesuré sur un pilote**, jamais transposé.
- **Échelle des données vérifiée** (µV, pas V) avant tout fit : piège `return_epochs=True`.
- `dreyer2023a` et `liu2024` restent **réservés** comme réplication fraîche de la question
  « modèle de fondation » (ils ont servi à claim 2, jamais au point 4).

---

## 4. Les étapes

### Étape 0 — Batterie d'entraînement (aucun résultat scientifique, que des refus possibles)

**Question.** Chaque modèle retenu, fixe ou en croissance, s'entraîne-t-il correctement ?

Sur le dataset de développement, pour chaque modèle :

| critère | refus si |
|---|---|
| départ | perte initiale à plus de 0.15 nat de ln(n_classes) sans explication (init du classifieur) |
| capacité | sur-apprentissage d'un petit sous-ensemble impossible (perte train ne descend pas sous 0.05) |
| optimum | `restored_epoch` collé au début (≤ 5) ou à la fin (≥ budget − 5) sur plus de 20 % des fits |
| référence publiée | ShallowFBCSPNet et Deep4Net hors de l'IC du chiffre publié (Schirrmeister 2017, **à relever dans le PDF**, pas de mémoire) |
| hyperparamètres | lr, batch, patience, `selection_monitor`, budget d'époques réglés sur le dev **puis gelés** et écrits dans le préreg |
| effet de la carte | d'abord **sans GPU** : sur les fits existants (`dynamics_final`, cartes déjà hétérogènes), part de variance de la sonde expliquée par le modèle de carte une fois le sujet pris en compte, comparée à celle des seeds. Rejeu sur deux cartes seulement si cette part n'est pas négligeable, et alors sur quelques fits |

En plus, pour chaque bras en croissance :

| critère | refus si |
|---|---|
| sélection réelle | `kept/proposed = 1.0` (gromo ne sélectionne pas) ; règle : fan-out du module ≥ 2× les places à remplir |
| spectre | valeurs singulières des candidats sur moins d'une décade |
| cible | largeur cible atteinte sur moins de 80 % des fits (le plafond d'août venait de l'early stopping) |
| calendrier | tous les événements de croissance dans les 10 premières époques |
| jumeau | pas de jumeau fixe à la largeur finale exacte, même architecture |

**Hyperparamètres : d'où ils viennent (règle tranchée le 04/10).**
- Une config **par modèle, la même sur tous les datasets** (pratique des benchmarks MOABB /
  braindecode). Pas de réglage par dataset.
- Modèles publiés : valeurs du papier ou des défauts braindecode, citées, pas réglées.
- Ce qui dépend légitimement du dataset est rendu invariant par construction, pas réglé :
  fréquence d'échantillonnage commune (les noyaux temporels gardent leur sens), budget
  exprimé en pas de gradient plutôt qu'en époques si les tailles de datasets l'exigent.
- Hyperparamètres de croissance (`grow_every`, plancher singulier, largeur cible, part
  held-out) : aucune valeur publiée, donc calibrés.
- **Calibrer sur le comportement ne fuit pas, calibrer sur l'accuracy fuit.** Les critères
  de comportement de cette étape (courbes, cible atteinte, sélection réelle…) sont écrits
  avant et peuvent être vérifiés sur des sujets de n'importe quel dataset. Tout réglage qui
  regarde une **accuracy** se fait sur le dataset de développement seul.

**Avant tout fit : catalogue des courbes de croissance anormales de la campagne précédente**,
lu dans `dynamics_final` (zéro GPU). Chaque type d'anomalie devient un critère de refus
ci-dessus, avec sa définition chiffrée.

Premier relevé, 04/10, sur `gd_curves_mean.csv.gz` (1 116 courbes, moyennes sur les folds) :

| constat | mesure | conséquence pour la campagne |
|---|---|---|
| vraies explosions de gradient | par fold : **692 fits** avec `grad_norm_max > 1e3`, jusqu'à 1e12, toutes `fix_`/`grow_` shallow et sccnet ; **0** sur les `bd_*` et la famille deep. Groupées par process : dans une cellule, souvent une seule seed sur trois explose (ex. zhou2016 `fix_shallow` within : seed 1 30/60 fits, seeds 0 et 2 : 0/60) | **hypothèse du log RÉFUTÉE le 04/10** (jobs 709112-3, 709147-9) : rejeu instrumenté des mêmes cellules (même code, données, seed, protocole, carte turing) → **0 pic sur 173 fits**, entrée du log jamais sous 5e-3 (dérivée ≤ 200), et le contrefactuel ε=1e-4 donne une trajectoire identique. Les pics de la campagne (jusqu'à 1e12) dépassent de toute façon la borne 1/ε = 1e6. Ils dépendent du **contexte d'exécution**, pas du modèle. **Co-location RÉFUTÉE le 04/10** (jobs 709522/709523, code 58c6beeb, margpu020) : les 9 cellules co-localisées comme dans la campagne (fraction 0.094) → 0 pic sur 8 cellules complètes (la 9e, gsea_s0, OOM à mi-parcours) ; la même cellule `fix_shallow` seed 1 **seule** → **16 fits avec pic** (jusqu'à 2.3e9). Seule différence : la carte. Pics sur la carte PCI `5E:00.0` de margpu020, aucun sur la `3B:00.0` ni sur margpu022. Tous les pics tombent sur le batch 0 ou 1 d'une époque, entrée du log saine (≥ 0.02), gradient concentré sur `conv_spat.weight` ou `conv_time.bias`. **Carte SUSPECTE, pas confirmée (Fisher unilatéral p ≈ 0.045 sur 12 runs, aucun signal matériel direct relevé ; « batch 0/1 » non informatif : 2 batches par époque)** (job 709926, 4 cartes épinglées par UUID, 2 rejeux chacune) : `5E:00.0` (UUID `GPU-b0d82f38…`) → 14 fits avec pic au 1er rejeu, 0 au 2e ; `3B`, `B1`, `D9` → 0/6. Bilan : **2 runs sur 3 avec pic sur 5E, 0 sur 9 ailleurs** (dont margpu022). Défaut intermittent, localisé sur une carte. Mais la campagne a aussi des pics sur margpu018 (445 fits), 019, 017 et 028 : d'autres cartes sont probablement en cause, ce n'est pas un problème de la seule 5E. Effet sur les scores de la campagne précédente (`spike_scores.py`, jumeaux = autres seeds sans pic, même sujet × session) : +0.012 [−0.002, +0.031] sur 82 unités où le pic précède l'époque restaurée → aucune perte détectable à ~2-3 pts près ; les unités dont toutes les seeds ont des pics (ex. cho2017 `grow_sccnet`, margpu018) sont exclues faute de jumeau. **Piste carte écartée par Adam le 04/10** (pas d'alerte matérielle de l'Inria) : cause des pics **non élucidée**, on ne la poursuit pas. **Règles pour la nouvelle campagne** (valent quelle que soit la cause) : UUID de la carte enregistré à chaque fit (coût nul, permet de revérifier a posteriori) ; garde-fou qui refuse un fit avec `grad_norm` > seuil et le relance ; nombre de fits refusés rapporté. Les entraînements ne sont pas déterministes même sur une seule carte (trajectoires différentes dès l'époque 1-2 entre deux rejeux) : la comparaison bit à bit est impossible, d'où le critère sur les pics. Autre constat : la campagne n'est pas reproductible bit à bit (perte d'époque 1 déjà différente au rejeu) |
| **seuil de pic chiffré (13/10)** | `step0_catalogue/spike_threshold.py`, lecture seule de `results_final` (167 076 fits, 9 bras) : max par fit de `grad_norm_max`. Fits sains ≤ 10^3,75–10^4 (grow_sccnet, le bras aux normes les plus hautes), un isolé à 5,8e4 ; **aucun fit entre 5,8e4 et 5,6e6** ; pics : **222 fits** de 5,6e6 à 1e16, tous shallow/sccnet, 0 sur `bd_*` et la famille deep. **Correction du relevé du 04/10** : les « 692 fits > 1e3 » incluaient 471 fits normaux de grow_sccnet (1e3–1e4) ; le vrai compte est 222. Le ratio au médian du fit est inutilisable : la mémorisation effondre la norme tardive, et des fits sains atteignent 1e4–1e5 | **seuil = 1e6 sur `grad_norm_max`** : ×17 au-dessus du plus grand sain, ×5,6 sous le plus petit pic, zéro fit entre les deux. Valide pour les familles déjà mesurées ; pour tout nouveau modèle (deep growable, LaBraM), l'étape 0 vérifie que ses fits sains restent ≤ 1e4 (≥ ×100 de marge), sinon le seuil est refusé pour ce modèle et recalibré. Un fit au-dessus est **refusé et relancé** ; nombre de refus rapporté par modèle |
| « gradient qui s'effondre » | norme finale ≈ 1–3 % de la norme d'époque 5–15 sur la plupart des bras, avec perte train → 1e-4 | c'est la mémorisation du train, pas une pathologie d'Adam. `eps` d'AdamW déjà mesuré le 25/08 : il absorbe ≤ 0,07 % du pas — le monter ne change rien |
| pas d'early stopping | `patience=200` = `max_epochs=200` ; modèle scoré à l'époque médiane 10–34 | **choix mesuré, pas un défaut** (carré budget × sélection, +0.085, p=1e-7, cf. `deep4-instability`) ; seul effet de bord : la croissance continue pendant le sur-apprentissage, à garder en tête pour le critère « calendrier » |
| **modèle scoré quasi non entraîné** (relevé 04/10, `gd_fits`, tous les bras) | époque restaurée < 5 sur **19 à 37 % des fits de TOUS les bras** (bd_* compris : 23–30 %), = 1 sur 10–20 %. Par dataset (bras grow) : shin2017a 59 %, physionetmi 39 %, alexmi 25 %, lee2019_mi 19 % … bnci2014_001 1 %, schirrmeister2017 0,5 %. Pour les bras grow, ce sont des fits scorés **avant la 1re croissance** : 33 % des `grow_deep` sont scorés comme le petit réseau de départ alors qu'ils ont grandi jusqu'à 32 ensuite | cause lue dans le code : `RestoreBestModel` garde la 1re époque à égalité (`>` strict), et `valid_acc` est quantifiée en 1/n_valid (3 essais de validation sur shin2017a, d'après le commentaire de `config.yaml`) → une époque 1 chanceuse verrouille la sélection. La campagne **échoue au critère « optimum » de l'étape 0 sur les 9 bras**. Conséquence directe pour la sonde : `restored_epoch` mesure en partie le bruit d'un split de validation minuscule. À régler avant l'étape 2 : taille minimale du split de validation, ou un moniteur non quantifié (perte lissée), puis re-mesurer |
| line search sans held-out | le « held-out » de la line search = les derniers 20 % des batches **d'entraînement** de l'époque (déjà vus par le réseau) ; et sous 4 batches (n_train < 256, soit **~90 % des événements de croissance**) il n'y a pas de held-out du tout : la line search tourne sur les batches mêmes qui ont servi aux statistiques | explique `s = 1.0` (max de la grille {0, 0.1, 0.5, 1}) sur 96–99 % des pas appliqués : sur les données qu'elles ont ajustées, la plus grande amplitude gagne toujours. La distribution de `s` est censurée et ne dit rien. Correctif : un vrai held-out disjoint du split de sélection, et une grille qui dépasse 1. **Chiffré le 13/10** (`step0_catalogue/line_search_censoring.py`, 542 187 événements) : **92,7 %** des événements sans held-out. Le held-out, même fait de batches déjà entraînés, change les décisions (tant que perte train > 0,1, abstentions 0 → 7,5 % grow_deep, 0,3 → 20,5 % grow_sccnet, 2,8 → 23,9 % grow_shallow) : il porte de l'information. Mais quand un pas s'applique, `s` est au sommet de la grille sur **97,8–100 %** des cas, avec ou sans held-out. **Critères de refus** : (1) held-out présent et disjoint de l'entraînement, du calcul des statistiques et du split de sélection sur **100 %** des événements (vérifié par estampille) ; (2) grille **censurée** si le sommet gagne sur > **20 %** des pas appliqués (même seuil que les bords de l'époque scorée) → refus du modèle tant que la grille n'est pas élargie. **TRANCHÉ par Adam le 13/10 : option A** — 10 % du fold retirés du côté entraînement pour **tous** les bras (mêmes essais d'entraînement partout ; le split de validation reste identique bit à bit, n_valid inchangé), grille élargie {0 ; 0,25 ; 0,5 ; 1 ; 2 ; 4}. Implémenté : `eegrow.training.splits.HoldoutValidSplit`, `GromoGrowth(scaling_grid=…)` (stats sur tout l'entraînement, line search sur le held-out, **refus** si un held-out demandé est absent), `train.growth_holdout` / `train.scaling_grid`, estampilles `grow_holdout`, `grow_n_select`, `n_fit_train`, `n_valid`, `n_holdout` ; 107/107 tests. Smoke via `run_cell` (bnci sujet 1, 11 époques) : 161/46/23 essais sur les deux bras, 20/20 événements `disjoint`. **Premier signal, non concluant (1 sujet, 11 époques)** : `s = 4` gagne 19/19 pas appliqués, la perte held-out ne varie que de 5e-4 sur toute la grille (effet quasi linéaire et minuscule des nouveaux neurones) → la grille élargie risque d'échouer aussi au critère de censure ; c'est ce que l'étape 0 tranchera |
| abstentions de `grow_shallow` | 74,9 % d'abstentions, mais **77 % de ses événements ont lieu quand la perte train est déjà < 0,01** (abstention 94,5 % là) ; au-dessus de 0,01 : 5–12 % | en majorité bénin : la croissance continue d'être tentée après la mémorisation. Le vrai défaut est en amont : `grow_shallow` mémorise avant d'atteindre sa cible (époque médiane d'atteinte 35, cible atteinte 29 %) |
| largeur cible | atteinte sur 29 % (shallow), 48 % (deep), 78 % (sccnet) des fits **scorés** — pour deep, l'écart vient presque entièrement des fits scorés avant la 1re croissance (ligne 1) | pas d'appariement en largeur avec les jumeaux |
| calendrier | `grow_sccnet` atteint sa cible à l'époque médiane **5** : une seule croissance 4 → 22 ; `grow_deep` en 3 pas fixes de +8 (époques 5, 10, 15) | `grow_sccnet` n'est pas une croissance progressive mais un gonflement unique : il échoue au critère « calendrier » |
| croissance après l'époque scorée | 18 % (sccnet), 33 % (shallow), 43 % (deep) des croissances appliquées tombent après l'époque restaurée | sans effet sur le score, mais du calcul perdu |
| effet immédiat d'une croissance | perte train en baisse (par construction), `valid_acc` médiane **+0,000** sur les 3 bras | cohérent avec `growth-no-local-effect` |
| contrôle manquant | aucun `fix_*` en `cross_subject` | grow − bd n'y est pas décomposable |

**Correctif du moniteur de sélection, mesuré le 06/10 sans GPU** (`restored_tie_rule/`).
Recalcul, sur les courbes `valid_acc` par pli des JSONL, de la sonde `grow_shallow`
within_session sous 3 règles : `orig` (1er argmax, comme a tourné `RestoreBestModel`),
`late` (égalité → époque la plus tardive), `cv` (courbe moyennée sur tous les plis du
sujet, pondérée par n_valid). Portes passées : `orig` reproduit `restored_epoch` fit par fit
et `restored_probe` au sujet près, et redonne exactement l'étage C (+0.348).

| dataset | n_valid/fit | fits < ép. 5 orig → late → cv | ICC(1,k) orig → late | partielle q_cent \| acc : orig / late / cv |
|---|---|---|---|---|
| shin2017a | 4 | 64 % → 15 % → 18 % | 0.39 → 0.61 | +0.52 / +0.19 / +0.50 |
| physionetmi | 8 | 33 % → 9 % → 12 % | 0.44 → 0.65 | +0.44 / +0.53 / +0.52 |
| liu2024 | 7 | 44 % → 14 % → 11 % | 0.45 → 0.48 | +0.23 / +0.37 / +0.27 |
| lee2019_mi | 16 | 15 % → 5 % → 2 % | 0.76 → 0.86 | +0.22 / +0.25 / +0.39 |
| cho2017 | 32 | 5 % → 2 % → 1 % | 0.69 → 0.70 | +0.26 / +0.42 / +0.15 |
| dreyer2023a | 39 | 3 % → 1 % → 1 % | 0.78 → 0.76 | +0.25 / +0.17 / +0.30 |

Groupé étage C (3 datasets) : orig +0.348, late **+0.443**, cv +0.411 ; écarts appariés
(bootstrap des sujets) late−orig +0.095 [−0.052, +0.233], cv−orig +0.063 [−0.078, +0.198].
Groupé frais (dreyer2023a + liu2024) : +0.245 / +0.262 / +0.283, écarts ±0.2.

Lecture :
- **Le signal de `restored_epoch` n'était pas le bruit du petit split** : il survit
  intact sous les deux règles corrigées. Le gain de prédiction n'est **pas résolu** (IC
  contiennent 0, et l'étage C est le jeu de données qui a fait trouver la variable).
- **Ce que la règle `late` corrige sans ambiguïté, c'est le comportement** : fits scorés
  avant l'époque 5 divisés par 3 à 4, fiabilité ICC(1,k) en hausse là où le split est
  petit (physionetmi 0.44 → 0.65, shin2017a 0.39 → 0.61).
- `cv` n'apporte rien de plus que `late` ici, et ne garde que k = 3 réplicats par sujet
  (une valeur par seed) : son ICC(1,k) passe sous 0.50 sur lee2019_mi et cho2017.

**Décision (validée par Adam le 06/10)** : `RestoreBestModel` départage les égalités vers
l'époque **la plus tardive**, pour tous les bras (sonde et fits scorés). Choisi sur un
critère de comportement, pas sur une accuracy : conforme à la règle de calibration
ci-dessus. Le garde-fou n_valid minimal reste à fixer dans le préreg (shin2017a à
n_valid = 4 garde 15 % de fits scorés avant l'époque 5 même sous `late`).

Implémenté le 06/10 (non commité) : paramètre `RestoreBestModel(tie_break=...)`, clé
`train.tie_break` dans `config.yaml`, lue par `pipelines._build_dl`. Le défaut reste
`earliest` pour que les résultats archivés se reproduisent, même convention que
`selection_monitor`. Tout lancement de l'étape 0 et des suivantes passe donc
`train.patience=200 train.selection_monitor=valid_acc train.tie_break=latest`, et le
préreg doit **refuser** un lancement sans ces trois overrides. Le JSONL des fits
n'enregistre ni `tie_break` ni `selection_monitor` : le préreg doit les estampiller (meta
de `FitRecorder`) ou contrôler la ligne `srun`. Tests : `tests/test_callbacks.py` (plateau
quantifié n_valid = 4, perte, valeur inconnue refusée, survie au `clone`), 14/14 verts.
Le 07/10, `FitRecorder` estampille en plus `selection` (moniteur, tie_break, stop_monitor,
patience) lu sur les callbacks vivants, pas sur la config ; 76/76 tests verts.

**07/10 — la décision ci-dessus est REMISE EN CAUSE, à re-trancher par Adam.** Le 06/10
n'avait mesuré que le bord *début* du critère « optimum ». Le bord *fin* (`restored_epoch`
≥ budget − 5 sur plus de 20 % des fits → refus) casse sous `late` : après mémorisation,
`valid_acc` re-touche son maximum jusqu'au bout, et le dernier argmax tombe en fin de budget.
Les fits basculés en fin avaient atteint leur maximum à l'époque médiane 21. Recalcul
(`restored_tie_rule/smooth_rule.py`, même sonde, mêmes portes) avec en plus `smW` = argmax de
la moyenne glissante centrée de `valid_acc` sur W époques :

| dataset (n_valid) | < ép. 5 : orig / late / sm5 / sm11 | ≥ budget−5 : orig / late / sm5 / sm11 |
|---|---|---|
| shin2017a (4) | 64 / 15 / 45 / 36 % | 0 / **49** / 0 / 0 % |
| liu2024 (7) | 44 / 14 / 27 / 22 % | 0 / **31** / 0 / 0 % |
| physionetmi (8) | 33 / 9 / 19 / 17 % | 0 / **36** / 0 / 1 % |
| lee2019_mi (16) | 15 / 5 / 8 / 7 % | 0 / **34** / 1 / 3 % |
| cho2017 (32) | 5 / 2 / 2 / 1 % | 1 / **25** / 4 / 9 % |
| dreyer2023a (39) | 3 / 1 / 1 / 1 % | 2 / **29** / 7 / 10 % |
| bnci2014_001 (46) | 0 / 0 / 0 / 0 % | 4 / **23** / 10 / 16 % |

Lecture :
- **`late` échoue au critère « optimum » sur les 7 datasets**, dont le dataset de dev. Il
  déplace le défaut d'un bord à l'autre.
- **Le lissage tient les deux bords dès n_valid ≥ 16** (sm5 et sm11 : ≤ 8 % au début,
  ≤ 16 % en fin). `sm21` repasse au-dessus de 20 % en fin sur bnci2014_001.
- **Sous n_valid ≈ 8, aucune règle ne passe** : avec 4 à 8 essais de validation, `valid_acc`
  ne contient pas l'information nécessaire pour choisir une époque. Ce n'est pas un réglage de
  règle, c'est le **plancher n_valid** : 16 est la plus petite valeur observée qui passe.
- ICC(1,k) : sm5/sm11 au niveau d'`orig` ou au-dessus (bnci 0.96, cho 0.74) ; la hausse
  d'ICC de `late` sur les petits splits venait en partie de la masse en fin de budget.
- Contrôle seulement (pas un critère de choix) : étage C groupé orig +0.347, late +0.443,
  sm5 +0.372, sm11 +0.408. Le signal survit sous toutes les règles.

**Proposition à trancher** : sélection sur `valid_acc` lissée (moyenne centrée, W = 5 ou 11 ;
les deux passent, W le plus petit est le changement minimal) au lieu de `tie_break=latest`,
et plancher **n_valid ≥ 16** comme critère de refus d'un dataset (shin2017a, physionetmi et
liu2024 tombent dessous en within_session tels quels). Coût d'implémentation : la moyenne
centrée voit W//2 époques dans le futur, donc `RestoreBestModel` doit garder un tampon
circulaire des W//2 + 1 derniers modules (pas un snapshot par époque).

**13/10 — implémenté et testé sur fits réels (go conditionnel d'Adam : « test que ça
fonctionne réellement et si oui alors go »).** `RestoreBestModel(smooth_window=W)` (tampon
circulaire, défaut 1 = règle historique), `train.smooth_window` dans `config.yaml`, lu par
`pipelines._build_dl`, estampillé dans `selection.smooth_window`. Tests : équivalence avec la
règle hors ligne sur 2 400 courbes aléatoires quantifiées (W ∈ {1,3,5,11} × 2 tie_break,
fits plus courts que la fenêtre inclus) + fit réel avec croissance ; 92/92 verts.

Test live (`restored_tie_rule/live/`) : bnci2014_001 within_session, 9 sujets × 2 sessions ×
5 folds, seed 0, pipeline du benchmark avec patience 200, `valid_acc`, W = 5, et un callback
qui score le **test** du fold à chaque époque. Critères écrits avant de lancer : A fidélité,
B bords < 20 %, C acc test sm5 pas en dessous d'orig/late (unité = sujet, bootstrap).

| | grow_shallow | bd_deep4 |
|---|---|---|
| A — époque restaurée = règle hors ligne, et acc rendue = acc test de cette époque | 90/90 | 90/90 |
| B — < ép. 5 : orig / late / **sm5** | 0 / 0 / **0** % | 6.7 / 2.2 / **2.2** % |
| B — ≥ budget−5 : orig / late / **sm5** | 4.4 / 22.2 / **10.0** % | 7.8 / 28.9 / **18.9** % |
| C — sm5 − orig | **+0.009** [+0.003, +0.014], 7/9 | **+0.013** [+0.002, +0.023], 8/9 |
| C — sm5 − late | +0.001 [−0.006, +0.007] | −0.004 [−0.010, +0.002] |
| *hors critères* — sm5 − dernière époque | **−0.009** [−0.015, −0.002], 2/9 | −0.000 [−0.012, +0.011] |
| *hors critères* — oracle (meilleure époque en test) − sm5 | +0.059 | +0.058 |

Lecture :
- **A, B, C passent sur les deux modèles.** Le 10.0 % de grow_shallow reproduit exactement
  le recalcul hors ligne (10 %), ce qui valide le recalcul lui-même. `late` recasse le bord fin
  en live (22 %, 29 %), comme prévu.
- **bd_deep4 passe le bord fin de justesse (18.9 % pour un seuil à 20 %).** Sur un autre
  seed ou un autre dataset, il peut basculer en refus.
- **Découverte hors critères, à trancher par Adam :** avec patience 200 (pas d'arrêt précoce),
  **ne rien sélectionner du tout** (garder la dernière époque) fait aussi bien que sm5 sur
  bd_deep4 et **mieux** sur grow_shallow (−0.009, IC hors de 0). L'oracle est 6 points
  au-dessus de toutes les règles : sur 46 essais, `valid_acc` ne sait pas désigner la bonne
  époque, la sélection récupère surtout ce que l'arrêt précoce avait détruit. Conséquence
  pour le cadrage : la *performance* n'a pas besoin de sélection, mais le *prédicteur*
  (`restored_epoch`) en a besoin, puisqu'il vaut 200 partout sans elle. Les deux questions
  doivent être séparées : quel modèle on score, et quelle époque sert de mesure.

**13/10 — TRANCHÉ par Adam :**
- **Score** = dernière époque (`smooth_window=1`, aucune restauration, patience 200). La
  performance n'a pas besoin de sélection, et le score ne dépend plus du split de validation
  qui porte le prédicteur : prédicteur et variable de contrôle de la partielle ne partagent
  plus le même bruit.
- **Prédicteur** = `restored_epoch` lissé W = 5, calculé hors ligne sur la courbe `valid_acc`
  enregistrée (la règle est celle de `RestoreBestModel(smooth_window=5)`, qui sert de
  référence). Recalculable a posteriori sous n'importe quelle règle.
- **Plancher n_valid ≥ 16** : critère de refus d'un dataset (la mesure n'a pas de sens
  dessous).
- **Condition de validité du score à la dernière époque** : le garde-fou de pic de gradient
  (refus + relance) de l'étape 0, puisque c'est le seul cas où la dernière époque est un
  modèle cassé.
- **Option gardée ouverte** : chaque fit enregistre aussi les poids du modèle choisi par le
  lissage, avec les indices du fold, pour pouvoir le re-scorer hors ligne. Tout score tiré de
  ces poids est **exploratoire**, déclaré a posteriori : le score principal reste la dernière
  époque, figé par le PREREG.

À ajouter à l'instrumentation : norme du gradient **par batch** et **par couche** (le
relevé actuel est par époque), et l'état de `v_hat` d'Adam après un pic (un pic de 1e6
gonfle le second moment pendant ~1/(1−β₂) pas, ce qui peut geler l'apprentissage sans
que la perte ne saute — hypothèse, non mesurée).

**Sortie.** Un tableau passe / refus par modèle et une config gelée. Un modèle refusé sort de
la campagne, il n'est pas « gardé avec une réserve ».

### Étape 1 — Un deep fidèle qui pousse vraiment

**Question.** Peut-on faire croître une architecture deep publiée sans la dénaturer ?

1a. **Lire le source de gromo** : un pooling agissant canal par canal (MaxPool en temps) peut-il
entrer dans `post_layer_function` sans fausser les statistiques de croissance ? Si oui,
Deep4Net fidèle devient possible. Réponse par lecture + test, pas par raisonnement.

1b. Candidats à jonction native (audit `GrowMo/audit_growable.py`, braindecode 1.7.0) :
- convolutionnels : **ContraWR**, **TIDNet** ;
- transformers : le **FFN** d'un petit **EEGConformer** — même famille que les modèles de
  fondation, ce qui rend le lien avec l'étape 4 direct.

**Critères.**
- Parité : le modèle growable à sa largeur de départ, poids copiés depuis braindecode, sort les
  mêmes logits (écart max ≤ 1e-5).
- Il passe l'étape 0 en entier.
- Retenir **au moins deux** deep fidèles de familles différentes (conv + transformer), pour
  qu'un résultat ne soit pas la propriété d'une seule architecture.

### Étape 2 — La sonde deep vaut-elle quelque chose, sur son propre terrain ?

**Question.** Avant tout modèle de fondation : une sonde deep choisit-elle mieux les donneurs
que l'accuracy **pour un receveur du même modèle** ? C'est le test qui a tué la sonde shallow.

- Fiabilité : `ICC(1,k) ≥ 0.50` corrigée Spearman-Brown, sur le nombre de fits par sujet qui
  sera réellement utilisé.
- Non-redondance : part de variance de la sonde indépendante de l'accuracy, rapportée.
- **Prédicteur primaire** : `restored_epoch` **sous le moniteur de validation corrigé à
  l'étape 0** (tranché le 05/10, § 5). Ce n'est pas l'instrument de septembre : sa fiabilité
  est re-mesurée ici, aucune valeur antérieure n'est reprise.
- Endpoint : A−B > 0 (sonde vs accuracy, à K fixé), groupé sur les datasets d'évaluation.
- **Contraste « croissance »** : la sonde du bras en croissance doit aussi battre le
  `restored_epoch` de son **jumeau fixe** (même architecture, largeur finale). Sans ça, le
  signal est celui de la dynamique d'apprentissage, pas de la croissance, et le bras fixe
  (moins cher) le fournit aussi. Ce contraste est secondaire : il ne refuse pas l'étape 4,
  mais il conditionne le droit d'appeler la sonde « sonde de croissance ».

**Refus** : si A−B n'est pas positif ici, l'étape 4 n'a pas lieu. Une sonde qui ne bat pas
l'accuracy pour son propre modèle n'a aucune raison de la battre pour un modèle de fondation.

### Étape 3 — Le mélange de datasets crée-t-il une marge pour la sélection ?

**Question.** Entre datasets MI harmonisés, existe-t-il du transfert négatif ?

Harmonisation, fixée avant : classes **gauche/droite** seulement (présence à **vérifier
dataset par dataset**), canaux communs, même fréquence d'échantillonnage, même fenêtre.

Mesures, par receveur :
- vivier de son propre dataset vs vivier multi-datasets complet ;
- plafond de sélection inter-datasets (même méthode que `selection-ceiling`) : le meilleur
  sous-ensemble observé bat-il le vivier complet ?

Pièges réglés d'avance :
- la sonde n'est **pas comparable** d'un dataset à l'autre (taille de départ, essais,
  classes) ⇒ classement **en percentiles à l'intérieur de chaque dataset** ;
- un top-K global serait raflé par un seul dataset ⇒ la **composition du top-K par dataset**
  est publiée, comparée au prorata.

**Issue : arbre de décision fermé, écrit dans le préreg de l'étape 3 avant sa lecture.**
- **Plafond inter-datasets franchi** (meilleur sous-ensemble > vivier complet, IC exclu, sur
  une part des receveurs fixée dans le préreg) → endpoint de l'étape 4 = **sélection contre
  le vivier entier**.
- **Sinon** → la sélection reste dominée même entre datasets, publié tel quel ; endpoint de
  l'étape 4 = **K imposé**, avec un K justifié de l'extérieur (coût d'adaptation chiffré) et
  écrit avant l'étape 4. Annoncé pour ce qu'il est : « lesquels, à K fixé », la question
  plus faible du point 4, à probabilité a priori basse.
Aucun autre embranchement ; le régime ne se choisit jamais après lecture.

### Étape 4 — Transfert à un modèle de fondation

**Question.** La sonde deep, classée inter-datasets, choisit-elle les données d'adaptation
d'un modèle de fondation mieux que le hasard **et** que l'accuracy ?

- Modèle portant l'endpoint : **LaBraM**. CBraMod et EEGPT descriptifs. BIOT exclu (montage
  bipolaire, 0/18 canaux communs). REVE seulement si Adam obtient l'accès Hugging Face.
- **Porte de transfert d'abord** : le bras « tout le vivier » doit être au-dessus de la chance,
  IC exclu, sur chaque (modèle × dataset). LaBraM gelé n'était qu'à +1 à +6 points : le régime
  (gelé / affiné) se choisit sur cette porte, jamais sur un contraste de sélection.
- Bras : sonde / accuracy / hasard (tirages multiples) / bas de classement / tout le vivier.
- Réplication fraîche sur `dreyer2023a` + `liu2024` seulement après lecture de l'endpoint.

### Étape 5 (conditionnelle) — Un autre paradigme

Seulement si l'étape 4 atteint son endpoint. Même chaîne, à l'identique, sur un paradigme ERP,
**séparée** de la MI de bout en bout. Le dataset sera choisi sur ses nombres de sujets réels,
mesurés, au moment venu. Raison de ne pas le faire avant : le signal de croissance n'a
probablement pas le même sens (motifs spectraux en MI, forme d'onde en ERP), donc un résultat
ERP serait une seconde hypothèse, pas une extension.

---

## 5. Décisions

Tranché le 05/10 par Adam :
1. **Prédicteur primaire = `restored_epoch`** (numéro d'époque, pas un temps : il ne dépend
   de la carte que par le non-déterminisme numérique, que l'étape 0 mesure).
   - *Pourquoi pas la taille finale* : l'étape 0 exige la largeur cible atteinte sur ≥ 80 %
     des fits ; un bras qui passe a donc une taille finale quasi constante, sans variance
     pour classer. Et à `restored_epoch` contrôlé, elle était déjà morte (+0.069
     [−0.070, +0.204]).
   - *Conditions* : défini sous le moniteur de validation corrigé (19–37 % des fits de la
     campagne précédente scorés avant l'époque 5) ; fiabilité re-mesurée à l'étape 2 ;
     contraste contre le jumeau fixe (étape 2).
2. **Dataset de développement = `bnci2014_001`** (n=9, ne peut porter aucun endpoint ;
   BCI IV 2a = dataset du chiffre publié Schirrmeister 2017, donc critère « référence
   publiée » vérifiable dessus, en 4 classes, plus la config gauche/droite de l'étape 3).
   - *Réserve* : c'est le dataset où la pathologie de l'époque restaurée est quasi absente
     (1 % des fits avant l'époque 5, contre 59 % sur shin2017a). Les réglages qui regardent
     une **accuracy** se font sur lui seul ; les critères de **comportement** de l'étape 0
     se vérifient aussi sur `shin2017a` et `physionetmi`.
3. **Régime de l'endpoint final** : pas choisi à l'avance, mais **règle pré-enregistrée** :
   l'arbre de décision de l'étape 3 (sélection contre le vivier entier si le plafond
   inter-datasets est franchi, K imposé de l'extérieur sinon).

Tranché le 03/10 par Adam :
- **GPU** : ce que Margaret laisse disponible, cartes hétérogènes. Conséquences : modèle de
  carte enregistré à chaque fit, aucune mesure en temps d'horloge dans la sonde, walltime
  dimensionné sur la carte la plus lente, invariance à la carte vérifiée à l'étape 0.
- **REVE** : plus tard, pas bloquant ; d'autres modèles de fondation peuvent porter le volet
  descriptif.

## 6. Hors périmètre

Mélanger des paradigmes ; tout réglage sur les sujets d'évaluation ; toute re-lecture des
résultats ICASSP comme preuve pour ou contre cette campagne.
