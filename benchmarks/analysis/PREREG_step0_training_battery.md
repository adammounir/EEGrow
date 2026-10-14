# Pré-registration — Étape 0 : batterie d'entraînement

**GELÉ le 14/10/2026**, avant tout fit de l'étape 0, par le commit qui l'introduit. Brouillon
écrit le 13/10. Les décisions laissées ouvertes dans le brouillon (§ 8 du 13/10) ont été
prises la nuit du 13 au 14/10 par Claude, sur délégation explicite d'Adam (« choisis ce qui
est le mieux ») ; chacune est marquée **[décidé 14/10]** avec sa source. Toute modification
après le premier fit lu se fait par avenant daté, et les critères déjà lus deviennent
exploratoires.

Référence : `CADRAGE_deep_growth_transfer.md`, § 4 Étape 0, et les décisions des 05, 06,
07 et 13/10 qui y sont consignées.

## 1. Ce que l'étape 0 décide, et ce qu'elle ne décide pas

Elle ne produit **aucun résultat scientifique**. Elle ne fait que **refuser** : un modèle
qui échoue à un critère sort de la campagne. Il n'est pas « gardé avec une réserve ».
Elle produit aussi une **config gelée** par modèle, la même sur tous les datasets.

Elle ne compare aucune accuracy entre bras ni entre configs candidates, et ne lit aucune
sonde. Le choix de la config (§ 3.2) se fait sur les seuls critères de refus.

## 2. Périmètre

**Lot 1 (ce document)** : les deux références publiées.

| bras | modèle | source des hyperparamètres d'architecture |
|---|---|---|
| `bd_shallow` | `braindecode.models.ShallowFBCSPNet` | défauts braindecode |
| `bd_deep4` | `braindecode.models.Deep4Net` | défauts braindecode |

L'implémentation testée est celle de l'env `bench` de Margaret (braindecode 1.5.2, torch
2.13+cu130), celle sur laquelle toutes les campagnes ont tourné.

**Les bras en croissance de l'étape 1** passent la même batterie, avec la § 5.2 en plus,
chacun par un avenant daté, sans changer les seuils.

**Dataset** : `bnci2014_001`, `within_session`, 5 plis, seeds 0/1/2 = **90 fits par
cellule, 270 par (bras, config)**. Plancher n_valid ≥ 16 : n_valid = 46–47 ici. `shin2017a`
(n_valid 4) et `physionetmi` (8) tombent sous le plancher ; le critère « optimum » n'y
sera pas vérifié quand ils entreront (lot ultérieur).

## 3. Protocole d'entraînement gelé

### 3.1 Commun à toutes les cellules

```
model=<bras> dataset=bnci2014_001 eval=within_session seed=<k>
train.max_epochs=200 train.patience=200
train.selection_monitor=valid_acc train.smooth_window=5 train.tie_break=earliest
train.restore_selected=false train.save_selected=true
train.growth_holdout=0.1 "train.scaling_grid=[0.0,0.25,0.5,1.0,2.0,4.0]"
train.lr_schedule=null train.lr=<lr> train.weight_decay=<wd>
```

| choix | valeur | d'où il vient |
|---|---|---|
| modèle scoré | dernière époque | tranché 13/10 (live, `restored_tie_rule/live/`) |
| `restored_epoch` (mesure) | argmax de `valid_acc` lissée, fenêtre centrée 5 → `selected_epoch` | tranché 13/10 |
| split | validation 20 % stratifiée (identique à l'historique) + 10 % du fold retirés de l'entraînement pour tous les bras | tranché 13/10 (option A) |
| batch, eps | 64, 1e-8 | config du benchmark |
| fréquence, bande | 250 Hz, 8–32 Hz | config du benchmark |
| schedule | constant | **[décidé 14/10]** : le cosinus a été mesuré nuisible sur cette grille (0.4276 vs 0.4818, 36 unités appariées, `config.yaml`) et il est incompatible avec les bras en croissance (`GromoGrowth` reconstruit l'optimiseur) ; un schedule différent entre `grow_*` et `bd_*` serait un facteur confondu dans l'endpoint |

### 3.2 Configs candidates d'optimiseur **[décidé 14/10]**

Le papier ne donne pas le lr (relu : § 2.5.5 et A.4 ne citent qu'« Adam »). Le code des
auteurs utilise `optim.Adam(model.parameters())` (lr 1e-3, sans weight decay) dans un
protocole cropped à deux phases qui n'est pas le nôtre. Deux candidats, chacun cité :

| candidat | `bd_shallow` | `bd_deep4` | source |
|---|---|---|---|
| **B** (braindecode) | AdamW lr 6.25e-4, wd 0 | AdamW lr **1e-2**, wd **5e-4** | `examples/model_building/plot_bcic_iv_2a_moabb_trial.py` l. 233-238 (« These values we found good for shallow network » / « For deep4 they should be ») |
| **H** (historique) | AdamW lr 6.25e-4, wd **0.01** | AdamW lr 6.25e-4, wd **0.01** | ce que toutes les campagnes ont tourné : `config.yaml` (lr) et le défaut d'AdamW torch 2.13 (wd, jamais passé avant le 14/10 ; lu dans le stamp `optimizer` des records). **Correction** : le brouillon du 13/10 disait wd 0, c'était faux |

**Règle de choix, fixée avant tout fit** : pour chaque bras, B est la config gelée si B
passe tous les critères de la § 5.1 ; sinon H si H les passe tous ; sinon le bras est
refusé. B est prioritaire parce que c'est la recette publiée par les mainteneurs ; H sert
de repli et garde la comparabilité avec l'historique. Les deux sont lancées en même temps
(la décision ne dépend pas de l'ordre) : 4 configs × 3 seeds = **12 cellules**.

## 4. Garde-fous : ils refusent, ils n'avertissent pas

Implémentés dans `benchmarks/step0_gate.py` (22 tests, `tests/test_step0_gate.py`) :

- **G-lancement**, sur **100 %** des records, avant toute lecture : estampille `selection`
  exactement `{valid_acc, lower_is_better false, earliest, smooth_window 5, swap false,
  stop valid_loss, patience 200}` ; `max_epochs` = `epochs` = 200 ; AdamW au lr et au wd
  de la config ; lr constant à chaque époque ; `n_holdout` > 0 ; poids sélectionnés
  sauvés ; `first_batch_loss` présent ; pour un bras en croissance, chaque événement
  `grow_holdout == "disjoint"`. Un seul écart invalide la cellule : elle est relancée,
  jamais lue.
- **G-complétude** : 90 fits par cellule ; `cv_ind` = 0…89 chacun une fois (MOABB les
  numérote sur toute la cellule, dans un ordre qui varie — relevé sur le smoke du 13/10) ;
  5 fits par (sujet, session) ; aucune clé (sujet, session, cv_ind) dupliquée (fits
  fantômes d'une cellule annulée, le JSONL étant en append) ; JSONL tronqué = refus.
- **G-pic** : un fit avec `grad_norm_max_fit` > **1e6** (ou non fini) fait refuser sa
  cellule, qui est relancée. Seuil calibré sur 167 076 fits (sains ≤ 5,8e4, pics ≥ 5,6e6,
  `step0_catalogue/spike_threshold.py`). Au bout de **2 relances sans succès**, la config
  est refusée pour instabilité. Le nombre de refus est rapporté par config.

**Effet de la carte [décidé 14/10, remplace l'analyse hors ligne du brouillon]** : la
carte de chaque cellule de la campagne précédente n'est pas dans ses records, donc
l'analyse préalable n'était pas faisable sans reconstitution. À la place : toutes les
cellules de l'étape 0 tournent sur **un seul modèle de carte** (NVIDIA GeForce RTX 2080 Ti,
14 cartes, margpu017/018/019/022, sondées par nvidia-smi le 14/10 ; margpu028 porte des RTX 2080 non Ti et margpu020/021/023 les cartes fantômes ou suspectes, tous exclus), avec une porte dans le sbatch qui refuse toute autre carte, et
l'UUID est enregistré à chaque fit. Si les pics se concentrent sur une carte (plus de la
moitié des pics d'une config sur une carte qui porte moins d'un tiers de ses fits), la
carte est exclue par UUID et ses cellules relancées.

## 5. Critères de refus

### 5.1 Tous les bras, par config

| critère | mesure | refus si |
|---|---|---|
| départ | `first_batch_loss` | voir § 5.3 |
| capacité | `benchmarks/step0_capacity.py` : bnci2014_001 sujet 1, session `0train`, 32 essais (8 par classe, `RandomState(seed)`), le module et l'optimiseur de la config, sans validation, 200 époques, batch 64 (un pas plein par époque), seeds 0/1/2 ; perte d'entropie croisée finale en mode **eval** sur les 32 essais | médiane sur les 3 seeds ≥ **0,05** **[protocole validé 14/10]** |
| optimum | `selected_epoch` | > **20 %** des 270 fits à ≤ 5, **ou** > 20 % à ≥ 195 (bd_deep4 était à 18,9 % sur le bord fin en live, config H, seed 0) |
| référence publiée | § 6 | § 6 |
| pics | G-pic | 2 relances sans succès |

### 5.2 Bras en croissance (entrent à l'étape 1, inchangé)

| critère | mesure | refus si |
|---|---|---|
| sélection réelle | `grow_n_kept / grow_n_proposed` ; fan-out | = 1,0 partout, ou fan-out < 2× les places à remplir |
| spectre | `grow_eig_proposed` | moins d'une décade entre la plus grande et la plus petite valeur |
| cible | `width_end == target_width` | atteinte sur < **80 %** des fits |
| calendrier | époques des événements appliqués | tous dans les 10 premières époques |
| jumeau | config | pas de jumeau fixe à la largeur finale exacte, même architecture |
| line search, held-out | `grow_holdout` | ≠ `"disjoint"` sur un seul événement (G-lancement) |
| line search, censure | `grow_s` des pas appliqués | le sommet de la grille (4) gagne sur > **20 %** des pas appliqués. Signal préalable non concluant : `s = 4` sur 19/19 au smoke du 13/10 |

### 5.3 Critère « départ » reformulé **[décidé 14/10]**

**Mesure faite avant le gel, qui le motive** (pas une lecture de l'étape 0 : 1 époque,
aucun critère lu). Perte du premier batch, avant toute mise à jour :

| | `run_cell`, 30 fits (sujets 1-3, seed 0) | hors pipeline, 10 seeds, entrée brute | entrée / écart-type global | entrée z-scorée par essai |
|---|---|---|---|---|
| `bd_shallow` | 1,936 (excès **+0,55**) | +0,537 | +0,537 | +0,409 |
| `bd_deep4` | 2,007 (excès **+0,62**) | +0,886 | +0,886 | +0,957 |

L'excès est **identique** quand on divise l'entrée par son écart-type (BatchNorm le rend
invariant à l'échelle) : il vient de l'init publiée des modules braindecode (logits
d'écart 1,2-1,6 à l'init), pas de notre pipeline. Le seuil de 0,15 nat du brouillon aurait
refusé les deux références **par construction**, ce qui ne teste rien.

Règle gelée :
- **Références publiées** (`bd_*`) : l'excès est mesuré et rapporté, **non refusant**.
  L'init fait partie de la spécification publiée.
- **Modèles dont nous écrivons l'init** (bras en croissance) : refus si la médiane de
  l'excès dépasse **0,15 nat** au-dessus de celle de la référence publiée de la même
  famille (même architecture), ou au-dessus de 0 sans référence de famille.

## 6. Référence publiée **[décidé 14/10 : reproduction sous le protocole des auteurs]**

Relevé dans le PDF (arXiv 1703.05051v5, Table 2), BCIC IV 2a = `bnci2014_001`, cropped,
moyenne des 9 sujets :

| bande | FBCSP | Deep ConvNet | Shallow ConvNet |
|---|---|---|---|
| 0–38 Hz | 68,0 | 70,9 | 73,7 |
| **4–38 Hz** | 67,8 | **70,1** | **71,9** |

**Bande 4–38 Hz** : c'est le défaut `low_cut_hz = 4` du script des auteurs et la bande de
l'exemple cropped actuel de braindecode.

**Protocole** : `benchmarks/step0_reference.py`, port ligne à ligne de
`robintibor/braindecode` (0.4) `examples/bcic_iv_2a_cropped.py` et des modules qu'il
appelle (relus le 14/10, détail dans la docstring) :
- session T pour entraîner, E pour tester ;
- ×1e6, puis Butterworth d'ordre 3, 4–38 Hz, **causal** ;
- standardisation exponentielle (1e-3, bloc initial 1000, eps 1e-4), version 0.4 ;
- ival [−500, 4000] ms, soit 1125 échantillons ;
- validation = les derniers 20 % de T, sans mélange ;
- `final_conv_length` 30 (Shallow) et 2 (Deep), entrée 1000, modèle dense ;
- crops aux pas de 0.4, batch ~60, RandomState((2017, 6, 28)) ;
- NLL de la log-softmax moyennée dans le temps ;
- Adam par défaut ;
- contrainte de norme max 2 et 0,5 ;
- arrêt en deux phases : 800 époques, 80 sans gain de `valid_misclass` ; reset du modèle
  et de l'optimiseur au meilleur ; phase 2 sur T complète jusqu'à `valid_loss` < perte
  train au meilleur, ou 2 × meilleure époque ;
- score = accuracy test de la dernière époque.

Écart inévitable : MOABB sert les fichiers run par run, donc filtre et standardisation
redémarrent à chaque run. 7 tests unitaires (`tests/test_step0_reference.py`) et un smoke
de 3 époques par modèle le 14/10 (288 essais par session, split 230/58).

**Seeds** 0/1/2, accuracy par sujet = moyenne des 3 seeds. **Refus** si la moyenne
publiée (4–38 Hz) tombe **hors** de l'IC bootstrap 95 % (10 000 rééchantillonnages des 9
sujets, `numpy.random.default_rng(0)`) de notre moyenne. Demi-largeur attendue d'environ
±8 points : le critère ne détecte qu'un écart grossier, et le rapport le dira.

## 7. Sortie

Un tableau passe / refus par (bras, config) et par critère ; le nombre de fits refusés par
G-pic ; la config gelée par bras selon la règle de la § 3.2. Les bras qui passent sont les
seuls admis aux étapes 1 à 4.

## 8. Exécution

- Code figé : le commit qui introduit ce document, poussé sur
  `origin/exp/deep-bottleneck` ; checkout Margaret **neuf**
  `/scratch/amounir/eegrow_step0`, jamais écrasé pendant un job ;
  `PYTHONPATH=$ROOT/src:$ROOT/benchmarks` ; garde-fou d'import.
- Cartes : RTX 2080 Ti seulement (porte dans le sbatch).
- Coût : mesuré en direct sur les premières cellules, jamais transposé ; walltime de
  sécurité 6 h par cellule.
- Lecture : `step0_gate.py` d'abord, sur toutes les cellules ; le script de lecture des
  critères de la § 5 est commité **avant** la lecture.

## Avenant 1 — 14/10/2026 : bras en croissance `grow_shallow` et `grow_deep4`

Écrit **après** la lecture du lot 1 (`RESULTAT_step0_training_battery.md`, config B gelée
pour les deux références) et **avant** tout fit des bras ci-dessous. Les seuils de la § 5
ne changent pas. L'avenant ne fait que rendre opérationnelles les définitions de la § 5.2,
que le brouillon laissait sans agrégation.

**Bras.** Chacun tourne sous la config B de sa famille (§ 3.2), car c'est la seule gelée :

| bras | modèle | largeur | config | référence de famille = jumeau |
|---|---|---|---|---|
| `grow_shallow` | `GrowingShallowFBCSPNet` (parité vérifiée) | 8 → 40 | lr 6.25e-4, wd 0 | `bd_shallow` (40) |
| `grow_deep4` | `GrowingDeep4Net` (`e78fb4d`, parité ≤ 1e-5) | 8 → 25 | lr 1e-2, wd 5e-4 | `bd_deep4` (25) |

- `grow_every` = 5 pour les deux. La largeur de départ (8) est celle de `grow_shallow`.
  Rien n'est réglé sur un résultat.
- Protocole § 3.1 inchangé : held-out disjoint de 10 %, grille {0 ; 0,25 ; 0,5 ; 1 ; 2 ; 4},
  dernière époque, 200 époques.
- 2 bras × 3 seeds = 6 cellules, sur les mêmes RTX 2080 Ti.

**Garde-fous** : ceux de la § 4, avec `--growing` (chaque événement de croissance doit avoir
`grow_holdout == "disjoint"`).

**Critères, avec leur définition opérationnelle** (implémentés dans
`step0_criteria.py --growing`, commité avec cet avenant, 7 tests sur records synthétiques) :

| critère | définition | refus si |
|---|---|---|
| départ (§ 5.3) | médiane de `first_batch_loss` − ln 4 | > excès de la référence lu au lot 1 + 0,15 : **0,610** (shallow), **0,853** (deep4) |
| capacité | voir ci-dessous | médiane des 3 seeds ≥ 0,05 |
| optimum | comme § 5.1 | > 20 % à ≤ 5 ou > 20 % à ≥ 195 |
| sélection réelle | sur les pas **appliqués** : part des pas où `kept = proposed` ; médiane de `n_proposed / places`, où places = largeur après − largeur avant | `kept = proposed` sur **tous** les pas, ou médiane < **2** |
| spectre | par événement, log10(max/min) des valeurs propres proposées > 0 ; médiane sur les événements | < **1** décade |
| cible | part des fits avec `width_end == target` | < **80 %** |
| calendrier | part des fits qui ont crû et dont **tous** les pas appliqués tombent à l'époque ≤ 10 | > **50 %** (gonflement unique, pas une croissance) |
| censure | part des pas appliqués avec `s = 4` (sommet de la grille) | > **20 %** |
| jumeau | par construction : à la cible, `GrowingDeep4Net` est Deep4Net et `GrowingShallowFBCSPNet` est ShallowFBCSPNet (parité) | — |

**Capacité d'un bras en croissance.** La croissance tourne pendant le test, avec son propre
`GromoGrowth` construit comme dans `pipelines._build_dl`. La line search a besoin d'essais
jamais vus en entraînement, d'où le protocole suivant :
- 10 essais par classe sont tirés au lieu de 8 ;
- 1 par classe est mis de côté comme held-out disjoint ;
- l'entraînement porte sur 36 essais, et la perte en mode eval est mesurée sur ces 36.

Le chemin `bd` de `step0_capacity.py` est inchangé, ses résultats sont lus.

**Signal préalable, non concluant** : sur le smoke du 13/10 (config H, 11 époques), la
censure valait 100 %, la cible 0 % et le calendrier 100 %. Le smoke s'arrêtait à 11
époques, donc seule la censure est informative ici, et le refus pour censure est un
résultat attendu possible. Il sera rapporté tel quel.

**Sortie** : passe / refus par bras et par critère. Un bras refusé ne passe pas à l'étape 2.
