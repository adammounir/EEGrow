# Jobs en cours

## À FAIRE juste après le dépouillement de l'étage 1 bis — matrice donneurs/receveurs

Demandé par Adam le 26/08, sur la base de la **figure 4.7 (p. 109) de la thèse Coelho
Rodrigues 2019** : trois matrices 52×52 de transfert appairé sur **Cho2017**, source en
colonnes, cible en lignes, réordonnées par sériation (§4.5.2), pour trois procédures de
transfert (DCT, RCT, RPA). Ce qu'elles montrent : des colonnes sombres = bons donneurs,
des lignes sombres = bons receveurs, et un panneau RPA **gris uniforme** là où DCT/RCT
sont contrastés — l'alignement n'améliore pas le transfert partout, il le rend moins
dépendant du couple.

**Ce n'est pas dans le plan actuel, et c'est structurel.** Le LOSO est exactement la marge
en lignes de cette matrice : il écrase tout l'axe source dans un seul agrégat. On obtient
donc les receveurs gratuitement (52 points, dès le dépouillement de ce soir) et on perd
totalement les donneurs.

**Ne pas la refaire avec les réseaux.** Une case de la matrice = un entraînement sur ~210
essais d'un seul sujet. C'est le régime où `RESULTS.md` §1 mesure braindecode à 0.63
contre 0.71 pour Riemann, et §3 trouve des cellules où les 8 réseaux sont pile à la
chance. Une matrice de convnets serait probablement une bouillie grise — indistinguable
du panneau RPA, donc ininterprétable : on ne saurait pas séparer « l'alignement a
homogénéisé » de « rien n'a appris ».

**Plan retenu** : refaire la matrice avec les pipelines riemanniens déjà présents
(`ts_lr`, `fgmdm`), sur Cho2017, × {`none`, `euclidean`}. CPU, donc sur les 46 nœuds des
partitions `normal`/`normal-best` qui dorment, sans toucher au budget GPU.

- coût réel : **52 entraînements, pas 2652** — un modèle par sujet source, évalué sur les
  51 autres cibles ; l'inférence est quasi gratuite ;
- ce n'est **pas** une cellule Hydra : `run_moabb_hydra.py` n'a pas de mode source→cible
  (même limitation que le sharding de folds). Script `pyriemann` autonome à écrire, sous
  `benchmarks/analysis/`, ~une demi-journée ;
- **usage** : pas un bras de plus dans l'étage 2. Le score de donneur par sujet devient une
  **covariable** pour expliquer les gains EA des réseaux — l'analogue par couple du
  Spearman(baseline, gain) = −0.467 mesuré sur bnci.

## Étage 1 — réplication Euclidean Alignment — SLURM **501098** + **501101** (26/08)

Checkout **`/scratch/amounir/eegrow_ea`** (branche `exp/ea-replication`, clone git, donc
`provenance()` stampe le SHA sur chaque ligne). Résultats sous `results_ea/results/`.
Logs `logs/%A_%a.out`. `gpu-best`, `gpu:hopper:1`, array `%6`.

18 cellules = 3 nets (`bd_eegnet`, `bd_shallow`, `bd_deep4`) × {`align=none`,
`align=euclidean`} × 3 seeds (42/43/44). Chaque cellule est un LOSO complet à 9 folds sur
bnci2014_001, `dataset.paradigm=LeftRightImagery` (2 classes — notre config livre le
4-classes, les chiffres du papier sont en 2 classes), protocole **corrigé**
(`train.patience=200 train.selection_monitor=valid_acc`).

- **501098** (array `1`) = la sonde, `bd_eegnet euclidean seed42` : **COMPLETED**, wall
  **450 s**, mean 0.8220, 18 lignes. Elle compte dans le lot (le sbatch saute les
  cellules déjà faites).
- **501101** (array `0,2-17%6`) = les 17 restantes. **TERMINÉ, 18/18 COMPLETED.**

### Résultat — **SOUS-PUISSANCE, pas échec**

| | Δ (EA − raw) | IC 95 % | p | win |
|---|---|---|---|---|
| poolé (3 nets) | **+3.17 pp** | [−0.34, +7.32] | 0.25 | 67 % |
| `bd_shallow` | +3.98 | [+1.31, +7.04] | **0.012** | **89 %** (8/9) |
| `bd_deep4` | +2.77 | [−2.14, +8.61] | 0.73 | 56 % |
| `bd_eegnet` | +2.75 | [−0.79, +6.52] | 0.25 | 67 % |

Écart-type **entre sujets** 6.28 pp → **MDE à n=9 = +6.69 pp**, au-dessus de la cible
+5.05. Il faudrait **13 sujets** ; ce dataset en a 9. Neuf sujets ne peuvent pas détecter
l'effet publié même reproduit à l'identique. Et ce n'est **pas** une affaire de seeds :
passer de 1 à 2 seeds n'a réduit la largeur de l'IC que de 8.2 à 7.5 pp. Voir
[[underpowered-not-null]].

**Où le gain vit** (par sujet retenu, poolé) : l'EA paie exactement où elle devrait —
sujet 5 (raw 59.3) **+14.7 pp**, sujet 7 (raw 67.8) **+12.0** ; rien sur les forts
(sujet 8 raw 96.5 → +0.06, sujet 9 raw 84.3 → −2.7). Spearman(baseline, gain) = −0.467.

**Notre baseline est à 79.82 contre 68.93 dans le papier.** Le protocole corrigé a déjà
sorti la plupart des sujets du régime où l'alignement a du bruit inter-sujet à retirer.
Ce n'est pas un défaut de notre EA : c'est que le +5.05 pp publié est en partie un proxy
de sous-entraînement.

## Étage 1 bis — réplication EA sur **cho2017** (52 sujets) — SLURM **501729 + 501914** — EN COURS

Checkout `eegrow_ea` au commit `ad5b85b`, sbatch `benchmarks/slurm/ea_cho_gpu.sbatch`,
résultats sous `results_cho/`.
Dépouillement : `analysis/ea_replication.py --root results_cho/results`.
**Les 6 cellules tournent en parallèle**, une carte chacune, réparties sur deux jobs :

| cellules | job | carte | lancé | s/fold mesuré | fin projetée |
|---|---|---|---|---|---|
| `bd_eegnet` × {none, euclidean} | **501914**_0-1 | H100 NVL (margpu012) | 15:52 UTC | 448 / 477 | **22:30 / 22:52 UTC** |
| `bd_shallow` × {none, euclidean} | 501729_2-3 | RTX 2080 Ti (margpu019/020) | 14:48 UTC | 432 / 410 | 21:12 / 20:59 UTC |
| `bd_deep4` × {none, euclidean} | 501729_4-5 | RTX 2080 Ti (margpu021/022) | 14:48 UTC | 366 / 367 | 20:13 / 20:14 UTC |

**Mur de la campagne : 22:52 UTC** (≈ 00h52 Paris), contre 01:30 UTC avant la bascule.

> **Relevé à 16:46 UTC**, `n_train=10320`, `epochs=200` sur les 6 cellules — le budget
> complet s'applique bien, `patience=200` neutralise l'arrêt anticipé comme prévu. Folds
> faits : deep4 18/52, shallow 15/52, eegnet 6/52 (le job hopper **reprend au fold 0**, le
> garde de reprise est au niveau du CSV, pas du fold — les 13 et 4 folds turing archivés en
> `.bak` ne comptent pas). Projection = (52 − faits) × médiane des 6 derniers folds, en
> excluant les folds turing des deux cellules eegnet. L'extrapolation initiale à partir de
> la sonde 10 sujets (463 s/fold hopper) donnait 22:33 — elle était juste à 4 % près.

> **Pourquoi deux jobs.** À 15:50 UTC les cartes hopper que `mkalla` tenait ce matin se
> sont libérées (3 dispos). Le mur de la campagne, ce sont les deux cellules `bd_eegnet`
> et elles seules : 10.7 h sur turing contre 6.7 h sur hopper. Les quatre autres cellules
> finissent à 20:25 et 21:05 quoi qu'il arrive, donc les redémarrer aurait jeté 1 h de
> calcul pour zéro gain de mur — elles restent sur turing. Seule la paire `bd_eegnet` a
> été relancée sur hopper (501914), avec les deux bras sur **le même nœud margpu012**.
>
> **La contrainte de carte constante est par PAIRE, pas globale.** Le test EA est un delta
> apparié *à l'intérieur* d'un modèle ; le Δ de `bd_eegnet` n'est jamais comparé à celui de
> `bd_shallow`. Il suffit donc que les deux bras d'une même paire partagent la carte, ce
> qui est vérifié ici (H100 NVL / H100 NVL, RTX 2080 Ti / RTX 2080 Ti). Formulation plus
> juste que celle de l'encadré ci-dessous, écrite quand j'imposais l'homogénéité aux six
> cellules à la fois.
>
> **Ordre de la bascule, pour ne pas perdre de folds.** 501914 a été soumis *avant*
> d'annuler 501729_0-1, et l'annulation n'est intervenue qu'une fois les deux tâches
> hopper RUNNING. La fenêtre de recouvrement (~1 min) est sans risque parce qu'une cellule
> passe ses ~3 premières minutes à charger les données, avant toute écriture dans le
> `__fits.jsonl`. Si hopper avait été pris entre-temps, on n'aurait rien perdu du tout.
> Les fits turing partiels sont archivés en `*.turing_cancelled.jsonl.bak` (907 419 et
> 284 386 octets) — à ne pas mélanger aux fits hopper dans une analyse de coût.

> **501357 remplacé/annulé.** Premier lancement à 13:20 UTC sur `gpu:hopper:1` : 1 cellule
> tournait, 5 attendaient. Annulé à 14:47 après ~1h30 de cellule 0. Voir l'encadré GPU
> ci-dessous — c'est la seule chose qui a changé, la science est identique.

> **Inventaire GPU au 26/08 15:50 UTC** (cartes libres et *utilisables*, cf. plancher
> sm_75) : 3 hopper (margpu012 ×2, margpu013), 5 ampere (margpu005/006/008, + margpu010
> ×2 en partition `gpu`), 3 rtx (margpu003 ×2, margpu002), 1 turing (margpu028). Les 15
> pascal et 3 volta libres ne comptent pas. Margaret est le **seul** cluster accessible :
> `~/.ssh/config` ne contient que `margaret02` et la passerelle `ssh-sif.inria.fr`.
>
> **Le vrai levier pour les campagnes suivantes n'est pas la carte, c'est le sharding des
> folds.** Une LOSO à 52 sujets, c'est 52 folds indépendants exécutés *en série* dans une
> seule cellule. Les répartir sur 4 cartes diviserait le mur par ~4 — bien plus que le
> ×1.60 hopper/turing. On ne peut pas le faire en restreignant `dataset.subjects` (ça
> change l'ensemble d'entraînement, donc l'estimand) : il faudrait exposer « évalue ces
> folds-là, entraîne sur tous les autres » dans `run_moabb_hydra.py`. À faire avant
> l'étage 2 si le volume augmente.

6 cellules = 3 nets × {none, euclidean} × **1 seed**. Une seule seed parce qu'à n=52 le
MDE tombe à ~2.4 pp : la puissance vient des sujets, pas des seeds — l'inverse exact du
raisonnement sur bnci, et pour la même raison. cho2017 est nativement 2 classes
(`LeftRightImagery` dans sa config), et ses 52 sujets sont déjà dans le cache MNE partagé
(`MNE-gigadb-data`, 10 Go) : aucun job GPU ne télécharge. Vérifié dans moabb 1.5.0 :
`Cho2017().subject_list` fait bien **52** entrées, aucune exclusion — donc n=52 pour la
puissance, pas 49.

### Le choix de la carte — deux erreurs successives, corrigées par la mesure

**`ampere` → `hopper` → `turing`.** Le pin initial venait de l'inventaire : ~18 cartes
ampere contre 3 hopper, donc le pool large draine en une vague. Faux — les 18 ampere
étaient **toutes allouées**, et l'estimation Slurm pour la sonde ampere donnait 37 h
d'attente (501230, annulé). D'où la règle corrigée : **le critère est ce qui est libre,
pas ce qui existe.** La règle est bonne ; je l'ai appliquée à une liste de deux entrées.
Un `sinfo -N` sur toute la partition montrait **10 nœuds entièrement IDLE** pendant que
5 des 6 cellules faisaient la queue derrière 2 cartes hopper tenues par `mkalla`. Un pin
mono-carte transforme une campagne 6-parallèle en campagne série : 6.7 h par cellule dans
les deux cas, mais 6.7 h de mur contre ~33 h.

**Toutes les cartes libres ne sont pas utilisables.** L'env `bench` porte
`torch 2.13.0+cu130`, compilé pour sm_75/80/86/90/100/120. Les **19 cartes pascal**
(P100, sm_60) et les **volta** (sm_70) plantent à la première convolution :

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible
RuntimeError: GET was unable to find an engine to execute this computation
```

(sonde 501434, rc=1 après 91 s). Le pool libre réellement exploitable était donc de
**14 cartes turing**, pas 35. À retenir pour toute campagne future sur ce cluster.

**Ce que coûte le fait de quitter hopper : ×1.60, mesuré.** Sondes 501315 (hopper) et
501433 (turing), cellule identique, mêmes 10 sujets :

| carte | fold, régime permanent |
|---|---|
| H100 (hopper) | **81.7 s** |
| RTX 2080 Ti (turing) | **131 s** |

Une carte de 2018 qui ne perd que 60 % contre une H100, c'est la signature d'une charge
**limitée par la latence** : ces convnets sont assez petits pour que le coût de lancement
des noyaux domine les FLOPs. C'est la même physique qui fait que `bd_eegnet` est le plus
cher des trois réseaux alors qu'il a le moins d'opérations. 6.7 h × 1.60 = **10.7 h par
cellule**, 6 en parallèle, contre 33 h sérialisées sur une hopper.

**La carte doit rester constante sur les 6 cellules.** Chaque contraste EA est apparié
dans un modèle à seed fixe ; un bras sur H100 et l'autre sur 2080 Ti laisserait la
sélection d'algorithme cuDNN différer entre les bras et ajouterait du bruit d'échelle
« seed » au delta, gratuitement. C'est pourquoi 501357 a été annulé au lieu d'être
recyclé : il n'achetait aucune minute de mur (les 6 cellules démarrent ensemble ici) et
il aurait coupé la paire `bd_eegnet` en deux générations de cartes. Les enregistrements
de fits hopper sont conservés sous
`bd_eegnet__seed42__fits.hopper_cancelled.jsonl.bak` (697 921 octets) et ne doivent pas
être mélangés aux fits turing dans une analyse de coût. Throttle passé de `%3` à `%6`.

### Chiffrage — sonde 501315, **COMPLETED**, 10 sujets, `bd_eegnet` raw

Écrite dans `results_cho_probe10/` : elle change l'estimand (un LOSO à 10 sujets entraîne
sur 9, pas sur 51) et ne doit jamais être confondue avec la campagne.

| mesure | valeur |
|---|---|
| WALL | 1053 s, dont 1002 s de fits → 51 s de chargement de données |
| fold, régime permanent | **81.7 s** à `n_train=1880` (le 1ᵉʳ fold à 138 s est le warmup CUDA) |
| régime d'entraînement | `stop_reason: budget`, **200 époques à chaque fold** |
| `MaxRSS` | **3.96 Go** |
| dims | 64 canaux, 750 échantillons |
| score moyen | 0.6959 AUC — loin du hasard, le harness marche sur cho2017 |

**Pourquoi l'extrapolation est fiable ici et pas en général** : `patience=200` avec
`max_epochs=200` désactive l'arrêt anticipé, donc chaque fit consomme exactement 200
époques et le temps est **linéaire en `n_train`**. Sans ça le nombre d'époques varie par
fold et aucune règle de trois ne tient.

À 52 sujets : `n_train` = 51/9 × 1880 ≈ 10 650 (×5.67), 52 folds →
**463 s/fold × 52 ≈ 6.7 h** pour `bd_eegnet`. Ratios inter-modèles mesurés sur le lot
bnci (même code, même protocole) : `bd_deep4` ×0.59, `bd_shallow` ×0.52 —
`bd_eegnet` est le **plus cher** des trois (46.5 s contre 27.5 et 24.0), les convolutions
séparables d'EEGNet sont limitées par la latence, pas par les FLOPs. Donc la sonde a
mesuré le pire cas. Total **≈28 GPU-h**.

Sur turing (×1.60) : `bd_eegnet` **743 s/fold → 10.7 h**, `bd_deep4` ≈ 6.3 h,
`bd_shallow` ≈ 5.6 h. Les 6 cellules tournant en parallèle, le mur de la campagne est
celui des deux cellules `bd_eegnet`, soit **10.7 h**.

Marges : mur demandé 2 jours contre 6.7 h de pire cellule ; mémoire 64 Go contre ~21 Go
projetés (3.96 Go × 5.2, majorant puisque l'overhead fixe ne monte pas). Les deux couvrent
un requeue en cours de route.

**Piège d'estimation à ne pas refaire** : mon premier chiffrage bnci était 8× trop haut —
j'avais lu les médianes v5 cross_subject (330 s pour `bd_shallow`) comme du temps *par
fit* alors que c'est du temps *par ligne*, et une cellule fait 18 lignes pour 9 fits.
D'où la règle : sonder, puis lire l'extrapolation sur la sonde.

**Garde-fou fixé d'avance** — Junqueira, Aristimunha, Chevallier & de Camargo,
arXiv 2401.10746, Table 2, BNCI2014_001 LOSO 2 classes :
No-EA 68.93 ± 12.61 → Offline-EA 73.98 ± 11.21, soit **+5.05 pp**. Le niveau absolu
n'est **pas** le test (leur harness n'est pas le nôtre) ; le test est le delta apparié
par sujet, où tout ce qui sépare les harnesses est commun aux deux bras et s'annule.

**Le niveau 0.822 n'est pas une fuite.** Vérifié : le classement par sujet de la sonde
(3, 8, 9 en tête ; 2, 5, 6 en bas) reproduit celui de v5 sur ce dataset. L'écart au
papier vient de la tâche 2 classes et du protocole corrigé — v5 tournait le protocole
cassé. L'EA aligne chaque sujet sur sa propre covariance sans jamais toucher `y`, donc le
sujet retenu est aligné avec son propre enregistrement non labellisé : c'est le réglage
« offline » du papier, rien ne traverse la frontière train/test.

## Contrôles fixes manquants — SLURM **500952** — **TERMINÉ 24/24** (26/08)

Soumis le 26/08/2026 ~07h50. Array `0-23%3`, `gpu-best`, `gpu:hopper:1` (même carte que
500573 : les comparaisons sont appariées fit par fit). 3 modèles (`fix_shallow`,
`fix_eegnex`, `fix_sccnet`) × carré 2×2 × 2 seeds = **24 cellules**. Durées mesurées :
`fix_shallow` 2:25–7:31, `fix_eegnex` 14:24–22:36, `fix_sccnet` 2:03–4:54.

### Résultat — famille GROWTH complète, `full_acc`, niveau sujet (n=9)

| paire | Δ | IC 95 % | p | holm | win |
|---|---|---|---|---|---|
| `grow_sccnet` vs `fix_sccnet` | +0.0176 | [+0.004, +0.031] | 0.055 | 0.22 | 7/9 |
| `grow_deep` vs `fix_deepeeg` | +0.0008 | [−0.018, +0.021] | 1 | 1 | 3/9 |
| `grow_shallow` vs `fix_shallow` | −0.0064 | [−0.016, +0.003] | 0.36 | 0.72 | 4/9 |
| `grow_eegnex` vs `fix_eegnex` | −0.0098 | [−0.020, −0.001] | 0.13 | 0.39 | 3/9 |

**0/4 significatives après Holm, aucun sign flip.** Les deux résultats publiés étaient
des artefacts de contrôle : `grow_eegnex` passe de −0.0892 (contre `bd_eegnex`, 0/9
sujets, holm=0.012) à −0.0098 ns contre son vrai jumeau ; `grow_shallow` passe de +0.0213
(contre `bd_shallow`) à −0.0064 ns, et vaut **−0.0307 p=0.0078** au protocole livré.
Seul `grow_sccnet` va dans le sens de la croissance, et seulement à budget plein
(+0.0051 ns livré → +0.0176 ; +0.0191 9/9 sujets holm=0.016 sur `full_loss`).
Le protocole déplace le verdict sur `grow_shallow` : DiD **+0.0243 p=0.027**.

Fichiers : `grid_fix_cells.txt`, `grid_fix_gpu.sbatch`, sorties dans
`grid_models/<model>/<arm>/results/` (à côté des 64 cellules de 500573, mêmes chemins).
Dépouillement : `analysis/growth_contrast.py --root $BUD/grid_models`.

### Pourquoi

Trois paires de croissance sur quatre opposaient `grow_*` à la **référence
braindecode**. Ce ne sont pas les mêmes codebases, donc l'écart apparié contient la
croissance **plus** tout ce qui les sépare — mesuré, l'init Xavier stock de braindecode
démarre `bd_shallow` 0.34 nats au-dessus de ln(k) contre +0.08 pour nos réseaux. Seul le
bras deep avait un contrôle de la même classe (`fix_deepeeg`), et c'est le seul qui ne
dit rien. Ces 24 cellules donnent ce contrôle aux trois autres : même classe, même
fichier, même init, mêmes callbacks, construit directement à la géométrie d'arrivée ;
seul `_can_grow` change.

### Le bug trouvé en route — `grow_eegnex` n'a jamais été width-matched

Dans EEGNeX, `filter_1` dimensionne **deux** endroits : la jonction growable (conv1
fan-out = conv2 fan-in) **et** la conv de queue `block_5 (filter_2 -> filter_1)`, dont la
sortie est aplatie dans le classifieur. La croissance n'élargit que la jonction. Donc
`grow_eegnex` (filter_1=2, cible 8) finit avec une jonction 8 et une **queue 2**, soit
**70 features** en entrée du classifieur, là où `bd_eegnex` (filter_1=8) en a **280**.

Conséquence directe : le **−0.089** de `grow_eegnex` (perte sur 9 sujets sur 9, le seul
résultat à survivre à Holm, et le seul « contre la croissance ») comparait un réseau à
classifieur 4× plus étroit à la référence. **Ce n'est pas une mesure de la croissance.**

Corrigé par un kwarg `filter_1_in` (miroir de `w2_in` sur le bras deep) qui découple les
deux largeurs ; `fix_eegnex` = `filter_1: 2, filter_1_in: 8`. Un contrôle naïf
`filter_1: 8` aurait construit un réseau différent et plus gros (56 580 params / 280
entrées contre 52 656 / 70).

**Rien à re-lancer** : `filter_1_in` vaut `filter_1` par défaut, donc `grow_eegnex`
construit exactement le même réseau qu'avant, bit pour bit. Les 64 cellules de 500573
tiennent.

### Vérifications faites avant de soumettre

- `tests/test_models.py::test_fixed_control_matches_grown_geometry` (nouveau) : chaque
  bras est **réellement grandi** jusqu'à sa cible via le callback du benchmark, puis les
  shapes de son `state_dict` sont diffées contre le jumeau gelé. 4/4 MATCH (c'est ce test
  qui a fait tomber eegnex). Suite complète : **51 passed**.
- Les trois YAML construits par le chemin du benchmark, en local **et** sur le checkout
  déployé : `can_grow=False`, largeurs 40 / 8 / 22, params 47 364 / 52 656 / 14 794 —
  identiques aux réseaux grandis.
- Garde in-job : assert `eegrow.__file__` sous `eegrow_budget/` **et** présence de
  `filter_1_in` dans la signature (un checkout périmé construirait la mauvaise géométrie
  en silence).

### `analysis/growth_contrast.py` — deux familles désormais

`FAMILIES` sépare **GROWTH** (`grow_X` vs `fix_X`, la seule question sur la croissance)
de **REFERENCE** (`grow_X` vs `bd_X`, « notre implémentation vaut-elle la référence ? »).
Holm s'applique **dans** une famille, jamais à travers. Une paire dont le contrôle n'est
pas encore là est sautée explicitement, pas rapportée vide.

## Généralisation aux 8 autres bras — SLURM **500573** — **TERMINÉ**

Lancé le 25/08/2026 ~23h00, fini le 26/08 au matin. Array `0-63%3`, `gpu-best`,
`gpu:hopper:1`. Carré 2×2 **complet** (`p20_loss`, `p20_acc`, `full_loss`, `full_acc`)
× 8 modèles non-ML × 2 seeds = **64 cellules**, **64/64 à rc=0**, 36 unités appariées
par cellule (9 sujets × 2 sessions × 2 seeds).

Analyse : `analysis/budget_models.py --root $BUD/grid_models`, sortie conservée dans
`/scratch/amounir/eegrow_budget/analysis_final.out`.

### L'unité d'analyse — à lire avant les chiffres

Une cellule fait 36 lignes = 9 sujets × 2 sessions × 2 seeds. Ce ne sont **pas** 36
observations indépendantes : les seeds sont de la réplication interne du même
sujet-session, et les deux sessions partagent un sujet. Un Wilcoxon sur 36 lignes
corrélées surestime sa propre significativité — mesuré, jusqu'à **4 ordres de grandeur**
(`grow_shallow`/`p20_acc` : p=3.8e-07 à n=36 contre p=0.0039 à n=9).

**Tous les p ci-dessous sont au niveau sujet (n=9)**, sessions et seeds moyennées. C'est
celui qui tient pour un papier. `analysis/budget_models.py --subject-level` et
`analysis/growth_contrast.py` produisent les deux niveaux. Attention au plancher : à n=9
le Wilcoxon bilatéral ne peut pas descendre sous **p=0.0039**, atteint ssi **les 9 sujets
vont dans le même sens** — c'est l'affirmation la plus forte disponible ici, et il faut
lire les tailles d'effet et les IC, pas les étoiles.

### Résultat 1 — le défaut livré sous-entraînait 6 bras sur 9

`full_acc` (budget plein + sélection `valid_acc`) contre le défaut `p20_loss` :

| modèle | shipped | corrigé | Δ (p, n=9) | interaction | additif ? |
|---|---|---|---|---|---|
| `bd_deep4` | 0.3501 | 0.4818 | **+0.1316** (0.004 = 9/9) | +0.0854 (0.004) | **non** |
| `grow_shallow` | 0.5316 | 0.6007 | **+0.0690** (0.004 = 9/9) | +0.0192 (0.02) | **non** |
| `bd_shallow` | 0.5147 | 0.5793 | **+0.0646** (0.004 = 9/9) | +0.0273 (0.02) | **non** |
| `grow_sccnet` | 0.6581 | 0.6959 | +0.0379 (0.004 = 9/9) | +0.0301 (0.2) | oui |
| `fix_deepeeg` | 0.3853 | 0.4190 | +0.0337 (0.008) | ns | oui |
| `grow_deep` | 0.3920 | 0.4198 | +0.0278 (0.03) | ns | oui |
| `bd_eegnex` | 0.6262 | 0.6568 | +0.0306 (**0.1**) | ns | oui |
| `bd_sccnet` | 0.6533 | 0.6756 | +0.0223 (**0.2**) | +0.0157 (0.07) | oui |
| `grow_eegnex` | 0.5579 | 0.5676 | +0.0097 (**0.7**) | ns | oui |

Holm sur les 9 bras : **5 survivent** (les quatre à 0.036 + `fix_deepeeg` à 0.04) ;
`grow_deep` tombe à 0.12. À n=36 on lisait 8 bras sur 9 et une interaction sur 5 — c'était
la corrélation intra-sujet, pas le signal.

Le **budget** reste positif sur les 9 bras sans exception (dans les deux colonnes de
sélection). La **sélection seule** est bruitée et parfois négative (`grow_eegnex` −0.0190,
`bd_eegnex` −0.0078) : c'est la **combinaison** qui porte l'effet, pas un knob isolé.
Interaction significative à n=9 sur **3 bras** (`bd_deep4`, `bd_shallow`, `grow_shallow`)
→ sur ceux-là aucun effet principal ne peut être cité seul. La diagonale seule l'aurait
raté : justification a posteriori du carré complet.

`grow_eegnex` est le bras le plus plat (p=0.7) et c'est aussi le seul qui ne grandit
quasiment pas (largeur finale médiane **8**, 2 événements) — il n'y a rien à
sous-entraîner. Aucun des 8 nouveaux bras n'a sa cellule livrée sous le seuil de chance
(0.295) ; `bd_deep4` reste le seul dans ce cas.

### Résultat 2 — croissance vs contrôle fixe, apparié (`analysis/growth_contrast.py`)

Appariement pris dans les configs, pas inventé : `grow_deep` ↔ **`fix_deepeeg`** (même
classe gelée à la géométrie d'arrivée), les trois autres contre leur référence
braindecode à la largeur cible. Δ = croissance − fixe, niveau sujet, IC bootstrap 95 % :

| paire | protocole livré | protocole corrigé | DiD (le protocole bouge-t-il le verdict ?) |
|---|---|---|---|
| `grow_shallow` vs `bd_shallow` | +0.0169 [+0.003, +0.030] p=0.055 | **+0.0213** [+0.008, +0.035] p=0.039 | +0.0044 ns |
| `grow_sccnet` vs `bd_sccnet` | +0.0048 ns (win 56 %) | **+0.0204** [+0.002, +0.037] p=0.074 | **+0.0156 p=0.012, holm=0.047** |
| `grow_deep` vs `fix_deepeeg` | +0.0067 **p=0.91, win 44 %** | +0.0008 **p=1, win 33 %** | −0.0059 ns |
| `grow_eegnex` vs `bd_eegnex` | −0.0683 p=0.02 | **−0.0892** [−0.111, −0.066] **win 0/9, holm=0.016** | −0.0209 p=0.074 |

**Aucun sign flip.** Correction d'une affirmation antérieure : « l'avantage de la
croissance sur le deep s'évapore avec le budget » est **faux**. `grow_deep` vs
`fix_deepeeg` n'a jamais été significatif — p=0.91 au protocole livré, la croissance perd
sur 5 sujets sur 9, IC [−0.015, +0.034]. Le +0.0067 était une différence de moyennes non
appariée, exactement l'erreur que ce contraste corrige.

Ce qui tient :
- **`grow_eegnex` est le résultat le plus net, et il est *contre* la croissance** :
  −0.089, la croissance perd sur **9 sujets sur 9**, seul à survivre à Holm. Le bras
  atteint pourtant sa largeur cible (8/8).
- **`grow_sccnet` est le seul cas de vraie dépendance au protocole** : DiD +0.0156,
  holm=0.047. Il passe de « rien » à une tendance positive.
- **`grow_shallow`** est la seule tendance positive robuste en direction (IC excluant 0
  aux deux protocoles, 7 sujets sur 9), mais ne passe pas Holm à n=9.

**Caveat à écrire dans le papier** : trois paires sur quatre opposent *notre*
implémentation growing à *celle de braindecode*, donc l'écart contient la croissance
**plus** tout ce qui sépare les deux codebases. Seule la paire deep est un contraste de
croissance propre — et c'est celle qui ne dit rien.

### Résultat 3 — le modèle scoré n'est pas le modèle qui a grandi

`width_lost` = largeur finale − largeur à l'époque restaurée, par fit (n=180/cellule) :

| bras | `frac_narrowed` | p90 | max |
|---|---|---|---|
| `grow_shallow` p20_loss (livré) | **0.39** | 18.9 / 40 | 22 |
| `grow_shallow` p20_acc | 0.37 | 14.1 | 26 |
| `grow_shallow` full_loss | 0.32 | 23 | 26 |
| `grow_shallow` full_acc | **0.07** | 0 | 24 |
| `grow_deep` p20_acc | 0.20 | 16 / 32 | 16 |
| `grow_deep` full_acc | 0.08 | 0 | 16 |
| `grow_sccnet` / `grow_eegnex` | 0.00–0.06 | 0 | ≤6 |

Sur `grow_shallow`, **4 fits sur 10** au défaut livré sont scorés sur un réseau plus
étroit que celui produit par l'entraînement, et le décile haut perd **la moitié de la
largeur cible**. Budget plein + `valid_acc` ramène à 7 %. Le benchmark mesurait donc,
sur une minorité substantielle de fits, une architecture qu'il n'annonçait pas — et
l'accuracy seule ne le révèle jamais.

**En médiane la perte n'est que de 0 à 1 filtre** : l'effet est concentré sur une
minorité de fits, c'est la distribution (`frac_narrowed`, p90) qui est le bon readout,
pas la médiane. Le `grow_step` n'est jamais le coupable : les 4 bras atteignent leur
cible dans **les deux** budgets — le confounding budget↔largeur finale annoncé au départ
**n'existe pas** (voir la prédiction réfutée du probe 500569 plus bas).

### Design (conservé)

- Modèles : `bd_shallow`, `bd_eegnex`, `bd_sccnet`, `fix_deepeeg`,
  `grow_shallow`, `grow_deep`, `grow_eegnex`, `grow_sccnet`.
- Cellules : `grid_models_cells.txt` (`MODEL ARM PATIENCE SEL SCHED SEED`).
- Sbatch : `grid_models_gpu.sbatch`, logs `grid_models_logs/500573_<task>.out`.
- Sortie : `/scratch/amounir/eegrow_budget/grid_models/<model>/<arm>/results/...`
  (`grid/` de bd_deep4 n'est pas touché).
- Reprise : chaque tâche saute sa cellule si le CSV existe (`--requeue` sûr).

**Pourquoi le carré complet et pas la diagonale** : sur `bd_deep4` la diagonale seule
aurait dit « +0.132, le budget était trop court » et aurait raté le fait que les effets
simples ne s'additionnent pas (interaction +0.085, p=1.1e-07). Refaire la diagonale
seule ici reproduirait l'ambiguïté 8 fois. Le surcoût est ~1 h de mur.

### Probe **500569** (`grow_shallow`, H100) — la porte avant de soumettre

Coût, mesuré sur le pire cas des bras growing (gap 8→40, donc le plus d'événements) :
**31.5 s/fold-row** à budget plein et **13.9 s** à patience 20, contre 23.7 / 7.3 pour
`bd_deep4`. La croissance ne coûte que **×1.33** : le `grow_step` (et son `eigh`) ne
tourne qu'une époque sur cinq et s'arrête dès `target_width` atteint.

**Une prédiction réfutée.** J'annonçais que patience 20 couperait les fits à l'époque 24
et laisserait 4 événements de croissance, donc un réseau inachevé. Mesuré : les fits
`p20_loss` durent **36 à 94 époques** (7 événements) et atteignent **40/40 dans les deux
bras**. L'extrapolation venait de `bd_deep4`, dont la valid_loss explose à l'époque 4 ;
`grow_shallow` n'a pas ce défaut. Pas de confounding budget↔largeur finale.

**Mais un effet structurel, mesuré pour la première fois** :

```
p20_loss  restored=23  width_at_restored=30   width_final=40
p20_loss  restored=16  width_at_restored=25   width_final=40
p20_loss  restored=19  width_at_restored=26   width_final=40
full_acc  restored=193 width_at_restored=40   width_final=40   (40/40 sur les 10 fits)
```

Le réseau finit de grandir, mais `RestoreBestModel` sur `valid_loss` rend un modèle à
**25–30 filtres sur 40**. Le modèle scoré n'est pas le modèle qui a grandi — le
benchmark mesurait une architecture plus petite que celle qu'il annonçait. Même cause
racine que `bd_deep4` (`valid_loss` sur 46 essais de validation), effet structurel et
non plus seulement temporel. Sujet 1 : 0.7170 contre 0.6770 (+0.040).

Note : le « 14/40 » de v5 n'est plus le comportement du code — c'était le
`statistical_threshold` absolu de gromo, remplacé depuis par le plancher relatif.

Détail à connaître de tout lecteur de largeur : `grow_width_after` n'est enregistré que
**sur les époques de croissance** (`skorch_integration._record_growth`), donc lire la clé
à une époque arbitraire rend `None` 4 fois sur 5 ; il faut la dernière valeur enregistrée
avant cette époque.

## bd_deep4 — budget × sélection (Margaret, GPU) — **run principal**

SLURM **500458** (cellules 0-7) + **500475** (`full_loss`, cellules 8-9), array,
partition `gpu-best`, `gpu:hopper:1` (H100 NVL). **TERMINÉ** le 25/08/2026 ~20h45,
18 min de mur pour les 10 cellules.

### Résultat — le carré 2×2, apparié sur 36 unités (9 sujets × 2 sessions × 2 seeds)

```
                    sel=valid_loss   sel=valid_acc
patience  20            0.3501           0.3390
patience 200            0.4074           0.4818
```

| contraste | effet | p (Wilcoxon) |
|---|---|---|
| budget, à sélection `valid_loss` | +0.0573 | 1e-04 |
| budget, à sélection `valid_acc` | +0.1428 | 7.4e-09 |
| sélection, à patience 20 | −0.0111 | 0.003 |
| sélection, à patience 200 | +0.0743 | 5.5e-07 |
| **interaction** | **+0.0854** | **1.1e-07** |
| les deux vs le défaut livré | +0.1316 | 6.9e-08 |

Les effets simples **ne s'additionnent pas** : il faut les deux knobs. Mécanisme —
`full_loss` et `full_acc` ont des trajectoires bit-identiques (`final_tr` 0.1366,
`best_vacc` 0.5592 identiques), mais `full_loss` restaure l'époque **5** quand le pic
de valid_acc est à l'époque **158**. `valid_loss` est un proxy cassé sur ces splits
de validation (46 essais) et il l'est deux fois : comme critère d'arrêt et comme
critère de sélection.

Corollaires : perte train finale **0.1366** à budget plein contre 1.1874 à l'arrêt
anticipé (ln 4 = 1.386) — « `bd_deep4` n'ajuste pas ses données » était un artefact du
budget. Le **cosine nuit** une fois réellement recuit (`lr_final` 0.0000) : 0.4276
contre 0.4818, restored 99 contre 158. Piste à retirer.

Portée : `patience` et `selection_monitor` sont **globaux**. Reste à mesurer lesquels
des 8 autres bras deep sont dans le même cas avant de conclure sur v5.

- Checkout **neuf** `/scratch/amounir/eegrow_budget` (archive envoyée du portable).
  `/scratch/amounir/eegrow` n'est PAS touché : il porte du travail v5 non commité.
  `PYTHONPATH=$BUD/src` écrase le `.pth` editable de l'env `bench` qui pointe sur
  l'ancien checkout ; le job l'`assert` avant de lancer.
- Env : conda `bench`, torch 2.13+cu130, **braindecode 1.5.2** (= la stack de v5,
  contre 1.7.0 en local). Env partagé non modifié.
- Cellules : `grid_cells.txt` (arm patience sel sched seed), une par tâche d'array.
- Sortie : `/scratch/amounir/eegrow_budget/grid/<arm>/results/...`, logs
  `grid_logs/500458_<task>.out`.
- Reprise : chaque tâche saute sa cellule si le CSV existe (`--requeue` sûr).
- Analyse : `PYTHONPATH=$BUD/src python analysis/deep4_budget.py --root $BUD/grid`

Probe **500457** (margpu012, H100), sujet 1 seed 42 — la porte avant de soumettre :

| | local MPS / bd 1.7.0 | H100 / bd 1.5.2 |
|---|---|---|
| `full_acc` | 0.5538 | 0.5728 |
| `p20_loss` | 0.4185 | 0.3764 |

47 s/sujet à budget plein contre 574 s en local (**×12** ; ×4.6 seulement sur les
cellules courtes, où le coût fixe d'épochage MOABB domine et ne dépend pas du device).
Les deux stacks concordent sur le sens et l'ordre de grandeur ; le +0.196 côté cluster
est obtenu sur **exactement** la stack de v5.

## bd_deep4 — budget × sélection (local, MacBook) — **ARRÊTÉ, objectif atteint**

Lancé le 25/08/2026 ~20h00 (PID 37506), **arrêté le 25/08 ~23h20** : la réplication
croisée qu'il devait fournir était acquise, le reste ne mesurait plus rien.

### Ce qu'il a livré — réplication sur 2 devices et 2 versions de braindecode

| bras | local (MPS, bd 1.7.0) | cluster (CUDA, bd 1.5.2) | écart |
|---|---|---|---|
| `p20_loss` | 0.3507 | 0.3501 | 6e-4 |
| `p20_acc` | 0.3432 | 0.3390 | 4e-3 |
| `full_acc` seed0 | 0.4801 | 0.4818 | 2e-3 |

Trois coins du carré sur quatre, dont **`full_acc`, celui qui porte le résultat**. Le
`+0.132` n'est donc pas un artefact de stack ni de device — c'est l'argument qu'on veut
pouvoir écrire dans le papier.

### Pourquoi la fin ne servait plus

Restaient `full_acc` seed1 (~75 min) et les deux `full_cos` (~150 min). La seed
supplémentaire n'ajoutait qu'une réplication de plus sur un effet déjà répliqué, et le
bras cosine est une piste **abandonnée** : le cluster l'a mesuré nuisible une fois
réellement recuit (0.4276 contre 0.4818). ~3 h 40 de MacBook libérées.

### Détails du run (conservés pour la reprise si jamais nécessaire)

Détaché (`nohup`), séquentiel, **reprenable** : relancer la même commande saute les
cellules dont le CSV existe (le CSV est écrit en dernier).

- Launcher : `benchmarks/exp_deep4_budget.py` (doc = design complet + ce qui a été réfuté avant)
- Analyse : `benchmarks/analysis/deep4_budget.py --root <out>`
- Sortie : `/private/tmp/claude-501/-Users-adammounir-Desktop-Inria-Exploration-Braindecode/8345a69a-724a-4bf4-8212-43fd080b8230/scratchpad/budget`
- Log : le même chemin + `.log`
- Reprise après interruption : relancer exactement la même commande, les cellules finies sont sautées
  (le CSV par cellule est écrit en dernier, donc sa présence = cellule terminée).

8 cellules = 4 bras × seeds {0,1}, bnci2014_001 / within_session / bd_deep4, 9 sujets.

| bras | patience | selection_monitor | schedule | ~min/cellule |
|---|---|---|---|---|
| `p20_loss` | 20 (défaut) | valid_loss | — | 12 |
| `p20_acc` | 20 | valid_acc | — | 12 |
| `full_acc` | 200 = pas d'early stopping | valid_acc | — | 85 |
| `full_cos` | 200 | valid_acc | cosine | 85 |

Total attendu ≈ **6 h 30**, mesuré (574 s/sujet à budget plein sur le probe, pas extrapolé).

### Ce qu'on teste

L'expérience LR a réfuté le pas de gradient (cosine +0.0018, lowlr explose pareil).
Le mécanisme mesuré est ailleurs : la valid loss touche son minimum à l'**époque 4** puis
explose, `stop_monitor=valid_loss` + patience 20 coupe à 24, et `RestoreBestModel` rend
**le modèle de l'époque 4**. Le benchmark scorait un réseau entraîné 4 époques.

Probe (sujet 1, seed 42, budget plein + sélection valid_acc) : **0.5538** contre 0.4185
à patience 20 et 0.272 pour v5. Perte train finale 0.076–0.118 au lieu de 0.869 —
« bd_deep4 n'ajuste pas ses données » était un artefact du budget. `ep_max_vacc` tombe
entre 85 et 182, ce qui réfute au passage l'idée que l'argmax de valid_acc serait un
pic chanceux précoce.

### Portée si ça se confirme

`selection_monitor` et `patience` sont des knobs **globaux**. Un effet de sélection ici
est un résultat sur le protocole du benchmark, pas un patch sur un modèle — et il
invalide les cellules deep de v5, qu'il faudra relancer.
