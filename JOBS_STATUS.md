# Jobs en cours

## 11/09 — « la croissance aide les mauvais sujets » : un tiers d'artefact, puis le plancher de chance

Analyse locale (pas de job cluster) : `benchmarks/analysis/rtm_sort_axis.py`, sur
`perf_final/scores`, `cross_session` / `align=none`, `grow_deep` contre `bd_deep4`,
3 graines × 6 datasets, 116 sujets.

**Le problème.** La figure « every subject, both arms » trie les sujets par le score de
`bd_deep4` puis lit `grow − bd` le long de cet axe. Le score de bd est donc à la fois la
position ET (au signe près) une moitié du delta : régression vers la moyenne garantie.
Test : classer avec la graine k, mesurer le delta avec la graine m ≠ k (même delta, même
niveau de bruit sur l'axe, seule l'indépendance change), moyenné sur les 6 paires.

**Étage 1 — un tiers de la pente est fabriqué par le tri.** Poolé sur 116 sujets, pente en
points de score par écart-type de bd (négatif = la croissance aide les faibles) :
NAÏVE **−0.0531** [−0.0640, −0.0396] → INDÉP **−0.0360** [−0.0479, −0.0218]. Artefact
**32 %**. Part d'artefact par dataset : bnci2014_001 10 %, lee2019_mi 23 %,
shin2017a **45 %**, zhou2016 61 %. L'effet survit, amputé d'un tiers.

**Étage 2 — le plancher de chance fait plus de dégâts que la régression.** Sujets où
`bd_deep4` apprend vraiment (score moyen > seuil exact sur les essais cumulés) :
**shin2017a 5/29**, lee2019_mi 33/54, les autres quasi intacts. Poolé sur les 70 sujets
restants : INDÉP **−0.0193** [−0.0330, −0.0041] — encore significatif, mais **36 % de la
pente naïve de départ**.

**Sensibilité qui ne passe pas.** Avec le critère strict (chacune des 6 lignes au-dessus du
seuil, n=70 → 33), la pente poolée **change de signe** : INDÉP **+0.0155** [−0.0010,
+0.0316]. Le critère strict sélectionne lui-même sur le bruit et n'est pas le bon test,
mais un reviewer le fera tourner : la conclusion ne survit pas à un changement de
définition de « ce sujet a un signal ».

**Étage 3 — le déficit global DOUBLE quand on retire les sujets à la chance.**
`grow − bd`, tous sujets → sujets où bd a appris : lee2019_mi **−0.0343 → −0.0732**
(n 54→33), shin2017a −0.0175 → −0.0984 (n=5, IC traverse 0), bnci2014_001 −0.0606 →
−0.0686, les autres inchangés. Le « la croissance n'est pas si loin derrière » reposait en
partie sur des sujets où la baseline ne décodait rien.

**Comptes de victoires, axe indépendant.** La croissance gagne sur la moitié basse contre
la moitié haute : bnci2014_001 17 % / 8 %, lee2019_mi 57 % / 19 %, shin2017a 51 % / 29 %,
bnci2015_001 11 % / 17 %, bnci2014_004 33 % / 33 %. **Les deux seuls datasets où la
croissance gagne une majorité des sujets faibles sont les deux où la baseline est à la
chance.**

**Ce qui est publiable.** Pas « la croissance aide les sujets difficiles » (faux : là où la
baseline apprend, la croissance perd encore sur les faibles). Mais « l'écart se resserre
quand le sujet est difficile » — interaction de magnitude, pas inversion de signe,
−0.019 par écart-type après correction des deux confondants. Claim secondaire, jamais
porteuse.

## 11/09 — 582872 (plancher singulier) : 8/8, le plancher mord mais l'accuracy ne suit pas

Le job est parti tout seul dans la nuit (les cartes ampere se sont libérées) — pas de
bascule turing, la classe de carte du contraste est restée homogène. Les 8 cellules ont
tourné entre 16:57 et 17:36, toutes COMPLETED. Intégrité : **8/8 cellules, 18 lignes
chacune, un seul `eegrow_sha` (9a9cc03e), `device=cuda`, sujets 1-9, 2 graines, aucune
manquante.**

**Étage 1 — le mécanisme marche.** Abaisser `MIN_SINGULAR_RATIO` de 0.10 à 1e-6 fait
exactement ce qu'on lui demande : le réseau garde plus de directions candidates et
atteint sa largeur cible bien plus souvent.

| bras | cible atteinte @0.10 | cible atteinte @1e-6 | largeur finale moy. |
|---|---|---|---|
| `grow_shallow` | 153/180 | **177/180** | 38.5 → 39.8 |
| `grow_deep_w32` | 130/180 | **160/180** | 115.4 → 120.9 |

L'axe est donc vivant : ce qui suit mesure un effet réel, pas un no-op. (Rappel :
sur `grow_deep` w2=8, le plancher est inerte par construction — fan-out = 8 candidats
pour 8 slots, il ne coupe jamais. D'où le choix de ces deux bras-là.)

**Étage 2 — l'endpoint ne bouge pas.** Delta = (1e-6) − (0.10), apparié sujet :

| bras | delta | IC95 sujet | MDE(n=9) | 1e-6 gagne |
|---|---|---|---|---|
| `grow_shallow` | −0.0071 | [−0.0163, **+0.0009**] | 0.0135 | 3/9 |
| `grow_deep_w32` | −0.0106 | [−0.0261, **+0.0061**] | 0.0255 | 3/9 |

Les deux sont non significatifs et sous leur MDE : **on ne peut pas déclarer un nul.**
Mais la borne haute de l'IC dit quelque chose de plus utile qu'un nul — tout gain
supérieur à **+0.001** (shallow) et **+0.006** (deep_w32) est exclu. L'hypothèse testée
était un gain substantiel ; c'est exactement ce que ces bornes écartent.

Lecture : le plancher mord sans déplacer l'accuracy, donc **les directions qu'il refusait
n'étaient pas des directions utiles**. L'abstention ne coûte rien. Le prior à 1 sujet
(`grow_shallow.yaml` : 0.690 → 0.552) annonçait au contraire une chute massive — elle
n'existe pas non plus sur 9 sujets ; ce prior ne mesurait que du bruit inter-sujets
(sd ≈ 0.17). Analyse de référence : `benchmarks/analysis/nostop_ratio.py`.

Détail à surveiller, pas un résultat : sur `grow_deep_w32` le `restored_epoch` médian
tombe de 89 à 63.5 à plancher bas. Le réseau grandit plus vite mais apprend
productivement moins longtemps (cf. `restored-epoch-supersedes-params`).

## 10/09 (nuit) — PLAN COMPLET (12/12) : le contraste dépend de l'alignement, et c'est la baseline qui bouge

582871 a rendu ses 5 dernières cellules (COMPLETED, 14:19 à 16:23). Intégrité vérifiée :
**12/12 cellules, 18 lignes chacune, un seul `eegrow_sha` (9a9cc03e), `device=cuda`
partout, sujets 1-9, sessions `0train`/`1test`, aucune manquante.**

| `grow_deep − bd_deep4` | delta | IC95 bootstrap sujet | MDE(n=9) | sujets gagnés |
|---|---|---|---|---|
| `align=none` | **−0.0660** | [−0.0999, −0.0338] | 0.0529 | 1/9 |
| `align=euclidean` | **+0.0146** | [−0.0189, +0.0448] | 0.0506 | 6/9 |

**L'interaction est significative : +0.0806, IC95 [+0.0481, +0.1185].** L'alignement ne
déplace donc pas les deux bras du même montant — il change bel et bien le contraste. Mais
la décomposition par bras dit *lequel* bouge :

- sur `bd_deep4` : **−0.0612**, IC95 [−0.0967, −0.0215] → l'alignement euclidien **dégrade**
  la baseline, significativement (0.4847 → 0.4235) ;
- sur `grow_deep` : **+0.0195**, IC95 [−0.0169, +0.0615], MDE 0.0621 → |effet| < MDE,
  **mesure vide**, pas un nul (0.4187 → 0.4381).

Autrement dit le contraste ne remonte pas parce que la croissance devient bonne, mais
parce que la référence devient mauvaise. Et même sous l'alignement le plus favorable, à
budget plein, le delta vaut **+0.015 et n'est pas significatif** — le +0.133 publié n'est
reproduit sous **aucun** des deux alignements.

Piège à ne pas rejouer : sous `euclidean`, `grow_deep` gagne sur **6/9 sujets** alors que
l'IC contient zéro. Le compte de signes n'est pas une preuve ; c'est l'appariement par
sujet qui tranche (cf. `unit-of-analysis-subject`).

L'alignement resserre aussi les deux bras (sd inter-lignes 0.1524 → 0.1107 sur `bd_deep4`,
0.1293 → 0.1092 sur `grow_deep`) : il homogénéise les sujets en coupant le haut.

Analyse de référence : `/tmp/gvb_full.py` sur Margaret (bootstrap **apparié** par sujet,
20 000 tirages, MDE ≈ 2.9·sd/√n affichée à côté de chaque effet).

**Deuxième bug du garde-fou GPU, corrigé (`11005d3`).** `scontrol requeue "$SLURM_JOB_ID"`
visait trop large : 9 échecs de sonde ont produit **15 événements REQUEUED**, et deux
cellules saines (tâches 3 et 5, `PROBE ok`, 3 min d'entraînement) sont mortes à la seconde
exacte où la tâche 7 se remettait en file — `CANCELLED ... DUE TO JOB REQUEUE` dans leur
propre log. La forme correcte `${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}` était déjà
dans `readout_ablation.sbatch` et `w1_pilot.sbatch`. Pour le job en vol, le déclencheur a
été coupé autrement : `scontrol update jobid=582871_{4,6} ExcNodeList=margpu020` → zéro
requeue ensuite, les 3 dernières cellules ont tourné d'une traite.

**582872 (nostop-ratio) : toujours 0 cellule.** Les 10 cartes ampere sont prises par des
jobs à **11 h à 35 h de walltime restant** (`gblayer`, `jjobard`). Le job est premier en
priorité sur sa classe, mais rien ne se libère avant ce soir. Les turing ont 4-6 cartes
libres. Basculer `--gres=gpu:ampere:1` → `gpu:turing:1` le ferait partir tout de suite et
garderait le contraste homogène (les 8 cellules bougent ensemble) — **décision d'Adam, pas
faite.**

## 10/09 (soir) — align=none complet à 3 graines ; la panne GPU est une carte morte, pas la concurrence

**`grow_deep` ne s'améliore PAS quand on retire l'early stopping.** 582394 a rendu la
graine 3, donc `align=none` est bouclé : 3 graines × 9 sujets, unité d'analyse = le sujet.

| | publié (budget coupé) | budget plein, 3 graines |
|---|---|---|
| `bd_deep4` | 0.275 | **0.4847** |
| `grow_deep` | 0.408 | **0.4187** |
| delta | +0.133 | **−0.0660** |

IC95 bootstrap sujet **[−0.0999, −0.0338]**, `grow_deep` gagne sur **1/9 sujets**,
sd inter-sujet 0.0547 → MDE(n=9, 80 %) 0.0529, franchi. Stable d'une graine à l'autre —
`bd_deep4` 0.4798 / 0.4943 / 0.4800, `grow_deep` 0.4260 / 0.4129 / 0.4172 : ce n'est pas
du bruit de graine. Le budget rend **+0.21 point** à `bd_deep4` et **+0.011** à
`grow_deep`. Le contraste publié mesurait un réseau coupé à l'époque 4, pas la croissance.

`align=euclidean` : 1 seule cellule (`grow_deep` graine 1, 0.4452), bras non apparié,
rien de lisible tant que `bd_deep4 euclidean` manque.

**La panne n'était pas la concurrence — c'est une carte morte.** 582231 en `%6` avait
perdu 12/18, 582394 en `%3` a reperdu 5/8 : baisser la concurrence n'a rien changé, donc
l'explication « trop de tâches pour 4 cartes » était fausse. Sonde 582835 (4 tâches de
10 s sur margpu020, chacune imprimant `SLURM_JOB_GPUS` à côté du verdict) :

```
SLURM_JOB_GPUS=0 -> available True      SLURM_JOB_GPUS=2 -> available True
SLURM_JOB_GPUS=1 -> available False     SLURM_JOB_GPUS=3 -> available True
                    RuntimeError: No CUDA GPUs are available
```

L'index physique **1** de margpu020 (`GPU-b0d82f38-d668-8ea7-a2da-dc32f41245c5`) est
inutilisable par torch alors que `nvidia-smi` le liste normalement — même signature que
margpu021. La cascade s'explique exactement : quand les cartes saines sont prises, la
seule libre est la morte ; la tâche la saisit, meurt en 4 s, **la rend**, la suivante la
reprend. Cinq cellules mortes en 20 secondes sur la même carte.

**Correctif** (`grow_vs_bd.sbatch` et `nostop_ratio.sbatch`) : sur échec du probe, la
tâche **tient la carte 60 s** — pendant cette fenêtre elle est hors du pool et les
cellules sœurs démarrent sur des cartes saines — puis `scontrol requeue`, avec un plafond
de 8 essais au-delà duquel on échoue vraiment. Un garde-fou doit finir par refuser, mais
il ne doit pas recycler le piège.

**582871** : les 5 cellules restantes (`--array=3-7%3`, POINTS=`grow_vs_bd_rest.txt`).
**582872** : nostop-ratio relancé sur **`--partition=tau,gpu`**. 582215 était resté
PENDING 45 min sans lancer une cellule : la partition `gpu` n'a que 4 ampere
(margpu010-011 ; margpu001 est volta donc sous sm_75), toutes allouées, derrière ~8 jobs
gblayer et un job cbarbano de 6 jours. margpu007 et margpu008 portent 3 ampere chacun et
sont dans `tau`. Pool éligible 4 → 10 cartes, **même classe** de carte des deux côtés
(`--gres=gpu:ampere:1`), donc aucune paire du contraste ne traverse ampere/turing.

---

## 10/09 — RÉSULTAT : le gain publié de `grow_deep` change de signe ; 582394 complète le design

**Les 4 cellules acquises de 582231** (`align=none`, graines 1-2, n=9 sujets, sessions et
graines moyennées avant l'appariement) :

| | `bd_deep4` | `grow_deep` | delta |
|---|---|---|---|
| publié (budget coupé) | 0.275 | 0.408 | **+0.133** |
| budget plein | **0.4870** | 0.4194 | **−0.0676** |

IC95 bootstrap au niveau **sujet** : **[−0.1033, −0.0339]**. `grow_deep` ne gagne que sur
**1 sujet sur 9**. Écart-type inter-sujet du delta 0.0568 → MDE(n=9, 80 %) = 0.0549 :
l'effet le dépasse, la cellule n'est pas sous-puissante pour ce contraste. La prédiction
inscrite dans l'en-tête de `grow_vs_bd.sbatch` est vérifiée — `bd_deep4` atteint 0.487
là où le carré budget × sélection annonçait 0.4818, et `grow_deep` reste à 0.42 comme
578293 le mesurait. **Le « gain de la croissance » sur le bras deep était un artefact de
budget, pas un effet.**

Ce qui reste à trancher : est-ce que l'inversion tient **sous alignement euclidien**, ou
est-ce que l'alignement portait le gain publié. C'est l'objet de 582394.

**Panne de 582231 : 12 cellules sur 18 mortes en 4 secondes, exit 90.**

```
NVIDIA GeForce RTX 2080 Ti, 11264 MiB
torch.cuda.is_available() est False
GPU PROBE FAILED -- abandon de la tache
```

`%6` tâches concurrentes sur margpu020, qui déclare `gpu:turing:4` et héberge d'autres
utilisateurs. Les tâches au-delà des cartes libres ont reçu une allocation dont torch ne
peut rien faire, pendant que `nvidia-smi` listait la carte — le mode de défaillance de
`cuda-ordinal-vs-nvidia-smi`, cette fois attrapé. **Aucun nettoyage nécessaire** : le
probe refuse avant l'entraînement, donc aucun CSV ni `fits.jsonl` n'a été écrit par ces
12 tâches. C'est la démonstration que le garde-fou vaut son coût.

**582394 reprend les 8 cellules manquantes** (graine 3 en `none`, tout le bras
`euclidean`) via `benchmarks/slurm/grow_vs_bd_rest.txt`, en **`%3`** pour laisser une
carte au nœud. `grow_vs_bd.sbatch` accepte désormais `POINTS` par l'environnement
(`sbatch --export=ALL,POINTS=...`) plutôt qu'un template dupliqué ; l'indexation des 8
lignes a été simulée avant soumission. Même `results_dir` que les 4 acquises : `stem`
sépare `(bd_deep4, none, 1)` de `(bd_deep4, euclidean, 1)`, aucune collision.

---

## 10/09 — 582231 : `grow_deep` contre `bd_deep4` à budget plein, et fin des bras `fix_*`

**Deux jobs vivants.**

| job | quoi | où | état |
|---|---|---|---|
| `582231` | `grow_deep` vs `bd_deep4`, 3 graines × 2 alignements | tau / **turing** (margpu020-021) | 12 cellules actives |
| `582215` | plancher singulier 0.1 vs 1e-6, `grow_shallow` + `grow_deep_w32` | gpu / **ampere** | 8 cellules, `PENDING (Resources)` |

**Règle posée par Adam : plus aucune comparaison contre `fix_*`.** L'endpoint est
`grow_*` contre `bd_*`. Les 6 cellules `fix_deepeeg` de 582231 (tâches 1, 4, 7, 10, 13,
16) ont été annulées, et le `fits.jsonl` partiel de la tâche 4 — 16 lignes, aucun CSV —
a été supprimé : une cellule annulée pollue le jsonl en append, c'est la faute que
`cancelled-cell-pollutes-fits-jsonl` documente.

**`benchmarks/slurm/grow_vs_bd.txt` n'est PAS édité** tant que l'array tourne. Les tâches
encore en file indexent ce fichier par numéro de ligne ; en retirer les lignes `fix_*`
maintenant décalerait tout ce qui reste et ferait tourner des cellules sur le mauvais
point, sans jamais échouer. Le fichier reste le procès-verbal de ce qui a été soumis ;
c'est ici qu'est enregistré ce qui a été annulé.

**Ce que 582231 vient trancher.** Le workbook publié annonce `grow_deep` +0.133 sur
`bd_deep4` (bnci2014_001 `within_session`), mais `bd_deep4` y score **0.275**, sous le
seuil exact de chance à 4 classes (0.295) : Deep4 y était scoré à l'époque 4. Le carré
budget × sélection donne **0.4818** au même bras une fois entraîné, et 578293 donne
**0.4214** à `grow_deep` à budget plein. Si ça tient, le +0.133 ne rétrécit pas, il
**change de signe**. Effet attendu ~0.06, MDE de la cellule +0.0486 : au-dessus, mais de
peu. Unité d'analyse = le sujet, n=9.

**Conséquence de la règle, à assumer.** Sans `fix_deepeeg`, `grow − bd` ne se décompose
plus en terme de croissance + terme d'architecture (10 484 paramètres contre 284 479,
2 étages contre 4). L'attribution devra venir de la lecture d'architecture déjà faite,
pas d'un bras supplémentaire.

**Pourquoi le seuil singulier n'est pas un axe sur `grow_deep`.** À `w2=8`,
`n_proposed` = fan-out = 8 pour 8 places : sur les 1 620 pas de croissance de 578293 le
plancher ne coupe rien dans 100 % des cas. Deux bras à 0.1 et 1e-6 y seraient identiques.
L'axe vit dans 582215, sur les deux bras où il mord.

**Protocole des deux arrays** : `train.patience=10000 train.selection_monitor=valid_acc`,
assertés avant le `srun` par une résolution `--cfg job`. Vérifié sur les premières
cellules : `epochs: 200`, `restored_epoch` renseigné.

---

## 10/09 — 578293 relu sur l'endpoint MÉCANIQUE : la sélection était morte, w2 la réveille, et ça ne paie toujours pas

**Aucun job lancé.** Le test que j'allais soumettre (`grow_deep` avec un `w2` élargi) est
**exactement 578293**, déjà terminé. Ce qui manquait n'était pas des cellules, c'était de
lire le bon endpoint dans des `fits.jsonl` déjà sur disque — 6 982 événements de croissance.

**Ce que `n_proposed` vaut, vérifié sur la campagne et plus seulement en sonde** : le
nombre de candidats que gromo propose = le **fan-out** de la jonction, donc `w2`. Mesuré
8 / 16 / 32 aux trois niveaux, sans exception.

**La sélection aux trois niveaux** (`kept = floor < proposed` = le plancher singulier
coupe vraiment ; `floor = proposed` = il ne coupe rien ; `kept = cap` = c'est le plafond
`target − width` qui décompte, pas une sélection) :

| niveau | évts | plancher coupe | plafond décompte | **rien n'est coupé** | décades médianes |
|---|---|---|---|---|---|
| `w2=8` (campagne) | 1 620 | 0.0 % | 0.0 % | **100.0 %** | 0.42 |
| `w2=16` | 1 667 | 3.2 % | 1.0 % | **95.8 %** | 0.68 |
| `w2=32` | 3 695 | **71.0 %** | 12.4 % | 16.6 % | 1.89 |

Hors des pas où le plafond mord, `floor/proposed` médian vaut 1.000 à `w2=8` (100 % des
pas), 1.000 à `w2=16` (98.6 %), et **0.906 à `w2=32`** (29 % seulement à 1.000). Donc le
réveil à `w2=32` est bien le plancher singulier et **pas** le plafond qui décompte vers la
cible — c'était l'artefact à écarter, il est écarté.

**Conclusion, et elle est plus forte que ce que je visais.** Sur la campagne `results_final`
le bras `grow_deep` n'a jamais sélectionné : rang saturé, 8 candidats pour 8 places, le
spectre est plat par construction. Mais 578293 a déjà fait tourner le niveau où la
sélection fonctionne, et son endpoint accuracy ne bouge pas — `grow − fix`, famille de 6,
**rien ne survit à Holm** (meilleur : eucl-w16 +0.0138, p brut 0.024, p Holm 0.145). Donc
ce n'est pas « on ne sait pas parce que la sélection était morte » : **à `w2=32` gromo
sélectionne réellement (71 % des pas, ~2 décades d'étalement) et n'achète rien de plus que
le jumeau gelé.** C'est un négatif mécaniquement interprétable, pas une mesure vide.

**Ce qui reste à ne pas confondre.** Le nul `grow − fix` de `results_final` sur le bras
deep (+0.003 pour un MDE de ±0.008) reste sous-puissant et ne doit pas être écrit comme un
nul. Ce que le tableau ci-dessus établit ne passe pas par le score : c'est mesuré à la
source, sur les propositions de gromo elles-mêmes.

**Corollaire pour le pilote goulot lui-même** : ses trois niveaux tiennent `w2_in = w2`,
donc au premier pas de chaque niveau `n_proposed == largeur courante` — la saturation de
rang est reconduite à chaque niveau, et la sélection ne se réveille qu'une fois la largeur
partie bien au-delà de `w2`. C'est pour ça que le réveil n'apparaît qu'à `w2=32`, où la
croissance va jusqu'à 128.

## 10/09 13h00 UTC — 581692 : ablation de readout, SOUMISE

**18 cellules** (`results_readout/`), `eegrow_sha=650b9ea79114`, branche `exp/deep-bottleneck`.
{`grow_shallow`, `grow_shallow_nolog`, `grow_shallow_generic`} × 3 seeds × 2 aligns, triées
par graine — les **6 premières cellules sont le plan complet** à la graine 0.

**La question.** D'où vient l'écart de **+0.108** entre `grow_shallow` et `grow_deep` ? Le
pilote w1 en a tué deux explications sur trois : la **capacité** (w1 8→32 ne vaut que
+0.0235, 4× trop peu) et la **régularisation** (dropout+GroupNorm *dégrade*). Reste le
**biais inductif** :

| | queue |
|---|---|
| shallow | `BatchNorm → Square → AvgPool → SafeLog` |
| deep | `BatchNorm → ELU → MaxPool` |

`Square → AvgPool` = moyenne des carrés = **la variance** du signal filtré spatialement ;
avec le log, c'est une log-puissance de bande. L'imagerie motrice est de l'ERD/ERS : la
quantité discriminante est la **puissance** mu/beta, une fonctionnelle **quadratique**.
ShallowFBCSPNet est donc du FBCSP écrit en couches — la statistique à estimer est *dans*
l'architecture. Une queue ReLU doit approximer x² par morceaux, ce qui se paie en unités
et en essais ; le bras deep a 8/16 filtres et 230 essais par sujet.

**Corollaire pour le papier.** La jonction growable de deep (`conv2b`) vit en **aval** du
collapse spatial à w1=8. Ce que ce collapse détruit ne se récupère pas en aval :
**l'axe croissable n'est pas l'axe limitant**. Ce n'est pas un fait sur la croissance,
c'est un fait sur l'endroit où la jonction a été posée — `growing_shallow` fait croître
`conv_spat`, c'est-à-dire le goulot lui-même, et sa croissance paie.

**Pourquoi ablater shallow et non augmenter deep.** deep diffère de shallow sur *quatre*
axes à la fois (profondeur, largeur, position de jonction, readout) — lui ajouter le
readout ne trancherait rien. On retire donc **un** facteur au bras qu'on sait bien mesurer,
à géométrie de pooling tenue fixe : les trois variantes ont **18372 paramètres et
`final_conv_length=69` à l'identique** (vérifié au smoke test et re-vérifié par le garde-fou
de chaque tâche). Ablation de *fonction*, pas changement de capacité déguisé.

**Facteur décomposé d'avance**, pour ne pas refaire le pari du facteur groupé `reg` :

| bras | readout | ce qu'il retire |
|---|---|---|
| `grow_shallow` | `Square → AvgPool → SafeLog` | rien (référence) |
| `grow_shallow_nolog` | `Square → AvgPool → Identity` | le log seul |
| `grow_shallow_generic` | `ELU → MaxPool → Identity` | tout le readout |

Le log porte un effet propre : la puissance de bande est à queue lourde, et le log
transforme l'ERD — modulation **multiplicative** — en décalage **additif**, lisible par un
classifieur linéaire. Sans le barreau intermédiaire, un effondrement de `generic` ne dirait
pas si c'est le carré ou la compression qui porte l'effet.

**Prédiction préenregistrée** : `power > nolog > generic ≈ grow_deep` (~0.44). **Si
`generic` tient à ~0.54, l'explication du readout est fausse** et l'écart est ailleurs —
ce sera à écrire tel quel.

**Appariement.** `grow_shallow` est **relancé** et non repris : ses chiffres v5 viennent
d'une autre campagne, sous un protocole différent (patience, selection_monitor) — apparier
deux campagnes est l'erreur que `xds-august-grid-obsolete` documente. `grow_deep` n'est
**pas** relancé : il est déjà dans `results_w1pilot` sous ce protocole, ces graines, ces
alignements, donc appariable tel quel. D'où aussi le même modèle de carte épinglé
(**2080 Ti**) et `dataset.resample=250`.

**Ce que ça ne teste pas** : toujours pas l'endpoint `grow − fix` (MDE ~+0.05 à n=9 contre
un effet visé de +0.0123). On teste l'accuracy **absolue**.

---

## 10/09 11h20 UTC — 580878 : pilote w1 × régularisation, EN COURS

> **Point 12h20 UTC.** 38 terminées, 12 en vol, 20 en file, 2 échecs (`_3`, `_8`, rattrapés
> par 580899/580903). Coût mesuré **15,5 min/cellule** (13:37–17:48 sur 38). Fin d'array
> attendue **~13 h 00 UTC**. Graine 0 complète (24 cellules = le plan factoriel entier),
> **lue** — voir la section « lecture graine 0 » plus bas.
>
> **Deux SHA dans les CSV**, `6c1ad3a614e6` et `53f24a118f37` : `git diff` entre les deux ne
> touche **que** `benchmarks/slurm/w1_pilot.sbatch` (28 lignes, la remise en file sur carte
> fantôme). Zéro ligne de modèle, d'entraînement ou de données — les cellules restent
> appariables. Vérifié, pas supposé.
>
> **Point 13h00 UTC — un TROISIÈME SHA est attendu, et c'est de mon fait.** Le déploiement
> de l'ablation de readout (`650b9ea79114`) a réécrit l'arbre cluster alors que **6 cellules
> w1 tournaient encore**. Celles qui ont démarré après portent donc `650b9ea79114`.
> `git diff 53f24a1 650b9ea` : `growing_shallow.py` + 4 fichiers **nouveaux** (2 yaml, 1
> sbatch, 1 txt). **Zéro ligne** de `growing_deepeeg.py`, de `loop.py` ou du chemin données —
> le bras deep n'a rien vu passer, les 72 cellules restent appariables. Dit ici plutôt que
> découvert à l'analyse.

**72 cellules** (`results_w1pilot/`), `eegrow_sha=6c1ad3a614e6`, branche `exp/deep-bottleneck`.
Plan factoriel : {`grow_deep`, `fix_deepeeg`} × {reg off, reg on} × {w1 8, 16, 32} × 3 seeds × 2 aligns.

**Pourquoi ce pilote.** 578293 a fait varier w2 (l'aval) et n'a rien trouvé. La mesure
sur ses 3 240 fits dit pourquoi : train loss 0.001–0.004 sur 230 essais, meilleure valid
loss à 0.1–0.2 nats sous ln 4, 18–28 % des fits jamais sous le prior uniforme, optimum de
validation époque 11–20 sur 200. **Le bras n'est pas limité en capacité, il est limité en
régularisation** — donc la croissance, qui ajoute de la capacité, ne peut rien payer.

Les deux facteurs jamais bougés jusqu'ici :
- **w1** — `conv_time 1→w1`, `conv_spat w1→w1` sur 22 électrodes, tenu à 8 partout contre
  25/25 pour Deep4Net. La jonction growable vit **en aval** de ce collapse spatial ; celle
  de `growing_shallow` est sur le signal brut (axe FBCSP). C'est l'hypothèse de mécanisme.
- **reg** — `drop_prob=0.5` + `norm_junction`. `growing_deepeeg.py` était la **seule** archi
  growing sans aucune régularisation (shallow/sccnet/eegnex à 0.5, Deep4Net à 3 Dropout).
  Facteur **groupé** : s'il gagne, un second passage devra le décomposer.

**Défauts OFF** (`drop_prob=0.0`, `norm_junction=False`) → sortie bit-identique à l'avant,
vérifiée localement : aucun résultat publié n'est re-daté.

**Cartes.** Recompte au moment de soumettre : **0 ampere, 0 hopper, 4 turing libres** sur
tout le cluster. Les 6 A100 de margpu007/008 sont tenues par des jobs `tau` (jjobard, ≤21 h
restantes) — même PriorityTier=100, rien à préempter. Première soumission **580874**
(ampere) annulée sans avoir écrit une ligne. Le pilote tourne donc sur **turing RTX 2080 Ti**,
classe épinglée *au modèle* : la sonde refuse toute autre carte, parce que les contrastes
sont appariés fit par fit. Les six niveaux étant relancés ici (w1=8/reg=off compris), aucune
paire ne traverse de frontière de matériel.

**Carte fantôme, de retour.** margpu021 annonce `gpu:turing:3` ; deux cartes répondent, la
troisième dit « No devices were found ». Sondée saine 20 min plus tôt sur un *autre* de ses
créneaux — le défaut est par carte, pas par nœud. La sonde `mem_get_info` l'a attrapée en
5 s (exit 90, tâches 3 et 8, aucun CSV). `ExcNodeList=margpu021` posé sur les 58 tâches en
attente, tâches 3 et 8 resoumises en **580899**. Sans cette sonde, le créneau fantôme aurait
consommé une tâche toutes les 5 s et fait « finir » l'array en quelques minutes.

**Ce que le pilote peut conclure** : l'accuracy **absolue** (référence : l'écart d'archi
`grow_shallow − grow_deep`, +0.108). **Pas** l'endpoint `grow − fix` — MDE +0.0486 à n=9
contre un effet visé de +0.0123. Si `reg` bouge l'accuracy, il faudra **relancer `grow − fix`
par-dessus**, une fois le bras sorti du régime de mémorisation : c'est la seule condition où
la question de la croissance a un sens.

Grille triée par **graine** : les 24 premières cellules sont le plan complet à la graine 0.

### Lecture graine 0 (24/24 cellules, n=9 sujets, 1 graine) — `w1_analysis.py`

**Accuracy absolue** (sessions moyennées, appariée par sujet) :

| bras | align | reg | w1=8 | w1=16 | w1=32 |
|---|---|---|---|---|---|
| grow_deep | euclidean | off | 0.4385 | 0.4562 | 0.4408 |
| grow_deep | euclidean | **ON** | 0.4153 | 0.4328 | 0.4643 |
| fix_deepeeg | euclidean | off | 0.4330 | 0.4605 | 0.4565 |
| fix_deepeeg | euclidean | **ON** | 0.3944 | 0.4381 | 0.4669 |
| grow_deep | none | off | 0.4292 | 0.4311 | 0.4444 |
| grow_deep | none | **ON** | 0.4089 | 0.4111 | 0.4186 |
| fix_deepeeg | none | off | 0.4161 | 0.4387 | 0.4416 |
| fix_deepeeg | none | **ON** | 0.4113 | 0.4215 | 0.4356 |

**Facteur reg — négatif.** 10 des 12 contrastes appariés `reg − noreg` sont négatifs
(moyenne −0.014). Trois IC95 excluent zéro, **les trois du côté négatif** (−0.023, −0.026,
−0.039). Aucun contraste positif n'est significatif. La régularisation ne sort pas le bras
du régime de mémorisation : elle le dégrade, surtout à w1 étroit.

**Facteur w1 — positif mais petit.** 6 contrastes sur 16 excluent zéro, tous positifs. Le
plus gros est `fix_deepeeg / eucl / reg : w32 − w8 = +0.0726 [+0.034, +0.106]` — mais son
point de départ (0.3944) est **la pire cellule de la table** : c'est le trou creusé par reg
qu'on rebouche, pas du terrain gagné. À reg off, le même contraste ne vaut que +0.0235.

**Endpoint.** Meilleure cellule toutes conditions confondues : **0.4669** ; meilleure sans
reg : 0.4605. Écart +0.006 — rien. Ni la largeur w1 ni la régularisation ne rattrapent
l'écart d'architecture de **+0.108**. La condition préenregistrée (« si `reg` bouge
l'accuracy, relancer `grow − fix` par-dessus ») **n'est pas déclenchée**.

**`grow − fix`** : 12 contrastes, moyenne −0.002, un seul IC exclut zéro (−0.017). MDE
0.018–0.054 contre un effet visé de +0.0123 → **sous-puissant, à ne pas lire comme un nul**.

À refaire sur les 3 graines quand l'array est fini : moyenner les graines réduit la sd et
donc le MDE ; n reste 9 (le sujet est l'unité).

## 10/09 10h30 UTC — 578293 : pilote goulot TERMINÉ, issue (b)

36/36 cellules, ~2 h de mur (cellule ≈ 3 à 8 min selon le niveau, mesurée en direct).
**Intégrité** : 648 lignes = 36 × 18, un seul `eegrow_sha=c34750c0fadc`, 9 sujets × 2
sessions × 3 seeds partout, aucune cellule fantôme.

**Endpoint (accuracy absolue, appariée par sujet, seeds et sessions moyennées, n=9)** :

| bras | align | w2=8 | w2=16 | w2=32 | amplitude |
|---|---|---|---|---|---|
| grow_deep | none | 0.4214 | 0.4193 | 0.4309 | 0.0116 |
| fix_deepeeg | none | 0.4216 | 0.4181 | 0.4196 | 0.0035 |
| grow_deep | euclidean | 0.4349 | 0.4525 | 0.4579 | 0.0230 |
| fix_deepeeg | euclidean | 0.4266 | 0.4388 | 0.4399 | 0.0133 |

Élargir w2 de 8 à 32 (×4 en largeur, ×3.3 en paramètres) déplace l'accuracy de
**+0.023 au mieux**, contre l'ordre de grandeur de référence **+0.108** inscrit dans le
sbatch. Là où ça bouge (euclidean), **les deux bras montent** : fix +0.013, grow +0.023.
→ **issue (b)** de la pré-inscription : le goulot w2=8 n'est pas la contrainte. Le nul
sur n=324 est le résultat, et devient un mécanisme à écrire.

**grow − fix, famille de 6, Holm** : rien ne survit. Le plus fort est
`euclidean w16` +0.0138, p brut 0.024, **p Holm 0.145**. Ordre : eucl-w16 (.024),
eucl-w32 (.080), none-w32 (.159), eucl-w8 (.227), none-w16 (.790), none-w8 (.973).

**Note de puissance, pour l'étage 2.** Le sd apparié observé ici est de 0.013 à 0.027,
pas les 0.052 pré-enregistrés — parce qu'on moyenne 3 seeds × 2 sessions avant de
tester (Spearman-Brown, cf. `icc-single-vs-mean`). MDE réalisé ≈ 0.014 à 0.029. C'est
une information de **planification** pour dimensionner l'étage 2 ; recalculer le MDE sur
le sd des mêmes données ne rétro-autorise pas à lire l'endpoint grow − fix sur n=9.

## 10/09 08h36 — 578293 : pilote goulot, RESOUMIS (578248 mort en 60 s)

**578248 : 36/36 cellules FAILED**, toutes sur
`ModuleNotFoundError: No module named 'eegrow.alignment'`, en moins d'une minute.
Zéro CSV produit.

**Cause.** L'env conda `bench` porte `__editable__.eegrow-0.1.0.pth` qui pointe en dur
sur `/scratch/amounir/eegrow/src` — un autre checkout, sans `alignment.py`. Mon sbatch
n'exportait que `PYTHONPATH=$ROOT/benchmarks`, donc `import eegrow` résolvait vers
l'autre tree. `pack_run.sh:82-84` exporte déjà `$ROOT/src:$ROOT/benchmarks` pour
exactement cette raison ; ce pilote ne passe pas par `pack_run.sh` et n'en héritait pas.
**L'échec bruyant est un coup de chance** : un tree d'août complet aurait rendu des CSV
de la bonne forme avec le bug s=0 de retour.

**Corrigé** en deux commits : `PYTHONPATH=$ROOT/src:$ROOT/benchmarks` + recopie du
garde-fou de provenance de `pack_run.sh:316-353` (chemin sous ROOT, abstention s=0
présente, tree capable de nommer son commit). Le garde-fou a immédiatement attrapé un
second trou du même genre : le sbatch n'exportait pas non plus `EEGROW_SHA` depuis
`$ROOT/.eegrow_sha`, donc **chaque ligne aurait porté `eegrow_sha=None`**.

Vérifié à la main sur le cluster avant resoumission :
`eegrow=/scratch/amounir/eegrow_budget/src/eegrow/__init__.py`, `sha=c34750c0fadc`.

**578293 soumis à 08h36**, déployé @ `c34750c`. 4 tâches R, guard ok dans le log de la
tâche 0. Coût par cellule **non encore mesuré** — bnci2014_001 n'était pas dans
`/scratch/amounir/mne_data` et la première vague le télécharge. ETA à donner une fois
la première cellule tombée, pas avant.

Design, endpoint et limites : voir la section 578248 ci-dessous, inchangés.

## 10/09 08h25 — 578248 : pilote goulot sur les deux bras deep (MORT, voir ci-dessus)

**36 cellules, bnci2014_001 within_session, partition `gpu`, A100 (margpu010/011),
array bridée à %4.** Branche `exp/deep-bottleneck`, déployée @ 3b10e09. Résultats dans
`benchmarks/results_deepbottleneck/` — répertoire dédié parce que le niveau w2=8
réutilise les labels `grow_deep` / `fix_deepeeg` et écraserait la campagne.

**Pourquoi.** `grow_deep − fix_deepeeg` = **+0.0034 [−0.0022, +0.0091]** sur n=324
sujets (within_session, align=none), contre **+0.0123 [+0.0075, +0.0172]** pour la
paire shallow au même n, MDE 0.0081. Nul borné, pas un manque de puissance. Ce qui
grossit dans `GrowingDeepEEGNet` est la fan-in de conv2b ; sa sortie reste `w2`, et
`w2=8` dans la campagne. La jonction grossit 8 → 32 puis se déverse dans un goulot de
8 canaux, qui est aussi tout ce que le classifieur voit.

**Design.** w2 ∈ {8, 16, 32}, facteur de croissance tenu à 4× (target = 4·w2), et le
jumeau gelé bouge à chaque niveau (`fix_deepeeg_wNN` suit `grow_deep_wNN`) — sinon la
paire diffère par la croissance ET la largeur. Le niveau w2=8 est relancé plutôt que
repris de `perf_final/` : il avait tourné sur turing, le pilote tourne sur ampere, et
aucune paire ne doit traverser une classe de carte.

**Ce que ça peut conclure.** Pas l'endpoint : MDE **+0.0486** à n=9 contre un effet
visé de +0.0123. Le pilote teste l'étage en amont — est-ce que l'accuracy **absolue**
bouge avec w2 (ordre de grandeur attendu si le goulot est réel : +0.108, l'écart
d'architecture `grow_shallow − grow_deep`). Si oui → étage 2 sur cho2017 +
physionetmi (n=161, MDE +0.0115) pour l'endpoint. Si non → le nul sur n=324 **est** le
résultat, et devient un mécanisme : la croissance paie quand la jonction growable est
le goulot de capacité (shallow), pas sinon (deep, sccnet).

**Attention en lisant les figures existantes.** `grow_deep` vs `bd_deep4` compare deux
architectures, pas la croissance — `GrowingDeepEEGNet` fait 2 étages, Deep4Net en fait
4 (25/50/100/200). Le code le dit à `growing_deepeeg.py:71` et `final_grid.py:176`. Le
contrôle est `fix_deepeeg`.

**Encore en attente de décision.** `543697_0` (motif_dxr) est toujours PD en
`DependencyNeverSatisfied` — job fantôme, jamais annulé.

## 06/09 15h50 — yang2025 : gate rouge sur UN sujet, réparé

Le patch MOABB tient : 51 sujets chauffés en 36,9 min, ~45 s chacun (contre 1316 s
pour sub-1 avant patch, où l'essentiel était la réextraction des 65,6 Go). Le cache
pèse 20 Go et compte 150 entrées valides.

Mais `stream_rc=1 gate_rc=1` :

    yang2025   OK=150   MANQUE=1   MENSONGE=0
        MANQUE: sub-40 (aucune session)
    [40/51] sub-40  16.7s  ÉCHEC ValueError: Invalid Annotation description(s)
                                  {'1', '2'} are not present in data

**Cause.** `sub-040/ses-03/eeg/data.bdf` faisait 14680064 octets, soit **14 MiB
pile**, quand la médiane des 186 fichiers est 438 Mo. Une taille ronde au bit près
n'est pas une fin de fichier légitime : c'est une écriture coupée sur frontière de
buffer. mtime 14h48 = l'instant où on a annulé 542617/542618, dont l'`extractall`
écrivait ce fichier. MNE déduit alors la durée de la taille, les 205 annotations
tombent hors plage, il ne reste aucun événement, et l'epoching lève.

**Ce n'était donc pas la source.** Le zip de 65,6 Go était intact ; `repair_yang_sub40.py`
(job 543489, 6 s) a réextrait ce seul membre : 14,7 Mo → **421,6 Mo**. Le script
extrait dans un `.part` et ne remplace qu'après contrôle de taille, pour qu'une
seconde interruption ne puisse pas recréer un tronqué en place.

**Ce que la porte a bien fait.** Elle a refusé au lieu d'avertir, et le
`--purge-raw-at-end` s'est auto-annulé (« PURGE FINALE REFUSÉE : des sujets ont
échoué, le brut reste »). Sans ça le zip partait, et réparer sub-40 aurait coûté un
retéléchargement de 65,6 Go. Cf. `gates-must-refuse-not-warn`.

**Chaîne relancée** : 543489 (réparation, COMPLETED) → `afterok` → **543544** (chauffe,
qui saute les 50 sujets en cache) → `aftercorr` → **543545** (D×R). 543231 annulé, sa
dépendance n'était plus satisfiable.

| job | dataset | état 15h50 | ETA UTC |
|---|---|---|---|
| 543544_1 | yang2025 chauffe | R, 50/51 déjà en cache | ~16h00 |
| 543545_1 | yang2025 D×R | PD (Dependency) | — |
| 541962_0 | stieger2021 chauffe | R, 27/62 | ~21h25 |
| 541963_0 | stieger2021 D×R | PD (Dependency) | — |

**Correction d'ETA sur stieger.** Les sujets ≤23 avaient 7 fichiers bruts (~490 s) ;
à partir de sub-24 ils en ont 11 (~610 s). Fin de chauffe repoussée de 20h15 à
~21h25 UTC. Quota 663 G, pente ~33 G/h nette du purge incrémental → atterrissage
prévu vers 860 G sur 1024.

`COST` propres : 9 datasets (brandl2020 à retenir à **133,0 s**, pas les 197,1 s de la
ligne contendue). zhou2020 : `AUCUN CSV`, toujours en attente de l'arbitrage d'Adam
(géométrie construite sur le premier sujet, montages 41 et 26 mélangés).

## 06/09 11h20 — QUOTA DÉBLOQUÉ (1024 → 753 G), CAMPAGNE RELANCÉE

Adam AFK, relance autonome. **271 G libres.**

### Ce qui a été rendu

| quoi | volume | pourquoi c'était sûr |
|---|---|---|
| `brainbert/btb` | 138 G | décompression de `btb_raw`, se refait par `unzip`, **sans re-télécharger** les 130 Go |
| brut zhou2020, chang2025, brandl2020, dreyer2023b | 133 G | `check_cache` vert AVANT suppression (job 541957 : `0 MANQUE, 0 MENSONGE`) |

`btb_raw/` (137 G), les poids `stft_large_pretrained.pth`, `popt_pipeline.py`, les
sbatch PopT et `results/results.csv` sont **intacts**. Arbitrage d'Adam : `btb/`
seul, pas `btb_raw/`.

### Chaîne relancée

| dataset | warm | D×R | note |
|---|---|---|---|
| forenzo2023 | 541959 | 541960 | 44/47, il ne manque que sub-23/24/25 ; brut déjà sur disque |
| — | 541961 | — | `afterok:541959` → rend les **158 G** de brut forenzo, le plus gros poste de l'arbre |
| stieger2021 | 541962 | 541963 | reprend à 120/184, suppression au fil de l'eau |
| yang2025 | 541964 | 541965 | `afterok:541961` : ne démarre qu'avec ~430 G devant lui |
| brandl2020 | — | 541966 | mesure de coût **refaite au propre** |

### Pourquoi brandl2020 est relancé alors que son cache est vert

Sa ligne `COST` était fausse. `EEGROW_CUDA_FRACTION=0.30` × 3 shards = plafond de
3,17 GiB sur une carte de 10,58 : 4 fits sur 48 sont morts en
`torch.OutOfMemoryError`, et les 197 s de médiane sont de la **contention**, pas un
coût — le log montre les fits de fin à 74 s et 118 s, dans la fourchette des autres
datasets. Relancé en `N_SHARDS=1 EEGROW_CUDA_FRACTION=0.9`, sur un répertoire neuf
(l'ancien est dans `dxr_motif/brandl2020_contended/`) pour que la médiane ne mélange
pas fits contendus et fits propres. ~48 fits × ~90 s ≈ 72 min en série.

**Mesuré à 11h46, shard unique** : fits à **124–137 s**, très réguliers, aucun OOM.
Le facteur de contention est donc **1,5** (197 s → ~130 s), et non « 3 à 10 » comme
écrit ici plus tôt. L'erreur était de supposer que brandl devait ressembler aux
20–57 s des autres datasets : avec 63 électrodes, 1125 points et une croissance qui
sature à 107 442 paramètres, **brandl2020 est intrinsèquement le dataset cher de la
campagne**. Dimensionner `motif_probe` sur ~130 s/fit pour lui, pas sur 40.
Cf. [[cell-cost-must-be-measured-live]] : la seule mesure valable est prise sur le
dataset lui-même, et une transposition depuis les autres se serait trompée d'un
facteur 3 dans le sens du sous-dimensionnement.

### 12h45 — forenzo OK, yang2025 remort sur une limite qui n'est PAS le quota d'espace

`COST forenzo2023 n_donneurs=25 fits=75 fit_s_median=37.7 fit_s_p90=39.5` — rc=0,
75 CSV, D×R bouclé en 17,9 min. **8 lignes `COST` propres sur 11.**

yang2025 a re-échoué à 11h49 (`stream_rc=3 gate_rc=1`), et cette fois le diagnostic
du 06/09 matin ne s'applique pas :

```
=== yang2025  51 sujets  quota_libre=420G
[1/51] sub-1   489.1s ÉCHEC OSError: [Errno 122] Disk quota exceeded
```

**420 G libres au démarrage, 419 G encore libres 3 min avant l'échec.** Écarté par
la mesure : quota espace user (604 G/1024), quota groupe `tau` (aucun), inodes (2 %
de 5,3 G), filesystem (4,6 To libres), home (46 G libres), et `TMPDIR=/tmp` qui est
un **tmpfs de 252 G avec 248 G libres** sur marg003.

Deux hypothèses déjà réfutées à la mesure (extraction sous `$HOME`, puis `/tmp`
saturé). **Ne pas en formuler une troisième sans la trace** — job **542077**
(`diag_yang.py`) rejoue sub-1 avec `traceback.print_exc()`, la chaîne
`__cause__/__context__` avec `errno` et `filename`, et un échantillonnage `quota -s`
toutes les 30 s. Cf. [[cell-cost-must-be-measured-live]].

Le brut yang2025 (104 G : l'archive de 65,6 G + son extraction) est **intact** —
`stream_rc=3` est l'abandon propre, il ne détruit rien.

#### 13h22 — répondu par la mesure : yang2025 se décompresse en place, sans fin visible

```
[du 13:16:35] 122G  mne_data/MNE-yang2025-data     [quota] 615G
[du 13:22:18] 148G  mne_data/MNE-yang2025-data     [quota] 642G
              4.0K  moabb_cache/MNE-BIDS-yang2025
               32K  /tmp
```

**~4,5 G/min dans `mne_data`.** Ni le home, ni le tmpfs, ni un quota caché : le
`quota_libre=420G` du démarrage est un instantané pris **avant** que l'expansion ne
commence. yang2025 demande beaucoup plus que ses 104 G apparents, et c'est pour ça
que le job va plus loin quand il y a plus de place devant lui (8 min à 11h40,
>18 min à 13h22) — le symptôme suit l'espace disponible, pas une limite fixe.

**Conséquence de conception : la suppression au fil de l'eau ne peut rien pour
yang2025.** `warm_stream.py` refuse (à raison) d'effacer un fichier réclamé par 51
sujets, donc l'archive de 65,6 G **et** son extension coexistent forcément. C'est le
seul dataset des 11 où le pic n'est pas bornable par le protocole.

Garde-fou posé pendant la mesure : `scancel 542176` si le quota atteint 940 G/1024,
pour ne pas emporter stieger2021 (541962) après 2 h de travail. Jamais déclenché.

#### 13h50 — RÉSOLU : l'expansion plafonne à 152 G, sub-1 passe

```
[du 13:22:18] 148G → [du 13:23:19] 152G → [du 13:23:50] 152G
OK
```

**Coût d'extraction unique, désormais payé.** yang2025 pèse 152 G de brut (archive
65,6 G + extension ~86 G) et n'augmente plus ; les 50 sujets restants n'ont plus
qu'à lire. Les deux échecs `EDQUOT` étaient la même chose vue deux fois : une
expansion en cours contre un quota déjà entamé, pas une limite exotique.

Relancé en **542617** (warm) → **542618** (D×R), avec `EXTRA=--purge-raw-at-end` :
c'est le seul dataset dont le brut ne peut pas être rendu au fil de l'eau, donc il
est rendu à la fin, sous réserve de la porte. 541965 (never-satisfied) annulé.

#### stieger2021, cadence mesurée

17/62 à 13h48, 670–785 s par sujet, 97 G rendus au fil de l'eau, `quota_libre`
stable à ~383 G. 45 sujets restants × ~730 s ≈ **9 h** → fin vers 22h50 UTC, dans
son mur de 2 jours.

#### 14h50 — LA VRAIE CAUSE : bug MOABB, l'archive est réextraite à CHAQUE sujet

Mesuré sur le process vivant (marg001, pid 2295686, `/proc/<pid>/io`) :

```
fichier ouvert : WBCIC_SHU_Motor_Imagery_dataset.zip
read_bytes  : 168 Go        write_bytes : 251 Go        en 57 min
wchan       : folio_wait_bit_common
```

251 Go écrits pour une archive de 65,6 Go, quota inchangé de 5 G : il décompresse,
écrase, recommence. `moabb/datasets/yang2025.py:347` :

```python
subj_str = f"sub-{subject:03d}"
existing = list(basepath.rglob(f"*{subj_str}*data.bdf"))   # ne matche JAMAIS
...
if zip_path.exists() and not existing:
    zf.extractall(str(basepath))                            # donc a chaque sujet
```

`rglob` sans séparateur teste le **nom du fichier**. Les 186 fichiers s'appellent
tous exactement `data.bdf` ; le sujet est un composant de répertoire
(`.../sub-001/ses-01/eeg/data.bdf`). `existing` est donc vide par construction.
Extrapolé : **~3,3 To écrits** sur un NFS partagé déjà à 91 % pour un seul dataset.

**Corrigé** dans l'env (`.bak` à côté), remplacé par un test sur les composants de
chemin : `[p for p in basepath.rglob("data.bdf") if subj_str in p.parts]`. Vérifié
en job 543227 : `0.19s` par sujet au lieu de 15-22 min, **0,0 Mo écrit**.

**À remonter en amont** — c'est un vrai bug MOABB, pas une particularité locale, et
n'importe qui utilisant Yang2025 écrit 3,3 To sans le savoir. Rien n'a été ouvert :
aucune action publique sans l'accord d'Adam. Cf. [[ask-before-public-pr-actions]].

Relancé en **543230** (warm) → **543231** (D×R).

#### brandl2020 : mesure propre obtenue

```
COST brandl2020 n_donneurs=16 fits=48 fit_s_median=133.0 fit_s_p90=141.7 device=cuda
=== brandl2020 DONE rc=0 csv=48        (shard done in 118.0 min)
```

48 fits sur 48, aucun OOM. Contendu 197,1 → propre 133,0 : **facteur 1,48**.
**8 lignes `COST` propres sur 11** (chang2025, dreyer2023b, guttmannflury2025_me,
guttmannflury2025_mi, hefmiich2025, kumar2024, forenzo2023, brandl2020).

### Reste à arbitrer par Adam

**zhou2020 est cassé, et ce n'est pas un problème de données** (son cache est vert,
137 entrées). `donor_receiver.py:206` dérive `n_chans` du **premier sujet** et
réutilise cette géométrie pour toutes les paires donneur/receveur :

```
RuntimeError: Calculated padded input size per channel: (1226 x 26).
Kernel size: (1 x 41). Kernel size can't be greater than actual input size
```

41 électrodes attendues, 26 présentes. Les dix autres datasets ont un montage
constant, zhou2020 non. Deux issues, qui ne disent pas la même chose du résultat :
restreindre au sous-ensemble commun d'électrodes, ou sortir zhou2020 de la campagne.
Transférer un donneur vers un receveur d'un autre montage n'a pas de sens sans
alignement — c'est un choix scientifique, pas une correction de bug. **Non relancé.**

### Leçon de venue

Le premier `du -sh /scratch/amounir/*` est parti **sur la frontale** : 623 k inodes
sur un NFS partagé, tué en cours. Les scans et les `rm` de cette session sont
repassés en jobs `normal` (541954-541958). `squeue`/`quota`/`grep` de logs restent
sans conséquence sur la frontale ; un parcours récursif, non.

---

## 06/09 — LA CAMPAGNE EST ARRÊTÉE : quota /scratch de 1 To saturé

**Ce n'était pas le débit, c'était le quota.** `quota -s` : 1024 G utilisés sur
1024 G. Le `df` montrait 4,2 To libres sur le filesystem — trompeur, la contrainte
est **par utilisateur**. Les chauffes sont mortes sur `OSError: [Errno 122] Disk
quota exceeded`.

**Bilan des 11 datasets** : 4 D×R complets (hefmiich2025, guttmannflury2025_mi,
guttmannflury2025_me, kumar2024). 7 tâches D×R en `DependencyNeverSatisfied`,
c'est-à-dire **jamais lancées** : stieger2021, yang2025, chang2025, forenzo2023,
zhou2020, dreyer2023b, brandl2020.

**La porte a fait son travail.** `warm_rc=0 gate_rc=1` sur les 7 : le script de
chauffe avalait les `OSError` par sujet et se croyait sorti propre ; c'est
`check_cache` qui a refusé, et `aftercorr` n'a donc jamais relâché le D×R. Aucun
résultat calculé sur un cache troué. Cf. [[gates-must-refuse-not-warn]].

**La campagne n'a jamais tenu dans le quota, et ce n'est pas 539433 qui l'a causé.**
Sans stieger2021 : mne_data 397 G + brainbert 281 G + popt 30 G + moabb_cache 110 G
+ divers 60 G = **878 G**. stieger2021 complet demande ~345 G (287 fichiers .mat
présents pour 165 G, soit ~48 % de 598 sessions) → **1223 G**, contre 1024 G de
quota. Le mur était structurel. Le tirage parallèle a avancé l'heure de la collision
et l'a fait tomber pendant la chauffe des dix autres, rien de plus.

**Le levier, mesuré : le brut est ~10× le cache traité, et il est jetable.**
zhou2020 68 G brut → 6,6 G cache. guttmannflury_mi 25 G → 1,9 G. kumar2024 13 G →
2,3 G. Le pipeline lit `moabb_cache`, pas `mne_data`.

Récupérable sans rien re-télécharger, ~242 G :
- `MNE-guttmannflury2025-data` 50 G — **doublon pur** (contient `ME/` et `MI/`, déjà
  présents dans les deux répertoires séparés `-mi-data` et `-me-data`).
- le brut des datasets dont le cache est **complet** : lee2019-mi 61, schirrmeister
  26, gutt-me 26, gutt-mi 25, kumar2024 13, gigadb 10, Dreyer2023A 9,3, eegfnirs
  5,9, weibo 4,1, hefmiich 3,9, bnci 3,8, eegbci 1,9, liu2024 0,9 ≈ 192 G.

À **garder** : le brut des 5 datasets au cache troué (zhou2020 68 G / 5 sujets
manquants, chang2025 44 G / 6, forenzo2023 40 G / 20, brandl2020 4,6 G / 9,
Dreyer2023B 2,9 G / 1 sujet).

**Non touchable sans l'accord d'Adam** : `brainbert` 281 G + `popt` 30 G = 311 G,
soit 30 % du quota, qui appartiennent aux chantiers de réplication BrainBERT/PopT.

**Redesign nécessaire pour stieger2021 et yang2025** : chauffer sujet par sujet en
effaçant le brut de chaque sujet dès ses epochs cachés. L'empreinte de pointe devient
quelques sujets au lieu de 345 G. yang2025 reste le cas dur — **fichier unique de
65,6 Go**, indivisible.

### Fait le 06/09, après arbitrage d'Adam

**Purge de 235 G** (quota 1024 G → 789 G). Gardée par `check_cache` : les 13 datasets
concernés ont été vérifiés **0 MANQUE, 0 MENSONGE** avant qu'un octet soit effacé.
Effacés : `MNE-{lee2019-mi,schirrmeister2017,guttmannflury2025,guttmannflury2025-mi,
guttmannflury2025-me,kumar2024,gigadb,Dreyer2023A,eegfnirs,weibo-2014,hefmiich2025,
bnci}-data`. `eegbci` et `liu2024` conservés (2,8 G, pas de config à vérifier — pas la
peine de risquer pour ça).

**Le plancher corrigé** (`motif_warm.sbatch`) : il lit `quota(1)`, plus `df`. Et
`warm_cache.py` interrompt la chauffe sur EDQUOT/ENOSPC au lieu de les avaler sujet par
sujet, et remonte l'échec dans son code de sortie.

**Jobs relancés** :
- `541266` / `541267` — warm + D×R des 5 datasets au cache troué (5,6,7,9,10). chang2025
  et dreyer2023b déjà chauffés, D×R en cours.
- `541276` / `541277` — stieger2021 en **flux** (`motif_warm_stream.sbatch`).
- `541278` / `541279` — yang2025 en flux + `--purge-raw-at-end`.
- `539161` annulé (7 tâches `DependencyNeverSatisfied`).

**Le piège de `warm_stream.py`, à ne pas réintroduire.** La garde de suppression a
d'abord lu le **nom de fichier** (« effacer si le basename contient l'identifiant du
sujet »). Testée sur les 287 `.mat` réels de stieger2021 *avant* tout effacement :
**570 revendications pour 287 fichiers**, parce que le numéro de **session** matche le
même motif — le sujet 1 réclamait `S2_Session_1.mat`. Lancée, elle effaçait le brut de
61 autres sujets. Remplacée par une carte `fichier -> sujets` et un **retard d'un
sujet** (le brut du sujet i ne part qu'une fois i+1 chauffé), avec 6 tests dans
`benchmarks/tests/test_warm_stream.py` qui fixent les deux topologies.

## 05/09 16h40 — le pôle série était stieger2021, corrigé par 539433

**Mesuré, pas estimé.** stieger2021 = 598 sessions de ~615 Mo, soit **~360 Go**,
téléchargées **une par une** par 539160_0 : 7,6 Go en 39 min, soit 3,3 Mo/s → **~30 h
pour ce seul dataset**, quand les dix autres finissent en ~3 h. Rien de calculatoire
là-dedans, c'est du réseau en série.

**539433** (`slurm/stieger_pull.sbatch`, array 0-7, CPU) tire les sujets 8..62 en
8 flux parallèles. **Purement additif** : il ne touche pas 539160_0, donc la chaîne
`aftercorr` vers 539161_0 reste intacte. Sûr parce que `Stieger2021.data_path` teste
`os.path.exists` avant chaque fichier (`stieger2021.py:417`) et que les tâches
prennent des sujets disjoints ; et parce qu'il **ne fait aucun epoching** — c'est
l'écriture concurrente dans le cache MOABB *traité* qui détruit une entrée partagée
(incident des 276 sujets), pas le remplissage de `MNE-Stieger2021-data`.

**Effet mesuré** : 3,3 → **15,6 Mo/s** (1,4 Go en 90 s), ×4,7. Reste ~350 Go → ~6 h.
Le débit par flux tombe de 3,3 à 1,9 Mo/s : le rendement marginal décroît, doubler
les flux ne doublerait pas le débit. Ne pas empiler d'aide supplémentaire sans
remesurer.

**ETA de la campagne** : ~3 h pour 9 datasets, ~2 h 30 pour yang2025 (fichier
**unique** de 65,6 Go — non parallélisable), **~7 h pour stieger2021** (6 h de
chauffe + ~1 h de D×R). Coût D×R mesuré sur hefmiich2025 : 37 sujets × 3 seeds =
111 fits à 37,7 s, 3 shards → **19 min** par dataset de cette taille.

## 05/09 — `motif` EN COURS : 11 datasets de plus (539160 chauffe → 539161 D×R)

**Pourquoi.** À n=6 datasets, « la croissance classe mieux les donneurs que
l'accuracy » est **non résolu** : 3/5 admis à écart positif, binomial p=0.50 ; un
modérateur y serait détecté avec une puissance ~0.3. La campagne mesure une
**prévalence**, pas un effet nouveau. Pré-registration :
`analysis/PREREG_motif_prevalence.md`, écrite avant le premier fit.

**Les 11 ajoutés** (LeftRightImagery, n ≥ 16, critères posés avant tout résultat) :
stieger2021 (62), yang2025 (51), hefmiich2025 (37), guttmannflury2025_mi (31),
guttmannflury2025_me (31), chang2025 (28), forenzo2023 (25), zhou2020 (20),
kumar2024 (18), dreyer2023b (21), brandl2020 (16). → 17 datasets au total.

Deux ne sont pas là pour le nombre : **dreyer2023b** est la seconde cohorte du *même
protocole* que dreyer2023a (l'un des deux échecs) — l'échec se réplique-t-il ? Et
**guttmannflury2025_me** est le jumeau en mouvement *exécuté* de `_mi`, mêmes 31
sujets, même matériel : le seul contraste intra-cohorte de la campagne.

| job | quoi | forme | dépendance |
|---|---|---|---|
| 539160 | chauffe du cache MOABB | array 0-10%6, partition `normal`, 0 GPU | — |
| 539161 | matrice D×R | array 0-10, 1 GPU, 3 shards co-tenants | `aftercorr:539160` |

**`aftercorr` et pas `afterok`** : la tâche D×R *i* part quand la tâche de chauffe *i*
sort en 0, donc dès que **ce** dataset est prêt — pas quand les onze téléchargements
sont finis. Le makespan est le max des chaînes par dataset, pas la somme des pires.

**Le D×R est aussi la sonde de coût, et c'est délibéré.** Le projet a transposé trois
fois un coût par cellule et s'est trompé les trois fois (une fois d'un ordre de
grandeur, une fois d'un facteur 7). Un fit donneur voit 100 % des essais d'un sujet, un
fold de sonde 80 % : le `fit_seconds` du D×R **majore** donc le coût de la sonde du même
dataset. La ligne `COST <dataset> ...` dans `/scratch/amounir/logs/motif_dxr_*` est ce
qui dimensionnera `motif_probe`. **Ne pas soumettre la sonde avant de l'avoir lue.**

Le D×R passe sur **GPU** et non CPU comme cho2017 : 1 140 s par fit donneur mesuré sur
ses CSV (`device=cpu`), contre 40 s pour le même réseau en sonde sur turing. Le CPU
était un choix de cohabitation d'un jour où les cartes étaient prises. Facteur ~25.

### Ce qui a failli passer — la porte G2, corrigée avant soumission

Première version : `ICC(1,1) > 0.05`. Passée sur les 6 datasets existants, elle en
excluait **4**, dont **les deux seuls où la règle échoue** (dreyer2023a, physionetmi).
Une porte qui retire précisément les négatifs biaise la prévalence vers le haut sans
rien signaler. Deux défauts s'additionnaient : le seuil portait sur la fiabilité d'un
réplicat isolé alors que la règle emploie une moyenne de 15 (Spearman-Brown : 0.05 sur
un réplicat vaut 0.44 sur quinze), et deux exports de sonde étaient mal routés
(`dynamics_fresh`), rendant un ICC `nan` lu comme un échec.

Corrigée en `ICC(1,15) > 0.30` ; n'exclut plus que shin2017a (0.174). La correction est
consignée **datée** dans la prereg, parce qu'elle a été faite après avoir vu qui elle
excluait — et la garantie n'est pas la bonne foi : **l'endpoint est rapporté deux fois,
avec et sans la porte**, et le script écrit en toutes lettres si les deux divergent.

`k` est égalisé à 15 réplicats/sujet : les ajoutés ont de 1 à 11 sessions, donc k irait
de 15 à 165, et « les nouveaux répondent mieux » aurait pu n'être que « ils ont plus de
fits ».

### Prior gelé (k égalisé), à ne pas reconstruire après coup

| dataset | ICC(1,15) | suspects | gap | gap_clean | classe |
|---|---:|---:|---:|---:|---|
| shin2017a | 0.174 | 0 | +0.101 | +0.101 | *exclu G2* |
| cho2017 | 0.691 | 7 | +0.330 | +0.076 | ROBUSTESSE |
| dreyer2023a | 0.779 | 21 | −0.119 | −0.133 | ABSENT |
| lee2019_mi | 0.606 | 5 | +0.082 | +0.358 | MOTIF |
| liu2024 | 0.453 | 0 | +0.053 | +0.053 | MOTIF_NON_TESTÉ |
| physionetmi | 0.441 | 1 | −0.189 | −0.164 | ABSENT |

**Ce que la campagne ne mesure PAS** : le gain end-to-end d'une allocation. Le lien entre
l'écran D×R et l'end-to-end est faible (P = 0.751, voir plus bas). La validation
end-to-end reste à n=3 et cette campagne ne la déplace pas.

---

## 05/09 — `dalloc_restored` TERMINÉ (531833/531834/531835, tous COMPLETED)

432/432 cellules sur disque, rapatriées dans `benchmarks/analysis/dalloc_restored/`.
Appariement **vérifié et non supposé** : 144 cellules contre 144 dans chaque bras,
1 fit par cellule des deux côtés, et le readout **refuse** de tourner si une ligne
n'a pas sa contrepartie exacte (fold × seed × budget × beta × sujet de test).

Lecture par `analysis/alloc_restored_readout.py`, suivant `PREREG_alloc_restored.md`.

**H_A — delta apparié par sujet de test, `restored_probe` − `acc_probe` :**

| dataset | n | delta | IC95 | p Holm | MDE pré-enreg. |
|---|---|---|---|---|---|
| cho2017 | 52 | +0.0077 | [−0.0031, +0.0183] | 0.331 | +0.0241 |
| lee2019_mi | 54 | **+0.0098** | [+0.0023, +0.0170] | **0.039** | +0.0163 |
| physionetmi | 109 | −0.0048 | [−0.0149, +0.0050] | 0.351 | +0.0184 |

Un seul dataset sur trois est significatif, et les deux nuls sont **sous la MDE
pré-enregistrée** : à écrire « non résolu », jamais « restored ne fait pas mieux »
(cf. [[underpowered-not-null]]).

**H_B — l'écran hors ligne prédit-il le signe ? 3/3.** Mais le chiffre à porter avec
n'est pas « 3/3 » : c'est que **2 des 3 signes ne sont pas statistiquement résolus**.
P(signe prédit) = 0.917 / 0.994 / 0.824 → P(les 3 vrais signes sont ceux prédits)
= **0.751**, soit ~25 % de chance qu'au moins un soit en fait inversé. Et sous
pile-ou-face pur 3/3 vaut p=0.125, non significatif. L'ordre des magnitudes prédit
par l'écran (cho2017 le plus fort) est **démenti** : cho2017 arrive second et n.s.

**Ce que ça change pour l'angle du papier :** `restored_probe` bat `acc_probe` d'au
mieux +0.010 sur un seul dataset, là où `acc_probe` seul capturait déjà tout le gain
contre l'uniforme (+0.024 à +0.067 sur lee2019_mi). Le signal de croissance ajoute
quelque chose de réel mais de marginal ; il ne renverse pas
[[allocation-continuum]].

### Suivi 05/09 — POURQUOI (post-hoc, `analysis/why_lee_responds.py`, zéro GPU)

**La question « pourquoi seulement lee2019_mi » était mal posée : cho2017 répond aussi.**
L'endpoint pré-enregistré moyennait sur une moitié de grille (beta=1, gros budget) où
les deux allocations convergent et n'apportent que des zéros. Dose-réponse sur la
divergence d'allocation : **+0.043/TVD (cho2017)**, +0.024 (lee2019_mi), **−0.053
(physionetmi)**.

**Mais l'avantage de cho2017 est de la robustesse, pas de l'information.** 7 sujets à
accuracy ≥ 0.90 valent rang 22.8/52 comme donneurs ; `acc_probe` leur donne **5.00 de
ses 5 places de top-5 pour chaque receveur**, `restored_probe` **0.00**. En les
retirant, capture d'acc 0.123 → **0.427** et l'écart s'effondre de +0.330 à **+0.076**.

**lee2019_mi est le seul cas propre** : y retirer les hauts performeurs *dégrade*
l'accuracy (0.389 → 0.163) et *fait grossir* l'avantage (+0.104 → +0.380).

**physionetmi est un échec cohérent** : pas d'aberrants, `acc_probe` capture 73 % du
plafond de toute règle globale ; il ne reste rien à prendre.

Réfutées et à ne pas ressortir : « rien à gagner » (physionetmi a le plus gros
gisement), « le nul est mécanique » (Jaccard 0.000 à beta 8), « c'est la fiabilité »
(causale *dans* un dataset, Spearman +1.00, mais l'accuracy est **plus** fiable partout
— ICC(1,1) 0.961 vs 0.130 sur cho2017 — en étant une pire règle). Détail dans
[[why-growth-rule-wins]].

## 04/09 (soir) — `dalloc_restored` SOUMIS : 531833 / 531834 / 531835

**432 cellules, 3 datasets, ≈ 80 GPU·h.** Teste `restored_probe` comme règle
d'allocation — la mesure que H1 pré-enregistre comme primaire et qui **n'avait jamais
été testée en intervention**, parce que le tuple `MEASURES` de `donor_allocate.py` la
déclarait sous le nom `restored_epoch` alors que le classement écrit `restored_probe` :
mesure déclarée, inatteignable, tout lancement se serait fait refuser. Corrigé.

| job | dataset | shards | cellules | budgets |
|---|---|---|---|---|
| 531833 | cho2017 | 6 | 144 | 3/5/10 |
| 531834 | lee2019_mi | 12 | 144 | 3/5/10 |
| 531835 | physionetmi | 12 | 144 | 5/10/20 |

Sharding dimensionné sur le coût **mesuré** sur les 1008 cellules `dalloc` terminées
(médiane par cellule : cho2017 5.3 min, lee2019_mi 11.9, physionetmi 10.9), pas estimé.
Sortie `/scratch/amounir/dalloc_restored/<dataset>`, séparée des globs de `dalloc/`.
`beta=0` n'est pas relancé (déjà sur disque, ne dépend d'aucune mesure) ; budget plein
non plus (il ne se déclenche qu'à beta=0).

**Appariement vérifié avant soumission** : les `ranking_*.csv` régénérés pour porter
`restored_probe` ont un écart **exactement nul** sur `params_probe`, `acc_probe`,
`width_probe`, `ceil_frac` face à ceux du cluster, mêmes sujets. Strictement additif.

Préreg : `benchmarks/analysis/PREREG_alloc_restored.md`, écrite avant le premier fit.
H_B y gèle **3 signes prédits** par l'écran hors ligne (cho2017 +, lee2019_mi +,
physionetmi −) ; le succès c'est 3/3, et 2/3 sera rapporté comme un demi-échec.

## 04/09 — **FILE VIDE, TOUT EST FINI.** Les deux campagnes sont lues.

`squeue -u amounir` ne renvoie rien. Les trois chantiers sont terminés et rapatriés :
`eegrow_shipped` (542 cellules, 23 848 lignes), `eegrow_dalloc` (1008/1008 cellules) et
la réplication fraîche (sonde 6/6 + deux matrices D×R).

### 1. Réplication confirmatoire de claim 2 — l'endpoint primaire est tombé

`benchmarks/analysis/claim2_fresh.py`, chaîne pré-enregistrée dans l'ordre imposé.

**liu2024 est écarté par les deux garde-fous, qui sont d'accord** : ICC(1,k) 0.453/0.454
sous le plancher 0.50, et garde-fou G à **−0.097** (p=0.75) contre +0.56/+0.67/+0.67 en
référence. Le diagnostic sur la sonde seule dit pourquoi : **acc_probe médiane 0.508 en
ROC-AUC, soit le hasard**, et **189/750 fits restaurés à l'époque ≤ 1** — liu2024 n'a que
~40 essais par sujet, la sonde n'y apprend pas. Ce dataset ne compte ni pour ni contre.

Donc **l'endpoint groupé sur 110 sujets n'est pas exécutable**, et ce qui reste est une
estimation mono-dataset que la préreg qualifie elle-même de descriptive, à un MDE
(+0.321) au-dessus de la cible (+0.288) :

| | ρ partielle | IC95 | p unilat. | Holm | MDE |
|---|---|---|---|---|---|
| H1 `restored_epoch` | +0.254 | [+0.017, +0.457] | 0.018 | franchi | +0.321 |
| H2 `params_probe` | +0.308 | [+0.049, +0.531] | 0.010 | franchi | +0.321 |

dreyer2023a passe tout le reste largement : ICC(1,k) 0.778/0.748, G **+0.671**,
k_probe = 15 pour les 60 sujets. **H3 s'inverse** : `params` survit au conditionnement
(+0.187) et `restored_epoch` non (−0.055), l'inverse de l'étage C — sous une colinéarité
de +0.857 qui n'était pas moindre historiquement (0.60–0.87).

### 2. Continuum d'allocation — 9 budgets sur 9, tous les IC excluent zéro

Delta apparié par sujet de test, β* contre β=0, **à budget d'essais fixe** :
cho2017 +0.058 / +0.049 / +0.026, lee2019_mi +0.071 / +0.044 / +0.029,
physionetmi +0.123 / +0.087 / +0.023. Courbe monotone, saturation vers β=2–4,
**33–51 % de l'écart au plafond `all_pool` récupérés** pour un quart à un dixième des
essais. Ne contredit pas le plafond de sélection : là on perdait les essais des sujets
écartés, ici le budget est tenu constant. La sélection est dominée, l'allocation non.

### Ce qui reste à faire (rien ne tourne)

Rapport HTML, message à Sylvain (**pas envoyé sans accord d'Adam**), commits, et la
re-lecture du `shipped` par frames par sujet.

## 04/09 15:40 — **311 fits fantômes trouvés et purgés dans la sonde fraîche** (données maintenant propres)

Le contrôle d'intégrité d'avant-analyse a trouvé une pollution que j'avais créée
moi-même. dreyer2023a avait **405/403/403 enregistrements de fit pour 300 attendus**, avec
des sujets à 10 fits au lieu de 5 — et de façon *variable selon la seed*, ce qui
ressemblait à un défaut de données.

**Cause, et elle est de moi.** En annulant `527597` (la sonde condamnée par
`CELL_TIMEOUT`) j'ai supprimé `eegrow_claims_fresh` **mais pas** `results_fresh`. Les
`__fits.jsonl` sont ouverts en **append** : `527804` a donc empilé ses 300 fits propres
par-dessus les ~103 fits partiels du job mort. liu2024 est intact — le job annulé
n'avait jamais atteint la 2ᵉ vague.

**Pourquoi le CSV de scores ne montrait rien.** Il est écrit **une fois, à la fin de la
cellule** ; le jsonl est écrit **fit par fit**. Un job tué laisse donc des scores propres
et des fits pollués. C'est l'image miroir exacte du piège déjà connu
([[jsonl-truncated-midwrite]], où un export en cours de campagne tronquait les fits sans
toucher aux scores). Même asymétrie, sens inverse : **ne jamais valider une campagne sur
le CSV de scores seul.**

**Et ce n'était pas cosmétique.** `donor_select_prep.py` construit `params_probe` comme
une *moyenne* sur les réplicats et `k_probe` comme leur *compte*. Les fits fantômes
contaminaient la moyenne et gonflaient k — or k pilote `ICC(1,k)`, la porte d'admission
de la réplication, via Spearman-Brown. La porte aurait été appliquée sur un k faux.

**Purge, et elle est vérifiable.** La relance ayant écrit un bloc complet et contigu, les
**300 dernières lignes** de chaque fichier sont exactement le run propre : `cv_ind`
0..299 strictement croissant, 60 sujets × 5 fits. Le préfixe (105/103/103 lignes,
`cv_ind` 0..104) est le run mort. Originaux conservés dans
`/scratch/amounir/results_fresh_backup_avant_purge/`. Après purge, les 6 fichiers sont
conformes (`cv_ind` complet, fits/sujet constant à 5) et **local et cluster sont
bit-identiques** (md5 vérifiés des deux côtés).

Le test qui a tout révélé, à passer désormais avant toute analyse : `cv_ind` doit couvrir
exactement `0..n_sujets*n_folds-1` sans répétition, et le nombre de fits par sujet doit
être constant. Un décompte de cellules ne l'aurait jamais vu.

## 04/09 15:10 — **PLUS AUCUN JOB EN FILE. La sonde fraîche est complète (6/6, 110 sujets).**

`527804` : `COMPLETED 0:0` en **1 h 28**. Sortie vérifiée fichier par fichier —
dreyer2023a **60 sujets × 3 seeds**, liu2024 **50 × 3**, six CSV, aucun trou, aucun
doublon. Les 110 sujets jamais vus de la réplication pré-enregistrée sont mesurés.

**Et la 4ᵉ correction du modèle de coût, qui est la variable qui manquait aux trois
autres : la classe de carte.** J'avais prévu 6 h ; il a atterri sur des **A40**
(margpu007, 46 Go) et non sur les 2080 Ti/turing où toutes mes mesures avaient été
prises. ~8,8 s/fit en solo sur A40 contre ~35 s/fit solo-équivalent sur 2080 Ti : un
**facteur ~4 entre classes**. Tout dimensionnement calibré sur turing est donc ~4×
pessimiste sur ampere, et un coût mesuré sans sa classe de carte n'est pas reproductible.
Le fond ne change pas (travail total fixe, seul le nombre de cartes raccourcit le wall
clock **à classe donnée**) — mais l'ironie est nette : c'est en relâchant le pin `ampere`
que le job a fini par tomber sur de l'ampere, et c'est ce qui l'a rendu 4× plus rapide.

### Tout ce que la pré-registration demande est sur disque

| pièce | état |
|---|---|
| matrice D×R dreyer2023a (`525777`) | 60/60 unités |
| matrice D×R liu2024 (`525778`) | 50/50 unités |
| sonde `grow_shallow` × 3 seeds × 2 datasets (`527804`) | 6/6 cellules, 110 sujets |

### Le prochain pas est un EXPORT, pas du calcul

`donor_select_prep.py` ne lit pas `results_fresh/` : il lit `analysis/dynamics_final/
gd_fits.csv.gz` et `analysis/perf_final/scores`, c'est-à-dire les **réductions** de
campagne. Les résultats frais doivent donc passer par la même étape d'export que celle
déjà en attente pour `shipped`. C'est une tâche **locale** — le checkout cluster n'a même
pas `perf_io.py`, et l'env d'analyse est le `.venv` local (pandas 3.0.5). Rapatriement de
`results_fresh` en cours.

Ordre imposé par la pré-registration, à ne pas inverser : export → **porte ICC(1,k) ≥ 0.50
sur la sonde seule** → garde-fou G (`acc_probe` doit prédire `q_cent` par dataset) → H1
groupée (primaire) → H2 (Holm) → H3 (descriptive, sans test).

## 04/09 10:50 — `dalloc` COMPLET ; la sonde resoumise une 3ᵉ fois (**527804**), correctement dimensionnée

### `dalloc` est fini : 1008/1008

lee2019_mi est passé de 268 à **336/336** en 4 à 28 min par shard — `donor_allocate.py`
a bien repris là où le TIMEOUT l'avait laissé, sans refaire les 268. Les trois datasets
sont à 336/336. L'analyse du continuum d'allocation (courbe en beta à budget fixé,
contrôle beta=0 contre `all_pool`, appariement à `n_train` égal contre `dsel2/`) est
maintenant possible.

### La sonde : 3ᵉ soumission, et je m'étais trompé trois fois sur son coût

`527597` a bien démarré (le garde-fou CUDA était réglé) et tournait depuis 2 h 34 sur
margpu018. En lisant SON log plutôt qu'en transposant : **89 fits en 2 h 34 = 104 s/fit**,
pour ~300 fits attendus (60 sujets × 5 folds, comme les 260 fits mesurés sur les 52
sujets de cho2017). Soit **8,6 h par cellule** contre un `CELL_TIMEOUT` à 4 h — et ce
timeout est un `timeout` dur (`pack_run.sh` ligne 523). Les 3 cellules allaient être tuées
à 10:00 UTC **sans écrire une seule ligne**. Annulé et resoumis avant l'échéance ; 0 CSV
produit, donc rien de perdu au-delà des 2 h 34 de carte.

Les trois estimations successives, pour ne pas refaire le tour :

| # | valeur | d'où elle venait | verdict |
|---|---|---|---|
| 1 | 4 h/cellule | cho2017, **par nombre de sujets** (60/50 vs 52) | faux |
| 2 | 1 h 15 | matrices D×R sur ces mêmes datasets, **par n_chans × n_times** | faux ×7 |
| 3 | **8,6 h** | **le log du job lui-même** | la vraie |

Les deux axes de transposition sont justes et aucun ne suffit : ni l'un ni l'autre ne
porte **la co-tenance** (K processus sur une carte divisent la vitesse de chaque fit par
~K) ni **le nombre de folds**.

Ce que ça change au dimensionnement : le travail total est fixe, ~17 GPU-heures. Une
seule carte ne peut pas le raccourcir — K=3 triple le débit apparent et divise la vitesse
par trois. **Seul le nombre de cartes achète du wall clock.** D'où `527804` : **G=3 K=1**,
chaque cellule seule sur sa carte à ~35 s/fit ≈ 2 h 55, deux vagues ≈ **6 h**.
`CELL_TIMEOUT=18000` (5 h), walltime 11 h. K=1 plutôt que K=2 à cartes égales : même wall
clock, mais la première vague est **écrite sur disque au bout de 3 h** au lieu que les six
cellules finissent ensemble et se perdent toutes sur un walltime raté.

## 04/09 04:35 — file vidée : **2 matrices réussies, 2 échecs qui sont les miens**, tout est resoumis

### Ce qui a réussi

**Les deux matrices D×R fraîches sont FAITES.** `525777` dreyer2023a **60/60 unités** en
2 h 24, `525778` liu2024 **50/50** en **19 minutes**, les deux `COMPLETED 0:0`.
La moitié « matrice » de la réplication pré-enregistrée est donc acquise ; il ne manque
que la sonde pour pouvoir joindre les deux et lire H1/H2/H3.

`dalloc` cho2017 **336/336** et physionetmi **336/336**, `COMPLETED` en 41-53 min/shard.

### Deux échecs, et les deux viennent de moi

**1. La sonde `525776` est morte en 1 seconde.** `PACK FATAL: EEGROW_CUDA_FRACTION
unset` : j'avais écrit `G=3 K=2` dans `fresh_probe.sbatch` sans jamais définir la
fraction CUDA. Le garde-fou a fait **exactement** ce qu'il devait faire — il a refusé
plutôt que d'avertir et de tourner sur une valeur inventée ([[gates-must-refuse-not-warn]]).
Coût réel : ~6 h de file d'attente perdues sur le chemin critique du chantier ICLR.

**2. Quatre shards `dalloc` lee2019_mi ont été tués en TIMEOUT à 1 h 30.** C'est
directement mon `scontrol update TimeLimit=01:30:00` d'hier soir. J'avais mesuré
28,6 s/cellule sur **cho2017** et généralisé ; lee2019_mi est plus lourd et a débordé.
cho2017 et physionetmi sont passés (41-53 min), lee2019_mi non : **268/336 cellules**,
**68 manquantes**. Le walltime honnête achetait bien du backfill, mais je l'ai calibré
sur le dataset léger au lieu du dataset lourd. `donor_allocate.py` sait reprendre
(`skip (already done)`, ligne 203), donc rien n'est à refaire — seulement à finir.

### La correction de fond : le coût ne suit pas le nombre de sujets

J'avais estimé 4 h/cellule pour les datasets frais « du même ordre de grandeur que
cho2017 » — en comparant **le nombre de sujets** (60 et 50 contre 52). Faux d'un ordre de
grandeur. Les logs de 525777/525778 mesurent **15 s par fit de 200 époques** contre 41 s
sur cho2017. Le bon axe est **n_chans × n_times** : cho2017 a 64 canaux, dreyer2023a 27
(× 1250) et liu2024 29 (× 1000). D'où une matrice liu2024 en 19 min au lieu de 12 h.

Le même axe donne la fraction CUDA **sans l'inventer**. `grid_memory_v2.json` est quasi
affine en n_chans × n_times pour `grow_shallow` sur 2080 Ti : bnci2014_002 18 750 → 788 MiB,
bnci2014_001 22 022 → 908, shin2017a 75 030 → 3668 (0.041 à 0.049 MiB/unité). dreyer2023a
33 750 → ~1650 MiB (**0.15** de la carte), liu2024 29 000 → ~1420 (**0.13**). Et
corroboration indépendante : 525777/525778 viennent de tourner à **0.30**, trois
co-tenants sur une carte, sans un OOM. Valeur retenue : **0.30**, double du pire cas.

### Resoumis

| job | quoi | changement |
|---|---|---|
| **527597** | sonde fraîche, 6 cellules | `EEGROW_CUDA_FRACTION=0.30` ; **1 carte** au lieu de 3, `G=1 K=3`, 8 h. À ~1 h 15/cellule, 3 cartes n'achetaient qu'une heure de wall clock pour une attente en file bien plus longue — et ma priorité est tombée à 2448. |
| **527598** | `dalloc` lee2019_mi, 68 cellules | walltime **natif 4 h**, plus de `scontrol`. Reprend, ne refait pas. |

Veille ré-armée (le flux précédent s'est terminé sur « plus aucun job »).

## 04/09 00:40 — **`eegrow_shipped` est TERMINÉE** (542/542), et le front frais recule à ~03 h 30

### La campagne est finie, et c'est vérifié et non déduit

Les **11 allocations** sont `COMPLETED` avec `ExitCode 0:0` — aucune `TIMEOUT`, aucune
`FAILED`. La plus longue (515942) a tourné 13 h 41 sur un plafond de 48 h, donc elles se
sont arrêtées faute de travail et non faute de temps. `515939`, la 4ᵉ allocation qui
attendait en `PD(Resources)` avec un walltime de 2 jours, a démarré et est **sortie en
27 secondes** : exactement le comportement prédit hier soir — elle n'avait plus une seule
cellule à réclamer, et c'est pour ça qu'il ne fallait pas l'annuler mais la laisser
libérer sa réservation toute seule.

Sortie mesurée : **542 claims, 542 CSV, 0 cellule orpheline**, 0 fichier vide ou tronqué,
**23 848 lignes** de résultats. Réparties en 308 `within_session` (12 datasets),
162 `cross_session` (6), 72 `cross_subject` (12).

*Réserve honnête* : je n'ai pas retrouvé le `.tsv` qui définissait la grille `shipped`
(le checkout d'origine n'est plus là, et `final_grid.tsv` — 1116 lignes — est celui de
`results_final`, pas de celui-ci). « 542 = la grille entière » repose donc sur la
signature d'auto-terminaison (claims == CSV, 0 non réclamée, 11 sorties propres bien
avant le walltime), pas sur un décompte de lignes. À reconfirmer au moment de l'export.

### Le front confirmatoire a reculé, et la raison est mécanique

Les 11 turing libérées n'ont pas profité aux jobs frais : elles ont été prises par mes
**propres shards `dalloc`** (6 en cours sur margpu017 et margpu022) et par deux autres
utilisateurs. Surtout, **ma priorité s'est effondrée de 7247 à 2448** — c'est le fairshare
qui encaisse les ~60 GPU-heures que `shipped` vient de consommer. Terminer une grosse
campagne coûte donc du rang sur la suivante ; ça n'a rien d'une anomalie, mais ça
invalide l'ETA d'hier soir.

Slurm donne maintenant une date ferme pour `525777` : **`StartTime=2026-09-04T01:28 UTC`**,
soit 03 h 30 Paris. Nouveaux ETA :

| | fin attendue (Paris) |
|---|---|
| `eegrow_dalloc` (6 R + 2 PD, 40 min/shard) | ~02 h |
| sonde fraîche `525776` (4 h après départ) | ~07 h 30 |
| matrices `525777/525778` | matinée, 15 h 30 au pire |
| analyses H1/H2/H3 + garde-fou G | **début d'après-midi**, pas fin de matinée |

Je n'ai pas mis les shards `dalloc` restants en `hold` pour dégager la voie : le backfill
les laisserait passer de toute façon (1 h 30 de walltime se glisse avant une réservation
à 01 h 28 sans la retarder), donc le gain serait nul et le coût réel sur la continuité
d'allocation.

### Débloqué par cette fin de campagne

Le ré-export des réductions + le **diff par sujet** (`params_probe`, `acc_probe`,
`k_probe`, `ceil_frac`, `width_probe`) — et non un diff des décomptes de cellules — puis
l'analyse de changement de classement (Kendall tau, inversions de signe, top-1) dans un
4ᵉ HTML. Ne dépend d'aucun GPU : faisable pendant que la file tourne.

## 03/09 19:45 — ETA mesuré, et le pin `ampere` corrigé (il coûtait ~14 h)

### L'ETA, mesuré sur disque et non estimé au walltime

| campagne | avancement mesuré | fin attendue (Paris) |
|---|---|---|
| `eegrow_shipped` (515937/38/42) | **498 CSV / 507 claims** — 9 cellules, toutes déjà en vol | **~00 h – 01 h cette nuit** |
| `515939` (4ᵉ alloc, PD Resources) | 0 cellule à prendre : les 9 sont claim'ées | démarre puis sort en ~15 min |
| `eegrow_dalloc` (525602-604) | 1 shard R, 11 PD, 40 min/shard mesuré | ~06 h demain (estimation backfill) |
| `525776` sonde fraîche | pas démarré | +4 h après démarrage → ~04-05 h |
| `525777/525778` matrices D×R | pas démarrées | +12 h max → matinée |
| analyses H1/H2/H3 + garde-fou G | dépendent de sonde ∩ matrices | **fin de matinée demain** |

Les 9 cellules restantes de `shipped`, avec l'heure de prise du claim (UTC) : schirrmeister
grow_deep seed0 14:45 / seed1 16:54 / seed2 **17:24**, cho2017 grow_shallow seed1 16:58 /
seed2 17:16, lee2019_mi seed1 17:13, physionetmi cross_subject seed0 13:58, weibo2014
seed1 17:32 / seed2 17:37. Le chemin critique est schirrmeister seed2 : 4,74 h mesurées
par cellule → ~22 h UTC.

### Le pin `ampere` était devenu le goulot, et c'est mesuré

`sinfo -O gres,gresused` sur tout le cluster : **aucune carte ampere libre nulle part**.
Sur le tau il n'y en a que 6 (margpu007-008) et elles sont prises par `jjobard` (array
517347, ~20 h restantes sur margpu007), `aszatkow` (un `bash` interactif avec **5 j 15 h**
restantes sur margpu008) et `ochabane`. Le raisonnement écrit dans les deux sbatch
(« ampere, PAS turing, parce que shipped occupe les turing ») était juste au moment où il
a été écrit et **faux maintenant** : `shipped` est à 98 % et rendra 11 cartes turing
(margpu017 ×4, margpu019 ×4, margpu022 ×3) cette nuit. Demander `ampere` revenait à
attendre ~20 h la classe la plus rare pendant que la plus abondante se libérait.

Corrigé **sans resoumettre**, par `scontrol update` sur les jobs en attente — l'âge du job
et donc la priorité sont conservés, ce qu'une resoumission aurait remis à zéro :

- `TresPerNode=gres/gpu:N` sans classe (N=3 pour la sonde, **N=1** pour les matrices —
  `donor_receiver.sbatch` ne demande qu'une carte, pas trois) ;
- `Partition=tau,gpu` (pas `gpu-best`, préemptible : acceptable pour une cellule de
  20 min, pas pour une matrice de 12 h) ;
- `ExcNodeList` complétée. **C'est le point qui rendait la relaxation dangereuse** : la
  protection contre les nœuds cassés était portée *en creux* par le pin ampere.
  `donor_receiver.sbatch` n'avait **aucun** `--exclude`, et `fresh_probe.sbatch` avait
  oublié **margpu020** — le nœud à carte fantôme qui a tué `515639_2` sans message le
  02/09. Liste appliquée : margpu001,004 (volta), 020,021 (fantômes), 023 (drained),
  024-027,029 (pascal sm_60, `idle` dans sinfo et inutilisables).

`Reason` est passé de `Priority` à `None` sur les trois. Fichiers corrigés et redéployés
(md5 identiques local/cluster : `5c4c938d…` et `b05e79cc…`).

## 03/09 19:40 — **le front confirmatoire est ouvert** : cache frais VERT, pré-registration écrite, sonde + 2 matrices lancées (525776-525778)

### Ce qui a changé en une heure

`warm_fresh` (525597) a **réussi en 18 min**, pas en 12 h : les données brutes étaient
déjà dans `mne_data`, il ne restait que la conversion BIDS. La porte du cache est
verte — **dreyer2023a 60/60, liu2024 50/50, zéro manquant, zéro mensonge**, sur la clé
de la campagne. Les deux datasets frais sont disponibles **maintenant**.

### Trois jobs soumis (525776, 525777, 525778)

| job | quoi | coût | walltime |
|---|---|---|---|
| 525776 `freshprobe` | sonde `grow_shallow within_session` × 2 datasets × 3 seeds | ~24 GPU-h, 6 cellules **en parallèle** (G=3 × K=2) | 12 h |
| 525777 `dxr_dreyer` | matrice D×R, 60 donneurs × 3 seeds = 180 unités | 3 shards locaux | 12 h |
| 525778 `dxr_liu` | matrice D×R, 50 donneurs × 3 seeds = 150 unités | 3 shards locaux | 12 h |

**La sonde et les matrices sont INDÉPENDANTES** — une matrice D×R mesure des scores de
transfert et n'a pas besoin du classement ; c'est seulement la corrélation finale qui
les joint. D'où les trois en parallèle plutôt qu'en série.

Fan-out local plafonné à **3** processus et pas plus : `donor_receiver.sbatch` demande
une seule carte avec `EEGROW_CUDA_FRACTION=0.30`, donc 3 × 0.30 = 0.90 tient et 6
aurait OOM la carte.

`RESULTS_DIR` / `CLAIMS` / `LOGS` surchargés **ensemble** vers `*_fresh` : `pack_run.sh`
l'exige (ligne 57) et partager `CLAIMS` avec `eegrow_shipped` en cours ferait moissonner
ses cellules vivantes par `reap_stale`.

### La pré-registration est écrite AVANT le premier fit

`benchmarks/analysis/PREREG_claim2_fresh_dreyer_liu.md`. Primaire H1 :
`restored_epoch` prédit `q_cent` contrôlé par `acc_probe`, unilatéral, **groupé sur les
deux datasets**. Cible à répliquer : **+0.288 [+0.164, +0.414]**.

**Puissance calculée avant, pas après** : MDE +0.321 à n=60 et +0.351 à n=50 — les deux
**sous-puissants** seuls — contre **+0.241 groupé, puissant**. D'où l'endpoint groupé
comme primaire et les deux datasets en descriptif. Un nul sur `liu2024` seul ne sera
pas rapporté comme un échec.

### La porte d'exclusion a été refaite, et deux candidats ont été RÉFUTÉS

L'échéance de [[prereg-gate-one-sided]] (corriger avant un 6ᵉ dataset) est arrivée.
Mesuré sur les 12 datasets :

1. **Censure bilatérale à 0.50 — réfutée.** Elle attrape bien `shin2017a` (0.002 en
   haut, **0.644 en bas** : il passait l'ancienne porte), mais elle **exclurait
   `physionetmi`** (0.514), le dataset au plus fort effet et à la plus grande dispersion.
2. **Sujets épinglés à une borne — réfutée, vide.** 0 % partout sauf schirrmeister (21 %),
   bnci2014_001 (11 %), bnci2014_002 (7 %).
3. **Fiabilité du classement `ICC(1,k) ≥ 0.50` — retenue.** Spearman ne voit que des
   rangs : ni l'étendue ni la censure ne l'atteignent, seule la reproductibilité du
   classement l'atténue.

**Correction sur le passé** : sous ce critère `bnci2014_001` **passe** (ICC 0.928, ses
89 réplicats/sujet). Il reste exclu, mais pour **la vraie raison — n=9**, MDE +0.94.
L'exclusion publiée était le bon verdict pour le mauvais motif ; l'analyse ne change
pas, la justification si.

### Correctif sur les campagnes d'allocation

Les 11 shards `dalloc` en attente demandaient **4 h pour 40 min de travail réel**
(mesuré : 28,6 s/cellule, 41 CSV en 1 171 s). `scontrol update TimeLimit=01:30:00` sur
525602/3/4 — aucune perte, aucun re-submit, et le backfill peut enfin les placer.

---

## 03/09 18:55 — **étage 5 lancé (525602-525604)**, cache frais en route (525597), et le jumeau n'est PAS fini

### Le jumeau, d'abord, parce que c'était la question

`eegrow_shipped` = **8 des 12 shards terminés**. 515937 (4 h 41, margpu017), 515938
(1 h 46, margpu019) et 515942 (7 h 57, margpu022) tournent encore, walltime 2 jours
chacun ; **515939 est PENDING(Resources)** derrière eux. Aucune date de fin fiable tant
que 515939 n'a pas démarré. `results_shipped/` compte **513 fichiers de fits**, répartis
12 / 6 / 12 répertoires sur within_session / cross_session / cross_subject. Le diff par
sujet prévu par [[eegrow-twin-reports]] attend la fin de la campagne — et il portera sur
les frames par sujet, jamais sur les décomptes de cellules.

### Ce que l'analyse gratuite a trouvé AVANT de dépenser un GPU

Les ~2 900 pools déjà entraînés aux étages 3 et 4 relus non pas comme cinq bras à
comparer, mais comme des points d'une relation **composition → qualité**. Chaque cellule
connaît sa composition exacte (colonne `chosen`) et son score. Cela donne ce qu'aucune
comparaison de points ne donnait : la PENTE et l'ÉTENDUE, donc le plafond.

**Le prix de sélectionner dépasse le plafond d'une règle parfaite dans 36 cas sur 36**
(3 datasets × 3 valeurs de K × 4 plis). Le « plafond » est le meilleur pool observé à K
— un maximum sur échantillon fini, donc *surestimé* par le bruit, ce qui va dans le sens
de la conclusion ; le 90ᵉ centile est imprimé à côté et il est plus bas partout.

| dataset | K | plafond (max) | plafond (p90) | prix de sélectionner | plis |
|---|---|---|---|---|---|
| cho2017 | 3 / 5 / 10 | +0.067 / +0.058 / +0.051 | +0.044 / +0.039 / +0.028 | +0.102 / +0.085 / +0.066 | 4/4 |
| lee2019_mi | 3 / 5 / 10 | +0.079 / +0.066 / +0.046 | +0.067 / +0.047 / +0.035 | +0.132 / +0.101 / +0.065 | 4/4 |
| physionetmi | 5 / 10 / 20 | +0.099 / +0.077 / +0.049 | +0.073 / +0.054 / +0.034 | +0.206 / +0.136 / +0.079 | 4/4 |

« Notre règle échoue » et « aucune règle ne peut réussir » cessent d'être
indiscernables. C'est la seconde qui est mesurée.

**Et la composition compte beaucoup** — ρ du score du pool avec l'accuracy moyenne de
ses membres : +0.56 / +0.67 / +0.67. La sélection n'est pas inutile : elle est
**dominée**. À accuracy du pool contrôlée, `params_probe` et `restored_epoch` ajoutent
encore +0.47 à +0.56 sur lee2019_mi et physionetmi — et **rien sur cho2017**
(−0.083 et +0.013 n.s.), qui est déjà le dataset où claim 2 était la plus faible
(+0.150). L'ordre des trois datasets est le même au niveau du pool qu'au niveau du
sujet : cohérence interne, pas réplication.

### 525602 / 525603 / 525604 — l'axe que les étages 3 et 4 n'ont pas parcouru

`donor_allocate.py` (nouveau). Toute méthode d'usage d'un vivier revient à **répartir un
budget d'essais** sur ses M donneurs : `w_i ∝ exp(beta·z_i)`, `n_i = B·w_i` écrêté au
nombre d'essais que le donneur possède, surplus redistribué par paliers. Les deux
protocoles publiés en sont des cas particuliers — `beta=0` étale sur tous les donneurs,
`beta` grand redevient la sélection top-K — donc **un seul balayage relie l'étage 3 à
l'étage 4** au lieu de deux points isolés.

Ce que ça répare : `all_pool` gagne 45/45 avec **plus de sujets ET plus d'essais**. Le
docstring de `donor_select.py` le dit lui-même, « la comparaison mélange plus de données
et meilleures données ». À budget d'essais fixé, `beta=0` contre `beta` grand ne fait
plus varier que **sur combien de sujets** le budget est réparti.

| job | dataset | budgets (équivalent-sujets) | cellules |
|---|---|---|---|
| 525602 | cho2017 | 3 / 5 / 10 + plein | 336 |
| 525603 | lee2019_mi | 3 / 5 / 10 + plein | 336 |
| 525604 | physionetmi | 5 / 10 / 20 + plein | 336 |

4 shards chacun, `--constraint=ampere|turing|hopper` (la vraie condition sm_75+ plutôt
qu'une liste de nœuds), partitions `tau,gpu,gpu-best`.

**Deux défauts attrapés avant la soumission, et c'est à ça que sert le `--dry-run`.**

1. *À budget plein, tous les beta sont identiques.* Quand B égale la somme des essais
   disponibles, l'écrêtage sature tous les donneurs et l'allocation devient l'identité :
   les sept beta produisaient la même cellule. **84 fits par dataset qui n'auraient été
   que des copies d'`all_pool`.** Ce n'est pas un bug mais une propriété : on ne
   repondère pas un vivier qu'on doit épuiser, sauf à dupliquer des essais — ce qui
   ferait entrer des doublons dans le split de validation de skorch et biaiserait le
   choix d'époque. Refusé. Le budget plein reste comme **ancre** (beta=0 doit retomber
   sur `all_pool`) et l'axe est porté par les budgets contraints.
2. *`pick_device()` prend `model_cfg`.* Mort en 20 s au premier smoke (525593).

**Smoke réel avant soumission, pas seulement un dry-run.** 525596 : budget visé 2021 →
**2021 essais livrés**, 606 → 606, 1010 → 1010, 39/39 donneurs à beta=0, sha de
provenance `58c6bee` identique aux cellules de `dsel2/`. Mais le shard n'avait tiré que
des cellules beta=0 — le chemin d'écrêtage restait non testé, donc 525601 l'a forcé :
`beta=8` sert **5 à 7 donneurs sur 39**, budget toujours exact. Le continuum va bien de
39 donneurs à 5.

**L'appariement est la condition de validité, et il tient par deux clous.** Les plis
viennent de `donor_select.folds_of` **importée**, jamais réimplémentée ; le checkout est
`eegrow_budget`, épinglé au même `EEGROW_SHA=58c6bee` que les cellules de `dsel2/` et
`dsel_all/`. Casser l'un des deux casserait l'appariement **en silence**, sans qu'aucune
colonne ne le signale.

### 525597 — le cache des deux datasets frais (CPU, aucun GPU)

`dreyer2023a` (60 sujets) et `liu2024` (50), tous deux 2-class main gauche/droite, même
paradigme `LeftRightImagery` que cho2017 et lee2019_mi. Sur `--partition=normal`,
marg001, 12 h de walltime : c'est du téléchargement et du décodage mono-thread, donc ça
part **pendant** que les cartes sont prises au lieu d'attendre derrière.

Pourquoi eux : `restored_epoch` et la lecture « la taille » de claim 2 ont été formulés
SUR les cinq datasets déjà mesurés, et un prédicteur post-hoc ne devient primaire que
sur des données fraîches. Les sept autres datasets du cache sont à n ≤ 14 (MDE ρ ≥ 0.69)
et ne peuvent rien trancher seuls. Ces deux-là sont à n=60 et n=50, soit la puissance de
lee2019_mi. **Aucune prédiction n'est encore écrite** : ce job rend le test possible, il
ne le fait pas.

À surveiller : `/scratch` est à **91 %** (5,0 To libres), le cache MOABB pèse déjà 79 Go
et `mne_data` 112 Go. Le job imprime `df`/`du` avant et après — un cache à moitié écrit
par manque de place est exactement l'état qui déclenche le `rmtree` destructeur de MOABB.

## 03/09 17:30 — **523644 + 523645 TERMINÉS** (COMPLETED 0:0) et analysés

| job | elapsed | nœud | carte |
|---|---|---|---|
| 523645 D×R shin2017a | **14:05** | margpu011 | ampere |
| 523644_0 `all_pool` | 48:35 | margpu006 | ampere |
| 523644_1 | 29:17 | margpu010 | ampere |
| 523644_2 | 56:34 | margpu018 | turing |
| 523644_3 | 1:00:21 | margpu019 | turing |

Intégrité : 32 cellules `all_pool` (4 plis × 8 graines) et 87 unités D×R (29 donneurs
× 3 graines) — comptes attendus, campagne fermée avant export, donc pas de
`__fits.jsonl` tronqué au sens de [[jsonl-truncated-midwrite]].

### Étage 4 — `all_pool` sur lee2019_mi : **15 comparaisons sur 15**, 4 plis concordants

`all_pool` (K=40) bat les 5 règles aux 3 K. Écarts par sujet de test, du plus petit au
plus grand : `params_top` K=10 **+0.046**, jusqu'à `params_bottom` K=3 **+0.210**.
`roc_auc` moyen d'`all_pool` = 0.8045 (n=54).

La base de l'étage 4 passe donc de **2 à 3 datasets, 45/45 comparaisons**. C'était le
point le plus étroit du dossier ; il ne l'est plus.

**Le contrôle de sous-entraînement est vacuous, et il faut le dire.** Les deux bras
sortent à `epochs médiane 200 [200, 200], budget 100 %` — sur les trois datasets. C'est
mécanique : `patience=200` = `max_epochs=200`, donc l'early stopping ne peut jamais se
déclencher et `stop_reason` ne discrimine rien. Le mode d'échec que le script cherchait
(`all_pool` coupé à `max_epochs` pendant que les règles s'arrêtent tôt) **ne peut pas se
produire**. Ce qui sauve la lecture n'est pas ce contrôle mais l'asymétrie de charge :
`all_pool` voit 4 à 8× plus d'essais par époque, donc **plus** de pas de gradient à
budget d'époques égal — il est sur-entraîné par rapport aux règles, pas sous-entraîné.
La direction de l'artefact possible est l'inverse de celle qui menacerait la conclusion.

### Étage 1-2 — shin2017a réplique, mais trébuche sur un angle mort de la porte

`ranking_shin2017a.csv` construit depuis la campagne `final` (2610 fits `grow_shallow` /
`within_session`, k_probe=45 réplicats/sujet — le plus dense des cinq).

| | n | ρ(#params) | ρ partielle | MDE | ICC_k | plafond | plancher |
|---|---|---|---|---|---|---|---|
| **shin2017a** | 29 | +0.256 [−0.103, +0.562] | **+0.210** [−0.196, +0.545] | 0.549 | 0.903 | **0.00** | **0.64** |

Même signe que les quatre autres, sous-puissant (MDE 0.549 > l'effet groupé ~0.29) —
donc non concluant seul et **non nul**, au sens de [[underpowered-not-null]].

Méta, primaire pré-enregistrée inchangée, sensibilité imprimée à côté :

| | n donneurs | ρ groupée | IC95 | I² |
|---|---|---|---|---|
| 3 datasets retenus (pré-enregistré) | 215 | **+0.288** | [+0.158, +0.414] | 19 % |
| + shin2017a (post-hoc) | 244 | +0.281 | [+0.154, +0.398] | **0 %** |

**La conclusion est insensible à la décision d'inclusion** — c'est le meilleur cas
possible pour le rapporter : l'estimation bouge de 0.007 et l'hétérogénéité tombe à zéro.

**Mais la porte pré-enregistrée est unilatérale.** Le critère est `ceil_probe >= 0.50`,
il ne regarde que le **plafond**. shin2017a le passe haut la main (plafond 0.00) tout en
étant censuré à **64 % au plancher** (`width_end <= 8`) — le défaut symétrique de celui
qui a fait exclure bnci2014_001. Dispersion du prédicteur, qui est la quantité qui
compte vraiment :

| dataset | plafond | plancher | CV(#params) | statut |
|---|---|---|---|---|
| physionetmi | 0.18 | 0.33 | 0.213 | retenu |
| lee2019_mi | 0.34 | 0.15 | 0.160 | retenu |
| cho2017 | 0.26 | 0.06 | 0.192 | retenu |
| **shin2017a** | 0.00 | **0.64** | **0.070** | zone grise |
| bnci2014_001 | **0.91** | 0.00 | 0.031 | exclu |

shin2017a a 2.3× la dispersion de bnci2014_001 mais 2.3 à 3× moins que les trois
retenus. La porte telle qu'écrite l'aurait accepté **en silence** : c'est exactement le
défaut de [[gates-must-refuse-not-warn]], transposé au critère lui-même. À corriger en
un critère sur la dispersion (`CV` ou étendue relative), pas sur un seul bord.

## 03/09 ~13:55 — **extension de la couverture D×R** : `all_pool` sur lee2019_mi (**523644**) + matrice sur shin2017a (**523645**)

**Le trou qui est comblé.** L'inventaire par étage montrait une pyramide inversée : 4
datasets à l'étage 1, 4 à l'étage 2, 3 à l'étage 3, **2 à l'étage 4** — or c'est l'étage 4
(`all_pool`) qui porte la conclusion la plus large de la page (« l'effet de sélection est
un ordre de grandeur plus petit que le coût de restreindre le vivier »). Les 30/30
comparaisons sont 5 règles × 3 K × **2 datasets** : elles ne sont pas indépendantes et la
base réelle est n=2. C'est la faiblesse du dossier, davantage que l'exclusion de
bnci2014_001 ou l'échec du garde-fou sur cho2017.

| étage | cho2017 | physionetmi | lee2019_mi | bnci2014_001 | shin2017a |
|---|---|---|---|---|---|
| 1 — matrice D×R | ✅ | ✅ | ✅ | ✅ | **523645** |
| 2 — prédicteur | +0,150 | +0,385 | +0,229 | +0,048 (sonde saturée) | à suivre |
| 3 — intervention | garde-fou tombe | −0,038 | −0,020 | non lancé | non prévu |
| 4 — `all_pool` | ✅ | ✅ | **523644** | non lancé | non prévu |

**523644 — `all_pool` sur lee2019_mi.** Dimensionné au `--dry-run`, pas extrapolé : vivier
de **40–41 donneurs × 200 essais = ~8 000 essais/fit**, contre 7 800 pour cho2017 mais sur
1 000 pas de temps au lieu de 750, soit **1,37× la charge cho2017** (dont les shards ont
fini en 19 à 39 min sur 3 h allouées). D'où `--mem=32G` au lieu de 24 G. 32 cellules
(4 plis × 8 seeds, pas de réplicat), 4 shards, `--array=0-3`. Sortie
`/scratch/amounir/dsel_all/lee2019_mi`, séparée de `dsel2/` comme les deux autres.

`--gres=gpu:ampere:1` **forcé**, alors que `donor_all.sbatch` demande `gpu:1` sans type :
le twin `eegrow_shipped` (515936-515942) demande `gpu:turing:3` sur la même partition
`tau` et a déjà 3 tâches en PENDING. Une demande turing de plus se met dans sa file. Effet
de bord à consigner : lee2019_mi tournera donc sur du matériel **plus** homogène que
cho2017/physionetmi (qui se sont dispersés sur margpu002/003/006/009/016/028) — le
contraste `all_pool − règle` est interne à un dataset, donc cela ne biaise aucune
comparaison rapportée.

**523645 — matrice D×R sur shin2017a.** Le seul 5ᵉ dataset individuellement lisible :
**n=29 sujets → MDE ρ = 0,50**, quand les 7 autres datasets disponibles sont à n ≤ 14
(MDE ≥ 0,69) et ne pourraient qu'alimenter la méta-analyse. Porte la méta de 215 à ~244
donneurs. `LOCAL_FANOUT=1` avec `N_SHARDS=3` sur un seul nœud : la matrice classe des
donneurs **entre eux**, donc l'homogénéité matérielle est une condition du protocole, pas
un confort. `EEGROW_CUDA_FRACTION=0.30` × 3 tenants = 0,90 (le défaut du sbatch tient pour
3 shards, pas pour 4). `--cpus-per-task=12`, `THREADS=4`, `--mem=64G` — au-dessus des 48 G
du défaut, qui était calibré pour les 52 sujets de cho2017 alors qu'ici il y a 3 processus
sur le même nœud.

**Ce qui n'est PAS lancé, et pourquoi.** L'étage 3 ne s'étend pas aux datasets restants,
et la raison est structurelle et non budgétaire : sélectionner K=5 dans un vivier de 11
sujets, c'est K/M ≈ 0,45, en pleine zone de saturation où toutes les règles convergent —
c'est exactement l'effet mesuré en figure 03 (+0,043 → +0,022 quand K/M monte). Le
protocole y serait aveugle par construction.

**Vérifié avant soumission.** `ranking_lee2019_mi.csv` est **bit-identique** (md5
`f913d718…`) entre le checkout cluster, `benchmarks/analysis/` et le ré-export du 03/09 :
la troncature JSONL du 31/08 n'a pas touché lee2019_mi, l'étage 4 part donc de la même
sonde que l'étage 3 déjà publié. Cache `MNE-BIDS-shin2017-a` présent. Aucun des deux
répertoires de sortie n'existait.

**Statut à 13:57** : `523644_0` RUNNING sur margpu006, shards 1-3 PENDING(Resources),
`523645` PENDING(Priority).

**ETA MESURÉE à 14:05 (12:05 UTC cluster).** Chiffres pris dans le log en cours, pas
extrapolés du dimensionnement.

- `523644_0` : 1ʳᵉ cellule bouclée, `fit 357.6s` → **6,0 min/cellule**, 8 cellules/shard →
  shard ≈ **48 min**, fin vers **14:45**. Premier point : `f0__all_pool__d0__seed0`,
  8000 essais d'entraînement issus de 40 donneurs, **roc_auc 0,7828** sur 14 sujets
  held-out. Chargement des 54 sujets depuis le cache : 41 s.
- `523644_[1-3]` : PENDING(Resources) — toutes les cartes **ampere** sont prises (le
  `--gres=gpu:ampere:1` que j'ai forcé est aussi ce qui les bloque ; les nœuds `idle` de
  `gpu-best` sont des pascal, exclus par le template). Estimation Slurm **16:55**, mais
  c'est l'estimation pessimiste calée sur l'expiration du walltime 3 h du shard 0 ; le
  backfill devrait les prendre dès ~14:45. Collecte des 32 cellules : **fin d'après-midi**.
- `523645` : PENDING(Priority), estimation Slurm de démarrage **04/09 18:06**. La cause est
  interne : il est derrière **mes propres** `eegrow_shipped` (515937/38/39, priorité 7867
  contre 7246, walltime 2 jours chacun) sur la partition `tau`. Durée propre inconnue ;
  repère mesuré sur la matrice lee2019_mi (513725) : **9 unités par shard en 246 min** à
  54 sujets / 200 essais / 62 ch / 1000 pts. shin2017a a 29 sujets ; la charge réelle sera
  lisible sur la ligne `data ready` au démarrage.

## 03/09 14:15 — **audit GPU complet** : les 24 cartes « libres » n'existent pas, mais une partition entière était inexploitée

Déclencheur : `523644_[1-3]` PENDING(Resources) et `523645` annoncé à **04/09 18:06**.
`sinfo` montrait 5 nœuds `idle` — l'audit établit que c'était une illusion, et trouve
ailleurs de quoi tout débloquer.

### Ce que l'env peut réellement utiliser

`torch==2.13.0+cu130` de l'env `bench` embarque **sm_75, 80, 86, 90, 100, 120**. Donc :

| Type | CC | Utilisable | Cartes | Libres au 03/09 12:10 |
|---|---|---|---|---|
| **pascal** (P100) | sm_60 | ❌ | 19 (margpu024-027, 029) | 19 — **capacité morte** |
| **volta** | sm_70 | ❌ | 5 (margpu001, 004) | 0 |
| turing | sm_75 | ✅ | 23 | 5 (dont 2 sur nœuds exclus) |
| rtx | sm_75 présumé | ⚠️ non vérifié | 4 | 0 |
| ampere | sm_80/86 | ✅ | 22 | **2, partition `gpu`** |
| hopper | sm_90 | ✅ | 3 | 0 |

**Les 19 pascal sont inutilisables, c'est mesuré et non déduit.** Job `501434_0`
(`p-pascal`, margpu029, 26/08) : `Tesla P100-PCIE-16GB with CUDA capability sm_60 is not
compatible`, puis mort réelle sur `RuntimeError: GET was unable to find an engine to
execute this computation` dans `F.conv2d`. L'exclusion des pascal dans les templates est
donc juste et permanente tant qu'on ne réinstalle pas un build cu126.

**Piège évité au passage.** Relâcher `--gres=gpu:ampere:1` en `gpu:1` rouvrait aussi les
**volta** (margpu001, 004), sm_70, tout aussi incompatibles — et *absents* de la liste
`--exclude`, qui ne couvre que pascal + les cartes fantômes. D'où le passage par
`Features=ampere|turing|hopper`, qui exprime la vraie contrainte (sm_75+) au lieu de
l'énumération de nœuds.

### La trouvaille : la partition `gpu`

`scontrol show partition` : `gpu` = margpu[001,010-011], `AllowGroups=ALL
DenyAccounts=services-accounts`. Le compte `tau` y a donc accès, et **aucun template ne
l'a jamais visée**. margpu010 et margpu011 portent `ampere:2` avec **1 carte libre
chacune**, 88 et 80 CPU libres, 1030G et 500G de RAM.

**Pas d'exposition à la préemption**, contrairement à ce que `PreemptType=preempt/partition_prio`
+ `PriorityTier=1` (contre 100 pour `tau`) laisse craindre : la préemption par priorité de
partition suppose deux partitions partageant un nœud, or margpu[010-011] n'appartiennent
qu'à `gpu`. Aucune partition de tier supérieur ne peut les réclamer.

### Ce qui a été appliqué (sans resoumission, `scontrol update` sur jobs PENDING)

- `523645` → `Partition=tau,gpu`, **`gres/gpu:ampere:1` conservé**. Le protocole n'est pas
  touché : même classe de carte que les 4 autres matrices, et `LOCAL_FANOUT=1` garde les 3
  shards sur un seul nœud. Démarré sur **margpu011 à 12:12**, soit **~30 h gagnées** sur
  l'estimation Slurm.
- `523644_[1-3]` → `Partition=tau,gpu-best,gpu`, `TresPerNode=gres/gpu:1`,
  `Features=ampere|turing|hopper`. Les 3 shards ont démarré **dans les 16 s**.

**Résultat : 4 shards `all_pool` + la matrice shin2017a tournent en parallèle, file vide.**

| job | nœud | carte | démarré |
|---|---|---|---|
| 523644_0 | margpu006 | ampere | 11:55 |
| 523644_1 | margpu010 | ampere | 12:12 |
| 523644_2 | margpu018 | turing | 12:12 |
| 523644_3 | margpu019 | turing | 12:12 |
| 523645 | margpu011 | ampere | 12:12 |

Santé au démarrage vérifiée sur les deux nouveaux logs : `device=cuda`, aucun
avertissement de compatibilité, chargement du cache en cours.

### Hétérogénéité matérielle assumée sur `all_pool`, et pourquoi c'est acceptable

Les shards 0-1 sont sur ampere, les shards 2-3 sur turing. **Ce serait inadmissible sur la
matrice D×R** — elle classe les donneurs entre eux, une dérive systématique par carte
biaiserait le classement, d'où `LOCAL_FANOUT=1` maintenu. Ici le contraste est
`all_pool − règle`, et le bras « règle » de la campagne `dsel2` a *déjà* tourné sur du
tau/gpu-best mixte : mélanger côté `all_pool` rend les deux bras **plus** comparables, pas
moins. Ordre de grandeur : effets de 0,06 à 0,28 contre un écart numérique inter-cartes
~1e-4.

### Ce qui reste indisponible, et n'est pas récupérable

margpu020 (1 turing libre) et margpu021 (1 turing libre) restent exclus : cartes fantômes
mesurées (`515639_2` mort en 12 s le 02/09). Les rouvrir donnerait 2 cartes contre un
risque de requeue en boucle, pour une campagne qui se termine dans l'heure — non rentable.
Après application, **le cluster est saturé en sm_75+** : plus une seule carte utilisable
libre.

**Limite que ces deux jobs ne lèvent pas.** Les cinq datasets sont **tous du motor
imagery**, quatre sur cinq en binaire gauche/droite. La conclusion porte sur « un sujet en
imagerie motrice », pas sur « l'EEG », et aucune extension prévue ne change cela.

## 03/09 12:46 — les deux rapports sont **régénérés sur la grille complète**

| | avant (31/08, grille à 956) | après (03/09, grille à 1116) |
|---|---|---|
| cellules | 956 | **1 116** |
| scores (perf) | 34 741 | **48 888** |
| unités sujet | 8 591 | **11 808** |
| folds (dynamics) | 120 249 | **167 076** |
| opportunités de croissance | 498 160 | **542 187** |
| neurones ajoutés | 1 067 782 | **1 371 399** |
| figures | 52 + 32 | **52 + 32** (aucune ajoutée, aucune perdue) |

Chaîne exécutée : `rsync` des 1 116 CSV de score → `perf_final/scores/` ; SLURM **523597**
(`normal`, 8 CPU, 706 s) a relancé `export_growth_dynamics.py` sur les 12 Go de JSONL vers
`/scratch/amounir/gd/out_1116` — l'export du 31/08 dans `out/` est **conservé à côté**, pas
écrasé, pour que les figures publiées avant aujourd'hui restent rattachables aux frames qui
les ont produites. `gd_provenance.csv` sort avec le **même `eegrow_sha=58c6beebc12e`**
qu'au 31/08 : rien à réconcilier entre les deux exports.

**Un résultat change de signe de conclusion.** L'ancienne page disait « `grow − fix` va de
−0,0031 à +0,0081, **0 sur 6** survivent à Holm ». La grille complète dit **+0,0001 à
+0,0136, 2 sur 6** survivent, et `shallow` / cross-session sort à **+0,0136 [+0,0078,
+0,0197] sur 71 sujets**, intervalle franchement au-dessus de zéro. La croissance gagne
donc quelque part — mais l'écart type graine-à-graine sur la même grille est **0,0436**,
soit trois fois l'effet : c'est réel et plus petit que relancer la même configuration.

**Les deux pages ne peuvent plus mentir sur leur propre couverture.** Le bandeau et le
constat 05 disaient « campagne en cours » via une chaîne constante — vraie à l'écriture,
silencieusement fausse la nuit où la grille s'est terminée. Ils **comptent** maintenant
`final_grid.tsv` et affichent `n_scoré / n_planifié`.

### La page D×R était concernée elle aussi — et pas pour la raison attendue

`build_dxr_report.py` ne lit que des agrégats, mais les cinq scripts qui **produisent**
ces agrégats (`donor_matrix`, `donor_summary`, `donor_predictor`, `donor_select_prep`,
`donor_figures`) lisent `perf_final/scores` et `dynamics_final/gd_fits.csv.gz`. La
campagne finale entre dans D×R par **la sonde** `grow_shallow` / `within_session` /
`align=none` : `params_probe`, `acc_probe`, `ceil_frac`, et les `ranking_*.csv` qui ont
servi d'entrée à l'étage 3.

Le défaut n'est pas « il manquait des cellules ». **Aucune cellule bnci2014_001 n'est
nouvelle** — mêmes 3 graines. Mais un CSV de score s'écrit **une fois, à la fin** de la
cellule (absent ou complet), alors qu'un `__fits.jsonl` s'écrit **fold par fold** : l'export
du 31/08 a lu bnci2014_001/grow_shallow/within_session à **493 folds sur 810**, et une
cellule tronquée est indiscernable d'une cellule complète en aval.

| | ancien | nouveau |
|---|---|---|
| folds de la sonde bnci | 493 | **810** |
| ρ brut bnci | +0,285 | **+0,217** |
| **ρ partiel bnci** | +0,239 | **+0,048** |
| plafond de la sonde bnci | 93 % | 91 % |
| sonde des 3 autres datasets | — | **bit-identique** (delta 0 sur les 5 colonnes) |
| **ρ méta rapporté (n=215)** | +0,288 [+0,159, +0,412] | **inchangé, au chiffre près** |

Le garde-fou pré-spécifié `ceil_probe < 0.50` exclut bnci2014_001 (il est à 0,91) : le
défaut n'atteint donc pas l'analyse rapportée. **Sur 380 lignes de texte de la page,
exactement 3 nombres changent, tous sur la ligne bnci2014_001.** C'est l'argument pour
pré-spécifier le garde-fou avant de voir les données.

L'étage 3 (3 840 cellules GPU) est **sauf** : il tourne sur `ranking_cho2017`,
`ranking_lee2019_mi` et `ranking_physionetmi`, tous trois bit-identiques. Rien à relancer
sur le cluster.

**À refaire pour le jumeau `shipped` le jour où il tombe** : ré-exporter les réductions
même si la liste de cellules paraît inchangée, et diffuser les frames **par sujet**, pas
les décomptes de cellules — un décompte de cellules ne voit pas une liste de folds
tronquée.

## 03/09 09:00 — la campagne `final` est **TERMINÉE : 1116/1116**

Les 40 cellules schirrmeister2017 qui restaient hier sont passées dans la nuit. Plus
aucun job `final` en queue. Contrôle de santé : **0 CSV de moins de 1 ko**, 50 004 lignes
de scores, et la répartition par eval tombe exactement sur la grille émise —
648 `within_session` + 324 `cross_session` + 144 `cross_subject` = 1116.

C'est la table à partir de laquelle le papier est écrit, et elle remplace v4 et v5.
Conséquence immédiate sur le dépôt : la note du `.gitignore` qui exclut
`perf_final/scores/` invoque « une campagne ENCORE EN COURS (1082/1116 au 02/09) ». Cette
raison a expiré — la voie prévue est maintenant ouverte : publier un agrégat dans
`results_final_published/`, la grille étant terminée.

## 03/09 09:00 — jumeau `shipped` : **438/558**, mais le débit est tombé

| | |
|---|---|
| CSV écrits | **438 / 558** (242 within + 138 cross_session + 58 cross_subject), 0 vide |
| passes démarrées | 6 / 10 — `g1k3`, `g1k9`, `g2k6`, `g3k8`, `g3k9`, et la grosse `g3k10` (333 cellules) |
| passes jamais démarrées | `g3k1` (41), `g3k2` (13), `g3k6` (13), `g3k7` (13) = **80 cellules** |
| jobs | 4 COMPLETED, **2 RUNNING** (515941 margpu017 depuis 8 h 47 ; 515942 margpu022 depuis 53 min), **6 PENDING** |

**438 cellules en 10 h, puis 7 en 3 h.** Ce n'est pas une panne, c'est l'ordre de
passage : les petites cellules sont parties d'abord et l'early stopping tire vraiment à
`patience=20` (c'est toute la raison pour laquelle ce jumeau coûte 238 GPU-h et non 870).
Il ne reste que la queue lourde, et il n'y a que **2 nœuds** pour la traiter.

**Le goulot est le `--exclusive` sur turing**, comme prévu. margpu018/019/020 sont tenus
par `jjobard` (`distraq`, 12 h 42 à 16 h 36 restantes), margpu019 aussi par `strivaud`
(7 h 28), margpu028 par `tweikert` (9 h 59). margpu023 est `drained*` (« Feature test :
Not responding »), 021 exclu du sbatch. Aucun turing entier ne se libère avant ~7 h.

**ETA.** À capacité actuelle (2 nœuds, ~2,3 cellules/h sur la queue lourde) : ~50 h, soit
le **5 septembre en soirée**. Si les nœuds de jjobard se libèrent ce soir et que les
6 jobs PENDING démarrent : ~15-20 h, soit le **4 septembre**. Fourchette honnête :
**4-6 septembre**, inchangée depuis hier.

Vérification de portée refaite ce matin : `wc -l` sur les 10 passes de
`/scratch/amounir/passes_shipped` somme à **558**, et les logs confirment
`protocol='train.patience=20 train.selection_monitor=valid_loss'` et
`sha=58c6beebc12e` sur chaque allocation — le protocole et le checkout sont bien inscrits
par cellule, ce pour quoi le raccourci `PROTOCOL=shipped` avait été écarté.

## 02/09 — jumeau PROTOCOLE LIVRÉ soumis (jobs 515931-515942, tag `shipped`)

**Ce qu'il mesure.** « Le protocole livré RÉORDONNE les bras » est écrit dans
`final_grid.py`, `final_grid.sbatch`, `CAMPAIGNS.md` et `config.yaml`, et c'est
l'affirmation qui a justifié de jeter v5 et de dépenser 817 GPU-h. Sa seule preuve est
SLURM 500573 : **bnci2014_001, n=9 sujets, 8 bras, 2 graines, un seul eval**. Assez pour
justifier l'override sur notre propre campagne, pas assez pour dire au domaine que ses
classements publiés sont des artefacts de protocole. Ce jumeau produit **deux classements
complets sur 12 datasets** dont la seule différence est `patience` et `selection_monitor`.

| | corrigée (`final`) | jumeau (`shipped`) |
|---|---|---|
| protocole | `patience=200 selection_monitor=valid_acc` | `patience=20 selection_monitor=valid_loss` |
| cellules | 1116 (558 raw + 558 alignées) | **558 raw** — égalité d'ensembles vérifiée avec la moitié `align=none` |
| coût | 1740 GPU-h projetés | **238 GPU-h mesurés** (v5 a tourné ce protocole : pas de facteur de budget) |
| pire cellule | 98.1 h (extrapolée) | 22.5 h (mesurée) — `CELL_TIMEOUT=48 h` |
| code / cartes | `eegrow_budget` @ 58c6bee, turing | **identiques** — sinon le matériel entre dans la différence |

Triplet séparé : `results_shipped` / `logs/pack_shipped` / `eegrow_claims_shipped`.
Grille `benchmarks/slurm/shipped_grid.tsv`, wrapper `benchmarks/slurm/shipped_grid.sbatch`,
passes `/scratch/amounir/passes_shipped` (12 allocations, `G=` réinjecté à la main comme
pour `passes_final` — `plan_campaign.py` ne l'émet toujours pas).

**Les 12 sont PD (Priority) : zéro nœud turing libre.** 017/018/028 tenus par `final`,
019/020/022 par jjobard (9 jobs, 9 h 35 à 14 h 50 restantes). Départ attendu quand
jjobard rend ses nœuds ou quand `final` épuise sa grille. ETA fin de campagne :
**4-6 septembre**.

## 02/09 — campagne `final` : 1076/1116, **40 cellules restantes, toutes schirrmeister2017**
38 `within_session` + 2 `cross_subject`; surtout `grow_deep` et les `bd_*`. 103 CSV
écrits dans les 24 dernières heures, 7 claims actifs, logs vivants — la campagne avance,
c'est sa queue lourde (27 Go de RAM hôte par locataire, donc K=1/K=3).

## 02/09 19:15 — bras `all_pool` **collecté (64/64) et lu**. Ne pas sélectionner bat toutes les règles, partout.

**Le résultat, en une ligne : entraîner sur le vivier entier bat les cinq règles, aux
trois K, sur les deux datasets, avec les 4 plis concordants en signe dans chacune des
30 comparaisons. Aucune exception.**

|  | cho2017 (vivier 39) | physionetmi (vivier 81) |
|---|---|---|
| `all_pool` roc_auc/sujet | **0.7041** (n=52) | **0.8468** (n=109) |
| écart au meilleur K, meilleure règle | +0.045 (K=10, `acc_top`) | +0.060 (K=20, `params_top`) |
| écart au pire | +0.119 (K=3, `random`) | +0.285 (K=5, `params_bottom`) |

**Le mode de défaillance banal est exclu, et il l'est dans le bon sens.** L'addendum
imposait de lire la convergence AVANT les écarts : un pool 4× plus gros au même
`patience=200` qui perdrait n'aurait simplement pas convergé. Mesuré :

| | `restored_epoch` médiane [p10, p90] | `n_train` médian |
|---|---|---|
| cho2017 `all_pool` | 106 [36, 181] | 7 880 |
| cho2017 règles K=10 | 90 [32, 171] | 2 040 |
| physionetmi `all_pool` | 142 [76, 192] | 3 661 |
| physionetmi règles K=20 | 125 [58, 182] | 898 |

Toutes les cellules sortent sur `budget` (200 époques) des deux côtés, donc la
comparaison est symétrique. Le meilleur époque restauré est bien à l'intérieur du budget
en médiane. Le p90 de `all_pool` sur physionetmi est à 192/200 : une part des fits
progressait encore à la fin, ce qui **sous-estime** `all_pool`. Le biais résiduel joue
donc contre la conclusion qu'on tire, pas pour elle.

**La reformulation qui rend le résultat citable** (dérivée descriptive, post-hoc sur du
post-hoc, à présenter comme telle) : sélectionner à K fixé aide, mais ne récupère qu'une
fraction du déficit que créer le fait de sélectionner tout court.

| dataset | K | déficit de `random` vs `all_pool` | part récupérée par `params_top` | `resid_top` | `acc_top` |
|---|---|---|---|---|---|
| cho2017 | 3 | +0.1189 | 13.2 % | 12.8 % | **29.5 %** |
| cho2017 | 5 | +0.0961 | 10.8 % | 8.6 % | **37.3 %** |
| cho2017 | 10 | +0.0708 | **1.1 %** | **0.9 %** | **36.0 %** |
| physionetmi | 5 | +0.2344 | 28.5 % | 18.5 % | **34.6 %** |
| physionetmi | 10 | +0.1551 | 31.2 % | 36.1 % | **37.7 %** |
| physionetmi | 20 | +0.0862 | **30.7 %** | 25.0 % | 26.7 % |

Deux lectures, et la seconde est celle qui compte pour le papier :

1. `acc_top` récupère le plus dans 5 lignes sur 6. Sur cho2017 à K=10, les règles fondées
   sur la taille récupèrent **1 %** du déficit — c'est-à-dire rien — pendant que
   l'accuracy en récupère 36 %. C'est le troisième endroit indépendant où l'accuracy bat
   `#params` comme critère de sélection, après la rho partielle marginale (l'accuracy
   gagne sur 3 datasets/4) et l'endpoint primaire de claim 3.
2. **L'effet de sélection est d'un ordre de grandeur plus petit que le coût de
   sélectionner.** Le contraste entre critères que claim 3 mesurait vaut ~0.04 ; jeter
   des sujets coûte 0.06 à 0.28. La question « quel critère » est donc dominée par la
   question « pourquoi restreindre le pool », et c'est exactement la référence qui
   manquait au protocole.

**Ce qu'on ne peut PAS en dire** (rappel de l'addendum, gelé) : K n'est pas apparié, donc
chaque écart mélange « plus de données » et « meilleures données ». Aucun écart
`all_pool` − règle ne se lit comme un effet de critère. Aucune valeur p n'est produite.
Le verbe « confirme » reste interdit : ce bras recadre claim 3, il ne le renverse ni ne
le valide.

Sorties : `/scratch/amounir/dsel2/all_pool_{cho2017,physionetmi}.csv`.

---

## 02/09 18:30 — le bras `all_pool` manquant est lancé (**515638** cho2017, **515639** physionetmi)

**Ce qui manquait.** Sylvain avait nommé trois comparateurs le 01/09 : « tout le monde
dedans », « que les cinq meilleurs », « des sujets au hasard ». Les deux derniers étaient
dans les 5 règles depuis la v1 ; **le premier n'a jamais existé**. Cause technique :
`draw_candidates` plafonne M strictement sous la taille du vivier (`donor_select.py`,
`if m <= 0 or m >= len(pool)`) et `--k` n'est jamais allé au-delà de 20 sur des viviers
de 39 et 82. Sans ce bras, la seule phrase autorisée est « parmi les pools de taille K,
tel critère vaut mieux que tel autre » — jamais « sélectionner vaut mieux que ne pas
sélectionner », qui est la question du relecteur.

**Statut du bras : POST-HOC ET DESCRIPTIF.** Écrit après lecture de claim 3
(physionetmi P(K=5) = −0.0377, cho2017 garde-fou G non tenu). Déclaré comme tel dans
`analysis/PREREG_ADDENDUM_all_pool.md`. Il n'amende pas l'endpoint primaire, n'entre dans
aucune correction de multiplicité, et le verbe « confirme » lui est interdit. Il peut
recadrer l'interprétation de claim 3, jamais la renverser.

**Deux limites structurelles, à ne pas effacer en rédigeant.**
1. *K n'est pas apparié* : 39 ou 82 sujets contre 3 à 20. Tout écart mélange « plus de
   données » et « meilleures données ». Interdit d'en lire un effet de critère.
2. *Pas de réplicat* : le pool est déterministe étant donné le pli, sa seule variance est
   l'initialisation (8 seeds × 4 plis = 32 fits). Unité = le pli, F=4. Assez pour une
   référence, pas assez pour un test — donc on n'en fait pas un test.

**Dimensionnement, mesuré au `--dry-run`, pas extrapolé.**

| dataset | vivier | essais/fit | vs la plus grosse cellule de règle |
|---|---|---|---|
| cho2017 | 39 sujets × 200 | 7 800 | K=10 → 2 000, soit **3,9×** |
| physionetmi | 82 sujets × 45 | 3 690 | K=20 → 897, soit **4,1×** |

32 cellules par dataset, 4 shards, 8 cellules par shard, `--mem=24G` (au lieu de 16 G :
`pool_xy` concatène tout le pool en mémoire) et `--time=03:00:00`.

**Soumis à 18 h 26.** 515638 (cho2017) et 515639 (physionetmi), `--array=0-3` chacun.
7 shards sur 8 démarrés en < 1 min sur margpu002/003/006/009/016/028.

**Isolation.** Sortie `/scratch/amounir/dsel_all/<dataset>`, séparée de `dsel2/`. Nom au
format v1 (`f0__all_pool__d0__seed3.csv`) et nom de règle absent de `RULES` : **ni le
glob v1 ni le glob v2 ne peuvent ramasser ces cellules**. Analyse par
`analysis/donor_all_analysis.py`, qui n'imprime **aucune valeur p** et sort la
convergence (`epochs`, `stop_reason`) AVANT les écarts — parce qu'un pool 4× plus gros
avec le même `patience=200` qui perdrait n'aurait probablement pas convergé, et ce mode
de défaillance banal doit être exclu avant toute lecture.

**Piège consigné.** `run_cell` nomme la cellule sans le K
(`f{pli}__all_pool__d0__seed{seed}`). Deux tailles de pool écriraient donc le même
chemin et la reprise sauterait la seconde. Le bras n'a qu'une taille par construction ;
si ça change un jour, changer le nom d'abord.

**Mesure obtenue au passage — l'argument BNCI est chiffré.**

| dataset | étendue `#params` | ratio max/min | `ceil_frac` |
|---|---|---|---|
| physionetmi | 33 255 – 95 319 | 2,87× | 0,18 |
| cho2017 | 40 151 – 104 284 | 2,60× | 0,26 |
| lee2019_mi | 48 566 – 97 768 | 2,01× | 0,34 |
| **bnci2014_001** | 41 116 – 46 084 | **1,12×** | **0,93** |

Sur bnci2014_001 le prédicteur varie de 12 % au total contre un facteur 2 à 3 ailleurs.
Ce n'est pas un dataset difficile, c'est un dataset où la variable n'existe pas — et
c'est celui que Sylvain avait explicitement demandé (« prends-en un qui sature »). Il
voulait un dataset où l'**accuracy** sature ; c'est le **prédicteur** qui y sature. À lui
remonter tel quel : c'est le seul endroit où le protocole s'écarte de sa consigne.

---

## 02/09 12:50 (heure cluster) — claim 3 **collecté à 100 %** et **lu**. Le résultat est une réfutation.

Les 2400 cellules sont là : physionetmi 960/960, cho2017 **960/960**, lee2019_mi 480/480.
`514438_10` a rendu la dernière cho2017 à 12:46. Plus aucun job claim 3 en file.

L'analyse pré-enregistrée a donc été lancée sur les deux datasets de la famille, dans
l'ordre déclaré, **sans toucher au fichier de décision** (gelé depuis le 02/09 11 h 40).

### physionetmi — garde-fou OK, endpoint primaire significatif et NÉGATIF

| K | `resid_top` − `random` (a) | `params_bottom` − `random` (b) | G |
|---|---|---|---|
| 5  | **+0.0434** [+0.0277, +0.0592] | **−0.0504** [−0.0659, −0.0347] | OK |
| 10 | **+0.0561** [+0.0466, +0.0658] | **−0.0647** [−0.0784, −0.0517] | OK |
| 20 | **+0.0217** [+0.0150, +0.0284] | **−0.0421** [−0.0508, −0.0335] | OK |

**P (K=5) : `resid_top` − `acc_top` = −0.0377 [−0.0499, −0.0258], p < 1e-4, n = 64.**
Soit **−87 % de l'effet de sélection au même K**. Le contraste va donc dans le sens
opposé à l'hypothèse de départ, et il est grand : ce n'est pas un nul, c'est une
réfutation directionnelle, et elle se rapporte comme telle.

Secondaire S (Holm sur 3) : le contraste n'existe **qu'à K=5**. K=10 : −0.0024
[−0.0091, +0.0045], Holm = 1.00. K=20 : −0.0014 [−0.0057, +0.0029], Holm = 1.00.
Avec MDE = 0.0103 et 0.0068, ces deux-là sont des nuls *informatifs* (assez puissants
pour exclure un effet de la taille de celui de K=5), pas des non-réponses.

Prédiction T **tenue** : le gain sur `random` décroît de +0.0434 (K=5) à +0.0217 (K=20).
L'explication de l'échec de v1 par la saturation du vivier survit à son test.

### cho2017 — le garde-fou G ÉCHOUE aux trois K. (c) n'est pas lu.

| K | (a) `resid_top` − `random` | (b) `params_bottom` − `random` | G |
|---|---|---|---|
| 3  | +0.0152 [+0.0054, +0.0247] | **+0.0192** [+0.0064, +0.0325] | ÉCHOUE |
| 5  | +0.0083 [+0.0014, +0.0153] | +0.0021 [−0.0069, +0.0109] | ÉCHOUE |
| 10 | +0.0007 [−0.0058, +0.0071] | −0.0016 [−0.0080, +0.0046] | ÉCHOUE |

Ce n'est pas un manque de puissance : (b) est **significativement POSITIF** à K=3
(MDE 0.0191), c'est-à-dire que les donneurs les PLUS PETITS y font mieux que `random` —
le signe inverse de (b). Et à K=10 les deux contrastes sont plats avec MDE ≈ 0.0096.
Le classement de donneurs de v1 **ne transporte pas jusqu'à cho2017**.

**Conséquence, écrite avant de voir les chiffres :** cho2017 avait été choisi *parce
que* c'est le dataset qui discrimine (ρ(#params, acc) = +0.042 contre +0.519 sur
physionetmi), et ce choix est antérieur aux résultats. C'est donc le test le plus
sévère qui échoue, pas un dataset de complaisance. Holm sur la famille de 2 ne
s'applique pas mécaniquement ici : cho2017 ne produit aucun P (le garde-fou refuse
avant), il produit un **échec de précondition**. Le compte-rendu honnête est :
« sur physionetmi le contraste est significatif et défavorable à la taille ; sur
cho2017 la précondition même de la mesure ne se réplique pas ». Rapporter le
physionetmi seul serait exactement la sélection sur le résultat que la
pré-registration interdit.

À NE PAS ÉCRIRE (interdits littéraux du prereg) : « la taille gagne »,
« l'accuracy gagne », « c'est équivalent », « indiscernables ».

### Campagne : 1102/1143 (96,4 %), 41 restantes, **toutes sur schirrmeister2017**

L'anomalie de la veille est **résolue d'elle-même** : g3k10 (666/666) et orphans
(27/27) sont complètes. Il ne reste que `grid_g1k3` à 11/52 — 39 `within_session` et
2 `cross_subject`, toutes sur schirrmeister2017, le dataset le plus lourd de la grille
(14 sujets, 128 canaux, 1000 échantillons).

**Coût par cellule mesuré, et il est bien pire que mon estimation d'hier.** Je comptais
4,4 h/cellule ; c'était le débit du worker, pas le coût de la cellule. `509153` occupe
**3 créneaux simultanés** (EEGROW_CUDA_FRACTION=0.283, 3 GPU) : il a rendu 11 cellules
en 44,2 h, soit 3 × 44,2 / 11 ≈ **12 h de mur par cellule**. Les trois cellules en vol
sur margpu018 tournent depuis 01:00, 04:57 et 07:09 et ne sont pas finies.

| créneaux | ETA sur 41 cellules |
|---|---|
| 6 (état actuel : `509153` + `512168` depuis 11:45) | ≈ 82 h, **3,4 jours** |
| 15 (si les 3 `final_g1k3` PENDING démarrent) | ≈ 33 h, **1,4 jour** |

`509153` a 4 j 16 h de reste, `512168` 6 j 23 h : les deux tiennent le scénario à 6
créneaux sans requeue. Les trois `final_g1k3` (514212 Resources, 514213/514214
Priority) attendent toujours un nœud turing entier — le verrou reste
`--gres=gpu:turing:3` + `--exclusive`, pas la priorité.

**Arbitrage toujours ouvert** (inchangé, je ne le tranche pas seul) : ouvrir la
campagne à ampere/rtx libérerait ces 9 créneaux tout de suite, au prix d'un mélange de
générations de GPU au milieu d'un benchmark comparatif.

## 02/09 11:50 (heure cluster) — rattrapage claim 3 **réussi**, comptabilité campagne enfin reproduite

### Claim 3 : 2 datasets sur 3 sont à leur n pré-déclaré

| dataset | cellules | cible | état |
|---|---|---|---|
| physionetmi | **960** | 960 | **complet** — les 12 shards de `514428` COMPLETED |
| lee2019_mi | **480** | 480 | **complet** — les 13 shards de `514439` COMPLETED |
| cho2017 | 663 | 960 | `514208_4` en vol (77/120 à 33 min), puis `514438` (13 shards, `afterany`) |

ETA cho2017 ≈ **40-50 min** : ~18 min pour finir `514208_4` (0,43 min/cellule mesurée),
puis 13 shards à ~20 trous chacun ≈ 10 min, et la file gpu-best a 11 GPU libres.
**physionetmi et lee2019_mi sont lisibles maintenant** — mais on ne lit rien avant
cho2017 : Holm sur la famille de 2, et rapporter physionetmi seul serait la sélection
sur le résultat que la pré-registration interdit explicitement.

### Un 17ᵉ shard mort, et le ledger corrigé

`514208_5` est mort sur margpu021 après le comptage précédent. Le ledger passe de
16 shards / 960 cellules à **17 shards / 1020 cellules** : 514206 → 4,5,6,7 ;
514207 → 2,5,6,7 ; **514208 → 0,5,7** ; 514209 → 0,2,3,4,6,7. `514208` avait été
soumis avant le correctif, donc ses shards atterrissent encore sur margpu021 — et
`514208_3` y a fini en 19 min pendant que `_5` y mourait en 13 s. La carte fantôme,
encore.

### Campagne : 1096/1143 (95,9 %), et le chiffre est enfin reproductible

Le « 1143 » n'était pas faux, c'est ma façon de compter qui l'était : il faut résoudre
`results_final/<split>/<dataset>/<stem>.csv` **grille par grille**, pas faire l'union
des clés (les 1280 clés et les 638 orphelins venaient de là).

| grille | fait / total |
|---|---|
| g1k3 | **11 / 52** |
| g3k10 | 663 / 666 |
| orphans | 24 / 27 |
| g1k9, g2k6, g3k1, g3k2, g3k6, g3k7, g3k8, g3k9 | complets |

**Reste 47 cellules, dont 41 sur la seule grille g1k3** : c'est tout le chemin critique.

### Correction : `512163-512167` n'étaient pas bloquées, elles ont fini en 32 s

Elles ont démarré sur margpu017 et se sont terminées `COMPLETED` en 32-38 s — c'est le
comportement correct d'un pack dont toutes les cellules ont déjà un CSV, pas un échec.
Mon diagnostic précédent (« 9 CPU + 120 G qui ne rentrent pas ») était faux. **Le vrai
verrou est `--gres=gpu:turing:N` + `--exclusive`** dans `final_grid.sbatch` : la
campagne exige le type turing et un nœud entier, or les 5 nœuds turing tau sont pleins
(nous sur 017/018, **jjobard** sur 019/020/022/028) pendant que 11 GPU rtx/ampere/hopper
sont libres sur gpu-best. Les jobs de jjobard ont entre 58 min et 18 h de reste.

**Reste à arbitrer (Adam)** : ouvrir la campagne à ampere/rtx ferait tomber 3 workers de
plus sur g1k3 tout de suite, mais mélangerait les générations de GPU au milieu d'un
benchmark comparatif. Ce n'est pas un choix d'ordonnancement, c'est un choix de
provenance — je ne le fais pas seul.

### ETA campagne

g1k3 tourne à **4,4 h/cellule** par worker (11 cellules du 31/08 10 h 55 au 02/09
07 h 09). Deux workers actifs (`509153` sur margpu018, `512168` sur margpu017) →
41 × 4,4 / 2 ≈ **3,8 jours**. Avec les 3 `final_g1k3` en attente, qui démarreront quand
jjobard libère les turing (≤ 18 h) → **1,5 à 2 jours**. `509153` a 4 j 17 h de reste :
ça tient, mais seul il finirait à ~25 cellules sur 41.

**Anomalie non résolue** : g3k10 (3 manquantes) et orphans (3) ont vu leur pack sortir
`COMPLETED` en 32 s sans les calculer. Soit un garde-fou les refuse, soit elles sont
hors grille effective. `514215` (`final_orph`) est en attente et couvrirait les
orphelines. À élucider avant de déclarer la campagne complète — cf.
[[gates-must-refuse-not-warn]].

---

## 02/09 14:05 — **16 shards tués par une carte fantôme sur margpu021**, rattrapage soumis

### Le diagnostic, avant le correctif

`sacct -X` sur les cinq arrays de claim 3 : **16 shards FAILED, tous en 12-22 s, tous
sur margpu021, et aucun ailleurs.** Même erreur mot pour mot dans les trois logs
inspectés, levée par `pick_device` (`benchmarks/utils.py:49`) :

    RuntimeError: CUDA_VISIBLE_DEVICES='0' but torch.cuda.is_available() is False:
    the pinned device does not exist. Refusing to fall back to CPU.

`scontrol show node margpu021` : `CfgTRES=...,gres/gpu=3` pour **2** cartes turing
réellement utilisables. SLURM place donc des jobs sur une 3ᵉ carte qui n'existe pas.
Ce n'est pas un nœud en panne — `514207_1` a fini ses 120 cellules **sur margpu021**,
et `514209_1` y tourne encore : les deux vraies cartes marchent. C'est la 3ᵉ qui ment.

Le garde-fou a fait exactement son travail : il a **refusé le CPU**. Zéro cellule
calculée sur le mauvais matériel, zéro CSV de mauvaise provenance. La perte est propre.

Le round-robin (`donor_select.py:547`) répartit les cellules modulo `n_shards`, donc
les trous sont étalés sur tous les K et toutes les règles : c'est une perte de
**puissance**, pas un biais.

| array | dataset | shards morts | cellules perdues |
|---|---|---|---|
| 514206 | cho2017 reps 0-7 | 4,5,6,7 | 240 |
| 514207 | physionetmi reps 8-15 | 2,5,6,7 | 240 |
| 514208 | cho2017 reps 8-15 | 0,7 | 120 |
| 514209 | lee2019_mi | 0,2,3,4,6,7 | 360 |
| | | **16 shards** | **960** |

### Coûts MESURÉS (COMPLETED), pas extrapolés

| dataset | s/cellule | source |
|---|---|---|
| physionetmi | 20 s | 514187, 8 shards × 60 cellules en ~20 min |
| **cho2017** | **28 s** | 514206_2, 60 cellules en 28:12 |
| lee2019_mi | ~58 s | 514209_1/5, 60 cellules en ~29 min |

Les 28 s de cho2017 **remplacent l'extrapolation à 44 s** que j'avais faite sur le
nombre d'essais. Rattrapage complet ≈ **9 h GPU**.

### Le correctif, et pourquoi il ne resoumet PAS les mêmes shards

`slurm/donor_select2.sbatch`, deux lignes :

1. `--exclude` : `margpu021` ajouté (il n'y était pas — l'exclusion ne couvrait que les
   pascal sm_60, une tout autre raison).
2. `--partition=tau,gpu-best` au lieu de `gpu-best`. Mesuré : tau a
   **PriorityTier=100 contre 1**, `PreemptType=preempt/partition_prio`, et je suis dans
   le groupe `tau`. Ce qu'il ne faut PAS en attendre : les nœuds turing tau sont tenus
   par **jjobard, tau lui aussi** — même tier, donc **pas de préemption**. Le seul nœud
   tau préemptible aujourd'hui est margpu008 (2 jobs gpu-best d'un autre utilisateur).
   Le gain vient de la priorité, pas de la préemption. Il a payé tout de suite :
   `514428_0` a démarré **en 12 s sur margpu017, partition tau**.

Resoumettre les indices 2,5,6,7 aurait reproduit le même découpage. `run_cell` saute
toute cellule dont le CSV existe (`donor_select.py:300`), donc on relance le **plan
complet** avec `N_SHARDS=12` : le round-robin redistribue les trous en ~20 par shard au
lieu de blocs de 120, et les jobs deviennent assez courts pour passer en backfill.
`--dependency=afterany` là où un array du même dataset tourne encore : deux processus
sur la même cellule écriraient le même chemin en concurrence.

| job | dataset | array | trous | attente |
|---|---|---|---|---|
| `514428` | physionetmi | 0-11 (`N_SHARDS=12`) | 240 | aucune, RUNNING |
| `514438` | cho2017 | 0-12 (`N_SHARDS=13`) | ~180 après 514208 | `afterany:514208` |
| `514439` | lee2019_mi | 0-12 (`N_SHARDS=13`) | 360 | `afterany:514209` |

`--exclude` étendu à `margpu007` pour 514428 seulement : trois shards de 514208 y
tournent, et un job tau pourrait les préempter.

**`N_SHARDS=12` était un mauvais choix, et le job l'a dit tout seul.** `514428_0` a
terminé en 1 min avec « 80/80 cells, 0.0 min » : **zéro trou**. Raison arithmétique,
pas matérielle — `gcd(8, 12) = 4`, donc le nouveau shard *s* ne recouvre que les anciens
shards *s* mod 4 et *s* mod 4 + 4. Les trous ne se redistribuent pas, ils se
**concentrent** : 0 pour *s* ≡ 0, 80 pour *s* ≡ 2. Pour redistribuer, le nouveau modulo
doit être **premier avec l'ancien**. 514430/514431 ont donc été annulés avant de démarrer
(dépendance, aucun calcul perdu) et resoumis en `N_SHARDS=13`. 514428 est laissé tel
quel : son pire shard fait 80 cellules ≈ 27 min, soit ce qu'aurait coûté le rattrapage
naïf — rien à gagner à le relancer.

**Conséquence de tau à signaler : on a préempté un collègue.** `514428_1` et `_2` ont
démarré sur margpu008 en requeuant `513328` et `513329` de **gblayer** (gpu-best,
~50 min de calcul chacun, retour en PENDING). C'est la sémantique voulue de la partition
tau (tier 100 vs 1), pas un bug, et nos jobs n'y tiennent que quelques minutes — mais
c'est un coût imposé à quelqu'un. Opt-out en une ligne si on ne veut pas : remettre
`--partition=gpu-best` ou ajouter `margpu008` à `--exclude`.

### Effet sur la lecture des verdicts

Rien ne change au protocole. L'endpoint pré-déclaré reste **n=64** (16 réplicats × 4
plis) sur physionetmi ET cho2017, Holm sur la famille de 2, lee2019_mi hors famille.
Le rattrapage sert précisément à ne pas lire à n=32 d'un côté et n=64 de l'autre.

### Campagne benchmark : ce qui bloque vraiment

`509153` tourne depuis 2 j 05 h sur margpu018 (4 j 19 h restants). Les dix autres
(`512163-512168`, `514212-514215`) sont PENDING, et la raison n'est pas la priorité :
elles demandent **9 CPU + 120 G** sur des nœuds turing qui n'ont que 32 CPU et sont
déjà à 24/32. `514428_0`, qui demande 4 CPU / 16 G, s'est glissé sur margpu017 en
12 secondes **devant elles**. À arbitrer : redimensionner ces demandes.

Le « 50 cellules restantes sur 1143 » du 02/09 12:15 **n'est toujours pas reproductible**
depuis le disque (1280 clés de grille, 431 avec résultat, 638 résultats sans ligne de
grille). Il faut passer par la comptabilité de `pack_run` elle-même.

## 02/09 12:15 — **CLAIM 2 COMBINÉE : +0.288 [+0.159, +0.412]**, et 6 arrays en vol

### Le résultat demandé en premier — coût GPU : ZÉRO

Les 4 matrices D×R étaient déjà **complètes** sur disque (327 + 162 + 156 + 27
cellules) : ce qui manquait n'était pas du calcul, c'était de les **combiner**.
`benchmarks/analysis/donor_meta.py`, écrit et tourné ce midi.

| | rho partielle | IC95 | p | n |
|---|---|---|---|---|
| **primaire** (datasets fixes, bootstrap donneurs stratifié) | **+0.288** | [+0.159, +0.412] | 1e-4 | 215 |
| secondaire (Fisher-z, effets fixes) | +0.294 | [+0.164, +0.414] | <1e-4 | 215 |
| sans exclusion (les 4 datasets) | +0.284 | [+0.155, +0.408] | 1e-4 | 224 |

**Hétérogénéité Q(2) = 2.45, p = 0.29, I² = 19 %.** C'est le chiffre important, et pas
le +0.288 : il dit *formellement* que les quatre estimations par dataset (+0.385,
+0.229, +0.150, +0.239) sont compatibles avec **un seul effet commun**. Le désaccord
apparent entre datasets est intégralement expliqué par le bruit d'échantillonnage —
[[underpowered-not-null]] démontré par un test, plus par un argument.

Analyse primaire = datasets **fixes**, pas un effet aléatoire : 4 datasets ne sont pas
un tirage dans une population de datasets, et un tau² estimé sur k=4 n'a aucune
précision. Corollaire à écrire dans le papier : cette estimation **ne généralise pas à
un 5ᵉ dataset**, elle répond à « l'effet est-il présent sur ces quatre-là, vus
conjointement ». L'exclusion de bnci2014_001 est pré-spécifiée sur `ceil_probe ≥ 0.50`
(93 % de censure : le prédicteur y est une borne, étendue 41–46 k contre 40–104 k), un
critère qui porte sur la **sonde** et pas sur le résultat ; les deux analyses sont
imprimées et concordent.

### Six arrays soumis

| job | quoi | cellules | statut |
|---|---|---|---|
| `514187_[0-7]` | claim 3 v2 physionetmi, reps 0-7 | 480 | RUNNING (81 faites à 11:15) |
| `514206_[0-7]` | **réplication cho2017**, M=20, K∈{3,5,10} | 480 | PENDING |
| `514207_[0-7]` | physionetmi reps 8-15 | +480 | après 514187 |
| `514208_[0-7]` | cho2017 reps 8-15 | +480 | après 514206 |
| `514209_[0-7]` | lee2019_mi, **exploratoire** | 480 | PENDING |
| `514212-514215` | campagne, 4 allocations `turing:1` coopérantes | 50 restantes | PENDING |

**Le doublement de réplicats (514207/514208) est déclaré MAINTENANT**, avant que le
script de verdict ait tourné sur la moindre cellule dsel2. La raison est quantitative :
à n=32 le MDE vaut 0.0107 alors que la marge d'équivalence est à 0.0127 — on ne peut
quasiment pas distinguer « effet » de « équivalent ». À n=64 le MDE tombe à 0.0076 et
la question devient tranchable. Décider ça *après* avoir vu le résultat à n=32 serait
de l'arrêt optionnel ; les jobs sont donc chaînés en `--dependency` dès maintenant.

### Réplication cho2017 : pré-registration séparée, endpoint INCHANGÉ

`analysis/PREREG_donor_select2_cho2017.md`, écrite avant la première cellule.
`donor_select2_analysis.py` n'est **pas** modifié — les cellules physionetmi y tombent
en ce moment, et un fichier de décision qu'on édite pendant que les données arrivent
n'est plus une pré-registration.

Ce qui se conserve d'un dataset à l'autre n'est ni M ni K mais leurs **ratios** :
M/vivier 49 % → 51 %, K/M 12/25/50 % → 15/25/50 %. Recouvrement `resid_top` ∩
`acc_top`, mesuré au `--dry-run` :

| | K le plus bas | K moyen | K le plus haut |
|---|---|---|---|
| physionetmi | 7 % (K=5) | 29 % (K=10) | 56 % (K=20) |
| **cho2017** | 1 % (K=3) | **11 % (K=5)** | 52 % (K=10) |
| lee2019_mi | 4 % (K=3) | 28 % (K=5) | 62 % (K=10) |

L'endpoint primaire reste **K=5**, celui de l'original, alors que cho2017 aurait un
recouvrement plus faible à K=3. Choisir le K le plus favorable dataset par dataset
ferait deux expériences à deux endpoints, qui ne se répliquent pas l'une l'autre.
Multiplicité : Holm sur la famille des 2 (physionetmi, cho2017), le résultat annoncé
est le pire des deux après correction. lee2019_mi est **exploratoire** et hors famille
— avec ρ(params, acc) = +0.647 les deux règles y sont largement confondues (28 % de
recouvrement à K=5 contre 11 %), donc l'inclure dans la famille confirmatoire diluerait
la correction pour presque aucune puissance ; il sert à tester (a)+(b) sur un 3ᵉ dataset.

### Campagne : le diagnostic, qui n'est pas celui que je croyais

**Elle est à 95,6 % : 50 cellules restantes sur 1143**, dont **39 sur
`within_session/schirrmeister2017`** (+ 6 bnci2014_001, 3 bnci2015_001, 2
cross_subject/schirrmeister2017).

Les 6 jobs PENDING depuis 22 h ne sont pas en retard dans la file : **il n'y a zéro GPU
turing libre sur tout le cluster**. Un autre utilisateur (`jjobard`) tient les 20 cartes
turing avec 20 jobs mono-GPU de 24 h, lancés entre 3 h et 20 h 36 avant maintenant.
`--exclusive` demande un nœud entier, et un pool que quelqu'un fragmente en jobs
mono-GPU ne libère jamais de nœud entier : c'est de la famine structurelle, attendre ne
la résout pas.

**Ce que je n'ai PAS fait, et pourquoi :**

- *Retirer `--exclusive`* — il est là pour une raison mesurée : `--mem` n'est pas
  appliqué sur ce cluster (les nœuds turing rapportent `AllocMem=0`), SLURM avait
  co-planifié deux allocations 120 G sur margpu020 pendant v5, et l'arbitre est devenu
  l'OOM killer du noyau : **85 cellules tuées**. MaxRSS mesurée de 509153 = 28 G, et
  schirrmeister demande 27 G de RAM hôte par locataire. On ne touche pas à ça.
- *Basculer sur les H100/ampere libres* — casserait l'appariement pour `bd_shallow` :
  `grow_shallow` schirrmeister est **déjà fait sur turing**, donc la différence
  `grow_shallow − bd_shallow` aurait une différence de matériel dedans, exactement ce
  que l'en-tête de `final_grid.sbatch` interdit. **Décision d'Adam, pas la mienne.**
  Blocs qui pourraient bouger sans casser d'appariement (paire entière restante) :
  la famille sccnet (`bd_sccnet` + `fix_sccnet` + `grow_sccnet`, 18 cellules) et
  `grow_deep` + `bd_deep4` (12 cellules). Pas `bd_shallow`, pas `fix_deepeeg__easubject`
  (son jumeau `raw` est déjà sur turing).

**Ce que j'ai fait** : 4 allocations `turing:1` coopérantes de plus (514212-514215), en
recopiant à l'identique une ligne que `submit.sh` émet déjà. Elles partagent le
répertoire de claims atomique — mécanisme prévu pour ça, `pack_run.sh` documente
explicitement que plusieurs allocations coopèrent. 3 sur `grid_g1k3` (couvre les 41
cellules schirrmeister), 1 sur `grid_orphans` (les 9 bnci, qui **n'ont jamais été
soumises** — `grid_orphans.tsv` n'apparaît nulle part dans `submit.sh`). Au lieu
d'attendre un nœud à 3 GPU, on prend les nœuds au fur et à mesure.

**À vérifier avant d'y toucher** : `nvidia-smi` dans l'allocation 509153 montre 2 cartes
dont une à 1 MiB / util N/A. Ça *ressemble* à un GPU inutilisé qu'on paie depuis 2 jours,
mais [[cuda-ordinal-vs-nvidia-smi]] dit exactement de ne pas le déduire — il faut sonder
`mem_get_info` par ordinal CUDA. Rien tenté : un faux pas ici tue une cellule à 98 h dont
51 h sont déjà investies.

**À décider** : 512163-512167 demandent `turing:3 --exclusive` pour 7 jours et **toutes
leurs cellules sont déjà faites** — s'ils démarrent, ils balaient, ne réclament rien et
sortent, après avoir monopolisé des nœuds entiers. Les annuler libérerait de la priorité
fairshare. Je ne les touche pas sans ton accord.

## 02/09 11:10 — claim 3 v2 **SOUMISE** : array `514187_[0-7]`

Sonde **514183** revenue (margpu006, 1 min 50, MaxRSS 3.09 G). Temps de fit **mesurés**,
un point par K, pas une extrapolation :

| K | essais d'entraînement | fit | cellules |
|---|---|---|---|
| 5 | 225 | 12.3 s | 160 |
| 10 | 450 | 15.3 s | 160 |
| 20 | 897 | 29.3 s | 160 |

Total = 160 × 56.9 s ≈ **2 h 30 de GPU cumulé**. Le chargement des données (52 s) est
payé une fois par shard, le scoring des ~27 sujets held-out est dans le bruit.

Soumis : `N_SHARDS=8 sbatch --array=0-7 --time=01:00:00` → **514187**, 60 cellules par
shard (round-robin, donc chaque shard porte autant de K=5 que de K=20) ≈ **21 min**.
1 h demandée = 3× la marge, ce qui la rend éligible au backfill au lieu de la mettre
derrière les jobs longs. 514187_0 et _1 tournent déjà sur margpu003. Les 3 cellules de
la sonde sont reprises telles quelles (saut par CSV existant), il en reste 477.

Le verdict s'imprime avec `analysis/donor_select2_analysis.py`, déjà fumée-testée sur
480 cellules synthétiques : les 3 branches (positif / équivalent / indéterminé) sortent.

### Réplication hors physionetmi — classements construits, protocole **à redimensionner**

`ranking_cho2017.csv` et `ranking_lee2019_mi.csv` construits (coût nul : ils dérivent de
`dynamics_final` + `perf_final`, aucun GPU) et déposés sur le cluster. Mais ils ne se
lancent **pas** avec les paramètres de physionetmi :

| dataset | n | vivier (4 plis) | ceil_frac méd. | ρ(params, acc) |
|---|---|---|---|---|
| physionetmi | 109 | 82 | 0.20 | +0.519 |
| lee2019_mi | 54 | 40 | 0.30 | +0.647 |
| cho2017 | 52 | 39 | 0.04→ | **+0.042** |
| bnci2014_001 | 9 | 7 | **0.96** | +0.159 |

- `M=40` sur un vivier de 39-40 rend **tout le vivier** candidat → on retombe exactement
  sur le défaut de v1 qu'on vient de corriger. Il faut M ≈ 20, et K=20 devient
  impossible (le contrôle de saturation disparaît).
- **bnci2014_001 est disqualifié deux fois** : vivier de 7 sujets, et `ceil_frac` médian
  à 0.96 — `params_probe` y est une borne (plafond de largeur 40) et non une mesure.
- **cho2017 est le dataset discriminant** : ρ(params, acc) ≈ 0, donc `params_top` et
  `acc_top` y sont quasi décorrélées et le contraste (c) y a un levier maximal — là où
  lee2019_mi (ρ = +0.65) les confond largement.

Rien n'est lancé là-dessus : changer (M, K) change l'endpoint, donc ça demande une
pré-registration à part, pas une improvisation en ligne de commande.

## 02/09 09:00 — claim 3 **v2** prête, sonde de coût **514183** en vol

Le correctif du défaut de v1 (1 pool par pli → contraste à 4 réplicats). Code :
`donor_select.py` (section « V2 »), `analysis/donor_select2_analysis.py`,
`slurm/donor_select2.sbatch`. Sortie séparée `/scratch/amounir/dsel2/physionetmi`.

**Le correctif** : `--candidates 40 --reps 8`. On tire 40 candidats par (pli, réplicat),
**partagés par les 5 règles**. Les règles déterministes acquièrent donc une distribution
de pools (unité de rééchantillonnage = 4 plis × 8 réplicats = **32**), et deux règles
d'un même réplicat sont comparées sur le même ensemble de sujets disponibles — la
différence mesurée est « quel critère a choisi quoi », plus « qui était disponible ».

**`--k 5 10 20`**, pools **emboîtés** en K (l'ordre est calculé une fois par
(pli, réplicat, règle), les K sont des préfixes — `random` compris, via une permutation
tronquée). Recouvrement mesuré au `--dry-run`, avant tout GPU :

| `resid_top` ∩ `acc_top` | K=5 | K=10 | K=20 |
|---|---|---|---|
| | **7 %** | 29 % | 56 % |

K=5 est donc le seul régime où le contraste a de la prise → **endpoint primaire déclaré
à K=5**, avant les données et sur un critère qui n'en contient aucune. Recouvrement des
candidats entre réplicats : **49 %** — de vrais tirages, pas des copies.

**Pré-registration v2** (`donor_select2_analysis.py`, écrite avant les données) :
garde-fou G = (a)+(b) doivent se répliquer à chaque K, sinon on ne lit pas (c) ;
primaire P = `resid_top` − `acc_top` à K=5, bootstrap **des réplicats stratifié par pli**
(les 109 sujets sont le dataset entier, donc fixes ; le seul hasard est le tirage du
pool) ; secondaire = les 3 K avec Holm ; **prédiction T** = le gain sur random doit
DÉCROÎTRE avec K — c'est mon explication de l'échec de v1, et si elle est fausse le
script l'écrit ; équivalence déclarée seulement si un TOST passe à ±25 % (et non « non
significatif donc pareil », l'erreur du matin).

**Vérifications faites avant soumission :** les 3 branches du verdict (positif, TOST
équivalent, indéterminé) exercées sur données synthétiques, 480 cellules → MDE 0.0107 à
n=32, effet injecté de +0.012 récupéré à +0.0095 ; le mode v1 **reproduit au bit près**
les 40 pools distincts des 72 cellules publiées (`select` garde `rng.choice` pour
`random`, qui ne consomme pas le générateur comme la permutation v2) ; l'analyse refuse
de tourner si deux règles d'un même réplicat n'ont pas vu les mêmes candidats.

**Sonde 514183** = shard 0 sur 160, soit exactement 1 cellule par K, pour mesurer le coût
avant de dimensionner l'array. 480 cellules au total. **L'array n'est PAS soumis.**

## 02/09 08:45 — **CLAIM 3 EST MESURÉ.** (a) et (b) passent, (c) non.

Job 514161, 6 shards, 72 cellules, **16 min de mur**, 0 erreur. 109 sujets de test,
`bd_shallow` aval fixe, K=20, 4 plis. Analyse : `donor_select_analysis.py`.

| contraste (apparié, n=109) | delta | IC 95 % | Holm | MDE |
|---|---|---|---|---|
| `resid_top` − random | **+0.0390** | [+0.0314, +0.0464] | <1e-4 | 0.0108 |
| `acc_top` − random | +0.0367 | [+0.0262, +0.0471] | <1e-4 | 0.0151 |
| `params_top` − random | +0.0315 | [+0.0216, +0.0414] | <1e-4 | 0.0143 |
| `params_bottom` − random | **−0.0516** | [−0.0653, −0.0379] | <1e-4 | 0.0197 |
| `params_top` − `acc_top` | −0.0051 | [−0.0153, +0.0053] | — | 0.0147 |
| `resid_top` − `acc_top` | +0.0023 | [−0.0090, +0.0136] | — | 0.0161 |

**(a) OUI** — choisir les donneurs bat l'aléatoire, franchement. **(b) OUI** — le
contrôle négatif s'inverse proprement (écart top−bottom **+0.083**), donc l'effet est
bien attribuable au tri et pas à autre chose qui bougerait en même temps.
**(c) NON TRANCHÉ** — et voir la correction ci-dessous, ce n'est PAS « indiscernables ».

### CORRECTION 02/09 (après-midi) — j'avais sur-conclu sur (c)

Deux défauts dans ma lecture du matin, trouvés en ré-auditant après question d'Adam.

**1. Le contraste (c) n'a que 4 réplicats, pas 109.** Les règles déterministes ne tirent
**qu'un seul pool par pli** : les 27 sujets d'un pli partagent exactement les mêmes 20
donneurs. La variance « sur quel pool cette règle est-elle tombée » est donc entièrement
confondue avec le pli, et le bootstrap sur les sujets ne la voit pas — même faute de
niveau d'analyse que `unit-of-analysis-subject`, transposée du sujet au pool. Les seeds
(×3) ne font varier que l'initialisation, pas le pool.

| contraste, par pli | pli 0 | pli 1 | pli 2 | pli 3 | signes |
|---|---|---|---|---|---|
| `resid_top` − `acc_top` | +0.0080 | +0.0128 | +0.0075 | **−0.0192** | 3/4 |
| `params_top` − `acc_top` | +0.0019 | +0.0007 | −0.0022 | **−0.0213** | 2/4 |
| `resid_top` − random | +0.0355 | +0.0458 | +0.0368 | +0.0380 | **4/4** |

Le pli 3 renverse (c) à lui tout seul (`acc_top` y fait 0.8416 contre 0.8224 pour
`resid_top`). Avec 4 clusters aucun IC n'est estimable de façon fiable sur (c).
**(a) et (b) ne sont PAS touchés** : 4/4 plis concordants, IC bootstrap-cluster
[+0.0361, +0.0435] pour `resid_top` − random, quasi identique à l'IC sujet.

**2. Le TOST ne démontre l'équivalence qu'à ±40 %, pas en dessous.**

| marge (fraction de l'effet de sélection) | ±50 % | ±40 % | ±30 % | ±25 % | ±20 % |
|---|---|---|---|---|---|
| `resid_top` − `acc_top` équivalent ? | ✔ | ✔ | ✘ | ✘ | ✘ |

L'IC monte à +0.0135, soit **35 % de l'effet de sélection** : un avantage modeste mais
scientifiquement intéressant de `#params` sur l'accuracy n'est pas exclu. Écrire
« équivalence informative » était trop fort.

**Lecture correcte : (c) est INDÉTERMINÉ.** Ni « la taille gagne », ni « l'accuracy
gagne », ni « c'est pareil ». Le protocole n'a pas la résolution pour trancher, et c'est
un défaut de **design** (1 pool par pli), pas de taille d'échantillon — ajouter des
sujets n'y changerait rien.

**Recouvrement mesuré (le protocole a bien de la prise) :** `params_top` ∩ `acc_top`
= 50-65 % selon le pli — la moitié du pool est commune, ce qui atténue mécaniquement ce
contraste-là. `resid_top` ∩ `acc_top` = 25-35 % seulement : c'est le contraste qui a de
la prise, c'est celui à garder. Vivier 81-82 sujets, K=20 = **24 % du vivier**.

**Marge restante :** le meilleur des 6 tirages aléatoires est à +0.011/+0.026 de la
médiane selon le pli, les règles à +0.027/+0.053. Les règles dépassent donc le meilleur
tirage aléatoire — il reste de la marge, l'effet n'est pas saturé. Écart-type
inter-cellules 0.036.

**Confond `n_train` (vérifié) :** `resid_top` 909 essais, `random` 900, `params_bottom`
900, `params_top` 892, `acc_top` 889. Étendue 2 %, et `acc_top` gagne son +0.037 avec le
MOINS de données — le petit avantage de volume va à `resid_top`, il ne peut donc pas
avoir masqué une supériorité de `#params`.

**Ce qui n'a JAMAIS été testé :** un seul dataset (physionetmi, et c'est celui au
transfert le plus faible et au critère le moins fiable, ICC_k 0.81) ; un seul K ; aucune
règle combinée. Le K est l'axe manquant le plus important : à K=20/82 on prend déjà le
quart du vivier, c'est le régime où tous les critères raisonnables convergent.

**Correctif de design pour trancher (c)** : répliquer le tirage du pool pour les règles
déterministes — 10 plis au lieu de 4, ou top-K sur des sous-échantillons bootstrap du
vivier, pour que chaque règle ait elle aussi une distribution de pools et non un point.

**Exploratoire (post-hoc, à re-tester prospectivement).** `resid_top` et `acc_top` ne
partagent que **28.7 %** de leurs donneurs, leurs gains par sujet corrèlent à r=+0.257,
14 sujets sont aidés par l'un seul et 8 par l'autre. Deux routes quasi disjointes vers le
même gain. La borne « oracle » +0.061 est sélectionnée sur l'issue → gonflée, ne pas la
citer comme résultat.

Coût mesuré : fit poolé (912 essais, 200 époques, `bd_shallow`) = **39.9 s**, MaxRSS
2.9 G. Répliquer sur lee2019_mi + cho2017 ≈ 1 h GPU.

## 02/09 08:15 — claim 3 : la file bloquait (résolu)

Sonde de coût **514152** (une cellule, 3 h demandées) `PD (Priority)`. Ce n'est pas un
problème de code : la partition `tau` n'a plus un seul GPU utilisable de libre.

- margpu007/008 (ampere) : `AllocTRES gres/gpu=3` sur `CfgTRES gres/gpu=3`, état
  `MIXED+PLANNED` — réservés par le backfill. Les CPU libres qu'on y voit ne servent à
  rien, le GPU est la ressource limitante.
- margpu017-022 (turing) : `allocated` / `mixed`.
- margpu024-027, 029 (pascal) : libres mais **inutilisables** (sm_60, torch 2.13+cu130
  → "no kernel image"). C'est pour ça qu'ils sont `idle`. Désormais dans `--exclude`.
- Devant moi : 9 jobs `jjobard` à **priorité 8438 contre 7203** ici, 24 h chacun. Mon
  `FairShare` est à 0.218 (`EffectvUsage` 0.459) — j'ai beaucoup consommé, je paie.

**Correctif appliqué** : `donor_select.sbatch` demande `gpu:1` générique au lieu de
`gpu:ampere:1`. L'argument d'homogénéité matérielle tenait sur le fond mais l'affectation
cellule→carte se fait par index de shard, donc **indépendamment de la règle** : un effet
carte est du bruit réparti, pas un biais. On l'enregistre (`node`, `gpu_name` dans chaque
CSV) au lieu de l'interdire — testable à l'analyse.

Watcher unique `b8v9f80zd`, poll 10 min. **L'array complet (72 cellules) n'est PAS
soumis** : il attend le temps de fit mesuré par la sonde.

## 02/09 (matin) — LES 4 MATRICES D×R SONT LÀ. **Claim 2 passe sur physionetmi.**

513724 / 513725 / 513726 : 9 + 18 + 30 tâches COMPLETED, 27 + 162 + 327 CSV rapatriés
dans `benchmarks/analysis/dxr_<dataset>/`. Avec cho2017 (513589), quatre matrices.

| dataset | n | rho(#params) | rho(acc) | **rho partielle \| acc** | MDE | ICC_k | plafond sonde | transfert |
|---|---|---|---|---|---|---|---|---|
| bnci2014_001 | 9 | +0.285 | +0.600 | +0.239 | 1.14 | 0.98 | **93 %** | +0.384 |
| cho2017 | 52 | +0.155 | +0.383 * | +0.150 [-0.11, +0.42] | 0.40 | 0.96 | 26 % | +0.056 |
| lee2019_mi | 54 | +0.502 * | +0.550 * | +0.229 [-0.01, +0.54] | 0.39 | 0.96 | 34 % | +0.100 |
| physionetmi | 109 | +0.520 * | +0.430 * | **+0.385 [+0.199, +0.548]** * | 0.27 | 0.81 | 18 % | +0.024 |

**Ce qui change par rapport à hier soir.** Le nul de cho2017 n'était PAS une réfutation :
son IC de partielle [-0.106, +0.421] CONTIENT le +0.385 de physionetmi. Les deux
datasets à n≈52 ont un MDE de 0.39-0.40, c'est-à-dire qu'ils ne pouvaient pas détecter
l'effet mesuré sur physionetmi. Sur le seul dataset qui a la puissance (n=109, MDE 0.27),
la taille finale du réseau prédit la qualité de donneur **au-delà de ce que l'accuracy du
sujet prédit déjà** — c'est exactement l'énoncé de claim 2.

**Les deux caveats à ne jamais retirer.**
1. Le dataset qui fait passer claim 2 est aussi celui où le transfert est le plus faible
   (+0.024 au-dessus de la chance) et le critère le moins fiable (ICC_k 0.81 contre 0.96
   ailleurs). Porte 0 passe, mais de peu : à citer avec la partielle.
2. La co-variable est la censure. Le plafond de la sonde va de 18 % (physionetmi) à 93 %
   (bnci2014_001) et la partielle va dans l'autre sens. bnci2014_001 n'est donc pas un
   contre-exemple : à 93 % de saturation le prédicteur n'a plus de variance, et n=9 donne
   un MDE de 1.14 — la ligne est indécidable par construction, elle ne sert qu'au
   contraste dataset-qui-sature / dataset-qui-ne-sature-pas demandé par Sylvain.

Scores `within_session` de bnci2014_001 / bnci2015_001 resynchronisés depuis
`/scratch/amounir/results_final` (CSV seulement) : c'est ce qui débloque la sonde de
bnci2014_001, absente hier.

Figures : `benchmarks/analysis/figures_dxr_summary/` (forêt 4 datasets + censure,
scatters partiels par sujet, portes 0/1) et `figures_dxr_<dataset>/` (6 figures chacun).
Script : `benchmarks/analysis/donor_summary.py`.


## 01/09 (22 h 00 UTC) — 3 matrices D×R relancées + figures. **Correction sur la censure.**

### Ce qui tourne

| job | dataset | n | unités | shards | nœuds |
|---|---|---|---|---|---|
| 513724 | bnci2014_001 (**sature** — l'autre dataset demandé par Sylvain) | 9 | 27 | 9 | marg003-004 |
| 513725 | lee2019_mi | 54 | 162 | 18 | marg004-005 |
| 513726 | physionetmi (**n le plus grand disponible**) | 109 | 327 | 30 | marg006+ |

57 tâches, toutes RUNNING. `--exclude=marg[001-002,013-032]` : classe CPU et mémoire
homogènes (515 Go), la même que la matrice cho2017, sinon le classement de donneurs
hérite d'une différence de machine. Un seul watcher pour les trois.

**Pourquoi lee2019_mi et physionetmi alors que l'étage 0 les avait écartés.** Ils
avaient été écartés comme *test propre de l'orthogonalité* (`#params` y corrèle avec
l'accuracy à ρ = +0.52 et +0.65, donc une corrélation avec la qualité de donneur y
serait ambiguë). Ils restent valables pour deux autres choses : la **puissance**
(n = 109 → MDE ρ = 0.26 contre 0.38 sur cho) et la **ρ partielle | accuracy**, qui
reste interprétable même quand les deux variables sont liées. Sur ces deux datasets,
la marginale ne vaut rien ; c'est la partielle qu'on lira.

**bnci2014_001 grow_shallow within_session n'est PAS à relancer** : les fits sont en
cours d'écriture par la campagne (509152/509153, `grow_shallow__seed{0,1,2}__fits.jsonl`
modifiés à 21 h 38). Le CSV de scores arrive en fin de cellule. Ne pas dupliquer.

### CORRECTION — la sonde est censurée elle aussi

J'ai écrit dans la section précédente que la sonde « reste sous la cible ». **Faux.**
`grow_shallow.yaml` borne la croissance des deux côtés (`n_filters_time: 8`,
`target_n_filters_time: 40`) et les deux variables tapent dans les bornes :

| | plancher (largeur 8, aucune croissance) | plafond (largeur 40) | censuré |
|---|---|---|---|
| sonde de campagne, 780 fits | 6 % | 26 % | **32 %** |
| fit donneur, 156 fits | 7 % | 36 % | **43 %** |

Ce qui sauve le prédicteur est la **moyenne sur k=15 réplicats** : aucun sujet n'a une
moyenne collée à la borne (min 14.1, médiane 30.7, max 38.9 sur 40), donc la variable
garde du contraste. Mais 11 sujets sur 52 ont ≥ 50 % de leurs fits au plafond : le haut
de l'échelle est comprimé et **le ρ de claim 2 est mesuré sur une règle tronquée**.

Conséquence à trancher : le nul de claim 2 (ρ = +0.155) est mesuré sur un prédicteur
atténué par la censure. Le test propre serait de relever `target_n_filters_time`
au-dessus de 40 et de refaire la sonde — c'est une re-campagne, pas une analyse.
Cette figure doit accompagner le nul, jamais être omise.

### Figures

`benchmarks/analysis/donor_figures.py --dxr dxr_cho2017` → `figures_dxr/` :

1. `dxr_01_heatmap.png` — la matrice, lignes triées par qualité de donneur, colonnes par
   difficulté de receveur. Les bandes horizontales sont visibles à l'œil : c'est porte 1.
2. `dxr_02_donor_quality.png` — qualité par sujet, les 3 seeds + leur moyenne.
   ICC(1) = 0.885, ICC_k = 0.96.
3. `dxr_03_predictors.png` — les deux nuages côte à côte. `#params` est un nuage,
   l'accuracy a une pente. C'est la figure du nul.
4. `dxr_04_paired.png` — la distribution bootstrap de |ρ(#params)|−|ρ(acc)|, IC sur zéro.
5. `dxr_05_ceiling.png` — la censure des deux variables (le tableau ci-dessus).
6. `dxr_06_predictor_scale.png` — la moyenne par sujet rattrape la censure des fits.


## 01/09 (21 h 40 UTC) — Matrice D×R cho2017 TERMINÉE. **Claim 2 ne passe pas.**

Job array 513589 : 26/26 COMPLETED, 156/156 CSV, ~2 h 45 de bout en bout.
Rapatriés dans `benchmarks/analysis/dxr_cho2017/`. Analyse :
`.venv/bin/python benchmarks/analysis/donor_matrix.py --dxr benchmarks/analysis/dxr_cho2017`

| | résultat | verdict |
|---|---|---|
| **Porte 0** — y a-t-il du transfert ? | +0.0561 au-dessus de la chance [+0.0464, +0.0659], n=52 donneurs | **PASS** |
| **Porte 1** — la qualité de donneur est-elle une propriété du sujet ? | ICC(1) = 0.885 sur 3 seeds, ICC_k = 0.96 | **PASS, fort** |
| **Contrôle** — sonde vs fit donneur | rho = +0.650 ; 36 % des fits donneur au plafond de croissance (largeur 40) | mesures d'accord |
| **Claim 2** — `#params` prédit-il, et mieux que l'accuracy ? | voir ci-dessous | **ÉCHEC** |

```
rho(#params (sonde)       , q_cent) = +0.155  [-0.126, +0.423]      non significatif
rho(accuracy (sonde)      , q_cent) = +0.383  [+0.096, +0.630]  *   significatif
rho(#params (fit donneur) , q_cent) = +0.289  [+0.009, +0.534]  *   significatif
rho partielle (#params sonde       | accuracy) = +0.150 [-0.106, +0.421]
rho partielle (#params fit donneur | accuracy) = +0.261 [-0.006, +0.514]
|rho(#params)| - |rho(accuracy)| = -0.228 [-0.541, +0.186]  -> indiscernables
MDE rho a 80 % de puissance, n=52 : 0.38
```

**Ce que ça dit.** Le prédicteur déclaré (`#params` de la sonde, k=15 réplicats) ne
prédit pas la qualité de donneur. Le prédicteur ennuyeux — l'accuracy du sujet — la
prédit, sur exactement les mêmes 52 sujets. La comparaison appariée sort
« indiscernables », donc on ne peut pas écrire « l'accuracy bat #params » ; mais
l'estimation ponctuelle est du mauvais côté et **rien ne soutient l'inverse**.

**Ce que ça ne dit pas.** MDE = 0.38 à n=52 : détecter rho = 0.155 demanderait
**n ≈ 325 sujets**, hors de portée sur un dataset. Ce n'est donc pas « l'effet est
nul », c'est « s'il existe il est trop petit pour cette narrative ». Corollaire à
retenir : **une réplication D×R sur bnci2014_001 (n=9) ne peut pas tester claim 2** —
son MDE serait ~0.8. Elle ne sert que le contraste sature / sature-pas de la narrative
efficacité paramétrique.

**Le plafond de croissance, mesuré et annoté dans le script.** `grow_shallow.yaml` fixe
`target_n_filters_time: 40` ; 36 % des fits donneur finissent exactement à 40/40
(107042 params) parce que le donneur voit 100 % des essais là où un fold en voit 80 %.
Deux conséquences encodées dans `donor_matrix.py` pour ne pas relire le résultat à
l'envers : (1) `#params (fit donneur)` est censuré, son rho est un plancher ; (2) le
rho sonde/donneur est atténué — +0.650 malgré la censure, donc les deux mesures
s'accordent bien.

**Suite.** Claim 3 (interventionnel) n'a plus de prémisse sur cho2017 : le lancer
reviendrait à sélectionner par une variable dont on vient de mesurer qu'elle ne classe
pas les donneurs. Repli disponible et **déjà mesuré** : l'efficacité paramétrique
(grow_shallow domine bd_shallow 25/25 sur le front accuracy/params), qui est l'autre
branche que Sylvain proposait et qui tient dans 4 pages ICASSP. Décision à Adam.

Trous ouverts : scores `within_session/grow_shallow` de bnci2014_001 et bnci2015_001
absents en local (0 fichier) — à resync depuis `results_final`. Aucune figure produite
(le script n'imprime que des nombres) : heatmap D×R triée + nuage `#params` vs qualité
de donneur à faire pour le Discord.


## 01/09 (soir) — Chantier « donneur-receveur » (réunion Sylvain). Étage 0 : **GO**, sous trois contraintes.

### La narrative visée (ICLR)

On arrête de vendre la croissance comme un **classifieur** — la campagne dit qu'elle
n'en est pas un meilleur — et on la vend comme un **instrument de mesure de données** :

> La taille à laquelle un réseau growing s'arrête sur un sujet est une mesure **de ce
> sujet**. Elle est peu coûteuse, orthogonale à l'accuracy, et elle prédit la valeur de
> ce sujet comme **donnée d'entraînement**.

Trois claims, dont seul le 3ᵉ fait un papier ICLR (les deux premiers sont des
corrélations, un reviewer les lira comme telles) :

1. la croissance dimensionne **par sujet** — mesuré, ci-dessous ;
2. `#params` prédit la qualité de donneur — matrice D×R, à lancer ;
3. **sélectionner** le training set par `#params` bat aléatoire / accuracy / tout le
   monde, en cross-subject — interventionnel, c'est là que va le budget GPU.

### Deux corrections au plan sorti de la réunion

**La figure 2 est déjà appariée par sujet.** J'ai répondu « moyennes globales » à
Sylvain pendant la réunion ; c'est faux. `perf_io.by_subject` (`perf_io.py:224`) réduit
à une ligne par (dataset, sujet) et *tout* test du module consomme sa sortie — il n'y a
aucun chemin qui teste au niveau ligne. L'axe de `decomposition` dit déjà
« subject-level mean, 95 % bootstrap » et le `n_win/n` est un décompte de sujets. Les
**figures 15-23** (`subject_delta__*`) sont exactement le sujet-par-sujet demandé et
existent depuis le 31/08 — elles n'ont pas été montrées en réunion. La critique de
Sylvain porte **uniquement** sur les **Pareto (fig 33-35)**, qui elles agrègent pour de
vrai (`perf_figures.py:655` : un point = moyenne sur tous les sujets × datasets).

**La matrice D×R coûte N entraînements, pas N².** Un fit par donneur, puis inférence
sur les N−1 receveurs (gratuite devant l'entraînement). Donc **pas de subset 5+5** :
cho2017 complet = 52 donneurs × 3 seeds = 156 fits pour une matrice 52×52 ;
bnci2014_001 = 27 fits. Le subset était motivé par un coût quadratique qui n'existe
pas, et il coûterait toute la puissance statistique (cf. étage 0 §3).

### Étage 0 — mesuré ce soir, `benchmarks/analysis/donor_predictor.py`

Deux conditions doivent tenir *avant* qu'un protocole D×R ait un sens. Aucune n'avait
jamais été mesurée.

**1. Fiabilité.** Si `params_end` varie autant entre réplicats d'un sujet qu'entre
sujets, ce n'est pas une mesure du sujet mais du bruit d'optimisation. ICC(1) par ANOVA
sur `grow_shallow / within_session / align=none`, réplicats = seeds × folds × sessions :

| dataset | n_suj | k | ICC(1) | **ICC_k** | √ICC_k | CV_inter | CV_intra |
|---|---|---|---|---|---|---|---|
| cho2017 | 52 | 15 | 0.293 | **0.86** | 0.93 | 0.192 | 0.272 |
| lee2019_mi | 54 | 30 | 0.128 | 0.82 | 0.90 | 0.160 | 0.385 |
| physionetmi | 109 | 15 | 0.096 | 0.61 | 0.78 | 0.213 | 0.527 |
| bnci2014_001 | 8 | 62 | 0.168 | 0.93 | 0.96 | 0.038 | 0.061 |
| shin2017a | 29 | 45 | 0.042 | 0.66 | 0.81 | 0.070 | 0.268 |
| **schirrmeister2017** | 14 | 10 | −0.045 | **0.00** | 0.00 | **0.019** | 0.061 |

**Le piège que ce tableau a failli me faire écrire.** L'ICC(1) est bas partout
(0.02–0.29) et mon premier jet concluait « inutilisable ». C'est un faux négatif :
l'ICC(1) juge **un fit isolé**, alors que le prédicteur réel est la **moyenne des k
réplicats**, dont la fiabilité suit Spearman-Brown, `ICC_k = k·ICC / (1 + (k−1)·ICC)`.
Condamner l'instrument sur l'ICC(1) reviendrait à jeter un thermomètre parce qu'une
mesure isolée est bruitée. Après correction : **0.86 sur cho2017**.

→ **Contrainte de design nº1 : le sonde coûte 15 fits par sujet, pas 1.** À 1 seed ×
1 fold la fiabilité retombe à 0.29 et le prédicteur ne peut plus rien prédire. Toute
version « bon marché » du sonde qu'on serait tenté d'écrire dans le papier est à
vérifier contre cette ligne.

→ **Contrainte nº2 : schirrmeister2017 est hors-jeu.** CV inter-sujets de 1,9 %, ICC_k
nul : la croissance y sature à la même taille pour tout le monde. Il n'y a rien à
mesurer, et il faut l'écrire dans le papier plutôt que se le faire dire en review.
`bnci2014_004` (CV 8 %) est limite pour la même raison.

**2. Non-redondance.** Si `#params` est une fonction de l'accuracy, « meilleur
prédicteur que l'accuracy » est une phrase vide. Régression `#params ~ acc + log(essais)`
sur variables centrées-réduites, unité (dataset, sujet) :

| dataset | n | ρ(params, acc) | ρ partielle | R²(acc+essais) | résiduel | plafond |
|---|---|---|---|---|---|---|
| **cho2017** | 52 | **+0.042** | +0.063 | **0.042** | 0.98 | **0.91** |
| lee2019_mi | 54 | +0.647 | +0.647 | 0.230 | 0.88 | 0.79 |
| physionetmi | 109 | +0.519 | +0.529 | 0.342 | 0.81 | 0.64 |
| bnci2014_002 | 14 | −0.200 | −0.200 | 0.079 | 0.96 | 0.88 |

Sur cho2017, **98 % de la variance de `#params` est indépendante de l'accuracy et du
nombre d'essais**. Le dataset que Sylvain a choisi à l'instinct est précisément celui où
le prédicteur peut apporter une information neuve. Le « plafond » (√ICC_k × résiduel)
est la corrélation maximale que `#params` peut encore avoir **en propre** avec la
qualité de donneur : 0.91 sur cho2017, 0.64 sur physionetmi.

**3. Puissance.** MDE sur une corrélation, 80 %, α=.05 bilatéral :

| dataset | n | MDE ρ |
|---|---|---|
| cho2017 | 52 | 0.38 |
| lee2019_mi | 54 | 0.37 |
| physionetmi | 109 | 0.27 |
| bnci2014_001 | — | absent des scores |

→ **Contrainte nº3 : la corrélation se teste sur cho2017 complet (n=52), pas sur un
subset.** À n=10, le MDE monte à ρ≈0.76 : on ne pourrait conclure que si l'effet était
énorme, et un nul ne voudrait rien dire ([[underpowered-not-null]]). BNCI2014-001 (n=9)
sert de réplication qualitative — « le dataset qui sature » — jamais de preuve.

**Limite à déclarer** : les folds d'un même sujet partagent leurs données, donc la
variance intra est un peu sous-estimée et `ICC_k` un peu optimiste. Ça ne change pas
l'ordre de grandeur mais ça se dit dans le papier.

### À réparer avant l'étage suivant

`bnci2014_001` et `bnci2015_001` **n'ont pas de scores `within_session`** dans
`perf_final/scores` alors que leurs fits sont présents dans `dynamics_final/gd_fits`.
Ce sont justement les deux datasets « qui saturent » que Sylvain veut. À resynchroniser
depuis `results_final` quand la campagne finit.

### Suite

1. ✅ étage 0 (ce soir, sans GPU) — fait, GO.
2. ✅ dumbbell + Pareto décomposées — faites, cf. ci-dessous.
3. matrice D×R sur cho2017 (156 fits) — **pas avant** que les grilles courtes libèrent
   de la RAM ; chemin critique schirrmeister ~70 h.
4. claim 3 : sélection par `#params` vs aléatoire vs accuracy vs tout le monde.

### Étage 1 : les figures par sujet (01/09 soir, sans GPU)

Deux fonctions ajoutées à `perf_figures.py`, branchées dans `build_perf_report.py`,
rapport régénéré (52 figures, 6.1 Mo) et **artefact mis à jour en place** —
`https://claude.ai/code/artifact/44b84ba1-d91c-418a-a223-6c0b923cc6ac`, le lien déjà
partagé continue de marcher. Attention : la numérotation a bougé (12 figures insérées),
les anciennes « 33-35 » sont maintenant 42-44.

- `dumbbell(subj, arch, eval)` — le croquis de Sylvain : un sujet par ligne, triangle
  `bd_*`, rond `grow_*`, segment rouge quand la croissance est devant, ligne de chance
  en pointillé, sujets triés par le bras braindecode. Un panneau par dataset.
- `pareto_subjects(subj, eval)` — la Pareto **avant la médiane** : un point par sujet,
  un panneau par dataset, échelle log en abscisse. Les bras fixes sont une ligne
  verticale, les bras growing un nuage — et c'est toute la question.

Ce que les deux figures montrent, et qui n'était visible dans aucune des anciennes :

| within_session, `grow_shallow` | n | étalement taille (max/min) | ρ(taille, score) | CV taille | CV acc |
|---|---|---|---|---|---|
| **cho2017** | 52 | **×2.60** | **+0.04** | **0.190** | **0.191** |
| physionetmi | 109 | ×2.87 | +0.52 | 0.212 | 0.225 |
| lee2019_mi | 54 | ×2.01 | +0.65 | 0.159 | 0.193 |
| bnci2014_002 | 14 | ×1.74 | −0.20 | 0.126 | 0.152 |
| shin2017a | 29 | ×1.29 | +0.29 | 0.069 | 0.173 |
| **schirrmeister2017** | 14 | **×1.07** | +0.06 | **0.019** | 0.122 |

La phrase du papier est dans la première ligne : **sur cho2017 la taille finale varie
d'un sujet à l'autre exactement autant que l'accuracy (CV 0.19 des deux côtés), et les
deux variations sont indépendantes (ρ = +0.04)**. C'est claim 1 mesuré, et ça redit ce
que l'étage 0 disait par la régression (R² = 0.042).

**Découverte qui contraint le protocole.** En `cross_subject`, `params_end` est
*constant sur tous les sujets* de bnci2014_001, bnci2014_002 et bnci2015_001, et
l'étalement tombe à ×1.0-1.6 ailleurs. C'est attendu — en LOSO le modèle est entraîné
sur les N−1 autres sujets, donc presque les mêmes données à chaque fold — mais ça
confirme par la mesure le choix de `PROBE_EVAL = within_session` dans
`donor_predictor.py` : **l'instrument n'existe qu'en within_session**. En cross_subject
la colonne `subject` désigne le sujet de *test*, et y lire une taille comme « la taille
que ce sujet appelle » serait un contresens.

**À ne pas sur-vendre.** Le dumbbell trace `grow − bd`, le titre — pas le terme de
croissance. Sur cho2017/within il vaut +0.036 [+0.027, +0.046], 47/52 sujets, mais la
`decomposition` attribue l'essentiel à la **codebase**, pas à la croissance. La légende
de la figure le dit maintenant explicitement.

## 01/09 (18 h 50 UTC) — Étage 2 : la matrice D×R est LANCÉE (job array 513589)

**156 fits, 52 donneurs x 3 seeds, cho2017 complet.** Un fit par donneur, puis
inférence sur les 51 autres : le coût est en N, pas en N². Sortie
`/scratch/amounir/dxr/cho2017`, un CSV par (donneur, seed) écrit en dernier — donc
reprenable, et analysable partiellement pendant que ça tourne.

### Il n'y a PAS de GPU libre sur Margaret, et les 19 « idle » sont un piège

Vérifié nœud par nœud (`scontrol show node`, AllocTRES) : **tous** les GPU d'ampere,
turing, hopper, volta et rtx sont alloués, sur `tau`, `gpu` ET `gpu-best`. Les 19 GPU
que `sinfo` affiche `idle` (margpu024-027, margpu029) sont des **pascal** — sm_60, sur
lesquels torch 2.13+cu130 échoue avec « no kernel image is available ». `final_grid.sbatch`
les exclut déjà pour cette raison ; ne pas se laisser reprendre par le mot `idle`.

Une demande `gpu:ampere:1` a été mise en file puis **annulée** (513421) une fois la voie
CPU mesurée plus rapide en temps de mur.

### Le coût, mesuré et non extrapolé

Référence GPU, prise dans les enregistrements de la campagne elle-même
(`gd_fits`, 780 fits `grow_shallow`/cho2017/within_session) : **41,2 s par fit**,
200 époques, `stop_reason=budget` sur 100 % des fits.

Sondes CPU (2 donneurs, 1 seed, partition `normal`) :

| régime | fit donneur 1 | fit donneur 2 | transfert (51 receveurs) |
|---|---|---|---|
| CPU, 1 thread | 5001 s | — | 442 s |
| CPU, 8 threads | 539 s | 1078 s | 82-91 s |
| GPU turing (référence campagne) | 41 s | — | — |

Le passage 1 -> 8 threads rend **x9,3** : ce n'est pas un fit qui refuse de
paralléliser, c'est `OMP_NUM_THREADS=1` hérité de `pack_run.sh` qui est un réglage de
**co-tenance GPU**, pas une recommandation. D'où `--threads` sur le script.

CPU 8 threads reste **~20x** plus lent que le GPU par fit. Ce qui décide n'est donc pas
le débit par fit mais la disponibilité : 30 nœuds `normal` idle (56 CPU chacun) contre
zéro GPU. 26 tâches d'array x 8 threads sur marg003-006, 6 unités chacune :
**~2 h de mur**, contre 2 h 25 de calcul sur un GPU qu'il aurait fallu attendre.

Les 156 fits tournent sur **une seule famille de nœuds** (marg003-012, exclusion
explicite du reste) : la matrice est un CLASSEMENT de donneurs comparés entre eux, donc
la constance du matériel est la même exigence que le `--gres-type turing` de la
campagne, appliquée au CPU. Les scores de transfert ne sont en revanche PAS comparables
case à case aux scores within_session de la campagne — ni par le matériel, ni par le
protocole (100 % des essais contre 80 %, pas de fold). Ce qui traverse, c'est le rang.

**Piège SLURM rencontré** : `--array` + `--nodelist=marg[003-012]` fait demander les
**10 nœuds par tâche** (le nodelist fixe le minimum de nœuds), et 7 tâches seulement
démarrent en bloquant la partition. C'est `--exclude` qu'il faut, avec `--nodes=1`.
Première soumission (513581) annulée à 4 s pour ça.

### Le premier signal, sur 2 donneurs

`mean roc_auc` hors-diagonale : **donneur 1 = 0,566**, **donneur 2 = 0,501**. La chance
est 0,50. Deux donneurs ne prouvent rien, mais c'est exactement la forme dont claim 2 a
besoin : un donneur au-dessus de la chance, un donneur à la chance, donc une variance de
qualité de donneur à expliquer. À confirmer sur les 52.

Fait notable et à surveiller : le fit donneur atteint **96 698-107 042 paramètres** là où
la sonde de la campagne s'arrête en moyenne à **79 239**. Attendu — le donneur voit 160
essais d'entraînement contre 128 pour un fold — mais ça confirme que les deux ne sont
pas la même mesure, et que le prédicteur reste celui de la sonde (k=15, ICC_k 0,86) et
non celui du fit donneur (k=3).

### Le code, et ce qui a été validé avant de brûler du calcul

- `benchmarks/donor_receiver.py` — le protocole. Charge les 52 sujets **une fois**
  (46 s depuis le cache MOABB, 2,0 Go en float32), encodage de classes **global** (un
  `left_hand` qui vaudrait 0 chez le donneur et 1 chez le receveur ferait mesurer
  1 − accuracy, et la matrice serait pleine de scores sous la chance sans qu'aucune
  ligne ne soit fausse), protocole corrigé `patience=200 selection_monitor=valid_acc`
  écrit en dur ici parce que rien d'historique n'en dépend.
- `benchmarks/slurm/donor_receiver.sbatch` — l'enveloppe, avec le garde-fou
  d'import (`eegrow` doit venir de `$ROOT/src`, sinon c'est l'arbre d'août sans
  l'abstention s=0 et `params_end` est faussé à la hausse).
- `benchmarks/analysis/donor_matrix.py` — l'analyse, **validée sur deux contrôles
  synthétiques avant lancement** : matrice sans structure -> « indiscernables » ;
  matrice où l'on injecte qualité ~ `params` de la vraie sonde -> rho = +0,99,
  « #params GAGNE ». Un pipeline qui ne saurait rendre que zéro aurait passé le premier
  contrôle et échoué le second.

### Trois choix de protocole, tranchés

1. **Le donneur donne 100 %** de ses essais (moins le split interne skorch, nécessaire
   à la sélection d'époque). La question est « que vaut ce sujet comme jeu
   d'entraînement » ; un donneur qui garde 20 % ne répond pas à celle-là.
2. **Le critère est la qualité CENTRÉE PAR RECEVEUR.** Un receveur facile donne un
   score élevé à tous ses donneurs, et l'exclusion de la diagonale fait que chaque
   donneur est moyenné sur un ensemble de receveurs légèrement différent — biais
   systématique, pas bruit. Le z-score par colonne retire les deux. La version brute
   est reportée pour mémoire.
3. **Le prédicteur est le `params_end` de la sonde de la campagne**, pas celui du fit
   donneur. Un fit isolé a ICC(1) = 0,29 et ne prédit rien ; la moyenne des 15
   réplicats a ICC_k = 0,86.

### Ce que la matrice ne fera PAS

Une corrélation, même forte, dit que `#params` **classe** les donneurs. Elle ne dit pas
que **sélectionner** par `#params` bat l'aléatoire — c'est claim 3, un protocole
interventionnel séparé, et c'est lui qui fait le papier.

## 01/09 (15 h 20 UTC) — 509142 est morte, travail replacé. 973/1116 (87 %).

**509142 (margpu020) COMPLETED à 12 h 44 UTC**, exactement par le mécanisme prévu :
sa grille `g3k1` n'avait plus de cellule libre, donc `[ "$claimed" -eq 0 ] && break`.
Les deux travailleurs `--overlap` que j'y avais lancés à 11 h sont morts avec elle,
laissant **6 claims zombies** (2 schirrmeister, 2 bnci2014_001, 2 cho2017).
margpu020 est repassé en `alloc` pour quelqu'un d'autre — pas pour nous.

`reap_stale` a nettoyé les 6 (vérifié : `PACK reaped 6 stale claim(s)`). Mais il ne
**relance** que ce qui est dans la `ROWS` d'une allocation vivante : les 2 schirrmeister
repartent chez margpu018, les 4 cellules orphelines (cho/bnci) n'appartiennent à aucune
grille vivante et seraient restées bloquées indéfiniment.

**Replacé à 15 h 15**, ordinaux CUDA **sondés** et non déduits :

| hôte | job | grille | G | K | GPU_LIST | RAM libre |
|---|---|---|---|---|---|---|
| margpu018 | 509153 | `grid_orphans.tsv` (23 cellules) | 1 | 3 | 0 | 95 G |
| margpu017 | 509144 | `grid_g1k9.tsv` (lee2019) | 2 | 1 | 1,2 | 101 G |

margpu018 sonde `device_count=1` : **une seule** carte visible, ordinal 0 (index
nvidia-smi 1) ; l'index 0 est la fantôme `[N/A]`. margpu017 : ordinaux 1 et 2 sains
(8,0 et 9,1 G libres), ordinal 0 saturé à 2,9 G — à éviter.

**Débit mesuré** : 22 CSV entre 11 h 00 et 15 h 14 = **5,2 CSV/h**, contre ~3 CSV/h sur
les 24 h précédentes. L'accélération est réelle.

**Fausse alerte levée** : la cellule `lee2019` à 43,1 h que j'avais signalée comme
possiblement bloquée a **fini** — le max observé pour lee2019 est passé de 32,76 à
43,97 h. Les cellules `cross_subject` lee2019 sont juste très longues.

**Chemin critique = schirrmeister2017** : 42 cellules libres, 3 créneaux seulement
(plafond RAM hôte, 33–46 G par cellule), médiane 4,95 h → **~70 h**. Tout le reste
(53 lee2019, 23 orphelines) se vide en ~20 h. Les 6 allocations en file (512163-512168)
sont toujours `PD (Priority)`.

## 01/09 (13 h 05) — LE TRAVAILLEUR DE margpu020 NE CALCULAIT RIEN. Corrigé, 28 cellules concurrentes.

### Le bug : `nvidia-smi` et CUDA ne numérotent pas les cartes pareil

Le travailleur overlap posé sur 509142 (margpu020) échouait **toutes** ses cellules en 19 s :

```
RuntimeError: CUDA_VISIBLE_DEVICES='2' but torch.cuda.is_available() is False:
the pinned device does not exist. Refusing to fall back to CPU.
```

47 cellules `schirrmeister2017` réclamées puis relâchées entre 10 h 20 et 10 h 39, en
boucle. `nvidia-smi` liste 0/1/2 avec la carte 1 fantôme (`[N/A]`) ; j'en avais déduit que
la carte libre était l'index 2. Mais CUDA ordonne par défaut en `FASTEST_FIRST`, **pas**
dans l'ordre de `nvidia-smi`, et sur margpu020 c'est l'**ordinal 2** qui est mort :

```
device_count = 3
  ordinal 0 : libre=0.3G / 10.6G   <- la cellule du parent
  ordinal 1 : libre=10.4G / 10.6G  <- la carte saine et libre
  ordinal 2 : lève une exception   <- la fantôme
```

Vérification après correction : `GPU_LIST=1` allume bien l'index **2** de `nvidia-smi`
(43 %, 1772 MiB). Le mapping est donc confirmé par la mesure, pas par déduction.

**Aucun résultat corrompu** : le garde `pick_device` (`utils.py` l.49) refuse de retomber
sur CPU. Sans lui, 47 cellules auraient été écrites en CPU, mélangées aux cellules GPU
dans les mêmes paires `grow_X − bd_X`. Le refus a coûté 20 minutes ; la retombée
silencieuse aurait coûté la campagne.

**Règle** : sur un nœud à carte fantôme, ne jamais déduire l'ordinal CUDA de
`nvidia-smi` — le lire avec `torch.cuda.mem_get_info(i)` dans l'allocation.

### 27 cellules que personne n'aurait jamais prises

`GRID` est figé au `sbatch` et `pack_run.sh` l.375 fait `mapfile -t ROWS < "$GRID"` **une
seule fois**, avant la boucle de balayage. Les passes `g2k6`, `g3k7`, `g3k9`, `g3k10`
n'ont plus aucune allocation vivante qui les porte : leurs 27 cellules restantes
(bnci2014_001 ×12, bnci2015_001 ×7, cho2017 ×6, bnci2014_002 ×2) étaient **structurellement
inatteignables**. Regroupées dans `passes_final/grid_orphans.tsv` et confiées à un
travailleur dédié.

### Ce qui tourne (mesuré, pas estimé)

| nœud | job | grille du parent | cellules | RAM utilisée / dispo |
|---|---|---|---|---|
| margpu017 | 509144 | g3k1 (épuisée) | 9 × lee2019 | 86 / 101 Go |
| margpu018 | 509153 | g1k3 | 3 × schirrmeister | 137 / 49 Go |
| margpu019 | 509152 | g1k9 | 9 × lee2019 | 137 / 50 Go |
| margpu020 | 509142 | g3k1 (épuisée) | 1 lee + 2 schirr + 4 cho | 116 / 70 Go |

28 cellules concurrentes contre ~20 avant. Dimensionnement fait sur les RSS **observés** :
lee2019 7,2 Go/cellule, schirrmeister 33 à 46 Go/cellule, cho2017 6 à 8 Go.
margpu018 et margpu019 sont saturés (≤ 50 Go libres) — ne rien y ajouter.

### Risque structurel : deux allocations vont mourir

`pack_run.sh` l.546 : `[ "$claimed" -eq 0 ] && break`. 509142 et 509144 sont liées à
`g3k1`, où il ne reste **0 cellule libre**. Dès que leur cellule courante finit, le
balayage suivant ne réclame rien et l'allocation sort — **en emportant ses travailleurs
overlap**. C'est ce qui a tué 509143 hier. Rien ne l'empêche : `ROWS` est déjà chargé.

Quand ça arrivera : relâcher les claims orphelins (essai à blanc d'abord — des claims sans
CSV appartiennent à des jobs **vivants**, les effacer déclencherait des doubles exécutions).

### File

Les 6 allocations soumises hier (512163-512168) démarrent au plus tôt le **06/09** :
margpu007/008/021/022/028 sont en `PLANNED` pour jjobard, margpu023 `drained*`, et les
nœuds pascal (024-027, 029) sont sm_60 où torch échoue. Aucune nouvelle allocation avant
5 jours ; l'overlap est le seul levier.

**Avancement : 949/1116 CSV, 977 claims.** Reste 145 cellules libres au lancement :
lee2019 76, schirrmeister 42, orphelines 27.

---

## 01/09 (12 h 25) — TRAVAILLEURS OVERLAP LANCÉS sur les GPU inactifs (accord d'Adam)

`srun --jobid=N --overlap` réutilise l'allocation du parent : **aucune nouvelle allocation,
aucune attente en file**. Les cartes sont les mêmes Turing (RTX 2080 Ti, 11 Go), donc
aucune contamination de classe matérielle dans les paires `grow_X − bd_X`.

### Dimensionnement — la RAM hôte, pas le GPU

`pack_run.sh` l.251 : sur `lee2019_mi`, `CrossSessionEvaluation` garde tout le dataset en
mémoire (4,4 Go de tableaux par processus) ; la première campagne packée a **perdu 30
cellules lee2019 à l'OOM killer du cgroup, tuées au chargement, sans traceback**.
`K = min(K_gpu, K_ram)`. Composition : `g1k9` = 94 × `lee2019_mi`, `g1k3` = 52 ×
`schirrmeister2017` (~27 Go/locataire).

RAM libre mesurée avant lancement (`srun --overlap free -g`) :

| job | nœud | dispo | GPU inactifs | décision |
|---|---|---|---|---|
| 509142 | margpu020 | 153 Go | **1** (GPU1 mort, répond `[N/A]`) | g1k3 G=1 K=2 → 2×27 = 54 Go |
| 509143 | margpu022 | 155 Go | 2 (GPU 0 et 2) | g1k9 G=2 K=3 → 6×13 = 78 Go |
| 509144 | margpu017 | 154 Go | 2 (GPU 1 et 2) | g1k9 G=2 K=3 → 6×13 = 78 Go |
| 509153 | margpu018 | **48 Go** | 1 | **écarté** — empile déjà 3 locataires |

Marge ≥ 75 Go partout. **+14 cellules concurrentes** (12 lee2019 + 2 schirrmeister)
contre 5 GPU qui calculaient.

### Le piège du mapping GPU

`pack_run.sh` l.491 : `gpu=$((p % G))` — les locataires vont sur les GPU physiques
`0..G-1`. Or les cartes libres ne sont pas les premières (parent sur GPU0 de margpu017 et
margpu020). Lancer tel quel aurait envoyé les nouveaux locataires **sur la carte du parent**
→ OOM GPU sur une cellule de plusieurs jours.

Correctif : copie `pack_run_overlap.sh` + variable `GPU_LIST`. **Ne jamais éditer
`pack_run.sh` en place** — bash relit son script au fil de l'exécution, une édition
corromprait les 5 allocations vivantes.

Lanceur : `/scratch/amounir/launch_overlap.sh`, journaux `/scratch/amounir/logs/overlap/`.

Garde-fous passés par les 3 : `guard ok … s=0 abstention present sha=58c6beebc12e`,
`PROTOCOL='train.patience=200 train.selection_monitor=valid_acc'`, cache complet.

`pgrep` montre 2 `srun` par job : c'est le fork interne de `srun`, pas un double
lancement (même motif chez les autres utilisateurs). Les bannières `PACK node=` sont
uniques par nœud.

### Risque accepté — et réalisé au bout de 5 minutes

Un travailleur overlap meurt avec son parent. **509143 a expiré pendant la vérification** :
sa cellule g3k1 s'est terminée, son balayage était déjà fini (« PACK sweep done » à
05 h 35), le job est sorti et a emporté notre travailleur et ses 6 locataires.

Dégâts réparés : 6 claims orphelins libérés (script de tri ci-dessous), 0 CSV perdu,
0 double exécution. Le format `owner` est `nœud PID jobid`, donc un claim est **sûrement**
périmé si son jobid n'est pas dans `squeue -t R` **et** qu'aucun CSV n'existe. Essai à
blanc systématique avant suppression : au même instant 22 claims sans CSV appartenaient à
des jobs vivants (cellules en cours) — les effacer aurait causé une double exécution.

    squeue -u $USER -h -t R -o "%i" > /tmp/am_live.txt   # puis le tri owner/CSV

**Leçon** : ne pas lancer d'overlap sur une allocation dont le journal dit « PACK sweep
done » — elle sort dès que sa dernière cellule finit. Seules les allocations encore en
balayage sont des hôtes stables.

### État après l'opération (01/09 12 h 40)

4 allocations R (509142, 509144, 509152, 509153), 509143 expirée.
2 travailleurs overlap vivants : 509144 (g1k9, 6 locataires, GPU 1 et 2 à 79 %/82 %) et
509142 (g1k3, 2 locataires, en chargement).
CSV **948/1116**, claims 970 → 22 en vol, 124 non réclamées.
6 allocations toujours PD (512163-512168).

## 01/09 (12 h 10) — LE GOULOT N'EST PAS LE CLUSTER, C'EST NOTRE RÉPARTITION. 6 allocations soumises.

Adam : « parallélise au maximum si tu trouves des cartes utilisables ». **Il n'y en a
aucune de libre** — mais nos propres allocations gaspillent 12 GPU sur 17.

### Pourquoi on ne peut pas prendre plus de matériel

| ressource | verdict |
|---|---|
| 15 GPU Pascal inactifs (`margpu024-027,029`) | **sm_60 : torch 2.13+cu130 échoue, « no kernel image is available »** |
| `margpu023` (turing:4) | DOWN+NOT_RESPONDING depuis le 10/08 |
| `margpu021` (turing:3) | GPU fantôme — annonce 3 cartes, répond « No devices were found » |
| `margpu028` (turing:2), `margpu007/008` (ampere) | MIXED+PLANNED, réservés à la file de `jjobard` (24 jobs PD prioritaires) |
| `margpu017-020, 022` | déjà les nôtres |

Ampere serait de toute façon exclu : chaque chiffre de tête est un `grow_X − bd_X` apparié
par sujet, une paire à cheval sur deux classes de cartes enferme une différence matérielle
dans la différence mesurée.

### Le vrai gaspillage (mesuré par `srun --jobid=N --overlap nvidia-smi`)

| job | grille | GPU actifs | restant dans sa grille |
|---|---|---|---|
| 509142 | g3k1 | 1/3 | 3 |
| 509143 | g3k1 | 1/3 | (idem) |
| 509144 | g3k1 | 1/3 | (idem) |
| 509152 | g1k9 | 1/1 | **93** |
| 509153 | g1k3 | 1/2 | **47** |

**5 GPU calculent sur les 17 réservés en exclusif.** `GRID=` est figé au `sbatch` : les
3 allocations g3k1 ne peuvent pas aider, et on ne peut pas les annuler — chacune tourne
une cellule `cross_subject` longue (79 % / 85 % / 97 % d'occupation GPU).

Ne pas se fier au « PACK sweep done » de 509143 (01/09 05 h 35) : le balayage de
réclamation est fini, mais une cellule tourne encore. Le journal ne le dit pas, `nvidia-smi` si.

### État par passe

| passe | total | csv | claims | en vol | non réclamées |
|---|---|---|---|---|---|
| grid_g1k9 | 94 | 1 | 10 | 9 | **84** |
| grid_g1k3 | 52 | 5 | 8 | 3 | **44** |
| grid_g3k10 | 666 | 653 | 653 | 0 | **13 orphelines** |
| grid_g2k6 | 52 | 46 | 46 | 0 | **6 orphelines** |
| grid_g3k7 | 26 | 20 | 20 | 0 | **6 orphelines** |
| grid_g3k9 | 40 | 38 | 38 | 0 | **2 orphelines** |
| grid_g3k1 | 82 | 79 | 82 | 3 | 0 |
| g3k2, g3k6, g3k8 | 104 | 104 | 104 | 0 | 0 |

27 cellules étaient **orphelines** : allocation morte, plus aucun travailleur pour les
réclamer. Elles ne seraient jamais sorties.

### Action — 6 allocations soumises (512163-512168), toutes PD

Une par grille sous-dotée, paramètres `G`/`K`/`EEGROW_CUDA_FRACTION` recopiés à
l'identique depuis `passes_final/submit.sh`. Le wrapper autorise explicitement le motif :
« *an extra allocation adds throughput with no split to get wrong* » — coopération par le
répertoire de claims atomique, donc aucun risque de double exécution.

| job | grille | couvre |
|---|---|---|
| 512163 | g3k10 | 13 orphelines |
| 512164 | g2k6 | 6 orphelines |
| 512165 | g3k7 | 6 orphelines |
| 512166 | g3k9 | 2 orphelines |
| 512167 | g1k9 | renfort sur les 84 |
| 512168 | g1k3 | renfort sur les 44 |

Toutes en `(Priority)` derrière `jjobard`. Elles démarreront quand un nœud Turing se
libère — y compris quand une allocation g3k1 finit sa cellule.

**Non fait, demande l'accord d'Adam** : lancer des travailleurs supplémentaires sur les
12 GPU inactifs *à l'intérieur* de nos allocations via `srun --jobid=N --overlap`. Ça
triplerait le débit sans attendre la file, mais `--mem` n'est pas appliqué sur ce cluster
et c'est exactement le scénario qui a tué 85 cellules en v5 par OOM — ici la victime
serait la cellule longue du parent.

## 01/09 (11 h 40) — ACCÈS RÉTABLI PAR LE VPN. 945/1116, mais le débit s'effondre.

Adam s'est connecté au VPN : `ssh-sif:22` **et** `margaret02:22` répondent tous les deux.
Le VPN n'est donc pas requis *en principe* (Sylvain a raison), mais il contourne en
pratique ce qui bloquait depuis le réseau d'Adam. Je n'ai pas coupé le VPN pour
distinguer « pare-feu réparé entre-temps » de « filtrage du réseau d'Adam contourné » —
non tranché, et sans importance tant que le VPN marche.

Relevé (`squeue` + comptage sous `/scratch/amounir/results_final`) :

| | |
|---|---|
| allocations R | 5 — 509142 (margpu020), 509143 (022), 509144 (017), 509152 (019), 509153 (018) |
| walltime restant | 5 j 06 h pour la plus courte (509142) → expire ~06/09 18 h |
| CSV | **945 / 1116 (85 %)** — cross_session 282, cross_subject 128, within_session 535 |
| claims (`/scratch/amounir/eegrow_claims_final`) | 961 → **16 en vol, 155 non réclamées** |
| débit 12 h | 34 CSV = **2.8/h** |
| débit 3 h | 2 CSV = **0.67/h** |

**Le mur de 7 j redevient un risque**, contrairement à ce que je disais le 31/08. 171
cellules restantes : à 2.8/h → 61 h → fin le 04/09 ; mais à 0.67/h → 255 h, soit bien
au-delà du 06/09. La fenêtre 3 h ne fait que 2 CSV, donc l'estimation est bruitée — c'est
un signal à re-mesurer, pas encore un fait.

Mécanisme (inchangé) : les cellules bon marché sont épuisées, les travailleurs sont
monopolisés par le chemin critique `cross_subject/lee2019_mi`.

16 en vol pour 5 travailleurs → **~11 claims périmés** (cellules mortes dont le
travailleur a disparu). C'est le vivier de la passe de rattrapage, toujours non lancée.

Quota : `/home` partagé à **95 %** (382 G libres sur 7 T). Pas encore bloquant, mais c'est
bien le sujet que l'admin signalait. `/scratch` n'est pas concerné, les résultats sont à
l'abri.

**Rien relancé, rien soumis, rien annulé. Relevé seul.**

Attention au nommage : les claims sont
`cross_session__bnci2014_001__bd_deep4__easubject__seed0` (plats, dans un répertoire à
part) alors que les CSV sont `<eval>/<dataset>/grow_deep__seed1.csv`. Un `comm` entre les
deux listes renvoie 961 faux « en vol ». Le seul comptage valide est la soustraction
claims − CSV.

## 01/09 (11 h 20) — ACCÈS COUPÉ : `ssh-sif` filtre le port 22. Aucun relevé possible.

Le trou dans les relevés depuis le 31/08 21 h n'est **pas** un arrêt de la campagne, c'est
une perte d'accès. Diagnostic mesuré, pas supposé :

| test | résultat | lecture |
|---|---|---|
| `ping 193.55.251.53` (ssh-sif) | 0 % perte, 4.8 ms | machine allumée et routable |
| `nc 193.55.251.53:22` | timeout silencieux | port filtré en `DROP` |
| `nc 128.93.193.18:22` (gitlab.inria) | OK | l'Inria est joignable en SSH |
| `nc 140.82.121.3:22` (github) | OK | pas de blocage SSH sortant local |

Ping qui passe + port 22 muet (ni `refused` ni `no route`) = règle de pare-feu, sur cette
destination seulement. Cohérent avec DINDART Guillaume sur Mattermost le 30/08 :
« I'm having some issues with the NFS quota and **firewall** on Margaret. »

**Ce n'est pas le VPN** — fausse piste que j'avais donnée le 31/08 et qu'Adam a relayée à
Sylvain ; Sylvain a confirmé : « Normalement, tu peux te connecter en ssh sans passer par
le VPN. » `gitlab.inria.fr:22` répond sans VPN, ce qui le prouve.

Les 5 allocations continuent d'écrire dans `/scratch/amounir/results_final` : le filtrage
bloque l'accès, pas les nœuds de calcul. **Rien relancé, rien soumis, rien annulé.**

À vérifier au retour de l'accès, en plus du relevé : le quota NFS que le même admin
signale. `/scratch` n'est pas sur NFS donc les résultats sont à l'abri, mais un `~` saturé
fait mourir un job sur une écriture de log refusée — cause classique de cellules mortes
silencieuses.

## 31/08 (23 h 45) — LE RAPPORT DE PERFORMANCE EXISTE AUSSI, et la croissance n'y gagne nulle part

Adam : « sur quelle figure je peux voir où la croissance a gagné ? » Réponse honnête :
**aucune**. Les 32 figures de dynamique sont toutes du mécanisme, zéro accuracy. Et le
rapport v5 qui avait les figures de perf (`win_matrix`, `growing_vs_fixed`,
`subject_delta_growing`) lit `results_v5_published` et date du **23/08 20 h 37** — donc
**avant `5337c56` (25/08, abstention à s=0)**. Ses bras `grow_*` ne sont pas le code
d'aujourd'hui. Il ne faut pas s'en servir pour répondre à « où ça gagne ».

Décision : **deux rapports appariés**, pas un. Contrainte dure — 8.5 Mo + 3.4 Mo passent,
mais 8.5 + 7.6 (v5) = 16.2 Mo dépassait le plafond Artifact de 16 Mo. Plus deux pipelines
(CSV de scores vs JSONL d'historique) et deux cadences. Chaque page renvoie à l'autre.

### Trois modules neufs

| fichier | rôle |
|---|---|
| `benchmarks/analysis/perf_io.py` | chargeur + stats. Unité = **le sujet**, sans exception |
| `benchmarks/analysis/perf_figures.py` | 17 fonctions de figure |
| `benchmarks/analysis/build_perf_report.py` | driver, importe `_CSS` de `build_growth_dynamics` |

Scores tirés du cluster (`/tmp/scores_final.tgz`, 474 Ko) → `perf_final/scores/`,
907 CSV = **34 605 folds → 8 561 unités sujet**, 12 datasets, 9 bras.
Rendu : **40 figures, 0 skipped**, 3.4 Mo.

- perf : https://claude.ai/code/artifact/44b84ba1-d91c-418a-a223-6c0b923cc6ac
- dynamique (republié avec le renvoi croisé) : https://claude.ai/code/artifact/1c817e93-e1a0-49e4-ab1f-0c2419c96c7c

### Ce que la décomposition dit

`grow − bd = (grow − fix) + (fix − bd)`, sur les 6 cellules (protocole × archi) où le
contrôle fixe existe, Holm sur la famille des 6 :

| protocole / archi | total | croissance | codebase |
|---|---|---|---|
| within / shallow | **+0.0263** [+0.0188, +0.0341] | +0.0051 | **+0.0187** |
| within / deep | **−0.0405** [−0.0517, −0.0289] | +0.0031 | **−0.0436** |
| within / sccnet | +0.0014 | −0.0031 | +0.0033 |
| cross-sess / shallow | +0.0101 | +0.0026 | +0.0075 |
| cross-sess / deep | **−0.0378** | +0.0003 | **−0.0381** |
| cross-sess / sccnet | +0.0043 | +0.0081 | −0.0038 |

1. **Le terme de croissance tient dans [−0.0031, +0.0081], et 0 sur 6 survit à Holm.**
2. **Le terme de codebase est le plus grand dans 5 cas sur 6.** Sur `deep`/within il vaut
   **−0.0436** : le `grow − bd` de −0.0405 est presque entièrement la ré-implémentation
   qui perd, pas la croissance. Publié sans le contrôle, ça se lit « la croissance nuit ».
3. **Sous-puissance** (`power.png`) : **13 des 21 contrastes** ont un effet plus petit que
   leur propre MDE ET un IC qui croise zéro. Ce ne sont pas des nuls, ce sont des
   mesures vides.
4. **Niveau de chance** : **3 699 / 8 561 cellules-sujet (43.2 %)** sont sous le seuil de
   chance sur la majorité de leurs folds. Pires : `fix_deepeeg`/within (414),
   `grow_deep`/within (406), `bd_deep4`/within (362). physionetmi et shin2017a sont
   essentiellement du bruit pour tout le monde. Les datasets concernés sont hachurés et
   exclus du pooling.
5. **Front params/accuracy** (`pareto__within_session`) : la famille sccnet domine —
   meilleur score à ~1.3e4 paramètres — et `bd_deep4` est le pire à 2.6e5.
   `grow_shallow` ≈ `fix_shallow` > `bd_shallow` à **moitié moins de paramètres**.
6. **Rang moyen** : `fix_sccnet` 1er en within, `bd_deep4` 1er en LOSO, `grow_deep`
   **dernier** en LOSO (5.05 / 6, CD = 0.47 → séparé).
7. **Pas d'effet « données rares »** (`train_size`) : la famille growing ne décolle pas
   à petit n ; en LOSO elle est uniformément sous braindecode.

### Trous de couverture réels (23 cellules en vol, 5 jobs)

`grow_shallow` a **0 fold** sur `bnci2014_001` et `bnci2015_001` en within/raw — donc le
dataset sur lequel le +5.06 de codebase avait été trouvé n'a pas encore son bras qui
grandit. `lee2019_mi` et `schirrmeister2017` n'ont que `grow_shallow`. À re-générer à la
fin (02–03/09) ; les deux commandes sont dans les docstrings des drivers.

### Quatre bugs de mon instrument, corrigés

- `attach_params` perdait **exactement 50 %** du grid : `gd_fits` écrit `align_tag = NaN`
  là où les CSV écrivent `"none"`, et une jointure sur NaN perd uniformément — donc ça
  ressemblait à un trou de couverture plausible.
- `pareto` centrait le score **dans (modèle, dataset)** au lieu de dans le dataset : tous
  les points s'écrasaient exactement sur zéro. Figure vide, pas fausse — pire.
- `_finish` réservait l'en-tête en **fraction de figure** : titre à travers le sous-titre
  sur toutes les figures courtes. Réservé en pouces.
- `chance_map` faisait `subplots_adjust` **après** `fig.colorbar(ax=[...])` → la barre
  était redessinée par-dessus le 3e panneau.

---

## 31/08 (22 h 30) — LES FIGURES DE DYNAMIQUE EXISTENT, et elles trouvent quatre choses

Trois modules écrits, 32 figures produites sur les 897 CSV / 948 JSONL actuels.
Rapport : `benchmarks/analysis/dynamics_final/growth_dynamics.html` (8.5 Mo) +
artefact <https://claude.ai/code/artifact/1c817e93-e1a0-49e4-ab1f-0c2419c96c7c>.

| fichier | rôle |
|---|---|
| `benchmarks/analysis/export_growth_dynamics.py` | réducteur **en streaming** (9.2 Go de JSONL, pic mémoire = 1 fichier) → `gd_fits` / `gd_curves_mean` / `gd_events` |
| `benchmarks/analysis/growth_dynamics.py` | 19 fonctions de figure |
| `benchmarks/analysis/build_growth_dynamics.py` | driver : rend les 32 figures + assemble la page |

Frames : 120 249 folds, 191 200 lignes de courbe, **498 160 opportunités de
croissance**. Export = 333 s.

### 1. La recherche linéaire REFUSE la majorité de ce qu'on lui propose

`SCALING_GRID = (0.0, 0.1, 0.5, 1.0)` et **s = 0 est un refus**. Sur les 498 160
opportunités, **34.8 % seulement sont appliquées**. Taux de refus par bras :
`grow_shallow` **76.5 %**, `grow_sccnet` 4.3 %, `grow_deep` 1.0 %.

Mesuré sur shin2017a / `grow_shallow` : la croissance tourne sur les 39 opportunités du
fold, se voit proposer 26 candidats à chaque fois, et **refuse 93.6 % du temps**. D'où
une largeur finale médiane de **8 sur une cible de 40** — pas un cap, pas un crash : un
« non » explicite et répété. Le refus est une **abstention**, pas un verrou (`done_`
reste False), donc un bras peut refuser 39 fois de suite.

La 2e vignette de la figure tranche entre les deux diagnostics : le taux de refus part
de 0 à la 1re opportunité et monte à 100 % vers la 15e-25e → **c'est de la saturation**,
pas un rejet dès le départ. Lecture plutôt saine du mécanisme.

### 2. Quand elle accepte, elle prend le plafond de sa propre grille

**s = 1.0 sur 98.8 % des 173 175 pas appliqués.** 1.0 est le **maximum** de la grille.
Une recherche qui rend sa borne supérieure quasi systématiquement est une recherche
dont la borne est contraignante : l'amplitude optimale est vraisemblablement > 1 et on
ne lui a jamais laissé le dire. Une ligne à changer dans `SCALING_GRID`, puis une
expérience.

### 3. Le gain premier ordre est 4 000× plus petit qu'une époque ordinaire

Contrôle : la baisse de train-loss de l'époque *précédente*, même fold, sans croissance.
Médiane du rapport `gain prédit / baisse d'une époque ordinaire` = **1e-3.6** pour
`grow_shallow`, 1e-0.9 pour `grow_deep`. Et la baisse **réalisée** sur l'époque de
croissance est indiscernable de celle d'une époque ordinaire (boîtes superposées).
Cohérent avec [[growth-no-local-effect]] du v5, mais mesuré ici sur le mécanisme lui-même.

`grow_first_order_improvement` est **NaN sur 100 % des événements `grow_sccnet`**
(BatchNorm2d à la jonction, cf. `loop._update_diagnostics`). Le docstring dit « trois de
nos quatre bras » ; la mesure dit **un sur trois**.

### 4. Deux trous de protocole que la campagne ne pourra pas combler

**a. Aucun contrôle fixe sur `cross_subject`.** Les 3 bras `fix_*` ont 0 dataset et
**0 claim** sous `cross_subject` : c'est la grille *planifiée*, pas une grille
incomplète (`passes_final/*.tsv` : 48 cellules par bras `bd_`/`grow_`, 0 pour `fix_`).
Donc en LOSO, `grow − bd` **ne sera jamais décomposable** en croissance + codebase —
et vu le +5.06 de codebase mesuré ce matin, c'est le protocole où ça comptait le plus.

**b. Les bras qui grandissent ne sont pas width-matched.** Folds atteignant la cible :
`grow_shallow` **25 %** (médiane 24 sur 40), `grow_deep` 47 % (24 sur 32),
`grow_sccnet` 72 % (22 sur 22). Toute affirmation d'efficacité paramétrique lit
`width_end` ; voilà ce que cette colonne contient.

### Deux corrections à mon propre instrument, trouvées en regardant les figures

- **Le 1er export filtrait sur `grow_applied`** → il jetait 93 % des pas et faisait
  croire que la recherche linéaire répondait toujours 1.0. Corrigé : on garde toutes les
  opportunités + une colonne `applied`.
- **Ordre des callbacks** : `FitRecorder` enregistre `width`/`n_params` *après* `gromo`,
  donc les valeurs stampées sur une époque de croissance sont **post**-croissance.
  Détecté parce que tous les ratios de paramètres sortaient à exactement 1.000.

### Vérifié au passage

100 % des 120 249 folds finissent sur `stop_reason=budget` à l'époque 200 exactement —
`patience=200` contre `max_epochs=200` rend l'early stopping arithmétiquement incapable
de se déclencher. **Aucune figure ici n'est confondue par une longueur d'entraînement
inégale.** En revanche l'époque du modèle *sélectionné* est médiane 9-13 pour les bras
fixes et 23-28 pour les bras qui grandissent : ~90 % du budget est dépensé au-delà de
l'optimum du fold.

## 31/08 (21 h) — RELEVÉ : 897/1116, une allocation de moins, débit tombé à 3 CSV/h

**Rien relancé, rien soumis. Relevé seul.** 18 h 57 UTC = 20 h 57 Paris.

| | |
|---|---|
| allocations `R` | **5** : 509142 (margpu020), 509143 (022), 509144 (017), 509152 (019), 509153 (018) |
| **allocation terminée depuis 14 h 45** | **509145 (margpu028)** — on est passé de 6 à 5 travailleurs |
| walltime restant | ≥ 5 j 21 h → le mur de 7 j n'est toujours pas en jeu |
| CSV | **897 / 1116 (80 %)** |
| claims | 920 → **23 en vol**, **196 non réclamées** |
| débit 12 h | 62 CSV = **5.2/h** |
| débit 3 h | 9 CSV = **3.0/h** |

**Le débit continue de se dégrader** (6 → 5.2 → 3.0 CSV/h) et une allocation est tombée.
ETA sur les 219 restantes : **42 h à 5.2/h (02/09)**, **73 h à 3.0/h (03/09 soir)**.

Cause visible dans la liste des cellules en vol : **11 des 23 sont
`cross_subject/lee2019_mi`** — le chemin critique déjà signalé, qui monopolise les
travailleurs pendant que les cellules bon marché sont épuisées. Le reste est
`grow_shallow` sur cho2017/weibo2014/physionetmi + 3 schirrmeister.

### Les figures de dynamique de croissance : la donnée est là, les figures n'existent pas

Vérifié sur les records de la campagne finale elle-même
(`results_final/**/*_fits.jsonl`, **948 fichiers**, écrits à côté des CSV).

**Couche donnée — COMPLÈTE.** Un fit `grow_shallow__easubject__seed0` de
within/bnci2014_001 porte :

- au niveau du fit : `subject` (**stampé, réel**, pas inféré), `session`, `cv_ind`,
  `stop_reason` (`budget`), `restored_epoch` (45), `width_start`→`width_end` (8→40),
  `params_start/end`, `optimizer`
- par époque : `grad_norm`, `grad_norm_max`, `lr`, `grow_s`, `grow_applied`,
  `grow_n_proposed`, `grow_n_kept`, `grow_width_after`,
  `grow_first_order_improvement`, `grow_eig_sum`, **`grow_eig_proposed` /
  `grow_eig_kept` (les spectres complets)**, `grow_select_loss`,
  `grow_param_update_decrease`, `adam_atten_*`
- exemple d'événement : `s=1.0`, 26 proposées → **7 gardées**, largeur → 15,
  gain premier ordre 5.87e-05, somme des vp 2.44e-04. 7 événements sur 200 époques.

**Couche export — COMPLÈTE.** `export_v5_tidy.MEAN_COLS` transporte déjà `grad_norm`,
`lr`, `grow_s`, `grow_first_order_improvement`, `grow_eig_sum`, `grow_n_kept` ;
`growth_events()` sort une ligne par événement ; `growth_io.load()` remonte
`subject`/`session`/`stop_reason`/`restored_epoch`.

**Couche figures — MANQUANTE.** Grep décisif sur tout le dépôt : `grad_norm`, `grow_s`,
`grow_eig*`, `grow_first_order_improvement` ne sont référencés **que** dans
`export_v5_tidy.py`. **Aucun code de tracé ne les lit.** `stop_reason` /
`restored_epoch` ne sont lus que par les analyses spécifiques deep4
(`deep4_lr.py`, `deep4_budget.py`, `budget_models.py`), sous forme de tables, jamais de
figures de campagne. Rien de supprimé dans l'historique git non plus : ce module n'a
jamais été écrit.

`explore_curves.py` (25/08) couvre l'*autre* moitié — courbes d'apprentissage, courbes
de perte, `growth_annotated_curves`, `growth_event_response`, `stopping_epoch`,
`width_trajectory`, `width_reached`, budget/Pareto, `selected_epoch_vs_data`.

**Reste à écrire** (8 figures, données déjà disponibles) : norme du gradient, learning
rate, spectres de valeurs propres (proposées vs gardées), gain premier ordre attendu vs
réalisé, facteur `s` de la line search, événements de croissance (neurones ajoutés par
étape et par couche), répartition des `stop_reason` par bras, et l'époque/le sujet où
chaque modèle s'est arrêté.

## 31/08 (14 h 45) — RELEVÉ : 881/1116, débit 6 CSV/h, le chemin critique frôle son plafond

**Rien relancé, rien soumis. Relevé seul.** Horloge cluster en UTC (12 h 45 UTC = 14 h 45 Paris).

| | |
|---|---|
| allocations `R` | **6** : 509142 (margpu020), 509143 (022), 509144 (017), 509145 (028), 509152 (019), 509153 (018) |
| allocations terminées | 509136–509141 (les 6 passes bon marché : `g3k2`, `g3k6`, `g3k8`, `g3k10`, `g3k7`, `g3k9`) |
| walltime restant | ≥ 5 j 23 h sur la plus vieille → **le mur de 7 j n'est pas en jeu** |
| CSV | **881 / 1116 (79 %)**, 0 CSV orphelin (sans claim) |
| claims | 910 → **29 cellules en vol**, **206 jamais réclamées** |

### Le débit a chuté d'un facteur 10, comme prévu

CSV écrits par heure (UTC) : 43–96/h jusqu'à 00 h le 31/08, puis **2 à 16/h**. Sur les
12 dernières heures : **73 CSV, soit ~6/h**. C'est l'entrée dans la queue chère
(`lee2019_mi`, `schirrmeister2017`, `cho2017`), pas une panne.

- volume restant : 206 / 6 ≈ **34 h** → ~02/09.
- makespan inchangé : `cross_subject/lee2019_mi/grow_deep__seed0` (+ sa jumelle EA) tourne
  depuis 31/08 02:13 UTC, estimée 98,1 h → **fin ~04/09 04 h UTC**. ETA campagne **~04/09**.

Les 12 cellules `cross_subject/lee2019_mi` avancent : logs tous rafraîchis dans les
minutes précédant le relevé, GPU à **99 % d'utilisation**.

### NOUVEAU — le chemin critique tape dans son propre plafond VRAM

`cross_subject/lee2019_mi/grow_shallow__seed0` et sa jumelle `__easubject__seed0` émettent
depuis 12:18 / 12:43 UTC :

```
[W831 12:42:48] CUDACachingAllocator: expandable_segments: memory mapping failed with OOM
                on device 0 while trying to map 20971520 bytes (free: 13565952, total: 11356864512)
```

Mesuré sur le nœud (`srun --jobid=509152 --overlap nvidia-smi`) : **4128 / 11264 Mio
utilisés, 9 locataires**. La carte est aux deux tiers vide — le « free: 13 Mo » est le
plafond **par process** (`g1k9`, fraction 0.094 ≈ 1,06 Gio), pas la carte. C'est
exactement le mécanisme de la famille 1.

Ce n'est **pas encore fatal** (avertissement de l'allocateur, pas d'exception ; PyTorch
retombe sur un autre chemin), mais c'est le même mur qui a tué 21 cellules, cette fois sur
des cellules à ~98 h. Si ça bascule en `OutOfMemoryError`, on perd tard la cellule la plus
chère de la campagne — et le rattrapage prévu, `within_session` bon marché, ne la couvre pas.

### Les mortes : 23 cellules, 2 échecs chacune, il leur reste 1 balayage

23 logs `pack_final` portent `OutOfMemoryError`, **toutes sans CSV**, **toutes à 2 OOM**
sur `MAX_SWEEPS=3` — un 3ᵉ balayage reste donc possible avant le report MISSING. Aucune
n'apparaît dans les 29 en vol : **les claims sont bien relâchés**, le rattrapage reste
idempotent sur `eegrow_claims_final`.

| famille | signature | cellules |
|---|---|---|
| 1 — notre plafond | `of which 6–9 GiB is free` (carte quasi vide, refus quand même) | **21** : bnci2014_001 `grow_sccnet` ×6 et `grow_shallow` ×6, bnci2015_001 `grow_shallow` ×6 + `grow_sccnet` ×1, bnci2014_002 `grow_shallow` ×2 |
| 2 — carte pleine | `of which 102.94 MiB is free. Process 4091814 has 9.51 GiB in use` | **2** : schirrmeister2017 `grow_shallow` seed1 + `easubject` seed2 (morts aujourd'hui 08:03 et 10:19) |

La famille 2 continue de tomber : les deux schirrmeister datent de ce matin, pas d'hier.
Elle est en `g3k1`, fraction déjà à 1.000 → monter la fraction n'y changera rien, seul
`--exclusive` (ou une allocation unique au lieu de trois) l'empêche.

**Décision inchangée** : rattrapage conçu, non lancé, à faire après la fin des passes en vol.

## 31/08 (16 h) — les deux « gros » contrastes décomposés : le +4.83 est de la codebase

Script : `/scratch/amounir/scratchpad_probe_two.py`. `grow − bd` se décompose en
`(grow − fix)` = la croissance, même classe, même init, mêmes callbacks, et `(fix − bd)` =
tout ce qui sépare notre implémentation de braindecode.

**`cross_session` / bnci2014_001 / EA — la croissance n'y est pour rien :**

| terme | Δ | IC 95 % | p | MDE | sujets |
|---|---|---|---|---|---|
| `grow_shallow − bd_shallow` (affiché) | +4.83 | [+3.50, +6.12] | 0.0039 | 2.02 | 9/9 |
| croissance (`grow − fix`) | **−0.23** | [−1.82, +1.32] | 0.65 | 2.38 | 4/9 |
| codebase (`fix − bd`) | **+5.06** | [+4.30, +5.73] | 0.0039 | 1.10 | 9/9 |

`fix_shallow` — le réseau construit directement à la géométrie d'arrivée, qui ne croît
jamais — fait **la totalité** de l'écart. Ce n'était pas un résultat sur la croissance.

**`cross_session` / shin2017a / EA — indécomposable :** total +4.02 (p=0.0039), mais
croissance +1.39 [−1.33, +4.08] (MDE 3.98) et codebase +2.63 [−0.50, +5.75] (MDE 4.50) :
**aucun des deux termes n'atteint son MDE**. Le design ne dit pas lequel porte l'effet.

### Le motif qui, lui, réplique

| famille | `fix − bd` | `grow − fix` |
|---|---|---|
| **shallow** (16 blocs) | **positif 16/16** (+0.12 à +5.06) | **négatif 12/16** |
| **sccnet** (22 blocs) | signe mélangé, ~0 | positif 13/22, signe instable (+5.46 weibo2014 … −4.03 alexmi) |

**Notre `fix_shallow` bat `bd_shallow` partout, et la croissance par-dessus ne rapporte
rien.** C'est le seul motif cohérent des 881 CSV — et ce n'est pas la revendication du
papier. Piste mécanique déjà notée le 26/08 : l'init Xavier stock de braindecode démarre
`bd_shallow` 0.34 nats au-dessus de ln(k) contre +0.08 pour nos réseaux.

**~~À trancher~~ TRANCHÉ (31/08, 17 h) : `fix_shallow` EST width-matched avec
`bd_shallow`.** Mesuré, `n_chans=22 n_times=1000 n_outputs=4` :

| | `bd_shallow` | `fix_shallow` |
|---|---|---|
| paramètres | **46 084** | **46 084** |
| multiset des shapes du `state_dict` | \{(40,1,25,1), (40,), (40,40,1,22), 5×(40,), (4,40,61,1), (4,)\} | **identique** |

Ce n'est donc **pas** le piège `grow_eegnex` : le +5.06 n'est pas de la géométrie. Reste
l'**initialisation**, seule différence structurelle entre les deux arms (chemin
d'entraînement identique par ailleurs : même `EEGClassifier`, même optimiseur, mêmes
callbacks, même règle de sélection ; `GromoGrowth` sort en `return` immédiat sur un arm
frozen, `skorch_integration.py:109`).

| tenseur | init braindecode (`shallow_fbcsp.py:205-218`) | init `fix_shallow` (défaut PyTorch via gromo) |
|---|---|---|
| `conv_time.weight` | Xavier `gain=1` — std 0.0445 | Kaiming-uniform — std **0.1164** (2.6×) |
| `conv_spat.weight` | Xavier — std 0.0337 | std **0.0195** |
| `conv_classifier.weight` | Xavier — std 0.0273 | std **0.0117** (2.3× plus petit) |
| `conv_classifier.bias` | `constant_(0)` | **non nul** (uniforme ±1/√fan_in) |

Le terme qui porte : Xavier sur un classifieur `(4, 40, 61, 1)` sort des logits ~2.3×
plus larges, donc une perte de départ au-dessus de ln(k). Re-mesuré sur bruit blanc, 20
graines : logit std **0.319** (bd) contre **0.137** (fix), excès sur ln(k) **+0.046** nats
contre **+0.0045** — même direction et même rapport ×10 que les +0.34 / +0.08 mesurés le
26/08 sur données réelles. `bd_shallow` passe ses premières époques à défaire son propre
init.

**Ce que ça change pour la lecture** : le +4.83 reste sans rapport avec la croissance,
mais il devient **imputable**, et à un terme qui n'a rien de fatal — un choix d'init dans
la référence. À vérifier avant d'en tirer quoi que ce soit : monter un arm
`bd_shallow_ourinit` (ShallowFBCSPNet + notre init) et voir si l'écart s'effondre. Si oui,
le résultat est « l'init Xavier de braindecode coûte 5 pp sur ShallowFBCSPNet », qui est
une contribution braindecode, pas un résultat GrowMo.

**Réserve de sélection** : ces deux contrastes sont le top-2 d'un balayage de 84 trié par
Δ décroissant. Le maximum d'un balayage est biaisé à la hausse — générateur d'hypothèses,
pas test.

## 31/08 (15 h 30) — PREMIER DÉPOUILLEMENT DES 881 CSV : rien de publiable encore

Script : `/scratch/amounir/scratchpad_audit_final.py`. 881 cellules lues, **33 113 lignes
de fit, 0 CSV illisible**. Les données sont saines ; c'est la lecture qui doit rester prudente.

### Complétude : 9 blocs complets sur 28, et ce sont les petits

Un bloc = un (protocole, dataset) à 54 cellules (9 modèles × 2 aligns × 3 seeds).

| complet (54/54) | partiel |
|---|---|
| `within` : alexmi, bnci2014_004, shin2017a, zhou2016 | `within` : cho2017 36, physionetmi 47, bnci2015_001 47, bnci2014_001 42, weibo2014 42, schirrmeister2017 **4** |
| `cross_session` : bnci2014_001, bnci2014_004, bnci2015_001, shin2017a, zhou2016 | `cross_session` : lee2019_mi **5** |
| — | `cross_subject` : **tout** (3 à 12 sur 54) |

**Le biais de survie se lit directement dans cette table.** Dans les blocs partiels des
passes *terminées*, les absentes sont presque exclusivement `grow_shallow` et
`grow_sccnet` (bnci2014_001 : les 12 cellules manquantes sont ces deux bras entiers ;
bnci2015_001 : 7 sur 7). Ces blocs-là ne peuvent pas être lus sur l'axe croissance.

### Niveau de chance : 3 cellules à surveiller

| bloc | cellule | acc | chance |
|---|---|---|---|
| `within` physionetmi | `bd_deep4` easubject | 0.492 | 0.500 |
| `within` shin2017a | `bd_deep4` easubject | 0.496 | 0.500 |
| `within` shin2017a | `grow_deep` easubject | 0.500 | 0.500 |

Le bras deep **avec EA** s'effondre au hasard sur ces deux datasets binaires. Tout
contraste qui touche ces cellules est ininterprétable — c'est l'audit v5 qui recommence.

### Contrastes appariés au sujet : 0 significatif, et ce n'est pas un résultat

84 contrastes `grow_X − bd_X` et `grow_X − fix_X` sur les 9 blocs complets, unité = le
sujet, IC bootstrap 10 000 tirages.

- **70 / 84 ont |Δ| < MDE.**
- **0 / 84 survivent à Holm** — mais c'est arithmétique, pas empirique : à n = 9 le
  Wilcoxon bilatéral plancher vaut 0.0039, donc sur une famille de 84 le meilleur `holm`
  atteignable est **0.327**. Aucun test ne *pouvait* passer. Il faudra des familles
  déclarées à l'avance et étroites, pas ce balayage.

Lus comme des candidats non corrigés (p < 0.05 **et** |Δ| > MDE), le signal est
**contradictoire selon le dataset**, pas globalement pour ou contre :

| bloc | contraste | Δ | IC 95 % | sujets |
|---|---|---|---|---|
| `cross_session` bnci2014_001 EA | grow_shallow − bd_shallow | **+4.83** | [+3.50, +6.12] | 9/9 |
| `cross_session` shin2017a EA | grow_sccnet − bd_sccnet | **+4.02** | [+1.31, +6.66] | 20/29 |
| `cross_session` bnci2014_001 EA | grow_sccnet − bd_sccnet | +1.61 | [+0.62, +2.57] | 8/9 |
| `within` bnci2014_004 | grow_sccnet − **fix**_sccnet | **−2.80** | [−4.20, −1.63] | 0/9 |
| `within` bnci2014_004 | grow_sccnet − bd_sccnet | −3.07 | [−4.37, −1.97] | 0/9 |
| `within` alexmi EA | grow_sccnet − bd_sccnet | −5.83 | [−8.82, −2.64] | 2/8 |

**Le motif le plus net est déjà connu et n'est pas un verdict sur la croissance.**
`grow_deep` perd lourdement contre `bd_deep4` (−1.4 à **−7.1 pp**, sur 4 blocs) mais est
**nul contre `fix_deepeeg`** (+0.79, +0.40, −0.29, −1.47, tous sous leur MDE). Or
`fix_deepeeg` est le seul contrôle de la même codebase : l'écart contre `bd_deep4` mesure
la codebase, pas la croissance. C'est exactement le caveat écrit le 25/08 — il réplique
sur 9 datasets au lieu d'un.

À noter : `bd_deep4` n'est plus le bras cassé de v5 (il est ici au-dessus de la chance
partout sauf en EA sur deux datasets binaires). Le correctif budget × sélection tient.

**Conclusion du dépouillement : aucun chiffre à sortir aujourd'hui.** Les blocs complets
sont les petits (n = 8–12 sujets, MDE 1–8 pp) ; le seul à avoir de la puissance,
shin2017a (n = 29), ne donne rien. Les datasets qui portent la puissance — cho2017 (52),
lee2019_mi, physionetmi — sont précisément ceux qui tournent encore.

## 31/08 (11 h 45) — POINT D'ÉTAPE : 871/1116, ETA ~04/09, 25 cellules mortes (2 causes)

**Rien n'a été relancé. Rien n'a été soumis. La grille tourne.**

### Avancement

871/1116 CSV (78 %), 435 alignés / 436 bruts. 6 allocations `R`, 0 `PD`, walltime 7 j
(la plus vieille à 23 h). Overrides vérifiés dans les logs :
`train.patience=200 train.selection_monitor=valid_acc` — la campagne finale est bien dans
la bonne case du carré 2×2, contrairement au gate cross-dataset du 29/08.

| pass | fraction | fait / total | reste |
|---|---|---|---|
| `g3k2` | 0.425 | 26 / 26 | — |
| `g3k6` | 0.141 | 26 / 26 | — |
| `g3k8` | 0.106 | 52 / 52 | — |
| `g3k10` | 0.084 | 653 / 666 | 13 trous OOM, **pass terminée** |
| `g3k7` | 0.121 | 20 / 26 | 6 trous OOM, **pass terminée** |
| `g3k9` | 0.094 | 38 / 40 | 2 trous OOM, **pass terminée** |
| `g2k6` | 0.141 | 16 / 52 | en vol |
| `g3k1` | 1.000 | 6 / 82 | en vol (+4 trous OOM) |
| `g1k3` | 0.283 | 0 / 52 | en vol |
| `g1k9` | 0.094 | 0 / 94 | en vol |

245 manquants = **220 à calculer** + **25 mortes**. 30 cellules en vol (claims sans CSV).

### ETA

Le débit s'est effondré en entrant dans la queue chère : 40–120 CSV/h entre h-22 et h-10,
puis **5,5 CSV/h sur les 10 dernières heures** (le reste est concentré sur `lee2019_mi` 116,
`schirrmeister2017` 60, `cho2017` 25).

- Gros du volume : 220 / 5,5 ≈ **40 h**, soit ~02/09.
- **Contrainte réelle = makespan** : `cross_subject/lee2019_mi/grow_deep__seed0` et
  `__easubject__seed0`, démarrées **31/08 02:13**, estimées 98,1 h → **fin vers 04/09 04 h**.
  Les 12 cellules `cross_subject lee2019_mi` sont toutes en vol depuis 02:13 sauf
  `bd_sccnet__easubject__seed0` (pas commencée).

**ETA campagne complète : 3 à 4 jours, ~04/09.** Le walltime 7 j n'est pas menacé.

### Les 25 cellules mortes — DEUX pannes différentes, pas une

Tous les trous des passes terminées sont des OOM. **0 récupérée** par les 3 sweeps : les
3 tentatives se heurtent au même mur.

**Famille 1 — notre propre plafond (21 cellules)**, `g3k10` / `g3k7` / `g3k9` :

```
Tried to allocate ... GPU 0 has a total capacity of 10.58 GiB of which 9.63 GiB is free
```

Carte quasi vide, refus quand même → c'est `EEGROW_CUDA_FRACTION`, dérivé par
`profile_grid_memory.py` **sur le réseau à l'initialisation**. Les bras `grow_*`
s'élargissent en cours de route et sortent du plafond au pic de `compute_s_update`.
**Réparable** : relancer ces cellules à une fraction plus haute.

**Famille 2 — la carte est réellement pleine (4 cellules)**, `g3k1`, fraction déjà à 1.000 :

```
GPU 0 has a total capacity of 10.58 GiB of which 1.02 GiB is free.
Process 1919980 has 5.68 GiB memory in use.
```

Le voisin, c'est nous. `g3k1` est soumis en **3 allocations qui coopèrent sur la même
grille** (lignes 13–15 de `passes_final/submit.sh`), chacune `--gres=gpu:turing:3` avec
`K=1` et `fraction=1.000`. Ce n'est sûr que si SLURM ne co-planifie jamais deux de ces
allocations sur le même nœud. Il l'a fait (OOM du 31/08 à 03:37 et 06:29).
**Monter la fraction ne réparera rien ici** — elle est déjà au maximum.

Cellules touchées : uniquement `grow_shallow` et `grow_sccnet`, uniquement `within_session`,
sur bnci2014_001 / bnci2014_002 / bnci2015_001 / schirrmeister2017 / physionetmi.

### Pourquoi c'est un problème de papier et pas d'ordonnanceur

Ce qui meurt, ce sont **les cellules qui ont le plus grossi**. Les retirer biaise
l'échantillon survivant vers celles qui ont le moins grossi — dans le sens flatteur, et
exactement sur l'axe de la revendication (efficacité paramétrique). Ces 25 cellules ne
peuvent pas être simplement déclarées manquantes.

### Passe de rattrapage — conçue, NON lancée

**Correction par rapport à la note du 30/08** : les 25 cellules ont bien **libéré leur
claim** (vérifié : `claims − CSV = 30`, dont 0 OOM). Un rattrapage peut donc réutiliser
`eegrow_claims_final` sans répertoire neuf, il re-prendra les cellules sans CSV.

Design : **une seule** allocation `G=3 K=1 fraction=1.000 --exclusive`, même `RESULTS_DIR`.
Le `--exclusive` (ou une allocation unique au lieu de trois) est le point non négociable,
sinon on refait la collision de la famille 2. À lancer **après** la fin des passes en vol,
pour ne pas leur prendre de GPU.

### Décision sur l'extension poolée × interpolation : NON

Le gate cross-dataset du 29/08 a déjà tourné (52 sujets, cho2017, sujets comme unité,
IC bootstrap appariés) : `pooled` n'est **jamais** significativement meilleur que `within`,
`lodo` est significativement **pire** dans les 6/6 conditions (−1,4 à −2,7 pp),
l'interpolation plafonne à ±0,8 pp avec un signe qui s'inverse selon le modèle.
Réserve : ce gate tournait avec `selection_monitor=valid_loss` (défaut config, pas
d'override sbatch), donc lecture directionnelle seulement — mais Q1/Q2 tiennent le modèle
fixe, l'erreur s'y annule largement.

→ On n'ajoute pas de section poolée. On garde le **négatif LODO** comme paragraphe de
section « limites » : significatif, cohérent sur 6 conditions, plus solide qu'une table
à moins d'1 pp non significatif.

---

## 30/08 (22 h) — 5e BLOQUANT EN VOL : 13 cellules `grow_shallow` tuées par le PLAFOND VRAM

**À décider avant la fin de la campagne. Rien n'a été relancé.**

322/1116 CSV écrits, 0 corruption, mais **13 cellules ont échoué en `torch.OutOfMemoryError`**,
toutes `grow_shallow`, toutes `within_session`, sur bnci2014_001 / bnci2014_002 / bnci2015_001.

Ce n'est pas une contention réelle. Le log dit :

```
GPU 0 has a total capacity of 10.58 GiB of which 9.28 GiB is free.
this process has 1.29 GiB memory in use. 1.28 GiB allowed
```

**9.28 Gio libres sur la carte** et le processus est refusé à 1.28 Gio : le plafond est
`EEGROW_CUDA_FRACTION`, que `plan_campaign` dérive de `profile_grid_memory.py`. Or ce
profilage mesure le réseau **avant croissance**. Les bras `grow_*` s'élargissent en cours
d'entraînement, donc l'empreinte dépasse un plafond calibré à l'initialisation. Le pic est
atteint dans `gromo/modules/conv2d_growing_module.py:1019` (`compute_s_update`, einsum de la
statistique S — elle scale en `(C·k)²`, donc elle grandit avec le réseau).

**Preuve que c'est la croissance et pas le dataset** : `bnci2014_002 grow_shallow easubject`
— seed1 **réussit**, seed0 et seed2 **OOM**. Même config, même passe, même fraction ; seule
la trajectoire de croissance diffère.

**Pourquoi ça compte pour le papier, et pas seulement pour le cluster.** Les cellules tuées
sont celles qui ont le plus grandi. Les laisser tomber ne coûte pas 1 % de cellules au
hasard : ça biaise l'échantillon survivant vers les cellules qui ont le **moins** grandi —
un biais de survivant sur l'axe exact de la revendication (efficacité paramétrique). Un
résultat obtenu comme ça est ininterprétable dans la bonne direction.

**Récupérable.** `pack_run.sh` relâche le claim en cas d'échec et rejoue la cellule jusqu'à
`MAX_SWEEPS=3`, puis la reporte MISSING (les commentaires du script décrivent exactement ce
scénario, l. 126-136). Donc ces 13 cellules vont brûler 2 balayages de plus **au même
plafond**, échouer pareil, et finir MISSING. Aucun CSV corrompu n'est écrit.

Exposition (cellules `grow_*` par passe, fraction) :

| passe | fraction | plafond | `grow_*` |
|---|---|---|---|
| `g3k10` | 0.084 | 0.89 Gio | **210** |
| `g3k9` / `g1k9` | 0.094 | 0.99 Gio | 22 / 28 |
| `g3k8` | 0.106 | 1.12 Gio | 16 |
| `g3k7` | 0.121 | 1.28 Gio | 14 |
| `g3k6` / `g2k6` | 0.141 | 1.49 Gio | 14 / 16 |
| `g1k3` | 0.283 | 2.99 Gio | 16 |
| `g3k2` | 0.425 | 4.50 Gio | 14 |
| `g3k1` | 1.000 | 10.58 Gio | 46 |

156 cellules `grow_*` sont déjà passées proprement, donc le plafond suffit pour la majorité :
c'est la queue de distribution des trajectoires de croissance qui déborde.

**Option recommandée (non lancée, attend le go d'Adam) :** ne rien perturber — la campagne
tourne et le système de claims rend un rattrapage idempotent — et préparer une **passe de
rattrapage `fraction=1.000`** qui relit le même `CLAIMS`/`RESULTS_DIR` et ne prend que les
cellules sans CSV. Elle ne coûte rien sur le chemin critique (les manquantes sont
`within_session`, le protocole bon marché) et supprime le biais de survivant. Le correctif
de fond, pour la suite, est de profiler le pic **après** croissance et non à l'init.

## 30/08 — LA GRILLE FINALE EST LANCÉE : jobs 509136–509145, 509152, 509153

**1116 cellules, 1740 GPU-h, 12 allocations, sortie dans `results_final`.** Commit
déployé et estampillé sur chaque ligne de résultat : `58c6bee`. Makespan attendu
**4–5 jours**, borné par une seule cellule (`cross_subject/lee2019_mi/grow_deep`, 98.1 h).

Décision d'Adam : **EA sur les trois paradigmes**, pas seulement les deux protocoles bon
marché. +72 cellules (`cross_subject × euclidean`), +733 GPU-h, makespan inchangé — la
cellule alignée tourne à côté de sa jumelle brute, pas après elle.

### Où en est la campagne

```
ssh margaret02 'squeue -u $USER -o "%.8i %.4t %.9M %R"
  find /scratch/amounir/results_final -name "*.csv" | wc -l   # sur 1116
  grep -hi "FATAL\|Traceback" /scratch/amounir/logs/final_*.log | head'
```

| au lancement | |
|---|---|
| jobs | 5 R (margpu018,019,020,022,028) + 7 PD (Priority) |
| gardes de provenance | **5/5 passés** — `sha=58c6beebc12e`, abstention s=0 présente |
| cache | `cache prêt pour la clé de la campagne`, 12 datasets |
| cellules réclamées | 267 après 25 min, logs de 87–111 Ko (entraînement actif) |
| RAM mesurée sur nœud | 142 / **187** Go (margpu019, 40 locataires) |
| erreurs | aucune |

### Un 4e bloquant, trouvé APRÈS la soumission — jobs 509146/509147 repris

`pack_run.sh` fait `G="${G:-${SLURM_GPUS_ON_NODE:-1}}"` et `plan_campaign` n'exportait
pas `G`. Sous `--exclusive`, SLURM accorde le **nœud entier** : `SLURM_GPUS_ON_NODE` est
le nombre de cartes *physiques*, jamais le `--gres` demandé. Mesuré : une passe
`--gres=gpu:turing:3` a démarré en `G=4` sur margpu019 ; margpu[018,022] ont 3 cartes,
margpu028 en a 2 — le nombre de locataires dépendait donc du nœud tiré au sort.

Sur les passes bornées par le GPU c'est de la capacité gratuite. Sur les passes que le
planificateur avait **délibérément rétrécies** pour tenir la RAM, `G` *est* la décision
qu'on jetait, puisque les locataires sont `G×K` :

| passe | G planifié | si G=4 | RAM demandée | contre |
|---|---|---|---|---|
| `g1k9` lee2019_mi | 1 | 36 locataires | 36 × 11.5 = **414 Go** | 187 Go |
| `g1k3` schirrmeister2017 | 1 | 12 locataires | 12 × 26.4 = **325 Go** | 187 Go |
| `g2k6` cho2017 | 2 | 24 locataires | 202 Go | sauvée : margpu028 n'a que 2 cartes |

`--mem` ne l'arrête pas — il n'est pas appliqué ici (`AllocMem=0`), donc l'arbitre est
l'OOM killer du noyau, qui envoie SIGKILL, qu'aucun trap n'attrape : l'incident qui a
coûté 85 cellules à v5.

Les deux passes condamnées étaient **PENDING**, donc rien de perdu : `scancel` (0 CSV,
0 claim orphelin) puis re-soumission avec `G=1` explicite → **509152, 509153**. Les
passes qui tournaient déjà tiennent même à G=4 (pire cas `g3k1` : 4 × 26.4 = 108 Go
contre 187) — je ne les ai pas touchées. Corrigé dans `plan_campaign.py` (`30bda87`) ;
le `submit.sh` du cluster est patché sur place (`submit.sh.orig-nog` conservé) pour
qu'une relance reparte juste. **Non redéployé** : les jobs en cours lisent `pack_run.sh`
depuis le disque et le réécrire sous eux est un risque gratuit.

**Allocations finales : 509136–509145 + 509152 + 509153.**

**Reprise après interruption** : les allocations coopèrent par le répertoire de claims
atomique (`/scratch/amounir/eegrow_claims_final`) et une cellule est « faite » quand
`<stem>.csv` existe — donc re-soumettre `/scratch/amounir/passes_final/submit.sh` reprend
où la campagne s'est arrêtée, elle ne recommence pas.

### Ce que l'audit avait trouvé avant de lancer (3 bloquants, tous corrigés)

| # | trouvé | conséquence si non corrigé | commit |
|---|---|---|---|
| 1 | `plan_campaign.py` lisait 4 colonnes, la grille en émet 6 | crash — puis, si le parseur avait été tolérant, la colonne `align` sautait à la ré-écriture des passes et **chaque cellule alignée aurait tourné en brut en écrasant sa jumelle** : 1044 CSV bien formés, aucune erreur | `10b8b54` |
| 2 | 216 cellules (**tout le bras `fix_*`**) exclues en silence, faute de mesure mémoire | l'ablation qui isole l'apport de la croissance absente d'une campagne qui se déclare complète | `10b8b54` |
| 3 | `/scratch/amounir/eegrow_budget` n'est pas un dépôt git | `eegrow_sha=None` sur **chaque ligne** de la campagne finale — exactement l'intraçabilité qui a coûté 1170 cellules v5 | `4ef76ee` |

Corrections associées : `fix_X` emprunte la mesure de `grow_X` (même géométrie terminale,
vérifiée champ par champ : 40/40, 22/22, 32/32 — borne supérieure, donc sûre) ;
`plan_campaign` refuse désormais d'écrire `submit.sh` tant qu'une cellule n'est pas placée ;
`pack_run.sh` refuse de démarrer si l'arbre ne sait pas nommer son commit.

### Vérifié, mesuré

| point | résultat |
|---|---|
| cache epochs (12 datasets, 250 Hz) | **501 entrées, 0 manquante, 0 mensonge** |
| clé de cache MOABB brut vs aligné | **identique** (`5d65788e…`) → le bras EA est un cache hit, coût ×1.00 |
| placement (grille finale, EA partout) | **1116 / 1116** cellules, 10 passes, exit 0 |
| colonnes des TSV de passe | 6 ; 558 `none` + 558 `euclidean` ; 0 stem dupliqué ; ré-union des passes **identique** à la grille source |
| coût | 1740 GPU-h ; 92.3 node-fulls de travail sérialisé |
| déploiement | 60 fichiers suivis, **sha identique des deux côtés** |
| **pire cellule** | **98.1 h** (`cross_subject/lee2019_mi/grow_deep`) → makespan ≈ 4.1 j |
| `CELL_TIMEOUT` | 144 h (47 % de marge), sous le mur de 7 j |
| nœuds turing utilisables | margpu[017-020,022,028] = **20 GPU** ; 021 (5.4 Go libres, GPU fantôme) et 023 (drained) exclus |
| baselines ML | **déjà faites** : 498/498 cellules dans `results_v5`, 12 datasets, 3 protocoles, ligne de provenance complète |
| dérive de config depuis les ML (19/08) | **aucune** → les lignes ML s'apparient avec la grille finale |
| disque | 45 T / 50 T (90 %), 5.2 T libres |

Les cellules ML portent `csp_lda__seed0.csv` et non `ml_csp_lda__…` (`ml_v5.sbatch`
retire le préfixe) : c'est ce qui m'avait fait conclure à tort qu'elles manquaient.

### Séquence exécutée le 30/08

```
python benchmarks/slurm/final_grid.py \
    --align-evals within_session cross_session cross_subject \
    --out benchmarks/slurm/final_grid.tsv
bash   benchmarks/slurm/deploy_final.sh
# les deux commandes suivantes SUR le cluster (conda activate bench) :
python benchmarks/slurm/plan_campaign.py \
    --grid /scratch/amounir/eegrow_budget/benchmarks/slurm/final_grid.tsv \
    --outdir /scratch/amounir/passes_final --root /scratch/amounir/eegrow_budget \
    --tag final --gres-type turing --wrapper benchmarks/slurm/final_grid.sbatch
bash   /scratch/amounir/passes_final/submit.sh
```

Deux pièges de cette séquence, tous deux silencieux :

- `--align-evals` **doit** être passé à la génération. Le défaut du script n'allume
  l'aligné que sur les deux protocoles bon marché ; regénérer la grille sans le flag
  retire 72 cellules `cross_subject` sans rien signaler.
- `--gres-type turing` n'est pas optionnel : sans lui `plan_campaign` émet `gpu:N`, qui
  écrase l'en-tête du wrapper, et une différence appariée grow_X − bd_X peut alors
  enjamber deux classes de cartes.

`plan_campaign` doit tourner **sur le cluster**, pas en local : les chemins `GRID=` qu'il
écrit dans `submit.sh` sont ceux de la machine où il s'exécute.

## 29/08 (soir) — les 4 arrays cho2017 sont terminées ; le gate est re-mesuré

| job | état |
|---|---|
| `504709` + `505009` pooled | **18/18** — bloc complet |
| `505184` lodo | 18/18 |
| `505185` tiers | 11/12 (index 10 mort) |
| `507767_10` relance | PENDING |

### Décomposition du gate, protocole corrigé, un seul arbre (n=52 sujets)

36 cellules `core`, toutes dans `eegrow_xds`, 27/08 14h22 → 29/08 16h03, `gpu:turing`.
Aucun appariement cross-arbre. Scripts `scratchpad/within_cho/{falsif,decompo}.py`.

| contraste | Δ | IC 95 % | p | holm | MDE |
|---|---|---|---|---|---|
| **GATE** grow+pooled+EA − bd+within+rien | **+3.20 pp** | [+1.72, +4.73] | 0.0001 | 0.0009 | 2.21 |
| EA seule (bd, within) | +1.89 | [+0.55, +3.25] | 0.0094 | 0.057 | 2.00 |
| croissance (within, EA) | +0.75 | [+0.17, +1.33] | 0.0165 | 0.082 | 0.87 |
| pooling seul (bd, EA) | +0.76 | [−0.09, +1.65] | 0.095 | 0.21 | 1.28 |
| croissance @ pooled/EA | +0.55 | [+0.04, +1.04] | 0.036 | 0.15 | 0.73 |
| rescaling seul — **contrôle négatif** | +0.12 | [−0.57, +0.84] | 0.75 | 0.75 | 1.05 |

**Ce qui change par rapport aux chiffres d'août.** Le pooling passe de **+1.79 → +0.76 pp**
et ne survit plus à Holm : l'essentiel du +1.79 était l'écart de protocole, pas le pooling.
L'ancien gate (baseline déjà alignée) passe de +2.19 → **+1.32 pp**. L'EA, elle, est stable
à +1.89. Le vrai gate « tout contre rien » vaut +3.20 pp et n'avait jamais été mesuré.

Le **contrôle négatif passe** : `scale` ne fait rien (+0.12 pp, p=0.75). Le gain de l'EA est
donc bien du blanchiment, pas de la normalisation d'amplitude.

### L'EA profite-t-elle plus au modèle qui croît ? (interaction)

| bras / base | Δ | IC 95 % | p | holm | MDE |
|---|---|---|---|---|---|
| within, base `scale` | **+1.06** | [+0.32, +1.81] | 0.0082 | **0.041** | 1.10 |
| within, base `none` | +0.80 | [−0.07, +1.68] | 0.084 | 0.20 | 1.30 |
| pooled, base `scale` | +0.75 | [+0.12, +1.40] | 0.027 | 0.11 | 0.94 |
| pooled, base `none` | +0.72 | [−0.01, +1.48] | 0.066 | 0.20 | 1.09 |
| **pooled − within** | −0.08 | [−1.22, +1.04] | 0.89 | 0.89 | 1.66 |

Les quatre estimations sont **du même signe et de la même taille** (+0.7 à +1.1 pp), mais
l'effet (~0.8) est sous le MDE (~1.0–1.3) : cohérent, pas encore établi. Une seule survit à
Holm. À traiter comme un signal à confirmer, pas comme un résultat.

**Le mécanisme « amplitude » est réfuté pour de bon.** Prédiction : l'interaction devait
disparaître sur `within` (un seul dataset, un seul amplificateur). Elle est **identique**
(différence −0.08 pp, p=0.89). Réserve honnête : MDE 1.66 pp > l'interaction elle-même, donc
on exclut la disparition, pas une atténuation partielle. Le motif d'août (significatif contre
`none`, nul contre `scale`) ne réplique pas non plus — c'était un artefact de protocole.

**Conséquence de design** : l'interaction ne dépend pas du pooling. On n'a donc pas besoin de
pooler pour capter la synergie EA × croissance.

## 29/08 (suite) — cellule perdue relancée ; le stamp sujet est branché

| job | état |
|---|---|
| `507767_10` tiers (relance) | PENDING — `grow_shallow`, `euclidean`, seed 1, `core+lowrank` |
| `505009` pooled grow | 3 RUNNING (9 h 52 / 11 h 38 / 11 h 41) |

Relancé avec `--exclude=margpu021,margpu028`. margpu021 est le nœud défectueux connu ;
margpu028 est exclu **pour rendre la relance informative**, pas parce qu'il est suspect
(il fait tourner `505009_4` depuis 11 h sans incident). Si la cellule repasse ailleurs on
n'apprend rien sur 028, mais on a la cellule ; si elle échoue encore, le défaut est dans
la cellule et non dans la carte. Une seule occurrence ne permet pas de trancher autrement.

### Stamp sujet — fait, testé, commité (`e9dc622`)

Chaque record de fit porte désormais `subject` / `session` / `cv_ind`. C'était la 8ᵉ
figure demandée par Stella et la **seule** qui ne se rattrape pas après coup : une colonne
absente de 396 cellules coûte les 817 GPU-h une deuxième fois.

Un piège évité, qui vaut d'être écrit : le hook évident,
`BaseEvaluation._maybe_save_model_cv`, reçoit bien le sujet — mais il n'est appelé que
depuis `_process_legacy`. En instrumentant le compteur sur une `WithinSessionEvaluation`
complète : **0 appel**. MOABB 1.5.0 passe par `_process_parallel` → `_build_task_list` →
`_evaluate_fold`. Un override là aurait produit 396 fichiers vides et aucune erreur.
D'où les deux gardes qui font échouer la cellule à sa première seconde plutôt qu'à sa
dernière (pas de splitter, ou `n_jobs != 1`). Vérifié : rien dans `slurm/` ne surcharge
`n_jobs`, `config.yaml` le fixe à 1.

Testé sur les trois protocoles contre le vrai flot MOABB (sujets complets, équilibrés
entre folds, distincts par record) et sur les cinq bras profonds de la grille finale
(listes de callbacks préservées élément par élément, recorder retrouvé dans chacune).

## 29/08 — tiers et lodo terminés, 1 cellule perdue sur 79 ; pooled encore en vol

| job | état | détail |
|---|---|---|
| `505184` lodo | **18/18 COMPLETED** | terminé |
| `505185` tiers | **11/12 COMPLETED, 1 FAILED** | l'index 10 (`grow_shallow`, `euclidean`, seed 1, `core+lowrank`) est mort |
| `505009` pooled grow | 3 tâches RUNNING (9 h 35 / 11 h 20 / 11 h 23) | dernier bloc |

68 CSV sur disque pour cho2017 : `core` 51, `core+interp` 6, `core+extrap` 6,
**`core+lowrank` 5 sur 6** — le trou est exactement la cellule perdue. La réponse-volume
à quatre points est bloquée dessus.

**Ce qui a tué la cellule**, et pourquoi ce n'est pas ce que la stack trace raconte :

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  callbacks.py:201 in on_grad_computed  ->  p.grad.detach().norm(2)
```

Les erreurs CUDA sont rapportées de façon asynchrone : le callback `GradientNorm` est le
premier endroit qui *synchronise* (`.item()`), donc c'est là que la faute remonte, pas là
où elle naît. margpu028 est sain (`mixed`, `gpu:turing:2`, et `505009_4` y tourne depuis
11 h). Une occurrence sur 79 cellules.

**Les 10 autres échecs des logs `xds_*` ne sont pas le même problème** et ne sont pas
nouveaux : tous viennent du job `504278`, tous sur **margpu021**, tous avec

```
RuntimeError: CUDA_VISIBLE_DEVICES='0' but torch.cuda.is_available() is False:
              the pinned device does not exist. Refusing to fall back to CPU.
```

C'est le défaut connu de margpu021 (il annonce `gpu:turing:3`, `nvidia-smi` n'en voit
aucun). Le garde a refusé de basculer sur CPU — comportement correct : un échec bruyant
plutôt qu'une cellule CPU silencieuse qui aurait l'air normale. margpu021 est déjà dans
l'`--exclude` de `final_grid.sbatch`.

**Conséquence pour la grille finale : aucune.** Les arrays `xds_*` sont indexés, donc un
index mort est perdu sec. `pack_run.sh` ne l'est pas — il relâche la revendication quand
aucun CSV n'a été écrit (`[ -s "$OUT" ] || rmdir "$CLAIM"`, l. 344) et la cellule est
re-revendiquée au balayage suivant. Une faute CUDA transitoire y coûte une re-exécution,
pas une cellule.

## 28/08 fin d'aprem — point cross-dataset cho2017 : 60 CSV, 2 jobs restants

`504709` (pooled bd) est **terminé** — les 6 cellules `pooled core` de `bd_shallow`
(none / scale / euclidean × 3 graines) sont sur disque. 60 CSV = **20 cellules complètes
× 3 graines**, aucune cellule partielle.

| job | cellules | état | reste |
|---|---|---|---|
| `505009` pooled grow | `grow_shallow pooled core`, aligns `none` (tâches 0-2) puis `scale` (3-5) | 0/1/2 RUNNING 7 h 36 (walltime 24 h), 3-5 PENDING | bras `none` ≈ 17 h/cellule → ~9 h ; puis `scale` ≈ 3 h 30 |
| `505185` tiers | `core+lowrank`, `bd_shallow` (6-8) puis `grow_shallow` (9-11) | 6/7/8 RUNNING 39 min–1 h 04 (walltime 10 h), 9-11 PENDING | bd ≈ 2 h ; puis grow ≈ 3 h 30 |

`core+interp` (tâches 0-5 de `505185`) est **complet**, 6 cellules. `core+extrap` complet
(package B). Il manque donc, pour fermer l'étage 2 : `pooled none/scale` du bras growing
et le tier `core+lowrank`. **ETA global ≈ 13 h** — tout devrait être sur disque demain
matin 29/08.

Analyses débloquées à ce moment-là : `scratchpad/within_cho/falsif.py` au nouveau
protocole (gate + effets principaux + interaction croissance × alignement), puis la
réponse-volume à quatre points (core / +lowrank / +interp / +extrap) et le contraste de
rang `interp` vs `lowrank` à volume quasi apparié, n=52, apparié sujet, IC bootstrap + Holm.

## 28/08 — audit des deux bugs bloquants : **déjà corrigés**, un troisième piège reste

Les deux bugs qui bloquaient le relancement de la grille principale étaient corrigés
depuis plusieurs jours ; les notes qui les disaient ouverts étaient périmées.

| bug | commit | date | test de régression |
|---|---|---|---|
| `drop_last=True` → 0 batch d'entraînement | `3cefa3a` | 23/08 | `test_callback_skips_growth_on_empty_iterator` |
| line search à s=0 → neurones morts | `5337c56` | 25/08 | `test_grow_step_abstains_instead_of_adding_dead_neurons` |

Les deux tests passent (28/08, 2 passed / 3.33 s).

**Déploiement vérifié fichier par fichier.** `/scratch/amounir/eegrow_budget` (le checkout
H100 destiné à la grille réduite) comparé à HEAD local : 48 fichiers `.py`, hashes
identiques sauf `benchmarks/pipelines.py` et `src/eegrow/training/callbacks.py`, dont le
diff est **purement documentaire** — AST identiques une fois les docstrings retirées. Il
manque seulement 4 utilitaires de lancement (`benchmarks/slurm/{estimate_eta,make_grid,
make_ml_grid,plan_campaign}.py`), à déployer par `scp` avant de construire la grille.
`eegrow_xds` a les deux correctifs. `/scratch/amounir/eegrow` (arbre d'août) a `drop_last`
mais **pas** l'abstention s=0 — une raison de plus de n'y apparier aucune cellule.

**Le vrai risque restant n'est pas un bug, c'est un défaut de configuration.**
`config.yaml` livre encore `selection_monitor: valid_loss` et `patience: null` (=20),
volontairement, pour ne pas re-dater `results_v5_published/`. Le protocole corrigé doit
donc être passé **explicitement** :

```
train.patience=200 train.selection_monitor=valid_acc
```

Une grille soumise sans ces deux overrides refait le protocole sous-entraîné, sans le
moindre avertissement et avec des CSV bien formés — jusqu'à **+0.13** d'accuracy sur
`bd_deep4`, soit davantage que les deux bugs réunis. Le commentaire de `config.yaml` qui
présentait encore la question comme ouverte a été corrigé pour dire l'inverse.

## 28/08 — `lodo` (505184) : 18/18, le transfert zéro-shot marche

18 cellules en ~1 h de GPU au total. Unité d'analyse = le sujet (3 seeds moyennés dedans),
IC bootstrap 10 000 tirages, Holm sur les 6 cellules.

**Le zéro-shot bat largement le hasard.** Aucune cellule ne s'en approche.

| modèle | align | acc | Δ vs 0.5 | IC 95 % | sujets > 0.5 | p (Holm) |
|---|---|---|---|---|---|---|
| bd_shallow | euclidean | 66.73 | **+16.73** | [+13.95, +19.63] | 51/52 | <0.001 |
| bd_shallow | none | 63.90 | +13.90 | [+11.17, +16.81] | 52/52 | <0.001 |
| bd_shallow | scale | 64.12 | +14.12 | [+11.47, +16.94] | 51/52 | <0.001 |
| grow_shallow | euclidean | 66.45 | **+16.45** | [+13.66, +19.31] | 50/52 | <0.001 |
| grow_shallow | none | 64.76 | +14.76 | [+12.01, +17.65] | 52/52 | <0.001 |
| grow_shallow | scale | 63.32 | +13.32 | [+10.62, +16.28] | 52/52 | <0.001 |

**Ne jamais voir la cible coûte 1.4 à 2.7 pp**, pas plus (`lodo − within`, apparié sujet) :
bd euclidean −1.39 [−2.59, −0.15] (p=0.061, MDE 1.79) ; bd none −2.33 (p=0.008) ;
bd scale −2.24 (p=0.012) ; grow euclidean −2.42 [−3.37, −1.46] (p<0.001, MDE 1.41) ;
grow none −1.42 (p=0.061) ; grow scale −2.73 (p=0.001).

**Le blanchiment paie dans les deux bras** (`euclidean − scale`, le seul contraste qui
isole la décorrélation spatiale, `scale` ayant déjà normalisé l'amplitude) : within bd
+1.77 [+0.61, +2.97], within grow +2.83 [+1.57, +4.13], lodo bd +2.62 [+1.55, +3.67],
lodo grow +3.13 [+2.07, +4.21] — les quatre survivent à Holm.

**Il paie-t-il *plus* en zéro-shot ? Pas de façon détectable.** L'interaction vaut +0.85 pp
[−0.53, +2.19] (bd) et +0.31 [−1.02, +1.66] (grow), **MDE ≈ 1.9 pp**. Les moyennes
suggéraient le contraire (2.83 → 3.13 d'écart apparent) ; le test apparié ne le confirme
pas. À ne pas écrire comme un effet. Ce design ne peut rien dire sous ~1.9 pp.

À noter pour la table de provenance : la colonne `pool_datasets` des CSV `lodo` liste le
tier entier (cho2017 compris) alors que `training_sets` retire bien le dataset cible et
qu'une `RuntimeError` garde contre la fuite. C'est l'étiquette qui est fausse, pas les
données. À corriger avant publication.

Script : `benchmarks/analyse_lodo.py` (sur le cluster).

## Tableau de bord — 28/08, 4 arrays en vol sur cho2017

Tout tourne sur le **même arbre** (`/scratch/amounir/eegrow_xds`), le **même protocole**
(post-`RestoreBestModel`) et la **même carte** (`gpu:turing`, `margpu021` exclu). C'est ce
qui autorise à apparier n'importe laquelle de ces cellules avec n'importe quelle autre.

| job | bloc | cellules | walltime | fin estimée |
|---|---|---|---|---|
| **504709** | `pooled`/`core`, les 12 cellules bd + grow `euclidean` | 12 | 10 h | ce soir |
| **505009** | `pooled`/`core`, les 6 cellules grow non-`euclidean` (rattrapage walltime) | 6 | 24 h | 30/08 midi |
| **505184** | `lodo`/`core`, 2 modèles × 3 alignements × 3 seeds | 18 | 6 h | ce soir |
| **505185** | `pooled` `euclidean`, tiers `core+interp` et `core+lowrank` | 12 | 10 h | demain |

Le bloc `within` (18 cellules, 3 alignements × 2 modèles × 3 seeds) est **déjà complet** au
nouveau protocole. Donc dès que `pooled` tombe, le carré `arm × align × model` se calcule
sans rien relancer.

Moniteur unique : `bjq7bfde6` (échec de cellule, complétion de chaque bloc, fin des quatre).

### Pourquoi `lodo` (505184) coûte presque rien

`cross_dataset.py` : le jeu d'entraînement du bras zéro-shot ne dépend pas du fold, donc
`if arm == "lodo": groups = [[...tous les sujets...]]` — **1 fit** scoré sur les 52 sujets,
contre 52 fits pour une cellule `pooled`. Soit ~1.5 % du coût, sur un pool plus petit en
prime (26 430 essais, cho2017 entièrement retiré). Les 18 cellules ≈ 1 GPU-h au total.

Ce que ce bras répond et qu'aucun autre ne répond : dans `pooled`, la cible est dans le jeu
d'entraînement, donc un modèle qui ignorerait complètement les autres datasets scorerait
déjà bien. `lodo` mesure si les 8 autres datasets portent une information **transférable**
vers cho2017. Question de fond du pooling, sans réponse jusqu'ici.

## ⚠️ La grille cross-dataset d'août est obsolète (27/08, après 504342)

**504342 TERMINÉ 18/18, rc=0, aucun `PREFLIGHT_FAIL`.** Le bloc `within` de cho2017 est
complet. Il devait retirer le type de carte comme décalage systématique ; il a trouvé
beaucoup plus gros.

Le replicat `within/euclidean` **ne réplique pas** la cellule d'août :

| modèle | août | 27/08 | écart | IC 95 % | p | sujets |
|---|---|---|---|---|---|---|
| bd_shallow | 66.11 | 68.12 | **+2.01 pp** | [+1.13, +2.83] | 3e-6 | 42/52 |
| grow_shallow | 66.56 | 68.87 | **+2.31 pp** | [+1.57, +3.05] | 1e-8 | 39/52 |

r = 0.97 entre replicats : décalage uniforme, pas du bruit. Même cache pool, même cible,
`n_train_trials` et `n_test` identiques au fold près.

**Ce n'est pas la carte GPU, c'est le protocole d'entraînement.** Commit `0efbdb5` (25/08),
`RestoreBestModel` : `EEGClassifier` n'embarque aucun `Checkpoint` et le `EarlyStopping` de
skorch a `load_best=False`, donc **tout score publié par ce benchmark venait du modèle 20
époques après son propre optimum** (`epochs - epoch_of_best` = 20, std 0.0 sur les 140 490
folds de v5). S'y ajoutent `5ac8b24` (budget explicite) et `5337c56` (abstention à s=0).
Preuve que ce n'est pas un budget plus long : `fit_seconds` de bd_shallow est inchangé
(29.87 → 29.62 s) pour +2.01 pp. grow_shallow, lui, double (30.5 → 57.6 s).

### Ce que ça invalide

Tous les chiffres de tête de la section « Résultats déjà en main » plus bas — gate
+2.19 pp, pooling seul +1.79/+1.74, blanchiment net +1.72, interaction +1.29 — sont
**internement cohérents mais mesurés avec un protocole depuis prouvé défectueux**. Ils ne
peuvent pas aller dans le papier tels quels.

Recalculés en appariant l'ancien `pooled` au nouveau `within`, le gate tombe à +0.18 pp
(p = 0.70, 21/52) et le pooling à −0.22 pp (p = 0.66). **Ces chiffres-là ne sont pas un
résultat non plus** : ils mélangent deux arbres. Règle : ne jamais apparier une cellule de
`/scratch/amounir/eegrow` (août) avec une de `eegrow_xds` — l'écart de protocole vaut ~2 pp,
soit le double des effets recherchés.

### Ce qui reste valide

Tout ce qui se calcule **entièrement dans le bloc `within` du 27/08** :

| contraste | Δ | IC 95 % | p | sujets |
|---|---|---|---|---|
| EA seul @ within/fixe | +1.89 pp | [+0.56, +3.26] | 0.0094 | 33/52 |
| croissance seule @ within/EA | +0.75 pp | [+0.17, +1.34] | 0.0165 | 32/52 |

### Test de falsification du mécanisme « amplitude » — **le mécanisme est réfuté**

Prédiction : `within` n'a qu'un dataset donc un seul amplificateur, l'interaction
croissance × alignement doit **disparaître**. Elle ne disparaît pas — elle change de base :

| bras | base | Δ | IC 95 % | p | holm | MDE |
|---|---|---|---|---|---|---|
| pooled (août) | none | +1.29 | [+0.39, +2.19] | 0.0074 | 0.037 | 1.32 |
| pooled (août) | scale | +0.17 | [−0.64, +1.00] | 0.68 | 0.89 | 1.21 |
| within (27/08) | none | +0.80 | [−0.07, +1.71] | 0.084 | 0.25 | 1.30 |
| within (27/08) | scale | **+1.06** | [+0.32, +1.81] | 0.0082 | **0.037** | 1.10 |

Le motif s'inverse : significatif contre `none` et nul contre `scale` sur le pool,
l'inverse sur `within`. La comparaison directe pooled − within vaut +0.49 pp (p = 0.44,
MDE 1.80) — non concluante, et de toute façon cross-arbre. **La prémisse même du mécanisme
(« nul contre `scale` ») est un candidat artefact de l'ancien protocole** : il faut la
re-mesurer sur `pooled` au nouveau protocole avant d'écrire quoi que ce soit là-dessus.

Scripts : `scratchpad/within_cho/falsif.py` et `replicat.py`.

### Conséquence sur le package B (504685, en cours)

Ses cellules `core+extrap` sont au **nouveau** protocole, sa baseline `core` du 12/08 à
l'**ancien**. Le contraste tel qu'apparié mesurerait ~2 pp de protocole pour ~1 pp d'effet
cherché. Le job n'est pas perdu : il devient interprétable dès qu'une baseline `core`
même-protocole existe.

### Re-mesure lancée : job **504709** (18 cellules `pooled`)

Soumis le 27/08 après go d'Adam. `benchmarks/slurm/xds_pooled_cho.sbatch`, array `0-17%4`,
`gpu:turing:1`, `--exclude=margpu021`, walltime 10 h, logs
`/scratch/amounir/logs/xds_pooled_cho_504709_*.log`.
`{bd,grow}_shallow` × `{euclidean, none, scale}` × 3 seeds, `--arm pooled --pool-tier core`.

**Ordre choisi : `euclidean` d'abord** (cellules 0-5) — ce sont les 6 qui débloquent la
baseline du package B et le bras de référence du gate.

**Coût réel ~63 GPU-h, pas 8-16.** L'estimation initiale était calquée sur le bloc `within`
(10 320 essais) ; `pooled` en entraîne 36 952. Mesuré depuis 504685 : 3.19 min/fold sur
43 472 essais → 2.71 min/fold sur `core` → **2.35 h/cellule bd**, et grow coûte 1.94× un bd
(57.6 s contre 29.6 s par fit) → **~4.6 h/cellule grow**.

**Pourquoi turing alors que pascal est libre.** turing est plein (23/23 GPU), pascal a
19 GPU au repos — pascal finirait en une nuit, turing prendra des jours. On reste sur
turing : les deux blocs auxquels ces cellules seront appariées (le bloc `within` du 27/08
et les `core+extrap` de 504685) tournent tous les deux sur `gpu:turing`. Passer sur pascal
réintroduirait le type de carte comme décalage systématique dans le chiffre principal,
soit exactement ce que 504278 avait pour but de retirer. Le coût est du temps d'attente,
pas un coût scientifique.

Moniteur `b6v38lbqb` (persistant) : échec de cellule, fin des 6 cellules `euclidean`, fin
de 504685, fin de 504709.

### 28/08 07:35 — état et **alerte walltime**

| cellules | contenu | état |
|---|---|---|
| 0-5 | `euclidean`, bd+grow, 3 seeds | **COMPLETED**, 2 h 00 – 3 h 43 |
| 6-8 | `none`, bd, 3 seeds | RUNNING, fold ~50/52, fin ~07:47-08:15 |
| 9 | `none`, grow, seed 0 | RUNNING, fold 15/52 après 4 h 36 |
| 10-17 | `none` grow ×2, `scale` bd ×3 + grow ×3 | PENDING (`JobArrayTaskLimit`) |

**Le coût par cellule dépend de l'alignement, ce que l'estimation n'avait pas prévu.**
Mesuré sur les timestamps de fold (régime établi, hors chargement du pool) :

| bras | min/fold | durée cellule |
|---|---|---|
| bd `euclidean` | 2.3 | 2 h 00 |
| grow `euclidean` | 3.8 – 4.3 | 3 h 19 – 3 h 43 |
| bd `none` | 7.45 | ~6 h 25 |
| grow `none` | **19.4** | **~16 h 50** |

Sans blanchiment le modèle ne cesse pas de progresser, l'early stopping ne déclenche
jamais et le budget complet est consommé — et la croissance ajoute de la capacité à
chaque époque. **Conséquence : les 6 cellules grow non-`euclidean` (9, 10, 11, 15, 16, 17)
dépassent le walltime de 10 h et seront tuées vers le fold 32.** `cross_dataset.py`
n'écrit son CSV qu'à la fin d'une cellule (aucun `*none*.csv` sur disque) : une cellule
tuée ne rend rien. `scontrol update TimeLimit` est refusé (Access/permission denied), la
partition `tau` autorise 7 jours.

Les 9 cellules bd et les 3 grow `euclidean` ne sont pas concernées.

### 28/08 07:45 — rattrapage : job **505009**

Après go d'Adam : `scancel 504709_{9,10,11,15,16,17}`, puis resoumission des 6 cellules
`grow_shallow` × `{none, scale}` × 3 seeds dans un array séparé,
`benchmarks/slurm/xds_pooled_cho_grow.sbatch`, `array=0-5%3`, **`--time=24:00:00`**,
`gpu:turing:1`, `--exclude=margpu021`, logs
`/scratch/amounir/logs/xds_pool_cho_grow_%A_%a.log`.
Mapping : `i/3` → `{none, scale}`, `i%3` → seed. Vérifié, 6 combinaisons uniques.

L'annulation a libéré un GPU immédiatement — 504709_12 (bd/scale seed 0) a démarré dans la
seconde, et 505009_0/1/2 (grow/none) tournent déjà. Fin attendue : les bd/scale ~6 h 25,
les grow ~16 h 50 par vague, 2 vagues à `%3` → **504709 complet ce soir, 505009 vers le
30/08**.

Périmètre final de la grille `pooled` cho2017 : 9 cellules bd (504709: 0-2, 6-8, 12-14) +
3 grow `euclidean` (504709: 3-5) + 6 grow non-`euclidean` (505009) = 18.

## Package B — contrôle négatif interpolation (27/08) : job **504685**, cible changée

Le package B **avait déjà tourné les 11-12/08** sur `bnci2014_001` (6 cellules
`core+extrap`, 6 `core+lowrank`, 6 `core+interp`, contre 6 `core`), toutes post-correctif
volts `004b8be` (11/08 15:58 ; les CSV invalidés s'arrêtent à 16:02, ceux-ci commencent à
20:03). Elles étaient sur disque sans avoir jamais été analysées.

Réanalyse à l'unité correcte (le sujet, seeds moyennées dedans, n = 9, Holm sur 6
contrastes, bootstrap B=20000) — **les six contrastes sont nuls** :

| modèle | contraste | Δ (pp) | IC 95 % | Holm | MDE | sujets |
|---|---|---|---|---|---|---|
| bd_shallow | extrap − core | −1.31 | [−2.71, −0.05] | 0.65 | 2.32 | 1/9 |
| bd_shallow | lowrank − core | −0.86 | [−2.70, +0.81] | 1.00 | 3.08 | 4/9 |
| bd_shallow | interp − core | −0.95 | [−2.58, +0.54] | 1.00 | 2.72 | 5/9 |
| grow_shallow | extrap − core | +0.21 | [−0.90, +1.43] | 1.00 | 2.04 | 4/9 |
| grow_shallow | lowrank − core | −0.32 | [−1.81, +1.13] | 1.00 | 2.56 | 5/9 |
| grow_shallow | interp − core | +0.12 | [−0.99, +1.26] | 1.00 | 1.92 | 3/9 |

**Ce n'est pas un résultat, c'est une absence de puissance.** `bnci2014_001` n'a que
9 sujets, l'écart-type inter-sujet du contraste vaut ~2.1 pp, donc le MDE plancher est de
~2.2 pp. Un contrôle négatif incapable de descendre sous 2.2 pp ne peut pas qualifier une
grille dont les effets visés valent ~1 pp. Pour un MDE de 1 pp : **n = 40 sujets**.

Piège de lecture identifié : `core+extrap` ajoute aussi +6520 essais (+17.6 %) et
+9 sujets, donc un nul peut vouloir dire « le bruit coûte ce que les données rapportent ».
Le contraste propre est `extrap − interp` (les deux ajoutent un dataset étranger) : mesuré
à −0.36 pp (bd) et +0.09 pp (grow), MDE 2.04 / 1.13 pp — non concluant lui aussi.

**Job 504685 relance donc le package B sur `cho2017` (52 sujets, MDE ≈ 0.86 pp).**
6 cellules, `pooled` / `euclidean` / `core+extrap`, ~16 GPU-h, PENDING (504342 occupe les
GPU). Baseline appariée = les cellules `cho2017__pooled__euclidean__core__seed{0,1,2}`
du 12/08 sous `/scratch/amounir/eegrow/` : **aucun commit ne touche `cross_dataset.py`,
`pool.py` ni `montage.py` entre `d1d16dd` (12/08) et `029dd9b`** (l'état de l'arbre
`eegrow_xds`), donc le code est identique et l'appariement par sujet tient. C'est pour ça
que le job part de `eegrow_xds` et non de `eegrow_interp`, dont l'arbre porte l'axe
field-interpolation non commité.

Analyse prête : `scratchpad/pkgB/analyse.py` et `analyse2.py`.

### Résultat (28/08) — la grille **est** sensible au contenu du pool

Appariement enfin propre : `core+extrap` = 504685, `core` = cellules 0-5 de 504709, **même
arbre `eegrow_xds`, même protocole post-`RestoreBestModel`, même carte `gpu:turing`**.
Intégrité vérifiée : 52 sujets × 3 seeds × 2 modèles × 2 tiers ; `core` = 36 949.7 essais /
247 sujets, `core+extrap` = 43 469.7 / 256. Script `scratchpad/pkgB_cho/analyse_cho.py`.

| modèle | `core` | `core+extrap` | Δ (pp) | IC 95 % | p (Wilcoxon) | Holm | MDE | sujets |
|---|---|---|---|---|---|---|---|---|
| bd_shallow | 68.88 | 69.67 | **+0.78** | [+0.35, +1.21] | 0.0006 | **0.0020** | 0.64 | 34/52 |
| grow_shallow | 69.43 | 69.64 | +0.21 | [−0.36, +0.76] | 0.38 | 0.48 | 0.82 | 28/52 |

**Ce que ça qualifie.** Le contrôle avait un seul but : montrer que la grille réagit au
contenu du pool. Elle réagit — sur le pire cas constructible (BNCI2014_004, 3 électrodes
enregistrées, 19 des 22 canaux inventés, le pire à 10.1 cm de toute mesure), à n = 52 avec
un MDE de 0.64 pp, l'effet ressort à p = 0.0006. Un nul d'interpolation mesuré sur cette
grille sera donc interprétable.

**Le signe est positif, et c'est le point intéressant.** Ajouter un dataset massivement
extrapolé **aide** le modèle fixe (+0.78 pp) au lieu de le dégrader. À `bnci2014_001` le
même contraste valait −1.31 pp (n = 9, non concluant) : la direction s'inverse une fois la
puissance suffisante. Lecture la plus simple : +17.6 % d'essais et +9 sujets rapportent
plus que le bruit d'interpolation ne coûte. Le contraste qui sépare les deux
(`extrap − interp`, à volume égal) **n'est pas encore mesuré sur cho2017** — c'est le
portage du tier `interp`, 6 cellules, en attente du go d'Adam.

La croissance, elle, ne bouge pas (+0.21, MDE 0.82) : elle absorbe déjà les données
supplémentaires autrement. À noter, pas à conclure — le nul est sous le MDE.

### 28/08 — pourquoi `core+interp` tel quel **ne** donnerait **pas** le contraste propre

Comptage des essais réellement en cache, par dataset ajouté à `core` (36 950 essais / 247 sujets) :

| tier ajouté | dataset | sujets | essais | Δ volume |
|---|---|---|---|---|
| `interp` | Shin2017A (rang plein, atteignable seulement par interpolation) | 29 | 1 740 | **+4.7 %** |
| `lowrank` | Zhou2016 (bien supporté, mais rang 14) | 4 | 1 199 | **+3.2 %** |
| `extrap` | BNCI2014_004 (19 des 22 canaux inventés) | 9 | 6 520 | **+17.6 %** |

Les trois tiers n'ajoutent donc **pas le même volume** : `extrap` en apporte 3.7× plus que
`interp`. Comme le +0.78 pp du package B est déjà explicable par le seul volume, le
contraste `extrap − interp` pris seul **resterait confondu** — il mesurerait surtout
« 6 520 essais valent plus que 1 740 ».

**Ce qui rend quand même les deux tiers manquants utiles — job 505185, lancé le 28/08.**
Deux lectures deviennent possibles, aucune ne l'était avec seulement `core` et
`core+extrap` :

1. **`interp` vs `lowrank` est quasi apparié en volume** (1 740 contre 1 199 essais,
   +4.7 % contre +3.2 %) et diffère surtout par le **rang** des données ajoutées : Shin2017A
   est de rang plein, Zhou2016 de rang 14 sur 22. C'est donc un contraste sur la qualité de
   l'interpolation, pas sur le volume.
2. Avec quatre points (0, +1 199, +1 740, +6 520 essais) on obtient une **courbe
   volume-réponse**. La qualité d'interpolation se lit alors comme le **résidu** : `extrap`
   tombe-t-il sous la tendance que ses 6 520 essais prédisent ? Avec deux points, cette
   question n'a pas de réponse.

**Ce que 505185 ne fait pas.** Il ne remplace pas le chiffre causal. Pour celui-là il reste
deux designs, à trancher :

1. **Tiers appariés en volume** : plafonner chaque dataset ajouté à ~1 200 essais
   (sous-échantillonnage déterministe par seed). Demande une option `--pool-cap` dans
   `pool.py`. Simple, mais l'appariement reste au niveau du volume, pas du contenu.
2. **Dégradation contrôlée** (le design causal, recommandé) : prendre **un** dataset `core`
   qui possède nativement les 22 canaux cibles, en garder 3 électrodes et interpoler les 19
   autres, puis comparer le pooling de la version vraie contre la version interpolée. Essais
   identiques, sujets identiques, seule l'interpolation change — **zéro confusion de volume
   par construction**. Demande une variante de build dans `pool.py`.

## Axe interpolation : field interpolation ajouté (27/08) — **pas encore lancé**

Le pool était construit avec les **splines sphériques**, c'est-à-dire la *baseline* du
papier de Mellot/Chevallier (arXiv:2403.15415), pas sa *méthode*. Corrigé sur la branche
`feat/cross-dataset-montage`, commit `8be3cb9` (local, **pas encore poussé sur la PR #5**) :

- `interpolate_to_montage(..., method="spline"|"field")`
- l'estimateur est une **clé de cache** (`<root>/<dataset>__field/`), vérifiée par sujet
  dans `pool.load()` : sinon un run FI relirait des arrays splines et le contraste
  reviendrait à zéro en ayant l'air propre
- les sujets qui ne reconstruisent rien (les ~250 du tier `core`) sont réutilisés tels
  quels — leur projection est une permutation, prouvé par test
- `--interp-method` sur `pool.py build` et `cross_dataset.py`
- 106 tests passent (98 avant)

**Job 504577 — TERMINÉ (rc=0), verdict : on garde les splines.** 20 sujets, vérité terrain
sur 13 électrodes. Au niveau sujet (n=20) :

| régime | Δ corr (FI − splines) | IC 95 % | p |
|---|---|---|---|
| interpolé (11 électrodes) | +0.0053 | [−0.0045, +0.0161] | 0.34 |
| extrapolé (C5, C6) | −0.1077 | [−0.128, −0.093] | 6e-10, 0/20 sujets |

C5/C6 sont les seules électrodes à `empty_sectors > 0`. Les splines sont indifférentes à
la découpe (0.8554 vs 0.8595), donc c'est une faiblesse de FI seul, pas des électrodes.
**Conséquence : pas de rebuild du pool, l'axe interpolation n'est plus bloquant.**

Deux corrections apportées au script après coup (commit `c3857dc`) :
- le `rho` gain-vs-écart était calculé sur 260 paires pour 13 valeurs d'écart distinctes
  (n gonflé ×20, signe instable). Unité = l'électrode, MDE affiché : la claim du papier
  est **non testée** ici (écarts 3.29–4.44 cm seulement), pas réfutée.
- le contraste `field − spline` **en décodage est structurellement nul** avec un modèle ML :
  CSP et le tangent space sont invariants par changement de base inversible des canaux, ce
  qui est exactement ce qui sépare deux interpolations linéaires des mêmes sources.
  Vérifié hors saturation (0.9667 / 0.9167, écart 0.0000). Le script le refuse désormais.
  Le contraste `reconstruit − enregistré` reste valide : **+0.0052, p = 0.70** — la
  reconstruction ne coûte rien en décodage, ce qui rend l'ablation `core` vs `core+interp`
  interprétable.

## Incident GPU fantôme margpu021 — 504278 partiel, retry **504342** (27/08)

Sur les 18 cellules de 504278, **8 tournent normalement** (cellules 0-4, 6-8, sur margpu020
et margpu022) et **10 ont échoué en 9-14 s** : cellules 5 et 9→17, toutes sur **margpu021**.

Cause : le nœud annonce `Gres=gpu:turing:3` mais un des trois slots n'expose aucun device.
Log : `nvidia-smi → "No devices were found"`, puis
`RuntimeError: CUDA_VISIBLE_DEVICES='0' but torch.cuda.is_available() is False`.
Le garde-fou de `benchmarks/utils.py:pick_device` a refusé le repli CPU — **aucune cellule
n'a été entraînée sur CPU en silence**, donc rien n'est contaminé. Mais l'échec en 9 s
libère le slot instantanément, et Slurm y a fait passer toute la file d'attente : d'où la
cascade qui a tué `grow_shallow` en entier (9-17) alors que `bd_shallow` est presque intact.

Retry : **504342**, `--array=5,9-17`, `--exclude=margpu021`, même `gpu:turing` (homogénéité
de carte avec les 8 survivantes). Script `benchmarks/slurm/xds_within_retry.sbatch`, logs
`/scratch/amounir/logs/xds_within_retry_504342_*.log`. Ajout d'un préflight `torch.cuda`
avant tout accès aux données, suivi d'un `sleep 120` en cas d'échec : si un autre nœud a la
même panne, il ne peut plus consommer qu'une ou deux cellules au lieu de l'array entier.

## Décision d'analyse (Adam, 27/08)

La comparaison de référence reste **`grow_X` vs `bd_X`** (modèles braindecode), pas
`grow_X` vs `fix_X`. Les contrôles `fix_*` de 500952 restent sur disque mais ne sont pas
le contraste principal des figures.

## Bloc `within` cross-dataset Cho2017 — SLURM **504278** — **EN COURS** (27/08)

Soumis le 27/08/2026 ~15h40. Array `0-17`, partition `tau`, `gpu:turing:1`, 3 h/cellule.
Checkout **`/scratch/amounir/eegrow_xds`** (branche `feat/cross-dataset-montage`, PR #5),
cache pool `/scratch/amounir/pool_cache` (290 sujets, manifeste validé).
Script : `benchmarks/slurm/xds_within.sbatch`. Logs : `/scratch/amounir/logs/xds_within_504278_*.log`.
Sorties : `/scratch/amounir/eegrow_xds/benchmarks/results_cross_dataset/cho2017/`.

18 cellules = `{bd_shallow, grow_shallow}` × `{none, scale, euclidean}` × seeds `{0,1,2}`,
`--target cho2017 --arm within --pool-tier core`. Coût attendu ~8 GPU-h (mesuré 0.44 h/cellule
sur la grille d'août).

### Pourquoi 18 et pas les 12 manquantes

`within/euclidean` existe déjà (grille d'août, `/scratch/amounir/eegrow/benchmarks/results_cross_dataset/`)
mais sur une carte non enregistrée. Or c'est le bras de référence du gate. Le re-lancer ici
coûte 2.6 GPU-h et retire le type de carte comme décalage systématique possible dans le
chiffre principal. Les anciennes cellules restent sur disque comme réplicat indépendant.
Même raison pour `gpu:turing` explicite plutôt que « ce qui est libre » : laisser Slurm mixer
pascal et ampere réintroduirait exactement ce que le re-run enlève.

### Ce que ça débloque

1. La baseline nue (`within/none`), donc un gate « tout allumé contre rien » sans note de bas de page.
2. La décomposition en effets principaux additifs (pooling / alignement / croissance).
3. **Le test de falsification de l'hypothèse d'amplitude.** L'interaction croissance × alignement
   mesurée sur `pooled` (+1.29 pp contre `none`, holm 0.037) est **nulle contre `scale`**
   (+0.17 pp, p=0.68) : elle porte sur la normalisation d'amplitude, pas sur le blanchiment.
   Mécanisme proposé : la line search de `grow_step` compare des magnitudes de gradient de part
   et d'autre d'une jonction, donc dans un pool où un dataset domine en amplitude c'est le gain
   de l'amplificateur qui décide *où* la capacité est allouée. Prédiction : l'interaction doit
   **disparaître** sur `within` (un seul dataset, un seul amplificateur). Si elle survit, le
   mécanisme est faux.

### Résultats déjà en main sur le pool (grille d'août, n=52, Holm sur 8 contrastes)

| contraste | Δ | IC95 bootstrap | p_holm |
|---|---|---|---|
| gate tout-allumé (grow+pooled+EA − fixe+within+EA) | +2.19 pp | [+1.29, +3.13] | 0.0002 |
| pooling seul, fixe | +1.79 pp | [+0.96, +2.65] | 0.0010 |
| pooling seul, growing | +1.74 pp | [+0.90, +2.64] | 0.0020 |
| blanchiment net du rescaling, growing | +1.72 pp | [+0.49, +3.09] | 0.049 |
| blanchiment net du rescaling, fixe | +1.55 pp | [+0.33, +2.91] | 0.073 |
| interaction croissance × alignement | +1.29 pp | [+0.40, +2.18] | 0.037 |
| croissance seule @ pooled/EA | +0.40 pp | [−0.13, +0.92] | 0.29 |

Le +1.72 pp de blanchiment net est à comparer au **+1.51 pp** du gate EA mono-dataset :
deux protocoles, deux estimateurs indépendants, même amplitude.

**Attention à la lecture du gate** : le bras de référence est `within/euclidean`, donc la
baseline a déjà l'EA. Le +2.19 pp est « pooling + croissance par-dessus l'EA », pas « tout
contre rien ». C'est ce que 504278 corrige.

### Suites conditionnées (NON lancées)

- **B — contrôle négatif d'interpolation**, `core+extrap` vs `core`, 6 cellules ≈ 21 GPU-h.
  Une grille incapable de distinguer `core+extrap` de `core` est insensible au contenu du pool,
  et alors aucun résultat d'interpolation n'est interprétable, y compris positif.
- **C — le vrai contraste d'interpolation**, `core+lowrank`, 12 cellules ≈ 43 GPU-h.
  À ne lancer que si B passe.

**Bloqueur sur B et C** : notre interpolation est en splines sphériques (Perrin 1989), or
Mellot, Collas, Chevallier, Engemann, Gramfort (arXiv:2403.15415) montrent que le *field
interpolation* (modèle direct de Maxwell, inversion par minimum-norm, ré-application aux
positions cibles) bat les splines sur **4 datasets sur 6**, avec un écart qui croît quand le
recouvrement de canaux diminue — soit exactement le régime `lowrank`/`extrap`. MNE expose les
deux par `interpolate_bads(method=dict(eeg="MNE"|"spline"))`. L'axe méthode doit exister avant
que B et C vaillent la peine.

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

## 06/09 ~17h — sonde motif soumise (543826) + jumeau `shipped` LU

**543826 — `motif_probe`, array 0-23, LES 24 TÂCHES DÉMARRÉES D'EMBLÉE, 0 en attente.**
La moitié manquante de la campagne prévalence : les 8 matrices D×R complètes n'étaient
pas lisibles sans un `restored_probe`/`acc_probe` par sujet à k>=15 (porte G1). Les
`restored_epoch` des CSV D×R ne s'y substituent pas, ils sont à k=3.

Dimensionnement : 27,8 GPU·h, cellule la plus longue = brandl2020 à 2,4 h, donc
**wall clock ~2,5 h** et non une nuit. Découpage en 24 tâches `--gres=gpu:1` (une
cellule = une graine d'un dataset = un processus sur une carte) plutôt qu'en 8
allocations de 3 cartes : 31 cartes utilisables étaient libres mais seulement 5 nœuds
avaient 3 cartes libres, donc l'allocation par nœud aurait fait la queue. Réparties sur
11 nœuds : margpu006/007/009/010/011/012/017/018/019/028. Claims vérifiés : 24 pris,
24 distincts, 3 par dataset.

Ce qui autorise le découpage : `gap` est une DIFFÉRENCE entre deux règles calculées sur
LES MÊMES fits, donc un effet de classe de carte s'annule dans la soustraction. C'est ce
qui rend l'homogénéité matérielle de `fresh_probe.sbatch` non portante ici.

**Périmètre arrêté à 8 datasets, et le motif est arithmétique.** `stieger2021`,
`yang2025` (en cours) et `zhou2020` (0 CSV) sont hors lecture. Les attendre porterait n
de 13 à 15 et FERAIT BAISSER la puissance du test de signe (0.58 -> 0.46 : le seuil de
rejet saute de 10/13 à 12/15). Addendum daté écrit dans `PREREG_motif_prevalence.md`
AVANT la soumission, règle d'arrêt = « ce qui est complet au 6 septembre », indépendante
de tout résultat.

**Jumeau `shipped` LU — il dormait terminé depuis le 04/09.**
`benchmarks/analysis/shipped_twin_readout.py` + `shipped_flips.py`, sorties dans
`benchmarks/analysis/shipped_twin/`. 542 cellules appariées sur 558 (restriction à
`align=none` : `results_final` porte aussi `euclidean`, pas `results_shipped`).

- **16 changements de vainqueur sur 30 (eval × dataset)**, dont 8/12 en within_session.
- tau de Kendall médian +0.75 ; 85 inversions de paires sur 710.
- Qualification (une inversion entre deux bras indiscernables est du bruit) :
  **42 cosmétiques + 34 matérielles + 9 DURES**. Les 9 dures sont les paires où les
  DEUX protocoles établissent un ordre et s'affirment le contraire.
- Le delta apparié (unité = le sujet) est **hétérogène entre bras**, et c'est le
  mécanisme : bd_deep4 +0.0594 en cross_session contre bd_sccnet +0.0100. Le protocole
  ne décale pas tous les bras également, il pénalise ceux qui convergent lentement.
- 6 des 9 inversions dures impliquent bd_deep4 vs fix_deepeeg/grow_deep — corroboré
  indépendamment par [[deep4-instability]] (bd_deep4 scoré à l'époque 4).

**Réserve à porter dans le papier** : les inversions dures sont sur n=9 à n=14 sujets,
où le bootstrap percentile sous-couvre. Les 6 liées à deep4 ont un mécanisme connu, les
3 autres (grow_sccnet sur BNCI2014-002 / BNCI2015-001) n'en ont pas encore.
