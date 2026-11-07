# PREREG — Adapter un modèle partagé en lui ajoutant des neurones : 5 modèles × 11 datasets (lecture 3)

07/11/2026. **GELÉ** dans le commit qui contient aussi `adapt_B.py`, ses deux sbatch et son
test, avant toute adaptation hors du dataset de réglage. Aucun nouvel entraînement de
modèle de base : les modèles partagés sont les fits cross-subject `<modèle>__bd` de la
campagne A, déjà sauvés.

**Rapport à la lecture 2.** La lecture 2 (`PREREG_growth_adaptation_A.md`, commit
2202484 : Shallow, lee2019_mi) tourne encore au moment du gel. Son rapport s'écrira tout
seul à la fin de la chaîne, mais **il ne sera pas ouvert avant ce commit** : le choix des
modèles, des datasets et des critères ci-dessous ne doit rien à son résultat. Seuls ont
été lus de la lecture 2 : son réglage du taux d'apprentissage sur bnci2014_001
(`tune.json`), qui ne contient aucune donnée de lee2019_mi.

## 1. Question

La même que la lecture 2, étendue à tout ce que la campagne A a entraîné : pour adapter
à un nouveau sujet un modèle entraîné sur les autres sujets, avec 10 essais par classe,
ajouter des neurones fait-il au moins aussi bien que LoRA à budget égal, et mieux que
réentraîner la tête ou recalibrer la BatchNorm (AdaBN) ?

La lecture 2 répond pour un modèle et un dataset. La lecture 3 demande si la réponse
tient sur 5 architectures (2 convolutives, 3 transformers) et 11 datasets.

## 2. Données

- **Modèles partagés** : `cross_subject`, graine 0, poids finaux (`__fits__final`) des 5
  bras bd de la campagne A, construits comme la campagne
  (`braindecode.models.<Arch>(n_chans, n_outputs, n_times)`, `module_kwargs: {}`) :
  ShallowFBCSPNet, EEGNeX, EEGConformer, CTNet, MSVTNet. Le fit du sujet i n'a jamais vu i
  (vérifié par le script).
- **Datasets** : les 12 de la campagne A moins bnci2014_001, réservé au réglage : alexmi,
  bnci2014_002, bnci2014_004, bnci2015_001, cho2017, lee2019_mi, physionetmi,
  schirrmeister2017, shin2017a, weibo2014, zhou2016. Données construites par
  `check_cache.build_campaign`, lues dans le cache que lisent les cellules (`cache_of`), en
  lecture seule ; un cache manquant refuse. Labels : `LabelEncoder` sur les chaînes, comme
  MOABB.
- **Découpage calibration / test** (règle fixée avant lecture, par le contenu des
  métadonnées) :
  1. ≥ 2 sessions (bnci2014_004, bnci2015_001, lee2019_mi, shin2017a, zhou2016) : première
     contre dernière session (ordre des noms MOABB), dans les deux sens, 3 tirages par sens ;
  2. une session, runs nommés `train` / `test` (bnci2014_002, schirrmeister2017) :
     calibration dans les runs `train`, test sur les runs `test`, 6 tirages ;
  3. une session, plusieurs runs (physionetmi : 3 runs) : calibration dans tous les runs
     sauf le dernier, test sur le dernier, 6 tirages ;
  4. une session, un run (alexmi, cho2017, weibo2014) : **dans chaque classe**,
     calibration dans la première moitié de ses essais (ordre rendu par
     `paradigm.get_data`), test sur la seconde moitié, 6 tirages. Par classe et non sur
     l'ensemble : mesuré avant le gel, cho2017 rend tous les essais d'une classe puis tous
     ceux de l'autre, si bien qu'une moitié globale ne contient qu'une classe (aucun k
     possible) ; alexmi et weibo2014 rendent des classes entremêlées.
  Dans tous les cas, 6 adaptations par (sujet, k, bras). Tirages sans remise, k par classe,
  `default_rng(1000·i + 10·d + r)` (comme la lecture 2).
- **k** ∈ {5, 10, 25, 50} ; un k n'est calculé pour un sujet que si chaque partie de
  calibration a au moins k essais par classe (sinon il est compté manquant). Un sujet dont
  une partie de test manque d'une classe est refusé et compté. **k = 10 est le critère
  principal.**
- **Score** : celui de la campagne (`campaignA_check.score_of`) : ROC-AUC à 2 classes,
  accuracy sinon. **Score d'un sujet** = moyenne de ses 6 adaptations.

## 3. Garde-fou G0 : on reconstruit bien le modèle de la campagne

Par sujet et par session, les logits du modèle reconstruit sont comparés à ceux que la
campagne a sauvés (`__fits__logits`). G0 passe si : mêmes essais et mêmes labels ;
max |Δ logit| ≤ 1e-2 × max(1, max |logit sauvé|) ; argmax identique sauf sur les quasi
ex æquo (écart top-2 du logit sauvé ≤ 2 max |Δ|) ; |Δ score| ≤ 5e-3. Un sujet qui échoue est
exclu et compté ; une cellule (modèle, dataset) où plus de 2 sujets échouent est refusée
entière.

Pourquoi pas la tolérance de la lecture 2 (1e-3, argmax partout) : mesuré avant le gel sur
une RTX 2080, G0 à 1e-3 échoue sur CTNet lee2019_mi s3 (4,8e-3) et Shallow
schirrmeister2017 s1 (5,9e-3) et passe à 0 sur Shallow lee2019_mi s3. Les logs de la
campagne disent pourquoi : les deux fits qui échouent ont été entraînés sur une A40
(ampere, où cuDNN calcule les convolutions en TF32 par défaut), celui qui passe sur une
1080 Ti (pas de TF32). L'écart vient de la précision de la carte de la campagne, pas d'un
autre modèle. Le test synthétique vérifie que la nouvelle règle refuse un modèle d'une
autre graine. La lecture 3 désactive TF32 partout, pour que toutes les cartes calculent en
float32 complet. Avec la règle ci-dessus, G0 passe sur les 5 modèles × {schirrmeister2017
s1, physionetmi s1} mesurés avant le gel (max |Δ| ≤ 5,9e-3 pour des logits d'échelle 28,
aucun argmax différent, Δ score = 0) ; sur lee2019_mi s3, seul max |Δ| a été mesuré
(≤ 4,8e-3).

## 4. Bras d'adaptation et jonctions

Même procédure que la lecture 2 (AdaBN avant, seuls les paramètres du bras entraînés,
BatchNorm en eval et dropout en train pendant l'entraînement, AdaBN après), avec trois
changements, tous sans effet sur ce qui est comparé :
- AdaBN recalcule **toutes** les BatchNorm du modèle (Shallow n'en a qu'une), plus celle des
  neurones ajoutés d'EEGNeX, en une passe sur tout le lot de calibration. Si la carte
  manque de mémoire pour cette passe, elle est refaite à l'identique sur CPU ;
- le gradient du lot complet est accumulé par morceaux de 50 essais. Comme la BatchNorm
  est en eval, c'est le même gradient ;
- calcul sur GPU, TF32 désactivé.

**Où grandir.** Les neurones sont ajoutés **là où la campagne A faisait grandir chaque
modèle**, et LoRA porte sur la couche dans laquelle ces neurones écrivent :

| modèle | jonction (campagne A) | croissance : 1 neurone = | LoRA rang r sur |
|---|---|---|---|
| Shallow | conv_time → conv_spat | un filtre temporel + ses poids spatiaux (lecture 2) | conv_spat, 40 × (40·C) |
| EEGNeX | block_1 conv (1→8, (1,64)) → BN → block_2 conv (8→32, (1,64)) | un filtre temporel, sa BN, ses 32 noyaux (1,64) | block_2 conv, 32 × (8·64) |
| Conformer | les 6 FFN : Linear(40,160) → GELU → Dropout → Linear(160,40) | une unité cachée par FFN | chaque Linear(160,40) |
| CTNet | les 6 FFN : idem | idem | idem |
| MSVTNet | les 2 FFN : linear1(72,72) → ReLU → Dropout → linear2(72,72) | une unité cachée par FFN | chaque linear2 |

Init : comme la lecture 2. Couche entrante des nouveaux neurones à l'init PyTorch par
défaut de la couche correspondante, couche sortante à 0 ; LoRA avec A Kaiming-uniforme
(borne 1/√fan_in) et B = 0. Les deux adaptateurs partent donc exactement du modèle partagé
(vérifié par le test, en eval sous `no_grad`, ce qui exclut le chemin rapide des
transformers PyTorch qui sauterait les hooks).

**Budget.** Le budget p ∈ {2, 8} fixe le rang de LoRA ; la croissance ajoute, par jonction,
le nombre de neurones dont le compte de paramètres est le plus proche de celui de LoRA
(au moins 1). Comptes à la forme de lee2019_mi (62 canaux, 1 000 échantillons) :

| modèle | tête | lora2 | grow2 (neurones/jonction) | lora8 | grow8 | complet |
|---|---|---|---|---|---|---|
| Shallow | 4 882 | 5 040 | 5 012 (2) | 20 160 | 20 048 (8) | 105 202 |
| EEGNeX | 498 | 1 088 | 2 114 (1) | 4 352 | 4 228 (2) | 58 514 |
| Conformer | 66 | 2 400 | 2 430 (5) | 9 600 | 9 720 (20) | 853 506 |
| CTNet | 1 202 | 2 400 | 2 430 (5) | 9 600 | 9 720 (20) | 153 082 |
| MSVTNet | 146 | 576 | 580 (2) | 2 304 | 2 320 (8) | 75 772 |

À p = 2, croissance et LoRA sont à 2 % près sauf pour EEGNeX (×1,9 : un seul neurone y
coûte déjà deux rangs). La tête (`final_layer` de braindecode) n'a le même budget que pour
Shallow : E2 compare donc deux bras de tailles différentes, et le rapport donne les
comptes.

**Entraînement** : AdamW, décroissance 0, lot complet, 100 pas, entropie croisée, graine
du tirage. **Taux d'apprentissage** : un par (modèle, bras), dans {1e-4, 3e-4, 1e-3,
3e-3, 1e-2, **3e-2**}, choisi sur bnci2014_001 (9 sujets, même protocole, k = 10) par le
score moyen. La grille gagne 3e-2 par rapport à la lecture 2 parce que, sur
bnci2014_001, trois bras de Shallow y choisissaient le bord haut (1e-2) avec une courbe
encore montante (constat fait sur le réglage seul). Le choix est écrit dans `tune.json`
puis lu tel quel ; aucun dataset de test ne sert à choisir.

## 5. Garde-fou G1, par modèle

Fine-tuning complet − zéro-shot à k = 50, groupé sur les datasets où k = 50 existe (même
statistique qu'au § 6) : borne basse > 0 exigée. Un modèle qui échoue est retiré du
critère principal et ses critères secondaires n'ont pas de verdict.

## 6. Critères

Différences appariées par sujet. Le contraste groupé est la moyenne des moyennes par
dataset, pondérées par √n_d (n_d = sujets du dataset ; convention de la méta-analyse de
MOABB, pour que physionetmi et ses 109 sujets ne fassent pas tout). IC95 bootstrap
stratifié : 10 000 tirages des sujets **à l'intérieur de chaque dataset**,
`default_rng(0)`. p unilatéral de supériorité = fraction des tirages ≤ 0.

**Principal** (k = 10, p = 2, modèles qui passent G1, Holm sur 3, marge NI −0,02 comme
la campagne A). Pour chaque sujet, la différence est moyennée sur les modèles retenus (un
sujet doit les avoir tous) :

| critère | contraste |
|---|---|
| **E1** | croissance − LoRA |
| **E2** | croissance − tête |
| **E3** | croissance − AdaBN |

Verdicts : SUPÉRIEUR si Holm rejette ; sinon NON INFÉRIEUR si borne basse > −0,02 ; sinon
INFÉRIEUR si borne haute < 0 ; sinon NON CONCLUANT.

**Secondaire** : E1 à E3 pour chaque modèle séparément (groupés sur les datasets), Holm
sur les 15, mêmes verdicts.

Lecture :
- E1 SUP ou NI et E2 SUP au principal → la croissance est un adaptateur crédible face à
  LoRA, au-delà de Shallow ; le secondaire dit sur quelles architectures ;
- E1 INF → LoRA fait mieux à budget égal ;
- E3 non SUP → à 20 essais, apprendre n'apporte rien au-delà d'AdaBN ; regarder k = 25 et
  50 (rapportés).

## 7. Rapporté, sans valeur décisionnelle

Scores moyens par modèle × dataset × bras à k = 10 ; contrastes groupés à chaque k (p = 2
et 8) ; comptes de paramètres par modèle et dataset ; taux choisis et courbes de réglage ;
sujets exclus et k manquants ; Shallow sur lee2019_mi rapproché de la lecture 2 (mêmes
tirages, seuls la grille de lr, le GPU et l'accumulation du gradient changent).

## 8. Ce qui ne sera pas fait

- Aucun critère, k, p, bras, jonction, règle de découpage ou taux ne sera changé après
  lecture.
- Pas de LoRA sur l'attention, pas de règle TINY, pas de tête + croissance.
- Pas d'autre graine du modèle partagé.

## 9. Limites connues avant lecture

- LoRA est mise sur la jonction que l'on fait grandir, pas sur l'attention où on la met
  d'habitude dans un transformer. C'est le contrôle juste pour la question « nouveaux
  neurones contre re-mélange des mêmes », pas la meilleure LoRA possible.
- EEGNeX : budgets inégaux à p = 2 (cf. § 4). La tête n'est à budget égal que pour Shallow.
- Un seul modèle partagé par sujet (graine 0).
- Découpage 4 (une moitié contre l'autre d'un même run) : calibration et test proches dans
  le temps, ce qui favorise tous les bras de la même façon.
- shin2017a n'a que 10 essais par classe et par session : à k = 10, les 3 tirages sont
  identiques et k = 25, 50 n'existent pas. physionetmi n'atteint pas toujours k = 10.
- Réglage sur un seul dataset (22 électrodes, 4 classes) pour tous les autres.
- Sujet et session confondus avec le montage du casque (lecture 1, § 9).

## 10. Coût et exécution (mesurés avant le gel)

Sonde sur RTX 2080 Ti, un sujet de chacun des 11 datasets, une adaptation par bras à
k = 10 et au k maximal, extrapolée aux 6 tirages et à tous les k : environ 140 heures-GPU
pour les 315 sujets × 5 modèles (EEGNeX 72 h, dont schirrmeister2017 à ~110 min par sujet ;
MSVTNet 23 h ; Conformer 19 h ; CTNet 18 h ; Shallow 8 h), plus ~5 h de réglage. G0 a passé
sur les 11 datasets × 5 modèles de la sonde. Cartes : turing, ampere ou hopper, ≥ 10 Gio,
sous `bench`, TF32 coupé ; margpu020 (carte morte) et margpu028 (8 Go, OOM mesuré)
exclus. Au plus 10 tâches GPU à la fois (throttle du tableau SLURM). Chaîne :
réglage (45 tâches) → choix du lr (CPU) → adaptation (lignes de `run_tasks.txt`) → lecture
(CPU), en `afterok`. Une tâche reprise après préemption saute les sujets déjà écrits.
