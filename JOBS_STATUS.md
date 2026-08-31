# Jobs en cours

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
