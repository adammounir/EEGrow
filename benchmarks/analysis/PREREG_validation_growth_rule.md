# PREREG — Validation de la règle de croissance sur les 6 modèles cibles

21/10/2026. Gelé avant le lancement de la campagne (commit qui contient ce fichier).
Lecture : `benchmarks/valid_read.py`, figé dans le même commit, avant tout résultat.

## 1. Question

Adam, 21/10 : avant de passer à l'étape suivante, la croissance doit être validée sur
**tous** les modèles qu'on veut utiliser. Ce sont 3 CNN (ShallowFBCSPNet, EEGNeX j2,
TIDNet) et 3 conv + transformer (MSVTNet, CTNet, EEGConformer, tous les FFN en
croissance). La règle doit être une **base commune** aux CNN et aux transformers.

« Sûr à 100 % » n'est pas atteignable statistiquement. Ce que la campagne établit,
par modèle et par règle : (S) l'entraînement est stable, (G) la croissance a lieu, et
(NI) le modèle qui grandit n'est pas inférieur, à une marge déclarée près, au même
modèle braindecode entraîné à taille fixe. Elle l'établit sur 3 datasets, dont 2 qui
n'ont jamais servi à choisir la règle.

## 2. Ce qui est déjà su (et pourquoi trois règles)

- **Le papier TINY** (Verbockhaven et al., TMLR 2024, arXiv 2405.19816, lu dans le PDF) :
  - L'entraînement (Algorithm 1, Fig. 5-6, Table 1) ajoute les neurones avec une
    **amplitude fixe** : norme quadratique moyenne de 1e-3 par neurone, des deux côtés
    (annexe E.3.1, éq. 197-198). Il n'y a ni recherche de s, ni mise à jour des poids
    existants.
  - La recherche de γ de l'annexe E.3.2 sert seulement à la Fig. 8, qui est une
    expérience **sans** descente de gradient.
  - Les transformers sont annoncés comme « future works ».
- **Campagne d'attribution du 20/10** (Shallow, bnci2014_001) :
  - La **dose** de s cause la perte.
  - match + s fixé à 0.25 donne +0.021 au-dessus de la variante A.
  - La variante A (grille 0→4 sans normalisation) vaut 0.585, contre 0.581 pour
    bd_shallow.
- **Étape 1 du 18/10** (MSVTNet, bnci) : la variante A est stable et vaut −0.007
  [−0.014, +0.001] contre bd_msvtnet.

Les règles, toutes déclarées explicitement sur chaque ligne de cellule :

| règle | normalisation | s appliqué | recherche de s | origine |
|---|---|---|---|---|
| **tiny** (principale) | `tiny_paper` (E.3.1), mise à jour des poids existants annulée | 1 (l'amplitude est celle du papier) | enregistrée, non appliquée | le papier TINY |
| m025 | `match_extending_layer` (gromo) | 0.25 | enregistrée, non appliquée | meilleur bras du 20/10 |
| varA | aucune | choisi par recherche sur {0, .25, .5, 1, 2, 4} (held-out 10 %) | appliquée | étape 1, variante A |

La règle **tiny** est la règle principale : c'est la base commune demandée, fidèle au
papier. Les deux autres servent de comparaison et de repli.

## 3. Plan

- **Modèles** (config `benchmarks/config/model/`), chacun contre son jumeau braindecode
  `bd_*`, qui a la même architecture à la largeur publiée :

| modèle | bras en croissance | jonctions | départ → cible |
|---|---|---|---|
| ShallowFBCSPNet | `grow_shallow` | conv_time → conv_spat | 8 → 40 |
| EEGNeX | `grow_eegnex_j2` | jonction 2 (block_4 → BN → block_5), en place | 4 → 32 |
| TIDNet | `grow_tidnet` | goulot du filtre spatial dense 0 | 8 → 72 |
| MSVTNet | `grow_msvtnet_all` | les 2 FFN | 8 → 72 chacun |
| CTNet | `grow_ctnet_all` | les 6 FFN | 16 → 160 chacun |
| EEGConformer | `grow_conformer_all` | les 6 FFN | 16 → 160 chacun |

  Sur les transformers, **une** jonction grandit par pas : celle que choisit
  `select_best_update` de gromo.
- **Recettes** : l'optimiseur de chaque dépôt officiel (CATALOGUE § 5), à savoir le
  type, lr, betas, eps, wd, amsgrad et batch. Elles sont écrites dans
  `valid_make_cells.py`.
  - **Écarts déclarés**, faits pour que tous les bras soient comparables : ni S&R
    (Conformer, CTNet, MSVTNet), ni schedule (cosinus de TIDNet, plateau d'EEGNeX), ni
    label smoothing (TIDNet), ni perte jointe (MSVTNet).
  - CTNet garde la largeur de braindecode (emb 40, 4 têtes) et non celle du papier
    pour 2a.
- **Protocole commun** (étape 0) :
  - within_session, 200 époques sans arrêt anticipé, score = dernière époque ;
  - held-out de croissance de 10 %, croissance toutes les 5 époques ;
  - RTX 2080 Ti seulement (pool margpu017/018/019/022) ;
  - `EEGROW_DEBUG_GROWTH=1`.
- **Datasets et graines** :
  - bnci2014_001 : 9 sujets, 4 classes, accuracy, graines 0-2 ;
  - cho2017 : 52 sujets, 2 classes, ROC-AUC, graine 0 ;
  - lee2019_mi : 54 sujets, 2 classes, ROC-AUC, graine 0.

  cho2017 et lee2019 sont lus dans le cache MOABB prétraité
  (`/scratch/amounir/moabb_cache`), car leurs fichiers bruts ont disparu de MNE_DATA.
- **120 cellules** : 6 modèles × 4 bras × (3 + 1 + 1) dataset-graines
  (`benchmarks/valid_cells.txt`, `valid_battery.sbatch`).

## 4. Critères (figés)

Unité = le sujet. Les scores sont moyennés sur les sessions et les graines, les
contrastes sont appariés par sujet, et l'IC à 95 % est un bootstrap percentile sur les
sujets (10 000 tirages, `default_rng(0)`).

### 4.1 Par (modèle, règle, dataset)
- **S, stabilité** : 0 fit avec une `train_loss` ou une `valid_loss` non finie à une
  époque quelconque, et `grad_norm_max_fit` ≤ 1e6 sur tous les fits (seuil de pic de
  l'étape 0).
- **G, croissance** : au moins 90 % des fits finissent plus larges qu'au départ. La
  largeur finale médiane, rapportée à la cible, est rapportée sans être un critère.
- **NI, non-infériorité**, sur Δ = moyenne par sujet de (grow − bd) :
  - PASSE si la borne basse de l'IC est > −0.02 (accuracy sur bnci, AUC sur cho et lee) ;
  - INFÉRIEUR si la borne haute de l'IC est < 0 ;
  - sinon NON CONCLUANT. Ce n'est ni un échec ni une validation.
- **C, complétude** : les mêmes sujets que le jumeau bd, avec les graines prévues.

### 4.2 Lisibilité d'un (modèle, dataset)
Le jumeau bd doit apprendre : la borne basse de l'IC de (bd − hasard) doit être > 0,
avec un hasard de 0.5 en AUC à 2 classes et de 0.25 en accuracy à 4 classes. Sinon,
la non-infériorité est triviale (deux modèles au hasard sont « non inférieurs »), et le
modèle est NON LISIBLE sur ce dataset.

### 4.3 Verdict d'une règle sur un modèle
- **VALIDÉE** si S, G, NI et C passent sur les 3 datasets. C'est un test
  d'intersection-union : chaque test est au seuil de 5 % et la conjonction ne demande
  pas de correction.
- **ÉCHEC** si un S, G ou C échoue, ou si un NI est INFÉRIEUR.
- **NON CONCLUANT** sinon.
- **NON LISIBLE** si un des 3 datasets n'est pas lisible.

### 4.4 Décision (sans relecture des résultats pour la choisir)
1. Si **tiny** est VALIDÉE sur les 6 modèles, elle devient la base commune et on
   passe à l'étape suivante.
2. Sinon, si m025 est VALIDÉE sur les 6, puis varA, dans cet ordre : cette règle
   devient la base, et l'échec de tiny est rapporté.
3. Sinon, aucune règle n'est adoptée automatiquement. On rapporte par modèle ce qui
   échoue (S, G, NI ou lisibilité). La suite se décide avec Adam : retirer un modèle,
   revoir sa recette, ou ajouter de la puissance pour un NON CONCLUANT.

Les comparaisons règle contre règle sont **exploratoires** et ne servent pas à
choisir.

## 5. Puissance

- Sur bnci2014_001 à n = 9, la demi-largeur de l'IC de (P_n4 − REF) valait environ
  0.012-0.015 le 20/10 (Shallow).
- Un modèle plus bruité aura un IC plus large. C'est exactement le cas que le statut
  NON CONCLUANT sépare de l'échec ([[underpowered-not-null]]).
- cho2017 et lee2019_mi (n ≈ 52-54) donnent la puissance que bnci n'a pas.

## 6. Ce qui ne sera pas fait
- Ne relancer aucune cellule pour « voir », sauf un échec technique : crash ou
  préemption, rapporté.
- Ne changer aucune recette ni aucun seuil après lecture.
- Ne pas dire « validé à 100 % ». Le résultat est un verdict par modèle, sous ces
  critères.
