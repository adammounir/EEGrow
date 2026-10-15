# Catalogue — quels décodeurs deep peut-on faire croître pour relancer le benchmark ?

14/10/2026.
- **Audit** : `GrowMo/audit_blockers.py`, sur Margaret, sous braindecode 1.5.2 (env `bench`, celui
  du benchmark) et 1.7.0 (env `bd`). Il part des 25 décodeurs MI ou « General » de
  braindecode entraînables from scratch. Sont exclus : les modèles de fondation, ceux pour le
  sommeil, l'EMG et l'émotion.
- **Implémentations** : `src/eegrow/models/growing_bd_junctions.py`.
- **Tests** : `tests/test_growing_bd_junctions.py`, 50 tests, passés sur Margaret sous 1.5.2.

**Méthode d'implémentation.** Le modèle braindecode est gardé entier : on remplace une seule
jonction par une paire gromo, et tous les `forward` restent ceux de braindecode.

**Critère de fidélité.** À la largeur braindecode, avec les poids copiés, les logits sont
identiques (écart max mesuré : 0). Le test vérifie aussi que la parité **échoue** si on modifie
un poids.

## 1. Implémentés, parité exacte, un pas de croissance gromo réel validé

| modèle | famille | params (22 ch × 1000) | jonction qui grandit | largeur braindecode | remarque |
|---|---|---|---|---|---|
| ShallowFBCSPNet | conv | 46 k | `conv_time → conv_spat` (latente, fusionnée par CombinedConv) | 40 | `grow_shallow`, étape 0 passée |
| Deep4Net | conv | 283 k | 1re jonction (latente) | 25 | `grow_deep4`, étape 0 passée |
| **EEGConformer** | conv + transformer | 790 k | FFN d'un bloc (6 au choix) **ou** patch conv→conv | 160 / 40 | conçu pour MI (BCI IV 2a) |
| **CTNet** | conv + transformer | 153 k | FFN d'un bloc (6 au choix) | 160 | MI (2a/2b) |
| **MSVTNet** | conv + transformer | 75 k | FFN d'une couche `nn.TransformerEncoderLayer` (2 au choix) | 72 | MI ; fast path fusionné désactivé (même fonction) |
| **TIDNet** | conv | 240 k | goulot conv1×1 → BN → conv(5,1) du filtre spatial dense | 72 | BN dans la jonction |
| **EEGNeX** | conv | 56 k | jonction 2 : `block_4` → BN → `block_5` | 32 | MI ; BN dans la jonction. La jonction 1 est refusée, voir § 3 |
| ContraWR | conv (STFT) | 1.6 M | conv1 → BN → ELU → conv2 d'un bloc résiduel (4 au choix) | 32-256 | conçu pour le sommeil, moins pertinent en MI |

**Trois réserves**, déjà écrites dans le code :
1. La boucle eegrow fait croître **une seule couche par modèle**. Pour un transformer, cela
   veut dire un FFN sur six. Faire croître tous les FFN demande d'étendre `grow_step`
   (gromo sait gérer plusieurs couches).
2. Une **BatchNorm dans la jonction** (TIDNet, EEGNeX, ContraWR) est l'identité pour les
   candidats pendant la line search. En entraînement, elle renormalise ensuite les nouveaux
   canaux : l'amplitude choisie n'est pas celle que l'entraînement voit.
3. Essai sur données aléatoires (sur Margaret, 6 époques) : la perte de l'EEGConformer saute
   juste après la croissance de son FFN (1.38 → 2.9 et 1.42 → 5.9). Ce n'est pas encore un
   résultat, mais c'est à surveiller sur vraies données.

## 2. Possibles moyennant du travail

- **TCFormer** (MI, 2025, 78 k) : MLP natif 48 → 96 → 48 (2 blocs). Il est **absent de
  braindecode 1.5.2**, la version du benchmark ; l'utiliser demande de monter l'env à 1.7.
- **SCCNet** (MI, 14 k) : une seule paire, bloquée uniquement par un `permute` entre la BN et
  la conv. Le bras historique `GrowingSCCNet` la réécrit ; sa parité avec braindecode reste à
  vérifier.

## 3. Non croissables avec gromo tel quel (motif lu dans la trace)

| motif | modèles |
|---|---|
| convolutions groupées / depthwise (gromo commente `groups`) | EEGNet, SincShallowNet, AttentionBaseNet, FBMSNet, EEGITNet, EEGTCNet, FBLightConvNet |
| Conv1d (pas de `Conv1dGrowingModule`) | ATCNet, BDTCN, IFNet, EEGSimpleConv, SSTDPN |
| concaténations / branches partout | EEGInceptionMI (8.9 M params) |
| padding `"same"` asymétrique (noyau pair, 31/32 ; `im2col` de gromo exige un int symétrique) | EEGNeX jonction 1 |
| trace vide, non analysé : à vérifier à la main | FBCNet, EEGSym |

## 4. Proposition pour la relance

Couvrir deux familles avec des modèles conçus pour la MI :
- **conv** : Shallow, Deep4, EEGNeX (j2), TIDNet ;
- **conv + transformer** : EEGConformer, CTNet, MSVTNet.

Chaque nouveau bras doit passer l'**étape 0** avant le benchmark : capacité, optimum, critères
de croissance, censure. Il faut d'abord choisir son optimiseur, lu dans le papier ou dans
l'exemple braindecode, jamais inféré.

## 5. Recettes d'entraînement lues dans le code officiel (15/10/2026)

Lues dans les dépôts cités par les docstrings braindecode, sans rien exécuter. Le commit lu
est indiqué, et chaque valeur vient d'une ligne de code. Pour les candidats MI, on lit la
config **intra-sujet** (*subject-dependent*), celle du benchmark.

| modèle | source (commit) | optimiseur | lr | betas | wd | batch | époques / arrêt | choix du modèle | augmentation, autres |
|---|---|---|---|---|---|---|---|---|---|
| EEGConformer | `eeyhsong/EEG-Conformer` `9ae149b`, `conformer.py` l. 217-222, 334 | Adam | 2e-4 | (0.5, 0.999) | 0 | 72 | 2000, sans arrêt | **meilleure accuracy de TEST** (l. 57, pas de validation) | S&R (8 segments), batch augmenté |
| CTNet | `snailpt/CTNet` `fb83814`, `main_subject_specific.py` l. 293-302, 704-723 | Adam | 1e-3 (défaut `ExP`, non surchargé par `main`) | (0.5, 0.999) | 0 | 72 (+3× S&R) | 1000, sans arrêt | min `val_loss`, validation = 30 % du train | S&R ×3. Config 2a = emb 16 / 2 têtes / 6 couches ; braindecode met par défaut emb 40 / 4 têtes (défauts de `ExP`) |
| MSVTNet | `SheepTAO/MSVTNet` `c3fa7fd`, `train_MSVTNet.ipynb` ; dpeeg 0.3.6 `6085816` (`classifier.py` l. 65-73, `experiments.py` l. 290-305) | Adam (torch) | 1e-3 (défaut dpeeg) | défaut (0.9, 0.999) | 0 | 64 | deux phases : 1500 max, patience 100 sur `val_inacc`, puis 600 | meilleur état de la phase 1, puis ré-entraînement | S&R ×1, normalisation, perte jointe sur les branches (`JointCrossEntoryLoss`) |
| TIDNet | `SPOClab-ca/ThinkerInvariance` `784a8b8`, `main.py` l. 99-118, `train.py` l. 490-491 | Adam **amsgrad** | 1e-3 | défaut | **0.01** (L2 couplé) | 60 | 100, cosinus avec 20 époques de montée | meilleure validation | label smoothing 0.2, alignement euclidien par défaut |
| EEGNeX | `chenxiachan/EEGNeX` `044d7fb`, `Run_model.ipynb`, `BenchmarkModels.py` (Keras) | Adam Keras | 1e-3 (défaut Keras) | défaut (ε = 1e-7) | 0 | 128 | 200, arrêt `val_loss` patience 20, lr ×0.5 sur plateau (patience 5) | **dernière époque** (rechargement du checkpoint commenté) | split 75/12.5/12.5 aléatoire |
| ContraWR | `ycq091044/ContraWR` `217753a`, `src/supervised.py` l. 72-73, 149 | Adam | 2e-4 | défaut | 0 | 128 | 100 | — | **sommeil seulement** : aucune recette MI n'existe |

**Ce que ça implique pour l'étape 1.**
- Aucune recette n'est un « optimiseur seul » : chacune vient avec son protocole (époques,
  arrêt, choix du modèle, augmentation S&R pour trois modèles sur six). L'étape 0 a tranché la
  même question pour Shallow et Deep4 : **optimiseur de la source** (type, lr, betas, wd,
  batch), **protocole commun du benchmark** (200 époques, `valid_acc` lissée W = 5, sans
  augmentation). Appliquer la même règle ici rend les bras comparables. Mais les modèles
  qui dépendent de S&R (Conformer, CTNet) risquent d'échouer la capacité ou l'optimum : l'étape 0
  le dira, ce n'est pas à inférer.
- Le chiffre publié de l'EEGConformer est choisi sur le test : il ne peut pas servir de
  critère « référence publiée ».
- La largeur de CTNet dans braindecode n'est pas celle du papier pour BCI IV 2a.
- ContraWR n'a pas de recette MI : à écarter, ou à garder avec une recette déclarée non
  publiée.
