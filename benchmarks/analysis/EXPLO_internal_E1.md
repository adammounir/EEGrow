# EXPLO — E1 : ce que la croissance fait à l'intérieur des modèles (campagne A, hors ligne)

09/11/2026. Analyse **exploratoire**, annoncée comme telle : elle génère des hypothèses,
elle ne décide rien dans un papier sans réplication. Plan et script
(`explo_internal_E1.py`) gelés dans le même commit, avant la lecture en masse des poids.
0 GPU-h : poids, historiques et logits déjà archivés, aucun réentraînement, aucune passe
avant sur des données.

Lu avant le gel (pour écrire le script, pas comme résultat) : les noms et formes des
tenseurs d'un fit par bras (bnci2014_001), les clés des `__fits.jsonl`, et les largeurs
finales de 3 fits par bras sur bnci2014_001 `within_session`. Vu à cette occasion :
Conformer tiny finit avec la largeur FFN [152, 16, 16, 16, 16, 16] sur un fit (~20 % de
la cible 960 au total), CTNet ~70 %, MSVTNet 100 %. C'est ce qui motive R1 ; R1 est donc
une **description** de ce motif sur toute la campagne, pas un test.

## 1. Pourquoi

À accuracy égale (campagne A, NI sur 13/15), un papier de premier rang ne s'intéresse à la
croissance que si elle change **ce que le modèle apprend à l'intérieur**, et si ce
changement est exploitable. Les transformers comptent le plus : c'est là que la croissance
pourrait devenir une technique réutilisable (où mettre la capacité, ce qu'un modèle complet
n'utilise pas).

Dans les 3 transformers de la campagne, les FFN de **chaque** bloc sont croissables ; à
chaque étape, une seule jonction grandit : celle dont le gain du premier ordre TINY est le
plus grand (`loop._grow_step_best`). Le bras random choisit la jonction **sur les mêmes
gains TINY** et ne change que la direction des neurones (`loop.py` l. 672). L'allocation
entre blocs est donc une propriété du critère TINY dans les deux bras.

## 2. Données

- Archive `campaignA` (lecture seule) ; cellules de `campaignA_grid` et `campaignS_grid`
  (`*_planned_av5.txt`) ; 5 modèles, bras bd / tiny / random (+ fix, tiny_bdinit pour
  Shallow) ; 3 protocoles ; graine 0.
- Par fit : `__fits.jsonl` (historique de croissance), `__fits__final` (poids, époque 200),
  `__fits__init` (poids à l'init), `__fits__logits` (logits de test).
- Une cellule dont le jsonl n'a pas le nombre de fits attendu (5 × sessions, sessions, 1)
  ou dont un fichier manque est exclue et comptée. Un fit divergé (poids ou logits non
  finis) est exclu des mesures concernées et compté.

## 3. Mesures (par fit)

**Jonctions.** Couche cachée h d'un FFN : W1 (h × d), W2 (d × h). Shallow : conv_time →
conv_spat (h = filtres temporels). EEGNeX : block_4 → block_5 (h = 32).

- `n_eff` : rapport de participation de l'importance des neurones
  s_j = ‖W1[j]‖ · ‖W2[:, j]‖ : (Σ s)² / Σ s². Nombre de neurones qui portent
  effectivement la jonction. `n_eff_frac` = n_eff / h.
- `srank_M`, `erank_M`, `norm2_M` (FFN linéaires) : rang stable, rang effectif (Roy &
  Vetterli, exp de l'entropie des σ normalisés) et norme spectrale de M = W2 · W1 (d × d),
  l'application linéaire du FFN sans la non-linéarité.
- `erank_W1`, `erank_W2` : rang effectif de W1 et de W2 mis à plat par neurone (h lignes).
- `drift_W1`, `drift_W2` : ‖W − W_init‖ / ‖W_init‖, seulement quand les formes init et
  final sont égales (bd, fix) : combien l'entraînement a déplacé le FFN de chaque bloc.

**Allocation (modèles à plusieurs jonctions : Conformer, CTNet, MSVTNet).**
- `width_frac_j` : largeur finale du bloc j / sa cible ;
- `foi_share_j` : moyenne, sur les étapes de croissance, de gain_j / Σ gains ;
- `pick_frac_j` : fraction des étapes où le bloc j est choisi.

**Sorties** (logits de test, softmax float32 comme le scoreur) : `nll`, `ece` (15 classes
de confiance, top-1), `acc` ; **accord** entre bras sur les mêmes essais de test
(appariés par sujet, protocole, `cv_ind`, sessions et `test_idx` identiques, sinon refus
de la paire et compte) : tiny–bd, random–bd, tiny–random (Shallow : aussi fix–bd,
tiny_bdinit–bd).

## 4. Ce qui sera rapporté (et rien d'autre)

Moyennes par sujet (moyenne de ses fits), puis groupées sur les datasets (poids √n_d,
bootstrap stratifié 10 000, `default_rng(0)`, comme les lectures 3-4), par protocole.

- **R1 — Où la croissance met la capacité.** Par modèle multi-jonction × bras ×
  protocole : `width_frac_j`, `foi_share_j`, `pick_frac_j` par bloc ; fraction de la
  largeur FFN de bd utilisée ; profil par dataset (tiny, within_session).
- **R2 — Le modèle complet utilise-t-il les mêmes blocs ?** Par bloc, chez bd :
  `n_eff_frac`, `drift_W1`, `drift_W2`, `norm2_M`, `srank_M`. Corrélation de Spearman
  entre blocs, par paire appariée (sujet, fit), entre l'allocation tiny (`foi_share`,
  `width_frac`) et l'usage bd (`n_eff_frac`, `drift_W2`, `norm2_M`), moyennée et groupée
  (Conformer et CTNet seulement : MSVTNet n'a que 2 blocs).
- **R3 — Même accuracy, même solution interne ?** Par modèle : Σ_blocs `n_eff` et
  `erank_M` moyen, tiny − bd et random − bd (contrastes groupés).
- **R4 — Sorties.** `nll`, `ece` : tiny − bd, random − bd ; accords tiny–bd,
  random–bd, tiny–random.
- **R5 — Shallow (petite init).** conv_time et conv_spat : `erank_W1`, `erank_W2`,
  `n_eff`, normes ; contrastes tiny − bd, random − bd, fix − bd, tiny_bdinit − bd,
  tiny − tiny_bdinit.

Lecture prévue, sans seuil de décision :
- R1 + R2 : si l'allocation tiny suit l'usage bd (Spearman > 0 groupé), la croissance
  **lit** quels blocs un modèle complet utilise, à une fraction du coût : piste « sonde de
  capacité » à confirmer par une ablation fonctionnelle (E1b, passes avant). Si elle ne le
  suit pas, l'allocation reflète l'échelle du critère TINY par bloc (biais de profondeur
  possible), et c'est ce biais qu'il faut comprendre avant toute technique.
- R3 : si n_eff(tiny) ≈ n_eff(bd) alors que tiny est plus étroit, bd « porte » des
  neurones inutiles que la croissance n'ajoute pas.

## 5. Limites connues avant lecture

- Poids seuls : rien ne dit ici qu'un bloc étroit est **inutile** (pas d'ablation), ni ce
  que le FFN calcule avec la non-linéarité ; M = W2 · W1 ignore ReLU/GELU.
- `n_eff` et les rangs dépendent de h : on rapporte absolu et fraction.
- Graine 0 partout : pas de variance de graine pour juger un accord entre bras ; l'accord
  tiny–random sert de point de comparaison partiel.
- L'arrêt de croissance de Conformer/CTNet tient au plancher `MIN_SINGULAR_RATIO = 0.1`
  et au budget de 39 étapes (choix eegrow) ; la campagne dose (810582) a montré que
  grandir jusqu'à la cible n'apporte pas d'accuracy.
