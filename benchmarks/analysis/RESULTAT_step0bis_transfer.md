# Résultat — étape 0-bis : transfert des critères de comportement hors de bnci

PREREG : `PREREG_step0bis_transfer.md`, gelé au commit `313ea03` avant tout lancement.
Code des cellules : `313ea03` (checkout `/scratch/amounir/eegrow_step0bis`, `.eegrow_sha`).
Lecture : `benchmarks/step0bis_criteria.py`, sorties dans `step0_read/step0bis/`.

## physionetmi — LU le 14/10/2026

Job 756409, 12 cellules, toutes `COMPLETED` rc=0 (50 min à 1 h 37 par cellule sur 2080 Ti).
Garde-fou : 12 cellules sur 12 admises, 545 fits chacune (109 sujets × 5 plis), 0 pic.

| bras | optimum ≤5 / ≥195 | prop./place | spectre (déc.) | cible | tout tôt | censure (s=64) | départ (excès ln 2) | verdict |
|---|---|---|---|---|---|---|---|---|
| bd_shallow | **28.2 %** / 0.3 % | n/a | n/a | n/a | n/a | n/a | +0.294 | **REFUS** (optimum) |
| bd_deep4 | 17.2 % / 1.0 % | n/a | n/a | n/a | n/a | n/a | +0.374 | passe |
| grow_shallow | 13.9 % / 0.4 % | 5.2 | 3.04 | 99.1 % | 0.1 % | 1.86 % | +0.062 | passe |
| grow_deep4 | 7.2 % / 0.7 % | 3.7 | 2.25 | 99.9 % | 0.0 % | 7.56 % | +0.370 | passe |

**Application de la règle § 4, sans correction :**
- Les deux bras `grow_*` passent tous les critères sur physionetmi. La normalisation et la grille
  de l'avenant 2 ne sont **pas** spécifiques à bnci sur ce dataset : la censure reste sous 20 %
  (1.9 % et 7.6 %).
- `bd_shallow` est refusé par l'optimum : 28.2 % de ses fits choisissent une époque ≤ 5 (seuil
  20 %). C'est le choix de l'époque qui est en cause, pas la croissance. Le contraste
  `grow_shallow` contre `bd_shallow` **n'est pas lisible à l'étape 2 sur physionetmi**.
  Aucune correction n'est lancée sans un nouveau PREREG.
- Le contraste `grow_deep4` contre `bd_deep4` reste lisible sur physionetmi.

Aucune accuracy n'a été lue (PREREG § 3).

## shin2017a — LU le 15/10/2026

Job 756889, 12 cellules, toutes `COMPLETED` rc=0 (33 min à 1 h 06 par cellule sur 2080 Ti,
margpu017/018/019/022), même commande et même checkout que physionetmi.
Garde-fou : 12 cellules sur 12 admises, 435 fits chacune (29 sujets × 3 sessions × 5 plis), 0 pic.

| bras | optimum ≤5 / ≥195 | prop./place | spectre (déc.) | cible | tout tôt | censure (s=64) | départ (excès ln 2) | verdict |
|---|---|---|---|---|---|---|---|---|
| bd_shallow | **53.3 %** / 0.0 % | n/a | n/a | n/a | n/a | n/a | +0.304 | **REFUS** (optimum) |
| bd_deep4 | **36.2 %** / 0.5 % | n/a | n/a | n/a | n/a | n/a | +0.375 | **REFUS** (optimum) |
| grow_shallow | **28.4 %** / 0.3 % | 5.2 | 2.00 | 95.7 % | 0.6 % | 1.82 % | +0.062 | **REFUS** (optimum) |
| grow_deep4 | **25.7 %** / 0.4 % | 3.7 | 3.34 | 95.6 % | 1.3 % | 11.16 % | +0.368 | **REFUS** (optimum) |

**Application de la règle § 4, sans correction :**
- Les quatre bras sont refusés sur shin2017a, **tous par le seul critère d'optimum** (> 20 % des
  fits avec `selected_epoch` ≤ 5). Aucun autre critère ne refuse.
- Les critères propres à la croissance passent tous pour les deux bras `grow_*` : sélection,
  spectre, cible, calendrier. La **censure reste sous 20 %** (1.8 % et 11.2 %) : la grille de
  l'avenant 2 n'est pas déclarée spécifique à bnci, ni ici ni sur physionetmi.
- Les deux bras `bd_*` sont refusés par l'optimum : selon le § 4, c'est le choix de l'époque, et
  non la croissance, qui est en cause. **Aucun contraste grow contre bd n'est lisible à l'étape 2
  sur shin2017a**, et aucun bras ne peut y porter d'endpoint.
- Aucune correction n'est lancée sans un nouveau PREREG.

Aucune accuracy n'a été lue (PREREG § 3).

## Bilan de l'étape 0-bis

| dataset | bd_shallow | bd_deep4 | grow_shallow | grow_deep4 |
|---|---|---|---|---|
| physionetmi | REFUS (optimum) | passe | passe | passe |
| shin2017a | REFUS (optimum) | REFUS (optimum) | REFUS (optimum) | REFUS (optimum) |

- Les critères de **croissance** (dont la censure de l'avenant 2) tiennent sur les deux datasets.
- Le critère qui ne transfère pas est l'**optimum**, c'est-à-dire la sélection de l'époque sur des
  splits minuscules (2 essais de held-out de croissance sur shin2017a). Il refuse les références
  braindecode comme les bras en croissance. C'est la pathologie déjà décrite dans
  `scored-epoch-quantised-valid` (59 % des fits sur shin2017a dans la campagne précédente).
  Elle est réduite, mais pas résolue.
- Seul contraste lisible à l'étape 2 hors bnci : `grow_deep4` contre `bd_deep4` sur physionetmi.
