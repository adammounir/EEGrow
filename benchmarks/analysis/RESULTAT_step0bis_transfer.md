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

## shin2017a — EN COURS

Téléchargement vérifié (29 sujets, 5.9 Go dans `/scratch/amounir/mne_data/MNE-eegfnirs-data`).
Job 756889, 12 cellules, lancé le 14/10/2026 avec la même commande et le même checkout.
