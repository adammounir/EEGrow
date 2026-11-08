# Empreinte du sujet dans les neurones ajoutés — campagne A (PREREG_growth_embedding_A.md)

## Verdicts (lee2019_mi, PREREG § 5)

- P1 ajoutés − bd32 = +0.183 [+0.081, +0.290] : **EMPREINTE PROPRE À LA CROISSANCE**
- P2 ajoutés − ajoutés_random = +0.009 [-0.111, +0.130] : **PAS PAR LA RÈGLE**
- La règle TINY écrit le sujet dans ses neurones (P1 et P2) : **NON**
- « Au-delà de la covariance » : décrit, pas testé (PREREG § 9).

## lee2019_mi (n = 54 sujets à 2 sessions, hasard = 0.019)

fits lus {'tiny': 108, 'bd': 108, 'random': 108, 'tiny_bdinit': 108}, exclus {'tiny': 0, 'random': 0, 'tiny_bdinit': 0, 'bd': 0}

| embedding | top-1 | AUC |
|---|---|---|
| added | 0.324 | 0.890 |
| init8 | 0.148 | 0.776 |
| added_random | 0.315 | 0.892 |
| added_bdinit | 0.370 | 0.880 |
| bd40 | 0.148 | 0.765 |
| bd32 | 0.141 | 0.723 |
| cov+bd32 | 0.954 | — |
| added_spectrum | 0.093 | 0.774 |
| added_topo | 0.296 | 0.892 |
| cov | 0.954 | 0.996 |
| cov+added | 0.954 | 0.996 |

| contraste (apparié par sujet) | IC95 | MDE |
|---|---|---|
| P1 added - bd32 | +0.183 [+0.081, +0.290] | 0.151 |
| P2 added - added_random | +0.009 [-0.111, +0.130] | 0.176 |
| added_bdinit - bd32 | +0.230 [+0.106, +0.354] | 0.180 |
| added - init8 | +0.176 [+0.056, +0.296] | 0.174 |
| added - bd40 | +0.176 [+0.065, +0.296] | 0.170 |
| added_random - bd32 | +0.174 [+0.066, +0.282] | 0.155 |
| cov+added - cov | +0.000 [+0.000, +0.000] | 0.000 |
| cov+bd32 - cov | +0.000 [+0.000, +0.000] | 0.000 |
| added - chance | +0.306 [+0.204, +0.407] | 0.149 |

## bnci2014_001 (n = 9 sujets à 2 sessions, hasard = 0.111)

fits lus {'tiny': 18, 'bd': 18, 'random': 18, 'tiny_bdinit': 18}, exclus {'tiny': 0, 'random': 0, 'tiny_bdinit': 0, 'bd': 0}

| embedding | top-1 | AUC |
|---|---|---|
| added | 0.611 | 0.850 |
| init8 | 0.333 | 0.647 |
| added_random | 0.833 | 0.904 |
| added_bdinit | 0.778 | 0.895 |
| bd40 | 0.333 | 0.796 |
| bd32 | 0.381 | 0.747 |
| cov+bd32 | 1.000 | — |
| added_spectrum | 0.278 | 0.713 |
| added_topo | 0.611 | 0.880 |
| cov | 1.000 | 1.000 |
| cov+added | 1.000 | 1.000 |

| contraste (apparié par sujet) | IC95 | MDE |
|---|---|---|
| P1 added - bd32 | +0.231 [+0.086, +0.431] | 0.267 |
| P2 added - added_random | -0.222 [-0.500, +0.056] | 0.412 |
| added_bdinit - bd32 | +0.397 [+0.056, +0.703] | 0.497 |
| added - init8 | +0.278 [+0.000, +0.611] | 0.473 |
| added - bd40 | +0.278 [+0.056, +0.500] | 0.339 |
| added_random - bd32 | +0.453 [+0.253, +0.664] | 0.315 |
| cov+added - cov | +0.000 [+0.000, +0.000] | 0.000 |
| cov+bd32 - cov | +0.000 [+0.000, +0.000] | 0.000 |
| added - chance | +0.500 [+0.222, +0.722] | 0.389 |

## bnci2015_001 (n = 8 sujets à 2 sessions, hasard = 0.125)

fits lus {'tiny': 16, 'bd': 16, 'random': 16, 'tiny_bdinit': 16}, exclus {'tiny': 0, 'random': 0, 'tiny_bdinit': 0, 'bd': 0}

| embedding | top-1 | AUC |
|---|---|---|
| added | 0.375 | 0.772 |
| init8 | 0.375 | 0.806 |
| added_random | 0.375 | 0.768 |
| added_bdinit | 0.250 | 0.772 |
| bd40 | 0.500 | 0.728 |
| bd32 | 0.241 | 0.625 |
| cov+bd32 | 0.956 | — |
| added_spectrum | 0.250 | 0.663 |
| added_topo | 0.438 | 0.772 |
| cov | 0.938 | 0.998 |
| cov+added | 1.000 | 1.000 |

| contraste (apparié par sujet) | IC95 | MDE |
|---|---|---|
| P1 added - bd32 | +0.134 [-0.153, +0.444] | 0.460 |
| P2 added - added_random | +0.000 [-0.375, +0.250] | 0.459 |
| added_bdinit - bd32 | +0.009 [-0.259, +0.294] | 0.419 |
| added - init8 | +0.000 [-0.375, +0.375] | 0.592 |
| added - bd40 | -0.125 [-0.562, +0.312] | 0.688 |
| added_random - bd32 | +0.134 [-0.188, +0.456] | 0.497 |
| cov+added - cov | +0.062 [+0.000, +0.188] | 0.175 |
| cov+bd32 - cov | +0.019 [+0.000, +0.056] | 0.053 |
| added - chance | +0.250 [+0.000, +0.562] | 0.439 |

## Exploratoire 7.1 — cross_subject, signature du sujet laissé de côté (lee2019_mi)

n = 54, hasard = 0.019. Rien n'en est conclu (PREREG § 7.1).

| contraste | IC95 |
|---|---|
| tiny - chance | +0.019 [-0.019, +0.074] |
| bd - chance | +0.019 [-0.019, +0.074] |
| tiny - bd | +0.000 [-0.074, +0.074] |

## Exploratoire 7.2 — EEGNeX, spectre des noyaux ajoutés (jonction 2, lee2019_mi)

n = 54, fits {'tiny': (108, 0), 'bd': (108, 0)}

| contraste | IC95 |
|---|---|
| eegnex added - bd | +0.037 [-0.028, +0.102] |
| eegnex added - chance | +0.074 [+0.009, +0.148] |

