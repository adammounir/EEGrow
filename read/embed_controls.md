# Contrôles de l'empreinte des neurones ajoutés — lecture

## lee2019_mi (n = 54, hasard = 0.019)

| embedding | top-1 | AUC |
|---|---|---|
| added | 0.370 | 0.782 |
| width_only | 0.019 | 0.589 |
| bd_matched | 0.039 | 0.565 |
| cov | 0.954 | 0.996 |
| cov+added | 0.944 | 0.996 |

| contraste (top-1, apparié par sujet) | IC95 |
|---|---|
| (a) added - bd_matched | +0.331 [+0.211, +0.457] |
| (b) added - width_only | +0.352 [+0.231, +0.472] |
| (c) added - cov | -0.583 [-0.704, -0.463] |
| (c) cov+added - cov | -0.009 [-0.028, +0.000] |

## bnci2014_001 (n = 9, hasard = 0.111)

| embedding | top-1 | AUC |
|---|---|---|
| added | 0.574 | 0.811 |
| width_only | 0.074 | 0.485 |
| bd_matched | 0.301 | 0.665 |
| cov | 1.000 | 1.000 |
| cov+added | 1.000 | 0.999 |

| contraste (top-1, apparié par sujet) | IC95 |
|---|---|
| (a) added - bd_matched | +0.273 [+0.115, +0.425] |
| (b) added - width_only | +0.500 [+0.130, +0.815] |
| (c) added - cov | -0.426 [-0.685, -0.185] |
| (c) cov+added - cov | +0.000 [+0.000, +0.000] |

## Verdicts (lee2019_mi, PREREG § 7)

- (a)+(b) : **EMPREINTE NON TRIVIALE**
- (c) : **REDONDANTE AVEC LA COVARIANCE**
