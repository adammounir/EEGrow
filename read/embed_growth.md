# Quand l'empreinte apparaît-elle ? — trajectoire des neurones ajoutés (EXPLO_growth_trajectory_A.md, exploratoire)

lee2019_mi, cross_session, n = 54 sujets, hasard = 0.019

exclus : {'tiny': {}, 'random': {}, 'bd': {}}

## Top-1 par ensemble et instant (ex æquo comptés ; « clones » = fraction d'items identiques à ceux d'un autre sujet ; « nuls » = neurones de norme nulle)

| bras | ensemble | instant | top-1 | clones | nuls |
|---|---|---|---|---|---|
| bd | bd40 | 0 | 0.019 | 1.00 | 0 |
| bd | bd40 | 10 | 0.046 | 0.00 | 0 |
| bd | bd40 | 25 | 0.093 | 0.00 | 0 |
| bd | bd40 | 50 | 0.130 | 0.00 | 0 |
| bd | bd40 | 100 | 0.157 | 0.00 | 0 |
| bd | bd40 | 200 | 0.148 | 0.00 | 0 |
| random | G1 | ins | 0.024 | 0.93 | 0 |
| random | G1 | 10 | 0.056 | 0.00 | 0 |
| random | G1 | 25 | 0.157 | 0.00 | 0 |
| random | G1 | 50 | 0.204 | 0.00 | 0 |
| random | G1 | 100 | 0.259 | 0.00 | 0 |
| random | G1 | 200 | 0.204 | 0.00 | 0 |
| random | added | ins | 0.019 | 0.00 | 0 |
| random | added | 100 | 0.287 | 0.00 | 0 |
| random | added | 200 | 0.315 | 0.00 | 0 |
| random | init8 | 0 | 0.019 | 0.98 | 0 |
| random | init8 | 10 | 0.028 | 0.00 | 0 |
| random | init8 | 25 | 0.093 | 0.00 | 0 |
| random | init8 | 50 | 0.111 | 0.00 | 0 |
| random | init8 | 100 | 0.120 | 0.00 | 0 |
| random | init8 | 200 | 0.102 | 0.00 | 0 |
| tiny | G1 | ins | 0.046 | 0.00 | 0 |
| tiny | G1 | 10 | 0.037 | 0.00 | 0 |
| tiny | G1 | 25 | 0.102 | 0.00 | 0 |
| tiny | G1 | 50 | 0.157 | 0.00 | 0 |
| tiny | G1 | 100 | 0.213 | 0.00 | 0 |
| tiny | G1 | 200 | 0.269 | 0.00 | 0 |
| tiny | added | ins | 0.046 | 0.00 | 0 |
| tiny | added | 100 | 0.315 | 0.00 | 0 |
| tiny | added | 200 | 0.324 | 0.00 | 0 |
| tiny | init8 | 0 | 0.019 | 0.99 | 0 |
| tiny | init8 | 10 | 0.019 | 0.00 | 0 |
| tiny | init8 | 25 | 0.037 | 0.00 | 0 |
| tiny | init8 | 50 | 0.111 | 0.00 | 0 |
| tiny | init8 | 100 | 0.139 | 0.00 | 0 |
| tiny | init8 | 200 | 0.148 | 0.00 | 0 |

## Contrastes (appariés par sujet, IC95 bootstrap)

| contraste | IC95 | MDE |
|---|---|---|
| T1 random added ins - chance | +0.000 [-0.019, +0.028] | 0.036 |
| T2 random added 200 - ins | +0.296 [+0.194, +0.398] | 0.146 |
| T4 tiny added ins - chance | +0.028 [-0.009, +0.065] | 0.056 |
| tiny added 200 - ins | +0.278 [+0.167, +0.389] | 0.160 |
| T5 tiny - random, added, ins | +0.028 [-0.019, +0.074] | 0.068 |
| T5 tiny - random, added, 100 | +0.028 [-0.093, +0.148] | 0.183 |
| T5 tiny - random, added, 200 | +0.009 [-0.111, +0.130] | 0.176 |
| T3 random G1 ins - chance | +0.006 [-0.012, +0.030] | 0.030 |
| T3 random G1 10 - chance | +0.037 [-0.009, +0.102] | 0.080 |
| T3 random G1 25 - chance | +0.139 [+0.065, +0.222] | 0.116 |
| T3 random G1 50 - chance | +0.185 [+0.102, +0.278] | 0.131 |
| T3 random G1 100 - chance | +0.241 [+0.139, +0.352] | 0.152 |
| T3 random G1 200 - chance | +0.185 [+0.102, +0.278] | 0.126 |

**Lecture (§ 4)** : T1 ≈ hasard, T2 > 0 → l'empreinte est écrite par l'entraînement après l'insertion (exploratoire).
