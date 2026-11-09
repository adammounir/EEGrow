# La signature du sujet dans l'adaptateur — 4 modèles × 5 datasets à 2 sessions (PREREG_growth_signature.md)

- adaptateurs ré-évalués : 17856, écart au score du passage d'adaptation > 0.005 : 0
- garde-fou G-sig (≤ 5 % d'écarts) : **PASSE**

## Critère principal (groupé sur les datasets à 2 sessions, Holm sur 4, α = 0.025)

| critère | k | écart [IC95] | p sup. | n sujets / datasets | MDE | verdict |
|---|---|---|---|---|---|---|
| T1 croissance (b=8) : son adaptateur − celui d'un autre | 10 | +0.015 [+0.009, +0.022] | 0.0000 | 108 / 5 | 0.009 | **SUPÉRIEUR** |
| T2 T1 croissance − T1 LoRA (b=8) | 10 | -0.007 [-0.010, -0.004] | 1.0000 | 108 / 5 | 0.004 | **INFÉRIEUR** |
| T1 croissance (b=8) : son adaptateur − celui d'un autre | 50 | +0.037 [+0.029, +0.045] | 0.0000 | 78 / 4 | 0.012 | **SUPÉRIEUR** |
| T2 T1 croissance − T1 LoRA (b=8) | 50 | -0.004 [-0.007, -0.001] | 0.9950 | 78 / 4 | 0.005 | **INFÉRIEUR** |

## Rapporté

| contraste | k=10 | k=50 |
|---|---|---|
| T1 croissance b=2 | +0.015 [+0.009, +0.022] | +0.034 [+0.027, +0.042] |
| T1 LoRA b=8 | +0.022 [+0.015, +0.029] | +0.041 [+0.032, +0.050] |
| T1 LoRA b=2 | +0.015 [+0.009, +0.022] | +0.035 [+0.027, +0.043] |
| T1 croissance − LoRA, b=2 | -0.000 [-0.003, +0.003] | -0.000 [-0.003, +0.002] |

### T1 croissance b=8 par modèle

| modèle | k=10 | k=50 |
|---|---|---|
| shallow | +0.026 [+0.018, +0.033] | +0.061 [+0.049, +0.072] |
| eegnex | +0.018 [+0.013, +0.023] | +0.026 [+0.020, +0.034] |
| conformer | +0.005 [-0.008, +0.017] | +0.033 [+0.020, +0.047] |
| msvtnet | +0.014 [+0.004, +0.024] | +0.027 [+0.016, +0.038] |
