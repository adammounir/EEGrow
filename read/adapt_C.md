# Une taille qui s'adapte au sujet — 4 modèles × 11 datasets (PREREG_growth_adaptive_size.md)

- modèles retenus (G1 de la lecture 3, CTNet exclu) : ['shallow', 'eegnex', 'conformer', 'msvtnet']
- taux d'apprentissage (réglés sur bnci2014_001) : {"shallow": {"grow1": 0.01, "grow2": 0.01, "grow4": 0.003, "grow8": 0.003, "grow16": 0.003, "lora1": 0.1, "lora2": 0.01, "lora4": 0.03, "lora8": 0.03, "lora16": 0.01}, "eegnex": {"grow1": 0.01, "grow2": 0.01, "grow4": 0.01, "grow8": 0.01, "grow16": 0.01, "lora1": 0.01, "lora2": 0.03, "lora4": 0.03, "lora8": 0.01, "lora16": 0.03}, "conformer": {"grow1": 0.03, "grow2": 0.01, "grow4": 0.01, "grow8": 0.003, "grow16": 0.003, "lora1": 0.01, "lora2": 0.01, "lora4": 0.01, "lora8": 0.01, "lora16": 0.003}, "msvtnet": {"grow1": 0.03, "grow2": 0.01, "grow4": 0.01, "grow8": 0.003, "grow16": 0.003, "lora1": 0.01, "lora2": 0.01, "lora4": 0.01, "lora8": 0.003, "lora16": 0.003}}
- meilleure taille fixe (réglée sur bnci2014_001, k = 10) : {"shallow": {"grow": 4, "lora": 16}, "eegnex": {"grow": 16, "lora": 16}, "conformer": {"grow": 16, "lora": 8}, "msvtnet": {"grow": 16, "lora": 16}}
- cellules refusées par G0 : []
- sujets exclus : {}

## Critère principal (groupé, Holm sur 6, α = 0.025 unilatéral, marge NI -0.02)

| critère | k | écart [IC95] | p sup. | n sujets / datasets | MDE | verdict |
|---|---|---|---|---|---|---|
| S1 taille du sujet − taille d'un autre (croissance) | 10 | -0.000 [-0.002, +0.002] | 0.6251 | 315 / 11 | 0.003 | **NON INFÉRIEUR** |
| S2 croissance − LoRA, tailles choisies | 10 | +0.000 [-0.001, +0.002] | 0.3609 | 315 / 11 | 0.002 | **NON INFÉRIEUR** |
| S3 croissance : taille choisie − meilleure taille fixe | 10 | +0.001 [-0.001, +0.002] | 0.2607 | 315 / 11 | 0.003 | **NON INFÉRIEUR** |
| S1 taille du sujet − taille d'un autre (croissance) | 50 | +0.004 [+0.002, +0.006] | 0.0001 | 157 / 7 | 0.003 | **SUPÉRIEUR** |
| S2 croissance − LoRA, tailles choisies | 50 | +0.001 [-0.001, +0.003] | 0.1668 | 158 / 7 | 0.002 | **NON INFÉRIEUR** |
| S3 croissance : taille choisie − meilleure taille fixe | 50 | +0.000 [-0.001, +0.002] | 0.3869 | 158 / 7 | 0.002 | **NON INFÉRIEUR** |

## Secondaire — par modèle (Holm sur 24)

| modèle | k | S1 taille du sujet − taille d'un autre (croissance) | S2 croissance − LoRA, tailles choisies | S3 croissance : taille choisie − meilleure taille fixe |
|---|---|---|---|---|
| shallow | 10 | -0.001 [-0.004, +0.002] **NON INFÉRIEUR** | +0.004 [+0.001, +0.007] **NON INFÉRIEUR** | -0.003 [-0.006, -0.000] **NON INFÉRIEUR** |
| shallow | 50 | +0.005 [+0.002, +0.008] **NON INFÉRIEUR** | +0.001 [-0.002, +0.005] **NON INFÉRIEUR** | -0.001 [-0.004, +0.002] **NON INFÉRIEUR** |
| eegnex | 10 | +0.002 [-0.000, +0.004] **NON INFÉRIEUR** | -0.002 [-0.005, +0.001] **NON INFÉRIEUR** | -0.011 [-0.013, -0.008] **NON INFÉRIEUR** |
| eegnex | 50 | +0.002 [-0.000, +0.004] **NON INFÉRIEUR** | -0.002 [-0.006, +0.003] **NON INFÉRIEUR** | -0.004 [-0.005, -0.003] **NON INFÉRIEUR** |
| conformer | 10 | -0.002 [-0.007, +0.004] **NON INFÉRIEUR** | +0.000 [-0.003, +0.004] **NON INFÉRIEUR** | +0.016 [+0.012, +0.021] **SUPÉRIEUR** |
| conformer | 50 | +0.006 [+0.000, +0.013] **NON INFÉRIEUR** | +0.003 [+0.001, +0.006] **NON INFÉRIEUR** | +0.007 [+0.002, +0.012] **NON INFÉRIEUR** |
| msvtnet | 10 | -0.001 [-0.004, +0.002] **NON INFÉRIEUR** | -0.001 [-0.004, +0.002] **NON INFÉRIEUR** | -0.000 [-0.003, +0.003] **NON INFÉRIEUR** |
| msvtnet | 50 | +0.003 [+0.000, +0.005] **NON INFÉRIEUR** | +0.001 [-0.003, +0.004] **NON INFÉRIEUR** | -0.001 [-0.003, +0.001] **NON INFÉRIEUR** |

## Rapporté (sans valeur décisionnelle)

| contraste | k=10 | k=50 |
|---|---|---|
| lora_cv − lora_perm | +0.001 [-0.001, +0.003] | +0.004 [+0.002, +0.006] |
| lora_cv − lora_fix | +0.003 [+0.001, +0.006] | -0.001 [-0.003, +0.001] |
| grow_oracle − grow_fix | +0.035 [+0.032, +0.037] | +0.016 [+0.015, +0.018] |
| lora_oracle − lora_fix | +0.043 [+0.041, +0.046] | +0.020 [+0.018, +0.023] |
| grow_fix − lora_fix | +0.003 [+0.001, +0.005] | -0.000 [-0.002, +0.002] |

### Tailles choisies (part des tirages, toutes cellules)

| modèle | k | famille | b=0 | b=1 | b=2 | b=4 | b=8 | b=16 |
|---|---|---|---|---|---|---|---|---|
| shallow | 10 | grow | 0.41 | 0.19 | 0.13 | 0.10 | 0.08 | 0.09 |
| shallow | 10 | lora | 0.39 | 0.15 | 0.15 | 0.10 | 0.10 | 0.11 |
| shallow | 50 | grow | 0.17 | 0.16 | 0.15 | 0.12 | 0.19 | 0.22 |
| shallow | 50 | lora | 0.16 | 0.10 | 0.11 | 0.16 | 0.23 | 0.24 |
| eegnex | 10 | grow | 0.39 | 0.18 | 0.00 | 0.00 | 0.16 | 0.27 |
| eegnex | 10 | lora | 0.30 | 0.19 | 0.13 | 0.14 | 0.10 | 0.14 |
| eegnex | 50 | grow | 0.10 | 0.09 | 0.00 | 0.00 | 0.15 | 0.66 |
| eegnex | 50 | lora | 0.13 | 0.14 | 0.14 | 0.21 | 0.16 | 0.22 |
| conformer | 10 | grow | 0.56 | 0.16 | 0.10 | 0.08 | 0.07 | 0.04 |
| conformer | 10 | lora | 0.59 | 0.16 | 0.07 | 0.07 | 0.06 | 0.04 |
| conformer | 50 | grow | 0.40 | 0.20 | 0.11 | 0.10 | 0.11 | 0.08 |
| conformer | 50 | lora | 0.41 | 0.18 | 0.10 | 0.09 | 0.11 | 0.10 |
| msvtnet | 10 | grow | 0.49 | 0.18 | 0.11 | 0.12 | 0.05 | 0.04 |
| msvtnet | 10 | lora | 0.54 | 0.17 | 0.09 | 0.11 | 0.06 | 0.03 |
| msvtnet | 50 | grow | 0.24 | 0.19 | 0.12 | 0.26 | 0.09 | 0.11 |
| msvtnet | 50 | lora | 0.30 | 0.16 | 0.11 | 0.21 | 0.11 | 0.11 |
