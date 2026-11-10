# E1b-bias — le coût de z_all est-il un décalage constant des logits ? (EXPLORATOIRE)

Fits lus : 18882. Portes : {'fits': 18882, 'groups': 19966, 'g1_fail': 0, 'g2_fail': 0, 'nat32_moved': 399, 'nat32_max_e6': 500000, 'excluded:e1b:g0': 2}.

A = acc(z_all) − acc(full) ; R = acc(z_all recentré) − acc(full) ; part récupérée = 1 − R/A (sur les moyennes).
Verdict (note gelée) : « biais » si borne basse de R > -0.01 ; « information » si borne haute de R < -0.01 ; sinon non concluant.

## Primaire : Conformer bd, datasets à 3-4 classes

- **within_session** : A = -0.041 [-0.051, -0.031] (n=35, 4 ds, MDE 0.015) ; R = -0.045 [-0.053, -0.037] (n=35, 4 ds, MDE 0.011) ; part récupérée -0.10 → **INFORMATION**
- **cross_session** : A = -0.047 [-0.066, -0.030] (n=13, 2 ds, MDE 0.026) ; R = -0.055 [-0.077, -0.036] (n=13, 2 ds, MDE 0.029) ; part récupérée -0.18 → **INFORMATION**
- **cross_subject** : A = -0.037 [-0.057, -0.018] (n=35, 4 ds, MDE 0.028) ; R = -0.080 [-0.095, -0.065] (n=35, 4 ds, MDE 0.021) ; part récupérée -1.14 → **INFORMATION**

**Conclusion écrite avant lecture : FFN = de l'information de classe (≥ 2/3 protocoles) : proposer la suite à Adam.**

## Secondaires (ne décident rien)

| modèle | bras | protocole | classes | A (accuracy) | R (recentré) | nat z − full | bias_share médian |
|---|---|---|---|---|---|---|---|
| conformer | bd | within_session | multi | -0.041 [-0.051, -0.031] (n=35, 4 ds, MDE 0.015) | -0.045 [-0.053, -0.037] (n=35, 4 ds, MDE 0.011) | -0.041 [-0.051, -0.031] (n=35, 4 ds, MDE 0.015) | 0.12 |
| conformer | bd | within_session | bin | -0.017 [-0.021, -0.012] (n=289, 8 ds, MDE 0.006) | -0.024 [-0.028, -0.020] (n=289, 8 ds, MDE 0.005) | -0.002 [-0.003, -0.000] (n=289, 8 ds, MDE 0.002) | 0.13 |
| conformer | bd | cross_session | multi | -0.047 [-0.066, -0.030] (n=13, 2 ds, MDE 0.026) | -0.055 [-0.077, -0.036] (n=13, 2 ds, MDE 0.029) | -0.047 [-0.066, -0.030] (n=13, 2 ds, MDE 0.026) | 0.11 |
| conformer | bd | cross_session | bin | -0.021 [-0.030, -0.012] (n=104, 4 ds, MDE 0.013) | -0.035 [-0.044, -0.026] (n=104, 4 ds, MDE 0.013) | +0.006 [+0.002, +0.011] (n=104, 4 ds, MDE 0.006) | 0.19 |
| conformer | bd | cross_subject | multi | -0.037 [-0.057, -0.018] (n=35, 4 ds, MDE 0.028) | -0.080 [-0.095, -0.065] (n=35, 4 ds, MDE 0.021) | -0.037 [-0.057, -0.018] (n=35, 4 ds, MDE 0.028) | 0.19 |
| conformer | bd | cross_subject | bin | -0.032 [-0.040, -0.024] (n=289, 8 ds, MDE 0.012) | -0.096 [-0.105, -0.086] (n=289, 8 ds, MDE 0.014) | -0.019 [-0.025, -0.014] (n=289, 8 ds, MDE 0.008) | 0.16 |
| conformer | tiny | within_session | multi | -0.051 [-0.064, -0.038] (n=35, 4 ds, MDE 0.019) | -0.033 [-0.042, -0.023] (n=35, 4 ds, MDE 0.014) | -0.051 [-0.064, -0.038] (n=35, 4 ds, MDE 0.019) | 0.23 |
| conformer | tiny | within_session | bin | -0.019 [-0.024, -0.014] (n=289, 8 ds, MDE 0.007) | -0.019 [-0.022, -0.015] (n=289, 8 ds, MDE 0.005) | -0.000 [-0.002, +0.001] (n=289, 8 ds, MDE 0.002) | 0.16 |
| conformer | tiny | cross_session | multi | -0.024 [-0.042, -0.006] (n=13, 2 ds, MDE 0.026) | -0.020 [-0.032, -0.010] (n=13, 2 ds, MDE 0.016) | -0.024 [-0.042, -0.006] (n=13, 2 ds, MDE 0.026) | 0.19 |
| conformer | tiny | cross_session | bin | -0.026 [-0.035, -0.016] (n=104, 4 ds, MDE 0.014) | -0.023 [-0.030, -0.015] (n=104, 4 ds, MDE 0.010) | +0.002 [-0.000, +0.005] (n=104, 4 ds, MDE 0.004) | 0.17 |
| conformer | tiny | cross_subject | multi | -0.027 [-0.047, -0.008] (n=35, 4 ds, MDE 0.028) | -0.027 [-0.038, -0.017] (n=35, 4 ds, MDE 0.015) | -0.027 [-0.047, -0.008] (n=35, 4 ds, MDE 0.028) | 0.22 |
| conformer | tiny | cross_subject | bin | -0.025 [-0.031, -0.018] (n=289, 8 ds, MDE 0.010) | -0.039 [-0.046, -0.033] (n=289, 8 ds, MDE 0.009) | -0.013 [-0.017, -0.009] (n=289, 8 ds, MDE 0.006) | 0.22 |
| ctnet | bd | within_session | multi | +0.002 [-0.004, +0.008] (n=35, 4 ds, MDE 0.009) | -0.001 [-0.005, +0.003] (n=35, 4 ds, MDE 0.005) | +0.002 [-0.004, +0.008] (n=35, 4 ds, MDE 0.009) | 0.27 |
| ctnet | bd | within_session | bin | -0.001 [-0.004, +0.001] (n=289, 8 ds, MDE 0.004) | -0.002 [-0.004, -0.000] (n=289, 8 ds, MDE 0.003) | +0.000 [-0.001, +0.002] (n=289, 8 ds, MDE 0.002) | 0.43 |
| ctnet | bd | cross_session | multi | -0.006 [-0.010, -0.003] (n=13, 2 ds, MDE 0.005) | -0.006 [-0.012, -0.000] (n=13, 2 ds, MDE 0.008) | -0.006 [-0.010, -0.003] (n=13, 2 ds, MDE 0.005) | 0.29 |
| ctnet | bd | cross_session | bin | -0.000 [-0.004, +0.003] (n=104, 4 ds, MDE 0.005) | -0.002 [-0.004, +0.001] (n=104, 4 ds, MDE 0.004) | -0.000 [-0.002, +0.001] (n=104, 4 ds, MDE 0.002) | 0.35 |
| ctnet | bd | cross_subject | multi | -0.000 [-0.006, +0.006] (n=35, 4 ds, MDE 0.009) | +0.004 [-0.001, +0.009] (n=35, 4 ds, MDE 0.007) | -0.000 [-0.006, +0.006] (n=35, 4 ds, MDE 0.009) | 0.33 |
| ctnet | bd | cross_subject | bin | -0.000 [-0.003, +0.002] (n=289, 8 ds, MDE 0.003) | +0.002 [+0.000, +0.003] (n=289, 8 ds, MDE 0.002) | +0.001 [-0.000, +0.002] (n=289, 8 ds, MDE 0.001) | 0.88 |
| ctnet | tiny | within_session | multi | +0.003 [-0.001, +0.006] (n=35, 4 ds, MDE 0.005) | +0.001 [-0.002, +0.005] (n=35, 4 ds, MDE 0.004) | +0.003 [-0.001, +0.006] (n=35, 4 ds, MDE 0.005) | 0.44 |
| ctnet | tiny | within_session | bin | -0.002 [-0.005, +0.001] (n=289, 8 ds, MDE 0.004) | -0.002 [-0.003, -0.000] (n=289, 8 ds, MDE 0.002) | -0.001 [-0.002, +0.000] (n=289, 8 ds, MDE 0.002) | 0.52 |
| ctnet | tiny | cross_session | multi | -0.002 [-0.005, +0.001] (n=13, 2 ds, MDE 0.004) | -0.001 [-0.004, +0.003] (n=13, 2 ds, MDE 0.005) | -0.002 [-0.005, +0.001] (n=13, 2 ds, MDE 0.004) | 0.48 |
| ctnet | tiny | cross_session | bin | +0.004 [+0.001, +0.008] (n=104, 4 ds, MDE 0.005) | -0.001 [-0.004, +0.003] (n=104, 4 ds, MDE 0.005) | -0.001 [-0.002, +0.000] (n=104, 4 ds, MDE 0.002) | 0.41 |
| ctnet | tiny | cross_subject | multi | -0.002 [-0.006, +0.003] (n=35, 4 ds, MDE 0.006) | +0.002 [-0.002, +0.006] (n=35, 4 ds, MDE 0.006) | -0.002 [-0.006, +0.003] (n=35, 4 ds, MDE 0.006) | 0.35 |
| ctnet | tiny | cross_subject | bin | -0.001 [-0.003, +0.001] (n=289, 8 ds, MDE 0.003) | +0.001 [-0.000, +0.002] (n=289, 8 ds, MDE 0.001) | +0.000 [-0.000, +0.001] (n=289, 8 ds, MDE 0.001) | 0.99 |
| msvtnet | bd | within_session | multi | -0.007 [-0.010, -0.003] (n=35, 4 ds, MDE 0.006) | -0.005 [-0.010, -0.000] (n=35, 4 ds, MDE 0.007) | -0.007 [-0.010, -0.003] (n=35, 4 ds, MDE 0.006) | 0.16 |
| msvtnet | bd | within_session | bin | +0.001 [-0.001, +0.003] (n=289, 8 ds, MDE 0.003) | +0.001 [-0.000, +0.003] (n=289, 8 ds, MDE 0.002) | +0.002 [-0.000, +0.004] (n=289, 8 ds, MDE 0.003) | 0.12 |
| msvtnet | bd | cross_session | multi | -0.006 [-0.014, +0.002] (n=13, 2 ds, MDE 0.011) | -0.003 [-0.008, +0.003] (n=13, 2 ds, MDE 0.008) | -0.006 [-0.014, +0.002] (n=13, 2 ds, MDE 0.011) | 0.22 |
| msvtnet | bd | cross_session | bin | -0.002 [-0.006, +0.001] (n=104, 4 ds, MDE 0.005) | -0.002 [-0.005, +0.001] (n=104, 4 ds, MDE 0.004) | -0.001 [-0.004, +0.003] (n=104, 4 ds, MDE 0.006) | 0.12 |
| msvtnet | bd | cross_subject | multi | +0.002 [-0.018, +0.025] (n=35, 4 ds, MDE 0.031) | +0.012 [-0.005, +0.032] (n=35, 4 ds, MDE 0.026) | +0.002 [-0.018, +0.025] (n=35, 4 ds, MDE 0.031) | 0.24 |
| msvtnet | bd | cross_subject | bin | +0.006 [-0.001, +0.014] (n=289, 8 ds, MDE 0.011) | +0.009 [+0.004, +0.016] (n=289, 8 ds, MDE 0.008) | -0.011 [-0.016, -0.006] (n=289, 8 ds, MDE 0.007) | 0.09 |
| msvtnet | tiny | within_session | multi | -0.007 [-0.012, -0.003] (n=35, 4 ds, MDE 0.006) | -0.007 [-0.012, -0.001] (n=35, 4 ds, MDE 0.008) | -0.007 [-0.012, -0.003] (n=35, 4 ds, MDE 0.006) | 0.21 |
| msvtnet | tiny | within_session | bin | -0.004 [-0.006, -0.002] (n=289, 8 ds, MDE 0.003) | -0.003 [-0.005, -0.001] (n=289, 8 ds, MDE 0.003) | +0.003 [-0.000, +0.006] (n=289, 8 ds, MDE 0.004) | 0.14 |
| msvtnet | tiny | cross_session | multi | -0.001 [-0.009, +0.007] (n=13, 2 ds, MDE 0.012) | +0.000 [-0.008, +0.008] (n=13, 2 ds, MDE 0.011) | -0.001 [-0.009, +0.007] (n=13, 2 ds, MDE 0.012) | 0.27 |
| msvtnet | tiny | cross_session | bin | -0.002 [-0.006, +0.001] (n=104, 4 ds, MDE 0.005) | -0.003 [-0.006, +0.000] (n=104, 4 ds, MDE 0.004) | +0.003 [-0.001, +0.008] (n=104, 4 ds, MDE 0.006) | 0.13 |
| msvtnet | tiny | cross_subject | multi | +0.001 [-0.018, +0.021] (n=35, 4 ds, MDE 0.028) | +0.007 [-0.010, +0.026] (n=35, 4 ds, MDE 0.025) | +0.001 [-0.018, +0.021] (n=35, 4 ds, MDE 0.028) | 0.24 |
| msvtnet | tiny | cross_subject | bin | -0.001 [-0.006, +0.005] (n=289, 8 ds, MDE 0.008) | +0.001 [-0.004, +0.005] (n=289, 8 ds, MDE 0.007) | -0.007 [-0.012, -0.003] (n=289, 8 ds, MDE 0.007) | 0.14 |

## Par dataset (Conformer bd)

- within_session / multi : alexmi A -0.033 R -0.040 ; bnci2014_001 A -0.049 R -0.050 ; schirrmeister2017 A -0.041 R -0.053 ; zhou2016 A -0.038 R -0.028
- within_session / bin : bnci2014_002 A -0.010 R -0.040 ; bnci2014_004 A -0.022 R -0.022 ; bnci2015_001 A -0.034 R -0.028 ; cho2017 A -0.017 R -0.027 ; lee2019_mi A -0.023 R -0.025 ; physionetmi A -0.007 R -0.014 ; shin2017a A -0.025 R -0.026 ; weibo2014 A -0.005 R -0.023
- cross_session / multi : bnci2014_001 A -0.055 R -0.070 ; zhou2016 A -0.034 R -0.033
- cross_session / bin : bnci2014_004 A -0.011 R -0.023 ; bnci2015_001 A -0.013 R -0.034 ; lee2019_mi A -0.017 R -0.028 ; shin2017a A -0.037 R -0.053
- cross_subject / multi : alexmi A -0.004 R -0.008 ; bnci2014_001 A -0.046 R -0.071 ; schirrmeister2017 A -0.052 R -0.150 ; zhou2016 A -0.044 R -0.060
- cross_subject / bin : bnci2014_002 A -0.048 R -0.065 ; bnci2014_004 A -0.029 R -0.065 ; bnci2015_001 A -0.013 R -0.034 ; cho2017 A -0.028 R -0.081 ; lee2019_mi A -0.016 R -0.139 ; physionetmi A -0.065 R -0.140 ; shin2017a A -0.008 R -0.058 ; weibo2014 A -0.013 R -0.076
