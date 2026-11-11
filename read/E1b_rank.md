# E1b-rank — supprimer les FFN abîme-t-il le classement des essais ? (EXPLORATOIRE)

Fits lus : 18882. Portes : {'fits': 18882, 'groups': 19966, 'g1_fail': 0, 'g2_fail': 0, 'pw_nan': 0, 'excluded:e1b:g0': 2}.

A = acc(z_all) − acc(full) ; K = pw(z_all) − pw(full), pw = AUC par paires de classes sur les écarts de logits (float64).
Verdict (note gelée) : « rang intact » si borne basse de K > -0.01 ; « rang perdu » si borne haute de K < -0.01 ; sinon non concluant.

## Primaire : Conformer bd, datasets à 3-4 classes

- **within_session** : A = -0.041 [-0.051, -0.031] (n=35, 4 ds, MDE 0.015) ; K = -0.001 [-0.007, +0.006] (n=35, 4 ds, MDE 0.009) → **RANG INTACT**
- **cross_session** : A = -0.047 [-0.066, -0.030] (n=13, 2 ds, MDE 0.026) ; K = +0.000 [-0.004, +0.004] (n=13, 2 ds, MDE 0.006) → **RANG INTACT**
- **cross_subject** : A = -0.037 [-0.057, -0.018] (n=35, 4 ds, MDE 0.028) ; K = +0.002 [-0.012, +0.018] (n=35, 4 ds, MDE 0.022) → **NON CONCLUANT**

**Conclusion écrite avant lecture : le coût est un changement d'échelle/forme des logits, pas de classement (≥ 2/3 protocoles) : axe clos.**

## Secondaires (ne décident rien)

| modèle | bras | protocole | classes | A (accuracy) | K (pw) | OvR z − full | pw full médian |
|---|---|---|---|---|---|---|---|
| conformer | bd | within_session | multi | -0.041 [-0.051, -0.031] (n=35, 4 ds, MDE 0.015) | -0.001 [-0.007, +0.006] (n=35, 4 ds, MDE 0.009) | -0.004 [-0.008, +0.001] (n=35, 4 ds, MDE 0.007) | 0.940 |
| conformer | bd | within_session | bin | -0.017 [-0.021, -0.012] (n=289, 8 ds, MDE 0.006) | -0.002 [-0.004, -0.000] (n=289, 8 ds, MDE 0.002) | -0.002 [-0.004, -0.000] (n=289, 8 ds, MDE 0.002) | 0.770 |
| conformer | bd | cross_session | multi | -0.047 [-0.066, -0.030] (n=13, 2 ds, MDE 0.026) | +0.000 [-0.004, +0.004] (n=13, 2 ds, MDE 0.006) | -0.006 [-0.011, -0.002] (n=13, 2 ds, MDE 0.006) | 0.943 |
| conformer | bd | cross_session | bin | -0.021 [-0.030, -0.012] (n=104, 4 ds, MDE 0.013) | +0.002 [-0.001, +0.006] (n=104, 4 ds, MDE 0.005) | +0.002 [-0.001, +0.006] (n=104, 4 ds, MDE 0.005) | 0.770 |
| conformer | bd | cross_subject | multi | -0.037 [-0.057, -0.018] (n=35, 4 ds, MDE 0.028) | +0.002 [-0.012, +0.018] (n=35, 4 ds, MDE 0.022) | -0.009 [-0.021, +0.004] (n=35, 4 ds, MDE 0.018) | 0.881 |
| conformer | bd | cross_subject | bin | -0.032 [-0.040, -0.024] (n=289, 8 ds, MDE 0.012) | -0.019 [-0.025, -0.014] (n=289, 8 ds, MDE 0.008) | -0.019 [-0.025, -0.014] (n=289, 8 ds, MDE 0.008) | 0.801 |
| conformer | tiny | within_session | multi | -0.051 [-0.064, -0.038] (n=35, 4 ds, MDE 0.019) | +0.001 [-0.003, +0.004] (n=35, 4 ds, MDE 0.005) | -0.007 [-0.011, -0.003] (n=35, 4 ds, MDE 0.005) | 0.940 |
| conformer | tiny | within_session | bin | -0.019 [-0.024, -0.014] (n=289, 8 ds, MDE 0.007) | -0.001 [-0.003, +0.000] (n=289, 8 ds, MDE 0.002) | -0.001 [-0.003, +0.000] (n=289, 8 ds, MDE 0.002) | 0.776 |
| conformer | tiny | cross_session | multi | -0.024 [-0.042, -0.006] (n=13, 2 ds, MDE 0.026) | -0.000 [-0.003, +0.002] (n=13, 2 ds, MDE 0.003) | -0.003 [-0.007, +0.000] (n=13, 2 ds, MDE 0.005) | 0.948 |
| conformer | tiny | cross_session | bin | -0.026 [-0.035, -0.016] (n=104, 4 ds, MDE 0.014) | -0.001 [-0.003, +0.001] (n=104, 4 ds, MDE 0.003) | -0.001 [-0.003, +0.001] (n=104, 4 ds, MDE 0.003) | 0.786 |
| conformer | tiny | cross_subject | multi | -0.027 [-0.047, -0.008] (n=35, 4 ds, MDE 0.028) | -0.003 [-0.011, +0.006] (n=35, 4 ds, MDE 0.012) | -0.004 [-0.013, +0.006] (n=35, 4 ds, MDE 0.014) | 0.893 |
| conformer | tiny | cross_subject | bin | -0.025 [-0.031, -0.018] (n=289, 8 ds, MDE 0.010) | -0.013 [-0.017, -0.009] (n=289, 8 ds, MDE 0.006) | -0.013 [-0.017, -0.009] (n=289, 8 ds, MDE 0.006) | 0.793 |
| ctnet | bd | within_session | multi | +0.002 [-0.004, +0.008] (n=35, 4 ds, MDE 0.009) | +0.000 [-0.001, +0.002] (n=35, 4 ds, MDE 0.002) | -0.001 [-0.003, +0.001] (n=35, 4 ds, MDE 0.003) | 0.914 |
| ctnet | bd | within_session | bin | -0.001 [-0.004, +0.001] (n=289, 8 ds, MDE 0.004) | +0.000 [-0.001, +0.002] (n=289, 8 ds, MDE 0.002) | +0.000 [-0.001, +0.002] (n=289, 8 ds, MDE 0.002) | 0.672 |
| ctnet | bd | cross_session | multi | -0.006 [-0.010, -0.003] (n=13, 2 ds, MDE 0.005) | -0.000 [-0.001, +0.001] (n=13, 2 ds, MDE 0.001) | -0.001 [-0.002, -0.000] (n=13, 2 ds, MDE 0.002) | 0.931 |
| ctnet | bd | cross_session | bin | -0.000 [-0.004, +0.003] (n=104, 4 ds, MDE 0.005) | -0.000 [-0.002, +0.001] (n=104, 4 ds, MDE 0.002) | -0.000 [-0.002, +0.001] (n=104, 4 ds, MDE 0.002) | 0.628 |
| ctnet | bd | cross_subject | multi | -0.000 [-0.006, +0.006] (n=35, 4 ds, MDE 0.009) | +0.000 [-0.002, +0.003] (n=35, 4 ds, MDE 0.003) | +0.001 [-0.001, +0.004] (n=35, 4 ds, MDE 0.004) | 0.865 |
| ctnet | bd | cross_subject | bin | -0.000 [-0.003, +0.002] (n=289, 8 ds, MDE 0.003) | +0.001 [-0.000, +0.002] (n=289, 8 ds, MDE 0.001) | +0.001 [-0.000, +0.002] (n=289, 8 ds, MDE 0.001) | 0.777 |
| ctnet | tiny | within_session | multi | +0.003 [-0.001, +0.006] (n=35, 4 ds, MDE 0.005) | +0.002 [+0.000, +0.003] (n=35, 4 ds, MDE 0.002) | +0.001 [-0.001, +0.002] (n=35, 4 ds, MDE 0.002) | 0.917 |
| ctnet | tiny | within_session | bin | -0.002 [-0.005, +0.001] (n=289, 8 ds, MDE 0.004) | -0.001 [-0.002, +0.000] (n=289, 8 ds, MDE 0.002) | -0.001 [-0.002, +0.000] (n=289, 8 ds, MDE 0.002) | 0.660 |
| ctnet | tiny | cross_session | multi | -0.002 [-0.005, +0.001] (n=13, 2 ds, MDE 0.004) | +0.000 [-0.000, +0.001] (n=13, 2 ds, MDE 0.001) | -0.000 [-0.001, +0.000] (n=13, 2 ds, MDE 0.001) | 0.930 |
| ctnet | tiny | cross_session | bin | +0.004 [+0.001, +0.008] (n=104, 4 ds, MDE 0.005) | -0.001 [-0.002, +0.000] (n=104, 4 ds, MDE 0.002) | -0.001 [-0.002, +0.000] (n=104, 4 ds, MDE 0.002) | 0.640 |
| ctnet | tiny | cross_subject | multi | -0.002 [-0.006, +0.003] (n=35, 4 ds, MDE 0.006) | +0.003 [+0.001, +0.005] (n=35, 4 ds, MDE 0.003) | +0.002 [+0.001, +0.004] (n=35, 4 ds, MDE 0.003) | 0.865 |
| ctnet | tiny | cross_subject | bin | -0.001 [-0.003, +0.001] (n=289, 8 ds, MDE 0.003) | +0.000 [-0.000, +0.001] (n=289, 8 ds, MDE 0.001) | +0.000 [-0.000, +0.001] (n=289, 8 ds, MDE 0.001) | 0.780 |
| msvtnet | bd | within_session | multi | -0.007 [-0.010, -0.003] (n=35, 4 ds, MDE 0.006) | -0.001 [-0.004, +0.001] (n=35, 4 ds, MDE 0.003) | -0.001 [-0.003, +0.001] (n=35, 4 ds, MDE 0.003) | 0.947 |
| msvtnet | bd | within_session | bin | +0.001 [-0.001, +0.003] (n=289, 8 ds, MDE 0.003) | +0.002 [-0.000, +0.004] (n=289, 8 ds, MDE 0.003) | +0.002 [-0.000, +0.004] (n=289, 8 ds, MDE 0.003) | 0.750 |
| msvtnet | bd | cross_session | multi | -0.006 [-0.014, +0.002] (n=13, 2 ds, MDE 0.011) | -0.001 [-0.002, +0.000] (n=13, 2 ds, MDE 0.002) | -0.002 [-0.004, -0.001] (n=13, 2 ds, MDE 0.002) | 0.947 |
| msvtnet | bd | cross_session | bin | -0.002 [-0.006, +0.001] (n=104, 4 ds, MDE 0.005) | -0.001 [-0.004, +0.003] (n=104, 4 ds, MDE 0.006) | -0.001 [-0.004, +0.003] (n=104, 4 ds, MDE 0.006) | 0.719 |
| msvtnet | bd | cross_subject | multi | +0.002 [-0.018, +0.025] (n=35, 4 ds, MDE 0.031) | +0.016 [+0.003, +0.031] (n=35, 4 ds, MDE 0.019) | +0.007 [-0.006, +0.021] (n=35, 4 ds, MDE 0.019) | 0.890 |
| msvtnet | bd | cross_subject | bin | +0.006 [-0.001, +0.014] (n=289, 8 ds, MDE 0.011) | -0.011 [-0.016, -0.006] (n=289, 8 ds, MDE 0.007) | -0.011 [-0.016, -0.006] (n=289, 8 ds, MDE 0.007) | 0.806 |
| msvtnet | tiny | within_session | multi | -0.007 [-0.012, -0.003] (n=35, 4 ds, MDE 0.006) | +0.002 [-0.000, +0.004] (n=35, 4 ds, MDE 0.003) | -0.001 [-0.002, +0.001] (n=35, 4 ds, MDE 0.002) | 0.950 |
| msvtnet | tiny | within_session | bin | -0.004 [-0.006, -0.002] (n=289, 8 ds, MDE 0.003) | +0.003 [-0.000, +0.006] (n=289, 8 ds, MDE 0.004) | +0.003 [-0.000, +0.006] (n=289, 8 ds, MDE 0.004) | 0.750 |
| msvtnet | tiny | cross_session | multi | -0.001 [-0.009, +0.007] (n=13, 2 ds, MDE 0.012) | -0.002 [-0.004, -0.001] (n=13, 2 ds, MDE 0.002) | -0.004 [-0.006, -0.002] (n=13, 2 ds, MDE 0.003) | 0.942 |
| msvtnet | tiny | cross_session | bin | -0.002 [-0.006, +0.001] (n=104, 4 ds, MDE 0.005) | +0.003 [-0.001, +0.008] (n=104, 4 ds, MDE 0.006) | +0.003 [-0.001, +0.008] (n=104, 4 ds, MDE 0.006) | 0.750 |
| msvtnet | tiny | cross_subject | multi | +0.001 [-0.018, +0.021] (n=35, 4 ds, MDE 0.028) | +0.007 [-0.002, +0.018] (n=35, 4 ds, MDE 0.014) | +0.006 [-0.006, +0.019] (n=35, 4 ds, MDE 0.017) | 0.887 |
| msvtnet | tiny | cross_subject | bin | -0.001 [-0.006, +0.005] (n=289, 8 ds, MDE 0.008) | -0.007 [-0.012, -0.003] (n=289, 8 ds, MDE 0.007) | -0.007 [-0.012, -0.003] (n=289, 8 ds, MDE 0.007) | 0.812 |

## Par dataset (Conformer bd)

- within_session / multi : alexmi A -0.033 K -0.002 ; bnci2014_001 A -0.049 K +0.000 ; schirrmeister2017 A -0.041 K -0.001 ; zhou2016 A -0.038 K +0.001
- within_session / bin : bnci2014_002 A -0.010 K -0.000 ; bnci2014_004 A -0.022 K +0.002 ; bnci2015_001 A -0.034 K -0.001 ; cho2017 A -0.017 K -0.003 ; lee2019_mi A -0.023 K -0.002 ; physionetmi A -0.007 K -0.002 ; shin2017a A -0.025 K -0.004 ; weibo2014 A -0.005 K -0.002
- cross_session / multi : bnci2014_001 A -0.055 K -0.002 ; zhou2016 A -0.034 K +0.003
- cross_session / bin : bnci2014_004 A -0.011 K +0.006 ; bnci2015_001 A -0.013 K -0.006 ; lee2019_mi A -0.017 K +0.000 ; shin2017a A -0.037 K +0.008
- cross_subject / multi : alexmi A -0.004 K +0.041 ; bnci2014_001 A -0.046 K -0.010 ; schirrmeister2017 A -0.052 K -0.017 ; zhou2016 A -0.044 K +0.001
- cross_subject / bin : bnci2014_002 A -0.048 K -0.006 ; bnci2014_004 A -0.029 K -0.021 ; bnci2015_001 A -0.013 K +0.004 ; cho2017 A -0.028 K -0.025 ; lee2019_mi A -0.016 K -0.012 ; physionetmi A -0.065 K -0.054 ; shin2017a A -0.008 K +0.013 ; weibo2014 A -0.013 K -0.004
