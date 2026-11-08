# Campagne A — lecture (PREREG_campaign_A.md § 3, PREREG_shallow_init_control.md § 3)

Score = dernière époque. Δ = moyenne par sujet ; IC95 bootstrap sur les sujets. NI : borne basse > −0.02. Case hors comptes si non lisible ou incomplète.

## Q1 — tiny − bd

| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict | même famille de carte |
|---|---|---|---|---|---|---|---|
| within_session | shallow | alexmi | 8 | +0.048 [-0.000, +0.100] (2/8) | 0.078 | NON INFÉRIEUR | 8 : NON INFÉRIEUR |
| within_session | shallow | bnci2014_001 | 9 | +0.035 [+0.023, +0.047] (0/9) | 0.018 | SUPÉRIEUR | 0 : — |
| within_session | shallow | bnci2014_002 | 14 | +0.029 [+0.008, +0.054] (1/14) | 0.034 | SUPÉRIEUR | 14 : SUPÉRIEUR |
| within_session | shallow | bnci2014_004 | 9 | -0.003 [-0.009, +0.005] (6/9) | 0.011 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| within_session | shallow | bnci2015_001 | 12 | +0.025 [+0.016, +0.033] (0/12) | 0.013 | SUPÉRIEUR | 12 : SUPÉRIEUR |
| within_session | shallow | cho2017 | 52 | +0.027 [+0.017, +0.038] (11/52) | 0.015 | SUPÉRIEUR | 0 : — |
| within_session | shallow | lee2019_mi | 54 | +0.057 [+0.043, +0.071] (6/54) | 0.020 | SUPÉRIEUR | 0 : — |
| within_session | shallow | physionetmi | 109 | +0.035 [+0.019, +0.050] (35/109) | 0.022 | SUPÉRIEUR | 0 : — |
| within_session | shallow | schirrmeister2017 | 14 | +0.014 [+0.005, +0.021] (2/14) | 0.012 | SUPÉRIEUR | 6 : SUPÉRIEUR |
| within_session | shallow | shin2017a | 29 | +0.022 [-0.001, +0.044] (9/29) | 0.032 | NON INFÉRIEUR | 0 : — |
| within_session | shallow | weibo2014 | 10 | +0.016 [-0.001, +0.032] (2/10) | 0.025 | NON INFÉRIEUR | 10 : NON INFÉRIEUR |
| within_session | shallow | zhou2016 | 4 | +0.036 [+0.016, +0.051] (0/4) | 0.028 | SUPÉRIEUR | 0 : — |
| within_session | eegnex | alexmi | 8 | +0.004 [-0.035, +0.048] (5/8) | 0.064 | NON CONCLUANT | 8 : NON CONCLUANT |
| within_session | eegnex | bnci2014_001 | 9 | +0.006 [-0.011, +0.023] (3/9) | 0.026 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| within_session | eegnex | bnci2014_002 | 14 | -0.001 [-0.010, +0.010] (6/14) | 0.015 | NON INFÉRIEUR | 14 : NON INFÉRIEUR |
| within_session | eegnex | bnci2014_004 | 9 | +0.005 [-0.003, +0.015] (4/9) | 0.014 | NON INFÉRIEUR | 0 : — |
| within_session | eegnex | bnci2015_001 | 12 | +0.002 [-0.003, +0.009] (4/12) | 0.009 | NON INFÉRIEUR | 0 : — |
| within_session | eegnex | cho2017 | 52 | -0.001 [-0.013, +0.011] (23/52) | 0.018 | NON INFÉRIEUR | 31 : NON INFÉRIEUR |
| within_session | eegnex | lee2019_mi | 54 | -0.012 [-0.021, -0.002] (34/54) | 0.014 | INFÉRIEUR | 8 : NON CONCLUANT |
| within_session | eegnex | physionetmi | 109 | +0.004 [-0.012, +0.020] (47/109) | 0.022 | NON INFÉRIEUR | 0 : — |
| within_session | eegnex | schirrmeister2017 | 14 | +0.011 [+0.003, +0.020] (4/14) | 0.013 | SUPÉRIEUR | 6 : SUPÉRIEUR |
| within_session | eegnex | shin2017a | 29 | -0.013 [-0.047, +0.017] (9/29) | 0.047 | NON CONCLUANT | 0 : — |
| within_session | eegnex | weibo2014 | 10 | -0.044 [-0.081, -0.014] (8/10) | 0.051 | INFÉRIEUR | 10 : INFÉRIEUR |
| within_session | eegnex | zhou2016 | 4 | -0.018 [-0.027, -0.004] (3/4) | 0.020 | INFÉRIEUR | 4 : INFÉRIEUR |
| within_session | msvtnet | alexmi | 8 | -0.008 [-0.058, +0.037] (4/8) | 0.073 | NON CONCLUANT | 8 : NON CONCLUANT |
| within_session | msvtnet | bnci2014_001 | 9 | +0.003 [-0.013, +0.019] (5/9) | 0.025 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| within_session | msvtnet | bnci2014_002 | 14 | +0.007 [-0.006, +0.018] (4/14) | 0.018 | NON INFÉRIEUR | 14 : NON INFÉRIEUR |
| within_session | msvtnet | bnci2014_004 | 9 | +0.005 [-0.001, +0.010] (4/9) | 0.008 | NON INFÉRIEUR | 0 : — |
| within_session | msvtnet | bnci2015_001 | 12 | +0.001 [-0.005, +0.009] (7/12) | 0.010 | NON INFÉRIEUR | 0 : — |
| within_session | msvtnet | cho2017 | 52 | +0.002 [-0.006, +0.010] (21/52) | 0.012 | NON INFÉRIEUR | 8 : NON INFÉRIEUR |
| within_session | msvtnet | lee2019_mi | 54 | +0.013 [+0.004, +0.021] (21/54) | 0.012 | SUPÉRIEUR | 23 : NON INFÉRIEUR |
| within_session | msvtnet | physionetmi | 109 | +0.004 [-0.011, +0.019] (52/109) | 0.022 | NON INFÉRIEUR | 0 : — |
| within_session | msvtnet | schirrmeister2017 | 14 | +0.005 [-0.002, +0.011] (4/14) | 0.010 | NON INFÉRIEUR | 6 : NON INFÉRIEUR |
| within_session | msvtnet | shin2017a | 29 | -0.016 [-0.042, +0.011] (15/29) | 0.039 | NON CONCLUANT | 29 : NON CONCLUANT |
| within_session | msvtnet | weibo2014 | 10 | +0.005 [-0.012, +0.023] (5/10) | 0.026 | NON INFÉRIEUR | 0 : — |
| within_session | msvtnet | zhou2016 | 4 | +0.005 [-0.001, +0.011] (2/4) | 0.010 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | alexmi | 8 | -0.013 [-0.058, +0.031] (3/8) | 0.069 | NON CONCLUANT | 8 : NON CONCLUANT |
| within_session | ctnet | bnci2014_001 | 9 | +0.009 [-0.006, +0.026] (6/9) | 0.025 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | bnci2014_002 | 14 | +0.008 [+0.000, +0.016] (4/14) | 0.011 | SUPÉRIEUR | 14 : SUPÉRIEUR |
| within_session | ctnet | bnci2014_004 | 9 | +0.002 [-0.007, +0.013] (4/9) | 0.015 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| within_session | ctnet | bnci2015_001 | 12 | +0.004 [-0.001, +0.010] (6/12) | 0.008 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | cho2017 | 52 | -0.007 [-0.018, +0.004] (34/52) | 0.016 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | lee2019_mi | 54 | -0.003 [-0.013, +0.007] (32/54) | 0.014 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | physionetmi | 109 | -0.004 [-0.018, +0.009] (50/109) | 0.020 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | schirrmeister2017 | 14 | -0.008 [-0.017, +0.001] (10/14) | 0.013 | NON INFÉRIEUR | 2 : INFÉRIEUR |
| within_session | ctnet | shin2017a | 29 | -0.011 [-0.035, +0.013] (16/29) | 0.035 | NON CONCLUANT | 0 : — |
| within_session | ctnet | weibo2014 | 10 | +0.008 [-0.014, +0.030] (4/10) | 0.034 | NON INFÉRIEUR | 0 : — |
| within_session | ctnet | zhou2016 | 4 | +0.000 [-0.013, +0.012] (1/4) | 0.020 | NON INFÉRIEUR | 0 : — |
| within_session | conformer | alexmi | 8 | -0.000 [-0.060, +0.056] (4/8) | 0.089 | NON CONCLUANT | 8 : NON CONCLUANT |
| within_session | conformer | bnci2014_001 | 9 | -0.010 [-0.027, +0.007] (7/9) | 0.026 | NON CONCLUANT | 0 : — |
| within_session | conformer | bnci2014_002 | 14 | +0.007 [-0.009, +0.024] (5/14) | 0.024 | NON INFÉRIEUR | 0 : — |
| within_session | conformer | bnci2014_004 | 9 | +0.002 [-0.002, +0.007] (3/9) | 0.007 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| within_session | conformer | bnci2015_001 | 12 | -0.001 [-0.005, +0.003] (6/12) | 0.005 | NON INFÉRIEUR | 1 : — |
| within_session | conformer | cho2017 | 52 | -0.005 [-0.012, +0.002] (32/52) | 0.010 | NON INFÉRIEUR | 11 : NON INFÉRIEUR |
| within_session | conformer | lee2019_mi | 54 | +0.003 [-0.002, +0.009] (24/54) | 0.009 | NON INFÉRIEUR | 10 : NON INFÉRIEUR |
| within_session | conformer | physionetmi | 109 | -0.002 [-0.014, +0.010] (58/109) | 0.017 | NON INFÉRIEUR | 109 : NON INFÉRIEUR |
| within_session | conformer | schirrmeister2017 | 14 | -0.002 [-0.014, +0.011] (8/14) | 0.019 | NON INFÉRIEUR | 1 : — |
| within_session | conformer | shin2017a | 29 | +0.003 [-0.013, +0.018] (11/29) | 0.023 | NON INFÉRIEUR | 29 : NON INFÉRIEUR |
| within_session | conformer | weibo2014 | 10 | +0.011 [+0.000, +0.020] (2/10) | 0.015 | SUPÉRIEUR | 10 : SUPÉRIEUR |
| within_session | conformer | zhou2016 | 4 | -0.001 [-0.012, +0.015] (3/4) | 0.023 | NON INFÉRIEUR | 4 : NON INFÉRIEUR |
| cross_session | shallow | bnci2014_001 | 9 | +0.022 [+0.012, +0.032] (1/9) | 0.016 | SUPÉRIEUR | 9 : SUPÉRIEUR |
| cross_session | shallow | bnci2014_004 | 9 | -0.003 [-0.009, +0.003] (6/9) | 0.009 | NON INFÉRIEUR | 0 : — |
| cross_session | shallow | bnci2015_001 | 12 | +0.014 [+0.003, +0.026] (3/12) | 0.017 | SUPÉRIEUR | 12 : SUPÉRIEUR |
| cross_session | shallow | lee2019_mi | 54 | +0.036 [+0.024, +0.048] (15/54) | 0.017 | SUPÉRIEUR | 54 : SUPÉRIEUR |
| cross_session | shallow | shin2017a | 29 | +0.055 [+0.024, +0.091] (8/29) | 0.049 | SUPÉRIEUR | 29 : SUPÉRIEUR |
| cross_session | shallow | zhou2016 | 4 | +0.007 [-0.005, +0.019] (1/4) | 0.021 | NON INFÉRIEUR | 4 : NON INFÉRIEUR |
| cross_session | eegnex | bnci2014_001 | 9 | +0.008 [-0.010, +0.026] (3/9) | 0.028 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| cross_session | eegnex | bnci2014_004 | 9 | -0.003 [-0.010, +0.003] (4/9) | 0.010 | NON INFÉRIEUR | 0 : — |
| cross_session | eegnex | bnci2015_001 | 12 | +0.009 [-0.001, +0.019] (3/12) | 0.015 | NON INFÉRIEUR | 12 : NON INFÉRIEUR |
| cross_session | eegnex | lee2019_mi | 54 | +0.006 [-0.005, +0.017] (25/54) | 0.016 | NON INFÉRIEUR | 0 : — |
| cross_session | eegnex | shin2017a | 29 | +0.002 [-0.020, +0.023] (11/29) | 0.032 | NON CONCLUANT | 29 : NON CONCLUANT |
| cross_session | eegnex | zhou2016 | 4 | +0.021 [-0.003, +0.058] (1/4) | 0.051 | NON INFÉRIEUR | 0 : — |
| cross_session | msvtnet | bnci2014_001 | 9 | -0.017 [-0.035, -0.000] (6/9) | 0.027 | INFÉRIEUR | 9 : INFÉRIEUR |
| cross_session | msvtnet | bnci2014_004 | 9 | -0.005 [-0.012, +0.000] (5/9) | 0.009 | NON INFÉRIEUR | 0 : — |
| cross_session | msvtnet | bnci2015_001 | 12 | -0.012 [-0.023, -0.002] (8/12) | 0.016 | INFÉRIEUR | 12 : INFÉRIEUR |
| cross_session | msvtnet | lee2019_mi | 54 | +0.016 [+0.004, +0.029] (21/54) | 0.018 | SUPÉRIEUR | 54 : SUPÉRIEUR |
| cross_session | msvtnet | shin2017a | 29 | +0.030 [+0.006, +0.054] (9/29) | 0.035 | SUPÉRIEUR | 29 : SUPÉRIEUR |
| cross_session | msvtnet | zhou2016 | 4 | +0.001 [-0.010, +0.012] (2/4) | 0.020 | NON INFÉRIEUR | 0 : — |
| cross_session | ctnet | bnci2014_001 | 9 | +0.008 [-0.008, +0.027] (4/9) | 0.026 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| cross_session | ctnet | bnci2014_004 | 9 | +0.005 [-0.002, +0.013] (3/9) | 0.012 | NON INFÉRIEUR | 0 : — |
| cross_session | ctnet | bnci2015_001 | 12 | +0.001 [-0.010, +0.013] (6/12) | 0.017 | NON INFÉRIEUR | 12 : NON INFÉRIEUR |
| cross_session | ctnet | lee2019_mi | 54 | +0.012 [+0.001, +0.024] (19/54) | 0.016 | SUPÉRIEUR | 0 : — |
| cross_session | ctnet | shin2017a | 29 | -0.025 [-0.049, -0.001] (19/29) | 0.035 | INFÉRIEUR | 29 : INFÉRIEUR |
| cross_session | ctnet | zhou2016 | 4 | +0.003 [-0.010, +0.017] (2/4) | 0.024 | NON INFÉRIEUR | 0 : — |
| cross_session | conformer | bnci2014_001 | 9 | -0.003 [-0.031, +0.026] (5/9) | 0.043 | NON CONCLUANT | 9 : NON CONCLUANT |
| cross_session | conformer | bnci2014_004 | 9 | -0.001 [-0.006, +0.006] (6/9) | 0.009 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| cross_session | conformer | bnci2015_001 | 12 | +0.002 [-0.007, +0.007] (2/12) | 0.011 | NON INFÉRIEUR | 12 : NON INFÉRIEUR |
| cross_session | conformer | lee2019_mi | 54 | -0.002 [-0.009, +0.005] (26/54) | 0.011 | NON INFÉRIEUR | 0 : — |
| cross_session | conformer | shin2017a | 29 | +0.032 [+0.018, +0.047] (5/29) | 0.021 | SUPÉRIEUR | 29 : SUPÉRIEUR |
| cross_session | conformer | zhou2016 | 4 | +0.015 [+0.003, +0.027] (0/4) | 0.019 | SUPÉRIEUR | 4 : SUPÉRIEUR |
| cross_subject | shallow | alexmi | 8 | -0.010 [-0.035, +0.010] (4/8) | 0.036 | NON CONCLUANT | 8 : NON CONCLUANT |
| cross_subject | shallow | bnci2014_001 | 9 | +0.002 [-0.017, +0.021] (5/9) | 0.029 | NON INFÉRIEUR | 0 : — |
| cross_subject | shallow | bnci2014_002 | 14 | +0.001 [-0.020, +0.021] (8/14) | 0.031 | NON INFÉRIEUR | 0 : — |
| cross_subject | shallow | bnci2014_004 | 9 | +0.002 [-0.002, +0.006] (4/9) | 0.006 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| cross_subject | shallow | bnci2015_001 | 12 | +0.001 [-0.014, +0.017] (8/12) | 0.024 | NON INFÉRIEUR | 12 : NON INFÉRIEUR |
| cross_subject | shallow | cho2017 | 52 | +0.009 [-0.000, +0.017] (22/52) | 0.012 | NON INFÉRIEUR | 0 : — |
| cross_subject | shallow | lee2019_mi | 54 | -0.002 [-0.008, +0.005] (25/54) | 0.009 | NON INFÉRIEUR | 15 : NON INFÉRIEUR |
| cross_subject | shallow | physionetmi | 109 | -0.004 [-0.011, +0.004] (53/109) | 0.011 | NON INFÉRIEUR | 60 : NON INFÉRIEUR |
| cross_subject | shallow | schirrmeister2017 | 14 | -0.024 [-0.045, -0.004] (11/14) | 0.032 | INFÉRIEUR | 3 : NON CONCLUANT |
| cross_subject | shallow | shin2017a | 29 | +0.009 [-0.009, +0.028] (11/29) | 0.027 | NON INFÉRIEUR | 0 : — |
| cross_subject | shallow | weibo2014 | 10 | +0.029 [+0.010, +0.052] (2/10) | 0.032 | SUPÉRIEUR | 0 : — |
| cross_subject | shallow | zhou2016 | 4 | +0.018 [+0.005, +0.035] (0/4) | 0.026 | SUPÉRIEUR | 4 : SUPÉRIEUR |
| cross_subject | eegnex | alexmi | 8 | +0.021 [-0.029, +0.071] (4/8) | 0.075 | NON CONCLUANT | 8 : NON CONCLUANT |
| cross_subject | eegnex | bnci2014_001 | 9 | -0.034 [-0.082, -0.001] (6/9) | 0.065 | INFÉRIEUR | 9 : INFÉRIEUR |
| cross_subject | eegnex | bnci2014_002 | 14 | +0.007 [-0.015, +0.026] (4/14) | 0.031 | NON INFÉRIEUR | 0 : — |
| cross_subject | eegnex | bnci2014_004 | 9 | +0.006 [-0.000, +0.014] (4/9) | 0.011 | NON INFÉRIEUR | 0 : — |
| cross_subject | eegnex | bnci2015_001 | 12 | -0.003 [-0.017, +0.010] (7/12) | 0.020 | NON INFÉRIEUR | 0 : — |
| cross_subject | eegnex | cho2017 | 52 | +0.008 [-0.001, +0.018] (19/52) | 0.013 | NON INFÉRIEUR | 12 : NON INFÉRIEUR |
| cross_subject | eegnex | lee2019_mi | 54 | +0.001 [-0.006, +0.007] (22/54) | 0.009 | NON INFÉRIEUR | 0 : — |
| cross_subject | eegnex | physionetmi | 109 | +0.007 [-0.003, +0.017] (51/109) | 0.014 | NON INFÉRIEUR | 32 : NON INFÉRIEUR |
| cross_subject | eegnex | schirrmeister2017 | 14 | +0.005 [-0.024, +0.037] (8/14) | 0.045 | NON CONCLUANT | 3 : NON CONCLUANT |
| cross_subject | eegnex | shin2017a | 29 | -0.003 [-0.023, +0.016] (12/29) | 0.029 | NON CONCLUANT | 6 : NON INFÉRIEUR |
| cross_subject | eegnex | weibo2014 | 10 | -0.007 [-0.029, +0.014] (6/10) | 0.033 | NON CONCLUANT | 0 : — |
| cross_subject | eegnex | zhou2016 | 4 | -0.008 [-0.031, +0.009] (2/4) | 0.034 | NON CONCLUANT | 0 : — |
| cross_subject | msvtnet | alexmi | 8 | +0.031 [-0.004, +0.069] (2/8) | 0.056 | NON INFÉRIEUR | 0 : — |
| cross_subject | msvtnet | bnci2014_001 | 9 | +0.003 [-0.011, +0.017] (4/9) | 0.021 | NON INFÉRIEUR | 0 : — |
| cross_subject | msvtnet | bnci2014_002 | 14 | -0.014 [-0.027, -0.001] (10/14) | 0.020 | INFÉRIEUR | 0 : — |
| cross_subject | msvtnet | bnci2014_004 | 9 | +0.003 [-0.010, +0.017] (7/9) | 0.020 | NON INFÉRIEUR | 9 : NON INFÉRIEUR |
| cross_subject | msvtnet | bnci2015_001 | 12 | -0.022 [-0.069, +0.030] (6/12) | 0.074 | NON CONCLUANT | 11 : NON CONCLUANT |
| cross_subject | msvtnet | cho2017 | 52 | +0.003 [-0.006, +0.014] (30/52) | 0.014 | NON INFÉRIEUR | 0 : — |
| cross_subject | msvtnet | lee2019_mi | 54 | +0.002 [-0.006, +0.010] (23/54) | 0.012 | NON INFÉRIEUR | 28 : NON INFÉRIEUR |
| cross_subject | msvtnet | physionetmi | 109 | +0.003 [-0.006, +0.011] (48/109) | 0.013 | NON INFÉRIEUR | 37 : NON INFÉRIEUR |
| cross_subject | msvtnet | schirrmeister2017 | 14 | -0.012 [-0.037, +0.019] (9/14) | 0.042 | NON CONCLUANT | 4 : NON CONCLUANT |
| cross_subject | msvtnet | shin2017a | 29 | -0.000 [-0.018, +0.019] (17/29) | 0.027 | NON INFÉRIEUR | 10 : NON CONCLUANT |
| cross_subject | msvtnet | weibo2014 | 10 | +0.008 [-0.019, +0.037] (3/10) | 0.042 | NON INFÉRIEUR | 10 : NON INFÉRIEUR |
| cross_subject | msvtnet | zhou2016 | 4 | -0.009 [-0.049, +0.024] (2/4) | 0.064 | NON CONCLUANT | 0 : — |
| cross_subject | ctnet | alexmi | 8 | -0.025 [-0.058, +0.010] (5/8) | 0.054 | NON CONCLUANT | 0 : — |
| cross_subject | ctnet | bnci2014_001 | 9 | -0.007 [-0.025, +0.012] (6/9) | 0.028 | NON CONCLUANT | 9 : NON CONCLUANT |
| cross_subject | ctnet | bnci2014_002 | 14 | -0.002 [-0.018, +0.015] (7/14) | 0.025 | NON INFÉRIEUR | 0 : — |
| cross_subject | ctnet | bnci2014_004 | 9 | -0.010 [-0.022, +0.000] (5/9) | 0.017 | NON CONCLUANT | 0 : — |
| cross_subject | ctnet | bnci2015_001 | 12 | -0.004 [-0.018, +0.010] (9/12) | 0.020 | NON INFÉRIEUR | 0 : — |
| cross_subject | ctnet | cho2017 | 52 | -0.006 [-0.016, +0.004] (30/52) | 0.014 | NON INFÉRIEUR | 33 : NON INFÉRIEUR |
| cross_subject | ctnet | lee2019_mi | 54 | +0.002 [-0.004, +0.009] (28/54) | 0.009 | NON INFÉRIEUR | 32 : NON INFÉRIEUR |
| cross_subject | ctnet | physionetmi | 109 | -0.002 [-0.013, +0.010] (49/109) | 0.016 | NON INFÉRIEUR | 20 : NON CONCLUANT |
| cross_subject | ctnet | schirrmeister2017 | 14 | +0.009 [-0.005, +0.023] (4/14) | 0.021 | NON INFÉRIEUR | 2 : SUPÉRIEUR |
| cross_subject | ctnet | shin2017a | 29 | -0.001 [-0.026, +0.025] (14/29) | 0.037 | NON CONCLUANT | 0 : — |
| cross_subject | ctnet | weibo2014 | 10 | +0.008 [-0.009, +0.024] (3/10) | 0.025 | NON INFÉRIEUR | 0 : — |
| cross_subject | ctnet | zhou2016 | 4 | +0.012 [-0.012, +0.036] (2/4) | 0.042 | NON INFÉRIEUR | 0 : — |
| cross_subject | conformer | alexmi | 8 | -0.002 [-0.037, +0.033] (2/8) | 0.056 | NON CONCLUANT | 8 : NON CONCLUANT |
| cross_subject | conformer | bnci2014_001 | 9 | +0.016 [-0.015, +0.051] (3/9) | 0.049 | NON INFÉRIEUR | 2 : NON CONCLUANT |
| cross_subject | conformer | bnci2014_002 | 14 | +0.001 [-0.009, +0.012] (8/14) | 0.016 | NON INFÉRIEUR | 0 : — |
| cross_subject | conformer | bnci2014_004 | 9 | -0.006 [-0.015, +0.003] (6/9) | 0.014 | NON INFÉRIEUR | 0 : — |
| cross_subject | conformer | bnci2015_001 | 12 | +0.057 [+0.009, +0.117] (4/12) | 0.082 | SUPÉRIEUR | 2 : NON CONCLUANT |
| cross_subject | conformer | cho2017 | 52 | +0.003 [-0.006, +0.011] (21/52) | 0.013 | NON INFÉRIEUR | 39 : NON INFÉRIEUR |
| cross_subject | conformer | lee2019_mi | 54 | +0.002 [-0.004, +0.010] (25/54) | 0.010 | NON INFÉRIEUR | 29 : NON INFÉRIEUR |
| cross_subject | conformer | physionetmi | 109 | -0.021 [-0.030, -0.012] (74/109) | 0.013 | INFÉRIEUR | 25 : INFÉRIEUR |
| cross_subject | conformer | schirrmeister2017 | 14 | -0.018 [-0.056, +0.013] (8/14) | 0.052 | NON CONCLUANT | 6 : NON CONCLUANT |
| cross_subject | conformer | shin2017a | 29 | +0.000 [-0.018, +0.018] (14/29) | 0.027 | NON INFÉRIEUR | 0 : — |
| cross_subject | conformer | weibo2014 | 10 | +0.002 [-0.011, +0.016] (5/10) | 0.021 | NON INFÉRIEUR | 0 : — |
| cross_subject | conformer | zhou2016 | 4 | -0.015 [-0.043, +0.012] (2/4) | 0.045 | NON CONCLUANT | 0 : — |

### Q1 groupé (Holm sur les 15 endpoints lisibles, α = 0.025 unilatéral)

| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |
|---|---|---|---|---|---|---|---|
| within_session | shallow | 12 | +0.028 [+0.023, +0.034] | 0.0000 | 0.0000 | 0.008 | SUPÉRIEUR |
| within_session | eegnex | 12 | -0.005 [-0.011, +0.001] | 0.9358 | 0.0000 | 0.009 | NON INFÉRIEUR |
| within_session | tidnet | 0 | — | — | — | — | AUCUN DATASET |
| within_session | msvtnet | 12 | +0.002 [-0.003, +0.007] | 0.2179 | 0.0000 | 0.008 | NON INFÉRIEUR |
| within_session | ctnet | 12 | -0.001 [-0.007, +0.004] | 0.6630 | 0.0000 | 0.008 | NON INFÉRIEUR |
| within_session | conformer | 12 | +0.000 [-0.006, +0.006] | 0.4403 | 0.0000 | 0.008 | NON INFÉRIEUR |
| cross_session | shallow | 6 | +0.022 [+0.015, +0.029] | 0.0000 | 0.0000 | 0.010 | SUPÉRIEUR |
| cross_session | eegnex | 6 | +0.007 [-0.000, +0.015] | 0.0271 | 0.0000 | 0.011 | NON INFÉRIEUR |
| cross_session | tidnet | 0 | — | — | — | — | AUCUN DATASET |
| cross_session | msvtnet | 6 | +0.002 [-0.004, +0.008] | 0.2486 | 0.0000 | 0.009 | NON INFÉRIEUR |
| cross_session | ctnet | 6 | +0.001 [-0.005, +0.007] | 0.4088 | 0.0000 | 0.009 | NON INFÉRIEUR |
| cross_session | conformer | 6 | +0.007 [+0.001, +0.013] | 0.0094 | 0.0000 | 0.009 | NON INFÉRIEUR |
| cross_subject | shallow | 12 | +0.003 [-0.002, +0.007] | 0.1429 | 0.0000 | 0.007 | NON INFÉRIEUR |
| cross_subject | eegnex | 12 | +0.000 [-0.007, +0.007] | 0.4913 | 0.0000 | 0.010 | NON INFÉRIEUR |
| cross_subject | tidnet | 0 | — | — | — | — | AUCUN DATASET |
| cross_subject | msvtnet | 12 | -0.000 [-0.008, +0.007] | 0.5374 | 0.0000 | 0.011 | NON INFÉRIEUR |
| cross_subject | ctnet | 12 | -0.002 [-0.008, +0.003] | 0.7768 | 0.0000 | 0.008 | NON INFÉRIEUR |
| cross_subject | conformer | 12 | +0.002 [-0.006, +0.009] | 0.3517 | 0.0000 | 0.011 | NON INFÉRIEUR |

## Q2 — tiny − random

| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict |
|---|---|---|---|---|---|---|
| within_session | shallow | alexmi | 8 | +0.013 [-0.015, +0.040] (3/8) | 0.042 | NON INFÉRIEUR |
| within_session | shallow | bnci2014_001 | 9 | +0.007 [-0.005, +0.018] (3/9) | 0.017 | NON INFÉRIEUR |
| within_session | shallow | bnci2014_002 | 14 | -0.005 [-0.019, +0.007] (8/14) | 0.020 | NON INFÉRIEUR |
| within_session | shallow | bnci2014_004 | 9 | -0.002 [-0.006, +0.002] (6/9) | 0.006 | NON INFÉRIEUR |
| within_session | shallow | bnci2015_001 | 12 | +0.001 [-0.003, +0.005] (6/12) | 0.006 | NON INFÉRIEUR |
| within_session | shallow | cho2017 | 52 | -0.002 [-0.008, +0.004] (27/52) | 0.008 | NON INFÉRIEUR |
| within_session | shallow | lee2019_mi | 54 | +0.003 [-0.004, +0.010] (21/54) | 0.010 | NON INFÉRIEUR |
| within_session | shallow | physionetmi | 109 | -0.000 [-0.012, +0.011] (53/109) | 0.016 | NON INFÉRIEUR |
| within_session | shallow | schirrmeister2017 | 14 | -0.000 [-0.007, +0.007] (6/14) | 0.011 | NON INFÉRIEUR |
| within_session | shallow | shin2017a | 29 | -0.002 [-0.017, +0.014] (15/29) | 0.022 | NON INFÉRIEUR |
| within_session | shallow | weibo2014 | 10 | -0.010 [-0.020, +0.001] (7/10) | 0.016 | NON INFÉRIEUR |
| within_session | shallow | zhou2016 | 4 | -0.017 [-0.030, -0.005] (4/4) | 0.020 | INFÉRIEUR |
| within_session | eegnex | alexmi | 8 | +0.019 [-0.017, +0.058] (3/8) | 0.057 | NON INFÉRIEUR |
| within_session | eegnex | bnci2014_001 | 9 | -0.009 [-0.026, +0.008] (6/9) | 0.026 | NON CONCLUANT |
| within_session | eegnex | bnci2014_002 | 14 | -0.000 [-0.013, +0.012] (4/14) | 0.019 | NON INFÉRIEUR |
| within_session | eegnex | bnci2014_004 | 9 | -0.004 [-0.009, +0.000] (7/9) | 0.008 | NON INFÉRIEUR |
| within_session | eegnex | bnci2015_001 | 12 | +0.006 [-0.002, +0.016] (3/12) | 0.014 | NON INFÉRIEUR |
| within_session | eegnex | cho2017 | 52 | -0.002 [-0.011, +0.007] (29/52) | 0.013 | NON INFÉRIEUR |
| within_session | eegnex | lee2019_mi | 54 | -0.003 [-0.014, +0.009] (31/54) | 0.017 | NON INFÉRIEUR |
| within_session | eegnex | physionetmi | 109 | -0.008 [-0.027, +0.010] (59/109) | 0.026 | NON CONCLUANT |
| within_session | eegnex | schirrmeister2017 | 14 | +0.005 [-0.004, +0.014] (7/14) | 0.013 | NON INFÉRIEUR |
| within_session | eegnex | shin2017a | 29 | -0.020 [-0.046, +0.006] (18/29) | 0.038 | NON CONCLUANT |
| within_session | eegnex | weibo2014 | 10 | -0.045 [-0.087, -0.012] (7/10) | 0.056 | INFÉRIEUR |
| within_session | eegnex | zhou2016 | 4 | -0.006 [-0.014, +0.001] (2/4) | 0.013 | NON INFÉRIEUR |
| within_session | msvtnet | alexmi | 8 | -0.033 [-0.085, +0.002] (4/8) | 0.071 | NON CONCLUANT |
| within_session | msvtnet | bnci2014_001 | 9 | +0.003 [-0.008, +0.013] (3/9) | 0.016 | NON INFÉRIEUR |
| within_session | msvtnet | bnci2014_002 | 14 | +0.007 [-0.004, +0.020] (5/14) | 0.017 | NON INFÉRIEUR |
| within_session | msvtnet | bnci2014_004 | 9 | +0.006 [-0.002, +0.015] (3/9) | 0.013 | NON INFÉRIEUR |
| within_session | msvtnet | bnci2015_001 | 12 | -0.001 [-0.009, +0.007] (6/12) | 0.012 | NON INFÉRIEUR |
| within_session | msvtnet | cho2017 | 52 | +0.003 [-0.005, +0.010] (24/52) | 0.011 | NON INFÉRIEUR |
| within_session | msvtnet | lee2019_mi | 54 | +0.001 [-0.006, +0.007] (25/54) | 0.010 | NON INFÉRIEUR |
| within_session | msvtnet | physionetmi | 109 | +0.009 [-0.007, +0.025] (46/109) | 0.023 | NON INFÉRIEUR |
| within_session | msvtnet | schirrmeister2017 | 14 | +0.001 [-0.010, +0.011] (7/14) | 0.015 | NON INFÉRIEUR |
| within_session | msvtnet | shin2017a | 29 | -0.005 [-0.037, +0.025] (16/29) | 0.046 | NON CONCLUANT |
| within_session | msvtnet | weibo2014 | 10 | +0.003 [-0.018, +0.026] (5/10) | 0.033 | NON INFÉRIEUR |
| within_session | msvtnet | zhou2016 | 4 | +0.007 [-0.010, +0.025] (1/4) | 0.031 | NON INFÉRIEUR |
| within_session | ctnet | alexmi | 8 | -0.008 [-0.038, +0.023] (5/8) | 0.046 | NON CONCLUANT |
| within_session | ctnet | bnci2014_001 | 9 | +0.007 [-0.011, +0.026] (4/9) | 0.028 | NON INFÉRIEUR |
| within_session | ctnet | bnci2014_002 | 14 | -0.013 [-0.026, -0.000] (10/14) | 0.019 | INFÉRIEUR |
| within_session | ctnet | bnci2014_004 | 9 | +0.013 [+0.004, +0.023] (1/9) | 0.015 | SUPÉRIEUR |
| within_session | ctnet | bnci2015_001 | 12 | -0.004 [-0.010, +0.003] (8/12) | 0.009 | NON INFÉRIEUR |
| within_session | ctnet | cho2017 | 52 | -0.003 [-0.014, +0.007] (24/52) | 0.015 | NON INFÉRIEUR |
| within_session | ctnet | lee2019_mi | 54 | -0.003 [-0.012, +0.005] (29/54) | 0.012 | NON INFÉRIEUR |
| within_session | ctnet | physionetmi | 109 | -0.001 [-0.017, +0.014] (52/109) | 0.022 | NON INFÉRIEUR |
| within_session | ctnet | schirrmeister2017 | 14 | -0.004 [-0.016, +0.009] (9/14) | 0.018 | NON INFÉRIEUR |
| within_session | ctnet | shin2017a | 29 | -0.012 [-0.039, +0.014] (15/29) | 0.038 | NON CONCLUANT |
| within_session | ctnet | weibo2014 | 10 | -0.019 [-0.039, +0.000] (7/10) | 0.029 | NON CONCLUANT |
| within_session | ctnet | zhou2016 | 4 | +0.013 [-0.005, +0.033] (1/4) | 0.029 | NON INFÉRIEUR |
| within_session | conformer | alexmi | 8 | -0.040 [-0.085, -0.000] (5/8) | 0.064 | INFÉRIEUR |
| within_session | conformer | bnci2014_001 | 9 | -0.006 [-0.017, +0.007] (6/9) | 0.018 | NON INFÉRIEUR |
| within_session | conformer | bnci2014_002 | 14 | +0.005 [-0.008, +0.023] (7/14) | 0.023 | NON INFÉRIEUR |
| within_session | conformer | bnci2014_004 | 9 | +0.005 [-0.002, +0.013] (4/9) | 0.012 | NON INFÉRIEUR |
| within_session | conformer | bnci2015_001 | 12 | +0.003 [+0.000, +0.006] (4/12) | 0.005 | SUPÉRIEUR |
| within_session | conformer | cho2017 | 52 | -0.001 [-0.009, +0.006] (30/52) | 0.011 | NON INFÉRIEUR |
| within_session | conformer | lee2019_mi | 54 | +0.001 [-0.005, +0.006] (25/54) | 0.008 | NON INFÉRIEUR |
| within_session | conformer | physionetmi | 109 | +0.003 [-0.006, +0.014] (51/109) | 0.014 | NON INFÉRIEUR |
| within_session | conformer | schirrmeister2017 | 14 | -0.003 [-0.019, +0.011] (7/14) | 0.022 | NON INFÉRIEUR |
| within_session | conformer | shin2017a | 29 | -0.001 [-0.017, +0.016] (14/29) | 0.024 | NON INFÉRIEUR |
| within_session | conformer | weibo2014 | 10 | -0.004 [-0.014, +0.006] (6/10) | 0.015 | NON INFÉRIEUR |
| within_session | conformer | zhou2016 | 4 | +0.008 [-0.002, +0.018] (1/4) | 0.017 | NON INFÉRIEUR |
| cross_session | shallow | bnci2014_001 | 9 | -0.007 [-0.022, +0.008] (6/9) | 0.023 | NON CONCLUANT |
| cross_session | shallow | bnci2014_004 | 9 | -0.002 [-0.005, +0.000] (7/9) | 0.004 | NON INFÉRIEUR |
| cross_session | shallow | bnci2015_001 | 12 | -0.001 [-0.013, +0.010] (6/12) | 0.017 | NON INFÉRIEUR |
| cross_session | shallow | lee2019_mi | 54 | -0.009 [-0.014, -0.003] (37/54) | 0.008 | NON INFÉRIEUR |
| cross_session | shallow | shin2017a | 29 | -0.007 [-0.021, +0.008] (15/29) | 0.021 | NON CONCLUANT |
| cross_session | shallow | zhou2016 | 4 | -0.027 [-0.043, -0.014] (4/4) | 0.025 | INFÉRIEUR |
| cross_session | eegnex | bnci2014_001 | 9 | -0.008 [-0.023, +0.007] (5/9) | 0.023 | NON CONCLUANT |
| cross_session | eegnex | bnci2014_004 | 9 | -0.007 [-0.010, -0.003] (8/9) | 0.005 | NON INFÉRIEUR |
| cross_session | eegnex | bnci2015_001 | 12 | +0.001 [-0.006, +0.007] (6/12) | 0.009 | NON INFÉRIEUR |
| cross_session | eegnex | lee2019_mi | 54 | +0.000 [-0.010, +0.012] (34/54) | 0.016 | NON INFÉRIEUR |
| cross_session | eegnex | shin2017a | 29 | +0.016 [-0.007, +0.041] (12/29) | 0.035 | NON INFÉRIEUR |
| cross_session | eegnex | zhou2016 | 4 | -0.000 [-0.021, +0.023] (2/4) | 0.036 | NON CONCLUANT |
| cross_session | msvtnet | bnci2014_001 | 9 | -0.009 [-0.023, +0.003] (6/9) | 0.020 | NON CONCLUANT |
| cross_session | msvtnet | bnci2014_004 | 9 | +0.000 [-0.006, +0.007] (5/9) | 0.010 | NON INFÉRIEUR |
| cross_session | msvtnet | bnci2015_001 | 12 | +0.005 [-0.008, +0.021] (7/12) | 0.022 | NON INFÉRIEUR |
| cross_session | msvtnet | lee2019_mi | 54 | -0.000 [-0.014, +0.013] (21/54) | 0.020 | NON INFÉRIEUR |
| cross_session | msvtnet | shin2017a | 29 | +0.008 [-0.009, +0.026] (12/29) | 0.025 | NON INFÉRIEUR |
| cross_session | msvtnet | zhou2016 | 4 | +0.008 [+0.006, +0.009] (0/4) | 0.003 | SUPÉRIEUR |
| cross_session | ctnet | bnci2014_001 | 9 | +0.014 [+0.005, +0.023] (2/9) | 0.014 | SUPÉRIEUR |
| cross_session | ctnet | bnci2014_004 | 9 | +0.001 [-0.004, +0.006] (3/9) | 0.007 | NON INFÉRIEUR |
| cross_session | ctnet | bnci2015_001 | 12 | -0.006 [-0.015, +0.002] (5/12) | 0.013 | NON INFÉRIEUR |
| cross_session | ctnet | lee2019_mi | 54 | -0.004 [-0.013, +0.005] (26/54) | 0.013 | NON INFÉRIEUR |
| cross_session | ctnet | shin2017a | 29 | -0.006 [-0.026, +0.014] (15/29) | 0.029 | NON CONCLUANT |
| cross_session | ctnet | zhou2016 | 4 | -0.006 [-0.037, +0.016] (1/4) | 0.042 | NON CONCLUANT |
| cross_session | conformer | bnci2014_001 | 9 | +0.018 [-0.010, +0.047] (3/9) | 0.044 | NON INFÉRIEUR |
| cross_session | conformer | bnci2014_004 | 9 | -0.003 [-0.008, +0.003] (7/9) | 0.008 | NON INFÉRIEUR |
| cross_session | conformer | bnci2015_001 | 12 | +0.003 [-0.001, +0.006] (4/12) | 0.006 | NON INFÉRIEUR |
| cross_session | conformer | lee2019_mi | 54 | -0.003 [-0.010, +0.004] (27/54) | 0.010 | NON INFÉRIEUR |
| cross_session | conformer | shin2017a | 29 | +0.008 [-0.004, +0.021] (12/29) | 0.018 | NON INFÉRIEUR |
| cross_session | conformer | zhou2016 | 4 | +0.017 [-0.006, +0.047] (1/4) | 0.043 | NON INFÉRIEUR |
| cross_subject | shallow | alexmi | 8 | -0.017 [-0.027, -0.008] (6/8) | 0.015 | INFÉRIEUR |
| cross_subject | shallow | bnci2014_001 | 9 | +0.005 [-0.017, +0.028] (3/9) | 0.035 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2014_002 | 14 | -0.002 [-0.021, +0.015] (6/14) | 0.026 | NON CONCLUANT |
| cross_subject | shallow | bnci2014_004 | 9 | +0.004 [-0.005, +0.013] (5/9) | 0.014 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2015_001 | 12 | +0.003 [-0.008, +0.014] (4/12) | 0.017 | NON INFÉRIEUR |
| cross_subject | shallow | cho2017 | 52 | +0.002 [-0.007, +0.010] (24/52) | 0.012 | NON INFÉRIEUR |
| cross_subject | shallow | lee2019_mi | 54 | -0.001 [-0.007, +0.006] (26/54) | 0.009 | NON INFÉRIEUR |
| cross_subject | shallow | physionetmi | 109 | -0.000 [-0.007, +0.006] (54/109) | 0.010 | NON INFÉRIEUR |
| cross_subject | shallow | schirrmeister2017 | 14 | -0.011 [-0.026, +0.004] (10/14) | 0.023 | NON CONCLUANT |
| cross_subject | shallow | shin2017a | 29 | -0.004 [-0.021, +0.013] (15/29) | 0.025 | NON CONCLUANT |
| cross_subject | shallow | weibo2014 | 10 | +0.017 [+0.003, +0.031] (3/10) | 0.021 | SUPÉRIEUR |
| cross_subject | shallow | zhou2016 | 4 | +0.002 [-0.007, +0.010] (2/4) | 0.014 | NON INFÉRIEUR |
| cross_subject | eegnex | alexmi | 8 | -0.010 [-0.058, +0.035] (4/8) | 0.073 | NON CONCLUANT |
| cross_subject | eegnex | bnci2014_001 | 9 | -0.006 [-0.058, +0.032] (4/9) | 0.070 | NON CONCLUANT |
| cross_subject | eegnex | bnci2014_002 | 14 | -0.006 [-0.015, +0.002] (9/14) | 0.013 | NON INFÉRIEUR |
| cross_subject | eegnex | bnci2014_004 | 9 | +0.005 [+0.001, +0.008] (2/9) | 0.006 | SUPÉRIEUR |
| cross_subject | eegnex | bnci2015_001 | 12 | -0.001 [-0.013, +0.012] (7/12) | 0.018 | NON INFÉRIEUR |
| cross_subject | eegnex | cho2017 | 52 | +0.002 [-0.006, +0.010] (24/52) | 0.011 | NON INFÉRIEUR |
| cross_subject | eegnex | lee2019_mi | 54 | -0.003 [-0.009, +0.004] (30/54) | 0.010 | NON INFÉRIEUR |
| cross_subject | eegnex | physionetmi | 109 | -0.005 [-0.015, +0.006] (57/109) | 0.015 | NON INFÉRIEUR |
| cross_subject | eegnex | schirrmeister2017 | 14 | -0.017 [-0.057, +0.019] (8/14) | 0.056 | NON CONCLUANT |
| cross_subject | eegnex | shin2017a | 29 | -0.008 [-0.025, +0.009] (15/29) | 0.025 | NON CONCLUANT |
| cross_subject | eegnex | weibo2014 | 10 | -0.009 [-0.022, +0.005] (6/10) | 0.021 | NON CONCLUANT |
| cross_subject | eegnex | zhou2016 | 4 | -0.007 [-0.038, +0.025] (2/4) | 0.054 | NON CONCLUANT |
| cross_subject | msvtnet | alexmi | 8 | +0.002 [-0.015, +0.021] (3/8) | 0.027 | NON INFÉRIEUR |
| cross_subject | msvtnet | bnci2014_001 | 9 | -0.004 [-0.033, +0.029] (6/9) | 0.048 | NON CONCLUANT |
| cross_subject | msvtnet | bnci2014_002 | 14 | -0.009 [-0.027, +0.011] (10/14) | 0.029 | NON CONCLUANT |
| cross_subject | msvtnet | bnci2014_004 | 9 | +0.003 [-0.010, +0.015] (3/9) | 0.019 | NON INFÉRIEUR |
| cross_subject | msvtnet | bnci2015_001 | 12 | -0.037 [-0.076, -0.001] (8/12) | 0.056 | INFÉRIEUR |
| cross_subject | msvtnet | cho2017 | 52 | -0.005 [-0.014, +0.004] (24/52) | 0.013 | NON INFÉRIEUR |
| cross_subject | msvtnet | lee2019_mi | 54 | -0.002 [-0.008, +0.004] (29/54) | 0.009 | NON INFÉRIEUR |
| cross_subject | msvtnet | physionetmi | 109 | +0.005 [-0.004, +0.014] (47/109) | 0.012 | NON INFÉRIEUR |
| cross_subject | msvtnet | schirrmeister2017 | 14 | +0.000 [-0.010, +0.010] (5/14) | 0.014 | NON INFÉRIEUR |
| cross_subject | msvtnet | shin2017a | 29 | +0.007 [-0.014, +0.026] (10/29) | 0.028 | NON INFÉRIEUR |
| cross_subject | msvtnet | weibo2014 | 10 | +0.016 [-0.003, +0.035] (4/10) | 0.029 | NON INFÉRIEUR |
| cross_subject | msvtnet | zhou2016 | 4 | -0.003 [-0.039, +0.024] (1/4) | 0.053 | NON CONCLUANT |
| cross_subject | ctnet | alexmi | 8 | -0.019 [-0.060, +0.017] (4/8) | 0.059 | NON CONCLUANT |
| cross_subject | ctnet | bnci2014_001 | 9 | -0.005 [-0.020, +0.012] (6/9) | 0.025 | NON INFÉRIEUR |
| cross_subject | ctnet | bnci2014_002 | 14 | +0.013 [-0.001, +0.029] (5/14) | 0.022 | NON INFÉRIEUR |
| cross_subject | ctnet | bnci2014_004 | 9 | -0.005 [-0.015, +0.007] (6/9) | 0.017 | NON INFÉRIEUR |
| cross_subject | ctnet | bnci2015_001 | 12 | -0.010 [-0.021, +0.001] (8/12) | 0.017 | NON CONCLUANT |
| cross_subject | ctnet | cho2017 | 52 | +0.001 [-0.007, +0.010] (29/52) | 0.013 | NON INFÉRIEUR |
| cross_subject | ctnet | lee2019_mi | 54 | +0.002 [-0.004, +0.009] (30/54) | 0.009 | NON INFÉRIEUR |
| cross_subject | ctnet | physionetmi | 109 | +0.009 [-0.001, +0.019] (50/109) | 0.014 | NON INFÉRIEUR |
| cross_subject | ctnet | schirrmeister2017 | 14 | -0.002 [-0.034, +0.027] (4/14) | 0.045 | NON CONCLUANT |
| cross_subject | ctnet | shin2017a | 29 | +0.019 [+0.003, +0.034] (11/29) | 0.023 | SUPÉRIEUR |
| cross_subject | ctnet | weibo2014 | 10 | +0.005 [-0.011, +0.021] (4/10) | 0.024 | NON INFÉRIEUR |
| cross_subject | ctnet | zhou2016 | 4 | +0.016 [-0.007, +0.050] (1/4) | 0.047 | NON INFÉRIEUR |
| cross_subject | conformer | alexmi | 8 | -0.017 [-0.056, +0.021] (5/8) | 0.059 | NON CONCLUANT |
| cross_subject | conformer | bnci2014_001 | 9 | +0.016 [-0.017, +0.056] (6/9) | 0.057 | NON INFÉRIEUR |
| cross_subject | conformer | bnci2014_002 | 14 | +0.001 [-0.007, +0.010] (9/14) | 0.013 | NON INFÉRIEUR |
| cross_subject | conformer | bnci2014_004 | 9 | +0.001 [-0.002, +0.005] (6/9) | 0.005 | NON INFÉRIEUR |
| cross_subject | conformer | bnci2015_001 | 12 | +0.019 [-0.000, +0.038] (5/12) | 0.029 | NON INFÉRIEUR |
| cross_subject | conformer | cho2017 | 52 | +0.001 [-0.007, +0.010] (27/52) | 0.013 | NON INFÉRIEUR |
| cross_subject | conformer | lee2019_mi | 54 | +0.001 [-0.005, +0.006] (25/54) | 0.008 | NON INFÉRIEUR |
| cross_subject | conformer | physionetmi | 109 | +0.002 [-0.004, +0.008] (52/109) | 0.009 | NON INFÉRIEUR |
| cross_subject | conformer | schirrmeister2017 | 14 | -0.037 [-0.073, -0.004] (8/14) | 0.051 | INFÉRIEUR |
| cross_subject | conformer | shin2017a | 29 | +0.005 [-0.011, +0.020] (11/29) | 0.022 | NON INFÉRIEUR |
| cross_subject | conformer | weibo2014 | 10 | -0.004 [-0.015, +0.008] (7/10) | 0.017 | NON INFÉRIEUR |
| cross_subject | conformer | zhou2016 | 4 | +0.010 [-0.005, +0.030] (1/4) | 0.030 | NON INFÉRIEUR |

### Q2 groupé (Holm sur les 15 endpoints lisibles, α = 0.025 unilatéral)

| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |
|---|---|---|---|---|---|---|---|
| within_session | shallow | 12 | -0.001 [-0.005, +0.002] | 0.7509 | 0.0000 | 0.005 | NON CONCLUANT |
| within_session | eegnex | 12 | -0.006 [-0.011, +0.000] | 0.9740 | 0.0000 | 0.008 | NON CONCLUANT |
| within_session | tidnet | 0 | — | — | — | — | AUCUN DATASET |
| within_session | msvtnet | 12 | +0.000 [-0.006, +0.006] | 0.4766 | 0.0000 | 0.008 | NON CONCLUANT |
| within_session | ctnet | 12 | -0.003 [-0.008, +0.002] | 0.8651 | 0.0000 | 0.007 | NON CONCLUANT |
| within_session | conformer | 12 | -0.002 [-0.007, +0.002] | 0.8419 | 0.0000 | 0.007 | NON CONCLUANT |
| cross_session | shallow | 6 | -0.009 [-0.014, -0.004] | 0.9998 | 0.0000 | 0.007 | NON CONCLUANT |
| cross_session | eegnex | 6 | +0.000 [-0.006, +0.007] | 0.4662 | 0.0000 | 0.009 | NON CONCLUANT |
| cross_session | tidnet | 0 | — | — | — | — | AUCUN DATASET |
| cross_session | msvtnet | 6 | +0.002 [-0.003, +0.007] | 0.2165 | 0.0000 | 0.007 | NON CONCLUANT |
| cross_session | ctnet | 6 | -0.001 [-0.007, +0.005] | 0.6255 | 0.0000 | 0.009 | NON CONCLUANT |
| cross_session | conformer | 6 | +0.007 [+0.000, +0.014] | 0.0221 | 0.0000 | 0.010 | NON CONCLUANT |
| cross_subject | shallow | 12 | -0.000 [-0.004, +0.004] | 0.5233 | 0.0000 | 0.005 | NON CONCLUANT |
| cross_subject | eegnex | 12 | -0.005 [-0.013, +0.002] | 0.9277 | 0.0000 | 0.010 | NON CONCLUANT |
| cross_subject | tidnet | 0 | — | — | — | — | AUCUN DATASET |
| cross_subject | msvtnet | 12 | -0.002 [-0.009, +0.004] | 0.7781 | 0.0000 | 0.009 | NON CONCLUANT |
| cross_subject | ctnet | 12 | +0.002 [-0.004, +0.008] | 0.2361 | 0.0000 | 0.008 | NON CONCLUANT |
| cross_subject | conformer | 12 | -0.000 [-0.006, +0.006] | 0.5344 | 0.0000 | 0.009 | NON CONCLUANT |

## C1 — tiny_bdinit − bd (endpoint du contrôle d'init de Shallow)

| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict |
|---|---|---|---|---|---|---|
| within_session | shallow | alexmi | 8 | +0.033 [-0.031, +0.108] (4/8) | 0.107 | NON CONCLUANT |
| within_session | shallow | bnci2014_001 | 9 | +0.010 [-0.003, +0.024] (4/9) | 0.021 | NON INFÉRIEUR |
| within_session | shallow | bnci2014_002 | 14 | +0.020 [+0.007, +0.035] (3/14) | 0.021 | SUPÉRIEUR |
| within_session | shallow | bnci2014_004 | 9 | -0.005 [-0.014, +0.002] (5/9) | 0.012 | NON INFÉRIEUR |
| within_session | shallow | bnci2015_001 | 12 | +0.016 [+0.009, +0.022] (1/12) | 0.010 | SUPÉRIEUR |
| within_session | shallow | cho2017 | 52 | -0.001 [-0.010, +0.007] (24/52) | 0.012 | NON INFÉRIEUR |
| within_session | shallow | lee2019_mi | 54 | +0.030 [+0.020, +0.040] (12/54) | 0.015 | SUPÉRIEUR |
| within_session | shallow | physionetmi | 109 | +0.021 [+0.007, +0.035] (39/109) | 0.020 | SUPÉRIEUR |
| within_session | shallow | schirrmeister2017 | 14 | +0.008 [-0.001, +0.016] (4/14) | 0.012 | NON INFÉRIEUR |
| within_session | shallow | shin2017a | 29 | +0.014 [-0.004, +0.034] (14/29) | 0.027 | NON INFÉRIEUR |
| within_session | shallow | weibo2014 | 10 | -0.003 [-0.028, +0.020] (4/10) | 0.037 | NON CONCLUANT |
| within_session | shallow | zhou2016 | 4 | +0.029 [+0.012, +0.046] (0/4) | 0.030 | SUPÉRIEUR |
| cross_session | shallow | bnci2014_001 | 9 | +0.009 [-0.008, +0.027] (4/9) | 0.027 | NON INFÉRIEUR |
| cross_session | shallow | bnci2014_004 | 9 | -0.004 [-0.008, +0.000] (7/9) | 0.006 | NON INFÉRIEUR |
| cross_session | shallow | bnci2015_001 | 12 | +0.012 [+0.001, +0.026] (4/12) | 0.018 | SUPÉRIEUR |
| cross_session | shallow | lee2019_mi | 54 | +0.024 [+0.014, +0.033] (13/54) | 0.014 | SUPÉRIEUR |
| cross_session | shallow | shin2017a | 29 | +0.029 [+0.006, +0.052] (9/29) | 0.034 | SUPÉRIEUR |
| cross_session | shallow | zhou2016 | 4 | +0.021 [+0.007, +0.035] (0/4) | 0.022 | SUPÉRIEUR |
| cross_subject | shallow | alexmi | 8 | -0.023 [-0.073, +0.023] (4/8) | 0.074 | NON CONCLUANT |
| cross_subject | shallow | bnci2014_001 | 9 | -0.009 [-0.031, +0.009] (4/9) | 0.030 | NON CONCLUANT |
| cross_subject | shallow | bnci2014_002 | 14 | +0.013 [-0.008, +0.037] (5/14) | 0.034 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2014_004 | 9 | +0.004 [-0.001, +0.008] (1/9) | 0.007 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2015_001 | 12 | -0.008 [-0.026, +0.009] (7/12) | 0.027 | NON CONCLUANT |
| cross_subject | shallow | cho2017 | 52 | +0.002 [-0.007, +0.011] (19/52) | 0.013 | NON INFÉRIEUR |
| cross_subject | shallow | lee2019_mi | 54 | +0.003 [-0.004, +0.010] (24/54) | 0.010 | NON INFÉRIEUR |
| cross_subject | shallow | physionetmi | 109 | -0.001 [-0.009, +0.007] (48/109) | 0.011 | NON INFÉRIEUR |
| cross_subject | shallow | schirrmeister2017 | 14 | +0.009 [-0.006, +0.025] (6/14) | 0.023 | NON INFÉRIEUR |
| cross_subject | shallow | shin2017a | 29 | +0.005 [-0.020, +0.030] (14/29) | 0.037 | NON INFÉRIEUR |
| cross_subject | shallow | weibo2014 | 10 | +0.027 [+0.009, +0.045] (1/10) | 0.027 | SUPÉRIEUR |
| cross_subject | shallow | zhou2016 | 4 | +0.009 [-0.010, +0.025] (1/4) | 0.027 | NON INFÉRIEUR |

### C1 groupé (Holm sur les 3 endpoints lisibles, α = 0.025 unilatéral)

| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |
|---|---|---|---|---|---|---|---|
| within_session | shallow | 12 | +0.014 [+0.007, +0.022] | 0.0000 | 0.0000 | 0.010 | SUPÉRIEUR |
| cross_session | shallow | 6 | +0.015 [+0.009, +0.021] | 0.0000 | 0.0000 | 0.008 | SUPÉRIEUR |
| cross_subject | shallow | 12 | +0.003 [-0.004, +0.008] | 0.1946 | 0.0000 | 0.009 | NON INFÉRIEUR |

## C2 — tiny − tiny_bdinit (rapporté, ne décide rien)

| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict |
|---|---|---|---|---|---|---|
| within_session | shallow | alexmi | 8 | +0.015 [-0.037, +0.073] (4/8) | 0.084 | NON CONCLUANT |
| within_session | shallow | bnci2014_001 | 9 | +0.025 [+0.006, +0.039] (1/9) | 0.026 | SUPÉRIEUR |
| within_session | shallow | bnci2014_002 | 14 | +0.009 [-0.003, +0.021] (4/14) | 0.018 | NON INFÉRIEUR |
| within_session | shallow | bnci2014_004 | 9 | +0.003 [-0.003, +0.009] (4/9) | 0.010 | NON INFÉRIEUR |
| within_session | shallow | bnci2015_001 | 12 | +0.009 [+0.002, +0.017] (3/12) | 0.011 | SUPÉRIEUR |
| within_session | shallow | cho2017 | 52 | +0.029 [+0.021, +0.037] (7/52) | 0.012 | SUPÉRIEUR |
| within_session | shallow | lee2019_mi | 54 | +0.027 [+0.019, +0.036] (11/54) | 0.013 | SUPÉRIEUR |
| within_session | shallow | physionetmi | 109 | +0.013 [-0.000, +0.027] (43/109) | 0.020 | NON INFÉRIEUR |
| within_session | shallow | schirrmeister2017 | 14 | +0.005 [+0.000, +0.010] (4/14) | 0.007 | SUPÉRIEUR |
| within_session | shallow | shin2017a | 29 | +0.008 [-0.011, +0.029] (12/29) | 0.029 | NON INFÉRIEUR |
| within_session | shallow | weibo2014 | 10 | +0.018 [+0.004, +0.038] (1/10) | 0.025 | SUPÉRIEUR |
| within_session | shallow | zhou2016 | 4 | +0.007 [-0.002, +0.014] (1/4) | 0.013 | NON INFÉRIEUR |
| cross_session | shallow | bnci2014_001 | 9 | +0.013 [+0.004, +0.022] (2/9) | 0.014 | SUPÉRIEUR |
| cross_session | shallow | bnci2014_004 | 9 | +0.001 [-0.005, +0.007] (6/9) | 0.009 | NON INFÉRIEUR |
| cross_session | shallow | bnci2015_001 | 12 | +0.001 [-0.007, +0.009] (4/12) | 0.013 | NON INFÉRIEUR |
| cross_session | shallow | lee2019_mi | 54 | +0.012 [+0.001, +0.024] (16/54) | 0.016 | SUPÉRIEUR |
| cross_session | shallow | shin2017a | 29 | +0.026 [+0.004, +0.050] (12/29) | 0.034 | SUPÉRIEUR |
| cross_session | shallow | zhou2016 | 4 | -0.014 [-0.028, -0.000] (4/4) | 0.023 | INFÉRIEUR |
| cross_subject | shallow | alexmi | 8 | +0.013 [-0.015, +0.042] (3/8) | 0.042 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2014_001 | 9 | +0.012 [-0.013, +0.035] (2/9) | 0.036 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2014_002 | 14 | -0.012 [-0.027, +0.003] (9/14) | 0.022 | NON CONCLUANT |
| cross_subject | shallow | bnci2014_004 | 9 | -0.002 [-0.008, +0.004] (5/9) | 0.009 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2015_001 | 12 | +0.009 [-0.004, +0.023] (5/12) | 0.021 | NON INFÉRIEUR |
| cross_subject | shallow | cho2017 | 52 | +0.006 [-0.002, +0.015] (21/52) | 0.012 | NON INFÉRIEUR |
| cross_subject | shallow | lee2019_mi | 54 | -0.005 [-0.012, +0.001] (30/54) | 0.009 | NON INFÉRIEUR |
| cross_subject | shallow | physionetmi | 109 | -0.002 [-0.010, +0.005] (59/109) | 0.011 | NON INFÉRIEUR |
| cross_subject | shallow | schirrmeister2017 | 14 | -0.033 [-0.056, -0.013] (11/14) | 0.032 | INFÉRIEUR |
| cross_subject | shallow | shin2017a | 29 | +0.004 [-0.020, +0.028] (17/29) | 0.036 | NON INFÉRIEUR |
| cross_subject | shallow | weibo2014 | 10 | +0.003 [-0.013, +0.018] (4/10) | 0.024 | NON INFÉRIEUR |
| cross_subject | shallow | zhou2016 | 4 | +0.009 [-0.001, +0.019] (1/4) | 0.018 | NON INFÉRIEUR |

### C2 groupé (sans test : rapporté)

| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |
|---|---|---|---|---|---|---|---|
| within_session | shallow | 12 | +0.014 [+0.009, +0.020] | 0.0000 | 0.0000 | 0.008 | RAPPORTÉ |
| cross_session | shallow | 6 | +0.007 [+0.001, +0.012] | 0.0081 | 0.0000 | 0.008 | RAPPORTÉ |
| cross_subject | shallow | 12 | +0.000 [-0.005, +0.005] | 0.4973 | 0.0000 | 0.007 | RAPPORTÉ |

## C3 — fix − bd (rapporté, ne décide rien)

| protocole | modèle | dataset | n | Δ [IC95] (neg) | MDE | verdict |
|---|---|---|---|---|---|---|
| within_session | shallow | alexmi | 8 | +0.010 [-0.027, +0.052] (2/8) | 0.061 | NON CONCLUANT |
| within_session | shallow | bnci2014_001 | 9 | +0.031 [+0.021, +0.041] (0/9) | 0.015 | SUPÉRIEUR |
| within_session | shallow | bnci2014_002 | 14 | +0.025 [+0.010, +0.040] (2/14) | 0.023 | SUPÉRIEUR |
| within_session | shallow | bnci2014_004 | 9 | -0.003 [-0.009, +0.003] (5/9) | 0.010 | NON INFÉRIEUR |
| within_session | shallow | bnci2015_001 | 12 | +0.020 [+0.012, +0.027] (0/12) | 0.011 | SUPÉRIEUR |
| within_session | shallow | cho2017 | 52 | +0.021 [+0.013, +0.030] (9/52) | 0.012 | SUPÉRIEUR |
| within_session | shallow | lee2019_mi | 54 | +0.017 [+0.009, +0.025] (14/54) | 0.011 | SUPÉRIEUR |
| within_session | shallow | physionetmi | 109 | +0.009 [-0.002, +0.020] (54/109) | 0.016 | NON INFÉRIEUR |
| within_session | shallow | schirrmeister2017 | 14 | +0.010 [-0.001, +0.019] (5/14) | 0.015 | NON INFÉRIEUR |
| within_session | shallow | shin2017a | 29 | -0.018 [-0.039, +0.002] (18/29) | 0.029 | NON CONCLUANT |
| within_session | shallow | weibo2014 | 10 | -0.001 [-0.018, +0.016] (4/10) | 0.026 | NON INFÉRIEUR |
| within_session | shallow | zhou2016 | 4 | +0.043 [+0.019, +0.067] (0/4) | 0.039 | SUPÉRIEUR |
| cross_session | shallow | bnci2014_001 | 9 | +0.021 [+0.004, +0.037] (1/9) | 0.025 | SUPÉRIEUR |
| cross_session | shallow | bnci2014_004 | 9 | -0.008 [-0.014, -0.002] (6/9) | 0.009 | NON INFÉRIEUR |
| cross_session | shallow | bnci2015_001 | 12 | +0.016 [+0.005, +0.028] (3/12) | 0.017 | SUPÉRIEUR |
| cross_session | shallow | lee2019_mi | 54 | +0.018 [+0.009, +0.027] (16/54) | 0.013 | SUPÉRIEUR |
| cross_session | shallow | shin2017a | 29 | +0.009 [-0.006, +0.025] (12/29) | 0.022 | NON INFÉRIEUR |
| cross_session | shallow | zhou2016 | 4 | +0.009 [-0.002, +0.022] (1/4) | 0.020 | NON INFÉRIEUR |
| cross_subject | shallow | alexmi | 8 | -0.002 [-0.038, +0.035] (5/8) | 0.056 | NON CONCLUANT |
| cross_subject | shallow | bnci2014_001 | 9 | -0.017 [-0.031, -0.003] (7/9) | 0.021 | INFÉRIEUR |
| cross_subject | shallow | bnci2014_002 | 14 | +0.012 [-0.007, +0.033] (8/14) | 0.030 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2014_004 | 9 | +0.004 [-0.003, +0.010] (2/9) | 0.010 | NON INFÉRIEUR |
| cross_subject | shallow | bnci2015_001 | 12 | -0.003 [-0.011, +0.005] (6/12) | 0.012 | NON INFÉRIEUR |
| cross_subject | shallow | cho2017 | 52 | +0.006 [-0.003, +0.015] (21/52) | 0.013 | NON INFÉRIEUR |
| cross_subject | shallow | lee2019_mi | 54 | +0.003 [-0.004, +0.009] (27/54) | 0.009 | NON INFÉRIEUR |
| cross_subject | shallow | physionetmi | 109 | +0.002 [-0.005, +0.009] (40/109) | 0.010 | NON INFÉRIEUR |
| cross_subject | shallow | schirrmeister2017 | 14 | +0.004 [-0.027, +0.033] (6/14) | 0.044 | NON CONCLUANT |
| cross_subject | shallow | shin2017a | 29 | -0.007 [-0.026, +0.013] (17/29) | 0.028 | NON CONCLUANT |
| cross_subject | shallow | weibo2014 | 10 | +0.022 [-0.001, +0.043] (3/10) | 0.034 | NON INFÉRIEUR |
| cross_subject | shallow | zhou2016 | 4 | +0.010 [-0.017, +0.028] (1/4) | 0.039 | NON INFÉRIEUR |

### C3 groupé (sans test : rapporté)

| protocole | modèle | datasets | Δ̄ [IC95] | p sup | p NI | MDE | verdict |
|---|---|---|---|---|---|---|---|
| within_session | shallow | 12 | +0.014 [+0.009, +0.019] | 0.0000 | 0.0000 | 0.007 | RAPPORTÉ |
| cross_session | shallow | 6 | +0.011 [+0.006, +0.016] | 0.0001 | 0.0000 | 0.007 | RAPPORTÉ |
| cross_subject | shallow | 12 | +0.003 [-0.003, +0.009] | 0.1587 | 0.0000 | 0.008 | RAPPORTÉ |

## Lecture de Q1 Shallow (PREREG_shallow_init_control.md § 3)

| protocole | Q1 | C1 | lecture |
|---|---|---|---|
| within_session | SUPÉRIEUR | SUPÉRIEUR | C1 SUPÉRIEUR : l'effet de Shallow est attribuable à la croissance |
| cross_session | SUPÉRIEUR | SUPÉRIEUR | C1 SUPÉRIEUR : l'effet de Shallow est attribuable à la croissance |
| cross_subject | NON INFÉRIEUR | NON INFÉRIEUR | Q1 NON INFÉRIEUR, C1 NON INFÉRIEUR : Q1 rapporté tel quel |

Identité C1 + C2 = Q1 vérifiée sur 30 cases, 765 sujets, erreur max 0.0e+00.

## Stabilité et croissance (comptes, par bras)

| protocole | modèle | dataset | bras | fits | divergés (score constant) | non finis | pics > 1e6 | ont grandi |
|---|---|---|---|---|---|---|---|---|
| within_session | shallow | alexmi | bd | 40 | 0 | 0 | 0 | — |
| within_session | shallow | alexmi | tiny | 40 | 0 | 0 | 0 | 40 |
| within_session | shallow | alexmi | random | 40 | 0 | 0 | 0 | 40 |
| within_session | shallow | alexmi | tiny_bdinit | 40 | 0 | 0 | 0 | 40 |
| within_session | shallow | alexmi | fix | 40 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2014_001 | bd | 90 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2014_001 | tiny | 90 | 0 | 0 | 0 | 90 |
| within_session | shallow | bnci2014_001 | random | 90 | 0 | 0 | 0 | 90 |
| within_session | shallow | bnci2014_001 | tiny_bdinit | 90 | 0 | 0 | 0 | 90 |
| within_session | shallow | bnci2014_001 | fix | 90 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2014_002 | bd | 70 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2014_002 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | shallow | bnci2014_002 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | shallow | bnci2014_002 | tiny_bdinit | 70 | 0 | 0 | 0 | 70 |
| within_session | shallow | bnci2014_002 | fix | 70 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2014_004 | bd | 225 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2014_004 | tiny | 225 | 0 | 0 | 0 | 225 |
| within_session | shallow | bnci2014_004 | random | 225 | 0 | 0 | 0 | 225 |
| within_session | shallow | bnci2014_004 | tiny_bdinit | 225 | 0 | 0 | 0 | 225 |
| within_session | shallow | bnci2014_004 | fix | 225 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2015_001 | bd | 140 | 0 | 0 | 0 | — |
| within_session | shallow | bnci2015_001 | tiny | 140 | 0 | 0 | 0 | 140 |
| within_session | shallow | bnci2015_001 | random | 140 | 0 | 0 | 0 | 140 |
| within_session | shallow | bnci2015_001 | tiny_bdinit | 140 | 0 | 0 | 0 | 140 |
| within_session | shallow | bnci2015_001 | fix | 140 | 0 | 0 | 0 | — |
| within_session | shallow | cho2017 | bd | 260 | 0 | 0 | 0 | — |
| within_session | shallow | cho2017 | tiny | 260 | 0 | 0 | 0 | 260 |
| within_session | shallow | cho2017 | random | 260 | 0 | 0 | 0 | 260 |
| within_session | shallow | cho2017 | tiny_bdinit | 260 | 0 | 0 | 0 | 260 |
| within_session | shallow | cho2017 | fix | 260 | 0 | 0 | 0 | — |
| within_session | shallow | lee2019_mi | bd | 540 | 0 | 0 | 0 | — |
| within_session | shallow | lee2019_mi | tiny | 540 | 0 | 0 | 0 | 540 |
| within_session | shallow | lee2019_mi | random | 540 | 0 | 0 | 0 | 540 |
| within_session | shallow | lee2019_mi | tiny_bdinit | 540 | 0 | 0 | 0 | 540 |
| within_session | shallow | lee2019_mi | fix | 540 | 0 | 0 | 0 | — |
| within_session | shallow | physionetmi | bd | 545 | 0 | 0 | 0 | — |
| within_session | shallow | physionetmi | tiny | 545 | 0 | 0 | 0 | 545 |
| within_session | shallow | physionetmi | random | 545 | 0 | 0 | 0 | 545 |
| within_session | shallow | physionetmi | tiny_bdinit | 545 | 0 | 0 | 0 | 545 |
| within_session | shallow | physionetmi | fix | 545 | 0 | 0 | 0 | — |
| within_session | shallow | schirrmeister2017 | bd | 70 | 0 | 0 | 0 | — |
| within_session | shallow | schirrmeister2017 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | shallow | schirrmeister2017 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | shallow | schirrmeister2017 | tiny_bdinit | 70 | 0 | 0 | 0 | 70 |
| within_session | shallow | schirrmeister2017 | fix | 70 | 0 | 0 | 0 | — |
| within_session | shallow | shin2017a | bd | 435 | 0 | 0 | 0 | — |
| within_session | shallow | shin2017a | tiny | 435 | 0 | 0 | 0 | 435 |
| within_session | shallow | shin2017a | random | 435 | 0 | 0 | 0 | 435 |
| within_session | shallow | shin2017a | tiny_bdinit | 435 | 0 | 0 | 0 | 435 |
| within_session | shallow | shin2017a | fix | 435 | 0 | 0 | 0 | — |
| within_session | shallow | weibo2014 | bd | 50 | 0 | 0 | 0 | — |
| within_session | shallow | weibo2014 | tiny | 50 | 0 | 0 | 0 | 50 |
| within_session | shallow | weibo2014 | random | 50 | 0 | 0 | 0 | 50 |
| within_session | shallow | weibo2014 | tiny_bdinit | 50 | 0 | 0 | 0 | 50 |
| within_session | shallow | weibo2014 | fix | 50 | 0 | 0 | 0 | — |
| within_session | shallow | zhou2016 | bd | 60 | 0 | 0 | 0 | — |
| within_session | shallow | zhou2016 | tiny | 60 | 0 | 0 | 0 | 60 |
| within_session | shallow | zhou2016 | random | 60 | 0 | 0 | 0 | 60 |
| within_session | shallow | zhou2016 | tiny_bdinit | 60 | 0 | 0 | 0 | 60 |
| within_session | shallow | zhou2016 | fix | 60 | 0 | 0 | 0 | — |
| within_session | eegnex | alexmi | bd | 40 | 0 | 0 | 0 | — |
| within_session | eegnex | alexmi | tiny | 40 | 0 | 0 | 0 | 40 |
| within_session | eegnex | alexmi | random | 40 | 0 | 0 | 0 | 40 |
| within_session | eegnex | bnci2014_001 | bd | 90 | 0 | 0 | 0 | — |
| within_session | eegnex | bnci2014_001 | tiny | 90 | 0 | 0 | 0 | 90 |
| within_session | eegnex | bnci2014_001 | random | 90 | 0 | 0 | 0 | 90 |
| within_session | eegnex | bnci2014_002 | bd | 70 | 0 | 0 | 0 | — |
| within_session | eegnex | bnci2014_002 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | eegnex | bnci2014_002 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | eegnex | bnci2014_004 | bd | 225 | 0 | 0 | 0 | — |
| within_session | eegnex | bnci2014_004 | tiny | 225 | 0 | 0 | 0 | 225 |
| within_session | eegnex | bnci2014_004 | random | 225 | 0 | 0 | 0 | 225 |
| within_session | eegnex | bnci2015_001 | bd | 140 | 0 | 0 | 0 | — |
| within_session | eegnex | bnci2015_001 | tiny | 140 | 0 | 0 | 0 | 140 |
| within_session | eegnex | bnci2015_001 | random | 140 | 0 | 0 | 0 | 140 |
| within_session | eegnex | cho2017 | bd | 260 | 0 | 0 | 0 | — |
| within_session | eegnex | cho2017 | tiny | 260 | 0 | 0 | 0 | 260 |
| within_session | eegnex | cho2017 | random | 260 | 0 | 0 | 0 | 260 |
| within_session | eegnex | lee2019_mi | bd | 540 | 0 | 0 | 0 | — |
| within_session | eegnex | lee2019_mi | tiny | 540 | 0 | 0 | 0 | 540 |
| within_session | eegnex | lee2019_mi | random | 540 | 0 | 0 | 0 | 540 |
| within_session | eegnex | physionetmi | bd | 545 | 0 | 0 | 0 | — |
| within_session | eegnex | physionetmi | tiny | 545 | 0 | 0 | 0 | 545 |
| within_session | eegnex | physionetmi | random | 545 | 0 | 0 | 0 | 545 |
| within_session | eegnex | schirrmeister2017 | bd | 70 | 0 | 0 | 0 | — |
| within_session | eegnex | schirrmeister2017 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | eegnex | schirrmeister2017 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | eegnex | shin2017a | bd | 435 | 0 | 0 | 0 | — |
| within_session | eegnex | shin2017a | tiny | 435 | 0 | 0 | 0 | 435 |
| within_session | eegnex | shin2017a | random | 435 | 0 | 0 | 0 | 435 |
| within_session | eegnex | weibo2014 | bd | 50 | 0 | 0 | 0 | — |
| within_session | eegnex | weibo2014 | tiny | 50 | 0 | 0 | 0 | 50 |
| within_session | eegnex | weibo2014 | random | 50 | 0 | 0 | 0 | 50 |
| within_session | eegnex | zhou2016 | bd | 60 | 0 | 0 | 0 | — |
| within_session | eegnex | zhou2016 | tiny | 60 | 0 | 0 | 0 | 60 |
| within_session | eegnex | zhou2016 | random | 60 | 0 | 0 | 0 | 60 |
| within_session | msvtnet | alexmi | bd | 40 | 0 | 0 | 0 | — |
| within_session | msvtnet | alexmi | tiny | 40 | 0 | 0 | 0 | 40 |
| within_session | msvtnet | alexmi | random | 40 | 0 | 0 | 0 | 40 |
| within_session | msvtnet | bnci2014_001 | bd | 90 | 0 | 0 | 0 | — |
| within_session | msvtnet | bnci2014_001 | tiny | 90 | 0 | 0 | 0 | 90 |
| within_session | msvtnet | bnci2014_001 | random | 90 | 0 | 0 | 0 | 90 |
| within_session | msvtnet | bnci2014_002 | bd | 70 | 0 | 0 | 0 | — |
| within_session | msvtnet | bnci2014_002 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | msvtnet | bnci2014_002 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | msvtnet | bnci2014_004 | bd | 225 | 0 | 0 | 0 | — |
| within_session | msvtnet | bnci2014_004 | tiny | 225 | 0 | 0 | 0 | 225 |
| within_session | msvtnet | bnci2014_004 | random | 225 | 0 | 0 | 0 | 225 |
| within_session | msvtnet | bnci2015_001 | bd | 140 | 0 | 0 | 0 | — |
| within_session | msvtnet | bnci2015_001 | tiny | 140 | 0 | 0 | 0 | 140 |
| within_session | msvtnet | bnci2015_001 | random | 140 | 0 | 0 | 0 | 140 |
| within_session | msvtnet | cho2017 | bd | 260 | 0 | 0 | 0 | — |
| within_session | msvtnet | cho2017 | tiny | 260 | 0 | 0 | 0 | 260 |
| within_session | msvtnet | cho2017 | random | 260 | 0 | 0 | 0 | 260 |
| within_session | msvtnet | lee2019_mi | bd | 540 | 0 | 0 | 0 | — |
| within_session | msvtnet | lee2019_mi | tiny | 540 | 0 | 0 | 0 | 540 |
| within_session | msvtnet | lee2019_mi | random | 540 | 0 | 0 | 0 | 540 |
| within_session | msvtnet | physionetmi | bd | 545 | 0 | 0 | 0 | — |
| within_session | msvtnet | physionetmi | tiny | 545 | 0 | 0 | 0 | 545 |
| within_session | msvtnet | physionetmi | random | 545 | 0 | 0 | 0 | 545 |
| within_session | msvtnet | schirrmeister2017 | bd | 70 | 0 | 0 | 0 | — |
| within_session | msvtnet | schirrmeister2017 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | msvtnet | schirrmeister2017 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | msvtnet | shin2017a | bd | 435 | 0 | 0 | 0 | — |
| within_session | msvtnet | shin2017a | tiny | 435 | 0 | 0 | 0 | 435 |
| within_session | msvtnet | shin2017a | random | 435 | 0 | 0 | 0 | 435 |
| within_session | msvtnet | weibo2014 | bd | 50 | 0 | 0 | 0 | — |
| within_session | msvtnet | weibo2014 | tiny | 50 | 0 | 0 | 0 | 50 |
| within_session | msvtnet | weibo2014 | random | 50 | 0 | 0 | 0 | 50 |
| within_session | msvtnet | zhou2016 | bd | 60 | 0 | 0 | 0 | — |
| within_session | msvtnet | zhou2016 | tiny | 60 | 0 | 0 | 0 | 60 |
| within_session | msvtnet | zhou2016 | random | 60 | 0 | 0 | 0 | 60 |
| within_session | ctnet | alexmi | bd | 40 | 0 | 0 | 0 | — |
| within_session | ctnet | alexmi | tiny | 40 | 0 | 0 | 0 | 40 |
| within_session | ctnet | alexmi | random | 40 | 0 | 0 | 0 | 40 |
| within_session | ctnet | bnci2014_001 | bd | 90 | 0 | 0 | 0 | — |
| within_session | ctnet | bnci2014_001 | tiny | 90 | 0 | 0 | 0 | 90 |
| within_session | ctnet | bnci2014_001 | random | 90 | 0 | 0 | 0 | 90 |
| within_session | ctnet | bnci2014_002 | bd | 70 | 0 | 0 | 0 | — |
| within_session | ctnet | bnci2014_002 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | ctnet | bnci2014_002 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | ctnet | bnci2014_004 | bd | 225 | 0 | 0 | 0 | — |
| within_session | ctnet | bnci2014_004 | tiny | 225 | 0 | 0 | 0 | 225 |
| within_session | ctnet | bnci2014_004 | random | 225 | 0 | 0 | 0 | 225 |
| within_session | ctnet | bnci2015_001 | bd | 140 | 0 | 0 | 0 | — |
| within_session | ctnet | bnci2015_001 | tiny | 140 | 0 | 0 | 0 | 140 |
| within_session | ctnet | bnci2015_001 | random | 140 | 0 | 0 | 0 | 140 |
| within_session | ctnet | cho2017 | bd | 260 | 0 | 0 | 0 | — |
| within_session | ctnet | cho2017 | tiny | 260 | 0 | 0 | 0 | 260 |
| within_session | ctnet | cho2017 | random | 260 | 0 | 0 | 0 | 260 |
| within_session | ctnet | lee2019_mi | bd | 540 | 0 | 0 | 0 | — |
| within_session | ctnet | lee2019_mi | tiny | 540 | 0 | 0 | 0 | 540 |
| within_session | ctnet | lee2019_mi | random | 540 | 0 | 0 | 0 | 540 |
| within_session | ctnet | physionetmi | bd | 545 | 0 | 0 | 0 | — |
| within_session | ctnet | physionetmi | tiny | 545 | 0 | 0 | 0 | 545 |
| within_session | ctnet | physionetmi | random | 545 | 0 | 0 | 0 | 545 |
| within_session | ctnet | schirrmeister2017 | bd | 70 | 0 | 0 | 0 | — |
| within_session | ctnet | schirrmeister2017 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | ctnet | schirrmeister2017 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | ctnet | shin2017a | bd | 435 | 0 | 0 | 0 | — |
| within_session | ctnet | shin2017a | tiny | 435 | 0 | 0 | 0 | 435 |
| within_session | ctnet | shin2017a | random | 435 | 0 | 0 | 0 | 435 |
| within_session | ctnet | weibo2014 | bd | 50 | 0 | 0 | 0 | — |
| within_session | ctnet | weibo2014 | tiny | 50 | 0 | 0 | 0 | 50 |
| within_session | ctnet | weibo2014 | random | 50 | 0 | 0 | 0 | 50 |
| within_session | ctnet | zhou2016 | bd | 60 | 0 | 0 | 0 | — |
| within_session | ctnet | zhou2016 | tiny | 60 | 0 | 0 | 0 | 60 |
| within_session | ctnet | zhou2016 | random | 60 | 0 | 0 | 0 | 60 |
| within_session | conformer | alexmi | bd | 40 | 0 | 0 | 0 | — |
| within_session | conformer | alexmi | tiny | 40 | 0 | 0 | 0 | 40 |
| within_session | conformer | alexmi | random | 40 | 0 | 0 | 0 | 40 |
| within_session | conformer | bnci2014_001 | bd | 90 | 0 | 0 | 0 | — |
| within_session | conformer | bnci2014_001 | tiny | 90 | 0 | 0 | 0 | 90 |
| within_session | conformer | bnci2014_001 | random | 90 | 0 | 0 | 0 | 90 |
| within_session | conformer | bnci2014_002 | bd | 70 | 0 | 0 | 0 | — |
| within_session | conformer | bnci2014_002 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | conformer | bnci2014_002 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | conformer | bnci2014_004 | bd | 225 | 0 | 0 | 0 | — |
| within_session | conformer | bnci2014_004 | tiny | 225 | 0 | 0 | 0 | 225 |
| within_session | conformer | bnci2014_004 | random | 225 | 0 | 0 | 0 | 225 |
| within_session | conformer | bnci2015_001 | bd | 140 | 0 | 0 | 0 | — |
| within_session | conformer | bnci2015_001 | tiny | 140 | 0 | 0 | 0 | 140 |
| within_session | conformer | bnci2015_001 | random | 140 | 0 | 0 | 0 | 140 |
| within_session | conformer | cho2017 | bd | 260 | 0 | 0 | 0 | — |
| within_session | conformer | cho2017 | tiny | 260 | 0 | 0 | 0 | 260 |
| within_session | conformer | cho2017 | random | 260 | 0 | 0 | 0 | 260 |
| within_session | conformer | lee2019_mi | bd | 540 | 0 | 0 | 0 | — |
| within_session | conformer | lee2019_mi | tiny | 540 | 0 | 0 | 0 | 540 |
| within_session | conformer | lee2019_mi | random | 540 | 0 | 0 | 0 | 540 |
| within_session | conformer | physionetmi | bd | 545 | 0 | 0 | 0 | — |
| within_session | conformer | physionetmi | tiny | 545 | 0 | 0 | 0 | 545 |
| within_session | conformer | physionetmi | random | 545 | 0 | 0 | 0 | 545 |
| within_session | conformer | schirrmeister2017 | bd | 70 | 0 | 0 | 0 | — |
| within_session | conformer | schirrmeister2017 | tiny | 70 | 0 | 0 | 0 | 70 |
| within_session | conformer | schirrmeister2017 | random | 70 | 0 | 0 | 0 | 70 |
| within_session | conformer | shin2017a | bd | 435 | 0 | 0 | 0 | — |
| within_session | conformer | shin2017a | tiny | 435 | 0 | 0 | 0 | 435 |
| within_session | conformer | shin2017a | random | 435 | 0 | 0 | 0 | 435 |
| within_session | conformer | weibo2014 | bd | 50 | 0 | 0 | 0 | — |
| within_session | conformer | weibo2014 | tiny | 50 | 0 | 0 | 0 | 50 |
| within_session | conformer | weibo2014 | random | 50 | 0 | 0 | 0 | 50 |
| within_session | conformer | zhou2016 | bd | 60 | 0 | 0 | 0 | — |
| within_session | conformer | zhou2016 | tiny | 60 | 0 | 0 | 0 | 60 |
| within_session | conformer | zhou2016 | random | 60 | 0 | 0 | 0 | 60 |
| cross_session | shallow | bnci2014_001 | bd | 18 | 0 | 0 | 0 | — |
| cross_session | shallow | bnci2014_001 | tiny | 18 | 0 | 0 | 0 | 18 |
| cross_session | shallow | bnci2014_001 | random | 18 | 0 | 0 | 0 | 18 |
| cross_session | shallow | bnci2014_001 | tiny_bdinit | 18 | 0 | 0 | 0 | 18 |
| cross_session | shallow | bnci2014_001 | fix | 18 | 0 | 0 | 0 | — |
| cross_session | shallow | bnci2014_004 | bd | 45 | 0 | 0 | 0 | — |
| cross_session | shallow | bnci2014_004 | tiny | 45 | 0 | 0 | 0 | 45 |
| cross_session | shallow | bnci2014_004 | random | 45 | 0 | 0 | 0 | 45 |
| cross_session | shallow | bnci2014_004 | tiny_bdinit | 45 | 0 | 0 | 0 | 45 |
| cross_session | shallow | bnci2014_004 | fix | 45 | 0 | 0 | 0 | — |
| cross_session | shallow | bnci2015_001 | bd | 28 | 0 | 0 | 0 | — |
| cross_session | shallow | bnci2015_001 | tiny | 28 | 0 | 0 | 0 | 28 |
| cross_session | shallow | bnci2015_001 | random | 28 | 0 | 0 | 0 | 28 |
| cross_session | shallow | bnci2015_001 | tiny_bdinit | 28 | 0 | 0 | 0 | 28 |
| cross_session | shallow | bnci2015_001 | fix | 28 | 0 | 0 | 0 | — |
| cross_session | shallow | lee2019_mi | bd | 108 | 0 | 0 | 0 | — |
| cross_session | shallow | lee2019_mi | tiny | 108 | 0 | 0 | 0 | 108 |
| cross_session | shallow | lee2019_mi | random | 108 | 0 | 0 | 0 | 108 |
| cross_session | shallow | lee2019_mi | tiny_bdinit | 108 | 0 | 0 | 0 | 108 |
| cross_session | shallow | lee2019_mi | fix | 108 | 0 | 0 | 0 | — |
| cross_session | shallow | shin2017a | bd | 87 | 0 | 0 | 0 | — |
| cross_session | shallow | shin2017a | tiny | 87 | 0 | 0 | 0 | 87 |
| cross_session | shallow | shin2017a | random | 87 | 0 | 0 | 0 | 87 |
| cross_session | shallow | shin2017a | tiny_bdinit | 87 | 0 | 0 | 0 | 87 |
| cross_session | shallow | shin2017a | fix | 87 | 0 | 0 | 0 | — |
| cross_session | shallow | zhou2016 | bd | 12 | 0 | 0 | 0 | — |
| cross_session | shallow | zhou2016 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_session | shallow | zhou2016 | random | 12 | 0 | 0 | 0 | 12 |
| cross_session | shallow | zhou2016 | tiny_bdinit | 12 | 0 | 0 | 0 | 12 |
| cross_session | shallow | zhou2016 | fix | 12 | 0 | 0 | 0 | — |
| cross_session | eegnex | bnci2014_001 | bd | 18 | 0 | 0 | 0 | — |
| cross_session | eegnex | bnci2014_001 | tiny | 18 | 0 | 0 | 0 | 18 |
| cross_session | eegnex | bnci2014_001 | random | 18 | 0 | 0 | 0 | 18 |
| cross_session | eegnex | bnci2014_004 | bd | 45 | 0 | 0 | 0 | — |
| cross_session | eegnex | bnci2014_004 | tiny | 45 | 0 | 0 | 0 | 45 |
| cross_session | eegnex | bnci2014_004 | random | 45 | 0 | 0 | 0 | 45 |
| cross_session | eegnex | bnci2015_001 | bd | 28 | 0 | 0 | 0 | — |
| cross_session | eegnex | bnci2015_001 | tiny | 28 | 0 | 0 | 0 | 28 |
| cross_session | eegnex | bnci2015_001 | random | 28 | 0 | 0 | 0 | 28 |
| cross_session | eegnex | lee2019_mi | bd | 108 | 0 | 0 | 0 | — |
| cross_session | eegnex | lee2019_mi | tiny | 108 | 0 | 0 | 0 | 108 |
| cross_session | eegnex | lee2019_mi | random | 108 | 0 | 0 | 0 | 108 |
| cross_session | eegnex | shin2017a | bd | 87 | 0 | 0 | 0 | — |
| cross_session | eegnex | shin2017a | tiny | 87 | 0 | 0 | 0 | 87 |
| cross_session | eegnex | shin2017a | random | 87 | 0 | 0 | 0 | 87 |
| cross_session | eegnex | zhou2016 | bd | 12 | 0 | 0 | 0 | — |
| cross_session | eegnex | zhou2016 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_session | eegnex | zhou2016 | random | 12 | 0 | 0 | 0 | 12 |
| cross_session | msvtnet | bnci2014_001 | bd | 18 | 0 | 0 | 0 | — |
| cross_session | msvtnet | bnci2014_001 | tiny | 18 | 0 | 0 | 0 | 18 |
| cross_session | msvtnet | bnci2014_001 | random | 18 | 0 | 0 | 0 | 18 |
| cross_session | msvtnet | bnci2014_004 | bd | 45 | 0 | 0 | 0 | — |
| cross_session | msvtnet | bnci2014_004 | tiny | 45 | 0 | 0 | 0 | 45 |
| cross_session | msvtnet | bnci2014_004 | random | 45 | 0 | 0 | 0 | 45 |
| cross_session | msvtnet | bnci2015_001 | bd | 28 | 0 | 0 | 0 | — |
| cross_session | msvtnet | bnci2015_001 | tiny | 28 | 0 | 0 | 0 | 28 |
| cross_session | msvtnet | bnci2015_001 | random | 28 | 0 | 0 | 0 | 28 |
| cross_session | msvtnet | lee2019_mi | bd | 108 | 0 | 0 | 0 | — |
| cross_session | msvtnet | lee2019_mi | tiny | 108 | 0 | 0 | 0 | 108 |
| cross_session | msvtnet | lee2019_mi | random | 108 | 0 | 0 | 0 | 108 |
| cross_session | msvtnet | shin2017a | bd | 87 | 0 | 0 | 0 | — |
| cross_session | msvtnet | shin2017a | tiny | 87 | 0 | 0 | 0 | 87 |
| cross_session | msvtnet | shin2017a | random | 87 | 0 | 0 | 0 | 87 |
| cross_session | msvtnet | zhou2016 | bd | 12 | 0 | 0 | 0 | — |
| cross_session | msvtnet | zhou2016 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_session | msvtnet | zhou2016 | random | 12 | 0 | 0 | 0 | 12 |
| cross_session | ctnet | bnci2014_001 | bd | 18 | 0 | 0 | 0 | — |
| cross_session | ctnet | bnci2014_001 | tiny | 18 | 0 | 0 | 0 | 18 |
| cross_session | ctnet | bnci2014_001 | random | 18 | 0 | 0 | 0 | 18 |
| cross_session | ctnet | bnci2014_004 | bd | 45 | 0 | 0 | 0 | — |
| cross_session | ctnet | bnci2014_004 | tiny | 45 | 0 | 0 | 0 | 45 |
| cross_session | ctnet | bnci2014_004 | random | 45 | 0 | 0 | 0 | 45 |
| cross_session | ctnet | bnci2015_001 | bd | 28 | 0 | 0 | 0 | — |
| cross_session | ctnet | bnci2015_001 | tiny | 28 | 0 | 0 | 0 | 28 |
| cross_session | ctnet | bnci2015_001 | random | 28 | 0 | 0 | 0 | 28 |
| cross_session | ctnet | lee2019_mi | bd | 108 | 0 | 0 | 0 | — |
| cross_session | ctnet | lee2019_mi | tiny | 108 | 0 | 0 | 0 | 108 |
| cross_session | ctnet | lee2019_mi | random | 108 | 0 | 0 | 0 | 108 |
| cross_session | ctnet | shin2017a | bd | 87 | 0 | 0 | 0 | — |
| cross_session | ctnet | shin2017a | tiny | 87 | 0 | 0 | 0 | 87 |
| cross_session | ctnet | shin2017a | random | 87 | 0 | 0 | 0 | 87 |
| cross_session | ctnet | zhou2016 | bd | 12 | 0 | 0 | 0 | — |
| cross_session | ctnet | zhou2016 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_session | ctnet | zhou2016 | random | 12 | 0 | 0 | 0 | 12 |
| cross_session | conformer | bnci2014_001 | bd | 18 | 0 | 0 | 0 | — |
| cross_session | conformer | bnci2014_001 | tiny | 18 | 0 | 0 | 0 | 18 |
| cross_session | conformer | bnci2014_001 | random | 18 | 0 | 0 | 0 | 18 |
| cross_session | conformer | bnci2014_004 | bd | 45 | 0 | 0 | 0 | — |
| cross_session | conformer | bnci2014_004 | tiny | 45 | 0 | 0 | 0 | 45 |
| cross_session | conformer | bnci2014_004 | random | 45 | 0 | 0 | 0 | 45 |
| cross_session | conformer | bnci2015_001 | bd | 28 | 0 | 0 | 0 | — |
| cross_session | conformer | bnci2015_001 | tiny | 28 | 0 | 0 | 0 | 28 |
| cross_session | conformer | bnci2015_001 | random | 28 | 0 | 0 | 0 | 28 |
| cross_session | conformer | lee2019_mi | bd | 108 | 0 | 0 | 0 | — |
| cross_session | conformer | lee2019_mi | tiny | 108 | 0 | 0 | 0 | 108 |
| cross_session | conformer | lee2019_mi | random | 108 | 0 | 0 | 0 | 108 |
| cross_session | conformer | shin2017a | bd | 87 | 0 | 0 | 0 | — |
| cross_session | conformer | shin2017a | tiny | 87 | 0 | 0 | 0 | 87 |
| cross_session | conformer | shin2017a | random | 87 | 0 | 0 | 0 | 87 |
| cross_session | conformer | zhou2016 | bd | 12 | 0 | 0 | 0 | — |
| cross_session | conformer | zhou2016 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_session | conformer | zhou2016 | random | 12 | 0 | 0 | 0 | 12 |
| cross_subject | shallow | alexmi | bd | 8 | 0 | 0 | 0 | — |
| cross_subject | shallow | alexmi | tiny | 8 | 0 | 0 | 0 | 8 |
| cross_subject | shallow | alexmi | random | 8 | 0 | 0 | 0 | 8 |
| cross_subject | shallow | alexmi | tiny_bdinit | 8 | 0 | 0 | 0 | 8 |
| cross_subject | shallow | alexmi | fix | 8 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2014_001 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2014_001 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | shallow | bnci2014_001 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | shallow | bnci2014_001 | tiny_bdinit | 9 | 0 | 0 | 0 | 9 |
| cross_subject | shallow | bnci2014_001 | fix | 9 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2014_002 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2014_002 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | shallow | bnci2014_002 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | shallow | bnci2014_002 | tiny_bdinit | 14 | 0 | 0 | 0 | 14 |
| cross_subject | shallow | bnci2014_002 | fix | 14 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2014_004 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2014_004 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | shallow | bnci2014_004 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | shallow | bnci2014_004 | tiny_bdinit | 9 | 0 | 0 | 0 | 9 |
| cross_subject | shallow | bnci2014_004 | fix | 9 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2015_001 | bd | 12 | 0 | 0 | 0 | — |
| cross_subject | shallow | bnci2015_001 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_subject | shallow | bnci2015_001 | random | 12 | 0 | 0 | 0 | 12 |
| cross_subject | shallow | bnci2015_001 | tiny_bdinit | 12 | 0 | 0 | 0 | 12 |
| cross_subject | shallow | bnci2015_001 | fix | 12 | 0 | 0 | 0 | — |
| cross_subject | shallow | cho2017 | bd | 52 | 0 | 0 | 0 | — |
| cross_subject | shallow | cho2017 | tiny | 52 | 0 | 0 | 0 | 52 |
| cross_subject | shallow | cho2017 | random | 52 | 0 | 0 | 0 | 52 |
| cross_subject | shallow | cho2017 | tiny_bdinit | 52 | 0 | 0 | 0 | 52 |
| cross_subject | shallow | cho2017 | fix | 52 | 0 | 0 | 0 | — |
| cross_subject | shallow | lee2019_mi | bd | 54 | 0 | 0 | 0 | — |
| cross_subject | shallow | lee2019_mi | tiny | 54 | 0 | 0 | 0 | 54 |
| cross_subject | shallow | lee2019_mi | random | 54 | 0 | 0 | 0 | 54 |
| cross_subject | shallow | lee2019_mi | tiny_bdinit | 54 | 0 | 0 | 0 | 54 |
| cross_subject | shallow | lee2019_mi | fix | 54 | 0 | 0 | 0 | — |
| cross_subject | shallow | physionetmi | bd | 109 | 0 | 0 | 0 | — |
| cross_subject | shallow | physionetmi | tiny | 109 | 0 | 0 | 0 | 109 |
| cross_subject | shallow | physionetmi | random | 109 | 0 | 0 | 0 | 109 |
| cross_subject | shallow | physionetmi | tiny_bdinit | 109 | 0 | 0 | 0 | 109 |
| cross_subject | shallow | physionetmi | fix | 109 | 0 | 0 | 0 | — |
| cross_subject | shallow | schirrmeister2017 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | shallow | schirrmeister2017 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | shallow | schirrmeister2017 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | shallow | schirrmeister2017 | tiny_bdinit | 14 | 0 | 0 | 0 | 14 |
| cross_subject | shallow | schirrmeister2017 | fix | 14 | 0 | 0 | 0 | — |
| cross_subject | shallow | shin2017a | bd | 29 | 0 | 0 | 0 | — |
| cross_subject | shallow | shin2017a | tiny | 29 | 0 | 0 | 0 | 29 |
| cross_subject | shallow | shin2017a | random | 29 | 0 | 0 | 0 | 29 |
| cross_subject | shallow | shin2017a | tiny_bdinit | 29 | 0 | 0 | 0 | 29 |
| cross_subject | shallow | shin2017a | fix | 29 | 0 | 0 | 0 | — |
| cross_subject | shallow | weibo2014 | bd | 10 | 0 | 0 | 0 | — |
| cross_subject | shallow | weibo2014 | tiny | 10 | 0 | 0 | 0 | 10 |
| cross_subject | shallow | weibo2014 | random | 10 | 0 | 0 | 0 | 10 |
| cross_subject | shallow | weibo2014 | tiny_bdinit | 10 | 0 | 0 | 0 | 10 |
| cross_subject | shallow | weibo2014 | fix | 10 | 0 | 0 | 0 | — |
| cross_subject | shallow | zhou2016 | bd | 4 | 0 | 0 | 0 | — |
| cross_subject | shallow | zhou2016 | tiny | 4 | 0 | 0 | 0 | 4 |
| cross_subject | shallow | zhou2016 | random | 4 | 0 | 0 | 0 | 4 |
| cross_subject | shallow | zhou2016 | tiny_bdinit | 4 | 0 | 0 | 0 | 4 |
| cross_subject | shallow | zhou2016 | fix | 4 | 0 | 0 | 0 | — |
| cross_subject | eegnex | alexmi | bd | 8 | 0 | 0 | 0 | — |
| cross_subject | eegnex | alexmi | tiny | 8 | 0 | 0 | 0 | 8 |
| cross_subject | eegnex | alexmi | random | 8 | 0 | 0 | 0 | 8 |
| cross_subject | eegnex | bnci2014_001 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | eegnex | bnci2014_001 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | eegnex | bnci2014_001 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | eegnex | bnci2014_002 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | eegnex | bnci2014_002 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | eegnex | bnci2014_002 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | eegnex | bnci2014_004 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | eegnex | bnci2014_004 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | eegnex | bnci2014_004 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | eegnex | bnci2015_001 | bd | 12 | 0 | 0 | 0 | — |
| cross_subject | eegnex | bnci2015_001 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_subject | eegnex | bnci2015_001 | random | 12 | 0 | 0 | 0 | 12 |
| cross_subject | eegnex | cho2017 | bd | 52 | 0 | 0 | 0 | — |
| cross_subject | eegnex | cho2017 | tiny | 52 | 0 | 0 | 0 | 52 |
| cross_subject | eegnex | cho2017 | random | 52 | 0 | 0 | 0 | 52 |
| cross_subject | eegnex | lee2019_mi | bd | 54 | 0 | 0 | 0 | — |
| cross_subject | eegnex | lee2019_mi | tiny | 54 | 0 | 0 | 0 | 54 |
| cross_subject | eegnex | lee2019_mi | random | 54 | 0 | 0 | 0 | 54 |
| cross_subject | eegnex | physionetmi | bd | 109 | 0 | 0 | 0 | — |
| cross_subject | eegnex | physionetmi | tiny | 109 | 0 | 0 | 0 | 109 |
| cross_subject | eegnex | physionetmi | random | 109 | 0 | 0 | 0 | 109 |
| cross_subject | eegnex | schirrmeister2017 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | eegnex | schirrmeister2017 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | eegnex | schirrmeister2017 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | eegnex | shin2017a | bd | 29 | 0 | 0 | 0 | — |
| cross_subject | eegnex | shin2017a | tiny | 29 | 0 | 0 | 0 | 29 |
| cross_subject | eegnex | shin2017a | random | 29 | 0 | 0 | 0 | 29 |
| cross_subject | eegnex | weibo2014 | bd | 10 | 0 | 0 | 0 | — |
| cross_subject | eegnex | weibo2014 | tiny | 10 | 0 | 0 | 0 | 10 |
| cross_subject | eegnex | weibo2014 | random | 10 | 0 | 0 | 0 | 10 |
| cross_subject | eegnex | zhou2016 | bd | 4 | 0 | 0 | 0 | — |
| cross_subject | eegnex | zhou2016 | tiny | 4 | 0 | 0 | 0 | 4 |
| cross_subject | eegnex | zhou2016 | random | 4 | 0 | 0 | 0 | 4 |
| cross_subject | msvtnet | alexmi | bd | 8 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | alexmi | tiny | 8 | 0 | 0 | 0 | 8 |
| cross_subject | msvtnet | alexmi | random | 8 | 0 | 0 | 0 | 8 |
| cross_subject | msvtnet | bnci2014_001 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | bnci2014_001 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | msvtnet | bnci2014_001 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | msvtnet | bnci2014_002 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | bnci2014_002 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | msvtnet | bnci2014_002 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | msvtnet | bnci2014_004 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | bnci2014_004 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | msvtnet | bnci2014_004 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | msvtnet | bnci2015_001 | bd | 12 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | bnci2015_001 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_subject | msvtnet | bnci2015_001 | random | 12 | 0 | 0 | 0 | 12 |
| cross_subject | msvtnet | cho2017 | bd | 52 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | cho2017 | tiny | 52 | 0 | 0 | 0 | 52 |
| cross_subject | msvtnet | cho2017 | random | 52 | 0 | 0 | 0 | 52 |
| cross_subject | msvtnet | lee2019_mi | bd | 54 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | lee2019_mi | tiny | 54 | 0 | 0 | 0 | 54 |
| cross_subject | msvtnet | lee2019_mi | random | 54 | 0 | 0 | 0 | 54 |
| cross_subject | msvtnet | physionetmi | bd | 109 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | physionetmi | tiny | 109 | 0 | 0 | 0 | 109 |
| cross_subject | msvtnet | physionetmi | random | 109 | 0 | 0 | 0 | 109 |
| cross_subject | msvtnet | schirrmeister2017 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | schirrmeister2017 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | msvtnet | schirrmeister2017 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | msvtnet | shin2017a | bd | 29 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | shin2017a | tiny | 29 | 0 | 0 | 0 | 29 |
| cross_subject | msvtnet | shin2017a | random | 29 | 0 | 0 | 0 | 29 |
| cross_subject | msvtnet | weibo2014 | bd | 10 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | weibo2014 | tiny | 10 | 0 | 0 | 0 | 10 |
| cross_subject | msvtnet | weibo2014 | random | 10 | 0 | 0 | 0 | 10 |
| cross_subject | msvtnet | zhou2016 | bd | 4 | 0 | 0 | 0 | — |
| cross_subject | msvtnet | zhou2016 | tiny | 4 | 0 | 0 | 0 | 4 |
| cross_subject | msvtnet | zhou2016 | random | 4 | 0 | 0 | 0 | 4 |
| cross_subject | ctnet | alexmi | bd | 8 | 0 | 0 | 0 | — |
| cross_subject | ctnet | alexmi | tiny | 8 | 0 | 0 | 0 | 8 |
| cross_subject | ctnet | alexmi | random | 8 | 0 | 0 | 0 | 8 |
| cross_subject | ctnet | bnci2014_001 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | ctnet | bnci2014_001 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | ctnet | bnci2014_001 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | ctnet | bnci2014_002 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | ctnet | bnci2014_002 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | ctnet | bnci2014_002 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | ctnet | bnci2014_004 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | ctnet | bnci2014_004 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | ctnet | bnci2014_004 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | ctnet | bnci2015_001 | bd | 12 | 0 | 0 | 0 | — |
| cross_subject | ctnet | bnci2015_001 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_subject | ctnet | bnci2015_001 | random | 12 | 0 | 0 | 0 | 12 |
| cross_subject | ctnet | cho2017 | bd | 52 | 0 | 0 | 0 | — |
| cross_subject | ctnet | cho2017 | tiny | 52 | 0 | 0 | 0 | 52 |
| cross_subject | ctnet | cho2017 | random | 52 | 0 | 0 | 0 | 52 |
| cross_subject | ctnet | lee2019_mi | bd | 54 | 0 | 0 | 0 | — |
| cross_subject | ctnet | lee2019_mi | tiny | 54 | 0 | 0 | 0 | 54 |
| cross_subject | ctnet | lee2019_mi | random | 54 | 0 | 0 | 0 | 54 |
| cross_subject | ctnet | physionetmi | bd | 109 | 0 | 0 | 0 | — |
| cross_subject | ctnet | physionetmi | tiny | 109 | 0 | 0 | 0 | 109 |
| cross_subject | ctnet | physionetmi | random | 109 | 0 | 0 | 0 | 109 |
| cross_subject | ctnet | schirrmeister2017 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | ctnet | schirrmeister2017 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | ctnet | schirrmeister2017 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | ctnet | shin2017a | bd | 29 | 0 | 0 | 0 | — |
| cross_subject | ctnet | shin2017a | tiny | 29 | 0 | 0 | 0 | 29 |
| cross_subject | ctnet | shin2017a | random | 29 | 0 | 0 | 0 | 29 |
| cross_subject | ctnet | weibo2014 | bd | 10 | 0 | 0 | 0 | — |
| cross_subject | ctnet | weibo2014 | tiny | 10 | 0 | 0 | 0 | 10 |
| cross_subject | ctnet | weibo2014 | random | 10 | 0 | 0 | 0 | 10 |
| cross_subject | ctnet | zhou2016 | bd | 4 | 0 | 0 | 0 | — |
| cross_subject | ctnet | zhou2016 | tiny | 4 | 0 | 0 | 0 | 4 |
| cross_subject | ctnet | zhou2016 | random | 4 | 0 | 0 | 0 | 4 |
| cross_subject | conformer | alexmi | bd | 8 | 0 | 0 | 0 | — |
| cross_subject | conformer | alexmi | tiny | 8 | 0 | 0 | 0 | 8 |
| cross_subject | conformer | alexmi | random | 8 | 0 | 0 | 0 | 8 |
| cross_subject | conformer | bnci2014_001 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | conformer | bnci2014_001 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | conformer | bnci2014_001 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | conformer | bnci2014_002 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | conformer | bnci2014_002 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | conformer | bnci2014_002 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | conformer | bnci2014_004 | bd | 9 | 0 | 0 | 0 | — |
| cross_subject | conformer | bnci2014_004 | tiny | 9 | 0 | 0 | 0 | 9 |
| cross_subject | conformer | bnci2014_004 | random | 9 | 0 | 0 | 0 | 9 |
| cross_subject | conformer | bnci2015_001 | bd | 12 | 0 | 0 | 0 | — |
| cross_subject | conformer | bnci2015_001 | tiny | 12 | 0 | 0 | 0 | 12 |
| cross_subject | conformer | bnci2015_001 | random | 12 | 0 | 0 | 0 | 12 |
| cross_subject | conformer | cho2017 | bd | 52 | 0 | 0 | 0 | — |
| cross_subject | conformer | cho2017 | tiny | 52 | 0 | 0 | 0 | 52 |
| cross_subject | conformer | cho2017 | random | 52 | 0 | 0 | 0 | 52 |
| cross_subject | conformer | lee2019_mi | bd | 54 | 0 | 0 | 0 | — |
| cross_subject | conformer | lee2019_mi | tiny | 54 | 0 | 0 | 0 | 54 |
| cross_subject | conformer | lee2019_mi | random | 54 | 0 | 0 | 0 | 54 |
| cross_subject | conformer | physionetmi | bd | 109 | 0 | 0 | 0 | — |
| cross_subject | conformer | physionetmi | tiny | 109 | 0 | 0 | 0 | 109 |
| cross_subject | conformer | physionetmi | random | 109 | 0 | 0 | 0 | 109 |
| cross_subject | conformer | schirrmeister2017 | bd | 14 | 0 | 0 | 0 | — |
| cross_subject | conformer | schirrmeister2017 | tiny | 14 | 0 | 0 | 0 | 14 |
| cross_subject | conformer | schirrmeister2017 | random | 14 | 0 | 0 | 0 | 14 |
| cross_subject | conformer | shin2017a | bd | 29 | 0 | 0 | 0 | — |
| cross_subject | conformer | shin2017a | tiny | 29 | 0 | 0 | 0 | 29 |
| cross_subject | conformer | shin2017a | random | 29 | 0 | 0 | 0 | 29 |
| cross_subject | conformer | weibo2014 | bd | 10 | 0 | 0 | 0 | — |
| cross_subject | conformer | weibo2014 | tiny | 10 | 0 | 0 | 0 | 10 |
| cross_subject | conformer | weibo2014 | random | 10 | 0 | 0 | 0 | 10 |
| cross_subject | conformer | zhou2016 | bd | 4 | 0 | 0 | 0 | — |
| cross_subject | conformer | zhou2016 | tiny | 4 | 0 | 0 | 0 | 4 |
| cross_subject | conformer | zhou2016 | random | 4 | 0 | 0 | 0 | 4 |
