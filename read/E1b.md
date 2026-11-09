# E1b — ablation des FFN au test (campagne A, poids finaux)

Tâches : 324 sujets, toutes présentes. Fits lus : 18882. Compteurs : {}.

## Exclusions par (modèle, bras, protocole)

- conformer / bd / cross_session : 298 fits, exclus 0
- conformer / bd / cross_subject : 324 fits, exclus 0
- conformer / bd / within_session : 2525 fits, exclus 0
- conformer / tiny / cross_session : 298 fits, exclus 0
- conformer / tiny / cross_subject : 324 fits, exclus 0
- conformer / tiny / within_session : 2525 fits, exclus {'g0': 1}
- ctnet / bd / cross_session : 298 fits, exclus 0
- ctnet / bd / cross_subject : 324 fits, exclus 0
- ctnet / bd / within_session : 2525 fits, exclus 0
- ctnet / tiny / cross_session : 298 fits, exclus 0
- ctnet / tiny / cross_subject : 324 fits, exclus 0
- ctnet / tiny / within_session : 2525 fits, exclus {'g0': 1}
- msvtnet / bd / cross_session : 298 fits, exclus 0
- msvtnet / bd / cross_subject : 324 fits, exclus 0
- msvtnet / bd / within_session : 2525 fits, exclus 0
- msvtnet / tiny / cross_session : 298 fits, exclus 0
- msvtnet / tiny / cross_subject : 324 fits, exclus 0
- msvtnet / tiny / within_session : 2525 fits, exclus 0

## Endpoints primaires (conformer, bras bd ; Holm à 0.025 unilatéral sur les 3 protocoles)

- **H1** within_session : m_deep − full = -0.005 [-0.007, -0.004] (n=324, 12 ds, MDE 0.002) → **NON INFÉRIEUR**
- **H1** cross_session : m_deep − full = -0.007 [-0.011, -0.004] (n=117, 6 ds, MDE 0.004) → **NON INFÉRIEUR**
- **H1** cross_subject : m_deep − full = -0.007 [-0.009, -0.005] (n=324, 12 ds, MDE 0.003) → **NON INFÉRIEUR**

- **H2** within_session : m_deep − m0 = +0.016 [+0.013, +0.019] (n=324, 12 ds, MDE 0.004) → **SUPÉRIEUR**
- **H2** cross_session : m_deep − m0 = +0.024 [+0.019, +0.029] (n=117, 6 ds, MDE 0.007) → **SUPÉRIEUR**
- **H2** cross_subject : m_deep − m0 = +0.023 [+0.017, +0.028] (n=324, 12 ds, MDE 0.007) → **SUPÉRIEUR**

- **H3** within_session : p_tiny − p_unif = +0.008 [+0.006, +0.010] (n=324, 12 ds, MDE 0.003) → **SUPÉRIEUR**
- **H3** cross_session : p_tiny − p_unif = +0.013 [+0.010, +0.017] (n=117, 6 ds, MDE 0.005) → **SUPÉRIEUR**
- **H3** cross_subject : p_tiny − p_unif = +0.011 [+0.008, +0.014] (n=324, 12 ds, MDE 0.004) → **SUPÉRIEUR**

## Secondaires (ne décident rien)

### conformer / bd / within_session

- m0 − full : -0.021 [-0.025, -0.018] (n=324, 12 ds, MDE 0.005)
- m1 − full : -0.002 [-0.003, -0.001] (n=324, 12 ds, MDE 0.002)
- m2 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m3 − full : -0.001 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m4 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m5 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.005 [-0.007, -0.004] (n=324, 12 ds, MDE 0.002)
- m_all − full : -0.032 [-0.036, -0.028] (n=324, 12 ds, MDE 0.006)
- z_deep − full : -0.001 [-0.002, +0.001] (n=324, 12 ds, MDE 0.002)
- z_all − full : -0.010 [-0.013, -0.007] (n=324, 12 ds, MDE 0.004)
- z_deep-z0 : +0.006 [+0.004, +0.008] (n=324, 12 ds, MDE 0.003)
- m_deep-m0 : +0.016 [+0.013, +0.019] (n=324, 12 ds, MDE 0.004)
- p_tiny-full : -0.005 [-0.006, -0.004] (n=324, 12 ds, MDE 0.002)
- p_unif-full : -0.013 [-0.016, -0.011] (n=324, 12 ds, MDE 0.003)
- p_tiny-p_unif : +0.008 [+0.006, +0.010] (n=324, 12 ds, MDE 0.003)
- rho_cost_alloc : +0.151 [+0.115, +0.186] (n=316, 12 ds, MDE 0.050)

### conformer / bd / cross_session

- m0 − full : -0.031 [-0.038, -0.025] (n=117, 6 ds, MDE 0.009)
- m1 − full : -0.004 [-0.006, -0.002] (n=117, 6 ds, MDE 0.003)
- m2 − full : -0.001 [-0.002, +0.000] (n=117, 6 ds, MDE 0.001)
- m3 − full : -0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)
- m4 − full : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.001)
- m5 − full : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.001)
- m_deep − full : -0.007 [-0.011, -0.004] (n=117, 6 ds, MDE 0.004)
- m_all − full : -0.039 [-0.048, -0.031] (n=117, 6 ds, MDE 0.013)
- z_deep − full : +0.002 [-0.001, +0.005] (n=117, 6 ds, MDE 0.004)
- z_all − full : -0.005 [-0.010, +0.000] (n=117, 6 ds, MDE 0.007)
- z_deep-z0 : +0.005 [+0.002, +0.009] (n=117, 6 ds, MDE 0.005)
- m_deep-m0 : +0.024 [+0.019, +0.029] (n=117, 6 ds, MDE 0.007)
- p_tiny-full : -0.008 [-0.012, -0.005] (n=117, 6 ds, MDE 0.005)
- p_unif-full : -0.022 [-0.027, -0.016] (n=117, 6 ds, MDE 0.008)
- p_tiny-p_unif : +0.013 [+0.010, +0.017] (n=117, 6 ds, MDE 0.005)
- rho_cost_alloc : +0.189 [+0.124, +0.255] (n=116, 6 ds, MDE 0.094)

### conformer / bd / cross_subject

- m0 − full : -0.030 [-0.036, -0.023] (n=324, 12 ds, MDE 0.009)
- m1 − full : -0.003 [-0.005, -0.002] (n=324, 12 ds, MDE 0.002)
- m2 − full : -0.001 [-0.002, -0.001] (n=324, 12 ds, MDE 0.001)
- m3 − full : -0.001 [-0.002, -0.000] (n=324, 12 ds, MDE 0.001)
- m4 − full : -0.001 [-0.001, -0.000] (n=324, 12 ds, MDE 0.001)
- m5 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.007 [-0.009, -0.005] (n=324, 12 ds, MDE 0.003)
- m_all − full : -0.043 [-0.050, -0.035] (n=324, 12 ds, MDE 0.011)
- z_deep − full : -0.001 [-0.003, +0.002] (n=324, 12 ds, MDE 0.003)
- z_all − full : -0.023 [-0.029, -0.017] (n=324, 12 ds, MDE 0.008)
- z_deep-z0 : +0.014 [+0.008, +0.019] (n=324, 12 ds, MDE 0.008)
- m_deep-m0 : +0.023 [+0.017, +0.028] (n=324, 12 ds, MDE 0.007)
- p_tiny-full : -0.003 [-0.004, -0.001] (n=324, 12 ds, MDE 0.002)
- p_unif-full : -0.014 [-0.018, -0.011] (n=324, 12 ds, MDE 0.005)
- p_tiny-p_unif : +0.011 [+0.008, +0.014] (n=324, 12 ds, MDE 0.004)
- rho_cost_alloc : +0.237 [+0.177, +0.297] (n=321, 12 ds, MDE 0.087)

### conformer / tiny / within_session

- m0 − full : -0.012 [-0.014, -0.008] (n=324, 12 ds, MDE 0.004)
- m1 − full : -0.001 [-0.002, -0.001] (n=324, 12 ds, MDE 0.001)
- m2 − full : -0.001 [-0.002, -0.001] (n=324, 12 ds, MDE 0.001)
- m3 − full : -0.001 [-0.001, -0.000] (n=324, 12 ds, MDE 0.001)
- m4 − full : -0.001 [-0.001, -0.000] (n=324, 12 ds, MDE 0.001)
- m5 − full : -0.000 [-0.001, -0.000] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.003 [-0.005, -0.001] (n=324, 12 ds, MDE 0.003)
- m_all − full : -0.017 [-0.020, -0.014] (n=324, 12 ds, MDE 0.004)
- z_deep − full : -0.001 [-0.002, -0.000] (n=324, 12 ds, MDE 0.002)
- z_all − full : -0.011 [-0.014, -0.008] (n=324, 12 ds, MDE 0.004)
- z_deep-z0 : +0.008 [+0.006, +0.011] (n=324, 12 ds, MDE 0.003)
- m_deep-m0 : +0.009 [+0.006, +0.011] (n=324, 12 ds, MDE 0.003)

### conformer / tiny / cross_session

- m0 − full : -0.019 [-0.024, -0.014] (n=117, 6 ds, MDE 0.007)
- m1 − full : -0.001 [-0.003, +0.001] (n=117, 6 ds, MDE 0.002)
- m2 − full : -0.000 [-0.001, +0.000] (n=117, 6 ds, MDE 0.001)
- m3 − full : -0.000 [-0.001, +0.000] (n=117, 6 ds, MDE 0.001)
- m4 − full : -0.000 [-0.001, +0.000] (n=117, 6 ds, MDE 0.001)
- m5 − full : +0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)
- m_deep − full : -0.004 [-0.006, -0.002] (n=117, 6 ds, MDE 0.003)
- m_all − full : -0.025 [-0.031, -0.019] (n=117, 6 ds, MDE 0.008)
- z_deep − full : +0.002 [+0.000, +0.004] (n=117, 6 ds, MDE 0.003)
- z_all − full : -0.003 [-0.007, +0.001] (n=117, 6 ds, MDE 0.006)
- z_deep-z0 : +0.005 [+0.002, +0.009] (n=117, 6 ds, MDE 0.006)
- m_deep-m0 : +0.014 [+0.010, +0.019] (n=117, 6 ds, MDE 0.007)

### conformer / tiny / cross_subject

- m0 − full : -0.025 [-0.030, -0.019] (n=324, 12 ds, MDE 0.008)
- m1 − full : -0.001 [-0.003, -0.000] (n=324, 12 ds, MDE 0.002)
- m2 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m3 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m4 − full : -0.000 [-0.001, -0.000] (n=324, 12 ds, MDE 0.000)
- m5 − full : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.000)
- m_deep − full : -0.003 [-0.005, -0.002] (n=324, 12 ds, MDE 0.002)
- m_all − full : -0.030 [-0.036, -0.024] (n=324, 12 ds, MDE 0.008)
- z_deep − full : -0.001 [-0.002, +0.001] (n=324, 12 ds, MDE 0.002)
- z_all − full : -0.016 [-0.021, -0.011] (n=324, 12 ds, MDE 0.007)
- z_deep-z0 : +0.012 [+0.007, +0.016] (n=324, 12 ds, MDE 0.007)
- m_deep-m0 : +0.021 [+0.016, +0.027] (n=324, 12 ds, MDE 0.007)

### ctnet / bd / within_session

- m0 − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m1 − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m2 − full : -0.001 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m3 − full : +0.001 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- m4 − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m5 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.001 [-0.002, +0.001] (n=324, 12 ds, MDE 0.002)
- m_all − full : -0.001 [-0.003, +0.000] (n=324, 12 ds, MDE 0.002)
- z_deep − full : +0.001 [-0.001, +0.002] (n=324, 12 ds, MDE 0.002)
- z_all − full : +0.001 [-0.001, +0.002] (n=324, 12 ds, MDE 0.002)
- z_deep-z0 : +0.000 [-0.001, +0.002] (n=324, 12 ds, MDE 0.002)
- m_deep-m0 : -0.001 [-0.002, +0.000] (n=324, 12 ds, MDE 0.002)
- p_tiny-full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- p_unif-full : +0.001 [-0.000, +0.002] (n=324, 12 ds, MDE 0.001)
- p_tiny-p_unif : -0.000 [-0.002, +0.001] (n=324, 12 ds, MDE 0.001)
- rho_cost_alloc : +0.005 [-0.021, +0.031] (n=313, 12 ds, MDE 0.036)

### ctnet / bd / cross_session

- m0 − full : -0.002 [-0.003, -0.001] (n=117, 6 ds, MDE 0.001)
- m1 − full : -0.001 [-0.002, -0.000] (n=117, 6 ds, MDE 0.001)
- m2 − full : -0.001 [-0.002, +0.000] (n=117, 6 ds, MDE 0.001)
- m3 − full : -0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)
- m4 − full : -0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)
- m5 − full : -0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)
- m_deep − full : -0.003 [-0.005, -0.001] (n=117, 6 ds, MDE 0.003)
- m_all − full : -0.004 [-0.006, -0.002] (n=117, 6 ds, MDE 0.003)
- z_deep − full : -0.001 [-0.002, +0.000] (n=117, 6 ds, MDE 0.002)
- z_all − full : -0.002 [-0.003, -0.000] (n=117, 6 ds, MDE 0.002)
- z_deep-z0 : +0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.002)
- m_deep-m0 : -0.001 [-0.003, -0.000] (n=117, 6 ds, MDE 0.002)
- p_tiny-full : -0.001 [-0.002, +0.000] (n=117, 6 ds, MDE 0.002)
- p_unif-full : -0.001 [-0.002, -0.000] (n=117, 6 ds, MDE 0.002)
- p_tiny-p_unif : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.001)
- rho_cost_alloc : +0.051 [+0.001, +0.102] (n=117, 6 ds, MDE 0.072)

### ctnet / bd / cross_subject

- m0 − full : -0.001 [-0.002, -0.000] (n=324, 12 ds, MDE 0.001)
- m1 − full : -0.001 [-0.002, +0.000] (n=324, 12 ds, MDE 0.001)
- m2 − full : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.001)
- m3 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m4 − full : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.001)
- m5 − full : +0.000 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.002 [-0.003, -0.000] (n=324, 12 ds, MDE 0.002)
- m_all − full : -0.002 [-0.003, -0.000] (n=324, 12 ds, MDE 0.002)
- z_deep − full : -0.001 [-0.002, +0.001] (n=324, 12 ds, MDE 0.002)
- z_all − full : +0.000 [-0.001, +0.002] (n=324, 12 ds, MDE 0.002)
- z_deep-z0 : -0.001 [-0.003, +0.001] (n=324, 12 ds, MDE 0.002)
- m_deep-m0 : -0.001 [-0.002, +0.001] (n=324, 12 ds, MDE 0.002)
- p_tiny-full : -0.001 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- p_unif-full : -0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.001)
- p_tiny-p_unif : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- rho_cost_alloc : +0.037 [-0.054, +0.129] (n=70, 10 ds, MDE 0.130)

### ctnet / tiny / within_session

- m0 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m1 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m2 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m3 − full : +0.000 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- m4 − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m5 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.002)
- m_all − full : -0.001 [-0.002, +0.000] (n=324, 12 ds, MDE 0.002)
- z_deep − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.002)
- z_all − full : -0.000 [-0.002, +0.001] (n=324, 12 ds, MDE 0.002)
- z_deep-z0 : +0.001 [-0.001, +0.002] (n=324, 12 ds, MDE 0.002)
- m_deep-m0 : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.002)

### ctnet / tiny / cross_session

- m0 − full : -0.002 [-0.002, -0.001] (n=117, 6 ds, MDE 0.001)
- m1 − full : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.001)
- m2 − full : -0.001 [-0.001, -0.000] (n=117, 6 ds, MDE 0.001)
- m3 − full : -0.001 [-0.001, -0.000] (n=117, 6 ds, MDE 0.001)
- m4 − full : -0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)
- m5 − full : -0.001 [-0.001, +0.000] (n=117, 6 ds, MDE 0.001)
- m_deep − full : -0.001 [-0.002, -0.000] (n=117, 6 ds, MDE 0.001)
- m_all − full : -0.002 [-0.003, -0.001] (n=117, 6 ds, MDE 0.002)
- z_deep − full : -0.001 [-0.002, +0.000] (n=117, 6 ds, MDE 0.002)
- z_all − full : -0.001 [-0.002, +0.000] (n=117, 6 ds, MDE 0.002)
- z_deep-z0 : -0.001 [-0.002, +0.001] (n=117, 6 ds, MDE 0.002)
- m_deep-m0 : +0.000 [-0.001, +0.001] (n=117, 6 ds, MDE 0.001)

### ctnet / tiny / cross_subject

- m0 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m1 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m2 − full : +0.000 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- m3 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m4 − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m5 − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : +0.001 [-0.000, +0.002] (n=324, 12 ds, MDE 0.001)
- m_all − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.002)
- z_deep − full : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- z_all − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- z_deep-z0 : +0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep-m0 : +0.001 [+0.000, +0.001] (n=324, 12 ds, MDE 0.001)

### msvtnet / bd / within_session

- m0 − full : -0.001 [-0.003, +0.001] (n=324, 12 ds, MDE 0.003)
- m1 − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.000 [-0.001, +0.000] (n=324, 12 ds, MDE 0.001)
- m_all − full : -0.001 [-0.003, +0.001] (n=324, 12 ds, MDE 0.003)
- z_deep − full : -0.001 [-0.002, +0.000] (n=324, 12 ds, MDE 0.001)
- z_all − full : +0.000 [-0.002, +0.002] (n=324, 12 ds, MDE 0.003)
- z_deep-z0 : -0.001 [-0.003, +0.000] (n=324, 12 ds, MDE 0.002)
- m_deep-m0 : +0.001 [-0.001, +0.003] (n=324, 12 ds, MDE 0.003)
- p_tiny-full : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.000)
- p_unif-full : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.001)
- p_tiny-p_unif : +0.000 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- rho_cost_alloc : -0.122 [-0.209, -0.035] (n=204, 11 ds, MDE 0.125)

### msvtnet / bd / cross_session

- m0 − full : -0.002 [-0.004, +0.001] (n=117, 6 ds, MDE 0.004)
- m1 − full : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.001)
- m_deep − full : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.001)
- m_all − full : -0.002 [-0.005, +0.001] (n=117, 6 ds, MDE 0.004)
- z_deep − full : -0.003 [-0.004, -0.002] (n=117, 6 ds, MDE 0.002)
- z_all − full : -0.002 [-0.005, +0.002] (n=117, 6 ds, MDE 0.005)
- z_deep-z0 : -0.005 [-0.008, -0.002] (n=117, 6 ds, MDE 0.005)
- m_deep-m0 : +0.002 [-0.000, +0.005] (n=117, 6 ds, MDE 0.004)
- p_tiny-full : +0.000 [+0.000, +0.000] (n=117, 6 ds, MDE 0.000)
- p_unif-full : -0.000 [-0.000, +0.000] (n=117, 6 ds, MDE 0.000)
- p_tiny-p_unif : +0.000 [-0.000, +0.001] (n=117, 6 ds, MDE 0.000)
- rho_cost_alloc : -0.094 [-0.237, +0.048] (n=101, 5 ds, MDE 0.205)

### msvtnet / bd / cross_subject

- m0 − full : -0.004 [-0.009, +0.000] (n=324, 12 ds, MDE 0.007)
- m1 − full : -0.001 [-0.001, -0.000] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.001 [-0.001, -0.000] (n=324, 12 ds, MDE 0.001)
- m_all − full : -0.005 [-0.009, +0.000] (n=324, 12 ds, MDE 0.007)
- z_deep − full : -0.001 [-0.002, -0.000] (n=324, 12 ds, MDE 0.001)
- z_all − full : -0.008 [-0.014, -0.003] (n=324, 12 ds, MDE 0.008)
- z_deep-z0 : +0.007 [+0.002, +0.013] (n=324, 12 ds, MDE 0.008)
- m_deep-m0 : +0.003 [-0.001, +0.008] (n=324, 12 ds, MDE 0.006)
- p_tiny-full : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.000)
- p_unif-full : -0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.000)
- p_tiny-p_unif : +0.000 [-0.000, +0.000] (n=324, 12 ds, MDE 0.000)
- rho_cost_alloc : -0.103 [-0.231, +0.028] (n=250, 8 ds, MDE 0.187)

### msvtnet / tiny / within_session

- m0 − full : -0.003 [-0.006, -0.001] (n=324, 12 ds, MDE 0.004)
- m1 − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : -0.000 [-0.001, +0.001] (n=324, 12 ds, MDE 0.001)
- m_all − full : -0.004 [-0.006, -0.001] (n=324, 12 ds, MDE 0.004)
- z_deep − full : +0.003 [+0.001, +0.004] (n=324, 12 ds, MDE 0.002)
- z_all − full : +0.001 [-0.002, +0.003] (n=324, 12 ds, MDE 0.004)
- z_deep-z0 : +0.004 [+0.001, +0.006] (n=324, 12 ds, MDE 0.003)
- m_deep-m0 : +0.003 [+0.001, +0.006] (n=324, 12 ds, MDE 0.004)

### msvtnet / tiny / cross_session

- m0 − full : -0.002 [-0.005, +0.001] (n=117, 6 ds, MDE 0.004)
- m1 − full : +0.001 [-0.000, +0.002] (n=117, 6 ds, MDE 0.001)
- m_deep − full : +0.001 [-0.000, +0.002] (n=117, 6 ds, MDE 0.001)
- m_all − full : -0.002 [-0.005, +0.001] (n=117, 6 ds, MDE 0.005)
- z_deep − full : +0.005 [+0.003, +0.008] (n=117, 6 ds, MDE 0.003)
- z_all − full : +0.002 [-0.001, +0.006] (n=117, 6 ds, MDE 0.005)
- z_deep-z0 : +0.007 [+0.003, +0.011] (n=117, 6 ds, MDE 0.005)
- m_deep-m0 : +0.003 [-0.000, +0.006] (n=117, 6 ds, MDE 0.004)

### msvtnet / tiny / cross_subject

- m0 − full : -0.008 [-0.013, -0.002] (n=324, 12 ds, MDE 0.007)
- m1 − full : +0.000 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- m_deep − full : +0.000 [-0.000, +0.001] (n=324, 12 ds, MDE 0.001)
- m_all − full : -0.009 [-0.014, -0.003] (n=324, 12 ds, MDE 0.008)
- z_deep − full : +0.003 [+0.001, +0.004] (n=324, 12 ds, MDE 0.002)
- z_all − full : -0.006 [-0.011, +0.000] (n=324, 12 ds, MDE 0.008)
- z_deep-z0 : +0.009 [+0.004, +0.014] (n=324, 12 ds, MDE 0.008)
- m_deep-m0 : +0.008 [+0.003, +0.013] (n=324, 12 ds, MDE 0.007)

