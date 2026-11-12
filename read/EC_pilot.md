# Pilote EC — les architectures se trompent-elles sur les mêmes essais ?

Note pré-enregistrée : `analysis/PREREG_EC_pilot.md`. G1 (primaire) : 5400 npz, 0 échec.

Strates du primaire : vraie classe × session ; une strate où l'un des 12 vecteurs est constant est écartée pour tous les couples.

| dataset | sujets | exclus (motif) | K_x | K_seed | R [IC 95 %] | med_x | AUC_art | verdict |
|---|---|---|---|---|---|---|---|---|
| cho2017 | 32 | {'jsonl_absent': 20} | 0.181 | 0.264 | 0.684 [0.637, 0.728] | 0.189 | 0.594 | **NON CONCLUANT** |
| lee2019_mi | 29 | {'compte': 6, 'jsonl_absent': 19} | 0.191 | 0.278 | 0.685 [0.644, 0.723] | 0.172 | 0.587 | **NON CONCLUANT** |

**Verdict global (règle gelée) : NON CONCLUANT.**

Groupement √n (ne décide rien) : R = 0.684 [0.654, 0.714], K_x = 0.186, K_seed = 0.271.

## Exclusions G0 par motif

- cho2017 [jsonl_absent] 20 : 32 (msvtnet s1: ); 33 (msvtnet s1: ); 34 (msvtnet s1: ); 35 (eegnex s1: ); 36 (eegnex s1: ); 37 (eegnex s1: ); 38 (eegnex s1: ); 39 (eegnex s1: ); 40 (eegnex s1: ); 41 (eegnex s1: ); 42 (eegnex s1: ); 43 (eegnex s1: ); 44 (eegnex s1: ); 45 (eegnex s1: ); 47 (eegnex s1: ); 48 (eegnex s1: ); 49 (eegnex s1: ); 50 (eegnex s1: ); 51 (eegnex s1: ); 52 (eegnex s1: )
- cho2017 : strates écartées 1 / 64, part d'essais perdue 1.5 % ; par modèle responsable : conformer 1 strates (1.5 %), ctnet 1 strates (1.5 %), eegnex 1 strates (1.5 %) ; sujets sans strate valide : []
- cho2017 : artefact {'n_all_wrong': 58, 'n_all_right': 1039, 'log_ptp_max': 0.40602701536623415, 'log_var_mean': 0.4537851382297302, 'kurt_max': 0.4954282964388835}
- lee2019_mi [compte] 6 : 21 (eegnex s1: 12 npz, 10 lignes jsonl, 10 attendus); 31 (eegnex s1: 6 npz, 6 lignes jsonl, 10 attendus); 38 (eegnex s1: 17 npz, 10 lignes jsonl, 10 attendus); 39 (eegnex s0: 12 npz, 10 lignes jsonl, 10 attendus); 46 (eegnex s2: 4 npz, 4 lignes jsonl, 10 attendus); 50 (eegnex s1: 12 npz, 10 lignes jsonl, 10 attendus)
- lee2019_mi [jsonl_absent] 19 : 32 (eegnex s1: ); 33 (eegnex s1: ); 34 (eegnex s1: ); 35 (eegnex s1: ); 36 (eegnex s1: ); 37 (msvtnet s1: ); 40 (msvtnet s1: ); 41 (msvtnet s1: ); 42 (msvtnet s1: ); 43 (msvtnet s1: ); 44 (msvtnet s1: ); 45 (msvtnet s1: ); 47 (eegnex s2: ); 48 (eegnex s2: ); 49 (eegnex s2: ); 51 (eegnex s1: ); 52 (eegnex s1: ); 53 (eegnex s1: ); 54 (eegnex s1: )
- lee2019_mi : strates écartées 2 / 116, part d'essais perdue 1.7 % ; par modèle responsable : conformer 1 strates (0.9 %), ctnet 1 strates (0.9 %), eegnex 2 strates (1.7 %), msvtnet 1 strates (0.9 %) ; sujets sans strate valide : []
- lee2019_mi : artefact {'n_all_wrong': 82, 'n_all_right': 920, 'log_ptp_max': 0.42888388123011667, 'log_var_mean': 0.41339475079533405, 'kurt_max': 0.46852465535524923}

## Secondaires (ne décident rien)

### cho2017

- κ_same = 0.237 ; κ_same − κ_x = 0.056 [0.048, 0.065] (effet du pli partagé)
- R par paire : eegnex|msvtnet 0.813 ; eegnex|ctnet 0.677 ; eegnex|conformer 0.687 ; msvtnet|ctnet 0.607 ; msvtnet|conformer 0.792 ; ctnet|conformer 0.514
- κ strates « classe » : K_x 0.181, K_seed 0.264, R 0.684 [0.637, 0.728], med_x 0.189 ; strates écartées 1/64 (1.5 % des essais)
- κ strates « aucune » : K_x 0.186, K_seed 0.271, R 0.687 [0.641, 0.730], med_x 0.191 ; strates écartées 0/32 (0.0 % des essais)
- Spearman(accuracy du sujet, κ_x) = 0.809 (p 0.000) ; (accuracy, κ_seed) = 0.896 (p 0.000)
- sujets proches du hasard (accuracy moyenne < quantile 95 % binomial à 0,5) : 21.9 % ; κ_x 0.083 contre 0.208 ; κ_seed 0.165 contre 0.292
- essais où l'argmax des logits ≠ l'argmax du softmax float32 : 0 ; argmax du softmax non unique (p0 == p1) : 0

- Shallow graine 0 (32 sujets) : κ_same(shallow, autres) = 0.254 ; κ_same entre les 4 à la graine 0 = 0.232 ; strates {'n_strata': 64, 'n_dropped': 1, 'trials_lost_share': 0.015337423312883436, 'by_model': {'conformer': {'strata': 1, 'trials_share': 0.015337423312883436}, 'ctnet': {'strata': 1, 'trials_share': 0.015337423312883436}}} ; sans strate valide []

### lee2019_mi

- κ_same = 0.276 ; κ_same − κ_x = 0.086 [0.077, 0.094] (effet du pli partagé)
- R par paire : eegnex|msvtnet 0.838 ; eegnex|ctnet 0.650 ; eegnex|conformer 0.704 ; msvtnet|ctnet 0.574 ; msvtnet|conformer 0.807 ; ctnet|conformer 0.525
- κ strates « classe » : K_x 0.195, K_seed 0.283, R 0.690 [0.651, 0.726], med_x 0.182 ; strates écartées 0/58 (0.0 % des essais)
- κ strates « aucune » : K_x 0.202, K_seed 0.292, R 0.693 [0.647, 0.734], med_x 0.193 ; strates écartées 0/29 (0.0 % des essais)
- Spearman(accuracy du sujet, κ_x) = 0.499 (p 0.006) ; (accuracy, κ_seed) = 0.481 (p 0.008)
- sujets proches du hasard (accuracy moyenne < quantile 95 % binomial à 0,5) : 3.4 % ; κ_x 0.100 contre 0.194 ; κ_seed 0.148 contre 0.283
- essais où l'argmax des logits ≠ l'argmax du softmax float32 : 0 ; argmax du softmax non unique (p0 == p1) : 0

- Shallow graine 0 (29 sujets) : κ_same(shallow, autres) = 0.241 ; κ_same entre les 4 à la graine 0 = 0.257 ; strates {'n_strata': 116, 'n_dropped': 2, 'trials_lost_share': 0.017241379310344827, 'by_model': {'conformer': {'strata': 1, 'trials_share': 0.008620689655172414}, 'eegnex': {'strata': 2, 'trials_share': 0.017241379310344827}, 'msvtnet': {'strata': 1, 'trials_share': 0.008620689655172414}, 'shallow': {'strata': 1, 'trials_share': 0.008620689655172414}}} ; sans strate valide []

- exclusions Shallow : {}
- physionetmi, eegnex, plafond de graine seul : 101 sujets, κ_seed = 0.130 [0.107, 0.154] ; exclus G0 {'jsonl_absent': 8} ; sans strate valide 0 ; strates écartées 1/202
