# BNCI2014_001 within-subject benchmark (sccnet)

Growable vs fixed baselines via braindecode `EEGClassifier`. Subjects 1-9, seeds 0-2 (9x3 = 27 runs/arm), 30 epochs, grow_every=5, width 4->16.

Per-subject test accuracy (mean ± std **over seeds**):

| subject | fixed-small | growable | fixed-target |
|---|---|---|---|
| S1 | 0.734 ± 0.077 | 0.792 ± 0.046 | 0.803 ± 0.053 |
| S2 | 0.477 ± 0.003 | 0.528 ± 0.025 | 0.553 ± 0.046 |
| S3 | 0.780 ± 0.012 | 0.847 ± 0.021 | 0.877 ± 0.027 |
| S4 | 0.639 ± 0.050 | 0.653 ± 0.040 | 0.718 ± 0.060 |
| S5 | 0.488 ± 0.089 | 0.505 ± 0.081 | 0.583 ± 0.045 |
| S6 | 0.412 ± 0.012 | 0.447 ± 0.079 | 0.502 ± 0.031 |
| S7 | 0.729 ± 0.025 | 0.863 ± 0.013 | 0.898 ± 0.018 |
| S8 | 0.750 ± 0.011 | 0.806 ± 0.031 | 0.817 ± 0.007 |
| S9 | 0.713 ± 0.041 | 0.799 ± 0.015 | 0.803 ± 0.021 |
| **mean (over subjects)** | **0.636 ± 0.131** | **0.693 ± 0.153** | **0.728 ± 0.139** |

Mean run-to-run std (over seeds, averaged over subjects) — lower is more stable:

| fixed-small | growable | fixed-target |
|---|---|---|
| 0.036 | 0.039 | 0.034 |
