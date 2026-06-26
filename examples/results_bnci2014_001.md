# BNCI2014_001 within-subject benchmark (sccnet)

Growable vs fixed baselines via braindecode `EEGClassifier`. Subjects 1-9, 30 epochs, grow_every=5, width 4->16. Single split/seed per subject.

Test accuracy per subject:

| subject | fixed-small | growable | fixed-target |
|---|---|---|---|
| S1 | 0.792 | 0.840 | 0.833 |
| S2 | 0.479 | 0.542 | 0.528 |
| S3 | 0.764 | 0.826 | 0.903 |
| S4 | 0.708 | 0.701 | 0.785 |
| S5 | 0.604 | 0.618 | 0.646 |
| S6 | 0.396 | 0.340 | 0.465 |
| S7 | 0.743 | 0.854 | 0.882 |
| S8 | 0.736 | 0.771 | 0.812 |
| S9 | 0.688 | 0.785 | 0.785 |
| **mean** | **0.657 ± 0.129** | **0.698 ± 0.161** | **0.738 ± 0.147** |
