# BNCI2014_001 within-session benchmark (SCCNet)

MOABB `WithinSessionEvaluation` (5-fold, both sessions), 4-class motor imagery, subjects 1-9. Growable vs fixed baselines, all via braindecode `EEGClassifier`. Width 4->16, 30 epochs, grow_every=5. Score = accuracy.

Per-subject accuracy (mean over sessions x CV folds):

| subject | fixed-small | growable | fixed-target |
|---|---|---|---|
| S1 | 0.708 | 0.743 | 0.741 |
| S2 | 0.470 | 0.444 | 0.559 |
| S3 | 0.710 | 0.833 | 0.823 |
| S4 | 0.510 | 0.557 | 0.592 |
| S5 | 0.316 | 0.372 | 0.411 |
| S6 | 0.418 | 0.458 | 0.507 |
| S7 | 0.609 | 0.757 | 0.755 |
| S8 | 0.688 | 0.755 | 0.775 |
| S9 | 0.758 | 0.830 | 0.835 |
| **mean (over subjects)** | **0.577 ± 0.145** | **0.639 ± 0.170** | **0.666 ± 0.144** |

_Spread is across subjects. 9 subjects x 2 sessions x 5-fold CV. Generated in 399s._

