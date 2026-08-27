# MOABB grid results — classical ML vs braindecode vs growing

Grid finished 2026-08-10. This document is the reading of the results; the numbers
themselves are in `results_published/`, and everything below follows from them via
`aggregate_published.py`.

## What ran

| | |
|---|---|
| datasets | 12 (MOABB motor imagery, known electrode positions) |
| protocols | `within_session` (12 datasets), `cross_session` (6, those with ≥ 2 sessions), `cross_subject` (12) |
| pipelines | 6 Riemann/CSP, 4 fixed braindecode, 4 growing counterparts |
| seeds | 5 per cell |
| runs | **2100** (14 × 5 × 30 protocol×dataset cells), no missing cell |
| scores | **189,062** rows (one per subject × session × seed × model) |
| sampling | 250 Hz throughout, epoch window imposed per dataset by MOABB |

The compared pairs are `grow_shallow`/`bd_shallow`, `grow_sccnet`/`bd_sccnet`,
`grow_eegnex`/`bd_eegnex`, `grow_deep`/`bd_deep4`: same target architecture, same
preprocessing, the only difference being that the growing arm starts narrow and widens
during training.

**Statistical convention.** The 5 seeds are replicates of one measurement, not 5
independent samples. They are therefore averaged *within* each (subject, session) before
any statistic; one observation = one held-out subject/session. Without that the effective
n is multiplied by five and every p-value is wrong. The metric follows MOABB: ROC-AUC on
the two-class `LeftRightImagery` datasets (BNCI2014-004, Cho2017, Lee2019-MI, PhysionetMI,
Shin2017A, Weibo2014), accuracy elsewhere — scores are never pooled across the two.

## 1. Riemannian ML is still ahead, except cross-subject

Mean absolute level per family over the 6 AUC datasets (the only ones comparable to each
other):

| family | `within_session` | `cross_session` | `cross_subject` |
|---|---|---|---|
| Riemann/CSP | **0.7135** | **0.7603** | 0.7347 |
| braindecode (fixed) | 0.6296 | 0.6487 | **0.7837** |
| growing | 0.6139 | 0.6438 | 0.7636 |

What this means is the data regime. In `within_session` and `cross_session` training runs
on a few hundred trials from a single subject: a Riemannian classifier, whose inductive
bias already encodes the signal's spatial covariance structure, beats a network that has
to learn it. In `cross_subject` the network sees dozens of subjects and the order
reverses — that is where, and only where, deep learning pays off. Counted per cell rather
than averaged, the best model across all families in `within_session` is Riemannian on 10
of the 12 datasets, and it is `ts_lr` (tangent space + logistic regression) on 8 of them.

## 2. Growing: one of four pairs wins, and modestly

Sign test on the (growing − fixed) deltas at the (protocol, dataset) level:

| pair | positive cells | median delta | p (sign) |
|---|---|---|---|
| `grow_shallow` vs `bd_shallow` | **24/27** | **+0.0066** | 4.9 × 10⁻⁵ |
| `grow_sccnet` vs `bd_sccnet` | 11/27 | −0.0066 | 0.44 |
| `grow_deep` vs `bd_deep4` | 6/27 | −0.0305 | 5.9 × 10⁻³ |
| `grow_eegnex` vs `bd_eegnex` | 2/27 | −0.0322 | 5.7 × 10⁻⁶ |

(27 rather than 30: see §3, three cells are at chance and are set aside. Including them:
27/30, +0.0065, p = 8.4 × 10⁻⁶ — the conclusion does not depend on the choice.)

So the effect on ShallowFBCSPNet is **real, systematic and small**: +0.7 point median,
but in the same direction almost everywhere, which is what the sign test measures. The
clearest cells are `lee2019_mi/within_session` (+0.0307 AUC, n = 108, p < 10⁻⁴),
`lee2019_mi/cross_session` (+0.0289), `cho2017/within_session` (+0.0256, n = 52).

On the other three architectures the fixed reference is better, and for EEGNeX it is
decisive (2 cells out of 27). The most economical reading: growing helps when capacity is
the limiting factor and the target network is small; on an already wide architecture,
starting narrow only spends epochs.

**A trap not to report backwards.** The largest positive deltas in the whole grid belong
to `grow_deep` and are not wins for growing:

| cell | growing | fixed | chance |
|---|---|---|---|
| `bnci2014_001` / `within_session` | 0.4080 | 0.2748 | 0.2500 |
| `bnci2014_001` / `cross_session` | 0.4192 | 0.3028 | 0.2500 |

`bd_deep4` sits 2.5 points above chance there, which is to say it does not train. A
+0.13 delta against a collapsed reference says Deep4 fails on those 9 subjects, not that
growing is worth thirteen points. That is why the absolute levels and the distance to
chance ship in `eegrow_benchmark_levels.csv` next to the deltas.

## 3. Three cells where no network learns

On `physionetmi/within_session`, `shin2017a/within_session` and
`shin2017a/cross_session`, **all 8 deep models** (4 fixed + 4 growing) sit at AUC
0.49–0.51, i.e. exactly chance. The Riemannian baselines, meanwhile, take off (`fgmdm` at
0.681 on `physionetmi/within_session`). So this is not an impossible dataset but a regime
where the networks have too few trials per session. Any paired delta computed in those
cells is noise between two coin flips, which is why the table in §2 sets them aside.

## 4. Provenance: what is provable and what no longer is

This has to be said before anyone cites these numbers.

A pair only measures growth if both its arms saw the same preprocessing. That was not
guaranteed: the production launch passed `dataset.resample=250` on the command line, some
retry scripts omitted it, and a relaunched cell silently fell back to the dataset's
native rate (500 Hz on Schirrmeister2017, 1000 Hz on Lee2019-MI). `regime_guard.py` is
the guard against that, and the `fix250` campaign (116 cells, `slurm/fix250_*.txt`) was
the remediation.

The guard read its evidence from hydra's run records, and those **no longer exist**: the
`rsync --delete` that erased the epoch cache took `benchmarks/multirun/` and
`benchmarks/outputs/` with it. Only the slurm logs and the result files remain.
`provenance_audit.py` gives the exact count (raw arm, 2100 cells):

| | cells | |
|---|---|---|
| native rate already 250 Hz → `resample` is a no-op, immune by construction | 630 | 30.0 % |
| certified at 250 Hz by a surviving log (27 already counted above) | 327 | — |
| **established, union of the two** | **930** | **44.3 %** |
| no trace, in either direction | 1170 | 55.7 % |

Two positives in that table. **No** cell is certified at any rate other than 250 Hz —
where the evidence survives, it always points the same way. And the **87** cells that can
be shown to have run at a native rate at some point are **all 87** certified at 250 Hz
today: the relaunch campaign did rewrite the bytes on disk. Lee2019-MI, the most exposed
dataset (factor 4), is certified on all 210 of its cells.

What remains is therefore ignorance, not known contamination — but it is ignorance, over
56 % of the cells, and no amount of analysis will close it. The guard itself refuses those
cells (`UNKNOWN` ≠ 250): the analyses above were produced with `EEGROW_ALLOW_MIXED=1`,
which is an explicit derogation and not an oversight.

**One approach that does not work**, documented so it is not retried: inferring the rate
from fit time. A convnet's cost is roughly linear in the number of input samples, so a
seed that ran at the native rate should stand out by a factor of native/250. The control
refutes the measurement: `lee2019_mi/cross_subject/bd_sccnet` has all five seeds certified
at 250 Hz and its per-seed median cost still spans 43.5 s to 242.3 s — a factor of 5.6,
larger than the 4.0 a genuine 1000 Hz contamination would produce. Early stopping and
heterogeneous GPUs dominate the cost; no threshold on that ratio can serve as a gate.

**Restoring provenance** means relaunching untraced cells, with the `sfreq` column now
written into every CSV and `resample: 250.0` pinned in the 12 dataset configs (commit
`8812860`), which makes the failure impossible to reproduce. The cost is not uniform, so
it is worth pricing the scopes separately. Hours below are the sum of the recorded fit
time over the cells to relaunch — sequential compute, not wall time on a slurm array:

| scope | untraced cells | compute |
|---|---|---|
| `bd_shallow` + `grow_shallow` — the one pair with a positive result | 173 | 70 h GPU |
| the 8 deep arms — every paired claim in §2 | 630 | 427 h GPU |
| the 6 Riemann/CSP pipelines — the levels in §1 | 540 | 2425 h CPU |
| everything | 1170 | 427 h GPU + 2425 h CPU |

Within the shallow pair, 60 of the 173 cells account for 65 of the 70 hours (the
`cross_subject` protocol on physionetmi, cho2017, shin2017a, schirrmeister2017,
bnci2015_001); the remaining 113 cost about 5 hours in total. The decision is which of
those scopes to buy, and it has to be taken before submission rather than after.

## Files

| file | contents |
|---|---|
| `results_published/eegrow_benchmark_all_scores.csv.gz` | long table, 189,062 scores, one per (protocol, dataset, subject, session, model, seed) — nothing is averaged |
| `results_published/eegrow_benchmark_levels.csv` | absolute levels per (protocol, dataset, model), seeds averaged within subject/session |
| `results_published/eegrow_benchmark_paired.csv` | the 120 paired growing − fixed contrasts, with Wilcoxon and fraction of positive deltas |
| `results_published/eegrow_moabb_grid_results.xlsx` | the same numbers as one spreadsheet, tables only — design, family means, sign tests, paired, levels, best model per cell, cells at chance, provenance |
| `aggregate_published.py` | produces the three files above from `results/` |
| `make_results_workbook.py` | produces the spreadsheet from the published CSVs |
| `provenance_audit.py` | the table in §4, reproducible |
| `where_grow_wins.py`, `where_grow_wins2.py` | the detailed paired analyses (§2) |
| `regime_guard.py` | the preprocessing guard |

The raw per-cell CSVs (2160 files) stay on the cluster, under
`benchmarks/results/<protocol>/<dataset>/<model>__seed<N>.csv`. `results_published/` is
their versioned aggregate, sufficient to redo the whole analysis without access to
Margaret.
