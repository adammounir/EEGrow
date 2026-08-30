# Campaign index — what ran, where it lives, and whether it can go in the paper

There are now seven distinct result sets across three checkouts and two machines, and
they were not produced by the same code. Two of them are **invalidated** and still read
as settled findings in documents that predate the diagnosis. This file is the single
place that says which is which; anything not listed here has no provenance and does not
enter a figure.

The rule this file exists to enforce: **a result is identified by its code, not by its
directory**. Three defects were found between 23 and 25 August, each silent, each
producing well-formed CSVs. A campaign is usable only if it ran after all three fixes.

| fix | commit | date | what it silently did before |
|---|---|---|---|
| `drop_last` | `3cefa3a` | 23/08 | 4 cells scored networks that took **zero** gradient steps |
| s=0 abstention | `5337c56` | 25/08 | growth added permanently dead neurons and counted them in `growable_width` / `n_params` |
| `RestoreBestModel` | `0efbdb5` | 25/08 | no checkpoint at all: every score came from the model `patience` epochs *past* its own best (~2 pp) |

And one that is **not** a code fix but a launch-time obligation, because
`config.yaml` deliberately ships the losing values (see `deep4-instability`):

    train.patience=200 train.selection_monitor=valid_acc

Worth up to **+0.13** accuracy on `bd_deep4`, and it *reorders* arms. A campaign
launched without both overrides is undertrained no matter how recent its code.

---

## Invalidated — never cite, never plot

### `benchmarks/results_published/` — the v4 grid, finished 2026-08-10
14 pipelines × 12 datasets × 3 protocols × 5 seeds = **2100 cells**, 189 062 score rows.
Read in `RESULTS.md`, which is why that file now opens with a banner.

**Why it is dead:** predates all three fixes and ran the shipped protocol. Every number
in it is affected by at least one, and the growth pairs by all four.

### `benchmarks/results_v5_published/` — the v5 grid, `eegrow_sha=2d7a2be` (03/08)
15 models × 12 datasets × 3 protocols × 5 seeds = **1848 cells**, 78 585 rows. Cluster
copy: `/scratch/amounir/results_v5` (1848 CSVs). Adds `fix_deepeeg` over v4.

**Why it is dead:** same three fixes, same protocol. Specifically —
- `shin2017a` / `physionetmi` / `alexmi` within_session and `shin2017a` cross_session
  were scored on networks at initialisation (`train_loss` is NaN on 100 % of their
  epochs, which is the direct proof);
- `growable_width` and `n_params` count dead neurons, so the **parameter axis of the
  efficiency figures** — the paper's central claim — is inflated on exactly the arms
  the claim is about, and it is **not** recomputable after the fact because the weights
  were never saved (the `.joblib` is an 8 KB MOABB object);
- fits stopped at a mean of 36.2 epochs against the 200 they were budgeted.

**What it is still good for, and only this:** planning. Its `time` and `epochs` columns
are honest measurements of how long this work takes, and `final_grid.py` builds the
cost model of the final campaign from them. Cost is not a score.

---

## Usable — corrected code and corrected protocol

### Budget × selection square — SLURM 500573, 26/08
`/scratch/amounir/eegrow_budget/grid_models`, 64/64 cells (4 protocol cells × 8 arms ×
2 seeds) plus the original 36-unit `bd_deep4` square. This is the experiment that
*established* the protocol above, so it is the one campaign whose value does not depend
on it. Analysis: `analysis/budget_models.py`, `analysis/deep4_budget.py`.

### Width-matched growth controls — SLURM 500952, 26/08
Same tree, 24/24 cells: `grow_X` vs `fix_X`, same class frozen at the geometry growth
ends on — the only contrast where growth is the sole difference. Verdict: 0/4 pairs
survive Holm, no sign flip. Analysis: `analysis/growth_contrast.py`.

**This no longer justifies dropping the `fix_*` arms, and they are back in the final
grid.** The scope is the problem: 24 cells is **n=9 subjects on bnci2014_001 alone**,
one eval, on hopper cards the turing final grid cannot be paired with, and the best pair
(`grow_sccnet`) sits at p=0.055. "0/4 survive Holm at n=9" is an underpowered null, not
a null — and the paper's framing (growth as architecture search at the price of one
training run) makes `fix_X` the ablation that isolates the contribution. On
`within_session` + `cross_session` it costs ~24 GPU-h with a 3.9 h worst cell, so it
hides entirely under the critical path, and it takes the ablation to ~250 subjects
across 12 datasets.

### Cross-dataset / alignment — jobs 504709, 505009, 505185
`/scratch/amounir/eegrow_xds/benchmarks/results_cross_dataset/cho2017`, **60 CSVs** =
20 complete cells × 3 seeds as of 28/08. Stem:
`{model}__{arm}__{align}__{tier}__seed{n}.csv`. Corrected **code**.
Status tracked in `JOBS_STATUS.md`.

**Read the protocol carefully before reusing a timing from this tree.** `cross_dataset.py`
takes its `train` block verbatim from `config.yaml`, which ships `patience: null` → 20
(verified in the deployed `eegrow_xds/benchmarks/config/config.yaml`). So these cells ran
the *undertrained* protocol, and the striking cost figure they produced — aligned cells
3–5× cheaper, grow `euclidean` 3h19–3h43 against grow `none` ~16h50 — is **early stopping
firing on whitened data and never firing on raw data**. The final grid sets
`patience=200 = max_epochs`, where early stopping cannot fire at all, so both arms run the
full budget and an aligned cell costs exactly what its raw twin costs. Planning the aligned
arm on the 3–5× would have under-budgeted it threefold. The *scores* from this tree are
fine for the corrected-protocol gate decomposition (all 36 `core` cells share one protocol
and one tree); it is the *timings* that do not transfer.

### Euclidean-alignment replication — `/scratch/amounir/eegrow_ea`
Stage 1 gate. `cho2017` passes (+1.51 pp, n=52); `bnci2014_001` is underpowered
(MDE 6.69 pp against a 5.05 pp target) and is a **non-conclusive null, not a null**.
Integration branch `exp/ea-replication`.

---

## Not yet run

### The final grid — `benchmarks/slurm/final_grid.tsv`
1044 cells, ~1007 GPU-h projected, `/scratch/amounir/eegrow_budget`, results tag `final`
(`/scratch/amounir/results_final`). This is the table the paper is written from and it
**replaces v4 and v5 outright**. Rationale for every cut is in `final_grid.py`'s
docstring; the protocol and the checkout are pinned in `final_grid.sbatch`.

---

## Where things live, and why it is not one place

| what | path |
|---|---|
| final campaign | `/scratch/amounir/eegrow_budget` — has all three fixes |
| cross-dataset | `/scratch/amounir/eegrow_xds` — has all three fixes |
| **the August tree** | `/scratch/amounir/eegrow` — has `3cefa3a` but **NOT** `5337c56`. Never pair a cell from here with one from anywhere else, and never launch from it. |
| results, per campaign | `/scratch/amounir/results_<tag>` |
| logs, claims | `/scratch/amounir/logs/pack_<tag>`, `/scratch/amounir/eegrow_claims_<tag>` |

`RESULTS_DIR`, `LOGS` and `CLAIMS` move together as one triplet — sharing a `RESULTS_DIR`
makes the runner skip cells it never ran, and sharing `CLAIMS` is worse still, because
`reap_stale()` cannot reap a claim whose owner is on another host: the cell is silently
never computed and no log says so. `plan_campaign.py --tag` sets all three at once.

All three sit **outside** the checkouts, because the deploy rsync mirrors those with
`--delete`.
