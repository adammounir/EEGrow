# MOABB benchmark (Hydra)

Config-driven benchmark of three pipeline families through MOABB's evaluation
protocols, parallelised **at the pipeline level**: one job = one
`(eval x dataset x model x seed)` point, expanded by `--multirun` and run
concurrently by a launcher (joblib locally, submitit/SLURM on Margaret).

* **ML baselines** (`model=ml_*`, CPU): CSP+LDA, CSP+SVM, TS+LR, TS+SVM, MDM, FgMDM.
* **braindecode references** (`model=bd_*`): ShallowFBCSPNet, SCCNet, EEGNeX, Deep4Net.
* **growing** (`model=grow_*`): the eegrow growables (`EEGClassifier` + `GromoGrowth`).

Both regimes come from one harness: `eval=within_session` (personal calibration) and
`eval=cross_subject` (leave-one-subject-out, where growth is expected to help most);
`eval=cross_session` is also available.

## Install

```bash
pip install -e ".[benchmark]"
```

## Run

Local smoke test (one job, 2 epochs, no GPU needed):

```bash
python benchmarks/run_moabb_hydra.py \
    model=grow_sccnet dataset=bnci2014_001 eval=within_session train.max_epochs=2
```

Local parallel sweep (all models, both protocols, 3 seeds):

```bash
python benchmarks/run_moabb_hydra.py -m hydra/launcher=joblib \
    model=glob(*) dataset=bnci2014_001 eval=within_session,cross_subject seed=0,1,2
```

Cluster sweep (Margaret, SLURM `tau` partition — edit `config/hydra/launcher/tau.yaml`
for your venv/module and resources first):

```bash
python benchmarks/run_moabb_hydra.py -m hydra/launcher=tau \
    model=glob(*) dataset=glob(*) eval=within_session,cross_subject seed=0,1,2 \
    data_dir=/data/tau/iceberg_1/titanic_1/datasets/mne_data
```

## Aggregate

Each job writes `results/<eval>/<dataset>/<model>__seed<k>.csv`. Then:

```bash
python benchmarks/aggregate.py            # -> results/summary.{md,csv}
```

## Add a dataset / model

Drop a YAML in `config/dataset/` (set `moabb_class`, `paradigm`; `sfreq` is inferred
from the epoch length, only set `dataset.sfreq` to force an override, and `resample`
to actually downsample) or `config/model/` (`kind: ml|bd|growing`, a `label`, the
arch + widths). `glob(*)` picks it up automatically.
