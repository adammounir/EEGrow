"""Build one benchmark pipeline from a Hydra model config.

Three families share the same scikit-learn ``Pipeline`` interface, so MOABB's
evaluation can treat them identically:

* ``kind: ml``       -- classic Riemannian / CSP baselines (CPU, no GPU);
* ``kind: bd``       -- a stock braindecode reference net via ``EEGClassifier``;
* ``kind: growing``  -- an eegrow growable net via ``EEGClassifier`` + ``GromoGrowth``.

Why one factory? Bruno's spec is "ML baselines + braindecode reference + growing",
benchmarked through the *same* MOABB protocol. Keeping the construction in one place
guarantees the three arms differ only where they should (the estimator), not in
training hygiene (loss, optimiser, early stopping, the float32 cast).

braindecode 1.x models *and* the eegrow growables both output **raw logits**, so every
deep arm uses ``CrossEntropyLoss`` -- no NLLLoss/LogSoftmax mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_float32(X):
    """MOABB serves float64 epochs; braindecode/skorch convs want float32.

    The classic ML arms keep float64 (pyriemann/CSP are happier in double), so the
    cast only sits in the deep pipelines.
    """
    return np.asarray(X, dtype="float32")


# --------------------------------------------------------------------- ML arms
def _ml_builders() -> dict[str, Any]:
    """The classic MOABB motor-imagery baselines (all CPU).

    Returns a name -> (callable building a fresh sklearn pipeline) map. A fresh
    pipeline per call matters: MOABB clones per fold, so the builder must be stateless.
    """
    from mne.decoding import CSP
    from pyriemann.classification import MDM, FgMDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    return {
        "csp_lda": lambda: make_pipeline(
            CSP(n_components=8), LDA(solver="lsqr", shrinkage="auto")),
        "csp_svm": lambda: make_pipeline(
            CSP(n_components=8), SVC(kernel="rbf")),
        "ts_lr": lambda: make_pipeline(
            Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=1000)),
        "ts_svm": lambda: make_pipeline(
            Covariances("oas"), TangentSpace(), SVC(kernel="rbf")),
        "mdm": lambda: make_pipeline(Covariances("oas"), MDM()),
        "fgmdm": lambda: make_pipeline(Covariances("oas"), FgMDM()),
    }


# --------------------------------------------------------------------- DL arms
def _build_dl(model_cfg, train_cfg, *, n_chans, n_times, n_outputs, sfreq,
              device, seed, record_path=None):
    """A braindecode reference (``kind: bd``) or eegrow growable (``kind: growing``).

    The architecture comes from ``model_cfg.arch`` (resolved on ``braindecode.models``
    or ``eegrow``); the training knobs (lr, epochs, batch) come from the shared
    ``train`` config, so a smoke test only needs ``train.max_epochs=2`` on the CLI.
    """
    import torch
    from braindecode import EEGClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from skorch.callbacks import EarlyStopping, EpochScoring
    from skorch.dataset import ValidSplit

    kind = model_cfg["kind"]
    if kind == "growing":
        import eegrow
        module = getattr(eegrow, model_cfg["arch"])
    else:
        import braindecode.models as bdm
        module = getattr(bdm, model_cfg["arch"])

    module_params = {
        "module__n_chans": n_chans,
        "module__n_outputs": n_outputs,
        "module__n_times": n_times,
    }
    if model_cfg.get("needs_sfreq"):
        module_params["module__sfreq"] = sfreq
    for k, v in (model_cfg.get("module_kwargs") or {}).items():
        module_params[f"module__{k}"] = v

    callbacks = [
        EpochScoring("accuracy", on_train=False, name="valid_acc",
                     lower_is_better=False),
    ]
    if kind == "growing":
        from eegrow import GromoGrowth
        # Growth runs on the training device for cuda/cpu; only MPS hops to CPU.
        module_params["module__device"] = device
        callbacks.append(
            ("gromo", GromoGrowth(grow_every=model_cfg["grow_every"], verbose=False)))
    # After growth (so an epoch's recorded width is the width it ended with) and
    # before EarlyStopping (which raises KeyboardInterrupt from on_epoch_end, skipping
    # every callback listed after it on that final epoch). Added to the fixed arms
    # too: their width is constant, which is what makes the curves comparable.
    if record_path is not None:
        from eegrow import FitRecorder
        callbacks.append(("record", FitRecorder(
            record_path, meta={"model": model_cfg.get("label"), "seed": seed})))
    callbacks.append(
        EarlyStopping(patience=max(15, int(train_cfg["max_epochs"]) // 10),
                      monitor="valid_acc", lower_is_better=False))

    clf = EEGClassifier(
        module=module,
        **module_params,
        criterion=torch.nn.CrossEntropyLoss,  # bd 1.x + eegrow models output logits
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(train_cfg["lr"]),
        max_epochs=int(train_cfg["max_epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        # EEGClassifier defaults to drop_last=True on the *training* iterator. With
        # batch_size=64, any cell whose skorch train split (0.8 x n_train) holds fewer
        # than 64 trials yields ZERO batches -- not a small last batch, no batch at all
        # -- so the net never takes a gradient step and the published score is its
        # initialisation. It is silent: no exception, no warning, a well-formed CSV.
        # Measured on results_v5: within_session shin2017a (13 trials in the split),
        # physionetmi (29), alexmi (38) and cross_session shin2017a (32) all had
        # train_loss = NaN on 100% of epochs. The growing arms never grew there either
        # (GromoGrowth lists the train iterator, gets [], and returns before grow_step),
        # which is a consequence of this and not a property of gromo.
        # False keeps the short final batch; BatchNorm tolerates it (a batch of 1 would
        # be the exception, and no split here lands on 8k+1 trials).
        iterator_train__drop_last=False,
        train_split=ValidSplit(0.2, random_state=seed, stratified=True),
        device=device,
        callbacks=callbacks,
        classes=list(range(n_outputs)),
        verbose=0,
    )
    return Pipeline([("cast", FunctionTransformer(to_float32)), ("clf", clf)])


def build_pipeline(model_cfg, train_cfg, *, n_chans, n_times, n_outputs, sfreq,
                   device, seed, record_path=None):
    """Dispatch on ``model_cfg.kind`` -> a fresh sklearn ``Pipeline``.

    ``record_path`` (deep arms only) is a JSONL file the fit appends its per-epoch
    curve and growth trajectory to; see ``eegrow.training.recording``. The ML arms
    have no epochs and no width, so it does not apply to them.
    """
    if model_cfg["kind"] == "ml":
        return _ml_builders()[model_cfg["name"]]()
    return _build_dl(model_cfg, train_cfg, n_chans=n_chans, n_times=n_times,
                     n_outputs=n_outputs, sfreq=sfreq, device=device, seed=seed,
                     record_path=record_path)
