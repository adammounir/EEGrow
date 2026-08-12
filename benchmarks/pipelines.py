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
from sklearn.base import BaseEstimator, TransformerMixin


def to_float32(X):
    """MOABB serves float64 epochs; braindecode/skorch convs want float32.

    The classic ML arms keep float64 (pyriemann/CSP are happier in double), so the
    cast only sits in the deep pipelines.
    """
    return np.asarray(X, dtype="float32")


class RMSScaler(BaseEstimator, TransformerMixin):
    """Divide by the training set's global RMS, so the net sees an O(1) input.

    A convnet handed volt-scale EEG (std ~5e-6) collapses: on one BNCI2014_001 fold,
    40 epochs, everything else fixed, ShallowFBCSPNet predicts one class for 99.7% of
    trials and scores 0.504 acc / 0.539 auc, against 0.649 / 0.733 at unit RMS, with the
    initial training loss falling from 9.71 to 1.04. BatchNorm does not make this go away:
    it makes the *forward* pass scale-free, but AdamW's weight decay still pulls against
    the ~1e6 first-layer weights that volt-scale inputs require, so the optimisation is not
    scale-free even when the architecture is.

    On *this* path the scaler is nearly a no-op, and that is measured rather than assumed.
    MOABB's array branch already multiplies by ``dataset.unit_factor = 1e6``, so the
    paradigm hands over microvolts and the nets were never in the dead regime. Re-running
    the grid's deep arms with this step added and comparing fold by fold against the
    original CSVs -- 8871 paired folds, BNCI2014-001 and Shin2017A, 8 architectures --
    gives a mean difference of **-0.002**. It is kept anyway so the deep arms do not
    silently depend on MOABB's unit default, and because ``pool.py`` reaches for Epochs and
    so does *not* get that factor (see ``pool.TARGET_RMS``).

    Only the deep arms carry it. The classic arms are scale-invariant by construction --
    CSP solves a generalised eigenproblem, the Riemannian arms work on covariances -- so
    adding it there could only introduce noise.

    The constant is fitted on train and applied to test, so no test statistic leaks: it is
    one scalar, exactly as MOABB's own scalers behave.
    """

    def __init__(self, target: float = 1.0):
        self.target = target

    def fit(self, X, y=None):
        X = np.asarray(X)
        rms = float(np.sqrt(np.mean(X.astype(np.float64) ** 2)))
        self.scale_ = self.target / rms if rms > 0 else 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype="float32") * np.float32(self.scale_))


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
              device, seed):
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
        train_split=ValidSplit(0.2, random_state=seed, stratified=True),
        device=device,
        callbacks=callbacks,
        classes=list(range(n_outputs)),
        verbose=0,
    )
    # ``rms`` before ``clf``: MOABB serves volts and the net cannot train on them. It is a
    # no-op on the cross-dataset path, where pool.load already delivers TARGET_RMS = 1.
    return Pipeline([("cast", FunctionTransformer(to_float32)),
                     ("rms", RMSScaler()), ("clf", clf)])


def build_pipeline(model_cfg, train_cfg, *, n_chans, n_times, n_outputs, sfreq,
                   device, seed):
    """Dispatch on ``model_cfg.kind`` -> a fresh sklearn ``Pipeline``."""
    if model_cfg["kind"] == "ml":
        return _ml_builders()[model_cfg["name"]]()
    return _build_dl(model_cfg, train_cfg, n_chans=n_chans, n_times=n_times,
                     n_outputs=n_outputs, sfreq=sfreq, device=device, seed=seed)
