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

    # Instrumentation goes in FIRST, ahead of the growth callback. See the order
    # contract below: `eps` must observe the optimizer that trained the epoch, and
    # growth replaces it. (These were previously appended after `gromo`, which the
    # comment below already described as wrong without the code agreeing.)
    from eegrow.training.callbacks import (AdamEpsDominance, GradientNorm,
                                           RestoreBestModel, StopReason)
    callbacks = [
        EpochScoring("accuracy", on_train=False, name="valid_acc",
                     lower_is_better=False),
        ("grad", GradientNorm()),
        ("eps", AdamEpsDominance()),
    ]
    if kind == "growing":
        from eegrow import GromoGrowth
        from eegrow.training.loop import MIN_SINGULAR_RATIO
        # Growth runs on the training device for cuda/cpu; only MPS hops to CPU.
        module_params["module__device"] = device
        callbacks.append(("gromo", GromoGrowth(
            grow_every=model_cfg["grow_every"],
            # How many neurons a growth step may propose, as a floor relative to the
            # best candidate's singular value. Per-model because the right value is an
            # arm's spectrum shape, not a global constant -- see loop.grow_step.
            min_singular_ratio=float(model_cfg.get("min_singular_ratio",
                                                   MIN_SINGULAR_RATIO)),
            verbose=False)))
    # ORDER IS THE CONTRACT HERE, and each position is load-bearing:
    #
    #   grad      instrumentation, anywhere before `record`.
    #   eps       instrumentation, but STRICTLY BEFORE gromo. Growth rebuilds the
    #             optimizer, and the neurons it splices in carry no second-moment
    #             accumulator yet (v=0 => attenuation 0 on every new coordinate). Read
    #             after growth, the metric would report a total eps collapse that is
    #             nothing but a fresh moment estimate -- i.e. it would manufacture
    #             exactly the finding it exists to test. Before growth, the number
    #             describes the optimizer that actually trained the epoch.
    #   gromo     grows the module.
    #   restore   snapshots/restores the module. AFTER gromo so a snapshot is the
    #             epoch's end state; BEFORE record so the record describes the model
    #             that is actually returned, which after a restore is not the last
    #             epoch of the curve written next to it.
    #   stop      writes stop_reason_, which record reads -> must precede it.
    #   record    writes the JSONL line.
    #   early     LAST. It raises KeyboardInterrupt from on_epoch_end, so every
    #             callback after it is skipped for that final epoch.
    #
    # The instrumentation callbacks are added to the fixed arms too: their width is
    # constant, which is what makes the curves and the selection comparable.

    # Which epoch's model gets scored, and which signal ends the fit. Two separate
    # decisions, deliberately not fused, and only one of them is settled.
    #
    # SETTLED: it must not be the last epoch. Early stopping ends a fit exactly
    # `patience` epochs after its best BY CONSTRUCTION -- v5 measured
    # `epochs - epoch_of_best` = 20 with std 0.0 on all 140 490 folds -- so every score
    # this benchmark ever published came from a model 20 epochs past its own optimum.
    #
    # SETTLED: `valid_acc` is a bad *stopping* signal here. The internal split is tiny
    # (46 trials on bnci2014_001, 3 on shin2017a), so accuracy moves in steps of
    # 1/n_valid = 0.0213 while skorch's relative threshold is 1e-4 of ~0.7 = 7e-05 --
    # 300x smaller than the smallest step the metric can take. The tolerance is not
    # loosely set, it is inoperative, and the run dies when a quantised metric fails to
    # beat an early lucky peak. On a continuous loss the same threshold works as
    # designed. Patience is unchanged: it was never the problem, the criterion was.
    #
    # SETTLED SINCE, and the other way: the *selection* must NOT be on the loss. The
    # 2x2 square (`exp_deep4_budget.py`, 36 paired units) has to be read as a square,
    # because the two knobs interact (+0.0854, p=1.1e-07) and neither effect can be
    # quoted alone::
    #
    #                       sel=valid_loss   sel=valid_acc
    #     patience  20          0.3501           0.3390
    #     patience 200          0.4074           0.4818
    #
    # The two full-budget cells train identically and differ only in the epoch handed
    # back -- `valid_loss` restores epoch 5, the accuracy peak is at epoch 158. So the
    # loss is a broken proxy on a 46-trial split as a SELECTION rule too, for the
    # reason already named above: confident-and-wrong trials blow up a mean loss long
    # before they touch an argmax of accuracy. Generalised across the other 8 arms
    # (SLURM 500573, `analysis/budget_models.py`): the budget helps all 9 without
    # exception, selection alone is noisy, and 6 of 9 arms were undertrained by the
    # shipped default at subject level.
    #
    # The default below is still `valid_loss` ONLY because flipping it re-dates every
    # result in `results_v5_published/`; the corrected protocol is passed explicitly
    # (`train.patience=200 train.selection_monitor=valid_acc`) until that migration is
    # done. Do not read the default as the recommendation -- it is the opposite.
    #
    # For a growing model there is a second, structural catch: the best epoch can be a
    # NARROWER model than the last one (grow_shallow restored width 19 against a final
    # 25), so selection interacts with the growth story and cannot be chosen on
    # validation resolution alone.
    monitor = str(train_cfg.get("selection_monitor", "valid_loss"))
    lower_is_better = monitor.endswith("loss")
    callbacks.append(("restore", RestoreBestModel(monitor=monitor,
                                                  lower_is_better=lower_is_better)))
    callbacks.append(("stop", StopReason()))
    if record_path is not None:
        from eegrow import FitRecorder
        callbacks.append(("record", FitRecorder(
            record_path, meta={"model": model_cfg.get("label"), "seed": seed})))
    # LR schedule, AFTER every instrumentation callback and before EarlyStopping.
    # The position is the same argument as `eps` above: `grad` and `eps` must describe
    # the optimizer that trained the epoch just finished, and stepping the scheduler
    # first would have them report the *next* epoch's step size.
    schedule = train_cfg.get("lr_schedule") or None
    if schedule is not None:
        if kind == "growing":
            # Not a limitation worth hiding behind a warning. skorch's LRScheduler
            # resolves `net.optimizer_` once, in on_train_begin; GromoGrowth calls
            # initialize_optimizer() at every growth step, so from the first growth
            # onward the scheduler would anneal an optimizer that no longer trains
            # anything -- silently, and with a perfectly ordinary curve to show for it.
            # Growth-aware rebinding is a separate piece of work; until it exists,
            # refuse rather than publish a schedule that stopped applying at epoch 10.
            raise NotImplementedError(
                f"train.lr_schedule={schedule!r} is not supported on growing arms: "
                "GromoGrowth rebuilds the optimizer and skorch's LRScheduler would "
                "keep annealing the discarded one. Fixed arms only for now.")
        if schedule != "cosine":
            raise ValueError(f"unknown train.lr_schedule={schedule!r} (null|cosine)")
        from skorch.callbacks import LRScheduler
        callbacks.append(("lrsched", LRScheduler(
            policy=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=int(train_cfg["max_epochs"]), eta_min=0.0,
            step_every="epoch")))

    stop_monitor = str(train_cfg.get("stop_monitor", "valid_loss"))
    # `patience` is a knob rather than a derived constant because it is the quantity
    # that decides how much of the budget a fit actually gets, and on `bd_deep4` that
    # turned out to be the whole story: measured on bnci2014_001 within, subject 1, its
    # validation loss bottoms at epoch 4 and then explodes 1.11 -> 3.27 while the TRAIN
    # loss is still falling, so patience 20 ends the run at epoch 24. Setting it >=
    # `max_epochs` disables early stopping outright (skorch needs `patience` misses in
    # a row and there are only `max_epochs - 1` epochs after the first), which is the
    # only way to measure what the net would have reached if simply left to train.
    # null => the historical default, so every existing config keeps its behaviour.
    patience = train_cfg.get("patience")
    patience = (max(15, int(train_cfg["max_epochs"]) // 10) if patience is None
                else int(patience))
    callbacks.append(
        EarlyStopping(patience=patience,
                      monitor=stop_monitor,
                      lower_is_better=stop_monitor.endswith("loss")))

    clf = EEGClassifier(
        module=module,
        **module_params,
        criterion=torch.nn.CrossEntropyLoss,  # bd 1.x + eegrow models output logits
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(train_cfg["lr"]),
        # Torch's default (1e-8), stated rather than inherited. It is the parameter
        # Stella's hypothesis is about, so it has to be settable to be testable at all
        # -- and a default that is written down is a default someone can argue with.
        # `AdamEpsDominance` measures whether it actually bites; do not raise it on
        # the strength of the hypothesis alone.
        optimizer__eps=float(train_cfg.get("optimizer_eps", 1e-8)),
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
