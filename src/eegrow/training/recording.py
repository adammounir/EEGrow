"""Persist what a fit actually did: growth trajectory, learning curve, capacity.

WHY THIS EXISTS
---------------
Nothing downstream can reach ``net.history``. MOABB clones the pipeline for every
fold and never hands the fitted estimator back -- it returns a table of scores. So a
trajectory that is not written from *inside* the fit is lost.

That is not a hypothetical gap. Growth width was only ever passed to ``logger.info``
(``skorch_integration.py``), and the benchmark builds the callback with
``verbose=False``. No run of the published grid recorded how wide its models actually
grew. Which is precisely how ``GrowingShallowFBCSPNet`` and ``GrowingDeepEEGNet``
went a whole campaign without their growth cap being connected at all: there was no
observable that would have shown it.

WHAT IT WRITES
--------------
One JSON object per fit, one per line, appended to ``path``:

    {"fit": 0, "n_train": 4608, "seconds": 91.3, "epochs": 47,
     "width_start": 8, "width_end": 40, "target_width": 40,
     "params_start": 40004, "params_end": 100324,
     "history": [{"epoch": 1, "train_loss": .., "valid_acc": .., "width": 8,
                  "n_params": 40004}, ...]}

One file per cell (the caller derives the path from the cell's stem), so the only
writer is this process and appends cannot interleave -- a JSONL line here is tens of
kilobytes, well past the 4 KiB that ``O_APPEND`` writes atomically.

The recorder is added to the *fixed* arms too. Their width is constant, which is the
point: the learning curves are then comparable, and "growing reaches the same score
for fewer parameter-epochs" becomes measurable instead of asserted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from skorch.callbacks import Callback


def _n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class FitRecorder(Callback):
    """Append one JSON line per fit: per-epoch curve plus start/end capacity.

    Parameters
    ----------
    path : str or Path
        JSONL file to append to. Created (with parents) on first write.
    meta : dict, optional
        Constant fields stamped on every record -- the cell's coordinates
        (eval/dataset/model/seed), so a record is self-describing once the files are
        concatenated.

    Notes
    -----
    Place this callback **after** the growth callback and **before**
    ``EarlyStopping``. After growth, so the width it reads for an epoch is the width
    the epoch ended with; before early stopping, because skorch's ``EarlyStopping``
    raises ``KeyboardInterrupt`` from ``on_epoch_end`` and every callback after it in
    the list is skipped for that final epoch. ``on_train_end`` still runs either way:
    ``NeuralNet.partial_fit`` catches the interrupt and notifies it regardless.
    """

    def __init__(self, path, meta: dict | None = None):
        # Stored VERBATIM, not normalised. Two separate sklearn/skorch contracts bite
        # here, and MOABB clones the pipeline for every fold so both are exercised:
        #
        #   1. skorch's ``Callback.get_params`` returns every attribute whose name does
        #      not end in ``_``, and ``clone`` feeds those back to ``__init__``. A
        #      counter named ``_fit`` therefore arrives as a constructor keyword and
        #      the fit dies with "unexpected keyword argument". Internal state ends
        #      in ``_``; only real hyperparameters do not.
        #   2. ``clone`` then checks that the round-trip returns the *same objects*.
        #      ``self.path = Path(path)`` fails that check with "the constructor
        #      either does not set or modifies parameter path", even though the value
        #      is equivalent. Conversion belongs at the point of use.
        self.path = path
        self.meta = meta
        # Note this counter restarts at 0 on every clone, and MOABB clones per fold --
        # so it does NOT identify a fold across a cell's records. The reader
        # (analysis/growth_io.py) numbers folds by line order instead, which is exact
        # because a cell's file has exactly one writing process.
        self.fit_idx_ = 0

    # ------------------------------------------------------------------ hooks
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.t0_ = time.time()
        self.params0_ = _n_params(net.module_)
        self.width0_ = getattr(net.module_, "growable_width", None)
        self.n_train_ = int(len(X)) if X is not None and hasattr(X, "__len__") else None

    def on_epoch_end(self, net, **kwargs):
        # Recorded into skorch's own history so the values sit next to train_loss and
        # valid_acc and come out of the same place at the end.
        net.history.record("n_params", _n_params(net.module_))
        width = getattr(net.module_, "growable_width", None)
        if width is not None:
            net.history.record("width", int(width))

    def on_train_end(self, net, X=None, y=None, **kwargs):
        keys = ("epoch", "train_loss", "valid_loss", "valid_acc", "width", "n_params",
                "dur")
        curve = [{k: h[k] for k in keys if k in h} for h in net.history]
        record = {
            **(self.meta or {}),
            "fit": self.fit_idx_,
            "n_train": self.n_train_,
            "seconds": round(time.time() - self.t0_, 3),
            "epochs": len(net.history),
            "max_epochs": int(getattr(net, "max_epochs", 0)),
            "width_start": self.width0_,
            "width_end": getattr(net.module_, "growable_width", None),
            "target_width": getattr(net.module_, "target_width", None),
            "params_start": self.params0_,
            "params_end": _n_params(net.module_),
            "history": curve,
        }
        self.fit_idx_ += 1
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
