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

import hashlib
import json
import time
import uuid
from pathlib import Path

from skorch.callbacks import Callback


def _n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _optimizer_stamp(net) -> dict | None:
    """The optimizer hyperparameters this fit actually ran under.

    Reconstructable from the config *for the run that produced it*, and not
    reconstructable at all once records from several campaigns sit in one frame -- a
    curve then carries no evidence of the learning rate or weight decay behind it.
    ``eps`` in particular is the parameter under active discussion (whether AdamW's
    default 1e-8 is too small for gradients this size), and it has never been set
    explicitly, which is exactly the kind of thing a reader assumes was recorded.

    Read off the live optimizer rather than the config, so it reports what ran and not
    what was requested. The scalar fields are the DEFAULT group's: the last one, where
    skorch puts every parameter no ``param_groups`` pattern claimed (the only group
    before ``train.no_decay_params``). With several groups, ``param_groups`` lists each
    one's tensor/parameter count and weight decay, so no group is reported for another.
    """
    opt = getattr(net, "optimizer_", None)
    if opt is None or not getattr(opt, "param_groups", None):
        return None
    group = opt.param_groups[-1]
    stamp = {"class": type(opt).__name__,
             "n_param_groups": len(opt.param_groups),
             "batch_size": getattr(net, "batch_size", None),
             "criterion": type(getattr(net, "criterion_", None)).__name__}
    for key in ("lr", "weight_decay", "eps", "betas", "momentum", "amsgrad"):
        if key in group:
            value = group[key]
            stamp[key] = list(value) if isinstance(value, tuple) else value
    if len(opt.param_groups) > 1:
        stamp["param_groups"] = [
            {"n_tensors": len(g["params"]),
             "n_params": int(sum(p.numel() for p in g["params"])),
             "weight_decay": g.get("weight_decay")} for g in opt.param_groups]
    return stamp


def _selection_stamp(net) -> dict | None:
    """Which epoch selection and which stopping rule this fit actually ran under.

    Same argument as :func:`_optimizer_stamp`, with a sharper edge: the corrected
    protocol is launch overrides (``patience``, ``selection_monitor``,
    ``smooth_window``) whose config defaults are the historical, losing values. A grid
    launched without them writes well-formed records indistinguishable from a correct
    one -- unless the record says what ran. Read off the live callbacks, not the config.

    Duck-typed (``tie_break`` / ``patience`` attributes) rather than ``isinstance`` so
    this module does not import ``callbacks``. ``None`` fields mean the callback was not
    in the list, which is itself worth recording.
    """
    cbs = getattr(net, "callbacks_", None) or []
    stamp = {"selection_monitor": None, "lower_is_better": None, "tie_break": None,
             "smooth_window": None, "swap": None, "stop_monitor": None,
             "patience": None}
    for _, cb in cbs:
        if hasattr(cb, "tie_break") and hasattr(cb, "best_epoch_"):
            stamp.update(selection_monitor=cb.monitor,
                         lower_is_better=bool(cb.lower_is_better),
                         tie_break=cb.tie_break,
                         smooth_window=int(getattr(cb, "smooth_window", 1)),
                         swap=bool(getattr(cb, "swap", True)))
        elif type(cb).__name__ == "EarlyStopping":
            stamp.update(stop_monitor=cb.monitor, patience=int(cb.patience))
    return stamp


def _device_stamp(net) -> dict:
    """The device the fit ran on, down to the physical card.

    The previous campaign had gradient spikes (up to 1e16) that tracked the execution
    context, not the model, and could only be traced to a card after the fact by
    re-pinning cells by UUID -- the records said nothing. A CUDA ordinal is not an
    identity either (it is renumbered by ``CUDA_VISIBLE_DEVICES``); the UUID is.
    """
    device = str(getattr(net, "device", None))
    stamp = {"device": device, "gpu_uuid": None, "gpu_name": None}
    if device.startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.device(device))
            stamp["gpu_uuid"] = str(getattr(props, "uuid", None))
            stamp["gpu_name"] = props.name
    return stamp


def _fingerprint(X, y) -> str | None:
    """SHA-1 of the training arrays, so a saved model can be matched to its fold.

    MOABB regenerates a fold from (seed, subject, session, cv_ind), but that is a
    convention about its splitter, and a model re-scored on the wrong fold returns a
    plausible number. The hash turns "this is the same fold" into a check.
    """
    import numpy as np
    try:
        h = hashlib.sha1(np.ascontiguousarray(np.asarray(X)).tobytes())
        if y is not None:
            h.update(np.ascontiguousarray(np.asarray(y)).tobytes())
        return h.hexdigest()
    except Exception:  # a skorch Dataset or anything not array-like
        return None


def _cpu_state(module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _shapes(module) -> dict:
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def _split_stamp(net) -> dict:
    split = getattr(net, "train_split", None)
    return {"n_fit_train": getattr(split, "n_train_", None),
            "n_valid": getattr(split, "n_valid_", None),
            "n_holdout": getattr(split, "n_holdout_", None)}


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
    save_selected : bool
        Save the module ``RestoreBestModel(swap=False)`` selected (``state_dict`` +
        width + epoch) next to ``path``, in ``<stem>__selected/<fit_uid>.pt``. Keeps
        the selected-epoch score recoverable offline when the net is scored at its
        last epoch; any such score is exploratory (the deep campaign's PREREG fixes
        the last epoch as the score). Default ``False``.
    save_final : bool
        Save the module as it ends training (the LAST epoch's weights, which are the
        scored ones under ``restore_selected=false``) in ``<stem>__final/<fit_uid>.pt``,
        same layout as ``save_selected``. The offline weight analyses (subject/session
        fingerprint, campaign A) need the scored network, not the selected one: the
        selected epoch varies per fit, and with it the width. Default ``False``.
    save_init : bool
        Save the weights as they enter training (``on_train_begin``) in
        ``<stem>__init/<fit_uid>.pt``: the origin of every trajectory, and what a
        "distance travelled" or a lottery-ticket style reading is measured from.
    save_growth : bool
        At every APPLIED growth event, keep a copy of each tensor whose shape the event
        changed, as it is right after the event -- the new neurons at birth, next to
        the old ones they join -- with the event's epoch, junction and widths. One file
        per fit, ``<stem>__growth/<fit_uid>.pt`` (a list, in event order). Only the
        resized tensors: a full state per event would cost ~40 conformer states per fit.
    save_checkpoints : sequence of int, optional
        Epochs (1-based, as in ``net.history``) at which the full weights are kept;
        written together at the end in ``<stem>__ckpt/<fit_uid>.pt`` as
        ``{epoch: state_dict}``. Campaign A: ``[10, 25, 50, 100]``.
    save_test_logits : bool
        A flag for the evaluation, not for this callback: the fold's test set is not
        in scope inside a skorch fit. ``benchmarks/subject_stamp.py`` reads it and, if
        set, writes the raw test logits under ``<stem>__logits/<last_fit_uid_>.npz``.

    Notes
    -----
    Place this callback **after** the growth callback and **before**
    ``EarlyStopping``. After growth, so the width it reads for an epoch is the width
    the epoch ended with; before early stopping, because skorch's ``EarlyStopping``
    raises ``KeyboardInterrupt`` from ``on_epoch_end`` and every callback after it in
    the list is skipped for that final epoch. ``on_train_end`` still runs either way:
    ``NeuralNet.partial_fit`` catches the interrupt and notifies it regardless.
    """

    def __init__(self, path, meta: dict | None = None, save_selected: bool = False,
                 save_final: bool = False, save_init: bool = False,
                 save_growth: bool = False, save_checkpoints=None,
                 save_test_logits: bool = False):
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
        self.save_selected = save_selected
        self.save_final = save_final
        self.save_init = save_init
        self.save_growth = save_growth
        self.save_checkpoints = save_checkpoints
        self.save_test_logits = save_test_logits
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
        self.fingerprint_ = _fingerprint(X, y) if X is not None else None
        # Drawn here, not at the end, so every file of this fit (init, growth, ckpt,
        # logits) carries the same id as its JSONL record.
        self.fit_uid_ = uuid.uuid4().hex
        self.init_path_ = None
        if self.save_init:
            self.init_path_ = self._save("init", {"state_dict": _cpu_state(net.module_),
                                                  "growable_width": getattr(net.module_, "growable_width", None),
                                                  "junction_widths": getattr(net.module_, "junction_widths", None)})
        self.shapes_ = _shapes(net.module_)
        self.growth_ = []
        self.ckpts_ = {}

    def on_epoch_end(self, net, **kwargs):
        # Recorded into skorch's own history so the values sit next to train_loss and
        # valid_acc and come out of the same place at the end.
        net.history.record("n_params", _n_params(net.module_))
        width = getattr(net.module_, "growable_width", None)
        if width is not None:
            net.history.record("width", int(width))
        row = net.history[-1]
        if self.save_growth and row.get("grow_applied"):
            shapes = _shapes(net.module_)
            state = net.module_.state_dict()
            changed = [k for k, v in shapes.items() if self.shapes_.get(k) != v]
            self.growth_.append({
                "epoch": row["epoch"], "junction": row.get("grow_junction"),
                "width_after": row.get("grow_width_after"),
                "junction_widths_after": row.get("grow_junction_widths_after"),
                "direction": row.get("grow_direction"),
                "shapes_before": {k: self.shapes_.get(k) for k in changed},
                "tensors": {k: state[k].detach().cpu().clone() for k in changed}})
            self.shapes_ = shapes
        if self.save_checkpoints and row["epoch"] in {int(e) for e in self.save_checkpoints}:
            self.ckpts_[int(row["epoch"])] = _cpu_state(net.module_)

    def _save(self, kind: str, payload: dict):
        """``torch.save`` one side file of this fit; returns its name for the record."""
        import torch
        folder = Path(self.path).with_name(Path(self.path).stem + f"__{kind}")
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{self.fit_uid_}.pt"
        torch.save({**payload, "train_fingerprint": self.fingerprint_, "meta": self.meta}, out)
        return out.name

    #: Per-epoch fields lifted out of ``net.history``. The ``grow_*`` ones exist only
    #: on epochs where a growth step actually ran, and ``grad_norm`` only when
    #: :class:`GradientNorm` is in the callback list -- both are filtered per row, so a
    #: fit without them writes a smaller record rather than failing.
    CURVE_KEYS = ("epoch", "train_loss", "valid_loss", "valid_acc", "width", "n_params",
                  "dur", "grad_norm", "grad_norm_max", "lr", "grow_s", "grow_applied",
                  "grow_width_after", "grow_n_proposed", "grow_n_kept",
                  "grow_first_order_improvement", "grow_eig_sum",
                  "grow_select_loss", "grow_losses", "grow_param_update_decrease",
                  "grow_eig_proposed", "grow_eig_kept", "grow_holdout", "grow_n_select",
                  "grow_normalization", "grow_direction", "grow_ls_s", "grow_rescale", "grow_rescale_fallbacks",
                  "grow_junction", "grow_junction_argmax", "grow_junction_foi", "grow_junction_ids",
                  "grow_junction_n_proposed", "grow_junction_n_kept",
                  "grow_junction_widths_after", "grow_split_a", "grow_split_b",
                  "grow_split_foi_a", "grow_split_foi_b",
                  "grow_fp64_retry", "grow_split_failed", "grow_split_fp64_retry",
                  "adam_atten_mean", "adam_atten_p05", "adam_eps_frac")

    def on_train_end(self, net, X=None, y=None, **kwargs):
        curve = [{k: h[k] for k in self.CURVE_KEYS if k in h} for h in net.history]
        # The spike refusal (deep campaign, step 0) reads these two fields: threshold
        # 1e6, calibrated on 167 076 fits (analysis/step0_catalogue/spike_threshold.py).
        gmax = [(h["grad_norm_max"], h["epoch"]) for h in net.history
                if h.get("grad_norm_max") is not None]
        gmax_fit, gmax_epoch = max(gmax) if gmax else (None, None)
        # Loss of the very first training batch: computed by the forward pass BEFORE
        # the first optimizer step (skorch's train_step), i.e. the initial loss the
        # step-0 "départ" criterion compares to ln(n_classes). Read off skorch's own
        # batch history, nothing recomputed.
        try:
            first_batch_loss = float(net.history[0]["batches"][0]["train_loss"])
        except (IndexError, KeyError, TypeError):
            first_batch_loss = None
        fit_uid = self.fit_uid_
        growth_path = self._save("growth", {"events": self.growth_}) if self.save_growth else None
        ckpt_path = self._save("ckpt", {"checkpoints": self.ckpts_}) if self.save_checkpoints else None
        n_growth, ckpt_epochs = len(self.growth_), sorted(self.ckpts_)
        self.growth_, self.ckpts_ = [], {}
        selected_path = None
        state = getattr(net, "selected_state_", None)
        if self.save_selected and state is not None:
            import torch
            folder = Path(self.path).with_name(Path(self.path).stem + "__selected")
            folder.mkdir(parents=True, exist_ok=True)
            selected_path = folder / f"{fit_uid}.pt"
            torch.save({**state, "epoch": getattr(net, "selected_epoch_", None),
                        "train_fingerprint": self.fingerprint_,
                        "meta": self.meta}, selected_path)
        if state is not None:
            net.selected_state_ = None  # release the copy either way
        final_path = None
        if self.save_final:
            import torch
            folder = Path(self.path).with_name(Path(self.path).stem + "__final")
            folder.mkdir(parents=True, exist_ok=True)
            final_path = folder / f"{fit_uid}.pt"
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.module_.state_dict().items()},
                        "growable_width": getattr(net.module_, "growable_width", None),
                        "junction_widths": getattr(net.module_, "junction_widths", None),
                        "epoch": net.history[-1]["epoch"] if len(net.history) else None,
                        "train_fingerprint": self.fingerprint_,
                        "meta": self.meta}, final_path)
        record = {
            **(self.meta or {}),
            "fit": self.fit_idx_,
            "fit_uid": fit_uid,
            "train_fingerprint": self.fingerprint_,
            "n_train": self.n_train_,
            # The three-way split the fit actually ran on (HoldoutValidSplit), read off
            # the live split object: `n_valid` is what the n_valid >= 16 floor is
            # checked on, `n_holdout` the growth line search's trials (0 = none).
            **_split_stamp(net),
            "seconds": round(time.time() - self.t0_, 3),
            "epochs": len(net.history),
            "max_epochs": int(getattr(net, "max_epochs", 0)),
            "width_start": self.width0_,
            "width_end": getattr(net.module_, "growable_width", None),
            "target_width": getattr(net.module_, "target_width", None),
            "params_start": self.params0_,
            "params_end": _n_params(net.module_),
            # Which epoch's weights the fitted estimator actually carries, stamped by
            # RestoreBestModel (None when nothing restored, i.e. the last epoch's).
            # Without it `width_end` above is ambiguous after a restore: the model
            # scored downstream may be narrower than the last epoch the curve shows.
            "restored_epoch": getattr(net, "restored_epoch_", None),
            "stop_reason": getattr(net, "stop_reason_", None),
            # The epoch the selection rule picked, restored or not. Under
            # RestoreBestModel(swap=False) the net is scored at its last epoch and
            # this is the restored_epoch MEASURE; `selected_path` holds its weights.
            "selected_epoch": getattr(net, "selected_epoch_", None),
            "selected_path": None if selected_path is None else selected_path.name,
            "final_path": None if final_path is None else final_path.name,
            "init_path": self.init_path_,
            "growth_path": growth_path,
            "n_growth_snapshots": n_growth if self.save_growth else None,
            "ckpt_path": ckpt_path,
            "ckpt_epochs": ckpt_epochs if self.save_checkpoints else None,
            "first_batch_loss": first_batch_loss,
            "grad_norm_max_fit": gmax_fit,
            "grad_norm_max_epoch": gmax_epoch,
            **_device_stamp(net),
            "optimizer": _optimizer_stamp(net),
            "selection": _selection_stamp(net),
            "history": curve,
        }
        self.fit_idx_ += 1
        # Read by the evaluation's logit scorer (benchmarks/subject_stamp.py), which runs
        # after the fit and must name its file after this record.
        self.last_fit_uid_ = fit_uid
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
