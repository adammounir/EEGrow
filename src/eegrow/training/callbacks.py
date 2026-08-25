"""Model selection and instrumentation callbacks that survive a model that grows.

Two things skorch does perfectly well for a fixed net and cannot do for a growing one:

``RestoreBestModel``
    give back the best epoch's weights instead of the last epoch's. skorch's
    ``EarlyStopping(load_best=True)`` snapshots ``state_dict()`` and restores it with a
    strict ``load_state_dict``, which raises ``size mismatch`` the moment the module
    grew after its best epoch -- and it always has here, because early stopping ends a
    fit exactly ``patience`` epochs after the best one and ``grow_every=5`` fits four
    growth events in that window.

``GradientNorm``
    record the global gradient norm per epoch. Nothing in skorch reports it, and it is
    the observable that separates "the optimizer is taking bad steps" from "this fold
    is just hard" -- the two readings of the diverging curves we cannot currently tell
    apart.

``AdamEpsDominance``
    record how much of the AdamW step ``eps`` is absorbing. Note this is *not* the same
    measurement as the one above and does not follow from it: the ``eps`` mechanism is a
    per-coordinate comparison, and a global norm cannot resolve it either way.

All three are deliberately plain ``Callback``s rather than subclasses of skorch's own:
subclassing ``EarlyStopping`` would tie *when we stop* to *which model we keep*, and
those are separate decisions with separate right answers on a quantised metric.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from skorch.callbacks import Callback


class RestoreBestModel(Callback):
    """Give the fitted net the weights of its best epoch, not its last.

    WHY THIS EXISTS
    ---------------
    ``EEGClassifier`` ships no ``Checkpoint``, and skorch's ``EarlyStopping`` defaults
    to ``load_best=False``. The consequence is not subtle and it is not occasional:
    early stopping fires ``patience`` epochs after the best epoch *by construction*, so
    on the v5 campaign ``epochs - epoch_of_best`` was exactly 20 on all 140 490 folds
    (std 0.0). Every score ever published from this benchmark is the model 20 epochs
    past its own best -- a median 4.3 accuracy points of internal validation thrown
    away, 10.6 on ``bd_deep4``, and on the arms that diverge it is the post-blow-up
    model that gets scored.

    WHY NOT ``EarlyStopping(load_best=True)``
    -----------------------------------------
    Because it restores a ``state_dict`` into the live module, and a growing module's
    tensors changed shape in the meantime::

        size mismatch for conv_time.layer.weight: copying a param with shape
        torch.Size([8, 1, 25, 1]) ..., the shape in current model is [12, 1, 25, 1]

    Not a rare edge: the window between the best epoch and the stop is precisely where
    growth keeps happening. So the snapshot has to be the *module*, not its weights --
    then the shapes travel with them and restoring is an object swap.

    WHY A SEPARATE MONITOR FROM THE STOPPING RULE
    ---------------------------------------------
    The default monitor is ``valid_loss`` even when the score reported downstream is
    accuracy, and that is on purpose. Internal validation splits here are tiny (8 of
    12 datasets hold fewer than 32 trials, three hold under 11), so validation accuracy
    moves in steps of 1/n_valid -- 0.0213 on bnci2014_001's 46-trial split. Its argmax
    is an early lucky peak far more often than it is a real optimum: on that dataset
    the accuracy peaks at epoch 30 while the loss is still falling until epoch 38, and
    the loss is still improving after the accuracy peak in 65 % of folds. A continuous
    criterion picks the better model out of the same trajectory.

    Note the trade-off honestly: a net can have rising validation loss while its
    accuracy still climbs (confident-but-wrong predictions inflate the loss first).
    ``monitor="valid_acc", lower_is_better=False`` restores the accuracy criterion for
    anyone who wants to measure that, and the recorded ``restored_epoch_`` makes the
    choice auditable after the fact rather than implicit.

    Parameters
    ----------
    monitor : str
        History key to select on. Default ``"valid_loss"``.
    lower_is_better : bool
        Whether a smaller ``monitor`` is better. Default ``True``.

    Attributes stamped on the net
    -----------------------------
    ``restored_epoch_``
        The epoch whose weights the net ends up carrying, or ``None`` if the last
        epoch was already the best. :class:`~eegrow.training.recording.FitRecorder`
        writes it into the fit record, so a downstream score can always be traced to
        the epoch that produced it.

    Notes
    -----
    Place this **before** ``FitRecorder`` in the callback list. skorch notifies
    callbacks in order, so the recorder then reports the width and parameter count of
    the model that is actually returned -- which after a restore can be narrower than
    the last epoch of the curve it writes alongside.

    The snapshot is a full ``deepcopy`` of the module, taken only when the monitor
    improves. That is bounded by the number of improvements, not the number of epochs,
    and these models top out around 300 k parameters; the copy is cheap next to the
    epoch that produced it.
    """

    def __init__(self, monitor: str = "valid_loss", lower_is_better: bool = True):
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        # Trailing underscore: skorch's ``Callback.get_params`` feeds every attribute
        # without one back through ``__init__`` when sklearn clones the pipeline, and
        # MOABB clones it per fold. State ends in ``_``; hyperparameters do not.
        self.best_score_ = None
        self.best_epoch_ = None
        self.best_module_ = None

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.best_score_ = -np.inf if not self.lower_is_better else np.inf
        self.best_epoch_ = None
        self.best_module_ = None
        net.restored_epoch_ = None

    def _improved(self, score) -> bool:
        # Strict: on a tie we keep the EARLIER model. Both are equally good on the
        # criterion and the earlier one is the smaller/less-trained of the two, which
        # is the conservative pick when the criterion is noisy.
        return score < self.best_score_ if self.lower_is_better else \
            score > self.best_score_

    def on_epoch_end(self, net, **kwargs):
        row = net.history[-1]
        if self.monitor not in row:
            return
        score = row[self.monitor]
        if score is None or not np.isfinite(score):
            # A diverged fold reports NaN loss. Never snapshot it, and never let it
            # win the comparison by accident -- NaN compares False either way, which
            # is the behaviour we want, but being explicit costs nothing.
            return
        if self._improved(score):
            self.best_score_ = score
            self.best_epoch_ = row["epoch"]
            self.best_module_ = copy.deepcopy(net.module_)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        if self.best_module_ is None:
            return
        last_epoch = net.history[-1]["epoch"] if len(net.history) else None
        if self.best_epoch_ == last_epoch:
            net.restored_epoch_ = None
            self.best_module_ = None  # release the copy
            return
        # Object swap, not ``load_state_dict``: the shapes come with the weights.
        # The optimizer is left pointing at the discarded parameters, which is correct
        # -- training is over, and ``predict`` reads ``module_``.
        net.module_ = self.best_module_
        net.restored_epoch_ = self.best_epoch_
        self.best_module_ = None


class GradientNorm(Callback):
    """Record the global L2 gradient norm of every training batch, per epoch.

    Stella's first ask, and the measurement that decides between the two readings of
    the diverging folds: an optimizer taking oversized steps (the AdamW-``eps``
    hypothesis) would show the norm collapsing toward zero while the step size stays
    put, whereas a genuinely hard fold shows a norm that stays large. Neither is
    currently observable, so the question has only been argued, not settled.

    Measured at ``on_grad_computed``, which skorch calls after ``backward()`` and
    before ``optimizer.step()`` -- i.e. on the gradients that are actually applied, and
    before any clipping a future config might introduce.

    Records ``grad_norm`` (mean over the epoch's batches), ``grad_norm_max`` (the worst
    single step, which is what a spike looks like and a mean hides) and ``lr``.
    """

    def __init__(self):
        self.batch_norms_ = []

    def on_epoch_begin(self, net, **kwargs):
        self.batch_norms_ = []

    def on_grad_computed(self, net, named_parameters, **kwargs):
        total = 0.0
        for _, p in named_parameters:
            if p.grad is not None:
                total += float(p.grad.detach().norm(2).item() ** 2)
        self.batch_norms_.append(total ** 0.5)

    def on_epoch_end(self, net, **kwargs):
        norms = self.batch_norms_
        net.history.record("grad_norm", float(np.mean(norms)) if norms else None)
        net.history.record("grad_norm_max", float(np.max(norms)) if norms else None)
        # The lr is constant under the current config (no scheduler), which is exactly
        # why it is worth recording: the moment a scheduler is added, every curve
        # already carries the schedule it ran under and no rerun is needed to know it.
        opt = getattr(net, "optimizer_", None)
        lr = None
        if opt is not None and opt.param_groups:
            lr = float(opt.param_groups[0].get("lr", float("nan")))
        net.history.record("lr", lr)


class AdamEpsDominance(Callback):
    r"""Measure how much of AdamW's step ``eps`` is currently eating, per epoch.

    WHY ``GradientNorm`` CANNOT ANSWER THIS
    ---------------------------------------
    Stella's hypothesis is that on the easy/small folds the gradient collapses toward
    zero, AdamW's denominator becomes dominated by ``eps``, the step degenerates to a
    fixed rescaling of the raw gradient, and performance dies. The observable we reached
    for first was the global gradient norm -- and it *cannot* settle the question, because
    the mechanism is per-coordinate and the norm is an aggregate over ~300 000 of them.
    A global norm of 2.0 is perfectly consistent with a handful of conv filters carrying
    all of it while thousands of coordinates sit at 1e-10. Reading a global norm to
    decide a per-coordinate comparison is a scale error, not a resolution one.

    THE ACTUAL QUANTITY
    -------------------
    torch's AdamW forms ``denom_i = sqrt(v_i)/sqrt(bc2) + eps`` and steps by
    ``-lr * m_hat_i / denom_i`` (verified against ``torch.optim.adam``, 2.13). The step
    an ``eps``-free Adam would take is the same thing without the ``+ eps``, so the ratio
    of what the optimizer *does* to what it *should* do is, exactly and elementwise,

    .. math:: a_i \;=\; \frac{\Delta_i}{\Delta^*_i} \;=\;
              \frac{\sqrt{\hat v_i}}{\sqrt{\hat v_i} + \varepsilon} \;\in\; (0, 1]

    with :math:`\hat v_i = v_i / (1 - \beta_2^t)`. ``a_i = 1/2`` is precisely the point
    where ``sqrt(v_hat_i) == eps``; ``a_i -> 0`` is Stella's failure mode; ``a_i ~ 1`` is
    an optimizer behaving as designed. This is a *direct* measurement of the effect --
    not a proxy for it -- and it needs no extra fit: the state is already in the
    optimizer.

    Recorded per epoch:

    ``adam_atten_mean``
        Mean :math:`a_i` over every coordinate with optimizer state. The single number
        to plot against the loss curve.
    ``adam_atten_p05``
        5th percentile of :math:`a_i` -- the damped tail a mean hides. A net can be
        healthy on average while the layer that matters is frozen.
    ``adam_eps_frac``
        Fraction of coordinates with :math:`\sqrt{\hat v_i} < \varepsilon`, i.e. where
        ``eps`` supplies more than half the denominator. **This is the number that
        confirms or refutes the hypothesis.**

    PLACEMENT
    ---------
    Before ``GromoGrowth`` in the callback list, and this is load-bearing rather than
    cosmetic. Growth rebuilds the optimizer and the neurons it just spliced in have no
    accumulator yet (``v = 0``), so measuring afterwards reads ``a_i = 0`` on every new
    coordinate and reports a spurious ``eps`` collapse that is really just a fresh
    moment estimate. Measured before, the epoch is described by the optimizer that
    actually trained it.

    Notes
    -----
    Costs one pass over the optimizer state per epoch (no gradients, no forward), which
    is negligible next to the epoch that produced it. Silently records ``None`` for a
    non-Adam optimizer rather than guessing: ``eps`` means something different in each
    optimizer family, and a number that quietly changes meaning is worse than a gap.
    """

    #: History keys written every epoch, so a reader can rely on their presence.
    KEYS = ("adam_atten_mean", "adam_atten_p05", "adam_eps_frac")

    def on_epoch_end(self, net, **kwargs):
        stats = self._measure(getattr(net, "optimizer_", None))
        for key in self.KEYS:
            net.history.record(key, stats.get(key))

    @staticmethod
    def _measure(opt) -> dict:
        """Attenuation statistics of ``opt``'s current state, or empty if not Adam."""
        if opt is None or not getattr(opt, "param_groups", None):
            return {}
        atten = []
        for group in opt.param_groups:
            eps = group.get("eps")
            betas = group.get("betas")
            if eps is None or betas is None:  # SGD & friends: no eps semantics
                continue
            beta2 = float(betas[1])
            for p in group["params"]:
                state = opt.state.get(p)
                if not state or "exp_avg_sq" not in state:
                    continue  # never stepped: no accumulator to read
                step = state.get("step", 0)
                step = float(step.item() if torch.is_tensor(step) else step)
                if step <= 0:
                    continue
                # Same bias correction the optimizer applies, so the ratio below is the
                # attenuation of the step actually taken and not of an idealised one.
                bc2 = 1.0 - beta2 ** step
                root_v = (state["exp_avg_sq"].detach() / bc2).sqrt_()
                atten.append((root_v / (root_v + eps)).flatten())
        if not atten:
            return {}
        a = torch.cat(atten).float()
        return {
            "adam_atten_mean": float(a.mean()),
            # torch.quantile caps out around 2^24 elements; these nets are three orders
            # of magnitude below that, but fall back rather than fail on a future one.
            "adam_atten_p05": float(a.quantile(0.05)) if a.numel() < 2 ** 24
            else float(np.quantile(a.numpy(), 0.05)),
            "adam_eps_frac": float((a < 0.5).float().mean()),
        }


class StopReason(Callback):
    """Stamp ``net.stop_reason_`` with what ended the fit: ``early`` or ``budget``.

    Recoverable after the fact (``epochs < max_epochs``) for a fit that ran to plan,
    but not for one that died on an exception or was cut short by anything else, and
    it is one of the fields Stella asked to have on every record. Cheap to write down
    at the time rather than infer later.

    Place it **before** ``FitRecorder``, which reads ``stop_reason_`` when it writes
    its record; skorch notifies in list order, so a later position leaves the field
    ``None`` on every fit. Its position relative to ``EarlyStopping`` does not matter:
    it only writes from ``on_train_end``, which skorch notifies whether the fit ended
    on the budget or on the ``KeyboardInterrupt`` early stopping raises.
    """

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        net.stop_reason_ = None

    def on_train_end(self, net, X=None, y=None, **kwargs):
        n = len(net.history)
        net.stop_reason_ = "budget" if n >= int(getattr(net, "max_epochs", 0)) \
            else "early"


def global_grad_norm(module: torch.nn.Module) -> float:
    """L2 norm of the concatenated gradients of ``module``, for one-off checks."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return total ** 0.5
