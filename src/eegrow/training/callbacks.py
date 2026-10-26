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
    Stopping and selection are separate arguments because they can disagree, and the
    benchmark measured that they do. The original argument for defaulting BOTH to
    ``valid_loss`` was that validation splits here are tiny (8 of 12 datasets hold
    fewer than 32 trials, three hold under 11), so validation accuracy moves in steps
    of 1/n_valid -- 0.0213 on bnci2014_001's 46-trial split -- and its argmax looked
    like a lucky early peak rather than a real optimum.

    That argument is WRONG for selection, and the 2x2 budget/selection square says so
    on 36 paired units (bnci2014_001 within_session, ``bd_deep4``)::

                          sel=valid_loss   sel=valid_acc
        patience  20          0.3501           0.3390
        patience 200          0.4074           0.4818

    The two full-budget cells have bit-identical trajectories and differ only in which
    epoch is handed back: ``valid_loss`` restores epoch 5, the accuracy peak is at
    epoch 158. So the peak is not early -- it was the early STOP that made it look
    early. The failure mode named in the docstring above is real and it is what breaks
    the loss: a net that becomes confident-and-wrong sees its loss explode (1.15 ->
    3.57 here) while its accuracy keeps climbing, and on 46 trials a handful of such
    trials dominates the mean. ``valid_loss`` is a broken proxy on splits this small,
    twice over -- as the stopping rule and as the selection rule.

    ``restored_epoch_`` is recorded so the choice stays auditable after the fact rather
    than implicit.

    WHICH EPOCH WINS A TIE
    ----------------------
    The historical rule kept the EARLIER of two tied epochs, on the argument that the
    less-trained model is the conservative pick on a noisy criterion. On a quantised
    ``valid_acc`` that argument backfires: with n_valid trials the metric takes only
    n_valid + 1 values, a lucky epoch 1 or 2 sets a plateau the trained net can only
    tie, and the strict comparison then hands back a net that barely trained. Measured
    offline on the ``grow_shallow`` probe fits (``benchmarks/analysis/restored_tie_rule``,
    06/10/2026): fits scored before epoch 5 fall from 64 % to 15 % on shin2017a
    (n_valid=4) and from 33 % to 9 % on physionetmi (n_valid=8) under
    ``tie_break="latest"``, and the across-fit reliability of ``restored_epoch_``
    rises (ICC(1,k) 0.39 -> 0.61 and 0.44 -> 0.65). Large splits barely move (cho2017,
    n_valid=32: 5 % -> 2 %), which is what a fix for quantisation should do.

    But ``"latest"`` breaks the OTHER edge (measured 07/10, ``smooth_rule.py`` in the
    same folder): once the net has memorised, ``valid_acc`` keeps re-touching its max
    until the budget ends, so the last argmax lands in the final 5 epochs on 23-49 % of
    fits, on every dataset including bnci2014_001. Neither tie rule is the fix.

    SMOOTHED SELECTION
    ------------------
    Both edges fail for the same reason: a single quantised reading is selected on, so
    whichever epoch got lucky wins. ``smooth_window=W`` selects instead on the centred
    rolling mean of ``monitor`` over W epochs (truncated at both ends, at least
    W//2 + 1 readings -- ``pandas.rolling(W, center=True, min_periods=W//2+1)``). An
    isolated lucky epoch is diluted by its neighbours; a real plateau keeps its height.
    Offline on the same probe fits (07/10): W=5 brings bnci2014_001 to 0 % early /
    10 % late, under the 20 % refusal on both edges, where ``"latest"`` sat at 23 %
    late. Below n_valid ~ 8 no rule passes either edge -- that is a floor on the
    dataset, not a property of the rule.

    The centred mean of epoch t is only known at epoch t + W//2, so the module of the
    last W//2 + 1 epochs is kept in a ring buffer and the snapshot is taken from it
    once its window closes. The epochs still open when training ends are scored on
    their truncated windows in ``on_train_end``. ``smooth_window=1`` is exactly the
    unsmoothed rule (no buffer, snapshot on improvement). The default stays 1 and
    ``"earliest"`` so that archived results reproduce.

    Parameters
    ----------
    monitor : str
        History key to select on. Default ``"valid_loss"``.
    lower_is_better : bool
        Whether a smaller ``monitor`` is better. Default ``True``.
    tie_break : {"earliest", "latest"}
        Which epoch to keep when the monitor ties the best so far. Default
        ``"earliest"`` (historical). ``"latest"`` fixes the early edge and breaks the
        late one (see above). Applies to the smoothed score when ``smooth_window > 1``.
    smooth_window : int
        Odd width W of the centred rolling mean selected on. Default 1 (no smoothing).
    swap : bool
        Whether the selected module replaces the last one. Default ``True`` (the net
        is scored at the selected epoch). ``False`` selects without restoring: the net
        is scored at its LAST epoch, and the selected module is only handed to
        ``FitRecorder`` to be saved. That is the deep campaign's protocol (13/10): at
        patience 200 the last epoch scores as well or better than any selection on a
        46-trial split (live, ``analysis/restored_tie_rule/live``), so the selection
        serves the ``restored_epoch`` measure, not the score -- and keeping it out of
        the score stops predictor and accuracy from sharing the same validation noise.

    Attributes stamped on the net
    -----------------------------
    ``restored_epoch_``
        The epoch whose weights the net ends up carrying, or ``None`` if the last
        epoch was already the best -- or if ``swap=False``, since the net then always
        carries its last epoch. :class:`~eegrow.training.recording.FitRecorder`
        writes it into the fit record, so a downstream score can always be traced to
        the epoch that produced it.
    ``selected_epoch_``
        The epoch the rule selected, whether or not it was restored (``None`` if no
        epoch was selectable).
    ``selected_state_``
        ``swap=False`` only: ``{"state_dict", "growable_width"}`` of the selected module
        (CPU tensors), for ``FitRecorder`` to save; released by the recorder once
        written. A dict and not the module itself: skorch refuses any
        ``torch.nn.Module`` set on the net outside ``initialize_module``.

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

    def __init__(self, monitor: str = "valid_loss", lower_is_better: bool = True,
                 tie_break: str = "earliest", smooth_window: int = 1,
                 swap: bool = True):
        if tie_break not in ("earliest", "latest"):
            raise ValueError(f"tie_break must be 'earliest' or 'latest', got {tie_break!r}")
        if (isinstance(smooth_window, bool) or not isinstance(smooth_window, (int, np.integer))
                or smooth_window < 1 or smooth_window % 2 == 0):
            raise ValueError(f"smooth_window must be a positive odd int, got {smooth_window!r}")
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.tie_break = tie_break
        self.smooth_window = smooth_window
        self.swap = swap
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
        # Smoothing state: every (epoch, score) seen, and the modules of the last
        # W//2 + 1 epochs -- the oldest is the one whose window closes next.
        self.scores_ = []
        self.buffer_ = []
        net.restored_epoch_ = None
        net.selected_epoch_ = None
        net.selected_state_ = None

    def _improved(self, score) -> bool:
        # ``tie_break="earliest"`` (strict) is the historical rule; the class docstring
        # says why it is the wrong one on a quantised monitor.
        if self.tie_break == "latest":
            return score <= self.best_score_ if self.lower_is_better else \
                score >= self.best_score_
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
            score = None
        if self.smooth_window == 1:
            if score is not None and self._improved(score):
                self.best_score_ = score
                self.best_epoch_ = row["epoch"]
                self.best_module_ = copy.deepcopy(net.module_)
            return
        # A non-finite reading still takes its slot (NaN), so window positions stay
        # epochs; the mean skips it, like pandas does.
        self.scores_.append(np.nan if score is None else float(score))
        self.buffer_.append((row["epoch"], copy.deepcopy(net.module_)))
        half = self.smooth_window // 2
        if len(self.buffer_) > half + 1:
            self.buffer_.pop(0)
        # The window of the epoch ``half`` positions back has just closed.
        centre = len(self.scores_) - 1 - half
        if centre >= 0:
            self._consider(centre, *self.buffer_[0])

    def _smoothed(self, i: int) -> float:
        half = self.smooth_window // 2
        win = np.asarray(self.scores_[max(0, i - half): i + half + 1])
        win = win[np.isfinite(win)]
        # Rounded: two windows holding the same quantised readings in a different
        # order must TIE, or float summation order breaks the tie rule at random.
        return round(float(win.mean()), 10) if len(win) >= half + 1 else np.nan

    def _consider(self, i: int, epoch, module):
        smoothed = self._smoothed(i)
        if np.isfinite(smoothed) and self._improved(smoothed):
            self.best_score_ = smoothed
            self.best_epoch_ = epoch
            self.best_module_ = copy.deepcopy(module)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        if self.smooth_window > 1:
            # Epochs whose window never closed: score them on what they got, in
            # epoch order so the tie rule still means earliest/latest. Indices
            # 0..n-half-1 were considered in on_epoch_end; the buffer holds the last
            # len(buffer_) indices.
            n, half = len(self.scores_), self.smooth_window // 2
            first_buffered = n - len(self.buffer_)
            for k, (epoch, module) in enumerate(self.buffer_):
                if first_buffered + k >= n - half:
                    self._consider(first_buffered + k, epoch, module)
            self.buffer_ = []
        if self.best_module_ is None:
            return
        net.selected_epoch_ = self.best_epoch_
        if not self.swap:
            net.selected_state_ = {
                "state_dict": {k: v.detach().cpu()
                               for k, v in self.best_module_.state_dict().items()},
                "growable_width": getattr(self.best_module_, "growable_width", None),
                # Multi-junction modules: the per-junction widths (the sum alone does
                # not say how to rebuild the module; the shapes in state_dict do too).
                "junction_widths": getattr(self.best_module_, "junction_widths", None)}
            self.best_module_ = None
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


class RequireParamGroups(Callback):
    """Refuse a fit whose ``optimizer__param_groups`` patterns match no parameter.

    skorch (1.4, ``NeuralNet._get_params_for_optimizer``) drops a pattern without a
    match in silence: the parameters it meant to exempt stay in the default group and
    the fit runs the old recipe under the new label. Checked on every optimizer the
    fit trains with (``on_epoch_begin``), since growth rebuilds it.
    """

    def __init__(self, patterns):
        self.patterns = list(patterns)

    def on_epoch_begin(self, net, **kwargs):
        import fnmatch
        names = [n for n, _ in net.module_.named_parameters()]
        dead = [p for p in self.patterns if not fnmatch.filter(names, p)]
        if dead:
            raise ValueError(f"optimizer param_groups patterns match no parameter: {dead}")
        expected = len(self.patterns) + 1
        if len(net.optimizer_.param_groups) != expected:
            raise ValueError(f"{len(net.optimizer_.param_groups)} optimizer param groups, "
                             f"expected {expected} (patterns {self.patterns} + default)")


def global_grad_norm(module: torch.nn.Module) -> float:
    """L2 norm of the concatenated gradients of ``module``, for one-off checks."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return total ** 0.5
