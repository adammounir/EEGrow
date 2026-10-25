"""Training loop with growth: gradient descent + periodic growth.

Key point: instead of fully resetting the optimizer after each growth (which loses
momentum across the WHOLE network), we preserve the optimizer state (Adam momentum)
of the unchanged parameters and only start fresh for the weights that grew.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from gromo.utils.training_utils import compute_statistics, evaluate_model

# gromo raises "input_extension_scaling is null" from ``apply_change`` -- and ONLY
# from there (growing_module.py:2147), not from the line search, which never applies
# anything. It therefore fires on exactly one event: the line search picked s=0 and we
# applied the change anyway, adding neurons whose input weights are identically zero.
# Those are an exact stationary point -- zero activation, zero gradient, dead for the
# rest of the fit while still counted by ``width`` and ``n_params``. Silencing it hid
# the one signal that would have shown it. ``grow_step`` now abstains at s=0 instead,
# so the warning has nothing left to fire on, and the filter is gone deliberately: if
# it ever fires again, something applied a null extension and we want to hear about it.


@dataclass
class TrainConfig:
    total_epochs: int = 150
    grow_every: int = 15
    lr: float = 6.25e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    log_every: int = 10


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_optimizer(model, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)


def rebuild_optimizer_preserving_state(
    old_opt: torch.optim.Optimizer, model: torch.nn.Module, cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, int]:
    """Rebuild the optimizer after a growth, keeping momentum of unchanged params.

    After ``apply_change``, gromo replaces the weight tensors that grew with NEW
    ``Parameter`` objects; the others (BN, classifier...) stay the SAME objects. We
    transfer the state for those identical objects and leave fresh state for the
    new/enlarged params.

    Returns
    -------
    (optimizer, number_of_params_whose_momentum_was_transferred)
    """
    old_state = old_opt.state  # dict keyed by Parameter object
    new_opt = make_optimizer(model, cfg)
    transferred = 0
    for p in model.parameters():
        if p in old_state:  # same object => unchanged param => keep its momentum
            new_opt.state[p] = old_state[p]
            transferred += 1
    return new_opt, transferred


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train_epoch(model, loader, opt, crit, device) -> float:
    model.train()
    tot = 0.0
    for x, y in loader:
        opt.zero_grad()
        loss = crit(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / len(loader)


#: The historical line-search grid. Censored: applied steps pick its top on 97.8-100 %
#: of events (analysis/step0_catalogue/line_search_censoring.py), so its maximum says
#: nothing about the best amplitude. Kept as the default so archived runs reproduce;
#: the deep campaign passes WIDE_SCALING_GRID.
SCALING_GRID = (0.0, 0.1, 0.5, 1.0)
#: Geometric, and twice past 1 so the step-0 censoring criterion (top of the grid wins
#: on > 20 % of applied steps => refused) can actually be met. Decided 13/10.
WIDE_SCALING_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

#: ``normalization="tiny_paper"``: the amplitude of the TINY paper's training protocol
#: (Verbockhaven et al., TMLR 2024, arXiv 2405.19816). Algorithm 1 (the protocol of
#: Fig. 5-6 and Table 1) is NewNeurons -> "Normalize according to E.3" -> add the
#: neurons -> train; App. E.3.1 eq. (197)-(198) sets the incoming (alpha) and outgoing
#: (omega) weights of the K new neurons to a mean squared norm per neuron of 1e-3:
#: ``alpha_k <- alpha_k * sqrt(1e-3 / (||(alpha_j)_j||^2 / K))``, same for omega. No
#: line search (the gamma* search of App. E.3.2 is Fig. 8, a run WITHOUT any gradient
#: descent), and no best-update step on the existing weights (Algorithm 1 adds neurons
#: only). Read 21/10/2026 in the PDF; nothing here is inferred.
TINY_PAPER_SQ_NORM = 1e-3


#: Keep a candidate neuron when its singular value is at least this fraction of the
#: largest one. Replaces gromo's ``statistical_threshold``, which is an ABSOLUTE cut
#: (default 1e-3) applied to a quantity whose scale is set by the loss gradients of
#: the data -- see ``grow_step``'s Notes for the measurement that made this necessary.
MIN_SINGULAR_RATIO = 0.1


def grow_step(model, train_loader, device, *, val_loader=None,
              max_added: int | None = None,
              min_singular_ratio: float = MIN_SINGULAR_RATIO,
              scaling_grid=SCALING_GRID, normalization: str | None = None,
              select: str = "first", split_half: bool = False,
              fixed_s: float | None = None, rescale: bool = False,
              direction: str = "tiny") -> dict:
    """One gromo growth: stats -> optimal update -> line search -> apply_change.

    ``direction="random"`` (campaign A control, needs ``normalization="tiny_paper"``)
    keeps every decision of the step and draws the new neurons' directions at random;
    see :func:`_randomize_candidates`.

    Two controls for the 20/10 attribution campaign, both off by default (historical):
    ``fixed_s`` applies that amplitude on every step, whatever the line search says
    (its losses are still recorded) -- a dose that the data cannot choose. ``rescale``
    restores, after ``apply_change``, each output channel of the growing layer to its
    pre-growth norm and compensates the following BatchNorm's running statistics
    (``model.post_growth_norm``), so the network computes the same function with the
    pre-growth weight scale: it separates the *direction* a step writes from the
    *scale* it leaves behind.

    ``select="first"`` (default, every historical arm) grows ``_growable_layers[0]``.
    ``select="best"`` is for a :class:`MultiJunctionGrowingModel`: candidates on every
    junction, then gromo's ``select_best_update``; see :func:`_grow_step_best`.
    ``select="sequential"`` is its control: the same candidates, but the junctions grow
    in turn (gromo's ``scheduling_method="sequential"``; TINY's Algorithm 1 loops over
    the depths), whatever the gains say.

    NB: ``compute_optimal_updates`` relies on ``eigh``, which is NOT implemented on
    MPS -- growth therefore requires a CPU device.

    Parameters
    ----------
    train_loader :
        Used to accumulate the growth statistics (the optimal new neurons).
    val_loader : optional
        If given, the line search (which scaling factor to apply) is evaluated on
        this *held-out* loader instead of ``train_loader``. gromo's candidate neurons
        already fit the train signal, so selecting the scaling on held-out data
        curbs the tendency to amplify neurons that only fit the training noise.
    max_added : int, optional
        Hard cap on the number of neurons added this step. After computing the
        optimal update we sub-select the ``max_added`` best candidates, so the width
        never overshoots the target (gromo otherwise adds a data-dependent count).
        Defaults to whatever ``model.target_width`` leaves room for. It used to
        default to "no cap", which was safe only because a step added one neuron at a
        time; now that a step can propose eighteen, an uncapped call overshoots the
        target on its first go -- and ``run_model`` below is exactly such a call.
    min_singular_ratio : float
        Keep a candidate neuron when its singular value is at least this fraction of
        the largest candidate's. Replaces gromo's absolute ``statistical_threshold``;
        see Notes. ``1.0`` reproduces "one neuron per step", ``0.0`` keeps every
        numerically valid candidate.
    scaling_grid : sequence of float
        The amplitudes the line search tries, in increasing order; must contain 0.0
        (the abstention). Ties go to the smaller amplitude. Default ``SCALING_GRID``.

    Returns
    -------
    dict
        What the step decided, for the caller to log: ``s`` (the scaling factor the
        line search won with), ``applied`` (False when it abstained), ``losses`` (the
        held-out loss at each point of the grid), ``width_before``/``width_after``,
        and gromo's own two diagnostics -- ``first_order_improvement`` and the
        ``eigenvalues_extension`` of the proposed neurons. None of this was observable
        before: ``s`` in particular was chosen and discarded inside this function.

    Notes
    -----
    **Abstention at s=0.** The line search returns a *direction* (the optimal new
    neurons) and an *amplitude*. gromo's own scaling setter couples the two: s=0 sets
    ``optimal_delta_scaling``, ``input_extension_scaling`` and
    ``output_extension_scaling`` all to zero, so ``apply_change`` then applies no delta
    to the existing weights and splices in neurons whose weights are identically zero.
    A zero-in/zero-out neuron is an exact stationary point of the loss: its activation
    is zero, so its gradient is zero, so it stays zero for the rest of the fit. It is
    dead on arrival -- and still counted by ``growable_width`` and ``n_params``, which
    is what makes it worse than not growing: it consumes the width budget (``max_added``
    is computed from ``target_width - growable_width``) and inflates the parameter axis
    the efficiency claim is measured on, in exchange for nothing.

    So s=0 is not "grow by zero", it is "the search found no useful amplitude". We
    discard the update instead of applying it, and the width is unchanged -- which the
    caller must read as *abstained*, not as *capped*, or it will stop growing forever
    on the first bad epoch.

    **How many neurons a step proposes.** gromo selects candidates with
    ``s >= min(statistical_threshold, s.max())`` (``utils/tools.py``), an ABSOLUTE cut
    with default 1e-3 on singular values whose scale is set by the loss gradients of
    the data. When every candidate falls below it the expression collapses to
    ``s >= s.max()`` and **exactly one** neuron survives -- not as a decision about the
    data, but because a constant chosen elsewhere sits above our whole spectrum.

    That is not a hypothetical. Measured on one growth step of each arm (22 channels,
    256 samples, 4 batches of 64), largest singular value and resulting count::

        arm            s.max      abs 1e-3    ratio 0.1
        grow_deep      1.07e-02   8           8
        grow_sccnet    2.08e-03   8           17
        grow_shallow   5.11e-04   1           18
        grow_eegnex    1.42e-06   1           11

    The two arms whose spectrum clears 1e-3 grow eight neurons at a time; the two below
    it grow exactly one. That reproduces the v5 campaign's per-arm growth rates on the
    nose -- ``grow_deep`` +8 per event (1204 of 1773 events), ``grow_shallow`` +1 (70 %
    of events), ``grow_eegnex`` +1 always -- and it is why ``grow_shallow`` could never
    reach its target: 32 neurons to add, one per event, 5 epochs per event, 160 epochs
    against fits that last 30.

    So the count was never really a property of growth, and raising ``grow_every`` would
    only have bought a linear factor on a mechanism that is scale-dependent by
    construction. We keep every numerically valid candidate and apply a *relative*
    floor instead: within ``min_singular_ratio`` of the best. It is scale-free, it
    leaves the arm that already worked (``grow_deep``, 8 either way) untouched, and it
    is still a threshold on the same quantity gromo ranks by -- we changed the units it
    is expressed in, not the criterion.
    """
    grid = tuple(float(v) for v in scaling_grid)
    if 0.0 not in grid or list(grid) != sorted(grid):
        raise ValueError(f"scaling_grid must be increasing and contain 0.0, got {grid}")
    if fixed_s is not None and not float(fixed_s) > 0.0:
        raise ValueError(f"fixed_s must be > 0 (s=0 is an abstention), got {fixed_s}")
    if normalization == "tiny_paper" and fixed_s is None:
        # The paper's amplitude is the normalisation itself; a line search on top would
        # be a different method (see TINY_PAPER_SQ_NORM). fixed_s=1 is the paper.
        raise ValueError("normalization='tiny_paper' needs fixed_s (1.0 = the paper)")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if direction == "random" and (normalization != "tiny_paper" or rescale):
        # The norm of a random draw is arbitrary: only the paper's normalisation, applied
        # after the draw, makes the two arms differ by the direction alone.
        raise ValueError("direction='random' needs normalization='tiny_paper' and no rescale")
    if select in ("best", "sequential"):
        if rescale:
            raise ValueError("rescale is a single-junction control (select='first')")
        if split_half and select != "best":
            raise ValueError("split_half compares argmaxes: it needs select='best'")
        return _grow_step_best(model, train_loader, device, val_loader=val_loader,
                               min_singular_ratio=min_singular_ratio, grid=grid,
                               normalization=normalization, split_half=split_half,
                               sequential=select == "sequential", fixed_s=fixed_s,
                               direction=direction)
    if select != "first":
        raise ValueError(f"select must be 'first', 'best' or 'sequential', got {select!r}")
    if split_half:
        raise ValueError("split_half compares junctions: it needs select='best'")
    crit_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    crit_mean = torch.nn.CrossEntropyLoss(reduction="mean")
    width_before = getattr(model, "growable_width", None)
    if max_added is None:
        cap = getattr(model, "target_width", None)
        if cap is not None and width_before is not None:
            max_added = max(1, int(cap) - int(width_before))
    model.set_growing_layers(index=0)
    compute_statistics(model, train_loader, loss_function=crit_sum, device=device)
    # statistical_threshold=0 disables gromo's absolute cut; `numerical_threshold`
    # (untouched, 1e-6) still guards the pseudo-inverse, which is a different job.
    # The selection then happens below, in units of the spectrum itself.
    model.compute_optimal_updates(statistical_threshold=0.0)
    layer = model._growable_layers[0]
    proposed = getattr(layer, "eigenvalues_extension", None)
    n_proposed = int(proposed.shape[0]) if proposed is not None else None
    # Snapshot the spectrum BEFORE the relative floor and the width cap trim it. The
    # sum alone cannot say whether a step was one dominant direction or a flat tail of
    # equivalent ones, which is the question the selection rule turns on -- and
    # `sub_select_optimal_added_parameters` below replaces the attribute, so this is
    # the only moment the untrimmed values exist. Bounded by the target width (<= 40
    # on every arm of the grid), so the record stays small.
    eig_proposed = _spectrum(proposed)
    keep = _n_candidates_to_keep(layer, min_singular_ratio, max_added)
    if keep is not None:
        # Candidates come out of the SVD in descending singular-value order, so
        # keeping the first `keep` is keeping the best `keep`. This also trims the
        # previous layer's matching outputs (sub_select_previous=True) -> hard cap.
        layer.sub_select_optimal_added_parameters(keep_neurons=keep)
    if direction == "random":
        _randomize_candidates(layer)
    if normalization is not None:
        # gromo bakes sqrt(singular value) into the extension weights, so new neurons
        # arrive at ~1e-2 amplitude and the line search never leaves the linear regime
        # (step 0, addendum 1: s = 4 wins 81-99 % of steps, loss linear in s). gromo's
        # own normalisation rescales each component to the std of the layer it extends
        # BEFORE the line search, so `s` becomes a relative amplitude. None = historical.
        _normalize(layer, normalization)
    # Read the two diagnostics off the layer NOW: `apply_change`/`delete_update` both
    # clear them, so after the branch below there is nothing left to record.
    diag = _update_diagnostics(layer)
    model.reset_computation()
    model.dummy_select_update()
    select_loader = train_loader if val_loader is None else val_loader
    best_loss, best_s = float("inf"), 0.0
    losses = {}
    for s in grid:
        model.set_scaling_factor(s)
        loss, _ = evaluate_model(model, select_loader, crit_mean,
                                 use_extended_model=True, device=device)
        losses[s] = float(loss)
        if loss < best_loss:
            best_loss, best_s = loss, s
    ls_s = best_s
    if fixed_s is not None:
        best_s = float(fixed_s)
        model.set_scaling_factor(best_s)
        best_loss, _ = evaluate_model(model, select_loader, crit_mean,
                                      use_extended_model=True, device=device)
    rescale_factors = None
    if best_s == 0.0:
        # Abstain -- see Notes. `delete_update` is exactly the cleanup gromo's own
        # `apply_change` runs after applying (growing_container.py), so the model is
        # left in the same state as a completed step, minus the change.
        model.set_scaling_factor(0.0)
        layer.delete_update()
        model.currently_updated_layer_index = None
    else:
        norms_before = _out_channel_norms(layer.layer.weight) if rescale else None
        model.set_scaling_factor(best_s)
        model.apply_change()
        _fix_conv_statistic_shapes(model)
        if rescale:
            rescale_factors = _restore_out_channel_norms(model, layer, norms_before)
    return {
        "fixed_s": None if fixed_s is None else float(fixed_s),
        "line_search_s": ls_s,
        "rescale_factors": rescale_factors,
        "rescale_fallbacks": (getattr(model, "_rescale_fallbacks", 0) if rescale else None),
        "s": best_s,
        "applied": best_s != 0.0,
        "direction": direction,
        "select_loss": float(best_loss),
        "losses": losses,
        "width_before": width_before,
        "width_after": getattr(model, "growable_width", None),
        # How many neurons gromo offered before the relative floor and the width cap
        # trimmed them. `n_proposed` >> `n_candidates` means the floor is doing the
        # work; `n_proposed == n_candidates == max_added` means the target is the
        # binding constraint. Distinguishing those two is the whole point of logging it.
        "n_proposed": n_proposed,
        "normalization": normalization,
        # The untrimmed singular values, next to `n_proposed` they came from.
        "eig_proposed": eig_proposed,
        **diag,
    }


def _fix_conv_statistic_shapes(model) -> int:
    """Repair the S/M statistics gromo re-creates with the wrong size after a growth.

    gromo ``Conv2dGrowingModule.layer_in_extension`` (conv2d_growing_module.py l. 1166-1186,
    commits b1ffd4d and 88c20ee) re-creates ``tensor_s``/``tensor_m`` with
    ``(in_channels + use_bias) * kh * kw`` rows, while every update (and the same class's
    own ``update_input_size``, l. 594-600) is ``in_channels * kh * kw + use_bias``. The two
    agree only without bias or with a 1x1 kernel, so a conv WITH bias and a kernel > 1x1
    (TIDNet's Conv(5, 1)) dies on its SECOND growth step: "The update tensor has a
    different size than the tensor statistic" (smoke 802688, 21/10/2026). The statistics
    are empty right after the extension, so re-creating them at the right size changes
    nothing else. Returns how many were repaired."""
    from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
    from gromo.utils.tensor_statistic import TensorStatistic
    fixed = 0
    for mod in model.modules():
        if not isinstance(mod, Conv2dGrowingModule) or not mod.use_bias:
            continue
        n = mod.in_channels * mod.layer.kernel_size[0] * mod.layer.kernel_size[1] + 1
        s = getattr(mod, "_tensor_s", None)
        if s is not None and s._shape is not None and s._shape != (n, n):
            if s.samples:
                raise RuntimeError(f"{s.name} holds {s.samples} samples at a wrong shape")
            mod._tensor_s = TensorStatistic((n, n), update_function=mod.compute_s_update,
                                            name=s.name)
            fixed += 1
        m = getattr(mod, "tensor_m", None)
        if m is not None and m._shape is not None and m._shape != (n, mod.out_channels):
            if m.samples:
                raise RuntimeError(f"{m.name} holds {m.samples} samples at a wrong shape")
            mod.tensor_m = TensorStatistic((n, mod.out_channels),
                                           update_function=mod.compute_m_update, name=m.name)
            fixed += 1
    return fixed


def _normalize(layer, normalization: str) -> None:
    """gromo's ``normalize_optimal_updates``, or the TINY paper's amplitude."""
    if normalization == "tiny_paper":
        _tiny_paper_normalize(layer)
    else:
        layer.normalize_optimal_updates(normalization_type=normalization)


#: ``direction`` values of :func:`grow_step`. ``"tiny"`` = the directions gromo computes
#: (every historical arm). ``"random"`` = campaign A's direction control, see
#: :func:`_randomize_candidates`.
DIRECTIONS = ("tiny", "random")


@torch.no_grad()
def _randomize_candidates(layer) -> None:
    """Replace the pending candidates' directions of ``layer`` by i.i.d. Gaussian ones.

    Campaign A's direction control (decided 25/10/2026): everything the growth rule
    decides is kept -- WHEN (the schedule), WHERE (the junction, ranked on the TINY
    gains), HOW MANY (the relative floor on the TINY spectrum and the width cap) and HOW
    LARGE (``_tiny_paper_normalize`` runs after this, so the mean squared norm per new
    neuron is the paper's 1e-3 on both sides) -- and only the direction of the new
    neurons' incoming (alpha, bias included) and outgoing (omega) weights is drawn at
    random. ``tiny - random`` therefore measures what the computed directions buy.

    Drawn on the CPU from torch's global generator (seeded per fit by ``seed_all``/
    skorch), then moved, so a seed gives the same directions on every card."""
    omega = layer.extended_input_layer
    alpha = layer.previous_module.extended_output_layer
    if omega is None or alpha is None:
        raise ValueError("direction='random' needs pending candidate neurons")
    for t in [omega.weight, alpha.weight] + ([alpha.bias] if alpha.bias is not None else []):
        t.copy_(torch.randn(t.shape, dtype=t.dtype).to(t.device))


@torch.no_grad()
def _tiny_paper_normalize(layer, sq_norm: float = TINY_PAPER_SQ_NORM) -> None:
    """TINY App. E.3.1 eq. (197)-(198) on the pending candidates of ``layer``, and the
    best update zeroed (Algorithm 1 adds neurons only) -- see ``TINY_PAPER_SQ_NORM``.

    omega = ``layer.extended_input_layer`` (new input columns of the second layer, one
    neuron per index of dim 1), alpha = ``layer.previous_module.extended_output_layer``
    (new output rows of the first layer, one neuron per index of dim 0, its bias
    included: it is part of the neuron's incoming weights). Each is rescaled so that
    ``||.||^2 / K = sq_norm``; with ``fixed_s=1`` gromo then splices them in as is."""
    omega = layer.extended_input_layer
    alpha = layer.previous_module.extended_output_layer
    if omega is None or alpha is None:
        raise ValueError("tiny_paper normalisation needs pending candidate neurons")
    k = int(omega.weight.shape[1])
    if int(alpha.weight.shape[0]) != k:
        raise ValueError(f"alpha has {alpha.weight.shape[0]} neurons, omega {k}")
    for lin, tensors in ((omega, [omega.weight]),
                         (alpha, [alpha.weight] + ([alpha.bias] if alpha.bias is not None
                                                   else []))):
        sq = sum(float(t.double().pow(2).sum()) for t in tensors)
        if not sq > 0:
            raise ValueError("a candidate extension is identically zero: nothing to scale")
        c = (sq_norm / (sq / k)) ** 0.5
        for t in tensors:
            t.mul_(c)
    if layer.optimal_delta_layer is not None:
        layer.optimal_delta_layer.weight.zero_()
        if layer.optimal_delta_layer.bias is not None:
            layer.optimal_delta_layer.bias.zero_()


def _out_channel_norms(w: torch.Tensor) -> torch.Tensor:
    """L2 norm of each output channel (dim 0) of a weight tensor."""
    return w.detach().flatten(1).norm(dim=1)


@torch.no_grad()
def _restore_out_channel_norms(model, layer, norms_before: torch.Tensor) -> list[float]:
    """Scale each output channel of ``layer`` back to ``norms_before`` and compensate
    the BatchNorm that follows it, so the function is unchanged in both train mode
    (batch statistics are scale-free) and eval mode (running mean x c, var x c^2).
    Refuses a layer with a bias or a model without ``post_growth_norm``: then the
    rescaling would change the function and the control would measure something else."""
    bn = getattr(model, "post_growth_norm", None)
    if not isinstance(bn, torch.nn.modules.batchnorm._BatchNorm):
        raise ValueError("rescale needs model.post_growth_norm, the BatchNorm right after "
                         "the growing layer")
    if getattr(layer.layer, "bias", None) is not None:
        raise ValueError("rescale assumes a bias-free growing layer (BatchNorm after it)")
    w = layer.layer.weight
    c = norms_before.to(w.device) / _out_channel_norms(w).clamp_min(1e-30)
    w.mul_(c.view(-1, *([1] * (w.dim() - 1))))
    bn.running_mean.mul_(c)
    # Eval mode, exact including eps: sqrt(var' + eps) = c * sqrt(var + eps). When that
    # would go negative (c^2 (var + eps) < eps) no running variance can reproduce it --
    # eps does not scale with the weights -- and we fall back to the naive var * c^2.
    # Train mode (batch statistics) is exact up to eps at the restored scale, which is
    # the scale the net trained at before the step; the running stats are re-estimated
    # within a few batches anyway (momentum). `model._rescale_fallbacks` counts them.
    exact = c * c * (bn.running_var + bn.eps) - bn.eps
    naive = c * c * bn.running_var
    fb = exact < 0
    bn.running_var.copy_(torch.where(fb, naive, exact))
    model._rescale_fallbacks = getattr(model, "_rescale_fallbacks", 0) + int(fb.sum())
    return [float(x) for x in c.cpu()]


def _dump_failed_statistics(model) -> None:
    """Diagnose a pinv that fails in float64 too (job 787665): are S and M finite, are
    the weights finite, what is S's spectrum on CPU? Printed, and saved under
    ``$EEGROW_DUMP_DIR`` when it is set. Never raises."""
    import os
    try:
        rows = []
        for layer in model._growing_layers:
            s, m = layer.tensor_s().detach().double().cpu(), layer.tensor_m().detach().double().cpu()
            row = {"junction": model.junction_of(layer), "S_shape": list(s.shape),
                   "S_finite": bool(torch.isfinite(s).all()),
                   "M_finite": bool(torch.isfinite(m).all()),
                   "S_absmax": float(s.abs().nan_to_num(0).max())}
            if row["S_finite"]:
                ev = torch.linalg.eigvalsh((s + s.t()) / 2)
                row.update(S_eig_max=float(ev.max()), S_eig_min=float(ev.min()),
                           S_rank_1e12=int((ev > ev.max() * 1e-12).sum()))
            rows.append((row, s, m))
        params_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        print(f"[pinv-failure] params_finite={params_finite} "
              f"{[r for r, _, _ in rows]}", flush=True)
        out = os.environ.get("EEGROW_DUMP_DIR")
        if out:
            os.makedirs(out, exist_ok=True)
            torch.save({"rows": [r for r, _, _ in rows], "S": [s for _, s, _ in rows],
                        "M": [m for _, _, m in rows], "params_finite": params_finite},
                       os.path.join(out, f"pinv_failure_{os.getpid()}.pt"))
    except Exception as exc:   # a diagnostic must not mask the real error
        print(f"[pinv-failure] diagnostic itself failed: {exc!r}", flush=True)


def _junction_candidates(model, loader, device, *, min_singular_ratio: float) -> list:
    """Statistics on ``loader``, optimal updates on EVERY growable junction, then the
    per-junction relative floor and width cap. The updates are left in place; returns
    one diagnostics dict per entry of ``model._growing_layers``, in that order.

    The floor and cap are applied BEFORE any selection: gromo's own flow trims inside
    ``compute_optimal_updates`` (``maximum_added_neurons``, ``statistical_threshold``)
    and only then ranks, so this is the native order with our relative floor in place
    of its absolute cut. Normalisation is NOT applied here: ``normalize_optimal_updates``
    rescales ``eigenvalues_extension`` (``scale_layer_extension``), which would turn
    the ranked quantity from a predicted loss decrease into an arbitrary amplitude.
    """
    crit_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    model.set_growing_layers(scheduling_method="all")
    compute_statistics(model, loader, loss_function=crit_sum, device=device)
    # float32 pinv of S fails to converge on CUDA when S is ill-conditioned (cuSOLVER
    # error 46, job 787660: 3/3 cells, always on a split-half pass, i.e. half the
    # samples). gromo takes the dtype of that computation as an argument: retry ONCE in
    # float64 and say so in the record; a second failure propagates.
    retried = False
    try:
        model.compute_optimal_updates(statistical_threshold=0.0)
    except torch.linalg.LinAlgError:
        try:
            model.compute_optimal_updates(statistical_threshold=0.0, dtype=torch.float64)
        except torch.linalg.LinAlgError:
            # Still failed (job 787665: 3/3 cells, on the full pass, not reproduced by
            # the same seed in 787668 -> intermittent). Last resort: the same float64
            # pinv through MAGMA instead of cuSOLVER. Same quantity, other SVD driver.
            _dump_failed_statistics(model)
            if not torch.cuda.is_available():
                raise
            previous = torch.backends.cuda.preferred_linalg_library()
            torch.backends.cuda.preferred_linalg_library("magma")
            try:
                model.compute_optimal_updates(statistical_threshold=0.0,
                                              dtype=torch.float64)
            finally:
                torch.backends.cuda.preferred_linalg_library(previous)
            print("[pinv-failure] recovered with MAGMA", flush=True)
        retried = True
    out = []
    for layer in model._growing_layers:
        proposed = getattr(layer, "eigenvalues_extension", None)
        n_proposed = int(proposed.shape[0]) if proposed is not None else None
        eig_proposed = _spectrum(proposed)
        keep = _n_candidates_to_keep(layer, min_singular_ratio, model.remaining(layer))
        if keep is not None:
            layer.sub_select_optimal_added_parameters(keep_neurons=keep)
        d = _update_diagnostics(layer)
        d.update(junction=model.junction_of(layer), n_proposed=n_proposed,
                 eig_proposed=eig_proposed, fp64_retry=retried)
        out.append(d)
    model.reset_computation()
    return out


def _argmax_junction(per: list) -> int:
    """Index (into ``per``) of the largest first-order improvement; refuses a None."""
    fois = [d["first_order_improvement"] for d in per]
    if any(f is None for f in fois):
        # A junction whose gain cannot be computed cannot be ranked; ranking the others
        # without it would silently starve it. Refuse instead of degrading.
        raise RuntimeError(f"first_order_improvement unavailable on a junction: {fois}")
    return max(range(len(fois)), key=lambda i: fois[i])


def _grow_step_best(model, train_loader, device, *, val_loader, min_singular_ratio,
                    grid, normalization, split_half, sequential=False,
                    fixed_s: float | None = None, direction: str = "tiny") -> dict:
    """``grow_step`` for several junctions: gromo's ``select_best_update``.

    1. candidates on every junction still under its target (:func:`_junction_candidates`);
    2. ``model.select_best_update()`` keeps the junction with the largest
       ``first_order_improvement`` (= ``parameter_update_decrease`` + sum of the
       squared singular values of the kept neurons) and deletes the others' updates;
    3. normalisation, line search, abstention and ``apply_change`` exactly as the
       single-junction path, on the selected junction only.

    ``split_half`` (measurement, étape 1): before the real step, the same candidates
    are computed on each half of the statistics batches and the two argmaxes are
    recorded. If they disagree often, the selection reads noise, not a bottleneck.
    The real step is unaffected: the half-updates are deleted and the statistics are
    recomputed on all batches.
    """
    crit_mean = torch.nn.CrossEntropyLoss(reduction="mean")
    width_before = model.growable_width
    widths_before = list(model.junction_widths)
    batches = train_loader if isinstance(train_loader, list) else list(train_loader)
    halves = {}
    if split_half:
        if len(batches) >= 2:
            cut = len(batches) // 2
            for tag, part in (("a", batches[:cut]), ("b", batches[cut:])):
                try:
                    per_h = _junction_candidates(model, part, device,
                                                 min_singular_ratio=min_singular_ratio)
                    halves[tag] = {"junction": per_h[_argmax_junction(per_h)]["junction"],
                                   "foi": [d["first_order_improvement"] for d in per_h],
                                   "fp64_retry": per_h[0]["fp64_retry"]}
                except torch.linalg.LinAlgError:
                    # A measurement, not the step: record the failure, never kill the
                    # fit. The reader counts these (`grow_split_failed`).
                    model.reset_computation()
                    halves[tag] = None
                for layer in model._growing_layers:
                    layer.delete_update()
        else:
            halves = {"a": None, "b": None}
    per = _junction_candidates(model, batches, device,
                               min_singular_ratio=min_singular_ratio)
    expected = _argmax_junction(per)
    if sequential:
        # Round-robin over the junctions still under their target: the next one after
        # the junction grown (or abstained on) last time. The gains are still computed
        # and recorded, so the reader can tell what the argmax would have chosen.
        ids = [d["junction"] for d in per]
        last = getattr(model, "_sequential_last", -1)
        nxt = min((j for j in ids if j > last), default=min(ids))
        idx = model.select_update(ids.index(nxt))
        model._sequential_last = nxt
    else:
        idx = model.select_best_update()
        if idx != expected:   # same quantity, same order: a mismatch is a bug, not noise
            raise RuntimeError(f"select_best_update chose {idx}, argmax is {expected}")
    layer = model.currently_updated_layer
    if direction == "random":
        # AFTER the junction is chosen on the TINY gains: only the direction changes.
        _randomize_candidates(layer)
    if normalization is not None:
        _normalize(layer, normalization)
    diag = _update_diagnostics(layer)
    select_loader = batches if val_loader is None else val_loader
    best_loss, best_s = float("inf"), 0.0
    losses = {}
    for s in grid:
        model.set_scaling_factor(s)
        loss, _ = evaluate_model(model, select_loader, crit_mean,
                                 use_extended_model=True, device=device)
        losses[s] = float(loss)
        if loss < best_loss:
            best_loss, best_s = loss, s
    ls_s = best_s
    if fixed_s is not None:
        # Same contract as the single-junction path: the line search is still run and
        # recorded (`line_search_s`), the applied amplitude is the imposed one.
        best_s = float(fixed_s)
        model.set_scaling_factor(best_s)
        best_loss, _ = evaluate_model(model, select_loader, crit_mean,
                                      use_extended_model=True, device=device)
    if best_s == 0.0:
        model.set_scaling_factor(0.0)
        layer.delete_update()
        model.currently_updated_layer_index = None
    else:
        model.set_scaling_factor(best_s)
        model.apply_change()
        _fix_conv_statistic_shapes(model)
    model.refresh_growable()
    chosen = per[idx]
    return {
        "fixed_s": None if fixed_s is None else float(fixed_s),
        "line_search_s": ls_s,
        "s": best_s,
        "applied": best_s != 0.0,
        "direction": direction,
        "select_loss": float(best_loss),
        "losses": losses,
        "width_before": width_before,
        "width_after": model.growable_width,
        "n_proposed": chosen["n_proposed"],
        "normalization": normalization,
        "eig_proposed": chosen["eig_proposed"],
        **diag,
        # Multi-junction record: which junction won, every junction's ranked gain (raw,
        # before normalisation), what each proposed, and the widths after the step.
        "junction": chosen["junction"],
        "junction_argmax": per[expected]["junction"],
        "junction_foi": [d["first_order_improvement"] for d in per],
        "junction_ids": [d["junction"] for d in per],
        "junction_n_proposed": [d["n_proposed"] for d in per],
        "junction_n_kept": [d["n_candidates"] for d in per],
        "junction_widths_before": widths_before,
        "junction_widths_after": list(model.junction_widths),
        "split_a": None if not halves or halves.get("a") is None else halves["a"]["junction"],
        "split_b": None if not halves or halves.get("b") is None else halves["b"]["junction"],
        "split_foi_a": None if not halves or halves.get("a") is None else halves["a"]["foi"],
        "split_foi_b": None if not halves or halves.get("b") is None else halves["b"]["foi"],
        "fp64_retry": chosen["fp64_retry"],
        "split_failed": None if not split_half else
        sum(halves.get(t) is None for t in ("a", "b")) if len(batches) >= 2 else None,
        "split_fp64_retry": None if not split_half else
        sum(bool(h and h.get("fp64_retry")) for h in halves.values()),
    }


def _spectrum(values, ndigits: int = 8) -> list | None:
    """A tensor of singular values as a plain list of rounded floats.

    Rounded because these end up in a JSONL line: full float repr triples the size of
    a growth epoch's record for digits no analysis reads. Guarded like every other
    diagnostic -- a spectrum that cannot be read is a ``None`` in a log line, never a
    dead fit.
    """
    if values is None:
        return None
    try:
        return [round(float(v), ndigits) for v in values.detach().flatten().cpu()]
    except Exception:  # noqa: BLE001 -- diagnostics never break a fit
        return None


def _n_candidates_to_keep(layer, min_singular_ratio: float,
                          max_added: int | None) -> int | None:
    """How many of the proposed neurons to keep: relative floor, then the width cap.

    Returns ``None`` when there is nothing to trim, which is the signal to skip
    ``sub_select_optimal_added_parameters`` entirely -- calling it with the count it
    already has is a no-op that still rebuilds the tensors.

    At least one neuron is always kept when any candidate exists: a step that survived
    the numerical threshold has *something* to offer, and whether it is worth applying
    is the line search's decision to make, not this function's.
    """
    eig = getattr(layer, "eigenvalues_extension", None)
    if eig is None or eig.numel() == 0:
        return None
    n_total = int(eig.shape[0])
    smax = float(eig.max())
    if smax <= 0:
        keep = 1
    else:
        keep = int((eig >= min_singular_ratio * smax).sum())
    keep = max(1, keep)
    if max_added is not None and max_added >= 1:
        keep = min(keep, max_added)
    return keep if keep < n_total else None


def _update_diagnostics(layer) -> dict:
    """gromo's own verdict on the update it just proposed, as plain floats.

    Three quantities, in increasing order of how likely they are to be unavailable:

    ``eigenvalues_extension``
        the per-neuron singular values the candidates were ranked on. Always there
        once ``compute_optimal_updates`` has run; their sum is the extension's own
        share of the predicted gain, and their spread is what the relative floor in
        :func:`_n_candidates_to_keep` cuts on.
    ``parameter_update_decrease``
        the predicted first-order gain from the optimal *delta* alone -- the part of
        the step that improves the weights already there.
    ``first_order_improvement``
        delta + new neurons, gromo's total. Often unavailable here: computing the
        new-neuron half goes through ``activation_gradient``, which probes the
        activation numerically and raises for a ``BatchNorm2d`` junction (it feeds a
        scalar to a layer that wants 4D). Three of our four arms grow at exactly such
        a junction, so this is the normal case, not a failure.

    Together they separate a step the line search scored at s=0 because it had nothing
    to offer from one that had a real direction the held-out loss refused to pay for --
    two different problems with two different fixes.

    Every read is guarded, and guarded broadly. These are properties that assert,
    numerically differentiate and allocate; a diagnostic that can abort a fit is worse
    than no diagnostic, so the failure mode is a ``None`` in a log line.
    """
    out = {"first_order_improvement": None, "parameter_update_decrease": None,
           "eigenvalues_extension_sum": None, "n_candidates": None,
           "eig_kept": None}
    for key, get in (
        ("eigenvalues_extension_sum", lambda: float(layer.eigenvalues_extension.sum())),
        ("n_candidates", lambda: int(layer.eigenvalues_extension.shape[0])),
        ("eig_kept", lambda: _spectrum(layer.eigenvalues_extension)),
        ("parameter_update_decrease",
         lambda: float(layer.parameter_update_decrease)),
        ("first_order_improvement", lambda: float(layer.first_order_improvement)),
    ):
        try:
            out[key] = get()
        except Exception:  # noqa: BLE001 -- see docstring: never break the fit
            pass
    return out


def run_model(name, model, train_loader, test_loader, *, grow: bool,
              cfg: TrainConfig, device, log_fn=print) -> dict:
    """Train a model (with or without growth) and return the history."""
    crit = torch.nn.CrossEntropyLoss()
    hist = {"epoch": [], "test_acc": [], "params": [], "name": name}
    log_fn(f"\n--- {name}: start width={model.growable_width}, "
           f"{count_params(model):,} params ---")
    t0 = time.time()
    opt = make_optimizer(model, cfg)
    for epoch in range(1, cfg.total_epochs + 1):
        train_epoch(model, train_loader, opt, crit, device)
        if grow and epoch % cfg.grow_every == 0 and epoch < cfg.total_epochs:
            grow_step(model, train_loader, device)
            opt, kept = rebuild_optimizer_preserving_state(opt, model, cfg)
        acc = accuracy(model, test_loader, device)
        hist["epoch"].append(epoch)
        hist["test_acc"].append(acc)
        hist["params"].append(count_params(model))
        if epoch % cfg.log_every == 0 or epoch == 1:
            log_fn(f"  [{name:>10}] epoch {epoch:>3}  test_acc={acc:.3f}  "
                   f"width={model.growable_width}  "
                   f"params={count_params(model):,}")
    hist["seconds"] = time.time() - t0
    hist["final_params"] = count_params(model)
    log_fn(f"  [{name}] done in {hist['seconds']:.0f}s -- "
           f"final acc {hist['test_acc'][-1]:.3f}, best {max(hist['test_acc']):.3f}, "
           f"final width {model.growable_width}, "
           f"{count_params(model):,} params")
    return hist
