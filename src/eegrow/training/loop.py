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


SCALING_GRID = (0.0, 0.1, 0.5, 1.0)


#: Keep a candidate neuron when its singular value is at least this fraction of the
#: largest one. Replaces gromo's ``statistical_threshold``, which is an ABSOLUTE cut
#: (default 1e-3) applied to a quantity whose scale is set by the loss gradients of
#: the data -- see ``grow_step``'s Notes for the measurement that made this necessary.
MIN_SINGULAR_RATIO = 0.1


def grow_step(model, train_loader, device, *, val_loader=None,
              max_added: int | None = None,
              min_singular_ratio: float = MIN_SINGULAR_RATIO) -> dict:
    """One gromo growth: stats -> optimal update -> line search -> apply_change.

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
    keep = _n_candidates_to_keep(layer, min_singular_ratio, max_added)
    if keep is not None:
        # Candidates come out of the SVD in descending singular-value order, so
        # keeping the first `keep` is keeping the best `keep`. This also trims the
        # previous layer's matching outputs (sub_select_previous=True) -> hard cap.
        layer.sub_select_optimal_added_parameters(keep_neurons=keep)
    # Read the two diagnostics off the layer NOW: `apply_change`/`delete_update` both
    # clear them, so after the branch below there is nothing left to record.
    diag = _update_diagnostics(layer)
    model.reset_computation()
    model.dummy_select_update()
    select_loader = train_loader if val_loader is None else val_loader
    best_loss, best_s = float("inf"), 0.0
    losses = {}
    for s in SCALING_GRID:
        model.set_scaling_factor(s)
        loss, _ = evaluate_model(model, select_loader, crit_mean,
                                 use_extended_model=True, device=device)
        losses[s] = float(loss)
        if loss < best_loss:
            best_loss, best_s = loss, s
    if best_s == 0.0:
        # Abstain -- see Notes. `delete_update` is exactly the cleanup gromo's own
        # `apply_change` runs after applying (growing_container.py), so the model is
        # left in the same state as a completed step, minus the change.
        model.set_scaling_factor(0.0)
        layer.delete_update()
        model.currently_updated_layer_index = None
    else:
        model.set_scaling_factor(best_s)
        model.apply_change()
    return {
        "s": best_s,
        "applied": best_s != 0.0,
        "select_loss": float(best_loss),
        "losses": losses,
        "width_before": width_before,
        "width_after": getattr(model, "growable_width", None),
        # How many neurons gromo offered before the relative floor and the width cap
        # trimmed them. `n_proposed` >> `n_candidates` means the floor is doing the
        # work; `n_proposed == n_candidates == max_added` means the target is the
        # binding constraint. Distinguishing those two is the whole point of logging it.
        "n_proposed": n_proposed,
        **diag,
    }


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
           "eigenvalues_extension_sum": None, "n_candidates": None}
    for key, get in (
        ("eigenvalues_extension_sum", lambda: float(layer.eigenvalues_extension.sum())),
        ("n_candidates", lambda: int(layer.eigenvalues_extension.shape[0])),
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
