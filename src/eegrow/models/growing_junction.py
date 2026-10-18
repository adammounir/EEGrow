"""Grow ONE junction of an unchanged braindecode model, in place.

WHY THIS EXISTS (cadrage, étape 1b)
-----------------------------------
``GrowingDeep4Net`` rewrote Deep4Net's head and reused its tail. That works for a
junction at the input, but the junctions the audit found in the other braindecode
models (``braindecode-growable-junctions``) sit in the MIDDLE of a forward that
braindecode writes itself: the FFN of a transformer block behind a residual add, the
first two convolutions of a ContraWR residual block, the bottleneck of a TIDNet dense
filter. Rewriting those forwards would be a re-implementation, and a re-implementation
is exactly what the parity criterion exists to catch.

So the model is NOT rewritten. The braindecode instance is kept whole, and the junction
``first -> (point-wise ops) -> second`` is replaced, in its parent, by a single module,
:class:`GrowingPair`, that computes the same function with two gromo bricks. Every other
module, and every ``forward``, is braindecode's own code. A gromo junction must have
only point-wise operations between its two layers (activation, normalisation, dropout:
the ``post_layer_function``) and one consumer for its hidden width; each subclass names
the junction and the audit criteria it was read against.

During the line search, ``extended_forward`` flips the pair to its extended mode and
runs braindecode's forward unchanged, so the candidate neurons are seen through the
very same residuals, poolings and heads as the real ones.
"""

from __future__ import annotations

import torch
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from torch import nn


class GrowingPair(nn.Module):
    """``second(first(x))`` with ``first``'s output (the hidden width) growable."""

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        self.first, self.second = first, second
        self.extended = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.extended:
            return self.second(self.first(x))
        x, x_ext = self.first.extended_forward(x, None)
        x, _ = self.second.extended_forward(x, x_ext)
        return x


class JunctionGrowingModel(SequentialGrowingModel):
    """A braindecode model (``self.net``) with one :class:`GrowingPair` inside.

    Subclasses build ``self.net`` and ``self.pair`` and call :meth:`_finish`.
    """

    def _finish(self, *, start: int, target: int) -> None:
        if not 1 <= start <= target:
            raise ValueError(f"need 1 <= start ({start}) <= target ({target})")
        self._can_grow = target > start
        self.target_width = target   # read by GromoGrowth to cap growth
        self.to(self.device)
        self._growable_layers = [self.pair.second] if self._can_grow else []
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")

    @property
    def growable_width(self) -> int:
        return self.pair.second.in_neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward pass including the candidate neurons (gromo line search)."""
        self.pair.extended = True
        try:
            return self.net(x)
        finally:
            self.pair.extended = False


def _pin_activation_gradient(pair: GrowingPair) -> None:
    """Set gromo's cached sigma'(0+) of the junction to its deterministic value.

    gromo multiplies the new neurons' share of ``first_order_improvement`` by the
    derivative at 0+ of the junction's ``post_layer_function``, computed ONCE and
    cached forever (``GrowingModule.activation_gradient``). A module it does not know
    is differentiated numerically -- including ``Dropout``, which in train mode returns
    0 or 1/(1-p) at random. Measured on 15/10/2026 (MSVTNet, p = 0.5, 20 seeds, train
    mode): the cached factor was 0 or 2, different between the two junctions in 11/20
    seeds. With one junction it only skews a logged diagnostic; once
    ``select_best_update`` RANKS junctions on it, a junction drawn at 0 is starved for
    the whole fit. Dropout is skipped here (inverted dropout has derivative 1 in
    expectation, and exactly 1 in eval mode, which is what gromo gets in eval);
    BatchNorm is skipped as gromo does; everything else uses gromo's table or the same
    numerical derivative gromo would take, on a deterministic module.
    """
    from gromo.modules.growing_module import GRADIENT_COMPUTATION_EPSILON
    from gromo.utils.utils import known_activations_zero_plus_gradient as known

    f = pair.first.post_layer_function
    mods = list(f) if isinstance(f, nn.Sequential) else [f]
    value = 1.0
    for m in mods:
        if isinstance(m, (nn.modules.dropout._DropoutNd,
                          nn.modules.batchnorm._BatchNorm)):
            continue
        if type(m) in known:
            value *= float(known[type(m)])
        else:
            value *= float(torch.func.grad(m)(torch.tensor(GRADIENT_COMPUTATION_EPSILON)))
    if not value > 0:
        raise ValueError(f"sigma'(0+) of junction {pair.second.name} is {value}: its "
                         "new neurons would never count in the ranking")
    pair.second._activation_gradient_previous_module = torch.tensor(value)


class MultiJunctionGrowingModel(SequentialGrowingModel):
    """A braindecode model (``self.net``) with SEVERAL :class:`GrowingPair` inside.

    Lecture (a) of étape 1 (15/10/2026): every junction gromo can grow is growable,
    and ONE grows per step -- the one gromo's own ``select_best_update`` picks, i.e. the
    largest ``first_order_improvement`` (``growing_container.py``). Each junction has
    its own target; a junction that reaches it leaves ``_growable_layers``, so the
    others keep competing for the rest of the budget.

    ``growable_width`` and ``target_width`` are the SUMS over junctions, so the growth
    callback's total cap and the recorder keep their meaning; the per-junction widths
    are ``junction_widths`` / ``junction_targets``.

    Subclasses build ``self.net`` and the pairs, and call :meth:`_finish_multi`.
    """

    def _finish_multi(self, pairs: list[GrowingPair], *, start: int,
                      targets: list[int]) -> None:
        if len(pairs) != len(targets) or not pairs:
            raise ValueError("need one target per pair, and at least one pair")
        for t in targets:
            if not 1 <= start <= t:
                raise ValueError(f"need 1 <= start ({start}) <= target ({t})")
        self.pairs = nn.ModuleList(pairs)
        for p in self.pairs:
            _pin_activation_gradient(p)
        self.junction_targets = [int(t) for t in targets]
        self.target_width = sum(self.junction_targets)  # read by GromoGrowth
        self.to(self.device)
        self.refresh_growable()

    def refresh_growable(self) -> None:
        """Junctions still under their target; call after every applied growth."""
        self._growable_layers = [p.second for p, t in
                                 zip(self.pairs, self.junction_targets)
                                 if p.second.in_neurons < t]
        self._can_grow = bool(self._growable_layers)
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")
        else:
            self._growing_layers = []

    def junction_of(self, layer) -> int:
        """Index in ``self.pairs`` of a growing layer (identity, not equality)."""
        for i, p in enumerate(self.pairs):
            if p.second is layer:
                return i
        raise ValueError("layer is not the second module of any junction")

    def remaining(self, layer) -> int:
        i = self.junction_of(layer)
        return self.junction_targets[i] - self.pairs[i].second.in_neurons

    @property
    def junction_widths(self) -> list[int]:
        return [int(p.second.in_neurons) for p in self.pairs]

    @property
    def growable_width(self) -> int:
        return sum(self.junction_widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward with candidate neurons. Junctions without an update stay plain:
        gromo's ``extended_forward`` returns ``(activity, None)`` when nothing is
        pending, so only the selected junction contributes candidates."""
        for p in self.pairs:
            p.extended = True
        try:
            return self.net(x)
        finally:
            for p in self.pairs:
                p.extended = False


def _copy_linear(dst, src: nn.Linear) -> None:
    dst.layer.weight.copy_(src.weight)
    if src.bias is not None:
        dst.layer.bias.copy_(src.bias)


def _copy_bn(dst: nn.modules.batchnorm._BatchNorm, src) -> None:
    dst.load_state_dict(src.state_dict())
