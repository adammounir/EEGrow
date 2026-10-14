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


def _copy_linear(dst, src: nn.Linear) -> None:
    dst.layer.weight.copy_(src.weight)
    if src.bias is not None:
        dst.layer.bias.copy_(src.bias)


def _copy_bn(dst: nn.modules.batchnorm._BatchNorm, src) -> None:
    dst.load_state_dict(src.state_dict())
