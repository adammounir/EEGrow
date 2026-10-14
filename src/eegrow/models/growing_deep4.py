"""GrowingDeep4Net -- braindecode's Deep4Net, unchanged, with its first junction growable.

WHY THIS EXISTS (cadrage, étape 1a)
-----------------------------------
``growing_deepeeg.py`` gave up on a faithful Deep4Net on the ground that "gromo cannot
grow a junction across a pooling layer". That was asserted, never tested. Tested on
14/10/2026 (gromo 0.0.1): a ``MaxPool2d`` placed in the ``post_layer_function`` of a
``RestrictedConv2dGrowingModule`` makes the statistics pass FAIL LOUDLY (a reshape
error in the cross tensor), with a stride of 3 and even with a stride of 1. The cause
is read in the source: ``compute_m_prev_update`` contracts the previous layer's
unfolded input, laid out on that layer's OUTPUT grid, with the current layer's
gradient, assuming the current layer's input grid is that same grid; any pooling in
between changes the grid. So no junction of Deep4Net that crosses a pooling can grow
without changing gromo -- the three pooled junctions (blocks 2, 3, 4) stay fixed.

But Deep4Net's FIRST junction crosses no pooling and no nonlinearity: ``conv_time``
(1 -> n_filters_time, kernel (filter_time_length, 1)) then ``conv_spat``
(n_filters_time -> n_filters_spat, kernel (1, n_chans)), fused by braindecode in a
``CombinedConv`` that stores the two weight sets separately -- exactly the junction
``GrowingShallowFBCSPNet`` already grows with parity checked. Growing it grows
``n_filters_time`` and nothing else: at its target width the network IS Deep4Net.

HOW
---
The tail (BatchNorm, ELU, MaxPool, the three conv blocks, the classifier) is not
rewritten: it is TAKEN from a ``braindecode.models.Deep4Net`` built with the same
arguments, so it cannot drift from braindecode's. Only the head is replaced by two
gromo bricks. ``load_braindecode_weights`` copies a Deep4Net's weights in, which is
what the parity criterion of étape 1 checks (max |logit difference| <= 1e-5).
"""

from __future__ import annotations

import torch
from braindecode.models import Deep4Net
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from torch import nn

Conv2dGrowingModule = RestrictedConv2dGrowingModule

#: Deep4Net children that come before the tail, replaced by the growable head.
_HEAD = ("ensuredims", "dimshuffle", "conv_time_spat")


class GrowingDeep4Net(SequentialGrowingModel):
    """Deep4Net whose ``n_filters_time`` (first junction) can grow via gromo.

    Parameters
    ----------
    n_chans, n_outputs, n_times : int
        As braindecode.
    n_filters_time : int
        INITIAL width of the temporal conv (the grown axis). braindecode: 25.
    target_n_filters_time : int | None
        Width to grow to; None or == n_filters_time freezes the model.
    **deep4_kwargs
        Any other ``Deep4Net`` argument (n_filters_spat, filter_time_length,
        final_conv_length, drop_prob, ...), passed through unchanged.
    """

    def __init__(self, n_chans: int, n_outputs: int, n_times: int,
                 n_filters_time: int = 25, target_n_filters_time: int | None = None,
                 device: torch.device | str | None = None, **deep4_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        if deep4_kwargs.get("split_first_layer") is False:
            raise ValueError("the growable junction is the split first layer")
        ref = Deep4Net(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                       n_filters_time=n_filters_time, **deep4_kwargs)
        cc = ref.conv_time_spat
        self.n_chans, self.n_outputs, self.n_times = n_chans, n_outputs, n_times
        target = n_filters_time if target_n_filters_time is None else target_n_filters_time
        self._can_grow = target > n_filters_time
        self.target_width = target   # read by GromoGrowth to cap growth

        self.ensuredims = ref.ensuredims
        self.dimshuffle = ref.dimshuffle
        k_time = cc.conv_time.kernel_size[0]
        self.conv_time = Conv2dGrowingModule(
            in_channels=1, out_channels=n_filters_time, kernel_size=(k_time, 1),
            use_bias=cc.conv_time.bias is not None, post_layer_function=nn.Identity(),
            previous_module=None, allow_growing=False, input_size=(n_times, n_chans),
            name="conv_time", device=self.device)
        self.conv_spat = Conv2dGrowingModule(
            in_channels=n_filters_time, out_channels=cc.conv_spat.out_channels,
            kernel_size=(1, n_chans), use_bias=cc.conv_spat.bias is not None,
            post_layer_function=nn.Identity(), previous_module=self.conv_time,
            allow_growing=self._can_grow, input_size=(n_times - k_time + 1, n_chans),
            target_in_channels=target, name="conv_spat", device=self.device)
        # No stride on the head in any configuration: braindecode's CombinedConv.forward
        # hard-codes stride=(1, 1), even under stride_before_pool=True (read in the 1.7
        # source, 14/10). The tail carries every other stride, and it is braindecode's.
        self.tail = nn.Sequential(*[m for n, m in ref.named_children() if n not in _HEAD])
        self.to(self.device)
        self._growable_layers = [self.conv_spat] if self._can_grow else []
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")

    @property
    def growable_width(self) -> int:
        return self.conv_time.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dimshuffle(self.ensuredims(x))
        return self.tail(self.conv_spat(self.conv_time(x)))

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward pass including candidate neurons (for the gromo line search)."""
        x = self.dimshuffle(self.ensuredims(x))
        x_ext: torch.Tensor | None = None
        x, x_ext = self.conv_time.extended_forward(x, x_ext)
        x, x_ext = self.conv_spat.extended_forward(x, x_ext)
        return self.tail(x)

    @torch.no_grad()
    def load_braindecode_weights(self, bd_model: nn.Module) -> "GrowingDeep4Net":
        """Copy a braindecode Deep4Net (same arguments) into this net."""
        cc = bd_model.conv_time_spat
        self.conv_time.layer.weight.copy_(cc.conv_time.weight)
        if cc.conv_time.bias is not None:
            self.conv_time.layer.bias.copy_(cc.conv_time.bias)
        self.conv_spat.layer.weight.copy_(cc.conv_spat.weight)
        if cc.conv_spat.bias is not None:
            self.conv_spat.layer.bias.copy_(cc.conv_spat.bias)
        src = dict(bd_model.named_children())
        for name, m in zip([n for n, _ in bd_model.named_children() if n not in _HEAD],
                           self.tail):
            m.load_state_dict(src[name].state_dict())
        return self
