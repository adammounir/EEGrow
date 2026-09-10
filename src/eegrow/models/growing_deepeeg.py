"""GrowingDeepEEGNet -- a *deep* EEG network with a growable junction (gromo).

A faithful Deep4Net is not growable: a MaxPool between every conv breaks the
junction (gromo cannot grow a junction across a pooling layer, which changes the
spatial dimension). The workaround is a VGG-style staged structure: inside a stage,
several convs are chained *without pooling* (the growable junction); pooling only
happens at the end of a stage (where it breaks the chain -- as expected).

The growable junction here lives in a *deep* stage, i.e. after a pooling has already
happened upstream.

Structure (2 stages)
--------------------
    input (B,C,T) -> (B,1,T,C)
    STAGE 1 :  conv_time (1->w1, kT x 1)
               conv_spat (w1->w1, 1 x C)        <- spatial collapse
               BN -> ELU -> MaxPool(time)       <- breaks the chain
    STAGE 2 :  [Dropout]                        <- opt-in, block start (Deep4Net order)
               conv2a (w1->w2, kT x 1)          <- stage start: previous=None
               [GroupNorm] -> ELU               <- norm opt-in, INSIDE the junction
               conv2b (w2->w2, kT x 1)          <- growable junction (deep)
               BN -> ELU -> MaxPool(time)
    conv classifier (w2->n_outputs) -> squeeze

conv2b grows its FAN-IN (= conv2a out_channels); its OUTPUT stays w2.

REGULARISATION (`drop_prob`, `norm_junction`) -- added 10/09/2026, DEFAULTS OFF
------------------------------------------------------------------------------
Measured on the 3240 fits of pilot 578293 (bnci2014_001, n=9): every arm drives the
train loss to 0.001-0.004 on 230 trials while the *best* valid loss reached is only
0.1-0.2 nats under ln(4)=1.3863, and 18-28% of fits never beat the uniform prior at
all. Widening w2 8->32 (10036 -> 65956 params) divides the final train loss by 4 and
moves the valid optimum from epoch 19 to epoch 11. The arm is not capacity-limited,
it is regularisation-limited -- which is *why* growth cannot pay here.

This file was the only growing architecture with no regularisation at all:
growing_shallow / growing_sccnet / growing_eegnex all carry drop_prob=0.5, and so
does the braindecode Deep4Net this model imitates (3 Dropout modules). Both knobs
therefore default to OFF, so every result produced before this commit is reproduced
bit-for-bit by the defaults; the pilot turns them on explicitly.

Why GroupNorm and not BatchNorm inside the junction. gromo applies
`post_layer_function` *independently* to the main and the candidate branch
(`_apply_extended_post_layer_function`, whose own docstring restricts that path to
"stateless modules such as nn.ReLU"). A BatchNorm there would (a) update its running
stats twice per line-search step and (b) be sized on conv2a's fan-out, which is
exactly what grows. `nn.GroupNorm(1, C, affine=False)` has no parameter and no
buffer, and its forward never reads `num_channels` -- it survives growth unchanged.
Known limitation, stated rather than hidden: during a line search the candidate
branch is normalised over its own channels, not jointly with the main branch, so the
scaling factor gromo selects is parameterised differently from the post-merge model.
That affects the choice of gamma at each growth event, not the merged architecture.
"""

from __future__ import annotations

import torch
from braindecode.modules import Ensure4d, SqueezeFinalOutput
from einops.layers.torch import Rearrange
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from torch import nn

# RestrictedConv2dGrowingModule: the base Conv2dGrowingModule does not implement
# fan-in growth (compute_m_prev_update); only the Restricted/Full variants do.
Conv2dGrowingModule = RestrictedConv2dGrowingModule


def _conv_out(length: int, kernel: int, stride: int = 1) -> int:
    return (length - kernel) // stride + 1


class GrowingDeepEEGNet(SequentialGrowingModel):
    """A deep 2-stage EEG network whose stage-2 junction can grow."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        w1: int = 8,
        w2: int = 16,
        w2_in: int | None = None,
        target_w2: int | None = None,
        filter_time_length: int = 25,
        filter_length_2: int = 10,
        pool_len: int = 3,
        pool_stride: int = 3,
        drop_prob: float = 0.0,
        norm_junction: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.n_chans, self.n_outputs, self.n_times = n_chans, n_outputs, n_times
        # What grows is the *intermediate* width of stage 2 -- conv2a's fan-out, which
        # is conv2b's fan-in. conv2b's OUTPUT stays at w2, and so do bn2 and the
        # classifier. A model grown from w2_in=8 to a target of 32 therefore ends up
        # 8 -> 32 -> 8, not 8 -> 32 -> 32.
        #
        # w2_in exists so that geometry can also be built directly, frozen: it is the
        # only honest fixed control for the growing arm (same architecture, same final
        # widths, no growth). Comparing grow_deep against braindecode's Deep4Net -- 4
        # stages, 25/50/100/200 filters -- compares architectures, not growth.
        # Default None => w2, which is the pre-existing behaviour unchanged.
        w2_in = w2 if w2_in is None else w2_in
        self.w1, self.w2, self.w2_in = w1, w2, w2_in
        target = w2_in if target_w2 is None else target_w2
        self._can_grow = target > w2_in
        # See GrowingShallowFBCSPNet: gromo does not enforce the target, the callback
        # does, and it needs this attribute. Missing here, this model grew to 77 for a
        # target of 32 in nine growth events.
        self.target_width = target

        # reshape (B,C,T) -> (B,1,T,C)
        self.ensuredims = Ensure4d()
        self.dimshuffle = Rearrange("b c t 1 -> b 1 t c")

        # ---- STAGE 1 -------------------------------------------------------
        t1 = _conv_out(n_times, filter_time_length)
        self.conv_time = Conv2dGrowingModule(
            in_channels=1, out_channels=w1, kernel_size=(filter_time_length, 1),
            use_bias=True, post_layer_function=nn.Identity(),
            previous_module=None, allow_growing=False,
            input_size=(n_times, n_chans), name="conv_time", device=self.device,
        )
        # conv_spat: (1, C) kernel collapses the electrode axis.
        self.conv_spat = Conv2dGrowingModule(
            in_channels=w1, out_channels=w1, kernel_size=(1, n_chans),
            use_bias=False, post_layer_function=nn.Identity(),
            previous_module=self.conv_time, allow_growing=False,
            input_size=(t1, n_chans), name="conv_spat", device=self.device,
        )
        self.bn1 = nn.BatchNorm2d(w1)
        self.act1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((pool_len, 1), stride=(pool_stride, 1))
        t1p = _conv_out(t1, pool_len, pool_stride)   # after pooling: chain is broken

        # ---- STAGE 2 (deep, AFTER a pooling) -------------------------------
        # Dropout at the START of the deep block, which is the braindecode Deep4Net
        # order (drop -> conv -> bn -> nonlin -> pool) and, like Deep4Net, block 1 is
        # left bare. It sits OUTSIDE the junction: pool1 has already broken the chain,
        # so gromo never crosses it and the line search never sees a stochastic mask.
        self.drop_prob = float(drop_prob)
        self.drop2 = nn.Dropout(p=self.drop_prob) if self.drop_prob > 0 else nn.Identity()

        # conv2a: stage start -> previous=None, allow_growing=False (chain broken).
        t2 = _conv_out(t1p, filter_length_2)
        # Stateless and width-agnostic -- see the module docstring for why this cannot
        # be a BatchNorm. `num_channels` is passed for readability only; GroupNorm's
        # forward never reads it, which is precisely what lets it survive growth.
        self.norm_junction = bool(norm_junction)
        act2a: nn.Module = nn.ELU()
        if self.norm_junction:
            act2a = nn.Sequential(nn.GroupNorm(1, w2_in, affine=False), nn.ELU())
        self.conv2a = Conv2dGrowingModule(
            in_channels=w1, out_channels=w2_in, kernel_size=(filter_length_2, 1),
            use_bias=False, post_layer_function=act2a,   # activation INSIDE the junction
            previous_module=None, allow_growing=False,
            input_size=(t1p, 1), name="conv2a", device=self.device,
        )
        # conv2b: the growable (deep) junction. Its fan-in is what grows (and with it
        # conv2a's fan-out); its fan-out stays w2, which is what bn2 and the
        # classifier are sized on.
        t2b = _conv_out(t2, filter_length_2)
        self.conv2b = Conv2dGrowingModule(
            in_channels=w2_in, out_channels=w2, kernel_size=(filter_length_2, 1),
            use_bias=False, post_layer_function=nn.Identity(),
            previous_module=self.conv2a, allow_growing=self._can_grow,
            input_size=(t2, 1), target_in_channels=target,
            name="conv2b", device=self.device,
        )
        self.bn2 = nn.BatchNorm2d(w2)
        self.act2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((pool_len, 1), stride=(pool_stride, 1))
        t2p = _conv_out(t2b, pool_len, pool_stride)

        # ---- classifier ----------------------------------------------------
        self.conv_classifier = nn.Conv2d(w2, n_outputs, (t2p, 1), bias=True)
        self.squeeze = SqueezeFinalOutput()
        self.to(self.device)

        # register the deep junction as growable with gromo
        self._growable_layers = [self.conv2b] if self._can_grow else []
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")

    @property
    def growable_width(self) -> int:
        """Width of the growable axis (fan-in of the deep junction conv2b)."""
        return self.conv2a.out_channels

    def _stage1(self, x):
        x = self.conv_spat(self.conv_time(x))
        return self.pool1(self.act1(self.bn1(x)))

    def _tail2(self):
        return [self.bn2, self.act2, self.pool2, self.conv_classifier, self.squeeze]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dimshuffle(self.ensuredims(x))
        x = self._stage1(x)
        x = self.conv2b(self.conv2a(self.drop2(x)))
        for m in self._tail2():
            x = m(x)
        return x

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward pass including candidate neurons (gromo line search).

        Stage 1 and conv2a are upstream of the growing junction, so x_ext does not
        exist yet (None). It is born at conv2b and then propagates through the tail.
        """
        x = self.dimshuffle(self.ensuredims(x))
        x = self._stage1(x)
        x = self.drop2(x)
        x_ext: torch.Tensor | None = None
        x, x_ext = self.conv2a.extended_forward(x, x_ext)
        x, x_ext = self.conv2b.extended_forward(x, x_ext)
        for m in self._tail2():
            x = m(x)
            if x_ext is not None:
                x_ext = m(x_ext)
        return x
