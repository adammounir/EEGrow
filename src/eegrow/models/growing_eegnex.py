"""GrowingEEGNeX -- a *growable* (gromo) analog of EEGNeX.

EEGNeX (braindecode, Chen et al. 2024) opens with **two stacked temporal
convolutions** that build a learned FIR-like filter bank, with only a BatchNorm
between them and *no pooling*:

    block_1 : conv (1  -> filter_1)  (1 x L)  -> BN
    block_2 : conv (f1 -> filter_2)  (1 x L)  -> BN

That ``block_1 -> block_2`` junction is the gromo textbook case, so we grow
``filter_1`` (the intermediate temporal width -- the "8" in EEGNeX-8,32). The
BatchNorm between the two convs is carried as ``block_1``'s ``post_layer_function``
(a :class:`GrowingBatchNorm2d`), so it grows together with the conv, exactly as
gromo's own VGG container does.

Everything downstream (the depthwise spatial conv, the dilated temporal stack, the
classifier) is kept identical to braindecode as *fixed* modules: their dimensions
do not depend on ``filter_1``.

Faithful analog, not bit-exact
------------------------------
braindecode's two temporal convs use ``padding="same"`` with an **even** kernel
(L=64), which torch realises with *asymmetric* padding. gromo's growth statistics
require an **integer** (symmetric) padding (it calls ``F.unfold(padding=...)`` and
``2 * padding`` when sizing the extension), so the two growable convs here use
symmetric padding instead. The architecture, blocks, dilations and classifier are
otherwise identical; only the exact temporal alignment of the first two convs
differs (a one-sample edge effect). This mirrors how ``GrowingDeepEEGNet`` is a
faithful analog of Deep4Net rather than a bit-exact copy. The downstream
fixed blocks keep braindecode's ``padding="same"`` (plain ``nn.Conv2d`` -- only the
*growable* convs are constrained by gromo).
"""

from __future__ import annotations

import torch
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint
from einops.layers.torch import Rearrange
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm2d
from torch import nn

# RestrictedConv2dGrowingModule: the base Conv2dGrowingModule does not implement
# fan-in growth (compute_m_prev_update); only the Restricted/Full variants do.
Conv2dGrowingModule = RestrictedConv2dGrowingModule


class GrowingEEGNeX(SequentialGrowingModel):
    """EEGNeX whose ``filter_1`` (first temporal filter bank) width can grow.

    Parameters
    ----------
    n_chans, n_outputs, n_times : int
        Signal parameters (same conventions as braindecode).
    filter_1 : int
        INITIAL width of the first temporal conv (= fan-in of the second temporal
        conv). This is the axis gromo grows.
    filter_2 : int
        Width of the second temporal conv (fixed here).
    target_filter_1 : int | None
        TARGET width to grow towards. If None or == filter_1, the model is frozen.
    depth_multiplier, kernel_block_1_2, kernel_block_4, dilation_block_4,
    avg_pool_block4, kernel_block_5, dilation_block_5, avg_pool_block5,
    drop_prob, max_norm_conv, max_norm_linear, activation :
        Same meaning and defaults as braindecode's EEGNeX.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        filter_1: int = 8,
        filter_2: int = 32,
        target_filter_1: int | None = None,
        depth_multiplier: int = 2,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        drop_prob: float = 0.5,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
        activation: type[nn.Module] = nn.ELU,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_2 * depth_multiplier

        target = filter_1 if target_filter_1 is None else target_filter_1
        self._can_grow = target > filter_1
        self.target_width = target  # cap for the growth callback (gromo does not enforce it)

        # symmetric integer padding for the growable temporal convs (gromo cannot
        # use torch's asymmetric "same"; see module docstring).
        pad_w = (kernel_block_1_2 - 1) // 2

        # --- head: reshape, identical to braindecode -------------------------
        # (B, C, T) -> (B, 1, C, T): n_chans on the height axis, time on width.
        self.ensure_dim = Rearrange("batch ch time -> batch 1 ch time")

        # --- the two growable temporal convs ---------------------------------
        # conv1: (1 -> filter_1). Its OUTPUT grows when conv2 grows its fan-in. The
        # BatchNorm of block_1 lives in the junction -> carried as a growing
        # post_layer_function.
        self.conv1 = Conv2dGrowingModule(
            in_channels=1,
            out_channels=filter_1,
            kernel_size=(1, kernel_block_1_2),
            padding=(0, pad_w),
            use_bias=False,
            post_layer_function=GrowingBatchNorm2d(filter_1, device=self.device),
            previous_module=None,
            allow_growing=False,
            input_size=(n_chans, n_times),
            name="conv1",
            device=self.device,
        )
        t1 = n_times + 2 * pad_w - (kernel_block_1_2 - 1)
        # conv2: (filter_1 -> filter_2). The growable junction: growing its fan-in
        # grows filter_1 (= conv1 fan-out). bias=False (a BN follows in the tail).
        self.conv2 = Conv2dGrowingModule(
            in_channels=filter_1,
            out_channels=filter_2,
            kernel_size=(1, kernel_block_1_2),
            padding=(0, pad_w),
            use_bias=False,
            post_layer_function=nn.Identity(),
            previous_module=self.conv1,
            allow_growing=self._can_grow,
            input_size=(n_chans, t1),
            target_in_channels=target,
            name="conv2",
            device=self.device,
        )

        # --- fixed tail: braindecode blocks 2(BN)/3/4/5 + classifier ---------
        # block_2's BatchNorm (conv2 carries Identity as post).
        self.bn2 = nn.BatchNorm2d(filter_2)

        # block_3: depthwise spatial conv (collapses the n_chans axis) + condense.
        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=filter_2,
                out_channels=self.filter_3,
                max_norm=max_norm_conv,
                kernel_size=(n_chans, 1),
                groups=filter_2,
                bias=False,
            ),
            nn.BatchNorm2d(self.filter_3),
            activation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4), padding=(0, 1)),
            nn.Dropout(p=drop_prob),
        )
        # block_4: dilated temporal conv.
        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_3,
                out_channels=filter_2,
                kernel_size=(1, kernel_block_4),
                dilation=(1, dilation_block_4),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filter_2),
        )
        # block_5: dilated temporal conv + condense + flatten.
        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter_2,
                out_channels=filter_1,
                kernel_size=(1, kernel_block_5),
                dilation=(1, dilation_block_5),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filter_1),
            activation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block5), padding=(0, 1)),
            nn.Dropout(p=drop_prob),
            nn.Flatten(),
        )
        self.to(self.device)

        # classifier: max-norm linear; in_features inferred by a dry run because the
        # symmetric padding shifts the time dimension by a couple of samples.
        in_features = self._infer_in_features(n_chans, n_times)
        self.final_layer = LinearWithConstraint(
            in_features=in_features,
            out_features=n_outputs,
            max_norm=max_norm_linear,
        )
        self.to(self.device)

        # --- register the growable layer with gromo --------------------------
        self._growable_layers = [self.conv2] if self._can_grow else []
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")

    @property
    def growable_width(self) -> int:
        """Width of the growable axis (= filter_1)."""
        return self.conv1.out_channels

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _infer_in_features(self, n_chans: int, n_times: int) -> int:
        """Run head + growable convs + tail (minus classifier) to read the size."""
        self.eval()
        x = torch.zeros(1, n_chans, n_times, device=self.device)
        x = self.ensure_dim(x)
        x = self.conv2(self.conv1(x))
        for module in self._tail():
            x = module(x)
        return x.shape[1]

    def _tail(self) -> list[nn.Module]:
        return [self.bn2, self.block_3, self.block_4, self.block_5]

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_dim(x)
        x = self.conv1(x)  # applies the BatchNorm post_layer_function
        x = self.conv2(x)
        for module in self._tail():
            x = module(x)
        return self.final_layer(x)

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward pass including candidate neurons (for the gromo line search)."""
        x = self.ensure_dim(x)
        x_ext: torch.Tensor | None = None
        x, x_ext = self.conv1.extended_forward(x, x_ext)
        x, x_ext = self.conv2.extended_forward(x, x_ext)
        for module in self._tail():
            x = module(x)
            if x_ext is not None:
                x_ext = module(x_ext)
        x = self.final_layer(x)
        return x
