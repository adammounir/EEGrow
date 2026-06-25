"""GrowingSCCNet -- a *growable* (gromo) version of SCCNet.

SCCNet (braindecode, Wei et al. 2019) is a spatial-first network: it chains a
*spatial* convolution (projects the ``n_chans`` electrodes onto ``Nu`` components)
and a *spatio-temporal* convolution (mixes across components over a ~0.1 s window),
with **only a BatchNorm between them and no pooling**. That junction is exactly the
gromo textbook case, so we can grow ``Nu`` (``n_spatial_filters``).

Reformulation of the second conv
--------------------------------
braindecode encodes the ``Nu`` components as the *height* dimension of the second
conv: it permutes ``(B, Nu, 1, T) -> (B, 1, Nu, T)`` and uses a conv with a kernel
height ``Nu`` (``in_channels=1``). gromo grows the *fan-in channels* of a conv, not
its kernel height, so we rewrite that identical operation as a channel-wise conv:

    spatial_filt_conv : (Nu -> Nu_smooth)  kernel (1, samples_100ms)

acting directly on ``(B, Nu, 1, T)``. This is mathematically the same map (the
weight is just transposed between the channel and height axes), so growing the
fan-in of ``spatial_filt_conv`` grows ``Nu`` (= the fan-out of ``spatial_conv``).

The BatchNorm that sits *between* the two convs is carried as the
``post_layer_function`` of ``spatial_conv`` (a :class:`GrowingBatchNorm2d`), exactly
as gromo's own VGG container does -- it then grows together with the conv.

braindecode's SCCNet stays the reference baseline: we copy its weights
(:meth:`load_braindecode_weights`), which also lets us check equivalence.
"""

from __future__ import annotations

import math
from warnings import warn

import torch
from braindecode.modules import LogActivation, Square
from einops.layers.torch import Rearrange
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm2d
from torch import nn

# RestrictedConv2dGrowingModule: the base Conv2dGrowingModule does not implement
# fan-in growth (compute_m_prev_update); only the Restricted/Full variants do.
Conv2dGrowingModule = RestrictedConv2dGrowingModule


class GrowingSCCNet(SequentialGrowingModel):
    """SCCNet whose ``n_spatial_filters`` (Nu) width can grow via gromo.

    Parameters
    ----------
    n_chans, n_outputs, n_times : int
        Signal parameters (same conventions as braindecode).
    sfreq : float
        Sampling frequency, used to size the temporal kernel (~0.1 s) and the
        pooling window (~0.5 s) exactly like braindecode's SCCNet.
    n_spatial_filters : int
        INITIAL number of spatial components (= fan-in of ``spatial_filt_conv``).
        This is the axis gromo grows.
    n_spatial_filters_smooth : int
        Number of spatio-temporal filters (fixed here).
    target_n_spatial_filters : int | None
        TARGET width to grow towards. If None or == n_spatial_filters, the model is
        frozen (useful for the equivalence test / baseline).
    drop_prob : float
        Dropout probability.
    activation : type[nn.Module]
        Activation after the second conv block (power-like). Default LogActivation.
    batch_norm_momentum : float
        Momentum of both BatchNorms.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        sfreq: float,
        n_spatial_filters: int = 22,
        n_spatial_filters_smooth: int = 20,
        target_n_spatial_filters: int | None = None,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = LogActivation,
        batch_norm_momentum: float = 0.1,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        self.n_spatial_filters = n_spatial_filters
        self.n_spatial_filters_smooth = n_spatial_filters_smooth

        target = (
            n_spatial_filters
            if target_n_spatial_filters is None
            else target_n_spatial_filters
        )
        self._can_grow = target > n_spatial_filters

        # --- kernel sizing, identical to braindecode SCCNet ------------------
        self.samples_100ms, self.kernel_size_pool = self._calc_kernel_sizes()
        num_features = self._calc_num_features()

        # --- head: reshape, identical to braindecode -------------------------
        # (B, C, T) -> (B, 1, C, T): n_chans on the height axis, time on width.
        self.ensure_dim = Rearrange("batch nchan times -> batch 1 nchan times")

        # --- the two growable convs ------------------------------------------
        # spatial_conv: (1 -> Nu) over all electrodes (kernel height = n_chans);
        # its OUTPUT grows when spatial_filt_conv grows its fan-in. Its
        # post_layer_function is the BatchNorm that sits inside the junction; being
        # a GrowingBatchNorm2d, it grows with the conv (gromo handles this in
        # _grow_post_layer_function).
        self.spatial_conv = Conv2dGrowingModule(
            in_channels=1,
            out_channels=n_spatial_filters,
            kernel_size=(n_chans, 1),
            use_bias=True,
            post_layer_function=GrowingBatchNorm2d(
                n_spatial_filters, momentum=batch_norm_momentum, device=self.device
            ),
            previous_module=None,
            allow_growing=False,
            input_size=(n_chans, n_times),
            name="spatial_conv",
            device=self.device,
        )
        # spatial_filt_conv: channel-wise rewrite of braindecode's permuted conv.
        # Growing its fan-in grows Nu (= spatial_conv fan-out). bias=False (a BN
        # follows in the tail, as in braindecode). Input is (B, Nu, 1, T).
        self.spatial_filt_conv = Conv2dGrowingModule(
            in_channels=n_spatial_filters,
            out_channels=n_spatial_filters_smooth,
            kernel_size=(1, self.samples_100ms),
            use_bias=False,
            post_layer_function=nn.Identity(),
            previous_module=self.spatial_conv,
            allow_growing=self._can_grow,
            input_size=(1, n_times),
            target_in_channels=target,
            name="spatial_filt_conv",
            device=self.device,
        )

        # --- fixed tail: strictly braindecode's order ------------------------
        self.batch_norm = nn.BatchNorm2d(
            n_spatial_filters_smooth, momentum=batch_norm_momentum
        )
        self.square = Square()
        self.dropout = nn.Dropout(drop_prob)
        self.temporal_smoothing = nn.AvgPool2d(
            kernel_size=(1, self.kernel_size_pool),
            stride=(1, self.samples_100ms),
        )
        self.activation = LogActivation() if activation is None else activation()
        self.flatten = nn.Flatten(1)
        self.final_layer = nn.Linear(num_features, n_outputs)
        self.to(self.device)

        # --- register the growable layer with gromo --------------------------
        self._growable_layers = [self.spatial_filt_conv] if self._can_grow else []
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")

    @property
    def growable_width(self) -> int:
        """Width of the growable axis (= n_spatial_filters Nu)."""
        return self.spatial_conv.out_channels

    # ------------------------------------------------------------------ sizing
    def _calc_kernel_sizes(self) -> tuple[int, int]:
        """Temporal conv (~0.1 s) and pooling (~0.5 s) sizes -- braindecode logic."""
        conv_kernel_samples = int(math.floor(self.sfreq * 0.1))
        pool_kernel_samples = int(math.floor(self.sfreq * 0.5))
        total_kernel_samples = conv_kernel_samples + pool_kernel_samples
        if self.n_times < total_kernel_samples:
            warn(
                f"Input window ({self.n_times / self.sfreq:.2f}s) is smaller than the "
                f"model's combined kernels ({total_kernel_samples / self.sfreq:.2f}s). "
                "Scaling temporal parameters down proportionally.",
                UserWarning,
                stacklevel=2,
            )
            scale = self.n_times / total_kernel_samples
            conv_kernel_samples = int(math.floor(conv_kernel_samples * scale))
            pool_kernel_samples = int(math.floor(pool_kernel_samples * scale))
        return max(1, conv_kernel_samples), max(1, pool_kernel_samples)

    def _calc_num_features(self) -> int:
        w_out_conv2 = self.n_times - self.samples_100ms + 1
        w_out_pool = (w_out_conv2 - self.kernel_size_pool) // self.samples_100ms + 1
        return self.n_spatial_filters_smooth * w_out_pool

    def _tail(self) -> list[nn.Module]:
        return [
            self.batch_norm,
            self.square,
            self.dropout,
            self.temporal_smoothing,
            self.activation,
            self.flatten,
            self.final_layer,
        ]

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_dim(x)
        x = self.spatial_conv(x)  # applies the BatchNorm post_layer_function
        x = self.spatial_filt_conv(x)
        for module in self._tail():
            x = module(x)
        return x

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward pass including candidate neurons (for the gromo line search)."""
        x = self.ensure_dim(x)
        x_ext: torch.Tensor | None = None
        x, x_ext = self.spatial_conv.extended_forward(x, x_ext)
        x, x_ext = self.spatial_filt_conv.extended_forward(x, x_ext)
        for module in self._tail():
            x = module(x)
            if x_ext is not None:
                x_ext = module(x_ext)
        return x

    # ------------------------------------------------- transfer from braindecode
    @torch.no_grad()
    def load_braindecode_weights(self, bd_model: nn.Module) -> "GrowingSCCNet":
        """Copy weights from a braindecode SCCNet (built with the same sfreq/n_times).

        The only non-trivial step is ``spatial_filt_conv``: braindecode stores it
        with shape ``(Nu_smooth, 1, Nu, samples)`` (Nu on the kernel-height axis);
        our channel-wise rewrite wants ``(Nu_smooth, Nu, 1, samples)`` -- a transpose
        of the channel and height axes, which leaves the function unchanged.
        """
        self.spatial_conv.layer.weight.data.copy_(bd_model.spatial_conv.weight.data)
        self.spatial_conv.layer.bias.data.copy_(bd_model.spatial_conv.bias.data)
        self.spatial_conv.post_layer_function.load_state_dict(
            bd_model.spatial_batch_norm.state_dict()
        )
        w = bd_model.spatial_filt_conv.weight.data  # (Nu_smooth, 1, Nu, samples)
        self.spatial_filt_conv.layer.weight.data.copy_(w.permute(0, 2, 1, 3))
        self.batch_norm.load_state_dict(bd_model.batch_norm.state_dict())
        self.final_layer.load_state_dict(bd_model.final_layer.state_dict())
        return self
