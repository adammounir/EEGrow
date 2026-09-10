"""GrowingShallowFBCSPNet -- a *growable* (gromo) version of ShallowFBCSPNet.

ShallowFBCSPNet (braindecode) chains, with no nonlinearity between them, a *temporal*
then a *spatial* convolution (braindecode fuses them in `CombinedConv` but stores the
two weight sets separately). We rewrite these two convs as gromo
`Conv2dGrowingModule` bricks:

    conv_time : (1 -> n_filters_time)              kernel (filter_time_length, 1)
    conv_spat : (n_filters_time -> n_filters_spat) kernel (1, n_chans)

The `conv_time -> conv_spat` junction has no nonlinearity between them
(post_layer_function = Identity), which is the textbook case for gromo. We can
therefore grow `n_filters_time` (the fan-out of conv_time = the fan-in of conv_spat).
Everything else (BatchNorm, Square, AvgPool, SafeLog, Dropout, classifier) is kept
identical to braindecode, as fixed modules.

braindecode's ShallowFBCSPNet stays the reference baseline: we simply copy its
weights (`load_braindecode_weights`), which also lets us check equivalence.

READOUT ABLATION (`readout`) -- added 10/09/2026, DEFAULT UNCHANGED
------------------------------------------------------------------
`Square -> AvgPool -> SafeLog` is not three arbitrary nonlinearities: mean-of-squares
over a window IS the variance of the spatially filtered signal, so the tail computes a
log band-power. Motor imagery is an ERD/ERS phenomenon -- the discriminative quantity
is the *power* of mu/beta, a quadratic functional of the signal -- so ShallowFBCSPNet
is FBCSP written as layers, and the statistic it must estimate is built into the
architecture. A ReLU-family tail (ELU + MaxPool) has to approximate x^2 piecewise, which
costs units and trials.

That is the standing explanation for the +0.108 gap to `grow_deep` (measured on
bnci2014_001, n=9): not depth, not width -- the missing readout. The w1 pilot (580878)
eliminated the two alternatives on that arm, capacity (w1 8->32 buys only +0.0235) and
regularisation (dropout+norm *hurt*, 10/12 paired contrasts negative).

This knob tests it on the arm we trust, one factor at a time and with the pool geometry
held fixed, so parameter count and final_conv_length are constant across settings:

    power   : Square  -> AvgPool -> SafeLog     braindecode, the default
    nolog   : Square  -> AvgPool -> Identity    keeps the variance, drops the log
    generic : ELU     -> MaxPool -> Identity    the ReLU-family readout

`nolog` sits in between on purpose: log matters for its own reason (band power is
heavy-tailed, and log turns the multiplicative ERD into an additive shift), so a single
grouped ablation could not say whether the quadratic or the compression carries the
effect -- the mistake the grouped `reg` factor made in the w1 pilot. The prediction
being tested is that `generic` collapses towards grow_deep's accuracy. If it does not,
the explanation is wrong and the gap is elsewhere.
"""

from __future__ import annotations

import torch
from braindecode.modules import Ensure4d, SafeLog, SqueezeFinalOutput, Square
from einops.layers.torch import Rearrange
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from torch import nn

# RestrictedConv2dGrowingModule: the base Conv2dGrowingModule does not implement
# fan-in growth (compute_m_prev_update); only the Restricted/Full variants do.
Conv2dGrowingModule = RestrictedConv2dGrowingModule


class GrowingShallowFBCSPNet(SequentialGrowingModel):
    """ShallowFBCSPNet whose ``n_filters_time`` width can grow via gromo.

    Parameters
    ----------
    n_chans, n_outputs, n_times : int
        Signal parameters (same conventions as braindecode).
    n_filters_time : int
        INITIAL width of the temporal conv (= fan-in of conv_spat). This is the
        axis gromo grows.
    n_filters_spat : int
        Number of spatial filters (fixed here).
    target_n_filters_time : int | None
        TARGET width to grow towards. If None or == n_filters_time, the model is
        frozen (useful for the equivalence test / baseline).
    filter_time_length, pool_time_length, pool_time_stride, drop_prob,
    batch_norm_alpha : see ShallowFBCSPNet.
    final_conv_length : int | "auto"
        Classifier kernel length. "auto" => computed by a dry run.
    readout : {"power", "nolog", "generic"}
        Which readout to put between conv_spat and the classifier. "power" is
        braindecode's and the default; the other two are the ablation. See the module
        docstring for what each one computes and why.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        target_n_filters_time: int | None = None,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: int | str = "auto",
        drop_prob: float = 0.5,
        batch_norm_alpha: float = 0.1,
        readout: str = "power",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat

        target = (
            n_filters_time if target_n_filters_time is None else target_n_filters_time
        )
        self._can_grow = target > n_filters_time
        # Cap for the growth callback. gromo does NOT enforce target_in_channels: it
        # adds a data-dependent number of neurons per step and never stops on its own.
        # GromoGrowth reads this attribute (`getattr(model, "target_width", None)`) to
        # both stop growing and trim the step via sub_select_optimal_added_parameters.
        # Without it the width is unbounded -- measured 8 -> 17 in nine growth events
        # here, and 8 -> 77 against a target of 32 on GrowingDeepEEGNet.
        self.target_width = target

        # --- head: reshape, identical to braindecode --------------------------
        self.ensuredims = Ensure4d()                       # (B,C,T) -> (B,C,T,1)
        self.dimshuffle = Rearrange("b c t 1 -> b 1 t c")  # -> (B,1,T,C)

        # --- the two growable convs ------------------------------------------
        # conv_time: network input (1 channel) -> n_filters_time. Its input does not
        # grow (it is the network input), but its OUTPUT grows when conv_spat grows
        # its input. bias_time=True as in braindecode.
        self.conv_time = Conv2dGrowingModule(
            in_channels=1,
            out_channels=n_filters_time,
            kernel_size=(filter_time_length, 1),
            use_bias=True,
            post_layer_function=nn.Identity(),
            previous_module=None,
            allow_growing=False,
            input_size=(n_times, n_chans),
            name="conv_time",
            device=self.device,
        )
        # conv_spat: (1, n_chans) kernel that collapses the electrode axis. This is
        # the growable module: growing its fan-in grows n_filters_time (and thus the
        # fan-out of conv_time). bias_spat=False since batch_norm=True in braindecode.
        t_after_time = n_times - filter_time_length + 1
        self.conv_spat = Conv2dGrowingModule(
            in_channels=n_filters_time,
            out_channels=n_filters_spat,
            kernel_size=(1, n_chans),
            use_bias=False,
            post_layer_function=nn.Identity(),
            previous_module=self.conv_time,
            allow_growing=self._can_grow,
            input_size=(t_after_time, n_chans),
            target_in_channels=target,
            name="conv_spat",
            device=self.device,
        )

        # --- fixed tail: strictly braindecode's order ------------------------
        # `readout` swaps ONLY the three modules of the band-power readout; the pool
        # geometry (kernel, stride) is held fixed, so final_conv_length and the whole
        # parameter count are identical across the three settings. The only thing that
        # changes is the *function* computed between conv_spat and the classifier.
        self.readout = readout
        if readout == "power":                      # braindecode, the default
            conv_nonlin: nn.Module = Square()
            pool: nn.Module = nn.AvgPool2d(kernel_size=(pool_time_length, 1),
                                           stride=(pool_time_stride, 1))
            pool_nonlin: nn.Module = SafeLog()
        elif readout == "nolog":                    # keep the variance, drop the log
            conv_nonlin = Square()
            pool = nn.AvgPool2d(kernel_size=(pool_time_length, 1),
                                stride=(pool_time_stride, 1))
            pool_nonlin = nn.Identity()
        elif readout == "generic":                  # the ReLU-family readout
            conv_nonlin = nn.ELU()
            pool = nn.MaxPool2d(kernel_size=(pool_time_length, 1),
                                stride=(pool_time_stride, 1))
            pool_nonlin = nn.Identity()
        else:
            raise ValueError(
                "readout must be 'power', 'nolog' or 'generic', got %r" % (readout,)
            )
        self.bnorm = nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha, affine=True)
        self.conv_nonlin = conv_nonlin
        self.pool = pool
        self.pool_nonlin = pool_nonlin
        self.drop = nn.Dropout(p=drop_prob)
        self.to(self.device)

        # --- classifier: "auto" length via a dry run -------------------------
        if final_conv_length == "auto":
            final_conv_length = self._infer_final_conv_length(n_chans, n_times)
        self.final_conv_length = int(final_conv_length)
        self.conv_classifier = nn.Conv2d(
            n_filters_spat, n_outputs, (self.final_conv_length, 1), bias=True
        )
        self.squeeze = SqueezeFinalOutput()
        self.to(self.device)

        # --- register the growable layer with gromo --------------------------
        self._growable_layers = [self.conv_spat] if self._can_grow else []
        if self._can_grow:
            self.set_growing_layers(scheduling_method="all")

    @property
    def growable_width(self) -> int:
        """Width of the growable axis (= n_filters_time)."""
        return self.conv_time.out_channels

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _infer_final_conv_length(self, n_chans: int, n_times: int) -> int:
        """Run head + tail (minus classifier) to read the temporal dim."""
        self.eval()
        x = torch.zeros(1, n_chans, n_times, device=self.device)
        x = self.dimshuffle(self.ensuredims(x))
        x = self.conv_spat(self.conv_time(x))
        x = self.drop(self.pool_nonlin(self.pool(self.conv_nonlin(self.bnorm(x)))))
        return x.shape[2]

    def _tail(self) -> list[nn.Module]:
        return [self.bnorm, self.conv_nonlin, self.pool, self.pool_nonlin,
                self.drop, self.conv_classifier, self.squeeze]

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dimshuffle(self.ensuredims(x))
        x = self.conv_time(x)
        x = self.conv_spat(x)
        for module in self._tail():
            x = module(x)
        return x

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward pass including candidate neurons (for the gromo line search)."""
        x = self.dimshuffle(self.ensuredims(x))
        x_ext: torch.Tensor | None = None
        x, x_ext = self.conv_time.extended_forward(x, x_ext)
        x, x_ext = self.conv_spat.extended_forward(x, x_ext)
        for module in self._tail():
            x = module(x)
            if x_ext is not None:
                x_ext = module(x_ext)
        return x

    # ------------------------------------------------- transfer from braindecode
    @torch.no_grad()
    def load_braindecode_weights(self, bd_model: nn.Module) -> "GrowingShallowFBCSPNet":
        """Copy weights from a braindecode ShallowFBCSPNet (split_first_layer=True).

        braindecode's two fused convs (`conv_time_spat.conv_time` and `.conv_spat`)
        are stored separately => direct copy into our two gromo bricks.
        """
        cc = bd_model.conv_time_spat  # CombinedConv
        self.conv_time.layer.weight.data.copy_(cc.conv_time.weight.data)
        self.conv_time.layer.bias.data.copy_(cc.conv_time.bias.data)
        self.conv_spat.layer.weight.data.copy_(cc.conv_spat.weight.data)
        self.bnorm.load_state_dict(bd_model.bnorm.state_dict())
        self.conv_classifier.load_state_dict(
            bd_model.final_layer.conv_classifier.state_dict()
        )
        return self
