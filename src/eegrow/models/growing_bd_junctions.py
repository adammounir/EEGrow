"""Three more braindecode models with one growable junction (cadrage, étape 1b).

Each class builds the braindecode model with its PUBLISHED defaults, and replaces one
junction by a :class:`GrowingPair` whose hidden width starts at ``start_width`` and
grows to ``target_width`` (default: braindecode's own width, so at its target the
network IS the braindecode model). Junctions read in the braindecode 1.7.0 source:

``GrowingEEGConformer``
    ``junction="ffn"``: the feed-forward block of transformer layer ``block``,
    ``Linear(emb, 4 emb) -> GELU -> Dropout -> Linear(4 emb, emb)``
    (``FeedForwardBlock``, ``modules/blocks.py``). Hidden width 4 x n_filters_time
    = 160. The residual add and the LayerNorm sit OUTSIDE the junction.
    ``junction="patch"``: ``patch_embedding.shallownet[0] -> shallownet[1]``,
    Conv2d -> Conv2d with nothing in between (the ShallowFBCSPNet junction, but not
    fused by ``CombinedConv``). Hidden width n_filters_time = 40.
``GrowingContraWR``
    ``_ResBlock`` number ``block``: ``conv1 -> bn1 -> ELU -> conv2`` (the residual is
    added after ``bn2``, outside the junction). The BatchNorm in the junction grows
    with gromo's ``GrowingBatchNorm2d``. Hidden width = the block's output width
    (32, 64, 128, 256).
``GrowingTIDNet``
    ``_DenseFilter`` number ``block`` of the spatial stage: the bottleneck
    ``Conv1x1 -> BN -> LeakyReLU -> Conv(5, 1)`` (the dense concatenation is applied
    after the junction). Hidden width bottleneck x s_growth = 72.

``GrowingCTNet``
    The FFN of transformer layer ``block``: ``trans.layers[block].feed_forward.module[0]``
    = ``Linear(40, 160) -> GELU -> Dropout -> Linear(160, 40)`` (inside CTNet's own
    residual add + LayerNorm). Hidden width 160.
``GrowingMSVTNet``
    The FFN of ``nn.TransformerEncoderLayer`` number ``block``: ``linear1 -> ReLU ->
    dropout -> linear2``, hidden 72 (``dim_feedforward == d_model`` in MSVTNet). The
    layer's fused fast path reads ``linear1.weight`` directly, so it is switched off
    (``activation_relu_or_gelu = 0``): the slow path is the same function, op by op.
``GrowingEEGNeX``
    ``junction=1``: ``block_1[1] Conv(1, 8, (1, 64)) -> BN -> block_2[0] Conv(8, 32)``,
    hidden 8; ``junction=2``: ``block_4[0] Conv(64, 32, dil 2) -> BN -> block_5[0]
    Conv(32, 8, dil 4)``, hidden 32. Growing a junction IN PLACE leaves every other
    width, and so the fixed tail, braindecode's: the reason ``grow_eegnex`` was not
    width-matched (its knob ``filter_1`` also sized the tail) does not apply.

Two caveats, inherent to the junctions and NOT fixed here:

* a BatchNorm inside the junction (ContraWR, TIDNet) is identity for the candidate
  neurons during the line search (gromo ``GrowingBatchNorm.extended_forward``), then
  starts at weight 1 / bias 0 / running stats (0, 1) once grown; in train mode it
  renormalises the new channels, so the amplitude the line search chose on the
  first layer's side is not the one training sees;
* the FFN junction is one block out of six: growing it grows ~1/6 of the
  transformer's FFN capacity. The eegrow loop grows one layer per model.
"""

from __future__ import annotations

import torch
from braindecode.models import CTNet, ContraWR, EEGConformer, EEGNeX, MSVTNet, TIDNet
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm2d
from gromo.modules.linear_growing_module import LinearGrowingModule
from torch import nn

from .growing_junction import GrowingPair, JunctionGrowingModel

Conv2dGrowingModule = RestrictedConv2dGrowingModule


def _input_hw(net: nn.Module, module: nn.Module, n_chans: int, n_times: int):
    """(H, W) of the 4-D input ``module`` receives in ``net`` (one dummy pass)."""
    seen = {}

    def hook(_m, args):     # must return None: a value would replace the inputs
        seen.setdefault("x", args[0].shape)

    h = module.register_forward_pre_hook(hook)
    was = net.training
    net.eval()
    with torch.no_grad():
        net(torch.zeros(2, n_chans, n_times))
    net.train(was)
    h.remove()
    return tuple(seen["x"][-2:])


def _growing_bn(bn: nn.BatchNorm2d, width: int) -> GrowingBatchNorm2d:
    return GrowingBatchNorm2d(width, eps=bn.eps, momentum=bn.momentum,
                              affine=bn.affine,
                              track_running_stats=bn.track_running_stats)


def _conv_pair(c1: nn.Conv2d, c2: nn.Conv2d, mid: nn.Module, *, start, target,
               hw_in, hw_mid, device, name):
    first = Conv2dGrowingModule(
        in_channels=c1.in_channels, out_channels=start, kernel_size=c1.kernel_size,
        stride=c1.stride, padding=c1.padding, dilation=c1.dilation,
        use_bias=c1.bias is not None, post_layer_function=mid, previous_module=None,
        allow_growing=False, input_size=hw_in, name=f"{name}.first", device=device)
    second = Conv2dGrowingModule(
        in_channels=start, out_channels=c2.out_channels, kernel_size=c2.kernel_size,
        stride=c2.stride, padding=c2.padding, dilation=c2.dilation,
        use_bias=c2.bias is not None, post_layer_function=nn.Identity(),
        previous_module=first, allow_growing=target > start, input_size=hw_mid,
        target_in_channels=target, name=f"{name}.second", device=device)
    return GrowingPair(first, second)


@torch.no_grad()
def _copy_conv(dst, src: nn.Conv2d) -> None:
    dst.layer.weight.copy_(src.weight)
    if src.bias is not None:
        dst.layer.bias.copy_(src.bias)


class GrowingEEGConformer(JunctionGrowingModel):
    """braindecode EEGConformer with its FFN (or patch-embedding) junction growable."""

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, *,
                 junction: str = "ffn", block: int = 0, start_width: int = 16,
                 target_width: int | None = None, device=None, **bd_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.net = EEGConformer(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                                **bd_kwargs)
        self.junction, self.block = junction, block
        if junction == "ffn":
            ffn = self.net.transformer[block][1].fn[1]
            lin1, act, drop, lin2 = ffn
            target = lin1.out_features if target_width is None else target_width
            first = LinearGrowingModule(
                lin1.in_features, start_width, use_bias=lin1.bias is not None,
                post_layer_function=nn.Sequential(act, drop), name="ffn.first",
                device=self.device)
            second = LinearGrowingModule(
                start_width, lin2.out_features, use_bias=lin2.bias is not None,
                previous_module=first, allow_growing=target > start_width,
                target_in_features=target, name="ffn.second", device=self.device)
            self._bd = (lin1, lin2)
            self.pair = GrowingPair(first, second)
            self.net.transformer[block][1].fn[1] = self.pair
        elif junction == "patch":
            sn = self.net.patch_embedding.shallownet
            c1, c2 = sn[0], sn[1]
            target = c1.out_channels if target_width is None else target_width
            hw_in = _input_hw(self.net, c1, n_chans, n_times)
            hw_mid = _input_hw(self.net, c2, n_chans, n_times)
            self._bd = (c1, c2)
            self.pair = _conv_pair(c1, c2, nn.Identity(), start=start_width,
                                   target=target, hw_in=hw_in, hw_mid=hw_mid,
                                   device=self.device, name="patch")
            sn[0], sn[1] = self.pair, nn.Identity()
        else:
            raise ValueError(f"junction must be 'ffn' or 'patch', got {junction!r}")
        self._finish(start=start_width, target=target)

    @torch.no_grad()
    def load_braindecode_weights(self, bd: EEGConformer) -> "GrowingEEGConformer":
        """Copy an EEGConformer (same arguments, start_width == its width) in."""
        own = dict(bd.state_dict())
        if self.junction == "ffn":
            pre, idx = f"transformer.{self.block}.1.fn.1.", (0, 3)
        else:
            pre, idx = "patch_embedding.shallownet.", (0, 1)
        for dst, i in zip((self.pair.first, self.pair.second), idx):
            dst.layer.weight.copy_(own.pop(f"{pre}{i}.weight"))
            dst.layer.bias.copy_(own.pop(f"{pre}{i}.bias"))
        missing, unexpected = self.net.load_state_dict(own, strict=False)
        assert not unexpected, unexpected
        assert all(".first." in k or ".second." in k for k in missing), missing
        return self


class GrowingContraWR(JunctionGrowingModel):
    """braindecode ContraWR with ``conv1 -> bn1 -> ELU -> conv2`` of one block growable."""

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, sfreq: float, *,
                 block: int = 0, start_width: int = 8, target_width: int | None = None,
                 device=None, **bd_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.net = ContraWR(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                            sfreq=sfreq, **bd_kwargs)
        self.block = block
        rb = self.net.convs[block]
        target = rb.conv1.out_channels if target_width is None else target_width
        hw_in = _input_hw(self.net, rb.conv1, n_chans, n_times)
        hw_mid = _input_hw(self.net, rb.conv2, n_chans, n_times)
        mid = nn.Sequential(_growing_bn(rb.bn1, start_width), rb.relu)
        self.pair = _conv_pair(rb.conv1, rb.conv2, mid, start=start_width,
                               target=target, hw_in=hw_in, hw_mid=hw_mid,
                               device=self.device, name=f"convs.{block}")
        rb.conv1, rb.bn1, rb.relu, rb.conv2 = (self.pair, nn.Identity(), nn.Identity(),
                                               nn.Identity())
        self._finish(start=start_width, target=target)

    @torch.no_grad()
    def load_braindecode_weights(self, bd: ContraWR) -> "GrowingContraWR":
        own = dict(bd.state_dict())
        pre = f"convs.{self.block}."
        for dst, src in ((self.pair.first, "conv1"), (self.pair.second, "conv2")):
            dst.layer.weight.copy_(own.pop(f"{pre}{src}.weight"))
            dst.layer.bias.copy_(own.pop(f"{pre}{src}.bias"))
        bn = self.pair.first.post_layer_function[0]
        bn.load_state_dict({k[len(pre) + 4:]: own.pop(k) for k in list(own)
                            if k.startswith(f"{pre}bn1.")})
        _, unexpected = self.net.load_state_dict(own, strict=False)
        assert not unexpected, unexpected
        return self


class GrowingTIDNet(JunctionGrowingModel):
    """braindecode TIDNet with the bottleneck of one dense spatial filter growable."""

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, *, block: int = 0,
                 start_width: int = 8, target_width: int | None = None, device=None,
                 **bd_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.net = TIDNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                          **bd_kwargs)
        self.block = block
        seq = self.net.dscnn.spatial.net[block].net
        c1, bn, act, c2 = seq[2], seq[3], seq[4], seq[5]
        target = c1.out_channels if target_width is None else target_width
        hw_in = _input_hw(self.net, c1, n_chans, n_times)
        hw_mid = _input_hw(self.net, c2, n_chans, n_times)
        mid = nn.Sequential(_growing_bn(bn, start_width), act)
        self.pair = _conv_pair(c1, c2, mid, start=start_width, target=target,
                               hw_in=hw_in, hw_mid=hw_mid, device=self.device,
                               name=f"spatial.{block}")
        seq[2], seq[3], seq[4], seq[5] = (self.pair, nn.Identity(), nn.Identity(),
                                          nn.Identity())
        self._finish(start=start_width, target=target)

    @torch.no_grad()
    def load_braindecode_weights(self, bd: TIDNet) -> "GrowingTIDNet":
        own = dict(bd.state_dict())
        pre = f"dscnn.spatial.net.{self.block}.net."
        for dst, i in ((self.pair.first, 2), (self.pair.second, 5)):
            dst.layer.weight.copy_(own.pop(f"{pre}{i}.weight"))
            if f"{pre}{i}.bias" in own:
                dst.layer.bias.copy_(own.pop(f"{pre}{i}.bias"))
        bn = self.pair.first.post_layer_function[0]
        bn.load_state_dict({k[len(pre) + 2:]: own.pop(k) for k in list(own)
                            if k.startswith(f"{pre}3.")})
        _, unexpected = self.net.load_state_dict(own, strict=False)
        assert not unexpected, unexpected
        return self


class GrowingCTNet(JunctionGrowingModel):
    """braindecode CTNet with the FFN of one transformer layer growable."""

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, *, block: int = 0,
                 start_width: int = 8, target_width: int | None = None, device=None,
                 **bd_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.net = CTNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                         **bd_kwargs)
        self.block = block
        holder = self.net.trans.layers[block].feed_forward.module
        lin1, act, drop, lin2 = holder[0]
        target = lin1.out_features if target_width is None else target_width
        self.pair = _linear_pair(lin1, lin2, nn.Sequential(act, drop), start=start_width,
                                 target=target, device=self.device, name="ffn")
        holder[0] = self.pair
        self._finish(start=start_width, target=target)

    @torch.no_grad()
    def load_braindecode_weights(self, bd: CTNet) -> "GrowingCTNet":
        return _load(self, bd, f"trans.layers.{self.block}.feed_forward.module.0.",
                     ((self.pair.first, "0"), (self.pair.second, "3")))


class GrowingMSVTNet(JunctionGrowingModel):
    """braindecode MSVTNet with the FFN of one transformer encoder layer growable."""

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, *, block: int = 0,
                 start_width: int = 8, target_width: int | None = None, device=None,
                 **bd_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.net = MSVTNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                           **bd_kwargs)
        self.block = block
        layer = self.net.transformer.trans.layers[block]
        act = layer.activation if isinstance(layer.activation, nn.Module) else \
            _Fn(layer.activation)
        target = layer.linear1.out_features if target_width is None else target_width
        self.pair = _linear_pair(layer.linear1, layer.linear2,
                                 nn.Sequential(act, layer.dropout), start=start_width,
                                 target=target, device=self.device, name="ffn")
        # _ff_block = linear2(dropout(activation(linear1(x)))): the pair takes linear1's
        # place, the three others become identities. No fused fast path (it would read
        # linear1.weight): the slow path computes the same thing op by op.
        layer.linear1, layer.linear2 = self.pair, nn.Identity()
        layer.dropout, layer.activation = nn.Identity(), nn.Identity()
        layer.activation_relu_or_gelu = 0
        self._finish(start=start_width, target=target)

    @torch.no_grad()
    def load_braindecode_weights(self, bd: MSVTNet) -> "GrowingMSVTNet":
        return _load(self, bd, f"transformer.trans.layers.{self.block}.",
                     ((self.pair.first, "linear1"), (self.pair.second, "linear2")))


class GrowingEEGNeX(JunctionGrowingModel):
    """braindecode EEGNeX with one of its two conv -> BN -> conv junctions growable."""

    _SITES = {1: ("block_1", 1, "block_2", 0), 2: ("block_4", 0, "block_5", 0)}

    def __init__(self, n_chans: int, n_outputs: int, n_times: int, *, junction: int = 1,
                 start_width: int = 2, target_width: int | None = None, device=None,
                 **bd_kwargs) -> None:
        super().__init__(in_features=n_chans, out_features=n_outputs, device=device)
        self.net = EEGNeX(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                          **bd_kwargs)
        self.junction = junction
        b1, i1, b2, i2 = self._SITES[junction]
        s1, s2 = getattr(self.net, b1), getattr(self.net, b2)
        c1, bn, c2 = s1[i1], s1[i1 + 1], s2[i2]
        target = c1.out_channels if target_width is None else target_width
        hw_in = _input_hw(self.net, c1, n_chans, n_times)
        hw_mid = _input_hw(self.net, c2, n_chans, n_times)
        self.pair = _conv_pair(c1, c2, nn.Sequential(_growing_bn(bn, start_width)),
                               start=start_width, target=target, hw_in=hw_in,
                               hw_mid=hw_mid, device=self.device, name=f"j{junction}")
        s1[i1], s1[i1 + 1], s2[i2] = self.pair, nn.Identity(), nn.Identity()
        self._finish(start=start_width, target=target)

    @torch.no_grad()
    def load_braindecode_weights(self, bd: EEGNeX) -> "GrowingEEGNeX":
        b1, i1, b2, i2 = self._SITES[self.junction]
        own = dict(bd.state_dict())
        for dst, key in ((self.pair.first, f"{b1}.{i1}"), (self.pair.second, f"{b2}.{i2}")):
            dst.layer.weight.copy_(own.pop(f"{key}.weight"))
            if f"{key}.bias" in own:
                dst.layer.bias.copy_(own.pop(f"{key}.bias"))
        bnp = f"{b1}.{i1 + 1}."
        self.pair.first.post_layer_function[0].load_state_dict(
            {k[len(bnp):]: own.pop(k) for k in list(own) if k.startswith(bnp)})
        _, unexpected = self.net.load_state_dict(own, strict=False)
        assert not unexpected, unexpected
        return self


class _Fn(nn.Module):
    """A functional activation (``F.relu``) as a stateless module."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _linear_pair(lin1: nn.Linear, lin2: nn.Linear, mid: nn.Module, *, start, target,
                 device, name):
    first = LinearGrowingModule(
        lin1.in_features, start, use_bias=lin1.bias is not None,
        post_layer_function=mid, name=f"{name}.first", device=device)
    second = LinearGrowingModule(
        start, lin2.out_features, use_bias=lin2.bias is not None, previous_module=first,
        allow_growing=target > start, target_in_features=target, name=f"{name}.second",
        device=device)
    return GrowingPair(first, second)


def _load(model, bd, prefix, pairs):
    own = dict(bd.state_dict())
    for dst, key in pairs:
        dst.layer.weight.copy_(own.pop(f"{prefix}{key}.weight"))
        if f"{prefix}{key}.bias" in own:
            dst.layer.bias.copy_(own.pop(f"{prefix}{key}.bias"))
    missing, unexpected = model.net.load_state_dict(own, strict=False)
    assert not unexpected, unexpected
    assert all(".first." in k or ".second." in k for k in missing), missing
    return model
