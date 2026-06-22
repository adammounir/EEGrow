"""Automatic converter: a plain ``nn.Sequential`` -> a growable gromo model.

Motivation
----------
A layer can only grow if it is a gromo ``Conv2dGrowingModule`` that knows how to
compute its optimal neurons. Plain ``nn.Conv2d`` cannot grow. Instead of rewriting
each architecture by hand, this module detects the growable junctions of an existing
``nn.Sequential`` and swaps the relevant convs for gromo bricks, preserving the
function at conversion time (weights are copied).

What is a growable junction?
----------------------------
gromo grows the *shared* width between two consecutive convs (the fan-out of conv A =
the fan-in of conv B). This is only valid if nothing between them changes the spatial
layout: pooling, stride, a channel-mixing op all break it. An activation between the
two convs is fine -- it becomes conv A's ``post_layer_function`` (which is how gromo
accounts for it in the growth math).

So we group the convs into maximal *chains* where consecutive convs are separated only
by a single activation (or nothing). Inside a chain (length >= 2): the first conv is
frozen (``allow_growing=False``), every following conv is growable and chained via
``previous_module``. Anything else (pooling, BN-between-convs, linear, flatten,
reshape) breaks the chain -- exactly the rule gromo's own VGG container uses.

Scope (v1)
----------
- Input must be an ``nn.Sequential`` (or a list of modules). Arbitrary ``forward``
  graphs (residuals, branches) are out of scope -- that needs FX tracing.
- Only stride-1, no-padding, single-group convs are converted (the EEG case);
  any other conv is left untouched and simply breaks the chain.
"""

from __future__ import annotations

import torch
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from torch import nn

# Activations that may sit between two convs and be folded as a post_layer_function.
FOLDABLE_ACTIVATIONS = (
    nn.ELU, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.Identity,
)


def _is_growth_compatible(conv: nn.Conv2d) -> bool:
    """A conv gromo can grow: stride 1, no padding, no dilation, single group."""
    return (
        tuple(conv.stride) == (1, 1)
        and tuple(conv.padding) == (0, 0)
        and tuple(conv.dilation) == (1, 1)
        and conv.groups == 1
    )


class GrowableSequential(SequentialGrowingModel):
    """A converted sequential model whose detected junctions can grow.

    Built by :func:`make_growable`; not meant to be instantiated directly.
    """

    def __init__(self, layers, growable, *, in_features, out_features, device):
        super().__init__(in_features=in_features, out_features=out_features,
                         device=device)
        self.layers = nn.ModuleList(layers)
        self._growable_layers = list(growable)
        if self._growable_layers:
            self.set_growing_layers(scheduling_method="all")
        self.to(self.device)

    @property
    def growable_width(self) -> int:
        """Current width of the (first) growable junction."""
        if not self._growable_layers:
            return 0
        return self._growable_layers[0].previous_module.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.layers:
            x = m(x)
        return x

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Forward including candidate neurons (gromo line search).

        ``x_ext`` (the correction carried by candidate neurons) is None until it is
        born at a growable conv, then it propagates through the rest of the layers --
        applied to every downstream module just like the main path.
        """
        x_ext: torch.Tensor | None = None
        for m in self.layers:
            if isinstance(m, RestrictedConv2dGrowingModule):
                x, x_ext = m.extended_forward(x, x_ext)
            else:
                x = m(x)
                if x_ext is not None:
                    x_ext = m(x_ext)
        return x


def _trace_conv_input_sizes(modules, example_input) -> dict[int, tuple[int, int]]:
    """Record the spatial (H, W) shape of the input feeding each Conv2d."""
    sizes: dict[int, tuple[int, int]] = {}
    was_training = any(m.training for m in modules)
    for m in modules:
        m.eval()  # eval so BatchNorm uses running stats (stable for a dry run)
    x = example_input
    with torch.no_grad():
        for idx, m in enumerate(modules):
            if isinstance(m, nn.Conv2d):
                sizes[idx] = tuple(x.shape[-2:])
            x = m(x)
    if was_training:
        for m in modules:
            m.train()
    return sizes


def _find_chains(modules):
    """Group conv indices into maximal chains joined by <=1 foldable activation.

    Returns a list of chains; each chain is a list of ``(conv_index, act_before)``
    where ``act_before`` is the activation module sitting between this conv and the
    previous conv of the chain (or None). Also returns the set of activation indices
    that get folded (and must be removed from the standalone module list).
    """
    chains: list[list[tuple[int, nn.Module | None]]] = []
    folded: set[int] = set()
    current: list[tuple[int, nn.Module | None]] = []
    prev_conv: int | None = None

    for idx, m in enumerate(modules):
        if not isinstance(m, nn.Conv2d):
            continue
        if not _is_growth_compatible(m):
            if current:
                chains.append(current)
                current = []
            prev_conv = None
            continue
        if prev_conv is None:
            current = [(idx, None)]
        else:
            between = modules[prev_conv + 1:idx]
            if len(between) <= 1 and all(
                isinstance(b, FOLDABLE_ACTIVATIONS) for b in between
            ):
                current.append((idx, between[0] if between else None))
                folded.update(range(prev_conv + 1, idx))
            else:
                chains.append(current)
                current = [(idx, None)]
        prev_conv = idx

    if current:
        chains.append(current)
    return chains, folded


def make_growable(
    model: nn.Sequential,
    example_input: torch.Tensor,
    *,
    growth_factor: float = 2.0,
    device: torch.device | str | None = None,
) -> GrowableSequential:
    """Convert an ``nn.Sequential`` into a growable gromo model.

    Parameters
    ----------
    model : nn.Sequential
        The model to convert. Its convs that form a junction (consecutive convs with
        only an activation between them) become growable; everything else is kept.
    example_input : torch.Tensor
        A representative input (with batch dim) used to trace activation shapes.
    growth_factor : float
        Target width of each growable junction = ``round(growth_factor * width)``.
    device : torch.device | str | None
        Device of the converted model (growth itself must run on CPU: gromo's
        optimal-update relies on ``eigh``, absent on MPS).

    Returns
    -------
    GrowableSequential
        Function-equivalent to ``model`` at conversion time (weights are copied).
    """
    if device is None:
        device = "cpu"
    modules = list(model)

    input_sizes = _trace_conv_input_sizes(modules, example_input)
    chains, folded = _find_chains(modules)

    gromo_for: dict[int, RestrictedConv2dGrowingModule] = {}
    growable: list[RestrictedConv2dGrowingModule] = []

    for chain in chains:
        if len(chain) < 2:
            continue  # a lone conv has no growable partner
        prev_g: RestrictedConv2dGrowingModule | None = None
        for pos, (idx, _act_before) in enumerate(chain):
            conv = modules[idx]
            is_first = pos == 0
            # activation between this conv and the NEXT one = this conv's post fn
            next_act = chain[pos + 1][1] if pos + 1 < len(chain) else None
            post = next_act if next_act is not None else nn.Identity()
            g = RestrictedConv2dGrowingModule(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                use_bias=conv.bias is not None,
                post_layer_function=post,
                previous_module=prev_g,
                allow_growing=not is_first,
                input_size=input_sizes[idx],
                target_in_channels=(
                    None if is_first else round(growth_factor * conv.in_channels)
                ),
                name=f"conv{idx}",
                device=device,
            )
            g.layer.weight.data.copy_(conv.weight.data)
            if conv.bias is not None:
                g.layer.bias.data.copy_(conv.bias.data)
            gromo_for[idx] = g
            if not is_first:
                growable.append(g)
            prev_g = g

    # rebuild the ordered module list: drop folded activations, swap converted convs
    new_layers: list[nn.Module] = []
    for idx, m in enumerate(modules):
        if idx in folded:
            continue
        new_layers.append(gromo_for.get(idx, m))

    with torch.no_grad():
        out = model.eval()(example_input)
    in_features = int(example_input.shape[1])
    out_features = int(out.shape[-1])

    return GrowableSequential(
        new_layers, growable,
        in_features=in_features, out_features=out_features, device=device,
    )
