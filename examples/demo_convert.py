"""Synthetic demo of the automatic converter.

We build an ORDINARY ``nn.Sequential`` EEG-style network -- plain ``nn.Conv2d``,
zero knowledge of gromo -- then call ``make_growable`` to turn it into a growable
model, run one growth step, and check the detected junction grew. No real data.

The converter auto-detects the growable junctions: consecutive convs separated only
by an activation (here, conv2a -> ELU -> conv2b). Convs separated by pooling are
correctly left non-growable.

Usage: python examples/demo_convert.py
"""

from __future__ import annotations

import torch
from braindecode.modules import Ensure4d, SqueezeFinalOutput
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eegrow.convert import make_growable
from eegrow.training import loop

DEVICE = "cpu"  # gromo growth requires CPU (eigh absent on MPS)
C, T, N_CLASSES = 22, 500, 4
w1, w2 = 8, 8


def plain_eeg_net() -> nn.Sequential:
    """A vanilla EEG net as nn.Sequential -- no gromo anywhere."""
    # spatial dims after each step, to size the classifier kernel
    t1 = T - 25 + 1
    t1p = (t1 - 3) // 3 + 1
    t2 = t1p - 10 + 1
    t2b = t2 - 10 + 1
    t2p = (t2b - 3) // 3 + 1
    return nn.Sequential(
        Ensure4d(),
        Rearrange("b c t 1 -> b 1 t c"),
        nn.Conv2d(1, w1, (25, 1)),                 # conv_time
        nn.Conv2d(w1, w1, (1, C), bias=False),     # conv_spat (junction: collapse)
        nn.BatchNorm2d(w1), nn.ELU(),
        nn.MaxPool2d((3, 1), (3, 1)),              # breaks the chain
        nn.Conv2d(w1, w2, (10, 1), bias=False),    # conv2a
        nn.ELU(),                                   # folded into conv2a
        nn.Conv2d(w2, w2, (10, 1), bias=False),    # conv2b (deep junction)
        nn.BatchNorm2d(w2), nn.ELU(),
        nn.MaxPool2d((3, 1), (3, 1)),
        nn.Conv2d(w2, N_CLASSES, (t2p, 1)),        # classifier
        SqueezeFinalOutput(),
    )


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, C, T)
    y = torch.randint(0, N_CLASSES, (64,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    plain = plain_eeg_net()

    print("=" * 64)
    print("Automatic converter demo -- nn.Sequential -> growable gromo model")
    print("=" * 64)

    # conversion must preserve the function (weights are copied)
    with torch.no_grad():
        ref = plain.eval()(x[:8])
    model = make_growable(plain, x[:8], growth_factor=4.0, device=DEVICE)
    with torch.no_grad():
        got = model.eval()(x[:8])
    max_diff = (ref - got).abs().max().item()
    print(f"junctions detected as growable : {len(model._growable_layers)}")
    print(f"output diff after conversion   : {max_diff:.2e}  (should be ~0)")

    p0 = loop.count_params(model)
    print(f"\ngrowable width (before) : {model.growable_width}")
    print(f"parameters     (before) : {p0:,}")

    # grow the first growable junction
    loop.grow_step(model, train_loader, DEVICE)

    p1 = loop.count_params(model)
    print(f"growable width (after)  : {model.growable_width}")
    print(f"parameters     (after)  : {p1:,}  (+{p1 - p0:,})")
    out = model(x[:4].to(DEVICE))
    print(f"forward after growth : output {tuple(out.shape)}  [OK]")

    assert max_diff < 1e-4, "conversion changed the function"
    assert p1 > p0, "growth did not increase the parameter count"
    print("\nConverter works: a plain nn.Sequential became growable and grew.")


if __name__ == "__main__":
    main()
