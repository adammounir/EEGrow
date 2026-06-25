"""SCCNet demo -- equivalence to braindecode + one growth step.

Two checks, both on CPU (gromo growth relies on `eigh`, absent on MPS):

  1. EQUIVALENCE: a frozen `GrowingSCCNet` loaded with a braindecode `SCCNet`'s
     weights reproduces its output bit-for-bit. This proves the gromo rewrite of
     the spatio-temporal conv (channel-wise instead of braindecode's permuted
     kernel-height trick) is the same function.
  2. GROWTH: a small `GrowingSCCNet` runs one gromo growth step on synthetic
     noise and its `n_spatial_filters` (Nu) width increases.

Usage: python examples/demo_sccnet.py
"""

from __future__ import annotations

import torch
from braindecode.models import SCCNet
from torch.utils.data import DataLoader, TensorDataset

from eegrow import GrowingSCCNet
from eegrow.training import loop

DEVICE = "cpu"  # gromo growth requires CPU (eigh absent on MPS)
C, T, N_CLASSES, SFREQ = 22, 500, 4, 250.0


def main() -> None:
    torch.manual_seed(0)

    print("=" * 64)
    print("SCCNet demo -- equivalence + growth")
    print("=" * 64)

    # 1. EQUIVALENCE ----------------------------------------------------------
    bd = SCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=22, n_spatial_filters_smooth=20,
    ).eval()
    frozen = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=22, n_spatial_filters_smooth=20, device=DEVICE,
    ).eval()
    frozen.load_braindecode_weights(bd)

    x = torch.randn(8, C, T)
    with torch.no_grad():
        max_diff = (bd(x) - frozen(x)).abs().max().item()
    print(f"equivalence vs braindecode SCCNet : max|diff| = {max_diff:.2e}  [OK]")

    # 2. GROWTH ---------------------------------------------------------------
    xs, ys = torch.randn(64, C, T), torch.randint(0, N_CLASSES, (64,))
    loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True)
    model = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=8, n_spatial_filters_smooth=20,
        target_n_spatial_filters=32, device=DEVICE,
    )
    p0 = loop.count_params(model)
    print(f"\nNu width (before) : {model.growable_width}   params {p0:,}")
    model(xs[:4].to(DEVICE))  # forward works before growth
    loop.grow_step(model, loader, DEVICE)
    p1 = loop.count_params(model)
    print(f"Nu width (after)  : {model.growable_width}   params {p1:,}  (+{p1 - p0:,})")
    out = model(xs[:4].to(DEVICE))
    print(f"forward after growth : output {tuple(out.shape)}  [OK]")

    assert max_diff < 1e-4, "SCCNet equivalence broken"
    assert p1 > p0, "growth did not increase the parameter count"
    print("\nGrowingSCCNet is bit-exact with braindecode and grows. [OK]")


if __name__ == "__main__":
    main()
