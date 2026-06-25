"""EEGNeX demo -- proves the growable analog runs end to end + one growth step.

NO real data, NO accuracy numbers: random noise, build a `GrowingEEGNeX` whose
first temporal filter bank (`filter_1`, the "8" in EEGNeX-8,32) can grow, run ONE
gromo growth step, and check the parameter count increased. Output shape parity
with braindecode's `EEGNeX` is checked too.

gromo growth relies on `eigh` (absent on MPS), so we stay on CPU.

Usage: python examples/demo_eegnex.py
"""

from __future__ import annotations

import torch
from braindecode.models import EEGNeX
from torch.utils.data import DataLoader, TensorDataset

from eegrow import GrowingEEGNeX
from eegrow.training import loop

DEVICE = "cpu"  # gromo growth requires CPU (eigh absent on MPS)
C, T, N_CLASSES = 22, 500, 4


def main() -> None:
    torch.manual_seed(0)

    print("=" * 64)
    print("EEGNeX demo -- growable first temporal filter bank (filter_1)")
    print("=" * 64)

    # output-shape parity with braindecode's EEGNeX
    bd_shape = tuple(EEGNeX(n_chans=C, n_outputs=N_CLASSES, n_times=T)
                     .eval()(torch.randn(4, C, T)).shape)

    xs, ys = torch.randn(64, C, T), torch.randint(0, N_CLASSES, (64,))
    loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True)
    model = GrowingEEGNeX(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        filter_1=4, filter_2=32, target_filter_1=16, device=DEVICE,
    )
    p0 = loop.count_params(model)
    print(f"filter_1 (before) : {model.growable_width}   params {p0:,}")
    out = model(xs[:4].to(DEVICE))
    print(f"forward before growth : output {tuple(out.shape)}  "
          f"(braindecode EEGNeX {bd_shape})  [OK]")

    loop.grow_step(model, loader, DEVICE)

    p1 = loop.count_params(model)
    print(f"filter_1 (after)  : {model.growable_width}   params {p1:,}  (+{p1 - p0:,})")
    out = model(xs[:4].to(DEVICE))
    print(f"forward after growth : output {tuple(out.shape)}  [OK]")

    assert out.shape[1:] == torch.Size(bd_shape[1:]), "output shape mismatch"
    assert p1 > p0, "growth did not increase the parameter count"
    print("\nGrowingEEGNeX runs and grows its first temporal filter bank. [OK]")


if __name__ == "__main__":
    main()
