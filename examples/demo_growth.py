"""Synthetic demo -- proves the growth mechanism runs end to end.

NO real data, NO accuracy numbers: we generate random noise, build a
`GrowingDeepEEGNet` with a growable deep junction, run ONE gromo growth step, and
check that:

  1. the forward pass works before growth;
  2. the growth step runs (stats -> optimal update -> line search -> apply);
  3. the parameter count increased (the junction grew);
  4. the forward pass still works after growth.

This is a smoke test of the gromo <-> braindecode bridge, not an experiment.
gromo growth relies on `eigh` (absent on MPS), so we stay on CPU.

Usage: python examples/demo_growth.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from eegrow import GrowingDeepEEGNet
from eegrow.training import loop

DEVICE = "cpu"  # gromo growth requires CPU (eigh absent on MPS)

# --- synthetic EEG signal (noise) ---------------------------------------------
N, C, T, N_CLASSES = 128, 22, 500, 4
torch.manual_seed(0)
x = torch.randn(N, C, T)
y = torch.randint(0, N_CLASSES, (N,))
train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def main() -> None:
    # growable deep junction: w2 = 8 -> 32
    model = GrowingDeepEEGNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        w1=8, w2=8, target_w2=32, device=DEVICE,
    )

    print("=" * 60)
    print("Synthetic demo -- growing a deep EEG junction")
    print("=" * 60)
    p0 = loop.count_params(model)
    print(f"junction width (before) : {model.growable_width}")
    print(f"parameters     (before) : {p0:,}")

    # 1. forward pass before growth
    out = model(x[:4].to(DEVICE))
    print(f"forward before growth : output {tuple(out.shape)}  [OK]")

    # 2. one gromo growth step
    loop.grow_step(model, train_loader, DEVICE)

    # 3. the junction grew
    p1 = loop.count_params(model)
    print(f"junction width (after)  : {model.growable_width}")
    print(f"parameters     (after)  : {p1:,}  (+{p1 - p0:,})")

    # 4. forward pass after growth
    out = model(x[:4].to(DEVICE))
    print(f"forward after growth : output {tuple(out.shape)}  [OK]")

    assert p1 > p0, "growth did not increase the parameter count"
    print("\nThe gromo <-> braindecode bridge works: the junction grew.")


if __name__ == "__main__":
    main()
