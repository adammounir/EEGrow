# eegrow

A bridge between **[gromo](https://github.com/growingnet/gromo)** (architecture
growth during training) and **[braindecode](https://braindecode.org)** (end-to-end
EEG decoding).

`eegrow` provides EEG decoding models whose convolution-junction width can **grow**
during training: instead of fixing the width up front, it is auto-sized via gromo.
Two models:

| Model | Growable junction | Notes |
|---|---|---|
| `GrowingShallowFBCSPNet` | temporal conv → spatial conv | mirrors `ShallowFBCSPNet`, can reload its braindecode weights |
| `GrowingDeepEEGNet` | a **deep** stage junction (after pooling) | 2-stage VGG-style net; grows a deep junction |

> **Why a staged network?** gromo cannot grow a junction *across a pooling layer*
> (pooling changes the spatial dimension). A growable deep network therefore chains
> several convs *without pooling* inside a stage (the junction that grows), pooling
> only at the end of the stage.

## Relationship to braindecode

A natural question: if the architectures are re-implemented here, what is braindecode
for? Re-implementation is **required by gromo, not a choice**: a layer can only grow
if it is a special `Conv2dGrowingModule` that knows how to compute its optimal
neurons. braindecode models use plain `nn.Conv2d`, which cannot grow — so a growable
model must rebuild the same architecture from gromo bricks.

braindecode is still used as:

1. **EEG primitives** — `Square`, `SafeLog`, `Ensure4d`, `SqueezeFinalOutput` are
   imported, not re-coded.
2. **Reference / baseline** — `GrowingShallowFBCSPNet.load_braindecode_weights()`
   copies the weights of a real braindecode `ShallowFBCSPNet`, so the growable
   version can be checked for equivalence against the canonical model.
3. **Data** — datasets, windowing and preprocessing (braindecode's main role) live
   on the data side, outside this repo.

A future, more elegant direction would be an automatic converter that takes a
braindecode model and swaps its `nn.Conv2d` layers for gromo growable bricks.

## Installation

`gromo` and `braindecode` are dependencies, never modified.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
```

`pip install -e .` pulls `torch`, `braindecode`, `einops` and `gromo` (from git).

## Quick start

Synthetic demo (random noise, no real data) proving the mechanism runs end to end —
forward pass, one growth step, and the parameter count increases:

```bash
python examples/demo_growth.py
```

```python
from eegrow import GrowingDeepEEGNet
from eegrow.training import loop

# growable deep junction: width 8 -> 32
model = GrowingDeepEEGNet(n_chans=22, n_outputs=4, n_times=500,
                          w1=8, w2=8, target_w2=32, device="cpu")

# ... standard training, with a periodic growth step:
loop.grow_step(model, train_loader, device="cpu")
```

## Technical notes

- **Growth requires CPU.** gromo's optimal-update computation relies on
  `torch.linalg.eigh`, not implemented on MPS (Apple Silicon). Regular
  forward/backward passes can stay on GPU/MPS; only `grow_step` must run on CPU.
- **Optimizer preserved after growth.** After a growth, only the weights that grew
  restart from a fresh optimizer state; the momentum of unchanged parameters is
  kept (`loop.rebuild_optimizer_preserving_state`).
- **gromo modules used.** `RestrictedConv2dGrowingModule` (the base
  `Conv2dGrowingModule` does not implement fan-in growth).

## License

MIT — see [LICENSE](LICENSE).
