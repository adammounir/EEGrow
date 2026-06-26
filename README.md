# eegrow

A bridge between **[gromo](https://github.com/growingnet/gromo)** (architecture
growth during training) and **[braindecode](https://braindecode.org)** (end-to-end
EEG decoding).

`eegrow` provides EEG decoding models whose convolution-junction width can **grow**
during training: instead of fixing the width up front, it is auto-sized via gromo.
Four models:

| Model | Growable junction | Notes |
|---|---|---|
| `GrowingShallowFBCSPNet` | temporal conv → spatial conv | mirrors `ShallowFBCSPNet`, can reload its braindecode weights (bit-exact) |
| `GrowingSCCNet` | spatial conv → spatio-temporal conv (grows `n_spatial_filters`) | mirrors `SCCNet`, can reload its braindecode weights (bit-exact) |
| `GrowingEEGNeX` | first temporal conv → second temporal conv (grows `filter_1`) | faithful analog of `EEGNeX`; symmetric padding on the two growable convs (see below) |
| `GrowingDeepEEGNet` | a **deep** stage junction (after pooling) | 2-stage VGG-style net; grows a deep junction |

> **Where does the BatchNorm go?** `SCCNet` and `EEGNeX` both put a BatchNorm
> *between* the two convs of the growable junction. gromo handles this the way its
> own VGG container does: the BatchNorm is the first conv's `post_layer_function`
> (a `GrowingBatchNorm2d`), so it grows together with the conv.

> **Why is `GrowingEEGNeX` only an analog?** EEGNeX's two temporal convs use
> `padding="same"` with an even kernel, which torch realises with *asymmetric*
> padding. gromo's growth statistics need an *integer* (symmetric) padding, so the
> two growable convs use symmetric padding instead (a one-sample edge effect). The
> blocks, dilations and classifier are otherwise identical; the downstream fixed
> convs keep braindecode's `padding="same"`.

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

## Automatic converter

Rather than re-implementing each model by hand, `make_growable` takes an existing
`nn.Sequential` and turns it into a growable gromo model: it detects the growable
junctions (consecutive convs separated only by an activation, no pooling), swaps
them for gromo bricks, and copies the weights (so conversion preserves the function).

```python
from eegrow import make_growable
from eegrow.training import loop

growable = make_growable(my_sequential_net, example_input, growth_factor=2.0)
loop.grow_step(growable, train_loader, device="cpu")
```

See `examples/demo_convert.py`. The converter is pure `torch` + `gromo` (no
braindecode dependency). Scope: `nn.Sequential` models with stride-1 convs; arbitrary
`forward` graphs (residuals, branches) would need FX tracing and are out of scope.

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

## Training with braindecode's `EEGClassifier`

The growable models train through the **standard braindecode/skorch API**, not only
the bespoke loop in `eegrow.training.loop`. A skorch callback (`GromoGrowth`) runs a
gromo growth step every `grow_every` epochs during `fit`, then rebuilds the optimizer
(preserving the momentum of the unchanged parameters):

```python
from eegrow import GrowingSCCNet, make_eeg_classifier

# starts at width 4, auto-grows towards 16 during fit -- no width tuning
model = GrowingSCCNet(n_chans=22, n_outputs=4, n_times=500, sfreq=250.,
                      n_spatial_filters=4, target_n_spatial_filters=16, device="cpu")
clf = make_eeg_classifier(model, max_epochs=80, grow_every=6, device="cpu")
clf.fit(X, y)          # X: (n, n_chans, n_times) float32, y: (n,) int64
clf.predict(X)
```

> **Why a callback?** `EEGClassifier` owns the fit loop; growth is an event between
> epochs that also invalidates the optimizer (gromo swaps the grown weight tensors
> for new `Parameter`s). The callback grows the module on **CPU** (gromo's `eigh`)
> and refreshes `net.optimizer_`. `target_width` is a *soft* cap: growth stops once
> it is reached (a final step may slightly overshoot).

`examples/benchmark_eegclassifier.py` benchmarks a growable model against two fixed
baselines (narrow `start` and `target` width) on a small learnable synthetic set.
First honest result: the growable model trains end-to-end and auto-sizes, but a
from-scratch `fixed-target` still generalises best — gromo's line search greedily
minimises the **train** loss. Closing that gap (held-out line search / growth
regularisation) and a **real-data MOABB benchmark** are the immediate next steps.

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
