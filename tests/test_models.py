"""Tests for the eegrow growable models.

Pins the manual validation as automated checks: every model runs a forward pass
and one gromo growth step (width increases, respects the target cap, forward still
works), `GrowingSCCNet` is bit-exact with braindecode, and `GrowingEEGNeX` matches
braindecode's output shape.

Growth relies on `torch.linalg.eigh` (absent on MPS) -> everything runs on CPU,
which is also what CI provides.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from eegrow import (
    GrowingDeepEEGNet,
    GrowingEEGNeX,
    GrowingSCCNet,
    GrowingShallowFBCSPNet,
)
from eegrow.training import loop

# small dims keep the whole suite fast on a CPU runner
C, T, N_CLASSES, N = 8, 256, 4, 32
SFREQ = 128.0
DEVICE = "cpu"


def _loader(seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, C, T, generator=g)
    y = torch.randint(0, N_CLASSES, (N,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def _assert_grows(model, cap: int) -> None:
    """One growth step: width strictly increases, stays <= cap, forward still works."""
    loader = _loader()
    x = next(iter(loader))[0]
    out_before = model(x)
    assert out_before.shape == (16, N_CLASSES)

    w0 = model.growable_width
    loop.grow_step(model, loader, DEVICE)
    w1 = model.growable_width

    assert w1 > w0, f"width did not grow ({w0} -> {w1})"
    assert w1 <= cap, f"width {w1} exceeded target cap {cap}"
    out_after = model(x)
    assert out_after.shape == (16, N_CLASSES)
    assert torch.isfinite(out_after).all()


# --------------------------------------------------------------------- growth
def test_shallow_grows():
    model = GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        n_filters_time=4, n_filters_spat=8, target_n_filters_time=16, device=DEVICE,
    )
    _assert_grows(model, cap=16)


def test_deepeeg_grows():
    model = GrowingDeepEEGNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        w1=4, w2=4, target_w2=16, device=DEVICE,
    )
    _assert_grows(model, cap=16)


def test_sccnet_grows():
    model = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=4, n_spatial_filters_smooth=8,
        target_n_spatial_filters=16, device=DEVICE,
    )
    _assert_grows(model, cap=16)


def test_eegnex_grows():
    model = GrowingEEGNeX(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        filter_1=4, filter_2=16, target_filter_1=16, device=DEVICE,
    )
    _assert_grows(model, cap=16)


# --------------------------------------------------------------- fidelity
def test_sccnet_equivalence_with_braindecode():
    """A frozen GrowingSCCNet loaded with braindecode weights is bit-exact."""
    from braindecode.models import SCCNet

    torch.manual_seed(0)
    bd = SCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=12, n_spatial_filters_smooth=8,
    ).eval()
    frozen = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=12, n_spatial_filters_smooth=8, device=DEVICE,
    ).eval()
    frozen.load_braindecode_weights(bd)

    x = torch.randn(8, C, T)
    with torch.no_grad():
        max_diff = (bd(x) - frozen(x)).abs().max().item()
    assert max_diff < 1e-5, f"SCCNet not equivalent: max|diff| = {max_diff:.2e}"


def test_eegnex_shape_parity_with_braindecode():
    from braindecode.models import EEGNeX

    torch.manual_seed(0)
    bd = EEGNeX(n_chans=C, n_outputs=N_CLASSES, n_times=T).eval()
    growing = GrowingEEGNeX(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        filter_1=8, filter_2=16, device=DEVICE,
    ).eval()
    x = torch.randn(8, C, T)
    with torch.no_grad():
        assert bd(x).shape == growing(x).shape


# --------------------------------------------------------------- frozen path
def test_frozen_models_have_no_growable_layers():
    """Without a target > initial width, a model is frozen (baseline mode)."""
    model = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=8, n_spatial_filters_smooth=8, device=DEVICE,
    )
    assert model._growable_layers == []
    x = torch.randn(4, C, T)
    assert model(x).shape == (4, N_CLASSES)


# --------------------------------------------------- skorch / EEGClassifier
def test_eegclassifier_grows_during_fit():
    """A growable model trained through braindecode's EEGClassifier grows during
    fit (callback fires), respects the target cap, and stays predictable."""
    import numpy as np

    from eegrow import make_eeg_classifier

    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")

    torch.manual_seed(0)
    model = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=4, n_spatial_filters_smooth=8,
        target_n_spatial_filters=12, device=DEVICE,
    )
    w0 = model.growable_width
    # one growth step within the 6 epochs (epoch 4): a single step stays under the cap.
    clf = make_eeg_classifier(
        model, max_epochs=6, grow_every=4, batch_size=16, device=DEVICE,
        verbose=False,
    )
    clf.fit(X, y)

    assert model.growable_width > w0, "EEGClassifier training did not grow the model"
    assert model.growable_width <= model.target_width, "single growth exceeded the cap"
    preds = clf.predict(X)
    assert preds.shape == (N,)
    assert set(np.unique(preds)).issubset(set(range(N_CLASSES)))


def test_eegclassifier_frozen_model_is_noop():
    """With no target width the growth callback is a no-op (plain baseline)."""
    from eegrow import make_eeg_classifier

    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")

    torch.manual_seed(0)
    model = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=8, n_spatial_filters_smooth=8, device=DEVICE,
    )
    clf = make_eeg_classifier(
        model, max_epochs=4, grow_every=2, batch_size=16, device=DEVICE,
        verbose=False,
    )
    clf.fit(X, y)
    assert model.growable_width == 8, "frozen model should not grow"
