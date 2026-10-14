"""One junction of EEGConformer / ContraWR / TIDNet grows; the rest is braindecode's.

Parity (étape 1 criterion): at braindecode's width, with braindecode's weights copied
in, the growable model returns braindecode's logits (max |diff| <= 1e-5). The parity
is also checked to FAIL when one junction weight is perturbed -- a parity test that
cannot fail would be measuring nothing.
"""

from __future__ import annotations

import pytest
import torch
from braindecode.models import CTNet, ContraWR, EEGConformer, EEGNeX, MSVTNet, TIDNet

from eegrow.models.growing_bd_junctions import (GrowingContraWR, GrowingCTNet,
                                                GrowingEEGConformer, GrowingEEGNeXJunction,
                                                GrowingMSVTNet, GrowingTIDNet)
from eegrow.training.loop import grow_step

C, T, K, SF = 22, 1000, 4, 250.0
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

VARIANTS = {
    "conformer_ffn_b0": (GrowingEEGConformer, EEGConformer, dict(junction="ffn", block=0), {}),
    "conformer_ffn_b5": (GrowingEEGConformer, EEGConformer, dict(junction="ffn", block=5), {}),
    "conformer_patch": (GrowingEEGConformer, EEGConformer, dict(junction="patch"), {}),
    "contrawr_b0": (GrowingContraWR, ContraWR, dict(block=0), dict(sfreq=SF)),
    "contrawr_b3": (GrowingContraWR, ContraWR, dict(block=3), dict(sfreq=SF)),
    "tidnet_d0": (GrowingTIDNet, TIDNet, dict(block=0), {}),
    "tidnet_d1": (GrowingTIDNet, TIDNet, dict(block=1), {}),
    "ctnet_b0": (GrowingCTNet, CTNet, dict(block=0), {}),
    "msvtnet_b0": (GrowingMSVTNet, MSVTNet, dict(block=0), {}),
    "msvtnet_b1": (GrowingMSVTNet, MSVTNet, dict(block=1), {}),
    "eegnex_j2": (GrowingEEGNeXJunction, EEGNeX, dict(junction=2), {}),
}
BD_WIDTH = {"conformer_ffn_b0": 160, "conformer_ffn_b5": 160, "conformer_patch": 40,
            "contrawr_b0": 32, "contrawr_b3": 256, "tidnet_d0": 72, "tidnet_d1": 72,
            "ctnet_b0": 160, "msvtnet_b0": 72, "msvtnet_b1": 72, "eegnex_j2": 32}


def _pair(name):
    G, B, kw, extra = VARIANTS[name]
    torch.manual_seed(0)
    bd = B(n_chans=C, n_outputs=K, n_times=T, **extra).eval()
    g = G(n_chans=C, n_outputs=K, n_times=T, **extra, **kw,
          start_width=BD_WIDTH[name]).eval()
    g.load_braindecode_weights(bd)
    return g, bd


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_target_width_is_braindecode_width(name):
    G, _, kw, extra = VARIANTS[name]
    m = G(n_chans=C, n_outputs=K, n_times=T, **extra, **kw, start_width=2)
    assert m.target_width == BD_WIDTH[name] and m.growable_width == 2


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_parity_with_braindecode(name):
    g, bd = _pair(name)
    x = torch.randn(4, C, T)
    with torch.no_grad():
        assert (g(x) - bd(x)).abs().max() <= 1e-5


@pytest.mark.parametrize("name", ["conformer_ffn_b5", "contrawr_b0", "tidnet_d0", "ctnet_b0",
                                  "msvtnet_b1", "eegnex_j2"])
def test_parity_fails_when_the_junction_differs(name):
    g, bd = _pair(name)
    with torch.no_grad():
        g.pair.second.layer.weight[0].add_(0.5)
        x = torch.randn(4, C, T)
        assert (g(x) - bd(x)).abs().max() > 1e-3


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_one_growth_step_widens_and_still_trains(name):
    G, _, kw, extra = VARIANTS[name]
    torch.manual_seed(0)
    start = max(2, BD_WIDTH[name] // 8)
    m = G(n_chans=C, n_outputs=K, n_times=T, **extra, **kw, start_width=start)
    gen = torch.Generator().manual_seed(1)
    X, y = torch.randn(64, C, T, generator=gen), torch.randint(0, K, (64,), generator=gen)
    ds = torch.utils.data.TensorDataset(X, y)
    train = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(48)), 16)
    hold = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(48, 64)), 16)
    res = grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                    normalization="match_extending_layer")
    assert sorted(res["losses"]) == list(GRID)
    assert res["n_proposed"] and res["n_proposed"] > 0
    if res["applied"]:
        assert start < m.growable_width <= BD_WIDTH[name]
    else:
        assert m.growable_width == start
    m.train()
    loss = torch.nn.functional.cross_entropy(m(X[:16]), y[:16])
    loss.backward()
    assert torch.isfinite(loss)
    # every grown parameter receives a gradient (no dead splice)
    assert m.pair.second.layer.weight.grad is not None
    assert m.pair.first.layer.weight.shape[0] == m.growable_width


def test_contrawr_junction_batchnorm_grows_with_the_layer():
    torch.manual_seed(0)
    m = GrowingContraWR(n_chans=C, n_outputs=K, n_times=T, sfreq=SF, block=0,
                        start_width=4)
    bn = m.pair.first.post_layer_function[0]
    gen = torch.Generator().manual_seed(1)
    X, y = torch.randn(48, C, T, generator=gen), torch.randint(0, K, (48,), generator=gen)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), 16)
    res = grow_step(m, loader, "cpu", val_loader=loader, scaling_grid=GRID,
                    normalization="match_extending_layer")
    assert bn.num_features == m.growable_width == bn.weight.shape[0], res["s"]


def test_exported():
    import eegrow
    assert eegrow.GrowingEEGConformer is GrowingEEGConformer
    assert eegrow.GrowingContraWR is GrowingContraWR
    assert eegrow.GrowingTIDNet is GrowingTIDNet
    assert eegrow.GrowingCTNet is GrowingCTNet
    assert eegrow.GrowingMSVTNet is GrowingMSVTNet
    assert eegrow.GrowingEEGNeXJunction is GrowingEEGNeXJunction


def test_eegnex_asymmetric_same_padding_junction_is_refused():
    with pytest.raises(ValueError, match="asymmetric"):
        GrowingEEGNeXJunction(n_chans=C, n_outputs=K, n_times=T, junction=1)
