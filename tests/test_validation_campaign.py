"""What the 21/10 validation campaign adds: the TINY paper's amplitude, ``fixed_s`` on
several junctions, every FFN of EEGConformer / CTNet growable, and the model recipes'
optimiser knobs.

``tiny_paper`` must be exactly App. E.3.1 eq. (197)-(198) of the TINY paper: mean
squared norm 1e-3 per new neuron on both sides, and Algorithm 1's "add the neurons
only" -- the existing weights do not move.
"""

from __future__ import annotations

import pytest
import torch
from braindecode.models import CTNet, EEGConformer
from omegaconf import OmegaConf

from eegrow import GrowingShallowFBCSPNet
from eegrow.models.growing_bd_junctions import (GrowingCTNetAll, GrowingEEGConformerAll,
                                                GrowingEEGNeXJunction, GrowingMSVTNetAll,
                                                GrowingTIDNet)
from eegrow.training import loop

C, T, K = 22, 1000, 4
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _loaders(seed=1):
    g = torch.Generator().manual_seed(seed)
    X, y = torch.randn(64, C, T, generator=g), torch.randint(0, K, (64,), generator=g)
    ds = torch.utils.data.TensorDataset(X, y)
    train = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(48)), 16)
    hold = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(48, 64)), 16)
    return train, hold, X, y


ALL = {"conformer": (GrowingEEGConformerAll, EEGConformer),
       "ctnet": (GrowingCTNetAll, CTNet)}


@pytest.mark.parametrize("name", sorted(ALL))
def test_all_ffn_parity_with_braindecode_and_it_can_fail(name):
    G, B = ALL[name]
    torch.manual_seed(0)
    bd = B(n_chans=C, n_outputs=K, n_times=T).eval()
    g = G(n_chans=C, n_outputs=K, n_times=T, start_width=160).eval()
    g.load_braindecode_weights(bd)
    assert len(g.pairs) == 6 and g.junction_targets == [160] * 6
    x = torch.randn(4, C, T)
    with torch.no_grad():
        assert (g(x) - bd(x)).abs().max() <= 1e-5
        g.pairs[5].second.layer.weight[0].add_(0.5)
        assert (g(x) - bd(x)).abs().max() > 1e-3


@pytest.mark.parametrize("name", sorted(ALL))
def test_all_ffn_grows_one_junction_per_step(name):
    G, _ = ALL[name]
    torch.manual_seed(0)
    m = G(n_chans=C, n_outputs=K, n_times=T, start_width=4)
    train, hold, X, y = _loaders()
    res = loop.grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                         select="best", fixed_s=0.25,
                         normalization="match_extending_layer")
    assert res["applied"] and res["s"] == 0.25 and res["fixed_s"] == 0.25
    assert res["line_search_s"] in GRID
    grown = [w for w in m.junction_widths if w > 4]
    assert len(grown) == 1 and m.junction_widths[res["junction"]] > 4
    m.train()
    loss = torch.nn.functional.cross_entropy(m(X[:16]), y[:16])
    loss.backward()
    assert torch.isfinite(loss)


def _shallow():
    return GrowingShallowFBCSPNet(n_chans=C, n_outputs=K, n_times=T, n_filters_time=4,
                                  n_filters_spat=8, target_n_filters_time=12)


SITES = {
    "shallow": (_shallow, "first"),
    "tidnet": (lambda: GrowingTIDNet(n_chans=C, n_outputs=K, n_times=T, start_width=8),
               "first"),
    "eegnex_j2": (lambda: GrowingEEGNeXJunction(n_chans=C, n_outputs=K, n_times=T,
                                                junction=2, start_width=4), "first"),
    "msvtnet_all": (lambda: GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T,
                                              start_width=8), "best"),
    "conformer_all": (lambda: GrowingEEGConformerAll(n_chans=C, n_outputs=K, n_times=T,
                                                     start_width=8), "best"),
}


@pytest.mark.parametrize("name", sorted(SITES))
def test_tiny_paper_amplitude_and_no_best_update(name):
    build, select = SITES[name]
    torch.manual_seed(0)
    m = build()
    layers = list(m._growable_layers)
    old = {id(l): (l.layer.weight.detach().clone(),
                   l.previous_module.layer.weight.detach().clone(),
                   None if l.previous_module.layer.bias is None
                   else l.previous_module.layer.bias.detach().clone()) for l in layers}
    train, hold, _, _ = _loaders()
    res = loop.grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                         select=select, normalization="tiny_paper", fixed_s=1.0)
    assert res["applied"] and res["s"] == 1.0
    layer = layers[res["junction"]] if select == "best" else layers[0]
    w2_old, w1_old, b1_old = old[id(layer)]
    n_old, k = w2_old.shape[1], layer.layer.weight.shape[1] - w2_old.shape[1]
    assert k > 0
    w2, w1 = layer.layer.weight.detach(), layer.previous_module.layer.weight.detach()
    # Algorithm 1 adds neurons only: the existing weights are untouched.
    torch.testing.assert_close(w2[:, :n_old], w2_old, rtol=0, atol=0)
    torch.testing.assert_close(w1[:n_old], w1_old, rtol=0, atol=0)
    # eq. (197)-(198): mean squared norm per new neuron = 1e-3, on each side.
    omega_sq = float(w2[:, n_old:].double().pow(2).sum()) / k
    alpha_sq = float(w1[n_old:].double().pow(2).sum())
    if b1_old is not None:
        b1 = layer.previous_module.layer.bias.detach()
        torch.testing.assert_close(b1[:n_old], b1_old, rtol=0, atol=0)
        alpha_sq += float(b1[n_old:].double().pow(2).sum())
    assert omega_sq == pytest.approx(1e-3, rel=1e-5)
    assert alpha_sq / k == pytest.approx(1e-3, rel=1e-5)


@pytest.mark.parametrize("norm,s", [(None, 1.0), ("tiny_paper", 1.0)])
def test_tidnet_grows_twice(norm, s):
    """gromo re-creates a biased conv's S at (in + 1) * k instead of in * k + 1 after a
    growth (smoke 802688): the SECOND step died. ``_fix_conv_statistic_shapes``."""
    torch.manual_seed(0)
    m = GrowingTIDNet(n_chans=C, n_outputs=K, n_times=T + 1, start_width=8)
    g = torch.Generator().manual_seed(1)
    tr = [(torch.randn(16, C, T + 1, generator=g), torch.randint(0, K, (16,), generator=g))
          for _ in range(3)]
    widths = [m.growable_width]
    for _ in range(3):
        r = loop.grow_step(m, tr, "cpu", val_loader=tr[:1], scaling_grid=GRID,
                           min_singular_ratio=0.5, normalization=norm, fixed_s=s)
        widths.append(r["width_after"])
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        m.train()
        for x, y in tr:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(x), y).backward()
            opt.step()
    assert widths[-1] > widths[1] > widths[0]


def test_tiny_paper_refuses_a_line_search():
    train, hold, _, _ = _loaders()
    with pytest.raises(ValueError, match="tiny_paper"):
        loop.grow_step(_shallow(), train, "cpu", val_loader=hold, scaling_grid=GRID,
                       normalization="tiny_paper")


def test_rescale_still_refused_on_several_junctions():
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8)
    train, hold, _, _ = _loaders()
    with pytest.raises(ValueError, match="rescale"):
        loop.grow_step(m, train, "cpu", val_loader=hold, select="best", rescale=True)


def _clf(**train):
    import pipelines
    cfg = OmegaConf.create({"lr": 1e-3, "max_epochs": 1, "batch_size": 8, **train})
    model = OmegaConf.create({"kind": "bd", "arch": "ShallowFBCSPNet", "label": "bd_x",
                              "module_kwargs": {}})
    pipe = pipelines._build_dl(model, cfg, n_chans=C, n_times=T, n_outputs=K, sfreq=250.0,
                               device="cpu", seed=0)
    clf = pipe.named_steps["clf"]
    clf.initialize()
    return clf.optimizer_


def test_recipe_optimiser_knobs():
    o = _clf()
    assert type(o) is torch.optim.AdamW and o.defaults["betas"] == (0.9, 0.999)
    assert o.defaults["amsgrad"] is False and o.defaults["weight_decay"] == 0.01
    o = _clf(optimizer="adam", betas=[0.5, 0.999], amsgrad=True, weight_decay=0.01)
    assert type(o) is torch.optim.Adam and o.defaults["betas"] == (0.5, 0.999)
    assert o.defaults["amsgrad"] is True and o.defaults["weight_decay"] == 0.01
    with pytest.raises(ValueError, match="optimizer"):
        _clf(optimizer="sgd")
