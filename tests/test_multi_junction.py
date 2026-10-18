"""Several junctions, one grows per step: gromo's ``select_best_update`` (étape 1, lecture a).

``GrowingMSVTNetAll`` makes the FFN of every MSVTNet encoder layer growable. Checked:
- parity with braindecode at braindecode's width (and that the parity can fail);
- a ``select="best"`` step changes exactly ONE junction, the argmax of the recorded
  first-order improvements, and the model still trains;
- each junction stops at its own target and then leaves the competition;
- the split-half measurement does not alter the real step's choice rule.
"""

from __future__ import annotations

import pytest
import torch
from braindecode.models import MSVTNet

from eegrow import GromoGrowth
from eegrow.models.growing_bd_junctions import GrowingMSVTNetAll
from eegrow.training.loop import grow_step

C, T, K = 22, 1000, 4
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
BD_FFN = 72   # dim_feedforward == d_model in braindecode's MSVTNet


def _data(n=64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    X, y = torch.randn(n, C, T, generator=gen), torch.randint(0, K, (n,), generator=gen)
    ds = torch.utils.data.TensorDataset(X, y)
    train = list(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(48)), 16))
    hold = list(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(48, n)), 16))
    return X, y, train, hold


def test_widths_and_targets_are_per_junction_and_summed():
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=2)
    n_layers = len(m.net.transformer.trans.layers)
    assert n_layers == 2
    assert m.junction_widths == [2] * n_layers
    assert m.junction_targets == [BD_FFN] * n_layers
    assert m.growable_width == 2 * n_layers and m.target_width == BD_FFN * n_layers
    assert len(m._growable_layers) == n_layers


@pytest.mark.parametrize("seed", range(10))
def test_activation_gradient_is_pinned_not_a_dropout_draw(seed):
    # Without the pin, train mode caches 0 or 1/(1-p)=2 per junction at random (11/20
    # seeds disagree), and select_best_update ranks on it.
    torch.manual_seed(seed)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8).train()
    assert [float(p.second.activation_gradient) for p in m.pairs] == [1.0, 1.0]


def _pair():
    torch.manual_seed(0)
    bd = MSVTNet(n_chans=C, n_outputs=K, n_times=T).eval()
    g = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=BD_FFN).eval()
    g.load_braindecode_weights(bd)
    return g, bd


def test_parity_with_braindecode():
    g, bd = _pair()
    x = torch.randn(4, C, T)
    with torch.no_grad():
        assert (g(x) - bd(x)).abs().max() <= 1e-5
    assert g._growable_layers == []   # at braindecode's width nothing is left to grow


@pytest.mark.parametrize("junction", [0, 1])
def test_parity_fails_when_a_junction_differs(junction):
    g, bd = _pair()
    with torch.no_grad():
        g.pairs[junction].second.layer.weight[0].add_(0.5)
        x = torch.randn(4, C, T)
        assert (g(x) - bd(x)).abs().max() > 1e-3


def test_best_step_grows_exactly_the_argmax_junction():
    torch.manual_seed(0)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8)
    X, y, train, hold = _data()
    res = grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                    normalization="match_extending_layer", select="best",
                    split_half=True)
    foi = res["junction_foi"]
    assert len(foi) == 2 and all(f is not None for f in foi)
    assert res["junction"] == res["junction_ids"][max(range(2), key=lambda i: foi[i])]
    assert res["split_a"] in (0, 1) and res["split_b"] in (0, 1)
    before, after = res["junction_widths_before"], res["junction_widths_after"]
    changed = [i for i in range(2) if after[i] != before[i]]
    if res["applied"]:
        assert changed == [res["junction"]]
        assert after[res["junction"]] > before[res["junction"]]
    else:
        assert changed == []
    assert m.junction_widths == after and m.growable_width == sum(after)
    m.train()
    loss = torch.nn.functional.cross_entropy(m(X[:16]), y[:16])
    loss.backward()
    assert torch.isfinite(loss)
    for p in m.pairs:   # shapes consistent on both sides of every junction
        assert p.first.layer.weight.shape[0] == p.second.layer.weight.shape[1]


def test_each_junction_stops_at_its_own_target():
    torch.manual_seed(0)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=4,
                          target_width=6)
    _, _, train, hold = _data()
    for _ in range(8):
        if not m._growable_layers:
            break
        grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                  normalization="match_extending_layer", select="best")
        assert all(w <= 6 for w in m.junction_widths), m.junction_widths
        full = [i for i, w in enumerate(m.junction_widths) if w == 6]
        assert all(m.pairs[i].second not in m._growable_layers for i in full)


def test_split_half_needs_best_and_select_is_validated():
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8)
    _, _, train, hold = _data()
    with pytest.raises(ValueError, match="split_half"):
        grow_step(m, train, "cpu", val_loader=hold, split_half=True)
    with pytest.raises(ValueError, match="select"):
        grow_step(m, train, "cpu", val_loader=hold, select="all")
    with pytest.raises(ValueError):
        GromoGrowth(select="round_robin")
    with pytest.raises(ValueError, match="split_half"):
        grow_step(m, train, "cpu", val_loader=hold, select="sequential", split_half=True)
    with pytest.raises(ValueError):
        GromoGrowth(split_half=True)


def test_exported():
    import eegrow
    assert eegrow.GrowingMSVTNetAll is GrowingMSVTNetAll


def _flaky(m, fail_float32_calls):
    """Make gromo's compute_optimal_updates raise LinAlgError on chosen float32 calls."""
    real = m.compute_optimal_updates
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        if kwargs.get("dtype", torch.float32) is torch.float32:
            calls["n"] += 1
            if calls["n"] in fail_float32_calls:
                raise torch.linalg.LinAlgError("simulated cuSOLVER error 46")
        return real(*args, **kwargs)

    m.compute_optimal_updates = wrapped


def test_ill_conditioned_pinv_is_retried_in_float64():
    torch.manual_seed(0)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8)
    _, _, train, hold = _data()
    _flaky(m, fail_float32_calls={1})    # first call = half a
    res = grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                    normalization="match_extending_layer", select="best",
                    split_half=True)
    assert res["split_failed"] == 0 and res["split_fp64_retry"] == 1
    assert res["fp64_retry"] is False and res["split_a"] in (0, 1)


def test_a_half_that_fails_twice_is_recorded_not_fatal(monkeypatch):
    torch.manual_seed(0)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8)
    _, _, train, hold = _data()
    real = m.compute_optimal_updates
    calls = {"n": 0}

    def wrapped(*args, **kwargs):   # half a fails in float32 AND float64
        calls["n"] += 1
        if calls["n"] in (1, 2):
            raise torch.linalg.LinAlgError("simulated")
        return real(*args, **kwargs)

    m.compute_optimal_updates = wrapped
    res = grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                    normalization="match_extending_layer", select="best",
                    split_half=True)
    assert res["split_failed"] == 1 and res["split_a"] is None
    assert res["split_b"] in (0, 1) and res["junction"] in (0, 1)


def test_sequential_alternates_junctions_and_records_the_argmax():
    torch.manual_seed(0)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=8)
    _, _, train, hold = _data()
    chosen = []
    for _ in range(4):
        res = grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                        normalization="match_extending_layer", select="sequential")
        chosen.append(res["junction"])
        assert res["junction_argmax"] in (0, 1)
        changed = [i for i in range(2)
                   if res["junction_widths_after"][i] != res["junction_widths_before"][i]]
        assert changed == ([res["junction"]] if res["applied"] else [])
    assert chosen == [0, 1, 0, 1]


def test_sequential_skips_a_junction_at_target():
    torch.manual_seed(0)
    m = GrowingMSVTNetAll(n_chans=C, n_outputs=K, n_times=T, start_width=4,
                          target_width=6)
    _, _, train, hold = _data()
    for _ in range(8):
        if not m._growable_layers:
            break
        res = grow_step(m, train, "cpu", val_loader=hold, scaling_grid=GRID,
                        normalization="match_extending_layer", select="sequential")
        assert m.junction_widths[res["junction"]] <= 6
        assert all(w <= 6 for w in m.junction_widths)
