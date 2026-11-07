"""adapt_A.py on synthetic inputs (PREREG_growth_adaptation_A.md): the branch reproduces
braindecode's CombinedConv, both adapters start as the identity function, growth equals a
widened conv_time_spat, parameter counts match the PREREG, AdaBN, Holm and verdicts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import adapt_A  # noqa: E402
from braindecode.models import ShallowFBCSPNet  # noqa: E402

C, T = 62, 1000


def model(seed=0):
    torch.manual_seed(seed)
    return ShallowFBCSPNet(n_chans=C, n_outputs=2, n_times=T).eval()


def test_combined_is_braindecode():
    m = model()
    cts = m.conv_time_spat
    x = torch.randn(3, 1, T, C)
    w, b = adapt_A.combined(cts.conv_time.weight, cts.conv_time.bias, cts.conv_spat.weight)
    assert torch.allclose(torch.nn.functional.conv2d(x, w, b), cts(x), atol=1e-4)


@pytest.mark.parametrize("kind", ["grow", "lora"])
def test_adapters_start_as_identity_and_count(kind):
    m = model()
    x = torch.randn(4, C, T)
    ref = m(x).detach()
    d = adapt_A.Delta(m, kind, 2, seed=1)
    assert torch.equal(m(x).detach(), ref)
    assert d.n_params() == {"grow": 5012, "lora": 5040}[kind]
    assert adapt_A.Delta(model(), kind, 8, seed=1).n_params() == {"grow": 20048, "lora": 20160}[kind]


def test_growth_equals_widened_conv():
    m = model()
    d = adapt_A.Delta(m, "grow", 3, seed=2)
    with torch.no_grad():
        d.ws.normal_()
    cts = m.conv_time_spat
    wt = torch.cat([cts.conv_time.weight, d.wt])
    bt = torch.cat([cts.conv_time.bias, d.bt])
    ws = torch.cat([cts.conv_spat.weight, d.ws], 1)
    w, b = adapt_A.combined(wt, bt, ws)
    x = torch.randn(2, 1, T, C)
    assert torch.allclose(cts(x), torch.nn.functional.conv2d(x, w, b), atol=1e-3)


def test_head_count_and_adabn():
    m = model()
    assert sum(p.numel() for p in m.final_layer.conv_classifier.parameters()) == 4882
    assert sum(p.numel() for p in m.parameters()) == 105202
    X = torch.randn(20, C, T) * 3 + 1
    adapt_A.adabn(m, X)
    with torch.no_grad():
        h = m.conv_time_spat(m.dimshuffle(m.ensuredims(X)))
    assert torch.allclose(m.bnorm.running_mean, h.mean((0, 2, 3)), rtol=1e-3, atol=1e-3)
    assert not m.training and m.bnorm.momentum == 0.1


def test_adapt_runs_and_trains_only_its_params():
    m = model()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    X = torch.randn(8, C, T)
    y = np.array([0, 1] * 4)
    for arm in ("zero", "adabn", "head", "lora2", "grow2"):
        lg, n = adapt_A.adapt(sd, arm, 1e-3, X, y, X, seed=0, shape=(C, 2, T))
        assert lg.shape == (8, 2) and np.isfinite(lg).all()
        assert n == {"zero": 0, "adabn": 0, "head": 4882, "lora2": 5040, "grow2": 5012}[arm]


def test_holm_and_verdict():
    assert adapt_A.holm([0.01, 0.04, 0.03]) == [True, False, False]
    assert adapt_A.holm([0.01, 0.02, 0.04]) == [True, True, True]
    assert adapt_A.verdict({"lo": -0.01, "hi": 0.03}, False) == "NON INFÉRIEUR"
    assert adapt_A.verdict({"lo": -0.05, "hi": -0.01}, False) == "INFÉRIEUR"
    assert adapt_A.verdict({"lo": -0.05, "hi": 0.02}, False) == "NON CONCLUANT"
    assert adapt_A.verdict({"lo": 0.01, "hi": 0.05}, True) == "SUPÉRIEUR"
