"""adapt_B.py on synthetic inputs (PREREG_growth_adaptation_B.md): every adapter starts as
the identity function, is live in eval under no_grad (no transformer fast path), growth
equals a widened junction, budgets match the PREREG, splits, AdaBN and the pooled
bootstrap."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import adapt_A  # noqa: E402
import adapt_B  # noqa: E402
import braindecode.models as bdm  # noqa: E402

C, T = 8, 1000
adapt_B._fastpath_off()


def model(name, n_chans=C, seed=0):
    torch.manual_seed(seed)
    return getattr(bdm, adapt_B.ARCH[name])(n_chans=n_chans, n_outputs=2, n_times=T).eval()


def out_weights(d):
    if isinstance(d, adapt_B.FFNDelta):
        return [p["w2"] if d.kind == "grow" else p["B"] for p in d.parts]
    return [d.ws if d.kind == "grow" else d.B]


@pytest.mark.parametrize("name", adapt_B.MODELS)
@pytest.mark.parametrize("kind", ["grow", "lora"])
def test_identity_at_start_and_live_in_eval(name, kind):
    m = model(name)
    x = torch.randn(4, C, T)
    with torch.no_grad():
        ref = m(x)
    d = adapt_B.make_delta(m, name, kind, 2, seed=1)
    for bn in d.bns:
        bn.eval()
    with torch.no_grad():
        assert torch.equal(m(x), ref)
        for w in out_weights(d):
            w.normal_()
        assert (m(x) - ref).abs().max() > 1e-4  # the branch is in the eval forward
    adapt_B.remove(d)
    with torch.no_grad():
        assert torch.equal(m(x), ref)


@pytest.mark.parametrize("name,m2,m8", [("shallow", 2, 8), ("eegnex", 1, 2), ("conformer", 5, 20),
                                        ("ctnet", 5, 20), ("msvtnet", 2, 8)])
def test_budgets(name, m2, m8):
    m = model(name, n_chans=62)
    assert (adapt_B.n_grow(m, name, 2), adapt_B.n_grow(m, name, 8)) == (m2, m8)
    unit, rank = adapt_B.unit_costs(m, name)
    nj = {"conformer": 6, "ctnet": 6, "msvtnet": 2}.get(name, 1)
    for p, mg in ((2, m2), (8, m8)):
        g = adapt_B.make_delta(m, name, "grow", p, 0)
        lo = adapt_B.make_delta(m, name, "lora", p, 0)
        assert sum(q.numel() for q in g.params) == nj * mg * unit
        assert sum(q.numel() for q in lo.params) == nj * p * rank
        adapt_B.remove(g)
        adapt_B.remove(lo)


def test_shallow_is_reading_2():
    for p, (g, lo) in ((2, (5012, 5040)), (8, (20048, 20160))):
        m = model("shallow", n_chans=62)
        assert sum(q.numel() for q in adapt_B.make_delta(m, "shallow", "grow", p, 0).params) == g
        assert sum(q.numel() for q in adapt_B.make_delta(m, "shallow", "lora", p, 0).params) == lo
    m, x = model("shallow", n_chans=62), torch.randn(3, 62, T)
    a = adapt_A.Delta(m, "grow", 2, seed=7)
    a.ws.data.normal_()
    with torch.no_grad():
        ya = m(x)
    a.handle.remove()
    b = adapt_B.make_delta(m, "shallow", "grow", 2, seed=7)
    b.ws.data.copy_(a.ws.data)
    with torch.no_grad():
        assert torch.equal(m(x), ya)


@pytest.mark.parametrize("name", ["conformer", "msvtnet"])
def test_ffn_growth_is_a_wider_ffn(name):
    m = model(name)
    d = adapt_B.make_delta(m, name, "grow", 2, seed=3)
    lin1, act, drop, lin2 = adapt_B.ffns(m, name)[0]
    part = d.parts[0]
    part["w2"].data.normal_()
    x = torch.randn(5, 7, lin1.in_features)
    with torch.no_grad():
        lin1(x)  # keeps x for the hook
        got = lin2(act(lin1(x)))
        W1 = torch.cat([lin1.weight, part["w1"]])
        b1 = torch.cat([lin1.bias, part["b1"]])
        W2 = torch.cat([lin2.weight, part["w2"]], 1)
        want = torch.nn.functional.linear(act(torch.nn.functional.linear(x, W1, b1)), W2, lin2.bias)
    assert torch.allclose(got, want, atol=1e-5)


def test_eegnex_growth_is_a_wider_front():
    m = model("eegnex")
    d = adapt_B.make_delta(m, "eegnex", "grow", 8, seed=3)
    d.ws.data.normal_()
    d.bn.eval()
    c1, bn1, c2 = m.block_1[1], m.block_1[2], m.block_2[0]
    x = torch.randn(2, 1, C, T)
    with torch.no_grad():
        got = c2(bn1(c1(x)))
        h = torch.cat([bn1(c1(x)), d.bn(torch.nn.functional.conv2d(x, d.wt, None, c1.stride, c1.padding))], 1)
        want = torch.nn.functional.conv2d(h, torch.cat([c2.weight, d.ws], 1), None, c2.stride, c2.padding)
    assert torch.allclose(got, want, atol=1e-4)


def test_adabn_every_batchnorm():
    m = model("eegnex")
    d = adapt_B.make_delta(m, "eegnex", "grow", 2, seed=0)
    x = torch.randn(6, C, T)
    adapt_B.adabn(m, x, d.bns)
    with torch.no_grad():
        h = m.block_1[1](m.block_1[0](x))
        z = torch.nn.functional.conv2d(m.block_1[0](x), d.wt, None, m.block_1[1].stride, m.block_1[1].padding)
    assert torch.allclose(m.block_1[2].running_mean, h.mean((0, 2, 3)), atol=1e-5)
    assert torch.allclose(d.bn.running_mean, z.mean((0, 2, 3)), atol=1e-5)
    assert not m.training and not d.bn.training


def test_adapt_arms_smoke():
    torch.manual_seed(0)
    sd = model("msvtnet").state_dict()
    Xc, yc = torch.randn(8, C, T), np.array([0, 1] * 4)
    Xt = torch.randn(6, C, T)
    shape, dev = (C, 2, T), torch.device("cpu")
    z, _ = adapt_B.adapt(sd, "msvtnet", "zero", 0.0, Xc, yc, Xt, 1, shape, dev)
    with torch.no_grad():
        assert np.allclose(z, model("msvtnet").eval()(Xt).numpy(), atol=1e-5)
    g, n = adapt_B.adapt(sd, "msvtnet", "grow2", 1e-2, Xc, yc, Xt, 1, shape, dev)
    a, _ = adapt_B.adapt(sd, "msvtnet", "adabn", 0.0, Xc, yc, Xt, 1, shape, dev)
    assert n == 2 * 2 * 145 and not np.allclose(g, a)
    _, nh = adapt_B.adapt(sd, "msvtnet", "head", 1e-3, Xc, yc, Xt, 1, shape, dev)
    assert nh == 146


def test_splits():
    s2 = np.array(["0"] * 4 + ["1"] * 4 + ["2"] * 4)
    rule, dirs, n = adapt_B.splits(s2, np.array(["0"] * 12), np.zeros(12, int))
    assert rule == "sessions" and n == 3 and dirs[0][1].sum() == 4 and dirs[0][2][8:].all()
    run = np.array(["0train", "0train", "1train", "2test", "2test"])
    rule, dirs, n = adapt_B.splits(np.array(["0"] * 5), run, np.zeros(5, int))
    assert rule == "train/test runs" and n == 6 and list(dirs[0][1]) == [1, 1, 1, 0, 0]
    rule, dirs, _ = adapt_B.splits(np.array(["0"] * 6), np.array(["0", "0", "1", "1", "2", "2"]), np.zeros(6, int))
    assert list(dirs[0][2]) == [0, 0, 0, 0, 1, 1]
    y = np.array([0] * 4 + [1] * 6)  # cho2017: one class, then the other
    rule, dirs, _ = adapt_B.splits(np.array(["0"] * 10), np.array(["0"] * 10), y)
    assert list(dirs[0][1]) == [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]


def test_sboot():
    c = adapt_B.sboot({"a": [0.1] * 4, "b": [0.3] * 16, "c": []})
    assert np.isclose(c["mean"], (2 * 0.1 + 4 * 0.3) / 6) and np.isclose(c["lo"], c["hi"])
    assert c["n"] == 20 and c["n_ds"] == 2 and c["p_sup"] == 0.0
    assert adapt_B.sboot({"a": []})["n"] == 0


def test_adabn_cpu_fallback_on_oom():
    m = model("eegnex")
    d = adapt_B.make_delta(m, "eegnex", "grow", 2, seed=0)
    x = torch.randn(6, C, T)
    adapt_B.adabn(m, x, d.bns, d.params)
    want = [bn.running_var.clone() for bn in adapt_B.bns_of(m, d.bns)]
    fwd, calls = m.forward, []

    def flaky(*a):
        calls.append(1)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("simulated")
        return fwd(*a)
    m.forward = flaky
    adapt_B.adabn(m, x, d.bns, d.params)
    assert len(calls) == 2
    for bn, w in zip(adapt_B.bns_of(m, d.bns), want):
        assert torch.allclose(bn.running_var, w, atol=1e-6)


def test_g0_tolerates_tf32_not_a_different_model():
    m = model("shallow")
    x = torch.randn(10, C, T)
    y = np.array([0, 1] * 5)
    sess = np.array(["0"] * 10)
    lg = adapt_B.logits_of(m, x)
    near = {"0": {"logits": lg + 1e-3, "y": y}}
    assert adapt_B.g0(m, near, x.numpy(), y, sess, torch.device("cpu"), 2)["0"]["ok"]
    other = {"0": {"logits": adapt_B.logits_of(model("shallow", seed=1), x), "y": y}}
    assert not adapt_B.g0(m, other, x.numpy(), y, sess, torch.device("cpu"), 2)["0"]["ok"]
