"""The two 20/10 attribution controls of ``loop.grow_step``: ``fixed_s`` and ``rescale``.

``rescale`` is only a valid control if it changes the weight SCALE and nothing else:
the grown network must compute the same function with and without it, in train mode
(batch statistics) and in eval mode (running statistics). That is what is pinned here.
"""

from __future__ import annotations

import copy

import pytest
import torch

from eegrow import GrowingShallowFBCSPNet
from eegrow.training import loop

C, T, N_CLASSES = 8, 256, 4
DEVICE = "cpu"


def _model():
    torch.manual_seed(0)
    m = GrowingShallowFBCSPNet(n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=4,
                               n_filters_spat=8, target_n_filters_time=12, device=DEVICE)
    # Non-trivial running statistics, as after some training.
    m.train()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for _ in range(5):
            m(torch.randn(16, C, T, generator=g) * 3)
    return m


def _batches(seed=1, n=4):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(16, C, T, generator=g),
             torch.randint(0, N_CLASSES, (16,), generator=g)) for _ in range(n)]


@pytest.mark.parametrize("norm", [None, "match_extending_layer"])
def test_rescale_preserves_the_function_and_restores_the_norms(norm):
    a = _model()
    b = copy.deepcopy(a)
    before = loop._out_channel_norms(b.conv_spat.layer.weight).clone()
    kw = dict(min_singular_ratio=0.1, normalization=norm, fixed_s=16.0)
    ra = loop.grow_step(a, _batches(), DEVICE, **kw)
    rb = loop.grow_step(b, _batches(), DEVICE, rescale=True, **kw)
    assert ra["applied"] and rb["applied"] and ra["s"] == rb["s"] == 16.0
    assert ra["width_after"] == rb["width_after"]
    assert rb["rescale_factors"] is not None and ra["rescale_factors"] is None
    # norms back to their pre-growth values, and the step had moved them
    after = loop._out_channel_norms(b.conv_spat.layer.weight)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=0)
    assert not torch.allclose(loop._out_channel_norms(a.conv_spat.layer.weight), before)
    x = _batches(seed=7, n=1)[0][0]
    with torch.no_grad():
        a.eval(); b.eval()
        torch.testing.assert_close(b(x), a(x), rtol=1e-3, atol=1e-4)
        a.train(); b.train()
        torch.testing.assert_close(b(x), a(x), rtol=1e-3, atol=1e-4)


def test_fixed_s_overrides_the_line_search_and_records_it():
    m = _model()
    r = loop.grow_step(m, _batches(), DEVICE, min_singular_ratio=0.1,
                       scaling_grid=(0.0, 0.25, 0.5, 1.0), fixed_s=64.0)
    assert r["applied"] and r["s"] == 64.0 and r["fixed_s"] == 64.0
    assert r["line_search_s"] in (0.0, 0.25, 0.5, 1.0)
    assert set(r["losses"]) == {0.0, 0.25, 0.5, 1.0}


def test_controls_off_by_default_leave_the_record_unchanged():
    r = loop.grow_step(_model(), _batches(), DEVICE, min_singular_ratio=0.1)
    assert r["fixed_s"] is None and r["rescale_factors"] is None
    assert r["s"] == r["line_search_s"]


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_fixed_s_must_be_positive(bad):
    with pytest.raises(ValueError, match="fixed_s"):
        loop.grow_step(_model(), _batches(), DEVICE, fixed_s=bad)
