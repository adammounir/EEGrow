"""`init: braindecode` draws GrowingShallowFBCSPNet's weights from bd_shallow's law.

braindecode's ShallowFBCSPNet (1.5.2, shallow_fbcsp.py l. 205-218) is xavier_uniform
(gain 1) with zero biases; ours defaults to PyTorch's init. Compared per tensor over
seeds, at the target geometry and at the 8-filter start of grow_shallow.
"""

from __future__ import annotations

import pytest
import torch
from braindecode.models import ShallowFBCSPNet

from eegrow import GrowingShallowFBCSPNet

C, K, T = 22, 4, 1000
SEEDS = range(20)


def _bd(seed):
    torch.manual_seed(seed)
    m = ShallowFBCSPNet(n_chans=C, n_outputs=K, n_times=T, final_conv_length="auto")
    s = m.conv_time_spat
    return {"time_w": s.conv_time.weight, "time_b": s.conv_time.bias,
            "spat_w": s.conv_spat.weight, "cls_w": m.final_layer.conv_classifier.weight,
            "cls_b": m.final_layer.conv_classifier.bias}


def _ours(seed, start, init):
    torch.manual_seed(seed)
    m = GrowingShallowFBCSPNet(n_chans=C, n_outputs=K, n_times=T, n_filters_time=start,
                               target_n_filters_time=40, init=init)
    return {"time_w": m.conv_time.layer.weight, "time_b": m.conv_time.layer.bias,
            "spat_w": m.conv_spat.layer.weight, "cls_w": m.conv_classifier.weight,
            "cls_b": m.conv_classifier.bias}


def _stats(nets, key):
    x = torch.cat([n[key].detach().flatten() for n in nets])
    return float(x.std()), float(x.abs().max())


@pytest.mark.parametrize("start", [40, 8])
def test_braindecode_init_matches_bd_shallow_law(start):
    bd = [_bd(s) for s in SEEDS]
    ours = [_ours(s, start, "braindecode") for s in SEEDS]
    assert bd[0]["time_w"].shape[0] == 40 and bd[0]["spat_w"].shape == (40, 40, 1, C)
    for key in ("time_w", "spat_w", "cls_w"):
        (s_bd, m_bd), (s_us, m_us) = _stats(bd, key), _stats(ours, key)
        assert abs(s_us / s_bd - 1) < 0.05, (key, s_us, s_bd)
        assert m_us <= m_bd * 1.0001, (key, m_us, m_bd)  # same uniform bound
    for key in ("time_b", "cls_b"):
        assert all(float(n[key].abs().max()) == 0.0 for n in ours)


def test_pytorch_default_is_unchanged_and_differs_from_bd():
    """The default (every existing cell) is still PyTorch's init: bit-identical to a
    build without the keyword, and not bd's law (conv_time std ~x2.6)."""
    torch.manual_seed(0)
    ref = GrowingShallowFBCSPNet(n_chans=C, n_outputs=K, n_times=T, n_filters_time=8,
                                 target_n_filters_time=40)
    a = _ours(0, 8, "pytorch")
    assert torch.equal(a["time_w"], ref.conv_time.layer.weight)
    assert torch.equal(a["spat_w"], ref.conv_spat.layer.weight)
    ratio = _stats([_ours(s, 40, "pytorch") for s in SEEDS], "time_w")[0] / \
        _stats([_bd(s) for s in SEEDS], "time_w")[0]
    assert ratio > 2


def test_unknown_init_refuses():
    with pytest.raises(ValueError, match="init must be"):
        GrowingShallowFBCSPNet(n_chans=C, n_outputs=K, n_times=T, init="xavier")
