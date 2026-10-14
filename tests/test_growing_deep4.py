"""GrowingDeep4Net: parity with braindecode's Deep4Net, and its first junction grows."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from braindecode.models import Deep4Net

from eegrow.models.growing_deep4 import GrowingDeep4Net


@pytest.mark.parametrize("n_chans,n_times", [(22, 1000), (22, 1001), (3, 1125)])
def test_parity_with_braindecode_deep4(n_chans, n_times):
    """Criterion of étape 1: same logits as Deep4Net with copied weights (<= 1e-5)."""
    torch.manual_seed(0)
    bd = Deep4Net(n_chans=n_chans, n_outputs=4, n_times=n_times).eval()
    g = GrowingDeep4Net(n_chans, 4, n_times, target_n_filters_time=50,
                        device="cpu").load_braindecode_weights(bd).eval()
    x = torch.randn(8, n_chans, n_times)
    with torch.no_grad():
        diff = (bd(x) - g(x)).abs().max().item()
    assert diff <= 1e-5, diff


def test_tail_is_braindecodes_own_modules():
    g = GrowingDeep4Net(22, 4, 1000, device="cpu")
    bd = Deep4Net(n_chans=22, n_outputs=4, n_times=1000)
    names = [type(m).__name__ for m in g.tail]
    ref = [type(m).__name__ for n, m in bd.named_children()
           if n not in ("ensuredims", "dimshuffle", "conv_time_spat")]
    assert names == ref
    assert sum(p.numel() for p in g.parameters()) == sum(p.numel() for p in bd.parameters())


def test_parity_holds_under_stride_before_pool():
    """CombinedConv never strides; the strided convs live in the borrowed tail."""
    torch.manual_seed(1)
    bd = Deep4Net(n_chans=22, n_outputs=4, n_times=1000, stride_before_pool=True).eval()
    g = GrowingDeep4Net(22, 4, 1000, stride_before_pool=True,
                        device="cpu").load_braindecode_weights(bd).eval()
    x = torch.randn(4, 22, 1000)
    with torch.no_grad():
        assert (bd(x) - g(x)).abs().max().item() <= 1e-5


def test_first_junction_grows_to_target():
    from braindecode import EEGClassifier
    from eegrow import GromoGrowth

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((96, 4, 500)).astype("float32")
    y = np.tile(np.arange(4), 24).astype("int64")
    model = GrowingDeep4Net(4, 4, 500, n_filters_time=8, target_n_filters_time=16,
                            device="cpu")
    net = EEGClassifier(model, criterion=torch.nn.CrossEntropyLoss, max_epochs=6,
                        batch_size=32, iterator_train__drop_last=False,
                        train_split=None,
                        callbacks=[("gromo", GromoGrowth(grow_every=1, verbose=False))],
                        device="cpu", classes=list(range(4)), verbose=0)
    net.fit(X, y)
    assert 8 < net.module_.growable_width <= 16
    assert net.predict(X).shape == (96,)


@pytest.mark.parametrize("stride", [3, 1])
def test_gromo_refuses_a_pooling_inside_a_junction(stride):
    """Étape 1a, by test: a MaxPool in post_layer_function breaks the statistics pass.

    If gromo ever supports it, this test fails and the pooled junctions of Deep4Net
    (blocks 2-4) become candidates -- re-read the source before using them.
    """
    from torch import nn
    from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule as R

    torch.manual_seed(0)
    l1 = R(in_channels=4, out_channels=5, kernel_size=(5, 1), use_bias=True,
           post_layer_function=nn.Sequential(nn.ELU(), nn.MaxPool2d((3, 1), (stride, 1))),
           name="l1", device="cpu")
    l2 = R(in_channels=5, out_channels=6, kernel_size=(5, 1), use_bias=True,
           previous_module=l1, name="l2", device="cpu")
    l1.next_module = l2
    l2.init_computation()
    loss = (l2(l1(torch.randn(16, 4, 60, 1))) ** 2).sum()
    loss.backward()
    with pytest.raises(RuntimeError):
        l2.update_computation()
        l2.compute_optimal_updates()
