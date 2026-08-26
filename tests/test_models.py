"""Tests for the eegrow growable models.

Pins the manual validation as automated checks: every model runs a forward pass
and one gromo growth step (width increases, respects the target cap, forward still
works), `GrowingSCCNet` is bit-exact with braindecode, and `GrowingEEGNeX` matches
braindecode's output shape.

Growth relies on `torch.linalg.eigh` (absent on MPS) -> everything runs on CPU,
which is also what CI provides.
"""

from __future__ import annotations

import pytest
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


def _assert_grows(model, cap: int) -> dict:
    """One growth step leaves the model coherent, whichever way the step decided.

    This used to assert "the width strictly increases", which was only ever true
    because ``grow_step`` applied the change unconditionally -- including at scaling
    factor 0, where the neurons it splices in have identically zero weights and are
    dead on arrival. Now that the step abstains at s=0, *not growing* is a legitimate
    outcome, and on the random labels this loader produces it is often the correct
    one: there is no signal for a new neuron to capture.

    So the invariant is the one that actually matters either way -- the proposal ran,
    the decision is internally consistent, the target cap holds, and the model still
    computes -- and the two branches are checked separately rather than one of them
    being asserted away.
    """
    loader = _loader()
    x = next(iter(loader))[0]
    out_before = model(x)
    assert out_before.shape == (16, N_CLASSES)

    w0 = model.growable_width
    result = loop.grow_step(model, loader, DEVICE)
    w1 = model.growable_width

    assert result["n_proposed"] >= 1, "gromo proposed no candidate neuron at all"
    assert result["width_after"] == w1, "reported width disagrees with the model"
    if result["applied"]:
        assert result["s"] > 0, "applied a change at scaling factor 0"
        assert w1 > w0, f"applied a change but width did not grow ({w0} -> {w1})"
    else:
        assert result["s"] == 0, "abstained at a non-zero scaling factor"
        assert w1 == w0, f"abstained but width moved ({w0} -> {w1})"
    assert w1 <= cap, f"width {w1} exceeded target cap {cap}"
    out_after = model(x)
    assert out_after.shape == (16, N_CLASSES)
    assert torch.isfinite(out_after).all()
    return result


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


# ------------------------------------------------------- the cap actually binds
# The four tests above take ONE growth step, which never reaches the target -- so by
# construction they cannot see a missing cap. And the cap does not live in gromo,
# which adds a data-dependent count per step and never stops on its own; it lives in
# the GromoGrowth callback, which reads `getattr(model, "target_width", None)` both
# to stop growing and to trim the step via sub_select_optimal_added_parameters.
#
# GrowingShallowFBCSPNet and GrowingDeepEEGNet did not declare that attribute, so for
# them the cap silently did not exist -- measured 8 -> 77 against a target of 32 on
# GrowingDeepEEGNet in nine growth events. The benchmark's grow/fixed pairs were
# therefore not width-matched, and the paired contrast did not mean what it said.
# These two tests drive the callback path -- the one the benchmark uses -- over
# repeated growths, on every growable model.
BUILDERS = {
    "shallow": lambda cap: GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=4,
        n_filters_spat=8, target_n_filters_time=cap, device=DEVICE),
    "deepeeg": lambda cap: GrowingDeepEEGNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, w1=4, w2=4, target_w2=cap,
        device=DEVICE),
    "sccnet": lambda cap: GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=4, n_spatial_filters_smooth=8,
        target_n_spatial_filters=cap, device=DEVICE),
    "eegnex": lambda cap: GrowingEEGNeX(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, filter_1=4, filter_2=16,
        target_filter_1=cap, device=DEVICE),
}
CAP = 10


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_growable_model_declares_target_width(name):
    """Every growable model must expose the attribute the growth callback reads."""
    model = BUILDERS[name](CAP)
    assert getattr(model, "target_width", None) == CAP, (
        f"{name} does not declare target_width; GromoGrowth would read None, skip "
        "sub_select_optimal_added_parameters and grow without any bound")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_growth_saturates_at_target(name):
    """Repeated growth lands exactly on the target and never overshoots it."""
    from braindecode import EEGClassifier

    from eegrow import GromoGrowth

    model = BUILDERS[name](CAP)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")

    widths = [model.growable_width]

    class _Probe(GromoGrowth):
        # A subclass rather than a second callback: the width has to be read *after*
        # the growth, and callback ordering is one more thing to get wrong.
        def on_epoch_end(self, net, **kwargs):
            super().on_epoch_end(net, **kwargs)
            widths.append(net.module_.growable_width)

    clf = EEGClassifier(
        model, criterion=torch.nn.CrossEntropyLoss, max_epochs=12, batch_size=16,
        callbacks=[("gromo", _Probe(grow_every=1, verbose=False))],
        train_split=None, device=DEVICE,
    )
    clf.fit(x, y)

    assert max(widths) <= CAP, f"{name} overshot the cap: {widths} (cap {CAP})"
    assert model.growable_width == CAP, (
        f"{name} stalled at {model.growable_width} instead of reaching {CAP}: {widths}")


# The frozen twin of each growing arm: the SAME class built directly at the geometry
# growth ends on. These are the benchmark's fixed controls (config/model/fix_*.yaml),
# and the whole growth contrast rests on them being width-matched.
#
# Two of the four need a dedicated "junction width" kwarg, because the width knob also
# sizes a FIXED module that growth never touches: `w2` sizes conv2b's output as well as
# the junction, and `filter_1` sizes block_5 (hence the classifier) as well as the
# junction. Building the naive twin (`w2=cap`, `filter_1=cap`) gives a genuinely
# different, larger network -- on EEGNeX, 280 classifier inputs against the grown net's
# 70. That is what this test exists to catch.
FIXED_TWINS = {
    "shallow": lambda cap: GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=cap,
        n_filters_spat=8, device=DEVICE),
    "deepeeg": lambda cap: GrowingDeepEEGNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, w1=4, w2=4, w2_in=cap,
        device=DEVICE),
    "sccnet": lambda cap: GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=cap, n_spatial_filters_smooth=8, device=DEVICE),
    "eegnex": lambda cap: GrowingEEGNeX(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, filter_1=4, filter_1_in=cap,
        filter_2=16, device=DEVICE),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_fixed_control_matches_grown_geometry(name):
    """The frozen control is the network growth ends on, parameter for parameter.

    Shapes, not values: the control is trained from its own init, so only the geometry
    has to match. Any key whose shape differs means the pair is not width-matched and
    the paired contrast is measuring architecture on top of growth.
    """
    from braindecode import EEGClassifier

    from eegrow import GromoGrowth

    grown = BUILDERS[name](CAP)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")
    EEGClassifier(
        grown, criterion=torch.nn.CrossEntropyLoss, max_epochs=12, batch_size=16,
        callbacks=[("gromo", GromoGrowth(grow_every=1, verbose=False))],
        train_split=None, device=DEVICE,
    ).fit(x, y)
    assert grown.growable_width == CAP

    fixed = FIXED_TWINS[name](CAP)
    assert not getattr(fixed, "_can_grow", True), f"{name} twin is not frozen"
    a = {k: tuple(v.shape) for k, v in grown.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in fixed.state_dict().items()}
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert not diff, f"{name} control is not width-matched with the grown net: {diff}"


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
def _eeg_clf(model, **kwargs):
    """A plain braindecode ``EEGClassifier`` with the gromo growth callback.

    There is no eegrow wrapper: this is exactly how a user wires growth -- a stock
    ``EEGClassifier`` plus ``GromoGrowth`` in ``callbacks``. ``CrossEntropyLoss`` is
    explicit because our models output raw logits (not log-probabilities).
    """
    from braindecode import EEGClassifier

    from eegrow import GromoGrowth

    grow_every = kwargs.pop("grow_every", 4)
    verbose = kwargs.pop("verbose", False)
    return EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        callbacks=[("gromo", GromoGrowth(grow_every=grow_every, verbose=verbose))],
        train_split=None,
        device=DEVICE,
        **kwargs,
    )


def test_eegclassifier_grows_during_fit():
    """A growable model trained through braindecode's EEGClassifier grows during
    fit (callback fires), respects the target cap, and stays predictable."""
    import numpy as np

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
    clf = _eeg_clf(model, max_epochs=6, grow_every=4, batch_size=16)
    clf.fit(X, y)

    assert model.growable_width > w0, "EEGClassifier training did not grow the model"
    assert model.growable_width <= model.target_width, "single growth exceeded the cap"
    preds = clf.predict(X)
    assert preds.shape == (N,)
    assert set(np.unique(preds)).issubset(set(range(N_CLASSES)))


def test_eegclassifier_frozen_model_is_noop():
    """With no target width the growth callback is a no-op (plain baseline)."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")

    torch.manual_seed(0)
    model = GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=8, n_spatial_filters_smooth=8, device=DEVICE,
    )
    clf = _eeg_clf(model, max_epochs=4, grow_every=2, batch_size=16)
    clf.fit(X, y)
    assert model.growable_width == 8, "frozen model should not grow"


class _StubNet:
    """Minimal skorch-``NeuralNet`` surface the ``GromoGrowth`` callback touches.

    Lets us drive ``on_epoch_end`` over a fixed batch list, without a real fit loop,
    so the callback's growth can be compared to a direct ``loop.grow_step``.
    """

    def __init__(self, module, batches, *, epoch, max_epochs):
        self.module_ = module
        self._batches = batches
        self.history = list(range(epoch))  # len(history) == current epoch
        self.max_epochs = max_epochs
        self.optimizer_ = torch.optim.AdamW(module.parameters(), lr=1e-3)

    def get_iterator(self, dataset_train, training=True):
        return list(self._batches)

    def initialize_optimizer(self):
        self.optimizer_ = torch.optim.AdamW(self.module_.parameters(), lr=1e-3)


def _fresh_sccnet():
    torch.manual_seed(0)
    return GrowingSCCNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, sfreq=SFREQ,
        n_spatial_filters=4, n_spatial_filters_smooth=8,
        target_n_spatial_filters=12, device=DEVICE,
    )


def test_callback_growth_matches_pure_gromo():
    """Integration: the EEGClassifier callback delivers the *same* growth as a direct
    gromo ``loop.grow_step``.

    Two identically-seeded models grow on the same batches: one through the
    ``GromoGrowth`` callback (skorch path), one through ``loop.grow_step`` (pure gromo,
    reproducing the callback's held-out split). The grown widths and every parameter
    must match -- the skorch wiring adds no divergence on top of gromo."""
    from eegrow import GromoGrowth

    g = torch.Generator().manual_seed(1)
    batches = [(torch.randn(8, C, T, generator=g), torch.randint(0, N_CLASSES, (8,),
                generator=g)) for _ in range(5)]

    cb = GromoGrowth(grow_every=1, verbose=False)

    # --- pure gromo path: replicate the callback's cap + held-out split exactly ---
    pure = _fresh_sccnet()
    max_added = max(1, pure.target_width - pure.growable_width)
    stats_loader, val_loader = cb._split_holdout(batches)
    loop.grow_step(pure, stats_loader, DEVICE, val_loader=val_loader,
                   max_added=max_added)

    # --- skorch callback path: same batches, driven on a stub net ---
    via_cb = _fresh_sccnet()
    net = _StubNet(via_cb, batches, epoch=1, max_epochs=2)
    cb.on_epoch_end(net, dataset_train=object())

    assert pure.growable_width == via_cb.growable_width > 4, "callback did not grow"
    pure_params = dict(pure.named_parameters())
    for name, p in via_cb.named_parameters():
        assert name in pure_params, f"param {name} missing from pure-gromo model"
        assert torch.allclose(p, pure_params[name], atol=1e-6), (
            f"param {name} differs between callback and pure-gromo growth")


def test_callback_skips_growth_on_empty_iterator():
    """Regression: an empty training iterator must skip growth, not crash.

    ``EEGClassifier`` sets ``iterator_train__drop_last=True``; on a CV fold smaller
    than ``batch_size`` the iterator yields *no* batch. Growth statistics can't be
    estimated on zero data, so the callback must return cleanly (leaving the model
    untouched and on its original device) rather than let gromo raise inside ``fit``.
    """
    from eegrow import GromoGrowth

    model = _fresh_sccnet()
    width_before = model.growable_width
    device_before = next(model.parameters()).device

    cb = GromoGrowth(grow_every=1, verbose=False)
    net = _StubNet(model, batches=[], epoch=1, max_epochs=2)  # empty iterator

    cb.on_epoch_end(net, dataset_train=object())  # must not raise

    assert model.growable_width == width_before, "width changed despite no data"
    assert next(model.parameters()).device == device_before, "device not restored"
    assert not getattr(cb, "done_", False), "growth wrongly marked done on empty epoch"


# --------------------------------------------------- s=0 abstention (dead neurons)
def test_grow_step_abstains_instead_of_adding_dead_neurons(monkeypatch):
    """When the line search picks s=0, the update is discarded, not applied.

    gromo couples the scaling factor to every part of the change: at s=0 the optimal
    delta is scaled by 0 and the new neurons are spliced in with identically zero
    weights. A zero-in/zero-out neuron sits at an exact stationary point -- zero
    activation, zero gradient -- so it stays zero for the rest of the fit while still
    counting toward ``growable_width`` and ``n_params``. Applying that change is
    strictly worse than skipping it: it burns width budget and inflates the parameter
    axis the efficiency claim is measured on, and buys nothing.

    Forced rather than hoped for: the line search is made to prefer s=0 outright, so
    the test pins the behaviour instead of depending on which way random data falls.
    """
    model = _fresh_sccnet()
    w0, p0 = model.growable_width, sum(p.numel() for p in model.parameters())

    # Loss strictly increasing in s => 0.0 wins on merit, not on the tie-break.
    def fake_evaluate(mdl, loader, crit, use_extended_model=True, device=None):
        return 1.0 + float(mdl._growable_layers[0].scaling_factor.item()), None

    monkeypatch.setattr(loop, "evaluate_model", fake_evaluate)

    result = loop.grow_step(model, _loader(), DEVICE)

    assert result["s"] == 0.0
    assert result["applied"] is False
    assert model.growable_width == w0, "abstained yet the width grew"
    assert sum(p.numel() for p in model.parameters()) == p0, (
        "abstained yet parameters were added -- these would be dead weights")
    assert torch.isfinite(model(next(iter(_loader()))[0])).all()


def test_callback_keeps_growing_after_an_abstention(monkeypatch):
    """An abstention must not be read as "target reached".

    ``GromoGrowth`` stops for good when a step leaves the width unchanged, which was
    the right reading when the only way that happened was hitting the cap. An
    abstention leaves the width unchanged too -- and if it latched ``done_`` the model
    would freeze at its seed width after a single unlucky epoch, never to grow again.
    """
    from eegrow import GromoGrowth

    model = _fresh_sccnet()
    cb = GromoGrowth(grow_every=1, verbose=False)

    def fake_evaluate(mdl, loader, crit, use_extended_model=True, device=None):
        return 1.0 + float(mdl._growable_layers[0].scaling_factor.item()), None

    monkeypatch.setattr(loop, "evaluate_model", fake_evaluate)

    g = torch.Generator().manual_seed(1)
    batches = [(torch.randn(8, C, T, generator=g),
                torch.randint(0, N_CLASSES, (8,), generator=g)) for _ in range(5)]
    net = _StubNet(model, batches, epoch=1, max_epochs=10)
    cb.on_epoch_end(net, dataset_train=object())

    assert model.growable_width == 4, "abstention should leave the width alone"
    assert not cb.done_, "an abstention was mistaken for reaching the target width"


# ------------------------------------------------ how many neurons a step proposes
def test_relative_floor_beats_gromos_absolute_threshold():
    """The neuron count must not hinge on where our gradients sit versus 1e-3.

    gromo keeps candidates with ``s >= min(statistical_threshold, s.max())``, an
    absolute cut defaulting to 1e-3. Half our arms have a singular-value spectrum
    entirely below it, so the expression collapses to ``s >= s.max()`` and exactly one
    neuron survives -- which is why ``grow_shallow`` needed 160 epochs to reach a
    target its fits never had time for. The relative floor has to keep strictly more
    than that on such a spectrum, and must still respect the width cap.
    """
    model = GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        n_filters_time=4, n_filters_spat=8, target_n_filters_time=16, device=DEVICE,
    )
    loader = _loader()

    strict = loop.grow_step(model, loader, DEVICE, min_singular_ratio=1.0)
    assert strict["n_candidates"] == 1, "ratio 1.0 should reproduce one-neuron steps"
    assert strict["n_proposed"] > 1, (
        "nothing to test: gromo proposed a single candidate before any thresholding")

    model = GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        n_filters_time=4, n_filters_spat=8, target_n_filters_time=16, device=DEVICE,
    )
    loose = loop.grow_step(model, loader, DEVICE, min_singular_ratio=0.1)
    assert loose["n_candidates"] > strict["n_candidates"], (
        "the relative floor kept no more neurons than the one-per-step default")
    assert model.growable_width <= 16, "the relative floor overshot the target width"


def test_grow_step_caps_itself_at_target_width():
    """``grow_step`` must respect ``target_width`` even when called without a cap.

    The cap used to live only in the ``GromoGrowth`` callback, which was safe while a
    step added one neuron at a time -- it could not overshoot in a single go. Now that
    a step can propose eighteen, any direct caller (``loop.run_model``, the tests) has
    to be protected by ``grow_step`` itself.
    """
    model = GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T,
        n_filters_time=4, n_filters_spat=8, target_n_filters_time=6, device=DEVICE,
    )
    loop.grow_step(model, _loader(), DEVICE)  # no max_added passed on purpose
    assert model.growable_width <= 6, (
        f"width {model.growable_width} overshot target_width=6")
