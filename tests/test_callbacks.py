"""Tests for ``eegrow.training.callbacks``.

The one that matters is :class:`RestoreBestModel`. Restoring the best epoch's weights
is a one-line change on a fixed network -- skorch has shipped ``load_best`` for years
-- and it is *impossible* that way on a growing one: the module's tensors change shape
between the best epoch and the last, so the strict ``load_state_dict`` behind skorch's
implementation raises ``size mismatch``. That is asserted here directly, because it is
the entire reason this module exists and a reader who does not know it will delete the
custom callback in favour of the built-in flag.

The window where this bites is not an edge case: early stopping ends a fit exactly
``patience`` epochs after its best, and growth keeps firing throughout that window.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from sklearn.base import clone

from eegrow import GromoGrowth, GrowingShallowFBCSPNet
from eegrow.training.callbacks import (AdamEpsDominance, GradientNorm,
                                      RestoreBestModel, StopReason)

C, T, N_CLASSES, N = 8, 256, 4, 64
DEVICE = "cpu"


def _xy(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")
    return x, y


def _model(cap: int = 12):
    torch.manual_seed(0)
    return GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=4,
        n_filters_spat=8, target_n_filters_time=cap, device=DEVICE)


def _clf(model, callbacks, max_epochs=8):
    from braindecode import EEGClassifier
    from skorch.dataset import ValidSplit

    return EEGClassifier(
        model, criterion=torch.nn.CrossEntropyLoss, max_epochs=max_epochs,
        batch_size=16, iterator_train__drop_last=False,
        train_split=ValidSplit(0.25, random_state=0, stratified=True),
        callbacks=callbacks, device=DEVICE, verbose=0,
        classes=list(range(N_CLASSES)),
    )


# ------------------------------------------------------------- the shape problem
def test_skorch_load_best_would_crash_on_a_grown_module():
    """Why ``EarlyStopping(load_best=True)`` is not the fix.

    Pins the mechanism rather than the sentiment: a state_dict captured before growth
    cannot be loaded back afterwards. If a future gromo made growth shape-preserving
    this test would fail, which is exactly when someone should reconsider the custom
    callback.
    """
    from eegrow.training import loop

    model = _model()
    snapshot = copy.deepcopy(model.state_dict())  # what skorch's load_best keeps

    g = torch.Generator().manual_seed(1)
    batches = [(torch.randn(16, C, T, generator=g),
                torch.randint(0, N_CLASSES, (16,), generator=g)) for _ in range(4)]
    result = loop.grow_step(model, batches, DEVICE, min_singular_ratio=0.1)
    assert result["applied"], "test needs a step that actually grew"

    try:
        model.load_state_dict(snapshot)  # what skorch's load_best does
    except RuntimeError as exc:
        assert "size mismatch" in str(exc)
    else:
        raise AssertionError(
            "load_state_dict succeeded across a growth event -- skorch's load_best "
            "would now work and RestoreBestModel may be redundant")


# ------------------------------------------------------------- RestoreBestModel
def test_restore_gives_back_the_best_epoch_across_growth():
    """The fitted net carries the best epoch's module, shapes and all."""
    restore = RestoreBestModel(monitor="valid_loss", lower_is_better=True)
    net = _clf(_model(), [("gromo", GromoGrowth(grow_every=2, verbose=False)),
                          ("restore", restore)])
    x, y = _xy()
    net.fit(x, y)

    losses = [row["valid_loss"] for row in net.history]
    best_epoch = int(np.argmin(losses)) + 1
    last_epoch = len(losses)

    if best_epoch == last_epoch:
        assert net.restored_epoch_ is None
        return
    assert net.restored_epoch_ == best_epoch, (
        f"restored epoch {net.restored_epoch_}, best was {best_epoch}")
    # The restored module must be the width the model had AT that epoch, which after
    # growth is narrower than the final one -- the shape travelled with the weights.
    width_at_best = net.history[best_epoch - 1]["width"]
    assert net.module_.growable_width == width_at_best
    assert torch.isfinite(torch.as_tensor(net.predict_proba(x))).all()


def test_restore_is_a_no_op_when_the_last_epoch_is_the_best():
    """Nothing to restore => the net is untouched and says so."""
    restore = RestoreBestModel(monitor="valid_loss", lower_is_better=True)
    net = _clf(_model(cap=4), [("restore", restore)], max_epochs=3)  # frozen width
    x, y = _xy()
    net.fit(x, y)

    losses = [row["valid_loss"] for row in net.history]
    if int(np.argmin(losses)) + 1 == len(losses):
        assert net.restored_epoch_ is None
    else:
        assert net.restored_epoch_ == int(np.argmin(losses)) + 1


def test_restore_ignores_non_finite_scores():
    """A diverged epoch reports NaN; it must never be selected as the best model."""
    restore = RestoreBestModel(monitor="valid_loss", lower_is_better=True)

    class _Net:
        def __init__(self):
            self.history = []
            self.module_ = torch.nn.Linear(2, 2)

    net = _Net()
    restore.on_train_begin(net)
    for epoch, loss in enumerate([1.0, float("nan"), 0.5, float("inf")], start=1):
        net.history.append({"epoch": epoch, "valid_loss": loss})
        restore.on_epoch_end(net)

    assert restore.best_epoch_ == 3, "NaN/inf epochs must not win the selection"


def test_restore_survives_sklearn_clone():
    """MOABB clones the pipeline per fold; the callback must round-trip."""
    net = _clf(_model(), [("restore", RestoreBestModel(monitor="valid_acc",
                                                       lower_is_better=False))])
    cb = dict(clone(net).callbacks)["restore"]
    assert cb.monitor == "valid_acc"
    assert cb.lower_is_better is False


# ------------------------------------------------------------------ instrumentation
def test_gradient_norm_and_stop_reason_are_recorded():
    """Stella's two asks land in the history where FitRecorder can pick them up."""
    net = _clf(_model(cap=4), [("grad", GradientNorm()), ("stop", StopReason())],
               max_epochs=3)
    x, y = _xy()
    net.fit(x, y)

    for row in net.history:
        assert row["grad_norm"] is not None and np.isfinite(row["grad_norm"])
        assert row["grad_norm_max"] >= row["grad_norm"]
        assert row["lr"] > 0
    # Ran the full budget with no EarlyStopping in the list.
    assert net.stop_reason_ == "budget"


def test_growth_events_are_recorded_with_their_scaling_factor():
    """The winning ``s`` was chosen and discarded; now it survives the fit.

    Without it no campaign can say whether a growth event bought anything -- an event
    at s=0 used to add neurons that were dead on arrival and looked identical, in
    every recorded field, to one that worked.
    """
    net = _clf(_model(), [("gromo", GromoGrowth(grow_every=2, verbose=False))])
    x, y = _xy()
    net.fit(x, y)

    events = [row for row in net.history if "grow_s" in row]
    assert events, "no growth opportunity was recorded"
    for row in events:
        assert row["grow_s"] in (0.0, 0.1, 0.5, 1.0)
        assert row["grow_applied"] == (row["grow_s"] != 0.0)
        assert row["grow_n_proposed"] >= row["grow_n_kept"] >= 1


# ------------------------------------------------------- the AdamW eps hypothesis
def test_attenuation_matches_the_closed_form_on_a_known_state():
    """``a = sqrt(v_hat)/(sqrt(v_hat)+eps)`` against a hand-built optimizer state.

    Pinned to the arithmetic rather than to a fitted net, because the whole value of
    this metric is that it is the *exact* ratio of the step AdamW takes to the step it
    would take with ``eps=0`` -- an approximation of it would answer a different
    question than the one asked. Three coordinates, one on each side of the ``eps``
    boundary and one exactly on it.
    """
    eps, beta2 = 1e-8, 0.999
    step = 100.0
    bc2 = 1.0 - beta2 ** step
    # Target sqrt(v_hat) values: far above eps, exactly eps, far below it.
    root_v_hat = torch.tensor([1e-4, 1e-8, 1e-12], dtype=torch.float64)
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    opt = torch.optim.AdamW([p], eps=eps, betas=(0.9, beta2))
    opt.state[p] = {"step": torch.tensor(step),
                    "exp_avg": torch.zeros(3, dtype=torch.float64),
                    "exp_avg_sq": (root_v_hat ** 2) * bc2}

    stats = AdamEpsDominance._measure(opt)
    expected = root_v_hat / (root_v_hat + eps)          # [~1, 0.5, ~1e-4]
    assert stats["adam_atten_mean"] == pytest.approx(float(expected.mean()), rel=1e-5)
    # `a < 0.5` is `sqrt(v_hat) < eps`: the coordinate sitting exactly ON the boundary
    # is not counted, so this is 1 of 3 and not 2 of 3.
    assert stats["adam_eps_frac"] == pytest.approx(1 / 3)


def test_a_healthy_fit_is_not_reported_as_eps_dominated():
    """The negative control, without which a confirmation could not be believed.

    If this metric read "eps dominates" on an ordinary net that is training fine, then
    reading it on a diverging fold would prove nothing. So: a net that learns must come
    back with attenuation near 1 and essentially no dominated coordinates.
    """
    net = _clf(_model(), [("eps", AdamEpsDominance())], max_epochs=6)
    net.set_params(optimizer=torch.optim.AdamW, optimizer__eps=1e-8)
    x, y = _xy()
    net.fit(x, y)

    for row in net.history[1:]:  # epoch 1 has one step: v_hat is still a single sample
        assert 0.0 < row["adam_atten_mean"] <= 1.0
        assert row["adam_atten_mean"] > 0.9, "a healthy AdamW is not eps-limited"
        assert row["adam_eps_frac"] < 0.05
        # p05 is NOT bounded above by the mean, and that it is not is the whole reason
        # both are recorded: the distribution is strongly left-skewed -- most
        # coordinates sit at ~1 while a thin tail drags the mean *below* the 5th
        # percentile -- so either statistic alone misrepresents the fit.
        assert 0.0 <= row["adam_atten_p05"] <= 1.0


def test_raising_eps_moves_the_metric_in_the_predicted_direction():
    """The metric responds to the knob, so a null result is informative.

    Same fit, same seed, ``eps`` raised by four orders of magnitude. If attenuation did
    not drop, the number would be inert and "no eps problem" would be unfalsifiable.
    """
    def atten(eps):
        torch.manual_seed(0)
        net = _clf(_model(), [("eps", AdamEpsDominance())], max_epochs=5)
        net.set_params(optimizer=torch.optim.AdamW, optimizer__eps=eps)
        net.fit(*_xy())
        return net.history[-1]["adam_atten_mean"]

    assert atten(1e-4) < atten(1e-8)
