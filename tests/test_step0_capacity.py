"""The capacity criterion detects a net that cannot descend, and passes one that can."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import step0_capacity as cap  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "benchmarks" / "config"


def _model(arm):
    return yaml.safe_load((CFG / "model" / f"{arm}.yaml").read_text())


def _train(**over):
    t = yaml.safe_load((CFG / "config.yaml").read_text())["train"]
    t.update({"lr_schedule": None, **over})
    return t


def _data(n_per_class=8, C=4, T=500, seed=0):
    """Class-coded oscillations: learnable, so a working net must memorise them."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 250.0
    X, y = [], []
    for c in range(4):
        for _ in range(n_per_class):
            sig = np.sin(2 * np.pi * (8 + 4 * c) * t)[None] * rng.uniform(1, 2, (C, 1))
            X.append(sig + 0.3 * rng.standard_normal((C, T)))
            y.append(c)
    return np.asarray(X, "float32"), np.asarray(y, "int64")


def test_balanced_subset_is_balanced_and_reproducible():
    y = np.repeat(np.arange(4), 72)
    a, b = cap.balanced_subset(y, 8, 0), cap.balanced_subset(y, 8, 0)
    assert np.array_equal(a, b) and len(a) == 32
    assert np.bincount(y[a]).tolist() == [8, 8, 8, 8]
    assert not np.array_equal(a, cap.balanced_subset(y, 8, 1))


def test_balanced_subset_refuses_a_short_class():
    with pytest.raises(ValueError):
        cap.balanced_subset(np.array([0] * 10 + [1] * 3), 8, 0)


def test_a_net_that_can_descend_passes():
    X, y = _data()
    res = cap.capacity_fit(_model("bd_shallow"), _train(), X, y, seed=0)
    assert res["passes"], res["final_loss_eval"]
    assert len(res["train_loss_curve"]) == cap.MAX_EPOCHS


def test_a_net_that_cannot_descend_is_refused():
    """lr = 0: the optimizer never moves, the loss stays at its init value."""
    X, y = _data()
    res = cap.capacity_fit(_model("bd_shallow"), _train(lr=0.0), X, y, seed=0,
                           max_epochs=5)
    assert not res["passes"] and res["final_loss_eval"] > 0.5


def test_growing_arms_are_refused_not_stood_in_for():
    X, y = _data()
    with pytest.raises(NotImplementedError):
        cap.capacity_fit(_model("grow_shallow"), _train(), X, y, seed=0)


def test_a_schedule_is_refused():
    X, y = _data()
    with pytest.raises(ValueError):
        cap.capacity_fit(_model("bd_shallow"), _train(lr_schedule="cosine"), X, y,
                         seed=0)
