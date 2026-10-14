"""The 0.4 cropped-training helpers, ported for the published-reference criterion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import step0_reference as ref  # noqa: E402


@pytest.mark.parametrize("n_preds", [479, 523, 126])
def test_crops_predict_every_sample_after_the_receptive_field(n_preds):
    blocks = ref.start_stop_blocks(1125, ref.INPUT_TIME_LENGTH, n_preds)
    n_rf = ref.INPUT_TIME_LENGTH - n_preds + 1
    predicted = np.unique(np.concatenate([np.arange(b - n_preds, b) for _, b in blocks]))
    assert np.array_equal(predicted, np.arange(n_rf - 1, 1125))
    assert blocks[0][0] == 0 and blocks[-1][1] == 1125
    assert all(b - a == ref.INPUT_TIME_LENGTH for a, b in blocks)


def test_trial_predictions_drop_the_overlap():
    n_preds, n_trials = 479, 3
    blocks = ref.start_stop_blocks(1125, ref.INPUT_TIME_LENGTH, n_preds)
    crop = [np.zeros((4, n_preds)) for _ in range(n_trials * len(blocks))]
    out = ref.trial_predictions(crop, 1125, n_preds, n_trials)
    n_rf = ref.INPUT_TIME_LENGTH - n_preds + 1
    assert [p.shape[1] for p in out] == [1125 - n_rf + 1] * n_trials


def test_balanced_batches_partition():
    rng = np.random.RandomState(0)
    bs = ref.balanced_batches(461, 60, rng, shuffle=True)
    assert sorted(np.concatenate(bs).tolist()) == list(range(461))
    assert len(bs) == 8 and max(map(len, bs)) - min(map(len, bs)) <= 1


def test_standardize_matches_braindecode_outside_the_init_block():
    """Same EWM as braindecode 1.7; only the init block convention differs."""
    from braindecode.preprocessing import exponential_moving_standardize

    x = np.random.default_rng(0).standard_normal((3000, 4)) * 5 + 2
    old = ref.exponential_running_standardize_04(x)
    new = exponential_moving_standardize(x.T, init_block_size=1000).T
    assert np.allclose(old[1000:], new[1000:], atol=1e-8)


def test_max_norm_caps_every_conv_weight():
    import torch

    m = ref.build_model("shallow", 22, 4)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=10.0)
    ref.max_norm_(m)
    convs = [x for x in m.modules() if isinstance(x, (torch.nn.Conv2d, torch.nn.Linear))]
    for c in convs[:-1]:
        assert c.weight.flatten(1).norm(dim=1).max() <= 2 + 1e-5
    assert convs[-1].weight.flatten(1).norm(dim=1).max() <= 0.5 + 1e-5
