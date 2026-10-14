"""Tests for the growth held-out (``HoldoutValidSplit``) and the line search reading it.

The point of the held-out is that the growth amplitude is chosen on trials the net
never trained on and the selection rule never looks at. Each of those three
properties -- untouched validation side, disjointness, actually used by the line
search -- fails silently if broken, so each is asserted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from skorch.dataset import Dataset, ValidSplit

from eegrow import FitRecorder, GromoGrowth, GrowingShallowFBCSPNet
from eegrow.training.loop import WIDE_SCALING_GRID
from eegrow.training.splits import HoldoutValidSplit

C, T, N_CLASSES = 8, 256, 4


def _data(n=230, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, C, T)).astype("float32")
    y = np.tile(np.arange(N_CLASSES), n // N_CLASSES + 1)[:n].astype("int64")
    rng.shuffle(y)
    return X, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_validation_side_is_the_historical_one(seed):
    """n_valid and the valid trials are bit-identical to ValidSplit(0.2) -- the floor
    and the restored_epoch measure must not move because growth got a held-out."""
    X, y = _data()
    ds = Dataset(X, y)
    _, valid = HoldoutValidSplit(0.2, stratified=True, random_state=seed,
                                 holdout=0.1)(ds, y)
    _, valid0 = ValidSplit(0.2, stratified=True, random_state=seed)(ds, y)
    assert np.array_equal(valid.indices, valid0.indices)


def test_three_sides_are_disjoint_and_cover_the_fold():
    X, y = _data()
    split = HoldoutValidSplit(0.2, stratified=True, random_state=0, holdout=0.1)
    train, valid = split(Dataset(X, y), y)
    tr, va, ho = set(train.indices), set(valid.indices), set(split.holdout_.indices)
    assert not (tr & va) and not (tr & ho) and not (va & ho)
    assert tr | va | ho == set(range(len(y)))
    assert split.n_holdout_ == round(0.1 * len(y)) == 23
    assert (split.n_train_, split.n_valid_) == (len(tr), len(va))
    # stratified: every class present in the held-out
    assert set(y[list(ho)]) == set(range(N_CLASSES))


def test_zero_holdout_is_exactly_validsplit():
    X, y = _data()
    ds = Dataset(X, y)
    split = HoldoutValidSplit(0.2, stratified=True, random_state=0, holdout=0.0)
    train, valid = split(ds, y)
    train0, valid0 = ValidSplit(0.2, stratified=True, random_state=0)(ds, y)
    assert np.array_equal(train.indices, train0.indices)
    assert split.holdout_ is None and split.n_holdout_ == 0


def test_a_held_out_too_small_for_every_class_is_refused():
    X, y = _data(n=20)
    with pytest.raises(ValueError, match="cannot hold all"):
        HoldoutValidSplit(0.2, stratified=True, random_state=0, holdout=0.1)(
            Dataset(X, y), y)


def test_bad_grid_is_refused_at_construction():
    with pytest.raises(ValueError):
        GromoGrowth(scaling_grid=(0.1, 1.0))          # no abstention
    with pytest.raises(ValueError):
        GromoGrowth(scaling_grid=(0.0, 2.0, 1.0))     # not increasing


def _fit(tmp_path, holdout, grid=None, max_epochs=6):
    from braindecode import EEGClassifier

    torch.manual_seed(0)
    model = GrowingShallowFBCSPNet(n_chans=C, n_outputs=N_CLASSES, n_times=T,
                                   n_filters_time=4, n_filters_spat=8,
                                   target_n_filters_time=40, device="cpu")
    path = tmp_path / "fits.jsonl"
    net = EEGClassifier(
        model, criterion=torch.nn.CrossEntropyLoss, max_epochs=max_epochs,
        batch_size=32, iterator_train__drop_last=False,
        train_split=HoldoutValidSplit(0.2, stratified=True, random_state=0,
                                      holdout=holdout),
        callbacks=[("gromo", GromoGrowth(grow_every=2, scaling_grid=grid,
                                         verbose=False)),
                   ("record", FitRecorder(path))],
        device="cpu", classes=list(range(N_CLASSES)), verbose=0)
    X, y = _data()
    net.fit(X, y)
    return net, json.loads(path.read_text().strip().splitlines()[-1])


def test_line_search_reads_the_disjoint_held_out(tmp_path):
    """Every growth event is scored on the 23 held-out trials, on the wide grid."""
    net, rec = _fit(tmp_path, holdout=0.1, grid=WIDE_SCALING_GRID)
    events = [h for h in rec["history"] if "grow_s" in h]
    assert events, "test needs growth events"
    for h in events:
        assert h["grow_holdout"] == "disjoint"
        assert h["grow_n_select"] == 23
        assert sorted(float(k) for k in h["grow_losses"]) == list(WIDE_SCALING_GRID)
    assert (rec["n_fit_train"], rec["n_valid"], rec["n_holdout"]) == (161, 46, 23)
    assert rec["n_fit_train"] + rec["n_valid"] + rec["n_holdout"] == rec["n_train"]


def test_historical_path_is_labelled_as_such(tmp_path):
    """Without a held-out the record says the line search saw training data."""
    _, rec = _fit(tmp_path, holdout=0.0)
    kinds = {h["grow_holdout"] for h in rec["history"] if "grow_s" in h}
    assert kinds <= {"train_tail", "none"} and kinds
    assert rec["n_holdout"] == 0


def test_a_missing_held_out_is_refused_not_degraded(tmp_path):
    """A split configured for a held-out that produced none must stop the fit."""

    class _Broken(HoldoutValidSplit):
        def __call__(self, dataset, y=None, groups=None):
            out = super().__call__(dataset, y=y, groups=groups)
            self.holdout_ = None
            return out

    from braindecode import EEGClassifier

    torch.manual_seed(0)
    model = GrowingShallowFBCSPNet(n_chans=C, n_outputs=N_CLASSES, n_times=T,
                                   n_filters_time=4, n_filters_spat=8,
                                   target_n_filters_time=40, device="cpu")
    net = EEGClassifier(
        model, criterion=torch.nn.CrossEntropyLoss, max_epochs=3, batch_size=32,
        train_split=_Broken(0.2, stratified=True, random_state=0, holdout=0.1),
        callbacks=[("gromo", GromoGrowth(grow_every=2, verbose=False))],
        device="cpu", classes=list(range(N_CLASSES)), verbose=0)
    with pytest.raises(RuntimeError, match="produced none"):
        net.fit(*_data())


def test_split_survives_sklearn_clone():
    from braindecode import EEGClassifier
    from sklearn.base import clone

    net = EEGClassifier(GrowingShallowFBCSPNet, module__n_chans=C,
                        module__n_outputs=N_CLASSES, module__n_times=T,
                        train_split=HoldoutValidSplit(0.2, stratified=True,
                                                      random_state=3, holdout=0.1),
                        callbacks=[("gromo", GromoGrowth(scaling_grid=WIDE_SCALING_GRID))])
    c = clone(net)
    assert c.train_split.holdout == 0.1 and c.train_split.random_state == 3
    assert tuple(dict(c.callbacks)["gromo"].scaling_grid) == WIDE_SCALING_GRID
