"""Tests for ``eegrow.training.recording.FitRecorder``.

The recorder is only useful if it survives being cloned: MOABB clones the whole
pipeline for every cross-validation fold, so a callback that breaks ``sklearn.clone``
breaks every cell of the benchmark and does so only on the cluster. Two distinct
contracts have to hold, and both were violated by the first version:

* skorch's ``Callback.get_params`` returns every attribute whose name does not end in
  ``_``, and ``clone`` passes them back as constructor keywords -- so internal state
  must end in ``_``;
* ``clone`` then checks the round-trip returns the *same objects*, so ``__init__``
  must store its arguments verbatim rather than normalising them.

Both are cheap to assert here and expensive to discover on a GPU node.
"""

from __future__ import annotations

import json

import torch
from sklearn.base import clone

from eegrow import FitRecorder, GromoGrowth, GrowingShallowFBCSPNet

C, T, N_CLASSES, N = 8, 256, 4, 32
DEVICE = "cpu"


def _xy(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, C, T, generator=g).numpy().astype("float32")
    y = torch.randint(0, N_CLASSES, (N,), generator=g).numpy().astype("int64")
    return x, y


def _clf(model, path):
    from braindecode import EEGClassifier

    return EEGClassifier(
        model, criterion=torch.nn.CrossEntropyLoss, max_epochs=6, batch_size=16,
        callbacks=[("gromo", GromoGrowth(grow_every=2, verbose=False)),
                   ("record", FitRecorder(path, meta={"model": "grow_shallow"}))],
        train_split=None, device=DEVICE,
    )


def _model(cap: int = 10):
    torch.manual_seed(0)
    return GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=4,
        n_filters_spat=8, target_n_filters_time=cap, device=DEVICE)


def test_recorder_survives_sklearn_clone(tmp_path):
    """`sklearn.clone` is what MOABB does per fold; it must round-trip."""
    net = _clf(_model(), tmp_path / "fits.jsonl")
    cloned = clone(net)
    rec = dict(cloned.callbacks)["record"]
    assert rec.path == tmp_path / "fits.jsonl"
    assert rec.meta == {"model": "grow_shallow"}


def test_recorder_writes_one_line_per_fit(tmp_path):
    """Two fits append two records to the same file, each self-describing."""
    path = tmp_path / "fits.jsonl"
    x, y = _xy()
    for _ in range(2):
        _clf(_model(), path).fit(x, y)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2, f"expected one record per fit, got {len(lines)}"
    for raw in lines:
        rec = json.loads(raw)
        assert rec["model"] == "grow_shallow"
        assert rec["width_start"] == 4
        assert rec["target_width"] == 10
        assert rec["width_end"] >= rec["width_start"]
        assert rec["params_end"] >= rec["params_start"]
        assert rec["epochs"] == len(rec["history"]) > 0
        # The trajectory is the whole point: width and capacity per epoch.
        first = rec["history"][0]
        assert "width" in first and "n_params" in first and "epoch" in first
        widths = [h["width"] for h in rec["history"]]
        assert widths == sorted(widths), f"width went backwards: {widths}"
        assert max(widths) <= rec["target_width"]


def test_recorder_on_a_frozen_model_records_a_constant_width(tmp_path):
    """A fixed arm gets a record too -- constant width is what makes it comparable."""
    path = tmp_path / "fits.jsonl"
    torch.manual_seed(0)
    frozen = GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=8,
        n_filters_spat=8, device=DEVICE)          # no target => frozen
    x, y = _xy()
    _clf(frozen, path).fit(x, y)

    rec = json.loads(path.read_text().strip())
    assert rec["width_start"] == rec["width_end"] == 8
    assert {h["width"] for h in rec["history"]} == {8}
    assert rec["params_start"] == rec["params_end"]
