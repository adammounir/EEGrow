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

import numpy as np
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


def test_growth_epochs_record_the_decision_not_only_its_outcome(tmp_path):
    """A growth epoch must carry the evidence for the choice it made.

    ``grow_s`` alone says which scaling factor won, never why. The line search over
    ``loop.SCALING_GRID`` and the singular spectrum are computed inside ``grow_step``
    and destroyed by ``apply_change``/``delete_update`` on the very next line, so a
    campaign that does not write them here cannot recover them at any price -- which
    for the final grid is 817 GPU-hours. This test is the guard on that.
    """
    from eegrow.training.loop import SCALING_GRID

    path = tmp_path / "fits.jsonl"
    x, y = _xy()
    _clf(_model(), path).fit(x, y)
    rec = json.loads(path.read_text().strip())

    grown = [h for h in rec["history"] if "grow_s" in h]
    assert grown, "no growth epoch in a fit with grow_every=2 and max_epochs=6"
    for h in grown:
        # The line search, keyed by scaling factor. JSON has no float keys, so the
        # grid comes back stringified -- that is the storage format, not a loss.
        losses = h["grow_losses"]
        assert set(losses) == {str(float(s)) for s in SCALING_GRID}, losses
        best = min(losses, key=losses.get)
        assert float(best) == h["grow_s"], f"s={h['grow_s']} does not minimise {losses}"
        assert h["grow_select_loss"] == losses[best]

        # Both spectra, bracketing the selection rule. `eigenvalues_extension` is set
        # by `compute_optimal_updates`, so unlike `first_order_improvement` it is
        # always available and a None here is a real regression.
        proposed, kept = h["grow_eig_proposed"], h["grow_eig_kept"]
        assert isinstance(proposed, list) and proposed, proposed
        assert isinstance(kept, list) and kept, kept
        assert len(kept) <= len(proposed), "selection cannot add candidates"
        assert h["grow_n_proposed"] == len(proposed)
        assert h["grow_n_kept"] == len(kept)
        # Descending singular values: the "keep the first k" cut in
        # `_n_candidates_to_keep` is only "keep the best k" if this holds.
        assert proposed == sorted(proposed, reverse=True), proposed


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


def test_record_stamps_the_selection_rule_that_ran(tmp_path):
    """The launch overrides must be readable off the record, not the config.

    A grid launched without ``smooth_window=5`` / ``selection_monitor=valid_acc`` /
    ``patience=200`` writes well-formed records; only this stamp tells them apart.
    """
    from braindecode import EEGClassifier
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit

    from eegrow.training.callbacks import RestoreBestModel

    path = tmp_path / "fits.jsonl"
    net = EEGClassifier(
        _model(), criterion=torch.nn.CrossEntropyLoss, max_epochs=3, batch_size=16,
        callbacks=[("restore", RestoreBestModel(monitor="valid_acc",
                                                lower_is_better=False,
                                                tie_break="latest",
                                                smooth_window=5)),
                   ("record", FitRecorder(path)),
                   ("stop", EarlyStopping(monitor="valid_loss", patience=200))],
        train_split=ValidSplit(0.25, random_state=0, stratified=True), device=DEVICE,
        classes=list(range(N_CLASSES)), verbose=0,
    )
    x, y = _xy()
    clone(net).fit(x, y)  # through clone, as MOABB does per fold

    rec = json.loads(path.read_text().strip().splitlines()[-1])
    assert rec["selection"] == {"selection_monitor": "valid_acc", "lower_is_better": False,
                                "tie_break": "latest", "smooth_window": 5,
                                "swap": True, "stop_monitor": "valid_loss",
                                "patience": 200}


def test_record_says_when_no_selection_callback_ran(tmp_path):
    """Absent callbacks stamp ``None`` rather than a default that looks like a choice."""
    path = tmp_path / "fits.jsonl"
    _clf(_model(), path).fit(*_xy())
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    assert set(rec["selection"].values()) == {None}


def _scored_last_clf(path, save_selected=True, max_epochs=8):
    """The deep campaign's protocol: select (smoothed) but score the LAST epoch."""
    from braindecode import EEGClassifier
    from skorch.dataset import ValidSplit

    from eegrow.training.callbacks import GradientNorm, RestoreBestModel

    return EEGClassifier(
        _model(), criterion=torch.nn.CrossEntropyLoss, max_epochs=max_epochs,
        batch_size=8,
        callbacks=[("grad", GradientNorm()),
                   ("gromo", GromoGrowth(grow_every=2, verbose=False)),
                   ("restore", RestoreBestModel(monitor="valid_acc", lower_is_better=False,
                                                smooth_window=3, swap=False)),
                   ("record", FitRecorder(path, meta={"subject": 3},
                                          save_selected=save_selected))],
        train_split=ValidSplit(0.25, random_state=0, stratified=True), device=DEVICE,
        classes=list(range(N_CLASSES)), verbose=0,
    )


def test_scored_last_epoch_keeps_the_last_module_and_saves_the_selected(tmp_path):
    """swap=False: the net keeps its last epoch; the selected one goes to disk.

    The saved weights must be the module AT the selected epoch -- width included,
    since growth continues after it -- and the fingerprint must name the fold.
    """
    path = tmp_path / "cell__fits.jsonl"
    net = _scored_last_clf(path)
    x, y = _xy()
    net.fit(x, y)
    rec = json.loads(path.read_text().strip().splitlines()[-1])

    assert net.restored_epoch_ is None and rec["restored_epoch"] is None
    assert rec["selection"]["swap"] is False
    assert rec["selected_epoch"] == net.selected_epoch_ is not None
    assert net.selected_state_ is None, "the recorder must release the copy"
    # The net carries its last epoch, i.e. the final width.
    assert net.module_.growable_width == rec["history"][-1]["width"]

    saved = torch.load(tmp_path / "cell__fits__selected" / rec["selected_path"],
                       weights_only=False)
    assert saved["epoch"] == rec["selected_epoch"]
    assert saved["growable_width"] == rec["history"][rec["selected_epoch"] - 1]["width"]
    assert saved["meta"] == {"subject": 3}
    import hashlib
    h = hashlib.sha1(x.tobytes()); h.update(y.tobytes())
    assert saved["train_fingerprint"] == rec["train_fingerprint"] == h.hexdigest()
    # Loadable into a model of that width: the re-score path actually works.
    width = saved["growable_width"]
    rebuilt = GrowingShallowFBCSPNet(
        n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=width,
        n_filters_spat=8, target_n_filters_time=10, device=DEVICE)
    rebuilt.load_state_dict(saved["state_dict"])


def test_selected_weights_are_not_saved_unless_asked(tmp_path):
    path = tmp_path / "cell__fits.jsonl"
    _scored_last_clf(path, save_selected=False).fit(*_xy())
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    assert rec["selected_path"] is None and rec["selected_epoch"] is not None
    assert not (tmp_path / "cell__fits__selected").exists()


def test_record_carries_the_spike_and_device_fields(tmp_path):
    """The step-0 spike refusal reads grad_norm_max_fit; the card is named per fit."""
    path = tmp_path / "cell__fits.jsonl"
    _scored_last_clf(path, save_selected=False).fit(*_xy())
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    curve = [h["grad_norm_max"] for h in rec["history"]]
    assert rec["grad_norm_max_fit"] == max(curve)
    assert rec["history"][rec["grad_norm_max_epoch"] - 1]["grad_norm_max"] == max(curve)
    assert rec["device"] == "cpu" and rec["gpu_uuid"] is None
    # Untrained 4-class net: the first batch's loss sits near ln(4) = 1.386.
    assert abs(rec["first_batch_loss"] - np.log(N_CLASSES)) < 0.5
    assert len(rec["fit_uid"]) == 32


def test_subject_stamp_keeps_every_recorder_flag(tmp_path):
    """The per-fold rebuild in subject_stamp must not drop save_selected."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from sklearn.pipeline import Pipeline
    from subject_stamp import with_stamp

    pipe = Pipeline([("clf", _scored_last_clf(tmp_path / "f.jsonl"))])
    stamped = with_stamp(pipe, {"subject": 7, "session": "0", "cv_ind": 2})
    rec = dict(stamped.named_steps["clf"].callbacks)["record"]
    assert rec.save_selected is True
    assert rec.meta == {"subject": 7, "session": "0", "cv_ind": 2}
