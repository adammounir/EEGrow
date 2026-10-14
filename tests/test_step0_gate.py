"""Step-0 gates refuse a cell on any single deviation, and admit a correct one.

Each mutation below is a failure mode met on a previous campaign (or a launch override
that defaults to the losing historical value); each must turn the verdict to refused.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import step0_gate as g  # noqa: E402

LR, WD = 6.25e-4, 0.01


def _record(subject, session, cv_ind, growing=False):
    hist = [{"epoch": e + 1, "train_loss": 1.0, "lr": LR} for e in range(g.MAX_EPOCHS)]
    if growing:
        hist[2].update(grow_s=1.0, grow_holdout="disjoint")
        hist[4].update(grow_s=2.0, grow_holdout="disjoint")
    return {
        "model": "bd_deep4", "seed": 0, "subject": str(subject), "session": session,
        "cv_ind": cv_ind, "fit_uid": f"{subject}{session}{cv_ind}",
        "n_holdout": 23, "epochs": g.MAX_EPOCHS, "max_epochs": g.MAX_EPOCHS,
        "selected_path": "x.pt", "first_batch_loss": 1.9,
        "grad_norm_max_fit": 12.0, "gpu_uuid": "GPU-a",
        "optimizer": {"class": "AdamW", "lr": LR, "weight_decay": WD},
        "selection": dict(g.SELECTION), "history": hist,
    }


def _cell(growing=False):
    recs, i = [], 0
    for s in (3, 1, 2, 4, 5, 6, 7, 8, 9):          # MOABB's order varies
        for sess in ("1test", "0train"):
            for _ in range(g.N_FOLDS):
                recs.append(_record(s, sess, i, growing))
                i += 1
    return recs


def _gate(recs, growing=False):
    return g.gate_cell(recs, dataset="bnci2014_001", lr=LR, weight_decay=WD,
                       growing=growing)


def test_a_correct_cell_is_admitted():
    v = _gate(_cell())
    assert not v["refused"], v["reasons"]
    assert v["n_fits"] == 90


def test_a_correct_growing_cell_is_admitted():
    assert not _gate(_cell(growing=True), growing=True)["refused"]


MUTATIONS = {
    "smooth_window default": lambda r: r["selection"].update(smooth_window=1),
    "swap default": lambda r: r["selection"].update(swap=True),
    "patience default": lambda r: r["selection"].update(patience=20),
    "valid_loss selection": lambda r: r["selection"].update(selection_monitor="valid_loss"),
    "early stop": lambda r: r.update(epochs=57),
    "no growth held-out": lambda r: r.update(n_holdout=0),
    "weights not saved": lambda r: r.update(selected_path=None),
    "wrong lr": lambda r: r["optimizer"].update(lr=1e-2),
    "silent wd": lambda r: r["optimizer"].update(weight_decay=0.0),
    "cosine schedule": lambda r: r["history"][-1].update(lr=0.0),
    "no first batch loss": lambda r: r.update(first_batch_loss=None),
    "spike": lambda r: r.update(grad_norm_max_fit=5.6e6),
    "nan spike": lambda r: r.update(grad_norm_max_fit=float("nan")),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_one_bad_fit_refuses_the_cell(name):
    recs = _cell()
    MUTATIONS[name](recs[37])
    v = _gate(recs)
    assert v["refused"], name


def test_spike_is_counted_and_named():
    recs = _cell()
    recs[5]["grad_norm_max_fit"] = 1e16
    v = _gate(recs)
    assert v["n_spikes"] == 1 and v["spiked_fit_uids"] == [recs[5]["fit_uid"]]


def test_line_search_on_training_data_is_refused_for_growing_arms():
    recs = _cell(growing=True)
    recs[10]["history"][4]["grow_holdout"] = "train_tail"
    assert _gate(recs, growing=True)["refused"]


def test_phantom_fits_of_a_relaunched_cell_are_refused():
    """The JSONL is append-only: a relaunch on top of a cancelled cell duplicates folds."""
    recs = _cell()
    recs.append(copy.deepcopy(recs[0]))
    v = _gate(recs)
    assert v["refused"] and any("phantom" in r for r in v["reasons"])


def test_a_missing_fold_is_refused():
    recs = _cell()
    del recs[41]
    assert _gate(recs)["refused"]


def test_a_missing_subject_is_refused():
    recs = [r for r in _cell() if r["subject"] != "7"]
    assert _gate(recs)["refused"]


def test_cli_writes_verdict_and_exits_nonzero_on_refusal(tmp_path):
    cell = g.cell_path(tmp_path, "within_session", "bnci2014_001", "bd_deep4", 0)
    cell.parent.mkdir(parents=True)
    cell.write_text("\n".join(json.dumps(r) for r in _cell()) + "\n")
    out = tmp_path / "v.json"
    rc = g.main(["--root", str(tmp_path), "--arm", "bd_deep4", "--seeds", "0", "1",
                 "--lr", str(LR), "--weight-decay", str(WD), "--out", str(out)])
    v = json.loads(out.read_text())
    assert rc == 1                                  # seed1 is missing
    assert v["refused_cells"] == ["seed1"]
    assert not v["cells"]["seed0"]["refused"]


def test_a_truncated_cell_is_refused_not_skipped(tmp_path):
    cell = g.cell_path(tmp_path, "within_session", "bnci2014_001", "bd_deep4", 0)
    cell.parent.mkdir(parents=True)
    lines = [json.dumps(r) for r in _cell()]
    cell.write_text("\n".join(lines) + "\n" + lines[0][:50])
    out = tmp_path / "v.json"
    rc = g.main(["--root", str(tmp_path), "--arm", "bd_deep4", "--seeds", "0",
                 "--lr", str(LR), "--weight-decay", str(WD), "--out", str(out)])
    assert rc == 1
