"""Euclidean alignment and the subject stamp have to work at the same time.

The two features were built on branches that never saw each other, and they touch the
same MOABB call from opposite sides: ``subject_stamp`` overrides ``_build_task_list``
on the *evaluation*, while ``aligned_paradigm`` wraps ``get_data`` on the *paradigm*.
Nothing in either module fails if the other is absent, which is exactly why their
combination needs its own test -- the failure mode is not an exception, it is 12
datasets of aligned cells whose fit records cannot name their subject, discovered at
analysis time.

Run against ``FakeDataset`` on purpose: the question is MOABB's control flow, not the
data. A stand-in estimator writes the recorder's JSONL itself, because instantiating a
real ``EEGClassifier`` here would test torch.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from eegrow.training.recording import FitRecorder  # noqa: E402

moabb = pytest.importorskip("moabb")


class _Const(BaseEstimator, ClassifierMixin):
    """Trivial classifier exposing a ``callbacks`` param, like a skorch net.

    It writes the record itself so the test exercises the stamping path and nothing
    else. ``X`` is kept so a caller can check what the paradigm actually served.
    """

    def __init__(self, callbacks=None):
        self.callbacks = callbacks

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for item in self.callbacks or []:
            cb = item[1] if isinstance(item, tuple) else item
            if isinstance(cb, FitRecorder):
                path = Path(cb.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as fh:
                    fh.write(json.dumps({**(cb.meta or {}),
                                         "rms": float(np.sqrt((X ** 2).mean()))}) + "\n")
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        p = np.zeros((len(X), len(self.classes_)))
        p[:, 0] = 1.0
        return p

    def decision_function(self, X):
        return np.zeros(len(X))


def _fake_dataset():
    from moabb.datasets.fake import FakeDataset

    return FakeDataset(n_sessions=2, n_subjects=4, n_runs=2, paradigm="imagery",
                       event_list=["left_hand", "right_hand"])


def _paradigm(aligned: bool):
    from aligned_paradigm import make_aligned_paradigm
    from moabb.paradigms import LeftRightImagery

    cls = LeftRightImagery
    if aligned:
        cls = make_aligned_paradigm(cls, level="subject", preserve_scale=True,
                                    rcond=1e-12)
    return cls()


def _run(aligned: bool):
    """One stamped, optionally aligned, WithinSessionEvaluation. Returns its records."""
    from moabb import evaluations as mev
    from subject_stamp import stamped

    tmp = Path(tempfile.mkdtemp())
    jsonl = tmp / "fits.jsonl"
    dataset = _fake_dataset()
    tag = "aligned" if aligned else "raw"
    evaluation = stamped(mev.WithinSessionEvaluation)(
        paradigm=_paradigm(aligned), datasets=[dataset], overwrite=True,
        random_state=0, n_jobs=1, hdf5_path=str(tmp), suffix=tag, n_splits=3)
    clf = _Const(callbacks=[("record", FitRecorder(str(jsonl),
                                                   meta={"model": "grow_shallow"}))])
    evaluation.process({tag: clf})
    return [json.loads(line) for line in jsonl.open()], dataset


def test_alignment_does_not_cost_the_subject_stamp():
    """Every fit of an aligned evaluation still names its own held-out subject."""
    records, dataset = _run(aligned=True)

    assert records, "no fit records written under the aligned paradigm"
    assert all(r["model"] == "grow_shallow" for r in records), "meta lost"
    got = sorted({r["subject"] for r in records})
    assert got == sorted(str(s) for s in dataset.subject_list), got
    assert {r["session"] for r in records} == {"0", "1"}
    # Equal fold counts: a fold stamped with a neighbour's id would still produce the
    # right *set* of subjects, and that is the bug worth catching.
    per = {s: sum(r["subject"] == s for r in records) for s in got}
    assert len(set(per.values())) == 1, per


def test_the_aligned_arm_actually_serves_different_data():
    """Guard against a silent no-op: `align=euclidean` must change what is fitted.

    Whitening is scale-preserving by construction here (`preserve_scale=True` rescales
    by one global factor), so equal RMS is expected and is *not* evidence the transform
    ran. The trial values themselves have to differ.
    """
    aligned, _ = _run(aligned=True)
    raw, _ = _run(aligned=False)

    assert len(aligned) == len(raw), "the two arms must fit the same folds"
    by_subject = lambda recs: {(r["subject"], r["session"], r["cv_ind"]): r["rms"]
                               for r in recs}
    a, b = by_subject(aligned), by_subject(raw)
    assert a.keys() == b.keys(), "the arms disagree on which folds exist"
    assert any(a[k] != b[k] for k in a), "alignment changed nothing -- silent no-op"
