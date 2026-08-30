"""Reading the fit records back has to survive the two things that changed under it.

Both changes are silent by construction, which is why they get a test rather than a
re-read of the figures:

* the alignment arm added a middle field to the filename, and a reader that folds it
  into the model name gives ``grow_shallow__easubject`` no ``bd_shallow`` to pair with;
* the subject stamp made the held-out subject a *recorded* fact, and the positional
  join that predates it would happily overwrite a recorded label with a guessed one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "analysis"))

import growth_io  # noqa: E402


def _record(fit: int, *, subject=None, session=None, epochs: int = 3) -> dict:
    rec = {
        "fit": 0,  # always 0 on disk: MOABB clones the callback, see growth_io
        "n_train": 100, "seconds": 1.0, "epochs": epochs, "max_epochs": 200,
        "width_start": 8, "width_end": 40, "target_width": 40,
        "params_start": 1000, "params_end": 2000,
        "history": [{"epoch": e, "train_loss": 1.0 / e, "valid_loss": 1.0 / e,
                     "valid_acc": 0.5 + 0.01 * e + 0.001 * fit,
                     "width": 8 + e, "n_params": 1000 + 10 * e,
                     "grow_applied": e == 2, "grow_s": 0.5 if e == 2 else None}
                    for e in range(1, epochs + 1)],
    }
    if subject is not None:
        rec |= {"subject": subject, "session": session, "cv_ind": fit}
    return rec


def _tree(tmp_path: Path, stems: dict[str, list[dict]]) -> Path:
    root = tmp_path / "results"
    d = root / "within_session" / "bnci2014_001"
    d.mkdir(parents=True)
    for stem, recs in stems.items():
        (d / f"{stem}__fits.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
    return root


@pytest.mark.parametrize("stem,expected", [
    ("grow_shallow__seed0", ("grow_shallow", "", "0")),
    ("grow_shallow__easubject__seed0", ("grow_shallow", "easubject", "0")),
    ("bd_deep4__seed12", ("bd_deep4", "", "12")),
    ("fix_sccnet__easubject__seed4", ("fix_sccnet", "easubject", "4")),
])
def test_the_alignment_tag_is_not_part_of_the_model_name(stem, expected):
    assert growth_io._split_stem(stem) == expected


def test_the_two_alignment_arms_stay_two_cells(tmp_path):
    """Same (eval, dataset, model, seed), different arm: they must not collide."""
    root = _tree(tmp_path, {
        "grow_shallow__seed0": [_record(0), _record(1)],
        "grow_shallow__easubject__seed0": [_record(0), _record(1)],
    })
    fits, curves = growth_io.load(root)

    assert set(fits.model) == {"grow_shallow"}, "the tag leaked into the model name"
    assert sorted(fits.align_tag.unique()) == ["", "easubject"]
    # The point of keeping align_tag in the cell key: without it these four folds are
    # two, and every per-epoch mean silently averages the ablation with its control.
    assert len(curves.groupby(list(growth_io.CELL) + ["fit"])) == 4


def test_a_stamped_campaign_keeps_its_recorded_subjects(tmp_path):
    """attach_subjects must return recorded labels, not overwrite them with guesses."""
    root = _tree(tmp_path, {"grow_shallow__seed0": [
        _record(0, subject="8", session="0train"),
        _record(1, subject="3", session="1test"),
    ]})
    _, curves = growth_io.load(root)
    assert set(curves.subject) == {"8", "3"}, "the stamp did not reach the epoch rows"

    # Scores deliberately name *different* subjects: if the positional join ran, it
    # would win and these are the values we would see.
    scores = pd.DataFrame({"eval": ["within_session"] * 2,
                           "dataset": ["bnci2014_001"] * 2,
                           "model": ["grow_shallow"] * 2,
                           "align_tag": ["", ""], "seed": [0, 0],
                           "subject": ["99", "77"], "session": ["x", "y"]})
    out = growth_io.attach_subjects(curves, scores, n_splits=1)
    assert set(out.subject) == {"8", "3"}, "recorded subjects were overwritten"
    assert not out.subject_inferred.any()


def test_an_unstamped_campaign_still_falls_back_to_the_join(tmp_path):
    """v5 has no stamp; its per-subject figures must keep working, and say so."""
    root = _tree(tmp_path, {"grow_shallow__seed0": [_record(0), _record(1)]})
    _, curves = growth_io.load(root)
    assert "subject" not in curves.columns

    scores = pd.DataFrame({"eval": ["within_session"] * 2,
                           "dataset": ["bnci2014_001"] * 2,
                           "model": ["grow_shallow"] * 2,
                           "align_tag": ["", ""], "seed": [0, 0],
                           "subject": ["8", "3"], "session": ["0train", "1test"]})
    out = growth_io.attach_subjects(curves, scores, n_splits=1)
    assert set(out.subject) == {"8", "3"}
    assert out.subject_inferred.all(), "an inferred label must be flagged as inferred"
