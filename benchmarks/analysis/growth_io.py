"""Read the per-fit JSONL records written by ``eegrow.training.recording.FitRecorder``.

The benchmark writes one ``<model>__seed<N>__fits.jsonl`` per cell, next to the cell's
score CSV, with one JSON object per fit (i.e. per cross-validation fold). This module
turns that tree into two tidy frames:

``fits``    one row per fit -- start/end width, start/end parameter count, epochs run,
            seconds, and the cell's coordinates. This is the frame that answers "did
            the model reach its target width", which nothing could answer before: the
            width was only ever passed to ``logger.info`` with ``verbose=False``.

``curves``  one row per (fit, epoch) -- loss, validation accuracy, width, parameter
            count. This is the frame the trajectory figures are drawn from.

Kept out of the notebook on purpose: parsing a directory tree is the part that breaks
silently when a path convention changes, so it belongs somewhere importable and
testable rather than in a cell nobody re-reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# eval/dataset are the two directory levels above the file, matching
# ``utils.results_path``: <results_dir>/<eval>/<dataset>/<model>__seed<N>__fits.jsonl
GLOB = "*/*/*__fits.jsonl"


def _split_stem(stem: str) -> tuple[str, str, str]:
    """``<model>[__<align tag>]__seed<N>`` -> ``(model, align_tag, seed)``.

    The alignment arm added a middle field (``utils.cell_stem``), and reading it back
    as part of the model name is not a cosmetic error: every pairing in the analysis
    matches ``grow_X`` against ``bd_X`` by string, so ``grow_shallow__easubject``
    silently has no partner and drops out of the comparison it was run for. No model
    label contains ``__``, which is what makes the middle field unambiguous.
    """
    head, _, seed = stem.partition("__seed")
    model, _, tag = head.partition("__")
    return model, tag, seed


def _records(root: Path):
    for path in sorted(Path(root).glob(GLOB)):
        ev, ds = path.parts[-3], path.parts[-2]
        stem = path.name[: -len("__fits.jsonl")]
        model, align_tag, seed = _split_stem(stem)
        index = 0
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:  # truncated final line: a job
                    # killed mid-write. Skip it rather than lose the whole cell.
                    print(f"  ! {path}:{lineno} unreadable ({exc}); skipped")
                    continue
                # The path is authoritative for the coordinates; `meta` only carries
                # what the callback was told, and a mismatch means a stale file.
                rec["eval"], rec["dataset"] = ev, ds
                rec["model"], rec["seed"] = model, int(seed) if seed else None
                # "" for the raw arm, so a campaign that predates alignment reads back
                # with the same value it would have been given had the column existed.
                rec["align_tag"] = align_tag
                # And the LINE ORDER is authoritative for the fold, not the `fit`
                # counter inside the record. MOABB clones the pipeline per fold, so
                # the callback is cloned too and its counter restarts at 0 every
                # time: every record claims fit=0. Grouping on that field would fold
                # a cell's ten trajectories into one and draw a sawtooth instead of
                # ten curves. Appends within a cell come from a single process, so
                # the file order is the fold order.
                rec["fit_reported"] = rec.get("fit")
                rec["fit"] = index
                index += 1
                yield rec


# The coordinates of a cell. ``align_tag`` is part of the identity, not a label: the
# raw and aligned arms are the same (eval, dataset, model, seed) point, so a groupby
# that omits it averages the two arms of the ablation into one number.
CELL = ("eval", "dataset", "model", "align_tag", "seed")
# Written by subject_stamp when the evaluation was wrapped; absent on older campaigns.
STAMP = ("subject", "session", "cv_ind")


def load(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(fits, curves)`` for every cell under ``root``."""
    fit_rows, curve_rows = [], []
    for rec in _records(root):
        history = rec.pop("history", []) or []
        fit_rows.append(rec)
        key = {k: rec[k] for k in CELL + ("fit",)}
        # Carried onto every epoch row so a per-subject curve figure needs no join at
        # all on a stamped campaign. Silently absent otherwise -- see attach_subjects.
        key.update({k: rec[k] for k in STAMP if k in rec})
        for point in history:
            curve_rows.append({**key, **point})

    fits = pd.DataFrame(fit_rows)
    curves = pd.DataFrame(curve_rows)
    if fits.empty:
        raise FileNotFoundError(
            f"no '{GLOB}' under {root}. These records only exist for campaigns run "
            "after FitRecorder was wired in; a grid produced before it has no growth "
            "trajectory at all, and none can be reconstructed from the score CSVs.")

    return _derive(fits), curves


def _derive(fits: pd.DataFrame) -> pd.DataFrame:
    """The four columns every growth figure reads, added in one place.

    They are recomputed on the tidy path too rather than trusted from the file: a
    column that can be derived should never be a thing a reader has to check was
    exported correctly.
    """
    # `grew` separates a growable arm from a frozen one without relying on the model
    # name: a fixed control is exactly a model whose width never moved.
    fits["grew"] = fits["width_end"] > fits["width_start"]
    fits["reached_target"] = fits["width_end"] == fits["target_width"]
    fits["params_ratio"] = fits["params_end"] / fits["params_start"]
    fits["stopped_early"] = fits["epochs"] < fits["max_epochs"]
    return fits


def load_tidy(directory) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(fits, budget, curves)`` from the exported tidy files.

    The JSONL tree is not portable: the v5 campaign's records are 688 MB across 1282
    files on the cluster, and `load` has to parse all of it to answer any question.
    Reducing it once and carrying the reduction home is what makes the figures
    reproducible off-cluster -- but only the reductions that stay small do:

    * ``eegrow_v5_fits.csv.gz`` -- every fold, 0.9 MB;
    * ``eegrow_v5_budget.csv.gz`` -- parameter-epochs per fold, a reduction of the
      full curves and therefore not recomputable from what is shipped;
    * ``eegrow_v5_curves_showcase.csv.gz`` -- the per-epoch history for ONE dataset.
      Full curves are 129 MB gzipped and no figure draws 128 801 trajectories; the
      ones that draw trajectories draw a single cell's.

    So ``curves`` here is deliberately narrower than ``load``'s: any figure using it
    must state which dataset it is showing rather than implying the whole grid.
    """
    d = Path(directory)
    fits = pd.read_csv(d / "eegrow_v5_fits.csv.gz")
    budget = pd.read_csv(d / "eegrow_v5_budget.csv.gz")
    curves = pd.read_csv(d / "eegrow_v5_curves_showcase.csv.gz")
    return _derive(fits), budget, curves


def attach_subjects(curves: pd.DataFrame, scores: pd.DataFrame,
                    *, n_splits: int = 5) -> pd.DataFrame:
    """Add ``subject``/``session`` to ``curves`` by joining on write order.

    ``FitRecorder`` stamps ``(eval, dataset, model, seed)`` but NOT the fold's
    coordinates: the callback is constructed once per cell, before MOABB clones the
    pipeline per fold, so it never learns which subject it is fitting. Folds are
    therefore only distinguishable by their order in the JSONL.

    That order is recoverable because both files are written by the same process in
    the same loop: MOABB appends one score row per (subject, session) after running
    that pair's ``n_splits`` folds, and the recorder appends one line per fold. So the
    k-th score row owns fit indices ``k * n_splits`` .. ``k * n_splits + n_splits - 1``.

    **Do not sort the score rows.** MOABB does not iterate subjects in sorted order
    (bnci2014_001 comes out 8, 8, 3, 3, 2, 2, 5, ...), and sorting them is precisely
    the assumption that makes this join wrong -- the two orderings guessable a priori
    (subject-major, session-major) correlate with the published per-subject ranks at
    Spearman +0.03 and +0.10, i.e. not at all.

    Validated, not assumed, and re-validated on the frames as they stand after the
    ``drop_last`` re-run: within each cell, the mean best ``valid_acc`` over a block of
    ``n_splits`` fits against the score of the row it is assigned to gives Spearman
    +0.84 to +0.99 for eight of the nine arms across all five seeds (mean +0.89 over
    45 cells). ``bd_deep4`` is the exception at +0.33 to +0.59 -- still positive, but
    its 297 k parameters overfit the 46-trial internal split, which makes ``valid_acc``
    a weak proxy for the test score rather than the join a weak join.

    This returns an *inferred* label, not a recorded one, and only for campaigns that
    have no recorded one. Since ``subject_stamp`` was wired in, the evaluation names the
    held-out subject on every fit record, so ``load`` already carries a real
    ``subject``/``session`` on the epoch rows; this function then returns them
    untouched. Read ``subject_inferred`` to know which of the two you are looking at --
    a figure that mixes a recorded label with a guessed one and says neither is how a
    per-subject claim gets made on the wrong subject.
    """
    # A stamped campaign needs no join, and doing one anyway would *overwrite* recorded
    # identities with guessed ones. Partial coverage is the interesting case: a tree
    # holding both old and re-run cells. Only the unstamped rows go through the join.
    if "subject" in curves.columns:
        stamped = curves[curves.subject.notna()].copy()
        stamped["subject_inferred"] = False
        rest = curves[curves.subject.isna()]
        if rest.empty:
            return stamped
        joined = attach_subjects(rest.drop(columns=["subject", "session"],
                                           errors="ignore"),
                                 scores, n_splits=n_splits)
        return pd.concat([stamped, joined], ignore_index=True)

    keys = [k for k in ("eval", "dataset", "model", "align_tag", "seed")
            if k in curves.columns and k in scores.columns]
    out = []
    for key, rows in scores.groupby(keys, sort=False):
        mask = pd.Series(True, index=curves.index)
        for col, val in zip(keys, key):
            mask &= curves[col] == val
        cell = curves[mask]
        if cell.empty:
            continue
        # Positional, in write order -- see above. `rows` keeps its file order:
        # `groupby` never reorders rows *within* a group.
        block = {fit: sub_sess
                 for k, sub_sess in enumerate(zip(rows.subject, rows.session))
                 for fit in range(k * n_splits, (k + 1) * n_splits)}
        cell = cell.copy()
        cell["subject"] = [block.get(f, (None, None))[0] for f in cell.fit]
        cell["session"] = [block.get(f, (None, None))[1] for f in cell.fit]
        cell["subject_inferred"] = True
        out.append(cell)
    return pd.concat(out, ignore_index=True)


def parameter_epochs(curves: pd.DataFrame) -> pd.DataFrame:
    """Cumulative parameter-epochs per fit -- the capacity actually paid for.

    Final parameter count flatters a growing model: it spent most of its epochs
    narrower than it ended. Summing ``n_params`` over epochs is the honest budget
    axis, and it is the quantity the efficiency claim is about. A fixed arm's budget
    is just ``n_params x epochs``, which falls out of the same sum.
    """
    keys = [k for k in CELL + ("fit",) if k in curves.columns]
    out = (curves.sort_values("epoch")
           .groupby(keys, as_index=False)
           .agg(param_epochs=("n_params", "sum"),
                epochs=("epoch", "max"),
                params_end=("n_params", "last"),
                best_valid_acc=("valid_acc", "max")))
    return out
