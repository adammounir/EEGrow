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


def _records(root: Path):
    for path in sorted(Path(root).glob(GLOB)):
        ev, ds = path.parts[-3], path.parts[-2]
        stem = path.name[: -len("__fits.jsonl")]
        model, _, seed = stem.partition("__seed")
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


def load(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(fits, curves)`` for every cell under ``root``."""
    fit_rows, curve_rows = [], []
    for rec in _records(root):
        history = rec.pop("history", []) or []
        fit_rows.append(rec)
        key = {k: rec[k] for k in ("eval", "dataset", "model", "seed", "fit")}
        for point in history:
            curve_rows.append({**key, **point})

    fits = pd.DataFrame(fit_rows)
    curves = pd.DataFrame(curve_rows)
    if fits.empty:
        raise FileNotFoundError(
            f"no '{GLOB}' under {root}. These records only exist for campaigns run "
            "after FitRecorder was wired in; a grid produced before it has no growth "
            "trajectory at all, and none can be reconstructed from the score CSVs.")

    # `grew` separates a growable arm from a frozen one without relying on the model
    # name: a fixed control is exactly a model whose width never moved.
    fits["grew"] = fits["width_end"] > fits["width_start"]
    fits["reached_target"] = fits["width_end"] == fits["target_width"]
    fits["params_ratio"] = fits["params_end"] / fits["params_start"]
    fits["stopped_early"] = fits["epochs"] < fits["max_epochs"]
    return fits, curves


def parameter_epochs(curves: pd.DataFrame) -> pd.DataFrame:
    """Cumulative parameter-epochs per fit -- the capacity actually paid for.

    Final parameter count flatters a growing model: it spent most of its epochs
    narrower than it ended. Summing ``n_params`` over epochs is the honest budget
    axis, and it is the quantity the efficiency claim is about. A fixed arm's budget
    is just ``n_params x epochs``, which falls out of the same sum.
    """
    out = (curves.sort_values("epoch")
           .groupby(["eval", "dataset", "model", "seed", "fit"], as_index=False)
           .agg(param_epochs=("n_params", "sum"),
                epochs=("epoch", "max"),
                params_end=("n_params", "last"),
                best_valid_acc=("valid_acc", "max")))
    return out
