"""Reduce the growth records to the frames the dynamics figures read -- by STREAMING.

WHY NOT ``export_v5_tidy``. That exporter calls ``growth_io.load``, which holds every
epoch row of the campaign in memory at once; its own docstring asks for a 96 GB job. The
final campaign's records are **9.2 GB of JSON across 948 files**, and the per-epoch rows
carry two variable-length lists (the candidate spectra), so the in-memory frame is
several times the file size and the peak is unbounded in the campaign's size. It also
cannot run while the grid is still writing: a single unreadable trailing line aborts a
load that took forty minutes to get there.

This module reduces **one cell at a time** and keeps only the reductions, so peak memory
is one JSONL file (54 MB at the worst) rather than the tree. That is what makes it
runnable *now*, on a campaign that is 80 % done and still being appended to.

WHAT COMES OUT, AND WHY EACH CUT IS THE RIGHT ONE

``fits``        one row per fold: the cell's coordinates, the stamped ``subject`` /
                ``session``, ``stop_reason``, ``restored_epoch``, ``epochs``, the width
                and parameter endpoints. Small (one row per fold), and it is the only
                frame from which "which subject did this arm stop early on" can be
                answered at all.

``curves_mean`` one row per (cell, epoch): the fold-mean and sd of every per-epoch
                diagnostic, **plus ``n_folds`` at that epoch**. Averaging curves of
                unequal length is the standing trap -- a fold that stopped is a fold
                removed from the mean, and the folds that survive to epoch 150 are the
                ones that were still improving at 130, so the mean drifts upward for a
                reason that is not training. Every figure that draws a mean here reads
                ``n_folds`` and marks the survivorship region rather than hiding it.

``events``      one row per growth step actually applied, across all datasets, WITH the
                two candidate spectra kept verbatim as JSON. This is the expensive
                choice and the deliberate one: the spectra are what turn "growth added
                seven neurons" into "growth was offered twenty-six candidates whose
                eigenvalues spanned four decades and kept the seven above the relative
                floor". Growth epochs are rare (7 in 200 on the fold measured), so
                keeping them in full costs less than keeping every epoch of one dataset.

                Each event also carries the losses on the epochs **around** it
                (``train_loss_before`` / ``_after``), computed here because they need
                the fold's full history and are not recoverable from any reduction. They
                are what makes gromo's *expected* first-order gain checkable against the
                gain that actually materialised -- the one figure that can say whether a
                growth step was a good decision rather than merely a decision.

Usage, from anywhere (the reducer is pure I/O, no GPU, no eegrow import)::

    python benchmarks/analysis/export_growth_dynamics.py \\
        /scratch/amounir/results_final /scratch/amounir/dynamics_final
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import growth_io  # noqa: E402  (for _split_stem: one parser for the path convention)

GLOB = "*/*/*__fits.jsonl"

#: Fit-level fields lifted verbatim. ``subject``/``session``/``cv_ind`` exist only on a
#: campaign run after ``subject_stamp`` was wired in; absent ones become NaN rather than
#: raising, so this reads an older tree too (minus the per-subject figures).
FIT_FIELDS = ("n_train", "seconds", "epochs", "max_epochs", "width_start", "width_end",
              "target_width", "params_start", "params_end", "restored_epoch",
              "stop_reason", "subject", "session", "cv_ind")

#: Per-epoch diagnostics averaged over folds. Everything the optimisation figures read.
MEAN_COLS = ("train_loss", "valid_loss", "valid_acc", "width", "n_params", "dur",
             "grad_norm", "grad_norm_max", "lr",
             "adam_atten_mean", "adam_atten_p05", "adam_eps_frac")

#: Scalar fields of a growth step.
EVENT_COLS = ("grow_s", "grow_n_proposed", "grow_n_kept", "grow_width_after",
              "grow_first_order_improvement", "grow_eig_sum", "grow_select_loss",
              "grow_param_update_decrease")

#: The candidate spectra, kept as JSON strings. Storing them as lists in a CSV would
#: round-trip through ``repr`` and come back as text anyway; JSON at least parses.
SPECTRA = ("grow_eig_proposed", "grow_eig_kept")


def _iter_cells(root: Path):
    """Yield ``(coords, [record, ...])`` one JSONL file at a time.

    The path is authoritative for the coordinates -- ``meta`` only carries what the
    callback was told, and a mismatch means a stale file. Line order is authoritative
    for the fold index: MOABB clones the pipeline per fold, so the recorder's own ``fit``
    counter restarts at 0 on every one of them and every record claims ``fit=0``.
    """
    for path in sorted(Path(root).glob(GLOB)):
        ev, ds = path.parts[-3], path.parts[-2]
        model, align_tag, seed = growth_io._split_stem(path.name[: -len("__fits.jsonl")])
        coords = {"eval": ev, "dataset": ds, "model": model,
                  "align_tag": align_tag, "seed": int(seed) if seed else None}
        records = []
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A fit killed mid-write, or the line the running job is appending
                    # right now. Skip the line, keep the cell -- this exporter is meant
                    # to run against a live campaign, so a partial tail is the normal
                    # case and not an error worth losing 90 folds over.
                    print(f"  ! {path.name}:{lineno} unreadable; skipped")
        if records:
            yield coords, records


def _reduce_cell(coords: dict, records: list[dict]):
    """One cell -> (fit rows, curves_mean rows, event rows). Nothing else is kept."""
    fit_rows, epoch_rows, event_rows = [], [], []

    for fit_idx, rec in enumerate(records):
        history = rec.get("history") or []
        opt = rec.get("optimizer") or {}
        fit_rows.append({
            **coords, "fit": fit_idx,
            **{k: rec.get(k) for k in FIT_FIELDS},
            # Read off the live optimizer at fit time, so it reports what ran rather
            # than what the config asked for.
            "opt_lr": opt.get("lr"), "opt_eps": opt.get("eps"),
            "opt_weight_decay": opt.get("weight_decay"),
        })

        # Indexed once so the growth events below can look at their neighbours in O(1).
        by_epoch = {h.get("epoch"): h for h in history}
        for h in history:
            epoch_rows.append({"fit": fit_idx, "epoch": h.get("epoch"),
                               **{c: h.get(c) for c in MEAN_COLS}})
            # EVERY epoch on which `grow_step` RAN, not only the ones it applied.
            # Filtering on `grow_applied` being true was the original cut here and it
            # was wrong in the way that matters: measured on shin2017a/grow_shallow,
            # the step runs on all 39 opportunities of a fold, proposes 26 candidates
            # every time, and the line search returns s=0 -- an *abstention* -- on
            # 93.6 % of them. Keeping only the applied ones throws away the decision
            # that dominates the mechanism and leaves a frame in which the line search
            # appears to always answer 1.0, because 1.0 is what it answers on the 6 %
            # of occasions it answers at all.
            if "grow_applied" not in h:
                continue
            ep = h.get("epoch")
            nxt, prv = by_epoch.get(ep + 1), by_epoch.get(ep - 1)
            event_rows.append({
                **coords, "fit": fit_idx, "epoch": ep,
                "applied": bool(h.get("grow_applied")),
                "subject": rec.get("subject"), "session": rec.get("session"),
                **{c: h.get(c) for c in EVENT_COLS},
                **{c: json.dumps(h.get(c)) for c in SPECTRA},
                # CALLBACK ORDER, and it inverts the naming if ignored. `FitRecorder`
                # records `width`/`n_params` in `on_epoch_end`, and it sits AFTER
                # `gromo` in the callback list (pipelines.py states the contract), so
                # the values stamped on a growth epoch are already POST-growth. The
                # honest "before" is the previous epoch's; the honest "after" is the
                # growth epoch's own. Reading `h` as "before" made every parameter
                # ratio come out at exactly 1.000, which is how this was caught.
                "width_before": (prv or {}).get("width"),
                "width_after": h.get("width"),
                "n_params_before": (prv or {}).get("n_params"),
                "n_params_after": h.get("n_params"),
                # The realised counterpart of `grow_first_order_improvement`. Growth is
                # applied at the END of epoch `ep`, so the loss it was predicted to
                # improve is the one measured on epoch `ep + 1`. `prv` is carried so a
                # figure can show the slope growth interrupted, and not mistake the
                # ordinary epoch-to-epoch descent for the growth step's own effect.
                "train_loss_prev": (prv or {}).get("train_loss"),
                "train_loss_before": h.get("train_loss"),
                "train_loss_after": (nxt or {}).get("train_loss"),
                "valid_loss_before": h.get("valid_loss"),
                "valid_loss_after": (nxt or {}).get("valid_loss"),
                "valid_acc_before": h.get("valid_acc"),
                "valid_acc_after": (nxt or {}).get("valid_acc"),
                "grad_norm_before": h.get("grad_norm"),
                "grad_norm_after": (nxt or {}).get("grad_norm"),
            })

    if not epoch_rows:
        return fit_rows, [], event_rows

    ep = pd.DataFrame(epoch_rows)
    cols = [c for c in MEAN_COLS if c in ep.columns]
    agg = {f"{c}_{s}": (c, s) for c in cols for s in ("mean", "std")}
    agg["n_folds"] = ("fit", "nunique")
    mean = ep.groupby("epoch", as_index=False).agg(**agg)
    for k, v in coords.items():
        mean[k] = v
    return fit_rows, mean.to_dict("records"), event_rows


def provenance(root: Path) -> pd.DataFrame:
    """The library versions and preprocessing behind the tree, deduplicated.

    Printed and shipped rather than assumed: a campaign that silently spans two
    ``eegrow_sha`` values is a campaign whose arms are not comparable, and the only
    place that shows up is here.
    """
    cols = ["sfreq", "resample_cfg", "fmin", "fmax", "device", "v_moabb",
            "v_braindecode", "v_torch", "v_gromo", "v_skorch", "v_sklearn", "v_mne",
            "eegrow_sha"]
    rows = []
    for p in sorted(glob.glob(str(root / "*/*/*__seed*.csv"))):
        if "__fits" in p:
            continue
        try:
            rows.append(pd.read_csv(p, nrows=1))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    pv = pd.concat(rows, ignore_index=True)
    return pv[[c for c in cols if c in pv.columns]].drop_duplicates()


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    fits, means, events = [], [], []
    t0, n_cells = time.time(), 0
    for coords, records in _iter_cells(root):
        f, m, e = _reduce_cell(coords, records)
        fits += f
        means += m
        events += e
        n_cells += 1
        if n_cells % 100 == 0:
            print(f"  {n_cells} cells, {len(fits):,} folds, {len(events):,} events, "
                  f"{time.time() - t0:.0f}s", flush=True)

    fits = pd.DataFrame(fits)
    if fits.empty:
        raise FileNotFoundError(f"no '{GLOB}' under {root}")
    # Derived once here so no figure has to re-derive them and get it subtly different.
    fits["grew"] = fits.width_end > fits.width_start
    fits["reached_target"] = fits.width_end == fits.target_width
    fits["params_ratio"] = fits.params_end / fits.params_start
    fits["stopped_early"] = fits.epochs < fits.max_epochs
    # How far past its own selected epoch the fit kept running. With EarlyStopping this
    # is `patience` by construction on every fold that stopped early -- which is exactly
    # the check worth being able to draw rather than assert.
    fits["epochs_past_best"] = fits.epochs - fits.restored_epoch

    files = {
        "gd_fits.csv.gz": fits,
        "gd_curves_mean.csv.gz": pd.DataFrame(means),
        "gd_events.csv.gz": pd.DataFrame(events),
        "gd_provenance.csv": provenance(root),
    }
    for name, frame in files.items():
        kw = {"compression": "gzip"} if name.endswith(".gz") else {}
        frame.to_csv(out / name, index=False, **kw)
        print(f"  {name}: {len(frame):,} rows, "
              f"{(out / name).stat().st_size / 1e6:.2f} MB")

    print(f"\n{n_cells} cells, {len(fits):,} folds in {time.time() - t0:.0f}s")
    # Coverage, printed rather than trusted: a re-export that silently drops an arm
    # looks exactly like a complete one until this table is read.
    print("\ndatasets per (model, eval):")
    print(fits.groupby(["model", "eval"]).dataset.nunique().unstack()
          .fillna(0).astype(int).to_string())
    if not fits.stop_reason.isna().all():
        print("\nstop_reason per model:")
        print(pd.crosstab(fits.model, fits.stop_reason).to_string())
    ev = pd.DataFrame(events)
    if not ev.empty:
        print(f"\ngrowth opportunities: {len(ev):,}, applied {int(ev.applied.sum()):,} "
              f"({100 * ev.applied.mean():.1f} %), "
              f"{ev.loc[ev.applied, 'grow_n_kept'].sum():,.0f} neurons added in total")
        print("\nabstention rate (s=0) per arm:")
        print((1 - ev.groupby("model").applied.mean()).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
