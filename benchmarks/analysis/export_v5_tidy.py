"""Reduce the v5 JSONL growth records to the files the figures read, on the cluster.

Supersedes the ad-hoc ``export_growth_v5b.py``: same three outputs, plus the two the
training-dynamics figures need and could not be drawn without.

WHY A REDUCTION AT ALL. The records are 688 MB across ~1300 files; the per-epoch frame
is 129 MB gzipped for 128 801 folds. Carrying that home is possible and pointless -- no
figure draws 128 801 trajectories. But the previous reduction went too far the other
way: it kept the full per-epoch history for **one** dataset (bnci2014_001), so every
learning-curve, overfitting and width-trajectory figure was a statement about one
dataset out of twelve, and "does this model overfit" had no cross-dataset answer.

WHAT THE TWO NEW FILES ARE, AND WHY THAT IS THE RIGHT CUT

``curves_mean`` -- per (eval, dataset, model, seed, epoch): the fold-mean of each
    curve, plus **n_folds at that epoch**. Averaging curves of unequal length is a trap:
    every fold stops when early stopping fires, so at epoch 120 the mean is taken over
    only the folds that survived to 120, and those are not a random subset -- they are
    the ones still improving. The mean therefore drifts *upward* for reasons that have
    nothing to do with training. Exporting n_folds is what lets a figure fade or
    truncate the curve where the population thins instead of drawing survivorship bias
    as a result. sd is over folds, so a band is drawable too.

``fold_summary`` -- per fold, where the interesting epoch is and what the losses were
    there: ``epoch_of_best`` says whether the 200-epoch budget or the patience-20 early
    stopping is the binding constraint, and ``valid_loss_at_best - train_loss_at_best``
    is the generalisation gap at the point the model is actually selected. Both are
    per-fold reductions of curves, so neither is recomputable from anything shipped.

Usage (on Margaret, from a job with ~96 GB -- `load` holds every epoch row in memory)::

    python benchmarks/analysis/export_v5_tidy.py /scratch/amounir/results_v5 /out/dir
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import growth_io  # noqa: E402
import pandas as pd  # noqa: E402

# The dataset every other eegrow figure uses, so per-fold trajectories stay comparable
# with the rest of the report.
SHOWCASE = "bnci2014_001"
# What ``fold_summary`` reduces: the three curves that define the selected epoch.
CURVE_COLS = ["train_loss", "valid_loss", "valid_acc"]
# What ``curves_mean`` averages. Wider than CURVE_COLS on purpose: these are the
# columns the *mechanism* figures read -- why growth helps, not just whether it does --
# and they are recorded per epoch inside the fit, so a reduction that drops them cannot
# be redone off-cluster. `grow_s` is the scaling factor the line search picked,
# `grow_first_order_improvement` the gain it expected, `grow_eig_sum` the spectrum it
# kept; together they are the only evidence a growth step was a decision rather than a
# scheduled event. Averaged over folds like the losses, and NaN on epochs where no
# growth happened, which is what makes a growth event visible in the mean.
MEAN_COLS = CURVE_COLS + [
    "grad_norm", "grad_norm_max", "lr", "grow_s", "grow_applied",
    "grow_n_proposed", "grow_n_kept", "grow_first_order_improvement",
    "grow_eig_sum", "grow_select_loss", "grow_param_update_decrease",
    "adam_atten_mean", "adam_atten_p05", "adam_eps_frac",
]
KEY = ["eval", "dataset", "model", "align_tag", "seed"]


def curves_mean(curves: pd.DataFrame) -> pd.DataFrame:
    """Fold-mean of every curve per (cell, seed, epoch), with the surviving fold count."""
    keys = [k for k in KEY if k in curves.columns]
    g = curves.groupby(keys + ["epoch"], as_index=False)
    # Intersected with what is present: a campaign predating a diagnostic simply has
    # fewer columns, and an exporter that raised on that could not read v5 at all.
    cols = [c for c in MEAN_COLS if c in curves.columns]
    agg = {f"{c}_{s}": (c, s) for c in cols for s in ("mean", "std")}
    agg["n_folds"] = ("fit", "nunique")
    agg["width_mean"] = ("width", "mean")
    agg["n_params_mean"] = ("n_params", "mean")
    return g.agg(**agg)


def growth_events(curves: pd.DataFrame) -> pd.DataFrame:
    """Every epoch on which a growth step was actually applied, all datasets.

    The cut that makes this affordable: growth epochs are *rare* (a fold grows a
    handful of times in 200 epochs), so keeping them in full across twelve datasets
    costs a fraction of what keeping every epoch of one dataset costs -- and it is the
    only file from which "what did growth decide, and was it the same decision
    everywhere" can be answered. Without it that question is a statement about
    bnci2014_001, which is what the showcase already was.

    The two spectra (``grow_eig_proposed``, ``grow_eig_kept``) are dropped here and
    kept only in the showcase: they are per-epoch *lists*, so they dominate the file
    size, and a figure that draws a spectrum draws one cell's.
    """
    if "grow_applied" not in curves.columns:
        return pd.DataFrame()
    ev = curves[curves.grow_applied.fillna(0).astype(bool)]
    return ev.drop(columns=["grow_eig_proposed", "grow_eig_kept"], errors="ignore")


def fold_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """One row per fold: where its best epoch was and what the losses were there."""
    c = curves.sort_values("epoch")
    keys = [k for k in KEY if k in curves.columns] + ["fit"]
    # idxmax on valid_acc gives the *selected* epoch -- the same criterion skorch's
    # EarlyStopping monitors, so the row it points at is the model that gets used.
    best = c.loc[c.groupby(keys).valid_acc.idxmax()]
    out = best[keys + ["epoch"] + CURVE_COLS].rename(columns={
        "epoch": "epoch_of_best", "train_loss": "train_loss_at_best",
        "valid_loss": "valid_loss_at_best", "valid_acc": "best_valid_acc"})
    last = c.groupby(keys, as_index=False).agg(
        epochs=("epoch", "max"),
        final_train_loss=("train_loss", "last"),
        final_valid_loss=("valid_loss", "last"),
        min_valid_loss=("valid_loss", "min"),
        min_train_loss=("train_loss", "min"),
        width_end=("width", "last"))
    out = out.merge(last, on=keys)
    # The quantity the overfitting figure is about, named once here rather than
    # recomputed in every notebook that wants it.
    out["gap_at_best"] = out.valid_loss_at_best - out.train_loss_at_best
    return out


def provenance(root: Path) -> pd.DataFrame:
    cols = ["sfreq", "resample_cfg", "fmin", "fmax", "device", "v_moabb",
            "v_braindecode", "v_torch", "v_gromo", "v_skorch", "v_sklearn", "v_mne",
            "eegrow_sha"]
    rows = [pd.read_csv(p, nrows=1)
            for p in sorted(glob.glob(str(root / "*/*/*__seed*.csv")))
            if "__fits" not in p]
    pv = pd.concat(rows, ignore_index=True)
    return pv[[c for c in cols if c in pv.columns]].drop_duplicates()


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    fits, curves = growth_io.load(root)
    print(f"loaded {len(fits):,} folds, {len(curves):,} epoch rows, "
          f"{curves.dataset.nunique()} datasets")

    files = {
        "eegrow_v5_fits.csv.gz": fits,
        "eegrow_v5_budget.csv.gz": growth_io.parameter_epochs(curves),
        "eegrow_v5_curves_showcase.csv.gz": curves[curves.dataset == SHOWCASE],
        "eegrow_v5_curves_mean.csv.gz": curves_mean(curves),
        "eegrow_v5_fold_summary.csv.gz": fold_summary(curves),
        "eegrow_v5_growth_events.csv.gz": growth_events(curves),
        "eegrow_v5_provenance.csv": provenance(root),
    }
    for name, frame in files.items():
        kw = {"compression": "gzip"} if name.endswith(".gz") else {}
        frame.to_csv(out / name, index=False, **kw)
        print(f"  {name}: {len(frame):,} rows, "
              f"{(out / name).stat().st_size / 1e6:.2f} MB")

    # Coverage, printed rather than trusted. It caught a running campaign's uneven
    # grid when there was one; it stays because a re-export that silently drops an
    # arm looks exactly like a complete one until this line is read.
    cov = (fits.groupby(["model", "eval"]).dataset.nunique().unstack()
           .fillna(0).astype(int))
    print("\ndatasets per (model, eval):")
    print(cov.to_string())


if __name__ == "__main__":
    main()
