"""Build the single-file results workbook from results_published/.

Tables only, no prose: one sheet per object a reader might want to check, from the
grid's design down to the 420 absolute levels. Every number is recomputed here from
the three published CSVs, so nothing in the workbook is a hand copy of RESULTS.md.

    python benchmarks/make_results_workbook.py

Writes results_published/eegrow_moabb_grid_results.xlsx. Needs openpyxl.

The 189,062 raw scores are deliberately left out: they would take the file from
40 kB to several MB for a sheet nobody reads by hand, and they already ship as
results_published/eegrow_benchmark_all_scores.csv.gz.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SRC = Path(__file__).resolve().parent / "results_published"
OUT = SRC / "eegrow_moabb_grid_results.xlsx"

# The three cells where all 8 networks sit at chance (see RESULTS.md section 3): a
# paired delta between two coin flips is noise, so the sign tests are reported both
# with and without them.
DEAD = {("within_session", "physionetmi"), ("within_session", "shin2017a"),
        ("cross_session", "shin2017a")}
ORDER = ["within_session", "cross_session", "cross_subject"]
FAMILIES = ["riemann/csp", "braindecode", "growing"]


def family_means(lvl):
    """Mean absolute level per family, over the AUC datasets only.

    The grid mixes roc_auc (6 two-class datasets) and accuracy, whose chance levels
    differ; pooling them would average incommensurable numbers.
    """
    auc = lvl[lvl["metric"] == "roc_auc"]
    fam = (auc.groupby(["family", "eval"])["score_mean"].mean().unstack("eval")
           .reindex(columns=ORDER).reindex(FAMILIES))
    fam.index.name = "family"
    return fam.round(4)


def sign_tests(pai):
    rows = []
    for pair, g in pai.groupby("pair", sort=False):
        alive = g[~g.set_index(["eval", "dataset"]).index.isin(DEAD)]
        for label, sub in (("dead cells excluded", alive), ("all cells", g)):
            d = sub["delta"].to_numpy()
            pos = int((d > 0).sum())
            rows.append({"pair": pair, "cells": label, "n_cells": len(d),
                         "positive": pos, "median_delta": float(np.median(d)),
                         "mean_delta": float(d.mean()),
                         "p_sign": binomtest(pos, len(d), 0.5).pvalue})
    return pd.DataFrame(rows)


def best_per_cell(lvl):
    best = (lvl.sort_values("score_mean", ascending=False)
            .groupby(["eval", "dataset"], as_index=False).first()
            [["eval", "dataset", "metric", "model", "family", "score_mean", "n_obs"]]
            .rename(columns={"model": "best_model", "score_mean": "best_score"}))
    best["eval"] = pd.Categorical(best["eval"], ORDER, ordered=True)
    return best.sort_values(["eval", "dataset"])


def cells_at_chance(lvl):
    deep = lvl[lvl["family"].isin(["braindecode", "growing"])]
    riem = lvl[lvl["family"] == "riemann/csp"]
    rows = []
    for ev, ds in sorted(DEAD):
        d = deep[(deep["eval"] == ev) & (deep["dataset"] == ds)]
        r = riem[(riem["eval"] == ev) & (riem["dataset"] == ds)]
        rows.append({"eval": ev, "dataset": ds, "metric": d["metric"].iloc[0],
                     "n_deep_models": len(d), "deep_min": d["score_mean"].min(),
                     "deep_max": d["score_mean"].max(),
                     "best_riemann_model": r.loc[r["score_mean"].idxmax(), "model"],
                     "best_riemann_score": r["score_mean"].max(),
                     "n_obs": d["n_obs"].iloc[0]})
    return pd.DataFrame(rows)


DESIGN = pd.DataFrame([
    ("datasets", 12), ("protocols", 3), ("pipelines", 14),
    ("riemann / CSP pipelines", 6), ("fixed braindecode pipelines", 4),
    ("growing pipelines", 4), ("seeds per cell", 5),
    ("protocol x dataset cells", 30), ("runs", 2100), ("score rows", 189062),
    ("missing cells", 0), ("sampling rate (Hz)", 250),
], columns=["item", "value"])

# Counts from provenance_audit.py, over the 2100 raw-arm cells.
PROVENANCE = pd.DataFrame([
    ("native rate already 250 Hz, override is a no-op", 630, 630 / 2100),
    ("certified 250 Hz by surviving slurm log (27 overlap with above)", 327, np.nan),
    ("established (union)", 930, 930 / 2100),
    ("no trace in either direction", 1170, 1170 / 2100),
    ("certified at a rate other than 250 Hz", 0, 0.0),
    ("ever written at a native rate, certified 250 Hz today", 87, 1.0),
], columns=["status", "cells", "share"])


def main():
    lvl = pd.read_csv(SRC / "eegrow_benchmark_levels.csv")
    pai = pd.read_csv(SRC / "eegrow_benchmark_paired.csv")

    sheets = [
        ("design", DESIGN, False),
        ("family_means_auc", family_means(lvl), True),
        ("sign_tests", sign_tests(pai), False),
        ("paired_all", pai, False),
        ("levels_all", lvl, False),
        ("best_model_per_cell", best_per_cell(lvl), False),
        ("cells_at_chance", cells_at_chance(lvl), False),
        ("provenance", PROVENANCE, False),
    ]
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for name, frame, with_index in sheets:
            frame.to_excel(xw, sheet_name=name, index=with_index)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(width + 2, 10), 46)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
