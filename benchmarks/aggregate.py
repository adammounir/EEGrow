"""Aggregate per-job CSVs into one cross-subject / within-session summary table.

Each job writes ``results/<eval>/<dataset>/<model>__seed<k>.csv`` (one row per
subject/session/fold). This walks that tree and reports, per (eval, dataset, model),
the mean +/- std of the per-subject score -- the number you put in the PR.

    python benchmarks/aggregate.py                       # reads benchmarks/results
    python benchmarks/aggregate.py --results-dir <dir> --out summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(results_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(results_dir.rglob("*.csv"))
              if p.name != "summary.csv"]
    if not frames:
        raise SystemExit(f"no result CSVs under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    # per-subject mean first (over sessions/folds/seeds), then mean +/- std over subjects
    keys = ["eval", "dataset", "model"]
    by_subject = (df.groupby(keys + ["subject"])["score"].mean().reset_index())
    agg = (by_subject.groupby(keys)["score"]
           .agg(["mean", "std", "count"]).reset_index()
           .sort_values(keys))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchmarks/results")
    ap.add_argument("--out", default="benchmarks/results/summary.md")
    args = ap.parse_args()

    df = load(Path(args.results_dir))
    if "dataset" not in df.columns:
        df["dataset"] = "unknown"
    agg = summarise(df)

    lines = ["# Benchmark summary\n",
             "Mean +/- std of per-subject accuracy.\n",
             "| eval | dataset | model | mean | std | n_subj |",
             "|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        std = 0.0 if pd.isna(r["std"]) else r["std"]
        lines.append(f"| {r['eval']} | {r['dataset']} | {r['model']} | "
                     f"{r['mean']:.3f} | {std:.3f} | {int(r['count'])} |")
    report = "\n".join(lines) + "\n"
    Path(args.out).write_text(report)
    agg.to_csv(Path(args.results_dir) / "summary.csv", index=False)
    print(report)
    print(f"(written to {args.out})")


if __name__ == "__main__":
    main()
