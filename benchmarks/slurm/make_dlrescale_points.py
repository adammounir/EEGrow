"""Emit the point list for re-running the published grid's deep arms at O(1) amplitude.

Why the list is *derived* rather than typed
------------------------------------------
The grid grew over several sweeps (native rate, then fix250, then the alignment ablation),
so the set of (eval, dataset, model, seed, align) points that actually ran is recorded in
the result CSVs and nowhere else. Typing a fresh cartesian product would silently re-run
points that were never in the grid and skip ones that were, and the re-run would then not
be comparable to what it replaces. Deriving it from ``benchmarks/results`` guarantees the
new numbers are a like-for-like replacement.

The classic arms are deliberately excluded. CSP solves a generalised eigenproblem and the
Riemannian arms work on covariances, so both are invariant to a global scale factor -- and
leaving their CSVs untouched keeps them as the control that says the deep arms moved for
numerical reasons. On the old grid, Spearman(log amplitude, score) is +0.929 (p = 0.0009)
for the deep arms and +0.119 (p = 0.78) for the classic ones.

    python benchmarks/slurm/make_dlrescale_points.py --out benchmarks/slurm/dlrescale.txt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

# MOABB spells datasets one way in its result CSVs and Hydra configs another.
DATASET_CONFIG = {
    "AlexandreMotorImagery": "alexmi",
    "BNCI2014-001": "bnci2014_001",
    "BNCI2014-002": "bnci2014_002",
    "BNCI2014-004": "bnci2014_004",
    "BNCI2015-001": "bnci2015_001",
    "Cho2017": "cho2017",
    "Lee2019-MI": "lee2019_mi",
    "PhysionetMotorImagery": "physionetmi",
    "Schirrmeister2017": "schirrmeister2017",
    "Shin2017A": "shin2017a",
    "Weibo2014": "weibo2014",
    "Zhou2016": "zhou2016",
}

# Points are ordered so the array's first tasks are the ones that settle the question:
# the two lowest-amplitude datasets, where the volt-scale collapse should be largest
# (BNCI2014-001 3.2e-06 with deep 0.483 against classic 0.564; Shin2017A 4.8e-06 with
# 0.573 against 0.751). If the rescale does nothing there, it does nothing anywhere, and
# that is worth knowing after one hour rather than five days.
FIRST = ("bnci2014_001", "shin2017a", "schirrmeister2017", "weibo2014")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results")
    ap.add_argument("--out", default="benchmarks/slurm/dlrescale.txt")
    a = ap.parse_args(argv)

    files = glob.glob(f"{a.results}/**/*.csv", recursive=True)
    if not files:
        raise SystemExit(f"no CSVs under {a.results}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dl = df[df.pipeline.str.startswith(("bd_", "grow_"))].copy()

    dl["align"] = dl["align"].fillna("none").astype(str)
    unknown = sorted(set(dl.dataset) - set(DATASET_CONFIG))
    if unknown:
        raise SystemExit(f"unmapped dataset names: {unknown}")
    dl["cfg"] = dl.dataset.map(DATASET_CONFIG)

    pts = (dl[["eval", "cfg", "pipeline", "seed", "align"]]
           .drop_duplicates()
           .rename(columns={"cfg": "dataset", "pipeline": "model"}))
    pts["seed"] = pts.seed.astype(int)
    pts["prio"] = pts.dataset.map(
        {d: i for i, d in enumerate(FIRST)}).fillna(len(FIRST)).astype(int)
    pts = pts.sort_values(["prio", "eval", "dataset", "model", "seed", "align"])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(
        f"{r.eval} {r.dataset} {r.model} {r.seed} {r.align}\n"
        for r in pts.itertuples()))
    print(f"{len(pts)} points -> {out}")
    print(pts.groupby("align").size().to_string())
    print(pts.groupby("eval").size().to_string())
    print(f"les {int((pts.prio < len(FIRST)).sum())} premiers points portent sur "
          f"{list(FIRST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
