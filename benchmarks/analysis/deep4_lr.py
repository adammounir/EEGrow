"""Read out the bd_deep4 learning-rate experiment (see benchmarks/exp_deep4_lr.py).

Primary endpoint: the final TRAIN loss, from the per-fit JSONL records. The diagnosis
being tested is that `bd_deep4` cannot fit its own training data, so the training loss
is the measurement -- held-out accuracy is a consequence, and a noisier one.

Secondary: held-out accuracy, paired across subject-sessions against the `constant`
arm. Paired because the units are the same sessions in every arm, which is where the
power is; the unpaired spread across subjects is several times the effect we are
looking for.

Usage: python deep4_lr.py --root <exp output dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ["constant", "lowlr", "cosine"]


def load_fits(root: Path) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for p in sorted((root / arm).rglob("bd_deep4__seed*__fits.jsonl")):
            seed = int(p.stem.split("seed")[1].split("__")[0])
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                h = r.get("history") or []
                if not h:
                    continue
                tr = [e["train_loss"] for e in h if e.get("train_loss") is not None]
                if not tr:
                    continue
                d = np.diff(tr)
                rows.append(dict(
                    arm=arm, seed=seed, fit=r.get("fit"), epochs=len(h),
                    final_tr=tr[-1], min_tr=float(np.min(tr)),
                    # The instability statistic the diagnosis is built on, recomputed
                    # here on the new runs so the comparison is like for like.
                    up_frac=float((d > 0).mean()) if len(d) else np.nan,
                    worst_up=float(d.max()) if len(d) else np.nan,
                    lr_final=h[-1].get("lr"),
                    stop_reason=r.get("stop_reason"),
                    restored_epoch=r.get("restored_epoch")))
    return pd.DataFrame(rows)


def load_scores(root: Path) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for p in sorted((root / arm).rglob("bd_deep4__seed*.csv")):
            if p.name.endswith("__fits.jsonl"):
                continue
            d = pd.read_csv(p)
            d["arm"] = arm
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    fits = load_fits(args.root)
    if fits.empty:
        print("no fit records found"); return 1
    print("=" * 74)
    print("PRIMARY: can the net fit its own training data?")
    print("=" * 74)
    print(f"(chance loss for 4 classes = ln(4) = {np.log(4):.3f}; "
          f"fix_deepeeg reaches 0.029 here)")
    g = (fits.groupby("arm")
             .agg(n_fits=("final_tr", "size"), epochs=("epochs", "median"),
                  final_tr=("final_tr", "median"), min_tr=("min_tr", "median"),
                  up_frac=("up_frac", "median"), worst_up=("worst_up", "median"),
                  lr_final=("lr_final", "median"))
             .reindex([a for a in ARMS if a in set(fits.arm)]))
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))

    base = fits[fits.arm == "constant"].set_index(["seed", "fit"])["final_tr"]
    for arm in ARMS[1:]:
        sub = fits[fits.arm == arm].set_index(["seed", "fit"])["final_tr"]
        common = base.index.intersection(sub.index)
        if len(common) < 5:
            continue
        d = sub.loc[common] - base.loc[common]
        w = stats.wilcoxon(d) if d.abs().sum() > 0 else None
        print(f"\n  {arm} - constant, paired on {len(common)} fits: "
              f"median {d.median():+.4f}"
              + (f", Wilcoxon p={w.pvalue:.2g}" if w else ""))

    sc = load_scores(args.root)
    if sc.empty:
        print("\n(no score CSVs yet)"); return 0
    print()
    print("=" * 74)
    print("SECONDARY: held-out accuracy")
    print("=" * 74)
    unit = ["dataset", "subject", "session", "seed"]
    print(sc.groupby("arm")["score"].agg(["size", "mean", "std"])
            .reindex([a for a in ARMS if a in set(sc.arm)])
            .to_string(float_format=lambda v: f"{v:.4f}"))
    b = sc[sc.arm == "constant"].set_index(unit)["score"]
    for arm in ARMS[1:]:
        s = sc[sc.arm == arm].set_index(unit)["score"]
        common = b.index.intersection(s.index)
        if len(common) < 5:
            continue
        d = s.loc[common] - b.loc[common]
        w = stats.wilcoxon(d) if d.abs().sum() > 0 else None
        print(f"  {arm} - constant, paired on {len(common)} units: "
              f"mean {d.mean():+.4f}, median {d.median():+.4f}"
              + (f", Wilcoxon p={w.pvalue:.2g}" if w else ""))
    # The threshold this cell has to clear at all, from the chance audit: 288 trials,
    # 4 classes, exact binomial -> 0.295. A gain that lands under it is not a result.
    print("\n  (exact 95% chance threshold for this cell: 0.295; v5 scored 0.272)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
