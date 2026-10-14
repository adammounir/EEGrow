"""Read the live fits against criteria A (fidelity), B (edges), C (utility)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def pick(acc, rule):
    acc = np.asarray(acc, float)
    if rule == "orig":
        return int(np.argmax(acc)) + 1
    if rule == "late":
        return int(np.where(acc == acc.max())[0][-1]) + 1
    if rule == "last":
        return len(acc)
    w = int(rule[2:])
    m = pd.Series(acc).rolling(w, center=True, min_periods=w // 2 + 1).mean().round(10)
    return int(np.nanargmax(m.to_numpy())) + 1


RULES = ["orig", "late", "sm5", "sm11", "last"]
rng = np.random.default_rng(0)
for path in sorted(HERE.glob("fits_*.jsonl")):
    rows = []
    bad = []
    for line in open(path):
        r = json.loads(line)
        va, ta, E = r["valid_acc"], r["test_acc"], r["epochs"]
        row = dict(subject=r["subject"], session=r["session"], fold=r["fold"], E=E,
                   oracle=max(ta))
        for k in RULES:
            e = pick(va, k)
            row[f"ep_{k}"] = e
            row[f"te_{k}"] = ta[e - 1]
        # A: live restore vs offline rule, and the returned weights vs the curve.
        live = r["restored_epoch"] if r["restored_epoch"] is not None else E
        if live != row["ep_sm5"] or abs(r["final_test_acc"] - ta[live - 1]) > 1e-9:
            bad.append((r["subject"], r["session"], r["fold"], live, row["ep_sm5"],
                        r["final_test_acc"], ta[live - 1]))
        rows.append(row)
    f = pd.DataFrame(rows)
    print(f"\n===== {path.stem}  ({len(f)} fits, {f.subject.nunique()} subjects)")
    print(f"A fidelity: {len(f) - len(bad)}/{len(f)} exact"
          + ("" if not bad else f"  MISMATCHES {bad[:5]}"))
    print("B edges (share of fits)       " + "  ".join(f"{k:>6s}" for k in RULES[:4]))
    print("  scored < epoch 5            " + "  ".join(
        f"{(f[f'ep_{k}'] < 5).mean():6.1%}" for k in RULES[:4]))
    print("  scored >= budget-5          " + "  ".join(
        f"{(f[f'ep_{k}'] >= f.E - 5).mean():6.1%}" for k in RULES[:4]))
    print("  median scored epoch         " + "  ".join(
        f"{f[f'ep_{k}'].median():6.0f}" for k in RULES[:4]))
    s = f.groupby("subject")[[f"te_{k}" for k in RULES] + ["oracle"]].mean()
    print("C test acc (subject mean)     " + "  ".join(
        f"{k:>6s}" for k in RULES + ["oracle"]))
    print("                              " + "  ".join(
        f"{s[c].mean():6.3f}" for c in s.columns))
    for a, b in [("sm5", "orig"), ("sm5", "late"), ("sm5", "last"), ("sm11", "sm5")]:
        d = (s[f"te_{a}"] - s[f"te_{b}"]).to_numpy()
        boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {a:>4s} - {b:<4s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{(d > 0).sum()}/{len(d)} subjects up")
    f.to_csv(HERE / f"summary_{path.stem}.csv", index=False)
