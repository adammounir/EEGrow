"""(1) ICASSP: CI half-width of grow - bd (paired by subject, seeds averaged) with seeds
0-2 vs 0-4. (2) Validation: where the smoothed valid_acc peaks, and how much is lost from
peak to the last epoch, per model x arm."""
import glob, json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def boot_hw(x):
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return (np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 2


d = pd.read_csv("read/icassp_all_scores.csv.gz")
d = d[d["align"] == "none"]
rows = []
for (ev, ds), g in d.groupby(["eval", "dataset"]):
    for back in ["shallow", "eegnex"]:
        gg = g[g["model"].isin([f"grow_{back}", f"bd_{back}"])]
        if gg.empty:
            continue
        out = {}
        for k, seeds in {"3": [0, 1, 2], "5": [0, 1, 2, 3, 4]}.items():
            p = gg[gg["seed"].isin(seeds)].groupby(["subject", "model"])["score"].mean().unstack()
            diff = (p[f"grow_{back}"] - p[f"bd_{back}"]).dropna().to_numpy()
            out[k] = (diff.mean(), boot_hw(diff), len(diff))
        rows.append((ev, ds, back, *out["3"][:2], *out["5"][:2], out["5"][2]))
r = pd.DataFrame(rows, columns=["eval", "dataset", "backbone", "d3", "hw3", "d5", "hw5", "n"])
r["hw_ratio_5_over_3"] = r["hw5"] / r["hw3"]
print("== (1) seeds 3 vs 5: CI half-width of grow - bd")
print(r.round(3).to_string())
print("median hw5/hw3 =", round(r["hw_ratio_5_over_3"].median(), 3),
      "| theoretical if seed noise dominated = %.3f" % np.sqrt(3 / 5))
print("sign changes 3 vs 5:", int((np.sign(r.d3) != np.sign(r.d5)).sum()), "/", len(r))
print("CI-excludes-0 changes:", int(((r.d3.abs() > r.hw3) != (r.d5.abs() > r.hw5)).sum()), "/", len(r))

print("\n== (2) validation learning curves (smoothed valid_acc, w=5)")
res = []
for f in glob.glob("valid_results/*__*/results/within_session/*/*__fits.jsonl"):
    label = f.split("/")[1]
    if not (label.endswith("__bd") or label.endswith("__tiny")):
        continue
    for line in open(f):
        try:
            r_ = json.loads(line)
        except json.JSONDecodeError:
            continue
        va = np.array([h["valid_acc"] for h in r_["history"]], float)
        tl = np.array([h["train_loss"] for h in r_["history"]], float)
        if len(va) < 200:
            continue
        sm = np.convolve(va, np.ones(5) / 5, mode="valid")
        pk = int(sm.argmax())
        res.append({"label": label, "peak_epoch": pk + 3, "peak_last20": pk + 3 > 180,
                    "drop_peak_to_end": sm.max() - sm[-1],
                    "end_minus_ep100": sm[-1] - sm[100 - 3],
                    "train_loss_end": tl[-1], "train_loss_rel_slope_last20": (tl[-20] - tl[-1]) / max(tl[-20], 1e-12)})
e = pd.DataFrame(res)
print(e.groupby("label").agg(n=("peak_epoch", "size"), peak_med=("peak_epoch", "median"),
                             frac_peak_last20=("peak_last20", "mean"),
                             drop_med=("drop_peak_to_end", "median"),
                             end_vs_100=("end_minus_ep100", "median"),
                             tl_end=("train_loss_end", "median"),
                             tl_still_falling=("train_loss_rel_slope_last20", "median")).round(3).to_string())
