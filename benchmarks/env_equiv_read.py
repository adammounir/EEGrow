"""Reader of PREREG_env_equivalence.md (frozen with it). Usage:

    python env_equiv_read.py --root /scratch/amounir/eegrow_v2 --out ../read/env_equiv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

# + avenant 1 (25/10 soir): GTX 1080 Ti (margpu024-027) and V100-SXM2 (margpu032), same references
CONDS = ["ref_s0_rerun", "ref_s1", "b126_turing", "b126_volta", "b126_pascal", "b126_1080ti", "b126_v100sxm2"]
ARMS = {"bd": ("bd_shallow", 0), "tiny": ("grow_shallow", 0)}


def load(root, label, arm, seed):
    d = Path(root) / "equiv" / label / "results/within_session/bnci2014_001"
    csv = d / f"{arm}__seed{seed}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    acc = df.groupby(df["subject"].astype(str))["score"].mean()
    fits = [json.loads(l) for l in open(d / f"{arm}__seed{seed}__fits.jsonl")]
    width = np.mean([f["width_end"] for f in fits if f.get("width_end") is not None]) if arm.startswith("grow") else None
    cards = sorted({f.get("gpu_name") for f in fits})
    return dict(acc=acc, wall=float(df["time"].sum()), width=width, cards=cards, torch=df["v_torch"].iloc[0])


def boot(d, n=10_000):
    rng = np.random.default_rng(0)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["# Équivalence env/carte (bench126, volta, pascal) — lecture", "",
             "| bras | condition | carte | torch | dev | Δ signé [IC95] | temps / ref | largeur tiny | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    ok = {}
    for k, (arm, _) in ARMS.items():
        ref = load(a.root, f"ref_s0__{k}", arm, 0)
        s1 = load(a.root, f"ref_s1__{k}", arm, 1)
        if ref is None or s1 is None:
            lines.append(f"| {k} | référence incomplète | | | | | | | — |"); continue
        dev_seed = float((s1["acc"] - ref["acc"]).abs().mean())
        for c in CONDS:
            seed = 1 if c == "ref_s1" else 0
            r = load(a.root, f"{c}__{k}", arm, seed)
            if r is None:
                lines.append(f"| {k} | {c} | | | | | | | incomplet |"); continue
            d = (r["acc"] - ref["acc"]).dropna().to_numpy()
            dev = float(np.abs(d).mean()); lo, hi = boot(d)
            verdict = ""
            if c.startswith("b126"):
                good = dev <= dev_seed and lo <= 0 <= hi
                verdict = "ÉQUIVALENT" if good else "REFUSÉ"; ok[(c, k)] = good
            w = "" if r["width"] is None else f"{r['width']:.1f}"
            lines.append(f"| {k} | {c} | {', '.join(map(str, r['cards']))} | {r['torch']} | {dev:.4f} | "
                         f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | ×{r['wall'] / ref['wall']:.2f} | {w} | {verdict} |")
        lines.append(f"| {k} | (étalon : graine 0 → 1) | | | {dev_seed:.4f} | | | | |")
    lines.append("")
    env_ok = ok.get(("b126_turing", "bd")) and ok.get(("b126_turing", "tiny"))
    for c in ("b126_volta", "b126_pascal", "b126_1080ti", "b126_v100sxm2"):
        card_ok = env_ok and ok.get((c, "bd")) and ok.get((c, "tiny"))
        lines.append(f"- **{c.split('_')[1]}** : {'ENTRE dans la campagne A' if card_ok else 'N’ENTRE PAS'}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
