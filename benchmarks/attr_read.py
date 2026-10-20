"""Read the 20/10 attribution campaign (benchmarks/attr_cells.txt, checkout eegrow_attr).

Written BEFORE any result was read. Unit = subject (n=9): each arm's score is averaged
over sessions and seeds per subject, contrasts are paired by subject, 95 % CI by
bootstrap over subjects (10 000 resamples, seed 0).

Contrasts, each tied to one question:
  Q1 dose     F_n{1,4,64,1024} and F_m{025,1,4,16} vs P_n4 (and vs REF)
  Q2 norm     F_m* vs F_n* -- does normalisation hurt at a given dose, or only via dose?
  Q3 scale    R_m16 - F_m16, R_n1024 - F_n1024, R_m64 - P_m64: rescaling keeps the
              direction and removes the scale; recovery => the scale is the culprit
  Q4 search   P_n64, P_n4096, P_m1, P_m64 vs P_n4: what the line-search protocols cost
  MSVTNet     M_n64, M_n4096 vs the existing variant-A cells (read separately)
Mechanism traces (EEGROW_DEBUG_GROWTH logs): conv_spat ||w||_F at the end of each fit
and its relative step du per epoch after growth stops; valid_acc curve from fits.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import statistics as st

DS = "bnci2014_001"


def subject_scores(root, label, arm):
    per = collections.defaultdict(list)
    for f in glob.glob(f"{root}/step0_results/{label}/results/within_session/{DS}/{arm}__seed*.csv"):
        for r in csv.DictReader(open(f)):
            per[r["subject"]].append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in per.items()}, {k: len(v) for k, v in per.items()}


def paired(a, b, n_boot=10000):
    ks = sorted(set(a) & set(b), key=int)
    d = [a[k] - b[k] for k in ks]
    if not d:
        return None
    rng = random.Random(0)
    bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    return {"n": len(d), "mean": sum(d) / len(d), "lo": bs[int(.025 * n_boot)],
            "hi": bs[int(.975 * n_boot) - 1], "n_neg": sum(x < 0 for x in d)}


DBG = re.compile(r"\[dbg\] ep=(\d+) epoch_end .*?j0s \S+ \S+ \S+ \S+ \|w\|=(\S+) fro=(\S+) du=(\S+)")


def traces(root, label):
    fits, cur = [], []
    for f in sorted(glob.glob(f"{root}/logs/{label}_*.out")):
        for line in open(f, errors="replace"):
            m = DBG.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            if cur and ep <= cur[-1][0]:
                fits.append(cur)
                cur = []
            du = None if m.group(4) == "na" else float(m.group(4))
            cur.append((ep, float(m.group(2)), float(m.group(3)), du))
        if cur:
            fits.append(cur)
            cur = []
    if not fits:
        return None
    fro_end = [f[-1][2] for f in fits]
    du_late = [x[3] for f in fits for x in f if x[0] > 100 and x[3] is not None]
    du_early = [x[3] for f in fits for x in f if 2 <= x[0] <= 5 and x[3] is not None]
    return {"n_fits": len(fits), "fro_end_med": st.median(fro_end),
            "du_early_med": st.median(du_early) if du_early else None,
            "du_late_med": st.median(du_late) if du_late else None}


def curves(root, label, arm, eps=(5, 10, 20, 30, 50, 100, 200)):
    acc = collections.defaultdict(list)
    s_all, ls_all = [], []
    for f in glob.glob(f"{root}/step0_results/{label}/results/within_session/{DS}/{arm}__seed*__fits.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = r["history"]
            for e in eps:
                if len(h) >= e:
                    acc[e].append(h[e - 1]["valid_acc"])
            for x in h:
                if x.get("grow_applied"):
                    s_all.append(float(x["grow_s"]))
                    if "grow_ls_s" in x:
                        ls_all.append(float(x["grow_ls_s"]))
    return ({e: st.median(v) for e, v in sorted(acc.items())},
            collections.Counter(s_all), collections.Counter(ls_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_attr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {lab: ("bd_shallow" if lab == "REF" else "grow_shallow") for lab in
            ["REF", "P_n4", "P_n64", "P_n4096", "P_m1", "P_m64", "F_n1", "F_n4", "F_n64",
             "F_n1024", "F_m025", "F_m1", "F_m4", "F_m16", "R_m16", "R_n1024", "R_m64"]}
    sc, cnt = {}, {}
    for lab, arm in arms.items():
        sc[lab], cnt[lab] = subject_scores(a.root, lab, arm)
    out = {"scores": {}, "contrasts": {}, "traces": {}, "curves": {}}
    lines = ["# Attribution campaign — reading", "",
             "| arm | n subj | rows/subj | mean | − P_n4 [95 % CI] (neg/9) | − REF [95 % CI] |",
             "|---|---|---|---|---|---|"]
    for lab in arms:
        s = sc[lab]
        if not s:
            lines.append(f"| {lab} | 0 | | | | |")
            continue
        m = sum(s.values()) / len(s)
        c1, c2 = paired(s, sc["P_n4"]), paired(s, sc["REF"])
        fmt = lambda c: "" if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ({c['n_neg']}/{c['n']})"
        rows = sorted(set(cnt[lab].values()))
        lines.append(f"| {lab} | {len(s)} | {rows} | {m:.3f} | {fmt(c1)} | {fmt(c2)} |")
        out["scores"][lab] = {"mean": m, "per_subject": s, "vs_P_n4": c1, "vs_REF": c2}
    lines += ["", "## Q3 rescale (scale vs direction): R − its unrescaled twin", ""]
    for r, f in [("R_m16", "F_m16"), ("R_n1024", "F_n1024"), ("R_m64", "P_m64")]:
        c = paired(sc[r], sc[f]) if sc[r] and sc[f] else None
        out["contrasts"][f"{r}-{f}"] = c
        lines.append(f"- {r} − {f}: " + ("n/a" if c is None else
                     f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ({c['n_neg']}/{c['n']} neg)"))
    lines += ["", "## Mechanism traces (conv_spat)", "",
              "| arm | fits | ‖w‖_F end (med) | du ep 2-5 (med) | du ep >100 (med) | applied s | line-search s (fixed arms) |",
              "|---|---|---|---|---|---|---|"]
    for lab in arms:
        if lab == "REF":
            continue
        t = traces(a.root, lab)
        cv, s_c, ls_c = curves(a.root, lab, arms[lab])
        out["traces"][lab], out["curves"][lab] = t, cv
        g = lambda k: "" if not t or t[k] is None else f"{t[k]:.3g}"
        lines.append(f"| {lab} | {t['n_fits'] if t else 0} | {g('fro_end_med')} | {g('du_early_med')} | "
                     f"{g('du_late_med')} | {dict(sorted(s_c.items()))} | {dict(sorted(ls_c.items()))} |")
    lines += ["", "## valid_acc median by epoch", "", "| arm | " + " | ".join(
        f"ep{e}" for e in (5, 10, 20, 30, 50, 100, 200)) + " |", "|---" * 8 + "|"]
    for lab, cv in out["curves"].items():
        lines.append(f"| {lab} | " + " | ".join(f"{cv.get(e, float('nan')):.3f}"
                                               for e in (5, 10, 20, 30, 50, 100, 200)) + " |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
