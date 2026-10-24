"""Read the MSVTNet 400-epoch test (PREREG_msvtnet_400_epochs.md). Frozen before read.

Usage::

    python msvt400_read.py --root /scratch/amounir/eegrow_valid --out ../read/msvt400
"""

from __future__ import annotations

import argparse
import json
import os

from ratio_read import widths
from valid_read import DATASETS, paired, scores


def ci(c):
    return "n/a" if not c else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, L = {}, ["# MSVTNet : 400 contre 200 époques — lecture", "",
                  "| dataset | Δ_bd [IC95] | Δ_tiny [IC95] | tiny−bd à 200 | tiny−bd à 400 | "
                  "tiny à la cible 200 / 400 | complet |", "|---|---|---|---|---|---|---|"]
    n_bridled = 0
    for ds, planned in DATASETS.items():
        s = {lab: scores(a.root, lab, ds) for lab in
             ("msvtnet__bd", "msvtnet__tiny", "msvtnet__bd_e400", "msvtnet__tiny_e400")}
        complete = all(v[0] and v[1] == planned for v in s.values())
        d_bd = paired(s["msvtnet__bd_e400"][0], s["msvtnet__bd"][0]) if complete else None
        d_t = paired(s["msvtnet__tiny_e400"][0], s["msvtnet__tiny"][0]) if complete else None
        c200 = paired(s["msvtnet__tiny"][0], s["msvtnet__bd"][0]) if complete else None
        c400 = paired(s["msvtnet__tiny_e400"][0], s["msvtnet__bd_e400"][0]) if complete else None
        w200, w400 = widths(a.root, "msvtnet__tiny", ds), widths(a.root, "msvtnet__tiny_e400", ds)
        fmt = lambda w: "—" if w["at_target"] is None else f"{w['at_target']:.0%}"
        n_bridled += bool(d_bd and d_bd["lo"] > 0)
        out[ds] = {"d_bd": d_bd, "d_tiny": d_t, "tiny_minus_bd_200": c200, "tiny_minus_bd_400": c400,
                   "w200": w200, "w400": w400, "complete": complete}
        L.append(f"| {ds} | {ci(d_bd)} | {ci(d_t)} | {ci(c200)} | {ci(c400)} | {fmt(w200)} / {fmt(w400)} | "
                 f"{'✅' if complete else '❌'} |")
    v = ("INCOMPLET" if not all(o["complete"] for o in out.values())
         else "BRIDÉ" if n_bridled >= 2 else "NON BRIDÉ")
    out["verdict"] = v
    L += ["", f"**Verdict (PREREG § 4) : {v}** ({n_bridled}/3 datasets avec borne basse de Δ_bd > 0)"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
