"""Read the JSONs written by ``profile_cell.py`` and print the comparison.

The point of the table is to answer one question per column: does the epoch cache
remove the preprocessing, does a cell need a whole GPU, and does the device have
room for co-tenants.

    python benchmarks/profile_report.py [profile_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "profile"
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    if not recs:
        raise SystemExit(f"no profile JSON under {d}")

    hdr = (f"{'cell':38s}{'cache':>7s}{'prep s':>9s}{'eval s':>9s}"
           f"{'total s':>9s}{'prep%':>7s}{'util%':>7s}{'idle%':>7s}{'GPU MiB':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        g = r.get("gpu", {})
        cell = f"{r['eval']}/{r['dataset']}/{r['model']}"
        print(f"{cell[:37]:38s}"
              f"{str(r['cache']):>7s}"
              f"{r['preprocess_all_subjects_s']:9.1f}"
              f"{r['evaluate_s']:9.1f}"
              f"{r['total_s']:9.1f}"
              f"{100 * r['preprocess_share']:7.1f}"
              f"{g.get('util_mean', float('nan')):7.1f}"
              f"{100 * g.get('util_below_20pct', float('nan')):7.1f}"
              f"{r.get('torch_peak_reserved_mib') or g.get('mem_used_max_mib', 0):9.0f}")

    # How many co-tenants a device holds. What bounds this is *reserved*, not
    # allocated: the caching allocator keeps every block it has ever taken, so
    # the device sees the reserved figure even though the model only needs the
    # allocated one. The gap between the two columns is the whole packing
    # question on Margaret, whose turing nodes are RTX 2080 Ti at 11 GB rather
    # than the 80 GB A100 the reference script was written for.
    alloc = max((r.get("torch_peak_alloc_mib") or 0) for r in recs)
    resv = max((r.get("torch_peak_reserved_mib") or 0) for r in recs)
    if resv:
        budget = 11000 * 0.85  # 15 % back for fragmentation and the CUDA context
        print(f"\npeak allocated {alloc} MiB, peak reserved {resv} MiB "
              f"({resv / max(alloc, 1):.1f}x)")
        print(f"  on an 11 GB card: K <= {int(budget // resv)} at the reserved "
              f"figure, K <= {int(budget // max(alloc, 1))} if the reserved "
              f"memory can be brought down to the allocated one")

    # The baseline for the cache is the UNCACHED run, not the cold one: a cold
    # run pays the preprocessing *and* writes the cache, so cold-minus-warm
    # flatters the cache by charging it for work it did once and never repeats.
    cold = next((r for r in recs if r["cache"] and r["seed"] == 0), None)
    warm = next((r for r in recs if r["cache"] and r["seed"] == 1), None)
    none = next((r for r in recs if not r["cache"]), None)
    if warm and none:
        base = none["preprocess_all_subjects_s"]
        hot = warm["preprocess_all_subjects_s"]
        print(f"\npreprocessing per job: {base:.1f} s uncached -> {hot:.1f} s warm "
              f"({base / max(hot, 1e-9):.1f}x)")
        if cold:
            first = cold["preprocess_all_subjects_s"]
            print(f"  the first job pays {first:.1f} s to populate the cache "
                  f"({first - base:+.1f} s vs uncached)")
            # 70 jobs = 14 pipelines x 5 seeds, what a dataset actually gets.
            with_cache = first + 69 * hot
            without = 70 * base
            print(f"  over a 14-model x 5-seed grid on this dataset: "
                  f"{without / 60:.0f} min -> {with_cache / 60:.0f} min "
                  f"({(without - with_cache) / 60:.0f} min saved)")
        print(f"  share of the cell that is preprocessing, uncached: "
              f"{100 * none['preprocess_share']:.0f} %")


if __name__ == "__main__":
    main()
