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

    # How many co-tenants a device holds, from the measured peak rather than from
    # the "less than a gigabyte" everyone assumes. The turing nodes here are RTX
    # 2080 Ti at 11 GB, not the 80 GB A100 the reference script was written for,
    # so this bound is real and not academic.
    peaks = [r.get("torch_peak_reserved_mib") or 0 for r in recs]
    if any(peaks):
        peak = max(peaks)
        print(f"\npeak reserved: {peak} MiB -> on an 11 GB card, "
              f"K <= {int(11000 * 0.85 // max(peak, 1))} co-tenants "
              f"(15 % kept back for fragmentation and the CUDA context)")

    # The seeds/models that share a dataset pay the preprocessing once if it is
    # cached and 70 times if it is not. That ratio is the whole argument.
    cold = next((r for r in recs if r["cache"] and r["seed"] == 0), None)
    warm = next((r for r in recs if r["cache"] and r["seed"] == 1), None)
    none = next((r for r in recs if not r["cache"]), None)
    if cold and warm:
        saved = cold["preprocess_all_subjects_s"] - warm["preprocess_all_subjects_s"]
        print(f"cache saves {saved:.1f} s of preprocessing per job "
              f"({cold['preprocess_all_subjects_s']:.1f} -> "
              f"{warm['preprocess_all_subjects_s']:.1f} s)")
        if none:
            print(f"control (no cache): {none['preprocess_all_subjects_s']:.1f} s "
                  "-- if this is close to the warm figure the saving was the OS "
                  "page cache, not cache_config")
        print(f"over a 14-model x 5-seed grid on this dataset: "
              f"{69 * saved / 3600:.1f} h saved")


if __name__ == "__main__":
    main()
