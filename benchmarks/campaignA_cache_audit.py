"""Cache vs raw, per subject: does the MOABB cache serve every session the raw files hold?

Found on 25/10/2026 by the campaign-A manifest: under the campaign's cache key,
bnci2014_004 subject 1 serves ONE session (4test, 160 trials) where the others serve
five, and MOABB raises nothing -- its BIDS cache returns whatever sessions it finds for
the key. This script derives the given subjects from the raw files (no cache), compares
with the cached read (sessions, trial counts, and bit-identity of the common sessions),
and with ``--repair`` writes the missing sessions into the cache (overwrite flags on for
those subjects only: the common sessions are rewritten from the same raw with the same
pipeline, and the script checks afterwards that they are bit-identical to before).

    python campaignA_cache_audit.py --dataset bnci2014_004 --subjects 1 [--repair]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def by_session(X, y, meta):
    out = {}
    for se in np.unique(meta["session"].values):
        m = meta["session"].values == se
        out[str(se)] = (X[m], np.asarray(y)[m], meta["run"].values[m])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subjects", required=True, help="comma-separated")
    ap.add_argument("--cache", default="/scratch/amounir/moabb_cache")
    ap.add_argument("--repair", action="store_true")
    a = ap.parse_args()

    from omegaconf import OmegaConf

    from check_cache import build_campaign
    from utils import cache_config

    ds, par, _ = build_campaign(a.dataset)
    cc = cache_config(OmegaConf.create({"enabled": True, "path": a.cache}))
    ok = True
    for s in [type(ds.subject_list[0])(x) for x in a.subjects.split(",")]:
        cached = by_session(*par.get_data(dataset=ds, subjects=[s], cache_config=cc))
        raw = by_session(*par.get_data(dataset=ds, subjects=[s]))
        print(f"{a.dataset} sub {s}: cache {({k: len(v[1]) for k, v in cached.items()})} "
              f"raw {({k: len(v[1]) for k, v in raw.items()})}")
        for k in sorted(set(cached) & set(raw)):
            same = (np.array_equal(cached[k][0], raw[k][0]) and np.array_equal(cached[k][1], raw[k][1]))
            print(f"   session {k}: cached == raw bit-identical: {same}"
                  + ("" if same else f" (max |diff| {np.abs(cached[k][0] - raw[k][0]).max():.3g})"))
            ok &= same
        missing = sorted(set(raw) - set(cached))
        if missing:
            print(f"   MISSING from cache: {missing}")
            ok = False
            if a.repair:
                rc = dict(cc, overwrite_epochs=True, overwrite_array=True)
                par.get_data(dataset=ds, subjects=[s], cache_config=rc)
                after = by_session(*par.get_data(dataset=ds, subjects=[s], cache_config=cc))
                good = sorted(after) == sorted(raw) and all(
                    np.array_equal(after[k][0], raw[k][0]) and np.array_equal(after[k][1], raw[k][1])
                    for k in raw)
                print(f"   REPAIRED: cache now {({k: len(v[1]) for k, v in after.items()})}, "
                      f"bit-identical to raw on every session: {good}")
                ok = good
    print("AUDIT OK" if ok else "AUDIT FOUND DIFFERENCES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
