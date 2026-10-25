"""Campaign A's data: build its cache from the raw files, verify it, and write the manifest.

WHY A FRESH CACHE (25/10/2026). The first manifest, read from the shared cache
``/scratch/amounir/moabb_cache``, found bnci2014_004 subject 1 serving 1 session of 5
and physionetmi subject 7 serving 30 trials of 45, with no error: MOABB's BIDS cache
returns whatever sessions / runs it finds under the key (bids_interface ``load``), and
``check_cache.py`` only sees whole missing sessions. A cache that can be silently
partial cannot be repaired in place with confidence, so campaign A reads its own cache,
derived here from the raw files, one process per dataset, and verified:

  --build    the target cache must hold nothing for this dataset; each subject is
             derived from raw and saved (the campaign's cache_config), then re-read
             from the cache and required to be bit-identical to what was derived
  --compare  the old cache is read too: trial counts and bit-identity per session go
             in the manifest (``old_cache``), i.e. which subjects earlier campaigns
             read truncated

The manifest then tells the cell generator what every subject serves: sessions, runs,
trials per class (MOABB's within-session 5-fold needs >= 5 per class per session), the
input shape, and the amplitude scale. Built through ``check_cache.build_campaign`` --
the single construction of the campaign's dataset and paradigm (same kwargs, same
Lee2019 session fix, same float fmin/fmax), hence the same cache key as the runner.

    python campaignA_manifest.py --dataset bnci2014_001 --cache /scratch/amounir/moabb_cache_A \
        --build --compare /scratch/amounir/moabb_cache --out ../manifest/bnci2014_001.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
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


def same(a, b):
    return sorted(a) == sorted(b) and all(
        np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1])
        and np.array_equal(a[k][2], b[k][2]) for k in a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--compare", default=None, help="an older cache to compare with")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from moabb.datasets.bids_interface import get_bids_root
    from omegaconf import OmegaConf

    from check_cache import build_campaign
    from utils import cache_config

    ds, par, keys = build_campaign(a.dataset)
    cc = cache_config(OmegaConf.create({"enabled": True, "path": a.cache}))
    ro = cache_config(OmegaConf.create({"enabled": True, "path": a.cache, "readonly": True}))
    target = Path(a.cache) / get_bids_root(ds.code, None).name
    if a.build and target.exists() and any(target.rglob("*_lockfile.json")):
        print(f"REFUSED: --build needs an empty target, {target} already holds entries")
        return 1
    old = (cache_config(OmegaConf.create({"enabled": True, "path": a.compare, "readonly": True}))
           if a.compare else None)
    rec = {"dataset": a.dataset, "moabb_class": type(ds).__name__, "cache": a.cache,
           "cache_keys": keys, "built": a.build, "subjects": {}, "n_chans": set(),
           "n_times": set(), "classes": set()}
    ok = True
    t0 = time.perf_counter()
    for s in ds.subject_list:
        t = time.perf_counter()
        if a.build:
            built = by_session(*par.get_data(dataset=ds, subjects=[s], cache_config=cc))
        X, y, meta = par.get_data(dataset=ds, subjects=[s], cache_config=ro)
        cur = by_session(X, y, meta)
        sub = {"n": int(len(y)), "x_std": float(np.std(X)), "sessions": {
            k: {"n": int(len(v[1])), "runs": sorted(map(str, np.unique(v[2]))),
                "per_class": dict(collections.Counter(map(str, v[1])))} for k, v in cur.items()}}
        if a.build:
            sub["reread_bit_identical"] = same(built, cur)
            ok &= sub["reread_bit_identical"]
            del built
        if old is not None:
            o = by_session(*par.get_data(dataset=ds, subjects=[s], cache_config=old))
            sub["old_cache"] = {"n": int(sum(len(v[1]) for v in o.values())),
                                "sessions": {k: int(len(v[1])) for k, v in o.items()},
                                "bit_identical": same(o, cur)}
            del o
        rec["n_chans"].add(int(X.shape[1]))
        rec["n_times"].add(int(X.shape[2]))
        rec["classes"].update(map(str, y))
        sub["load_s"] = round(time.perf_counter() - t, 1)
        rec["subjects"][str(s)] = sub
        oc = sub.get("old_cache")
        print(f"{a.dataset} sub {s}: {len(y)} trials, sessions {sorted(cur)}, std {sub['x_std']:.3g}"
              + (f", reread ok {sub['reread_bit_identical']}" if a.build else "")
              + (f", old cache {oc['n']} identical {oc['bit_identical']}" if oc else "")
              + f", {sub['load_s']}s", flush=True)
        del X, y, meta, cur
    for k in ("n_chans", "n_times", "classes"):
        rec[k] = sorted(rec[k])
    rec["total_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    diff = [s for s, v in rec["subjects"].items() if v.get("old_cache") and not v["old_cache"]["bit_identical"]]
    print(f"wrote {a.out}: {len(rec['subjects'])} subjects, chans {rec['n_chans']}, times "
          f"{rec['n_times']}, classes {rec['classes']}; differ from the old cache: {diff}")
    if not ok:
        print("BUILD FAILED: a re-read differs from what was derived")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
