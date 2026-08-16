"""Fill the MOABB cache, serially, under the key the campaign actually reads.

WHY A DEDICATED WARMER
----------------------
``profile_host_ram.py`` was used for this and it was the wrong tool. Its job is to
measure host RSS, so it builds its own paradigm -- with ``fmin=8, fmax=32`` as ints
where the runner passes floats. Those render as ``Band Pass Filter (8–32 Hz)`` and
``Band Pass Filter (8.0–32.0 Hz)``, hash differently, and therefore key two disjoint
caches in the same directory. Every "warm-up" filled the one the campaign never reads,
so every packed run in fact started cold: K tenants missed the same entry at the same
moment, all derived it, all wrote it, and collided -- and MOABB's error path
(``interface.erase()`` -> ``shutil.rmtree``) deletes the entry from the *shared*
directory when that happens. 32 cells of the first packed campaign and 19 physionetmi
sessions were lost to exactly this.

So the warmer does not build a paradigm. It imports ``check_cache.build_campaign``, the
same function the gate checks against, which makes drift between "what we warm" and
"what we verify" impossible by construction.

WHY SERIAL, AND WHY PER SUBJECT
-------------------------------
Serial because one writer never races another; the whole failure mode above needs two.
Per subject because that is MOABB's unit of caching, so a crash on one subject costs
that subject rather than the dataset, and because it lets this print progress against a
long job -- a warm-up of the full grid is hours, and a silent hours-long job is
indistinguishable from a hung one.

Already-complete subjects are skipped on the *checked* state (lockfile AND data present
under both digests), not on a marker of our own, so an interrupted warm-up resumes and
a subject whose data was destroyed is genuinely redone.

USAGE
-----
    python benchmarks/warm_cache.py --cache /scratch/amounir/moabb_cache
    python benchmarks/warm_cache.py --cache ... --datasets cho2017,schirrmeister2017
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_cache import DATASETS, build_campaign  # noqa: E402

SUBSES = re.compile(r"^sub-([^_]+)_ses-([^_]+)_")
DESC = re.compile(r"desc-([0-9a-f]+)")


def _state(base: Path, keys: dict) -> tuple[set, set]:
    """``(locks, data)`` as (subject, session, digest) triples restricted to ``keys``."""
    locks, data = set(), set()
    code = base / "code"
    if code.is_dir():
        for f in code.glob("*_lockfile.json"):
            m, d = SUBSES.match(f.name), DESC.search(f.name)
            if m and d and d.group(1) in keys:
                locks.add((m.group(1), m.group(2), d.group(1)))
    if base.is_dir():
        for f in base.rglob("*"):
            if (f.is_file() and "code" not in f.parts
                    and f.suffix in (".fif", ".npy")):
                m, d = SUBSES.match(f.name), DESC.search(f.name)
                if m and d and d.group(1) in keys:
                    data.add((m.group(1), m.group(2), d.group(1)))
    return locks, data


def warm(name: str, cache: Path, force: bool) -> None:
    from moabb.datasets.bids_interface import get_bids_root

    from omegaconf import OmegaConf

    from utils import cache_config

    ds, par, keys = build_campaign(name)
    base = cache / get_bids_root(ds.code, None).name
    # The runner's own cache_config, not a reconstruction of it. Same reasoning as for
    # the paradigm: a second copy of a configuration is a second thing that can drift.
    ccfg = cache_config(OmegaConf.create({"enabled": True, "path": str(cache)}))

    print(f"\n=== {name}  clés={' '.join(f'{h[:8]}:{k}' for h, k in keys.items())}",
          flush=True)
    locks, data = _state(base, keys)
    ok_subjects = set()
    if not force:
        # Sessions are DISCOVERED (their names are not 0..n-1: bnci2014_004 has
        # ``0train``/``3test``, shin2017a ``0imagery``), then every discovered session
        # is required under EVERY key. Deriving the expected set from what is on disk
        # instead -- i.e. iterating the (session, digest) pairs found -- silently
        # excuses the very entries that are missing, since a pair absent from both
        # locks and data never appears to be asked for. That bug marked zhou2016
        # complete while its epochs entries for sub-4 were gone.
        sessions: dict[str, set] = {}
        for s, e, _ in locks | data:
            sessions.setdefault(s, set()).add(e)
        for s, sess in sessions.items():
            if all((s, e, h) in locks and (s, e, h) in data
                   for e in sess for h in keys):
                ok_subjects.add(s)

    todo = [s for s in ds.subject_list if str(s) not in ok_subjects]
    print(f"    {len(ds.subject_list)} sujets, {len(todo)} à chauffer", flush=True)
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            par.get_data(dataset=ds, subjects=[s], cache_config=ccfg)
            msg = "ok"
        except Exception as exc:                    # a warmer reports, never crashes
            msg = f"ÉCHEC {type(exc).__name__}: {str(exc)[:140]}"
        print(f"    [{i}/{len(todo)}] sub-{s}  {time.time() - t0:7.1f}s  {msg}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/scratch/amounir/moabb_cache")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--force", action="store_true",
                    help="re-derive every subject, even those already complete")
    args = ap.parse_args()

    t0 = time.time()
    for name in args.datasets.split(","):
        try:
            warm(name, Path(args.cache), args.force)
        except Exception as exc:
            print(f"=== {name}: ÉCHEC GLOBAL {type(exc).__name__}: {exc}", flush=True)
    print(f"\n=== terminé en {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
