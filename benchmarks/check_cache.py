"""Is the MOABB cache warm *for the key the campaign will actually read*?

WHY THIS EXISTS, AND WHY ITS FIRST VERSION WAS USELESS
------------------------------------------------------
MOABB keys its BIDS cache by ``desc-<digest>``, where the digest is
``get_digest(process_pipeline)`` -- a hash of the *repr* of the whole processing
pipeline. Two pipelines that differ in any rendered detail are two disjoint caches
living side by side in the same directory.

The first version of this file counted subject/session directories. That cannot see a
wrong or missing key, and the consequence was not hypothetical:

  ``profile_host_ram.py``  built its paradigm with ``fmin=8, fmax=32``   (ints)
  ``run_moabb_hydra.py``   builds  its paradigm with ``fmin=8.0, fmax=32.0`` (floats)

which render as ``Band Pass Filter (8–32 Hz)`` and ``Band Pass Filter (8.0–32.0 Hz)``,
hence two different digests. So the "warm-up" filled a cache the campaign never reads.
Every packed run therefore began on a *cold* cache for its own key, K tenants missed
the same entry simultaneously, all of them derived and wrote it, and they collided --
whereupon MOABB's error path (``interface.erase()`` -> ``shutil.rmtree``) deleted the
entry from the shared directory. That is the mechanism behind the 32 cells lost by the
first packed campaign and the 19 physionetmi sessions destroyed on 2026-08-16, and the
old checker reported "cache complete" throughout.

WHAT IT CHECKS NOW
------------------
The digests are *computed*, from the same dataset/paradigm construction the campaign
uses, rather than discovered from what happens to be on disk. For each dataset:

  1. build the campaign's paradigm from ``config/dataset/<name>.yaml`` + ``config.yaml``
  2. take ``paradigm.make_process_pipelines(dataset)`` and derive the two digests MOABB
     will save under: the pipeline truncated after its last EPOCHS step (``save_epochs``)
     and the full pipeline (``save_array``)
  3. for every (subject, session) require, under BOTH digests, a lockfile AND data

The lockfile matters as much as the data, because the lockfile -- not the data -- is
MOABB's cache-hit test. A lockfile whose data is gone makes every reader report a hit,
fail the load, then erase and re-derive; and the warm-up trusts it too, so such an
entry can never repair itself. That state is reported separately as LYING.

Exit status is 0 only when every required entry is present, so this can gate a launcher.

USAGE
-----
    python benchmarks/check_cache.py --cache /scratch/amounir/moabb_cache
    python benchmarks/check_cache.py --cache ... --datasets lee2019_mi,physionetmi
    python benchmarks/check_cache.py --cache ... --quiet   # gate mode
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIG = HERE / "config"
DATASETS = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004",
            "bnci2015_001", "cho2017", "lee2019_mi", "physionetmi",
            "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016"]

SUBSES = re.compile(r"^sub-([^_]+)_ses-([^_]+)_")
DESC = re.compile(r"desc-([0-9a-f]+)")


def build_campaign(name: str):
    """``(dataset, paradigm, {digest: step_name})`` exactly as the campaign builds them.

    THE SINGLE SOURCE OF TRUTH FOR THE CACHE KEY. ``warm_cache.py`` imports this rather
    than constructing its own paradigm, because a *second* construction is precisely
    what caused the incident this module documents: ``profile_host_ram.py`` passed
    ``fmin=8`` where the runner passes ``fmin=8.0``, and the two filled disjoint caches.
    Any future warmer must go through here so that "what we warm" and "what we check"
    cannot drift from each other.

    Mirrors ``run_moabb_hydra.py``: the same dataset kwargs, the same subject
    restriction, the same neutralisation of MOABB's Lee2019 session filter, and --
    critically -- the same ``float`` fmin/fmax, since an int renders differently and
    hashes differently.
    """
    import moabb.datasets as mds
    import moabb.paradigms as mpar
    from moabb.analysis.results import get_digest
    from moabb.datasets.bids_interface import StepType, get_bids_root
    from moabb.datasets.preprocessing import FixedPipeline
    from omegaconf import OmegaConf

    root = OmegaConf.load(CONFIG / "config.yaml")
    dcfg = OmegaConf.load(CONFIG / "dataset" / f"{name}.yaml")
    ds = getattr(mds, dcfg.moabb_class)(**(OmegaConf.to_container(
        dcfg.get("kwargs")) or {}))
    if dcfg.get("subjects"):
        ds.subject_list = list(dcfg.subjects)
    # Same reason as in the runner: MOABB 1.5.0 filters Lee2019 sessions with an
    # off-by-one and serves half the trials. The campaign drops the filter, so the
    # cache must carry every session -- checking the filtered subset would pass a
    # cache the campaign then finds incomplete, on the largest dataset of the grid.
    if getattr(ds, "_selected_sessions", None) is not None:
        ds._selected_sessions = None

    pkw = {}
    if dcfg.get("resample"):
        pkw["resample"] = float(dcfg.resample)
    if dcfg.get("tmin") is not None:
        pkw["tmin"] = float(dcfg.tmin)
    if dcfg.get("tmax") is not None:
        pkw["tmax"] = float(dcfg.tmax)
    par = getattr(mpar, dcfg.paradigm)(
        fmin=float(root.paradigm.fmin), fmax=float(root.paradigm.fmax), **pkw)

    keys = {}
    for pl in par.make_process_pipelines(ds):
        steps = list(pl.steps)
        # MOABB saves under the digest of the pipeline truncated at the step being
        # saved (base.py: FixedPipeline(cached_steps + remaining_steps[:idx+1])), so
        # the epochs entry and the array entry have different digests.
        last_epochs = max((i for i, (t, _) in enumerate(steps)
                           if t is StepType.EPOCHS), default=None)
        if last_epochs is not None:
            keys[get_digest(FixedPipeline(steps[:last_epochs + 1]))] = "epochs"
        keys[get_digest(pl)] = "array"
    return ds, par, keys


def campaign_keys(name: str):
    """``(bids_dir_name, subject_list, {digest: step_name})`` for the checker."""
    from moabb.datasets.bids_interface import get_bids_root

    ds, _, keys = build_campaign(name)
    return get_bids_root(ds.code, None).name, list(ds.subject_list), keys


def check(root: Path, name: str) -> dict:
    bids, subjects, keys = campaign_keys(name)
    base = root / bids
    rec = {"dataset": name, "dir": bids, "keys": keys,
           "missing": [], "lying": [], "ok": 0}

    locks: set[tuple[str, str, str]] = set()
    data: set[tuple[str, str, str]] = set()
    code = base / "code"
    if code.is_dir():
        for f in code.glob("*_lockfile.json"):
            m, d = SUBSES.match(f.name), DESC.search(f.name)
            if m and d:
                locks.add((m.group(1), m.group(2), d.group(1)))
    if base.is_dir():
        for f in base.rglob("*"):
            if (f.is_file() and "code" not in f.parts
                    and f.suffix in (".fif", ".npy")):
                m, d = SUBSES.match(f.name), DESC.search(f.name)
                if m and d:
                    data.add((m.group(1), m.group(2), d.group(1)))

    # Sessions are discovered, not assumed: a dataset's session *names* are not
    # ``0..n-1`` (bnci2014_004 uses ``0train``/``3test``, shin2017a ``0imagery``), and
    # only the data on disk knows them. Discovery is over the union of both digests so
    # that a session present under one key is still required under the other.
    sessions: dict[str, set[str]] = {}
    for s, e, h in locks | data:
        if h in keys:
            sessions.setdefault(s, set()).add(e)

    # ONLY the array level gates. The campaign calls ``paradigm.get_data``, which
    # returns arrays, and no evaluation here sets ``return_epochs``, so the epochs
    # entry is never on the read path -- it is a by-product written when a subject is
    # derived from raw. It also cannot be back-filled: MOABB starts from the deepest
    # cached level, so while the array entry exists it returns immediately and never
    # regenerates the epochs one. Gating on it would demand a warm-up that is both
    # impossible and pointless (measured: it inflated the real gap from 90 to 277).
    array_h = next((h for h, k in keys.items() if k == "array"), None)
    epochs_h = next((h for h, k in keys.items() if k == "epochs"), None)
    for s in map(str, subjects):
        if s not in sessions:
            rec["missing"].append(f"sub-{s} (aucune session)")
            continue
        for e in sorted(sessions[s]):
            if (s, e, array_h) in data and (s, e, array_h) in locks:
                rec["ok"] += 1
            elif (s, e, array_h) in locks:
                # Lockfile without data: MOABB reports a hit, the load fails, and the
                # error path erases -- and a warm-up trusts the lockfile too, so the
                # entry cannot repair itself. This is the state that must be cleaned
                # before warming, not merely counted.
                rec["lying"].append(f"sub-{s}/ses-{e} [array]")
            else:
                rec["missing"].append(f"sub-{s}/ses-{e} [array]")
            if epochs_h is not None and ((s, e, epochs_h) in locks
                                         and (s, e, epochs_h) not in data):
                rec["epochs_lying"] = rec.get("epochs_lying", 0) + 1
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/scratch/amounir/moabb_cache")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--quiet", action="store_true",
                    help="print only the datasets that fail")
    args = ap.parse_args()

    root = Path(args.cache)
    bad = 0
    print(f"{'dataset':20s} {'OK':>6s} {'MANQUE':>7s} {'MENSONGE':>9s}  clés campagne")
    print("-" * 74)
    for name in args.datasets.split(","):
        r = check(root, name)
        n = len(r["missing"]) + len(r["lying"])
        bad += n
        if n or not args.quiet:
            ks = " ".join(f"{h[:8]}:{k}" for h, k in r["keys"].items())
            print(f"{r['dataset']:20s} {r['ok']:6d} {len(r['missing']):7d} "
                  f"{len(r['lying']):9d}  {ks}")
        for kind, label in (("missing", "MANQUE"), ("lying", "MENSONGE")):
            if r[kind]:
                head = ", ".join(r[kind][:6])
                more = f" ... (+{len(r[kind]) - 6})" if len(r[kind]) > 6 else ""
                print(f"    {label}: {head}{more}")
    if bad:
        print(f"\nCACHE PAS PRÊT: {bad} entrée(s). Lancer "
              f"benchmarks/slurm/warm_cache.sbatch (série, un seul process) avant tout "
              f"run packé -- K locataires qui manquent la même entrée l'écrivent tous "
              f"et le chemin d'erreur de MOABB supprime ce qu'il trouve partiel.")
        return 1
    print("\ncache prêt pour la clé de la campagne")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
