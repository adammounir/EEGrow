"""Did the two arms of each grow/bd pair see the same preprocessing?

A paired comparison only isolates growth if grow_X and bd_X ran on identical data.
That was not guaranteed: the production grid passed ``dataset.resample=250`` on the
CLI, but the retry scripts omitted it, so any relaunched cell silently fell back to
the dataset's native rate (500 Hz on schirrmeister2017, 1000 Hz on lee2019_mi, ...).
At a different rate a fixed 25-sample temporal kernel spans a different duration, so
such a pair measures preprocessing, not growth.

The result CSVs record no sampling rate, so we rebuild it from hydra's run
directories: ``overrides.yaml`` gives the cell identity (eval/dataset/model/seed),
``config.yaml`` the resolved ``dataset.resample``. When a cell was launched several
times the most recent run wins, because the CSV stem is ``<model>__seed<N>.csv`` and
gets overwritten (``suffix`` only namespaces MOABB's hdf5 cache, not the CSV).
"""

import glob
import os
import re
import collections


def _overrides(path):
    ov = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        if "=" in line:
            k, v = line.split("=", 1)
            ov[k.strip()] = v.strip()
    return ov


def _resolved_resample(path):
    txt = open(path).read()
    block = re.search(r"^dataset:\n((?:[ \t]+.*\n|\n)*)", txt, re.M)
    if not block:
        return "UNKNOWN"
    m = re.search(r"^\s+resample:\s*(.*)$", block.group(1), re.M)
    if not m:
        return "UNKNOWN"
    v = m.group(1).strip()
    return "NATIVE" if v in ("null", "None", "") else v


def cell_regimes(bench_root="."):
    """(eval, dataset, model, seed) -> sampling rate of the run that wrote the CSV."""
    launches = collections.defaultdict(list)
    trees = [
        (os.path.join(bench_root, "multirun", "*", "*") + os.sep, -3),
        (os.path.join(bench_root, "outputs", "*") + os.sep, -2),
    ]
    for pattern, ts_idx in trees:
        for d in glob.glob(pattern):
            ov_p, cf_p = d + ".hydra/overrides.yaml", d + ".hydra/config.yaml"
            if not (os.path.exists(ov_p) and os.path.exists(cf_p)):
                continue
            ov = _overrides(ov_p)
            if not {"eval", "dataset", "model", "seed"} <= set(ov):
                continue
            if "train.max_epochs" in ov or "dataset.subjects" in ov:
                continue  # smoke test, never wrote a production CSV
            ts = d.rstrip(os.sep).split(os.sep)[ts_idx]
            key = (ov["eval"], ov["dataset"], ov["model"], ov["seed"])
            launches[key].append((ts, _resolved_resample(cf_p)))
    return {k: max(v)[1] for k, v in launches.items()}


def mixed_pairs(regimes, pairs):
    """Pairs whose two arms ran at different rates -- not comparable."""
    bad = []
    for grow, fixed in pairs:
        for (ev, ds, m, sd) in regimes:
            if m != grow:
                continue
            rg = regimes[(ev, ds, grow, sd)]
            rf = regimes.get((ev, ds, fixed, sd))
            if rf is not None and rg != rf:
                bad.append((grow, fixed, ev, ds, sd, rg, rf))
    return sorted(bad)


def assert_paired(pairs, bench_root=".", allow_env="EEGROW_ALLOW_MIXED"):
    """Abort the analysis if any pair straddles two sampling rates."""
    regimes = cell_regimes(bench_root)
    if not regimes:
        print("[regime] aucun repertoire hydra trouve -- garde-fou inactif")
        return regimes, []
    bad = mixed_pairs(regimes, pairs)
    rates = collections.Counter(regimes.values())
    print(f"[regime] {len(regimes)} cellules datees, taux: "
          + ", ".join(f"{k}={v}" for k, v in rates.most_common()))
    if not bad:
        print("[regime] OK -- les deux bras de chaque paire ont le meme prétraitement")
        return regimes, bad
    print(f"[regime] {len(bad)} PAIRES A CHEVAL SUR DEUX TAUX -- non comparables :")
    for grow, fixed, ev, ds, sd, rg, rf in bad:
        print(f"  {ds:20s} {ev:15s} seed={sd}  {grow}={rg}  {fixed}={rf}")
    if os.environ.get(allow_env) != "1":
        raise SystemExit(
            f"\nAnalyse interrompue : {len(bad)} paires comparent deux pretraitements.\n"
            f"Relancer ces cellules a 250 Hz, ou forcer avec {allow_env}=1 "
            f"(les deltas concernes ne mesurent alors plus la croissance)."
        )
    print(f"[regime] {allow_env}=1 -> on continue malgre tout")
    return regimes, bad
