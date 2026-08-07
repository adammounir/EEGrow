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
times the most recent run *that actually wrote the CSV* wins, because the CSV stem is
``<label>__seed<N>.csv`` and gets overwritten (``suffix`` only namespaces MOABB's hdf5
cache, not the CSV).
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
    if v in ("null", "None", ""):
        return "NATIVE"
    # canonicalise: the dataset yaml pins ``250.0`` while the sbatch passes
    # ``dataset.resample=250`` on the CLI. Compared as strings those are two different
    # regimes and the guard would abort the analysis on a pair that is in fact
    # perfectly matched.
    try:
        return f"{float(v):g}"
    except ValueError:
        return v


def _resolved_label(path, fallback):
    """The stem the run used for its CSV.

    Not the same string as the hydra config key in the overrides: the classic-ML
    configs are keyed ``ml_csp_lda`` but carry ``label: csp_lda``, so matching on the
    override name alone finds no CSV and the cell drops out of the guard entirely --
    coverage lost silently, which is worse than a false alarm.
    """
    m = re.search(r"^model:\n(?:[ \t]+.*\n|\n)*?[ \t]+label:\s*(.*)$", open(path).read(),
                  re.M)
    return m.group(1).strip() if m else fallback


def _csv_mtime(bench_root, ev, ds, label, seed):
    """When the cell's result file was last written, or None if there is none."""
    p = os.path.join(bench_root, "results", ev, ds, f"{label}__seed{seed}.csv")
    return os.path.getmtime(p) if os.path.exists(p) else None


def cell_regimes(bench_root="."):
    """(eval, dataset, model, seed) -> sampling rate of the run that wrote the CSV.

    "Most recent" is decided on the mtime of the run's own ``config.yaml``, not on a
    timestamp parsed out of the directory path. The two hydra trees nest differently
    (``multirun/<date>/<overrides>/`` vs ``outputs/<date>/``), so a fixed negative
    index picked the *tree name* -- a constant. ``max()`` then never compared dates at
    all and fell through to the second tuple element, the rate string, where
    ``"NATIVE" > "250"``: every cell that had ever run at the native rate was reported
    as native forever, including the ones the 250 Hz relaunch had already fixed.

    A launch only counts if it actually *wrote* the CSV. hydra writes ``config.yaml``
    when the run starts and we write the CSV when it ends, so a successful run always
    has ``mtime(config) <= mtime(csv)``; a run that crashed leaves its config behind
    but never touches the CSV, which still holds the older result. Taking the plain
    latest launch would therefore credit a failed 250 Hz relaunch with fixing a cell
    whose CSV is still at the native rate -- silently the wrong way round.
    """
    launches = collections.defaultdict(list)
    trees = [
        os.path.join(bench_root, "multirun", "*", "*") + os.sep,
        os.path.join(bench_root, "outputs", "*") + os.sep,
    ]
    for pattern in trees:
        for d in glob.glob(pattern):
            ov_p, cf_p = d + ".hydra/overrides.yaml", d + ".hydra/config.yaml"
            if not (os.path.exists(ov_p) and os.path.exists(cf_p)):
                continue
            ov = _overrides(ov_p)
            if not {"eval", "dataset", "model", "seed"} <= set(ov):
                continue
            if "train.max_epochs" in ov or "dataset.subjects" in ov:
                continue  # smoke test, never wrote a production CSV
            # key on the resolved label, not the override name: that is what lands in
            # the CSV's ``model`` column and what the PAIRS of the analyses are written
            # in, so the guard and the thing it guards speak the same vocabulary.
            key = (ov["eval"], ov["dataset"], _resolved_label(cf_p, ov["model"]),
                   ov["seed"])
            launches[key].append((os.path.getmtime(cf_p), _resolved_resample(cf_p)))

    out = {}
    for key, runs in launches.items():
        csv_t = _csv_mtime(bench_root, *key)
        if csv_t is None:
            continue  # the cell has no result file, nothing to guard
        wrote = [r for r in runs if r[0] <= csv_t]
        if wrote:
            out[key] = max(wrote)[1]
        else:
            # every recorded launch postdates the CSV: the result was produced by a run
            # whose hydra directory is gone, so its rate is not recoverable. Say so
            # rather than crediting a later (failed) launch.
            out[key] = "UNKNOWN"
    return out


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
