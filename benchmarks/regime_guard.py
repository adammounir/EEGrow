"""Did the two arms of each grow/bd pair see the same preprocessing?

A paired comparison only isolates growth if grow_X and bd_X ran on identical data.
That was not guaranteed: the production grid passed ``dataset.resample=250`` on the
CLI, but the retry scripts omitted it, so any relaunched cell silently fell back to
the dataset's native rate (500 Hz on schirrmeister2017, 1000 Hz on lee2019_mi, ...).
At a different rate a fixed 25-sample temporal kernel spans a different duration, so
such a pair measures preprocessing, not growth.

The rate is established from the strongest evidence available for each cell, in this
order. The ordering is the point: the earlier sources say what the run *did*, the
later ones only what some run was *asked* to do.

1. ``sfreq`` column of the result CSV. Self-certifying: written by the run, into the
   file whose rows are being analysed. Runs after this module's sibling change.
2. The slurm log, matched to the file by timestamp. A run prints the exact path it
   writes and the second it writes it; when that second equals the file's mtime, the
   log describes the bytes on disk -- not an earlier run of the same cell, not a run
   that crashed. This is proof, not inference.
3. ``multirun/<date>/<overrides>/`` hydra directories. The directory name is the
   override string, so it is unique per cell and concurrent runs cannot collide.
4. ``outputs/<date_time>/`` hydra directories. Lossy, and known to have lost records:
   the name was a timestamp to the second, so two array tasks starting in the same
   second resolved to one directory and one of them was overwritten. When the winning
   run's record is the one that vanished, "latest launch that predates the CSV" falls
   back to an older, superseded launch and reports its rate with full confidence.
   That produced errors in *both* directions on the real grid -- one cell invented as
   contaminated (lee2019_mi/within_session/grow_eegnex/seed1, in fact 250 Hz) and
   seven genuinely contaminated cells missed. Kept as a last resort, reported apart.
"""

import collections
import csv as _csv
import datetime as dt
import glob
import os
import re

_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]")
_SFREQ = re.compile(r"sfreq=([0-9.]+)")
_SAVED = re.compile(r"saved \d+ rows -> (\S+\.csv)")
# a run logs the path just before closing the handle; allow for the rounding of both
# the log's whole-second timestamp and the filesystem's mtime
_MATCH_TOL = 2.0

# evidence tiers, strongest first
LOG = "csv-sfreq", "log-certifie", "hydra-multirun", "hydra-outputs"
CSV_SFREQ, LOG_CERT, HYDRA_MULTIRUN, HYDRA_OUTPUTS = LOG


def _canon(v):
    """One spelling per rate: the yaml pins ``250.0``, the CLI passes ``250``."""
    v = str(v).strip()
    if v in ("null", "None", ""):
        return "NATIVE"
    try:
        return f"{float(v):g}"
    except ValueError:
        return v


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
    return _canon(m.group(1)) if m else "UNKNOWN"


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


def _csv_path(bench_root, ev, ds, label, seed, align):
    """The cell's result file, or None if there is none.

    The stem is ``<label>__seed<N>.csv`` on the raw arm and
    ``<label>__<tag><level>__seed<N>.csv`` on an aligned one. The tag is built from the
    align config, which the overrides do not spell out, so the aligned arm is globbed.
    """
    d = os.path.join(bench_root, "results", ev, ds)
    if align == "none":
        hits = [os.path.join(d, f"{label}__seed{seed}.csv")]
    else:
        hits = glob.glob(os.path.join(d, f"{label}__*__seed{seed}.csv"))
    hits = [h for h in hits if os.path.exists(h)]
    return max(hits, key=os.path.getmtime) if hits else None


def _csv_sfreq(path):
    """The rate the run recorded in its own result rows, if it recorded one."""
    try:
        with open(path, newline="") as fh:
            r = _csv.reader(fh)
            header = next(r, None)
            if not header or "sfreq" not in header:
                return None
            row = next(r, None)
            return _canon(row[header.index("sfreq")]) if row else None
    except (OSError, StopIteration, IndexError, ValueError):
        return None


def _log_certificates(log_dirs):
    """{csv path -> [(save time, rate)]}, one entry per ``saved ... -> path`` line."""
    out = collections.defaultdict(list)
    for d in log_dirs:
        for f in glob.glob(os.path.join(d, "*.out")):
            rate = None
            for line in open(f, errors="replace"):
                m = _SFREQ.search(line)
                if m:
                    rate = _canon(m.group(1))
                m, t = _SAVED.search(line), _TS.match(line)
                if m and t and rate is not None:
                    when = dt.datetime.strptime(t.group(1),
                                                "%Y-%m-%d %H:%M:%S").timestamp()
                    out[os.path.realpath(m.group(1))].append((when, rate))
    return out


def _default_log_dirs(bench_root):
    """Slurm logs live beside the repo, not under ``benchmarks/``."""
    parent = os.path.dirname(os.path.abspath(bench_root))
    return sorted({d for pat in (os.path.join(bench_root, "*logs*"),
                                 os.path.join(parent, "*logs*"))
                   for d in glob.glob(pat) if os.path.isdir(d)})


def _hydra_launches(bench_root):
    """{cell key -> [(config mtime, rate, tier)]} over both hydra trees."""
    launches = collections.defaultdict(list)
    trees = [(os.path.join(bench_root, "multirun", "*", "*") + os.sep, HYDRA_MULTIRUN),
             (os.path.join(bench_root, "outputs", "*") + os.sep, HYDRA_OUTPUTS)]
    for pattern, tier in trees:
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
                   ov["seed"], ov.get("align", "none"))
            launches[key].append((os.path.getmtime(cf_p), _resolved_resample(cf_p),
                                  tier))
    return launches


def cell_evidence(bench_root=".", log_dirs=None):
    """(eval, dataset, model, seed, align) -> (rate, evidence tier).

    ``align`` belongs in the key: the alignment ablation relaunches the *same*
    (eval, dataset, model, seed) point with ``align=euclidean``, writing a separate
    CSV. Without it the two launches collide and "most recent wins" would attribute
    the aligned run's rate to the raw cell too.

    Only cells that have a result file are reported: the guard exists to qualify rows
    that enter an analysis, and a cell with no CSV contributes none.
    """
    if log_dirs is None:
        log_dirs = _default_log_dirs(bench_root)
    certs = _log_certificates(log_dirs)
    launches = _hydra_launches(bench_root)

    keys = set(launches)
    # a CSV whose hydra record was clobbered has no launch at all, but may still carry
    # its own sfreq or have a surviving log; recover those cells from the result tree
    for p in glob.glob(os.path.join(bench_root, "results", "*", "*", "*.csv")):
        ev, ds, stem = os.path.abspath(p).split(os.sep)[-3:]
        parts = stem[:-4].split("__")
        if len(parts) < 2 or not parts[-1].startswith("seed"):
            continue
        keys.add((ev, ds, parts[0], parts[-1][4:],
                  "none" if len(parts) == 2 else "euclidean"))

    out = {}
    for key in keys:
        path = _csv_path(bench_root, *key)
        if path is None:
            continue
        rp, mtime = os.path.realpath(path), os.path.getmtime(path)

        rate = _csv_sfreq(path)
        if rate is not None:
            out[key] = (rate, CSV_SFREQ)
            continue

        hits = {r for when, r in certs.get(rp, []) if abs(when - mtime) <= _MATCH_TOL}
        if len(hits) == 1:
            out[key] = (hits.pop(), LOG_CERT)
            continue

        # hydra: a launch only counts if it actually *wrote* the CSV. hydra writes
        # config.yaml when the run starts and we write the CSV when it ends, so a
        # successful run has mtime(config) <= mtime(csv); a run that crashed leaves its
        # config behind but never touches the CSV, which still holds the older result.
        wrote = [r for r in launches.get(key, []) if r[0] <= mtime]
        if wrote:
            _, rate, tier = max(wrote)
            out[key] = (rate, tier)
        else:
            out[key] = ("UNKNOWN", "aucune")
    return out


def cell_regimes(bench_root=".", log_dirs=None):
    """(eval, dataset, model, seed, align) -> rate of the run that wrote the CSV."""
    return {k: v[0] for k, v in cell_evidence(bench_root, log_dirs).items()}


def mixed_pairs(regimes, pairs):
    """Pairs whose two arms ran at different rates -- not comparable.

    A pair is compared within one alignment arm, so the arm is part of the match:
    grow_X aligned pairs with bd_X aligned, never with bd_X raw.
    """
    bad = []
    for grow, fixed in pairs:
        for (ev, ds, m, sd, al) in regimes:
            if m != grow:
                continue
            rg = regimes[(ev, ds, grow, sd, al)]
            rf = regimes.get((ev, ds, fixed, sd, al))
            if rf is not None and rg != rf:
                bad.append((grow, fixed, ev, ds, sd, al, rg, rf))
    return sorted(bad)


def assert_paired(pairs, bench_root=".", allow_env="EEGROW_ALLOW_MIXED"):
    """Abort the analysis if any pair straddles two sampling rates."""
    evidence = cell_evidence(bench_root)
    if not evidence:
        print("[regime] aucune cellule datee -- garde-fou inactif")
        return {}, []
    regimes = {k: v[0] for k, v in evidence.items()}
    bad = mixed_pairs(regimes, pairs)
    rates = collections.Counter(regimes.values())
    tiers = collections.Counter(t for _, t in evidence.values())
    print(f"[regime] {len(regimes)} cellules, taux: "
          + ", ".join(f"{k}={v}" for k, v in rates.most_common()))
    print("[regime] preuve: "
          + ", ".join(f"{k}={tiers[k]}" for k in LOG + ("aucune",) if tiers[k]))
    weak = tiers[HYDRA_OUTPUTS] + tiers["aucune"]
    if weak:
        print(f"[regime] ATTENTION : {weak} cellules reposent sur une trace hydra "
              f"non fiable (repertoire outputs/ ecrasable) ou sur rien du tout")
    if not bad:
        print("[regime] OK -- les deux bras de chaque paire ont le meme prétraitement")
        return regimes, bad
    print(f"[regime] {len(bad)} PAIRES A CHEVAL SUR DEUX TAUX -- non comparables :")
    for grow, fixed, ev, ds, sd, al, rg, rf in bad:
        print(f"  {ds:20s} {ev:15s} seed={sd} align={al:9s} "
              f"{grow}={rg}  {fixed}={rf}")
    if os.environ.get(allow_env) != "1":
        raise SystemExit(
            f"\nAnalyse interrompue : {len(bad)} paires comparent deux pretraitements.\n"
            f"Relancer ces cellules a 250 Hz, ou forcer avec {allow_env}=1 "
            f"(les deltas concernes ne mesurent alors plus la croissance)."
        )
    print(f"[regime] {allow_env}=1 -> on continue malgre tout")
    return regimes, bad
