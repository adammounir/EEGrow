"""How much of the published grid can still have its preprocessing proven, and how.

Background. A paired grow_X/bd_X comparison only isolates growth if both arms saw the
same input, and for a while they did not: the production launch passed
``dataset.resample=250`` on the command line while some retry scripts omitted it, so a
relaunched cell fell back to the dataset's native rate. At a different rate a fixed
25-sample temporal kernel spans a different duration, so such a pair measures resampling
rather than growth. ``regime_guard.py`` is the guard against that, and the fix250
campaign (``slurm/fix250_*.txt``, 116 cells) was the remediation.

Why this script exists. The guard reads its evidence from hydra's run records, and those
records no longer exist: the same ``rsync --delete`` that erased the epoch cache also
erased ``benchmarks/multirun/`` and ``benchmarks/outputs/``. What survives is the slurm
logs and the result files themselves. The guard therefore reports UNKNOWN for most cells
and -- correctly -- refuses to certify them. But "not provable" is not "contaminated",
and a paper needs the distinction stated in numbers rather than asserted. That is what
this script produces.

Three angles, and one that failed:

1. Datasets whose MOABB native rate is already 250 Hz (BNCI2014-001, BNCI2014-004,
   Zhou2016). There ``resample=250`` is a no-op and the cell ran at 250 Hz whether or
   not the override reached it. Immune by construction, no provenance required.

2. Per-cell certificates from the logs. A run prints ``saved N rows -> <path>`` in the
   same second it closes the handle, so the log line whose timestamp equals the file's
   mtime describes the bytes on disk -- not an earlier run of the same cell, not a run
   that crashed. This also settles the remediation question directly: it says, for every
   cell that ever ran at a native rate, whether the bytes now on disk came from a 250 Hz
   relaunch.

3. FALSIFIED -- fit-time homogeneity. The idea was that a convnet's cost is roughly
   linear in the number of input samples, so within one (protocol, dataset, model) group
   a seed that ran at a native rate should stand out by a factor of native/250, and cost
   homogeneity would extend one certificate to all five seeds. It does not work, and the
   control says so: ``lee2019_mi / cross_subject / bd_sccnet`` has all five seeds
   individually certified at 250 Hz and its per-seed median fit time still spans
   43.5 s to 242.3 s, a factor of 5.6 -- larger than the 4.0 a genuine 1000 Hz
   contamination would produce. Early stopping and heterogeneous GPUs dominate the cost,
   so the statistic cannot see the rate. It is computed and printed anyway, as the
   refutation is the useful output: no threshold on it may be promoted to a gate.

The honest summary is therefore the union of (1) and (2), and an explicit count of what
neither covers.
"""
import collections
import datetime as dt
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_guard import _canon, _log_certificates, _default_log_dirs  # noqa: E402,F401

# MOABB native sampling rates of the twelve datasets in the grid
NATIVE = {"bnci2014_001": 250, "bnci2014_002": 512, "bnci2014_004": 250,
          "bnci2015_001": 512, "alexmi": 512, "cho2017": 512, "lee2019_mi": 1000,
          "physionetmi": 160, "schirrmeister2017": 500, "shin2017a": 200,
          "weibo2014": 200, "zhou2016": 250}
TOL = 2.0          # log timestamp is whole-second, mtime is not
MIN_T = 5.0        # below this, fit time is fixed overhead rather than work
RATIO_MAX = 1.5    # the smallest contamination factor in this grid is 200/250 -> 1.25


def collect(bench_root="."):
    """One row per result CSV: what wrote it, at what rate, and what it cost."""
    certs = _log_certificates(_default_log_dirs(bench_root))
    rows = []
    for p in sorted(glob.glob(os.path.join(bench_root, "results", "*", "*", "*.csv"))):
        ev, ds, stem = os.path.abspath(p).split(os.sep)[-3:]
        parts = stem[:-4].split("__")
        if not parts[-1].startswith("seed") or ds not in NATIVE:
            continue
        mt = os.path.getmtime(p)
        evs = certs.get(os.path.realpath(p), [])
        match = [e for e in evs if abs(e[0] - mt) <= TOL]
        d = pd.read_csv(p)
        rows.append(dict(
            eval=ev, dataset=ds, model=parts[0], seed=parts[-1][4:],
            align="euclidean" if len(parts) > 2 else "none",
            certified=max(match)[1] if match else None,
            ever_native=any(_canon(r) != "250" for _, r in evs),
            n_writes=len(evs),
            t=pd.to_numeric(d["time"], errors="coerce").median()))
    return pd.DataFrame(rows)


def main(bench_root="."):
    df = collect(bench_root)
    raw = df[df["align"] == "none"].copy()
    raw["native"] = raw["dataset"].map(NATIVE)
    raw["noop"] = raw["native"] == 250
    raw["cert250"] = raw["certified"].map(lambda r: r is not None and _canon(r) == "250")
    tot = len(raw)

    print(f"cellules du bras brut : {tot}\n")

    print("=== 1. immunes par construction (taux natif deja 250 Hz) ===")
    noop = raw[raw["noop"]]
    print(f"{len(noop)} cellules ({100 * len(noop) / tot:.1f} %) sur "
          f"{', '.join(sorted(noop['dataset'].unique()))}\n")

    print("=== 2. certifiees par un log survivant ===")
    other = raw[~raw["cert250"] & raw["certified"].notna()]
    print(f"{int(raw['cert250'].sum())} cellules certifiees a 250 Hz ; "
          f"{len(other)} certifiees a un autre taux")
    nat = raw[raw["ever_native"]]
    print(f"cellules ayant tourne a un taux natif a un moment : {len(nat)} ; "
          f"dont certifiees 250 Hz apres relance : {int(nat['cert250'].sum())} ; "
          f"restees hors 250 : {len(nat) - int(nat['cert250'].sum())}")
    covered = raw[raw["noop"] | raw["cert250"]]
    print(f"\n--> etabli (1 ou 2) : {len(covered)} cellules "
          f"({100 * len(covered) / tot:.1f} %)")
    print(f"--> ni l'un ni l'autre : {tot - len(covered)} cellules "
          f"({100 * (tot - len(covered)) / tot:.1f} %) -- aucune trace, dans aucun sens\n")

    print("=== 3. discriminateur par le cout -- FALSIFIE, conserve comme refutation ===")
    worst = []
    for (ev, ds, mo), g in raw[~raw["noop"]].groupby(["eval", "dataset", "model"]):
        ts = g["t"].dropna()
        if len(ts) < 2 or ts.min() <= 0 or ts.median() < MIN_T:
            continue
        worst.append((ts.max() / ts.min(), ev, ds, mo, bool(g["cert250"].all()),
                      ts.min(), ts.max(), NATIVE[ds] / 250))
    worst.sort(reverse=True)
    n_flag = sum(1 for w in worst if w[0] > RATIO_MAX)
    n_flag_cert = sum(1 for w in worst if w[0] > RATIO_MAX and w[4])
    print(f"{len(worst)} groupes mesurables, {n_flag} au ratio > {RATIO_MAX} "
          f"-- dont {n_flag_cert} dont les 5 graines sont certifiees a 250 Hz")
    print("les groupes dont le cout varie le plus, avec leur certification :")
    for r, ev, ds, mo, allc, lo, hi, fac in worst[:6]:
        print(f"  ratio {r:5.2f} (contamination -> {fac:.1f}x)  {ds:18s} {ev:14s} "
              f"{mo:13s} t={lo:7.1f}..{hi:7.1f}s  "
              f"{'5/5 certifiees 250 Hz' if allc else ''}")
    if n_flag_cert:
        print(f"\nUn groupe entierement certifie a 250 Hz depasse le seuil : le cout ne "
              f"voit pas le taux.\nAucun seuil sur ce ratio ne doit servir de garde-fou.")
    return raw


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
