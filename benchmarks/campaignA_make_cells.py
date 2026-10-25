"""Cells and SLURM tasks of campaign A (analysis/PREREG_campaign_A.md).

ONE CELL = one (protocol, dataset, model, arm, seed, subject) = one ``run_cell.py`` call.
Within / cross_session: the subject evaluated; cross_subject: the subject held out
(``only_subject``; MOABB builds the folds on the full subject list). A line of
``campaignA_cells.txt`` is the WHOLE specification of its cell:

    CELL_ID PROT DS MODEL ARM SEED SUBJ RESULTS_DIR CSV [hydra overrides ...]

Nothing about the data is assumed: subjects and sessions come from the manifest
(``campaignA_manifest.py``, one JSON per dataset), the cross_session list is the
datasets whose subjects all have >= 2 sessions. Recipes and the tiny rule are imported
from the validation campaign (``valid_make_cells``), never retyped.

ONE TASK = a list of cells run in sequence by one array task, so small cells share a
job (MaxArraySize 1001, MaxJobCount 10000 on Margaret) while a large cell has its own.
Packing only ever groups cells of the same (protocol, dataset, model, arm, seed), and
the cost model (timing probe, 2080 Ti equivalent, calibrated on the measured
within-session walls of the validation) only sets the packing and the time limit,
never what runs. The time limit covers the slowest admitted card (P100, x2.05 measured
on 25/10) with a further x2 margin.

Archive (decided 25/10/2026, option B): init / growth / final / selected weights and
test logits on every cell; full checkpoints at epochs 10/25/50/100 on cross_session and
cross_subject only.

Usage (on Margaret, it reads the probe and the manifest there)::

    python campaignA_make_cells.py --manifest ../manifest --probe /scratch/amounir/eegrow_valid/probe/turing \
        --out-root /data/tau/iceberg_1/titanic_1/users/amounir/eegrow/campaignA --dest .
    python campaignA_make_cells.py ... --subset dryrun --out-root .../campaignA_dryrun
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

from utils import cell_stem
from valid_make_cells import MODELS, RULES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004", "bnci2015_001", "cho2017",
            "lee2019_mi", "physionetmi", "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016"]
SEEDS = {ds: [0, 1, 2] for ds in DATASETS}
for _ds in ("zhou2016", "alexmi", "bnci2014_001"):
    SEEDS[_ds] = [0, 1, 2, 3, 4]
PROTOCOLS = ["within_session", "cross_session", "cross_subject"]
ARMS = ["bd", "tiny", "random"]
CKPT = "[10,25,50,100]"
# The validation campaign's protocol (valid_battery.sbatch), plus the archive flags.
COMMON = ("train.max_epochs=200 train.patience=200 train.selection_monitor=valid_acc "
          "train.smooth_window=5 train.tie_break=earliest train.restore_selected=false "
          "train.growth_holdout=0.1 train.lr_schedule=null "
          "train.save_selected=true train.save_final=true train.save_init=true "
          "train.save_growth=true train.save_test_logits=true "
          "cache.enabled=true cache.readonly=true")
# Host RAM per cross_subject cell (the whole dataset is loaded): 1.5 x the probe's MaxRSS
# (816443, sacct), floor 16 G. schirrmeister2017 hit the 64 G limit in the probe: its
# value is the preflight's measurement (PREREG § 2), not a guess.
XS_MEM = {"schirrmeister2017": 128, "lee2019_mi": 40, "cho2017": 32, "physionetmi": 16}
MEM_FLOOR = 16
P100_SLOWDOWN = 2.05
TIME_MARGIN = 2.0
TIME_BUCKETS = [2, 4, 8, 16, 32, 48]
PACK_TARGET_H = 2.0
# measured within-session h per seed (validation, 2080 Ti, 200 epochs): (bd, tiny)
MEASURED = {
    "conformer": {"bnci2014_001": (1.09, 1.20), "cho2017": (2.47, 3.70), "lee2019_mi": (3.66, 5.85)},
    "ctnet": {"bnci2014_001": (0.69, 1.09), "cho2017": (2.52, 3.47), "lee2019_mi": (3.46, 5.72)},
    "eegnex": {"bnci2014_001": (0.65, 0.71), "cho2017": (2.95, 3.28), "lee2019_mi": (4.24, 4.52)},
    "msvtnet": {"bnci2014_001": (0.71, 0.91), "cho2017": (3.07, 3.84), "lee2019_mi": (4.16, 5.76)},
    "shallow": {"bnci2014_001": (0.19, 0.58), "cho2017": (0.65, 1.90), "lee2019_mi": (0.97, 2.82)},
    "tidnet": {"bnci2014_001": (0.93, 1.05), "cho2017": (3.41, 3.71), "lee2019_mi": (5.07, 5.75)},
}


def load_manifest(d):
    man = {}
    for ds in DATASETS:
        f = Path(d) / f"{ds}.json"
        if not f.exists():
            raise SystemExit(f"REFUSED: no manifest for {ds} ({f})")
        man[ds] = json.loads(f.read_text())
        bad = [s for s, v in man[ds]["subjects"].items() if v.get("reread_bit_identical") is False]
        if bad:
            raise SystemExit(f"REFUSED: {ds} cache re-read differs from its derivation for {bad}")
    return man


def check_within_folds(man, k=5):
    """MOABB's within-session split is a stratified k-fold per session: a class with
    fewer than k trials in one session would crash that cell 200 epochs in. Refuse."""
    bad = [(ds, s, se, pc) for ds, m in man.items() for s, sub in m["subjects"].items()
           for se, x in sub["sessions"].items() for pc in [x["per_class"]]
           if len(pc) != len(m["classes"]) or min(pc.values()) < k]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} sessions cannot be {k}-fold stratified, e.g. {bad[:3]}")


def cross_session_datasets(man):
    """Datasets whose EVERY subject has >= 2 sessions (MOABB skips a 1-session subject,
    which would make the cell an empty CSV); a dataset with a mix is refused loudly."""
    out = []
    for ds, m in man.items():
        n = [len(s["sessions"]) for s in m["subjects"].values()]
        if min(n) >= 2:
            out.append(ds)
        elif max(n) >= 2:
            raise SystemExit(f"REFUSED: {ds} mixes 1-session and multi-session subjects {collections.Counter(n)}")
    return out


class Cost:
    """GPU-h (2080 Ti) of one cell, from the timing probe; packing and time limits only."""

    def __init__(self, probe_dir, man):
        rows = {}
        for f in glob.glob(f"{probe_dir}/*.json"):
            r = json.load(open(f))
            rows[(r["dataset"], r["model"])] = r["s_per_trial_epoch"]
        arms = sorted({a for _, a in rows})
        for ds in DATASETS:  # impute the probe cells killed by host RAM: dataset factor vs lee
            present = [a for a in arms if (ds, a) in rows]
            fac = st.median(rows[(ds, a)] / rows[("lee2019_mi", a)] for a in present)
            for a in arms:
                rows.setdefault((ds, a), rows[("lee2019_mi", a)] * fac)
        self.spte = rows
        self.man = man
        cal = collections.defaultdict(list)
        for m, (grow, bd, *_rest) in MODELS.items():
            for ds, (h_bd, h_tiny) in MEASURED[m].items():
                N = sum(s["n"] for s in man[ds]["subjects"].values())
                for arm, h in ((bd, h_bd), (grow, h_tiny)):
                    cal[arm].append(h / (4 * N * 200 * rows[(ds, arm)] / 3600))
        self.cal = {a: st.median(v) for a, v in cal.items()}

    def __call__(self, prot, ds, cfg, subj):
        subs = self.man[ds]["subjects"]
        n_s = subs[subj]["n"]
        trials = {"within_session": 4 * n_s,
                  "cross_session": (len(subs[subj]["sessions"]) - 1) * n_s,
                  "cross_subject": sum(s["n"] for s in subs.values()) - n_s}[prot]
        return trials * 200 * self.spte[(ds, cfg)] * self.cal[cfg] / 3600


def arm_spec(model, arm):
    grow, bd, recipe, extra = MODELS[model]
    if arm == "bd":
        return bd, recipe
    ov = f"{recipe} {RULES['tiny']} {extra}".strip()
    if arm == "random":
        ov += " train.growth_direction=random"
    return grow, ov


def cells(man, out_root, subset=None):
    xsess = cross_session_datasets(man)
    for prot in PROTOCOLS:
        for ds in DATASETS:
            if prot == "cross_session" and ds not in xsess:
                continue
            for model in MODELS:
                for arm in ARMS:
                    cfg, ov = arm_spec(model, arm)
                    for seed in SEEDS[ds]:
                        for subj in man[ds]["subjects"]:
                            if subset and not subset(prot, ds, model, arm, seed, subj):
                                continue
                            rdir = f"{out_root}/{model}__{arm}/results"
                            # run_moabb_hydra's stem: utils.cell_stem (label == config name,
                            # no alignment tag) + "__subj<only_subject>"
                            csv = f"{rdir}/{prot}/{ds}/{cell_stem(cfg, '', seed)}__subj{subj}.csv"
                            ck = f" train.save_checkpoints={CKPT}" if prot != "within_session" else ""
                            # the cache the manifest was verified on (moabb_cache_A, or the
                            # old cache for lee2019_mi: PREREG § 2.2)
                            cp = f" cache.path={man[ds]['cache']}"
                            yield dict(prot=prot, ds=ds, model=model, arm=arm, cfg=cfg, seed=seed,
                                       subj=subj, rdir=rdir, csv=csv,
                                       ov=f"{COMMON}{cp}{ck} {ov}")


def bucket(h):
    need = h * P100_SLOWDOWN * TIME_MARGIN + 0.25
    return next((b for b in TIME_BUCKETS if b >= need), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dest", default=".")
    ap.add_argument("--subset", choices=["all", "dryrun", "ram", "b126"], default="all")
    a = ap.parse_args()
    man = load_manifest(a.manifest)
    check_within_folds(man)
    cost = Cost(a.probe, man)
    subset = None
    first = {ds: next(iter(man[ds]["subjects"])) for ds in DATASETS}
    if a.subset == "dryrun":
        # Full 200 epochs: every arm once on a 4-class dataset (within + cross_session),
        # the 2-class ROC-AUC path on cross_subject, and the GPU memory of the widest
        # nets on the 128-channel dataset (bd and tiny; random grows like tiny).
        subset = lambda p, ds, m, arm, seed, s: seed == 0 and s == first[ds] and (
            (ds == "bnci2014_001" and p in ("within_session", "cross_session"))
            or (ds == "bnci2014_004" and p == "cross_subject")
            or (ds == "schirrmeister2017" and p == "within_session" and arm != "random"))
    elif a.subset == "ram":
        # host RAM of the largest cross_subject pool (run with a few epochs only)
        subset = lambda p, ds, m, arm, seed, s: (seed == 0 and s == first[ds] and arm != "random"
                                                 and ds == "schirrmeister2017" and p == "cross_subject")
    elif a.subset == "b126":
        # the cu126 env on V100 / P100 with the new code paths (run with a few epochs only)
        subset = lambda p, ds, m, arm, seed, s: (seed == 0 and s == first[ds] and ds == "bnci2014_001"
                                                 and p == "within_session"
                                                 and (m, arm) in {("shallow", "tiny"), ("shallow", "random"),
                                                                  ("ctnet", "random"), ("conformer", "bd")})
    L = list(cells(man, a.out_root, subset))
    # Refuse two cells writing the same file (the 25/10 smoke: tiny and random shared a stem).
    dup = [k for k, n in collections.Counter(c["csv"] for c in L).items() if n > 1]
    if dup:
        raise SystemExit(f"REFUSED: {len(dup)} output paths shared by several cells, e.g. {dup[0]}")
    for c in L:
        if c["arm"] != "bd" and not ("train.growth_normalization=tiny_paper" in c["ov"]
                                     and "train.growth_fixed_s=1.0" in c["ov"]):
            raise SystemExit(f"REFUSED: growing cell without the explicit tiny rule: {c}")
        c["h"] = cost(c["prot"], c["ds"], c["cfg"], c["subj"])
        c["mem"] = max(MEM_FLOOR, XS_MEM.get(c["ds"], MEM_FLOOR)) if c["prot"] == "cross_subject" else MEM_FLOOR
    for i, c in enumerate(L):
        c["id"] = i
    # pack
    groups = collections.defaultdict(list)
    for c in L:
        groups[(c["prot"], c["ds"], c["model"], c["arm"], c["seed"])].append(c)
    tasks = []
    for key, cs in groups.items():
        cur, h = [], 0.0
        for c in sorted(cs, key=lambda c: -c["h"]):
            if cur and h + c["h"] > PACK_TARGET_H:
                tasks.append(cur)
                cur, h = [], 0.0
            cur.append(c)
            h += c["h"]
        if cur:
            tasks.append(cur)
    rows = []
    for t in tasks:
        h = sum(c["h"] for c in t)
        b = bucket(h)
        if b is None:
            raise SystemExit(f"REFUSED: task of {h:.1f} h exceeds the {TIME_BUCKETS[-1]} h bucket: {t[0]['csv']}")
        rows.append({"mem": t[0]["mem"], "time": b, "h": h, "ids": [c["id"] for c in t],
                     "canary": all(c["seed"] == 0 and c["ds"] in ("zhou2016", "bnci2014_004") for c in t)})
    # canary first (seed 0 of the two smallest datasets, every protocol and arm), then
    # the longest tasks first so the makespan is not set by a late giant
    rows.sort(key=lambda r: (not r["canary"], -r["h"]))
    dest = Path(a.dest)
    tag = "" if a.subset == "all" else f"_{a.subset}"
    with open(dest / f"campaignA_cells{tag}.txt", "w") as fh:
        for c in L:
            fh.write(f"{c['id']} {c['prot']} {c['ds']} {c['model']} {c['arm']} {c['seed']} {c['subj']} "
                     f"{c['rdir']} {c['csv']} model={c['cfg']} {c['ov']}\n")
    # arrays: same (mem, time) class, <= 1000 tasks each, in priority order
    arrays, order = collections.OrderedDict(), []
    for r in rows:
        k = (r["canary"], r["mem"], r["time"])
        arrays.setdefault(k, []).append(r)
    adir = dest / f"campaignA_arrays{tag}"
    adir.mkdir(exist_ok=True)
    for f in adir.glob("*.txt"):
        raise SystemExit(f"REFUSED: {adir} already holds arrays ({f.name}); use a fresh --dest")
    n = 0
    for (can, mem, time), rs in arrays.items():
        for j in range(0, len(rs), 1000):
            part = rs[j:j + 1000]
            name = f"A{n:02d}_{'canary' if can else 'main'}_mem{mem}_t{time}.txt"
            (adir / name).write_text("".join(",".join(map(str, r["ids"])) + "\n" for r in part))
            order.append((name, len(part), mem, time, sum(r["h"] for r in part)))
            n += 1
    xs = cross_session_datasets(man)
    summ = collections.Counter((c["prot"], c["arm"]) for c in L)
    print(f"{len(L)} cells, {len(rows)} tasks, {n} arrays -> {dest}")
    print("cross_session datasets (manifest):", xs)
    for k in sorted(summ):
        print(f"  {k[0]:15s} {k[1]:7s} {summ[k]:6d} cells")
    tot = collections.defaultdict(float)
    for c in L:
        tot[c["prot"]] += c["h"]
    print("GPU-h (2080 Ti eq.):", {k: round(v) for k, v in tot.items()}, "total", round(sum(tot.values())))
    print("longest cell:", max(L, key=lambda c: c["h"])["csv"], f"{max(c['h'] for c in L):.1f} h")
    for o in order:
        print("  array %s: %d tasks, mem %dG, time %dh, %.0f GPU-h" % o)
    json.dump({"n_cells": len(L), "n_tasks": len(rows), "arrays": order,
               "cross_session_datasets": xs, "gpu_h": tot,
               "calibration": cost.cal}, open(dest / f"campaignA_plan{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
