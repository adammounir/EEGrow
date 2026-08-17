"""What a campaign costs, and when it ends, from measured fit times.

Written to answer a question that has to be answered with numbers rather than a
guess -- how long the experiments take and what one iteration costs -- and kept in
the repo because those numbers are going into a paper and have to be reproducible.

The unit of work is a FOLD: one fit on one (subject, session), one row of a result
CSV, carrying its own `time` in seconds. A CELL is a (eval, dataset, model, seed)
quadruple -- one line of a pass TSV -- and costs the sum of its folds. A PASS is a
TSV run by an allocation of G GPUs holding K tenants per GPU, so its wall time is
its cells' serial cost divided by G*K, times the number of cooperating allocations
(REPLICAS in plan_campaign.py).

Passes run as independent allocations, so the campaign's wall time is the MAXIMUM
over passes, not the sum -- the sum is what it would cost if they queued behind one
another. Total GPU-hours is the cluster's accounting number, not the calendar one.

G, K and the allocation count are read from the emitted submit.sh rather than
recomputed, so the estimate describes what was actually submitted.

WHAT THE ESTIMATE CANNOT KNOW
  - A cell never measured falls back to the median of its (eval, model), then its
    eval, then everything. The provenance line reports how many cells took which
    path; treat a low `measured` share as a wide error bar, not as precision.
  - Times for the grow_* arms were measured BEFORE the growth cap was enforced,
    when a model kept widening past its target. Capped models are smaller, so those
    cells are over-estimated -- the bias is conservative, which is the right
    direction for a schedule but the wrong one for a claim about speed.
"""
import argparse
import collections
import csv
import glob
import os
import re
import statistics as st

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--passes", default="/scratch/amounir/passes_v5",
                help="directory holding the grid_*.tsv passes and submit.sh")
ap.add_argument("--results", default="/scratch/amounir/eegrow/benchmarks/results*",
                help="glob of result trees to measure fold times from")
args = ap.parse_args()
PASSES, RESULTS = args.passes, args.results

# ---------------------------------------------------------------- measurements
# Every result CSV ever produced, minus the ones that are not comparable:
# cross_dataset is a different protocol, *invalid* was the volts bug, and the
# *broken* smokes died mid-fit so their `time` is a truncation, not a duration.
SKIP = ("invalid", "cross_dataset", "broken", "results_probe", "results_gputest")
cell_secs = collections.defaultdict(list)   # (ev, ds, md) -> [total s per cell]
fold_secs = collections.defaultdict(list)   # (ev, ds, md) -> [s per fold]
n_folds = collections.defaultdict(list)

for root in glob.glob(RESULTS):
    if any(s in root for s in SKIP):
        continue
    for p in glob.glob(root + "/*/*/*.csv"):
        try:
            with open(p) as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        if not rows or "time" not in rows[0]:
            continue
        ev, md = rows[0].get("eval"), rows[0].get("model")
        if not (ev and md):
            continue
        ds = os.path.basename(os.path.dirname(p))   # config name, not MOABB name
        try:
            times = [float(r["time"]) for r in rows]
        except Exception:
            continue
        cell_secs[(ev, ds, md)].append(sum(times))
        fold_secs[(ev, ds, md)].extend(times)
        n_folds[(ev, ds, md)].append(len(times))

cost = {k: st.median(v) for k, v in cell_secs.items()}
by_em = collections.defaultdict(list)
by_e = collections.defaultdict(list)
allc = []
for (ev, ds, md), c in cost.items():
    by_em[(ev, md)].append(c)
    by_e[ev].append(c)
    allc.append(c)


def est(ev, ds, md):
    if (ev, ds, md) in cost:
        return cost[(ev, ds, md)], "measured"
    if (ev, md) in by_em:
        return st.median(by_em[(ev, md)]), "model"
    if ev in by_e:
        return st.median(by_e[ev]), "eval"
    return st.median(allc), "global"


# ------------------------------------------------------------------- the plan
# G, K and the allocation count come from submit.sh, so the ETA describes what
# was actually submitted rather than what the planner intended.
alloc = collections.Counter()
gk = {}
for line in open(f"{PASSES}/submit.sh"):
    m = re.search(r"--gres=gpu:(\d+).*GRID=(\S+?\.tsv),K=(\d+)", line)
    if m:
        g, tsv, k = int(m.group(1)), os.path.basename(m.group(2)), int(m.group(3))
        alloc[tsv] += 1
        gk[tsv] = (g, k)

print(f"cellules mesurees au moins une fois : {len(cost)} triplets (eval,dataset,model)")
print()
hdr = (f"{'passe':<16s} {'G':>2s} {'K':>3s} {'alloc':>5s} {'cells':>6s} "
       f"{'GPU-h':>8s} {'mur h':>7s} {'mur j':>6s} {'mes.':>5s}")
print(hdr)
print("-" * len(hdr))

tot_gpu_h = 0.0
walls = {}
prov = collections.Counter()
per_cell = []
for tsv in sorted(gk, key=lambda t: -gk[t][1]):
    g, k = gk[tsv]
    n_alloc = alloc[tsv]
    secs = 0.0
    n_meas = n_cell = 0
    for line in open(f"{PASSES}/{tsv}"):
        f = line.split()
        if len(f) < 4 or line.startswith("#"):
            continue
        ev, ds, md, _seed = f[0], f[1], f[2], f[3]
        c, how = est(ev, ds, md)
        secs += c
        per_cell.append(c)
        prov[how] += 1
        n_meas += how == "measured"
        n_cell += 1
    gpu_h = secs / 3600
    wall = gpu_h / (g * k * n_alloc)
    tot_gpu_h += gpu_h
    walls[tsv] = wall
    print(f"{tsv[:-4]:<16s} {g:>2d} {k:>3d} {n_alloc:>5d} {n_cell:>6d} "
          f"{gpu_h:>8.1f} {wall:>7.1f} {wall/24:>6.2f} {100*n_meas/max(n_cell,1):>4.0f}%")

crit = max(walls, key=walls.get)
print()
print(f"provenance des estimations : {dict(prov)}")
print(f"total serialise      : {tot_gpu_h:.0f} GPU-h ({tot_gpu_h/24:.1f} jours-GPU)")
print(f"passes en parallele  : le mur est le MAX, pas la somme")
print(f"chemin critique      : {crit[:-4]}  ->  {walls[crit]:.1f} h "
      f"({walls[crit]/24:.2f} j)")
print(f"2e plus long         : "
      f"{sorted(walls.items(), key=lambda x: -x[1])[1][0][:-4]} "
      f"{sorted(walls.values())[-2]:.1f} h")
print(f"mur si tout attendait sequentiellement : {sum(walls.values()):.1f} h")
print()

# ------------------------------------------------- the per-iteration question
allf = [t for v in fold_secs.values() for t in v]
allc2 = sorted(per_cell)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


print("cout d'une ITERATION (1 fit = 1 fold = 1 sujet/session)")
print(f"  n={len(allf)} fits mesures")
for lbl, p in (("mediane", .50), ("p90", .90), ("p99", .99)):
    print(f"  {lbl:8s} {q(allf, p):8.1f} s   ({q(allf, p)/60:6.2f} min)")
print(f"  max      {max(allf):8.1f} s   ({max(allf)/3600:6.2f} h)")
print(f"  moyenne  {st.mean(allf):8.1f} s")
print()
print("cout d'une CELLULE (1 modele x 1 dataset x 1 seed, tous folds)")
for lbl, p in (("mediane", .50), ("p90", .90), ("p99", .99)):
    print(f"  {lbl:8s} {q(allc2, p)/3600:8.2f} h")
print(f"  max      {max(allc2)/3600:8.2f} h")
print()
print("folds par cellule (mediane) par evaluation")
by_ev = collections.defaultdict(list)
for (ev, ds, md), ns in n_folds.items():
    by_ev[ev].extend(ns)
for ev in sorted(by_ev):
    print(f"  {ev:16s} {st.median(by_ev[ev]):5.0f} folds")
print()
# Restricted to the cells the campaign actually contains. The unrestricted version
# is dominated by ts_svm / mdm / fgmdm / ts_lr on schirrmeister2017 (95 h, 88 h,
# 57 h, 49 h), which are `kind: ml` arms and are NOT in this grid -- reporting them
# as the campaign's worst cells would have been simply false.
plan = set()
plan_cost = collections.Counter()
plan_model = collections.Counter()
plan_ds = collections.Counter()
for tsv in gk:
    for line in open(f"{PASSES}/{tsv}"):
        f = line.split()
        if len(f) < 4 or line.startswith("#"):
            continue
        ev, ds, md = f[0], f[1], f[2]
        c, _ = est(ev, ds, md)
        plan.add((ev, ds, md))
        plan_cost[(ev, ds, md)] = c
        plan_model[md] += c
        plan_ds[ds] += c

print(f"modeles de la campagne ({len(plan_model)}) et part du cout")
tot_s = sum(plan_model.values())
for md, c in plan_model.most_common():
    print(f"  {md:16s} {c/3600:8.1f} h  {100*c/tot_s:5.1f}%")
print()
print("les 8 (eval,dataset,modele) les plus chers PRESENTS dans le plan")
for (ev, ds, md), c in plan_cost.most_common(8):
    print(f"  {ev:15s} {ds:19s} {md:14s} {c/3600:7.2f} h")
print()
print("cout par dataset (top 6)")
for ds, c in plan_ds.most_common(6):
    print(f"  {ds:20s} {c/3600:8.1f} h  {100*c/tot_s:5.1f}%")
