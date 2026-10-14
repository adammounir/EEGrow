"""Step-0 reading: the refusal criteria of PREREG § 5-6 and the config rule of § 3.2.

Committed BEFORE any step-0 result is read. It reads nothing the PREREG does not name,
and it refuses to read a (arm, config) whose cells did not all pass the gates of
``step0_gate.py`` -- a criterion computed on a refused cell would be read and then
"relaunched", which is reading twice.

Criteria (all frozen in the PREREG):
  départ      median(first_batch_loss) - ln(4), REPORTED, not refusing, for bd_* (§ 5.3)
  capacité    median over seeds 0-2 of the eval-mode loss on 32 trials >= 0.05 -> refuse
  optimum     > 20 % of the 270 fits with selected_epoch <= 5, or > 20 % >= 195 -> refuse
  référence   published 4-38 Hz mean outside the 95 % bootstrap CI (10 000 resamples of
              the 9 per-subject means over seeds 0-2, default_rng(0)) -> refuse
  pics        reported from the gate (refusal there means relaunch, not a criterion)
Config rule: B if B passes everything, else H if H passes, else the arm is refused.

Usage::

    python step0_criteria.py --root /scratch/amounir/eegrow_step0/step0_results --out s0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step0_gate as gate  # noqa: E402

ARMS = {"bd_shallow": "shallow", "bd_deep4": "deep"}
CANDIDATES = {  # PREREG § 3.2
    "B": {"bd_shallow": (6.25e-4, 0.0), "bd_deep4": (1e-2, 5e-4)},
    "H": {"bd_shallow": (6.25e-4, 0.01), "bd_deep4": (6.25e-4, 0.01)},
}
SEEDS = (0, 1, 2)
CAPACITY_THRESHOLD = 0.05
OPT_EARLY, OPT_LATE, OPT_FRAC = 5, 195, 0.20
PUBLISHED = {"shallow": 0.719, "deep": 0.701}       # arXiv 1703.05051v5 Table 2, 4-38 Hz
N_BOOT = 10_000


def departure(recs):
    v = np.array([r["first_batch_loss"] for r in recs], float)
    return {"median_first_batch_loss": float(np.median(v)),
            "excess_nat": float(np.median(v) - math.log(4)), "refusing": False}


def capacity(rows, arm, cand):
    got = {r["seed"]: r["final_loss_eval"] for r in rows
           if r["arm"] == arm and r["candidate"] == cand}
    if sorted(got) != list(SEEDS):
        return {"status": "incomplete", "seeds": sorted(got), "refused": None}
    med = float(np.median([got[s] for s in SEEDS]))
    return {"status": "read", "losses": [got[s] for s in SEEDS], "median": med,
            "refused": not (math.isfinite(med) and med < CAPACITY_THRESHOLD)}


def optimum(recs):
    e = np.array([r["selected_epoch"] for r in recs], float)
    if np.isnan(e).any():
        return {"status": "selected_epoch missing", "refused": None}
    early, late = float(np.mean(e <= OPT_EARLY)), float(np.mean(e >= OPT_LATE))
    return {"status": "read", "n": int(len(e)), "frac_early": early, "frac_late": late,
            "refused": early > OPT_FRAC or late > OPT_FRAC}


def bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def reference(rows, model):
    by_subject = defaultdict(dict)
    for r in rows:
        if r["model"] == model and r.get("max_epochs") == 800 and r.get("max_increase") == 80:
            by_subject[r["subject"]][r["seed"]] = r["test_acc"]
    complete = {s: v for s, v in by_subject.items() if sorted(v) == list(SEEDS)}
    if sorted(complete) != list(range(1, 10)):
        return {"status": "incomplete", "subjects": sorted(complete), "refused": None}
    per_subject = [float(np.mean([complete[s][k] for k in SEEDS])) for s in range(1, 10)]
    lo, hi = bootstrap_ci(per_subject)
    pub = PUBLISHED[model]
    return {"status": "read", "per_subject": per_subject,
            "mean": float(np.mean(per_subject)), "ci95": [lo, hi], "published": pub,
            "refused": not (lo <= pub <= hi)}


def read_cells(root, arm, cand):
    lr, wd = CANDIDATES[cand][arm]
    cells, recs = {}, []
    for seed in SEEDS:
        p = gate.cell_path(Path(root) / cand / "results", "within_session",
                           "bnci2014_001", arm, seed)
        if not p.exists():
            cells[seed] = {"refused": True, "reasons": [f"missing {p}"], "n_spikes": 0}
            continue
        rs = gate.read_records(p)
        cells[seed] = gate.gate_cell(rs, dataset="bnci2014_001", lr=lr,
                                     weight_decay=wd, growing=False)
        recs += rs
    return cells, recs


def evaluate(root):
    root = Path(root)
    aux = root / "aux"
    cap_rows = ([json.loads(l) for l in (aux / "capacity.jsonl").read_text().splitlines()
                 if l.strip()] if (aux / "capacity.jsonl").exists() else [])
    ref_rows = [json.loads(l) for p in sorted(aux.glob("reference_*.jsonl"))
                for l in p.read_text().splitlines() if l.strip()]
    out = {"reference": {m: reference(ref_rows, m) for m in PUBLISHED}, "arms": {}}
    for arm, model in ARMS.items():
        res = {}
        for cand in CANDIDATES:
            cells, recs = read_cells(root, arm, cand)
            entry = {"gate": cells,
                     "n_spikes": sum(c.get("n_spikes", 0) for c in cells.values())}
            if any(c["refused"] for c in cells.values()):
                entry["status"] = "gate refused: relaunch, criteria NOT read"
                entry["passes"] = None
            else:
                entry.update(status="read", departure=departure(recs),
                             capacity=capacity(cap_rows, arm, cand),
                             optimum=optimum(recs))
                ref = out["reference"][model]
                verdicts = [entry["capacity"]["refused"], entry["optimum"]["refused"],
                            ref["refused"]]
                entry["passes"] = (None if any(v is None for v in verdicts)
                                   else not any(verdicts))
            res[cand] = entry
        if res["B"]["passes"] is True:
            frozen = "B"
        elif res["B"]["passes"] is False and res["H"]["passes"] is True:
            frozen = "H"
        elif res["B"]["passes"] is False and res["H"]["passes"] is False:
            frozen = "REFUSED"
        else:
            frozen = "undecided (incomplete)"
        out["arms"][arm] = {"candidates": res, "frozen_config": frozen}
    return out


def _fmt(v):
    return "—" if v is None else ("REFUS" if v else "passe")


def markdown(out):
    lines = ["| bras | config | gate | départ (excès nat) | capacité (méd.) | "
             "optimum (≤5 / ≥195) | pics |", "|---|---|---|---|---|---|---|"]
    for arm, a in out["arms"].items():
        for cand, e in a["candidates"].items():
            if e["status"] != "read":
                lines.append(f"| {arm} | {cand} | {e['status']} | | | | {e['n_spikes']} |")
                continue
            c, o = e["capacity"], e["optimum"]
            cap = (f"{c['median']:.4f} {_fmt(c['refused'])}" if c["status"] == "read"
                   else c["status"])
            lines.append(
                f"| {arm} | {cand} | admis | {e['departure']['excess_nat']:+.3f} | {cap} | "
                f"{o['frac_early']:.1%} / {o['frac_late']:.1%} {_fmt(o['refused'])} | "
                f"{e['n_spikes']} |")
    lines += ["", "| modèle | moyenne | IC95 | publié | verdict |", "|---|---|---|---|---|"]
    for m, r in out["reference"].items():
        if r["status"] != "read":
            lines.append(f"| {m} | {r['status']} {r.get('subjects')} | | {PUBLISHED[m]} | — |")
        else:
            lines.append(f"| {m} | {r['mean']:.3f} | [{r['ci95'][0]:.3f}, "
                         f"{r['ci95'][1]:.3f}] | {r['published']} | {_fmt(r['refused'])} |")
    lines += ["", "Config gelée : " + ", ".join(f"{a} → **{v['frozen_config']}**"
                                               for a, v in out["arms"].items())]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="prefix for .json and .md")
    ap.add_argument("--growing", action="store_true",
                    help="read the growing arms of addendum 1 instead of lot 1")
    ap.add_argument("--grid", type=float, nargs="+", default=None,
                    help="addendum 2: line-search grid (its max is the censoring top)")
    ap.add_argument("--normalization", default=None, help="addendum 2")
    a = ap.parse_args(argv)
    if a.growing:
        out = evaluate_growing(a.root, grid_top=max(a.grid) if a.grid else GRID_TOP,
                               grid=a.grid, normalization=a.normalization)
        Path(a.out + ".json").write_text(json.dumps(out, indent=2, default=float))
        md = markdown_growing(out)
        Path(a.out + ".md").write_text(md + "\n")
        print(md)
        return 0
    out = evaluate(a.root)
    Path(a.out + ".json").write_text(json.dumps(out, indent=2, default=float))
    md = markdown(out)
    Path(a.out + ".md").write_text(md + "\n")
    print(md)
    return 0



# ------------------------------------------------------------------ addendum 1
# Growing arms (PREREG addendum 1, 14/10/2026). Each is read under the frozen config B
# of its family, against its published reference of the same architecture.
GROWING = {
    # arm: (family reference arm, (lr, wd) of config B, target width)
    "grow_shallow": ("bd_shallow", CANDIDATES["B"]["bd_shallow"], 40),
    "grow_deep4": ("bd_deep4", CANDIDATES["B"]["bd_deep4"], 25),
}
#: Excess over ln 4 of the reference, read in lot 1 (RESULTAT_step0_training_battery.md).
REFERENCE_EXCESS = {"bd_shallow": 0.460, "bd_deep4": 0.703}
DEPARTURE_MARGIN = 0.15
TARGET_FRAC, CALENDAR_EPOCH, CALENDAR_FRAC = 0.80, 10, 0.50
FANOUT_RATIO, SPECTRUM_DECADES, CENSOR_FRAC = 2.0, 1.0, 0.20
GRID_TOP = 4.0


def growth_events(rec):
    """Applied growth events of one fit, each with the number of places it filled."""
    width = rec.get("width_start")
    out = []
    for h in rec.get("history", []):
        if "grow_s" not in h:
            continue
        if h.get("grow_applied"):
            after = h.get("grow_width_after")
            out.append({**h, "places": None if width is None or after is None
                        else int(after) - int(width)})
            width = after
    return out


def all_events(recs):
    return [h for r in recs for h in r.get("history", []) if "grow_s" in h]


def growth_criteria(recs, target, grid_top=GRID_TOP):
    applied = [e for r in recs for e in growth_events(r)]
    res = {}
    # sélection réelle: gromo kept every candidate on every applied step, or offered
    # fewer than 2 candidates per place filled (median over applied steps).
    kept = [e["grow_n_kept"] / e["grow_n_proposed"] for e in applied
            if e.get("grow_n_proposed")]
    ratio = [e["grow_n_proposed"] / e["places"] for e in applied
             if e.get("places") and e.get("grow_n_proposed")]
    res["selection"] = {
        "n_applied": len(applied), "frac_kept_all": float(np.mean([k == 1.0 for k in kept]))
        if kept else None, "median_proposed_per_place": float(np.median(ratio)) if ratio else None,
        "refused": (not applied or all(k == 1.0 for k in kept)
                    or (float(np.median(ratio)) if ratio else 0.0) < FANOUT_RATIO)}
    # spectre: decades spanned by the positive proposed eigenvalues, median over steps.
    spans = []
    for e in all_events(recs):
        v = np.asarray([x for x in (e.get("grow_eig_proposed") or []) if x and x > 0], float)
        if len(v) >= 2:
            spans.append(float(np.log10(v.max() / v.min())))
    med_span = float(np.median(spans)) if spans else None
    res["spectrum"] = {"median_decades": med_span,
                       "refused": med_span is None or med_span < SPECTRUM_DECADES}
    # cible
    hit = [r.get("width_end") == target for r in recs]
    res["target"] = {"frac_reached": float(np.mean(hit)),
                     "refused": float(np.mean(hit)) < TARGET_FRAC}
    # calendrier: a fit whose every applied step falls in the first 10 epochs is a
    # one-shot inflation, not a growth; refused if that is the majority of fits that grew.
    early = [max(e["epoch"] for e in growth_events(r)) <= CALENDAR_EPOCH
             for r in recs if growth_events(r)]
    res["calendar"] = {"frac_all_early": float(np.mean(early)) if early else None,
                       "refused": (not early) or float(np.mean(early)) > CALENDAR_FRAC}
    # censure de la line search
    s = [float(e["grow_s"]) for e in applied]
    top = float(np.mean([x >= grid_top for x in s])) if s else None
    res["censoring"] = {"frac_top": top, "refused": top is None or top > CENSOR_FRAC}
    return res


def read_growing_cells(root, arm, grid=None, normalization=None):
    ref, (lr, wd), _ = GROWING[arm]
    cells, recs = {}, []
    for seed in SEEDS:
        p = gate.cell_path(Path(root) / "B" / "results", "within_session",
                           "bnci2014_001", arm, seed)
        if not p.exists():
            cells[seed] = {"refused": True, "reasons": [f"missing {p}"], "n_spikes": 0}
            continue
        rs = gate.read_records(p)
        cells[seed] = gate.gate_cell(rs, dataset="bnci2014_001", lr=lr,
                                     weight_decay=wd, growing=True, grid=grid,
                                     normalization=normalization)
        recs += rs
    return cells, recs


def evaluate_growing(root, grid_top=GRID_TOP, grid=None, normalization=None):
    root = Path(root)
    aux = root / "aux"
    cap_rows = ([json.loads(l) for l in (aux / "capacity.jsonl").read_text().splitlines()
                 if l.strip()] if (aux / "capacity.jsonl").exists() else [])
    out = {}
    for arm, (ref, _, target) in GROWING.items():
        cells, recs = read_growing_cells(root, arm, grid, normalization)
        e = {"gate": cells, "n_spikes": sum(c.get("n_spikes", 0) for c in cells.values())}
        if any(c["refused"] for c in cells.values()):
            e.update(status="gate refused: relaunch, criteria NOT read", passes=None)
            out[arm] = e
            continue
        dep = departure(recs)
        dep["limit"] = REFERENCE_EXCESS[ref] + DEPARTURE_MARGIN
        dep["refusing"] = True
        dep["refused"] = dep["excess_nat"] > dep["limit"]
        e.update(status="read", departure=dep, capacity=capacity(cap_rows, arm, "B"),
                 optimum=optimum(recs), growth=growth_criteria(recs, target, grid_top),
                 twin={"arm": ref, "refused": False})
        # Addendum 2 reserve: the grid was chosen on subjects 1-3 (seed 0), so the
        # censoring is also REPORTED (not decided) on the subjects it never saw.
        unseen = [r for r in recs if str(r.get("subject")) not in {"1", "2", "3"}]
        e["censoring_subjects_4_9"] = growth_criteria(unseen, target, grid_top)["censoring"]
        verdicts = ([dep["refused"], e["capacity"]["refused"], e["optimum"]["refused"]]
                    + [g["refused"] for g in e["growth"].values()])
        e["passes"] = None if any(v is None for v in verdicts) else not any(verdicts)
        out[arm] = e
    return out


def markdown_growing(out):
    lines = ["| bras | départ (≤ limite) | capacité | optimum ≤5/≥195 | sélection "
             "(prop./place, tout gardé) | spectre (déc.) | cible | calendrier | censure s=4 "
             "| verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, e in out.items():
        if e["status"] != "read":
            lines.append(f"| {arm} | {e['status']} |||||||||")
            continue
        d, c, o, g = e["departure"], e["capacity"], e["optimum"], e["growth"]
        cap = f"{c['median']:.4f} {_fmt(c['refused'])}" if c["status"] == "read" else c["status"]
        sel = g["selection"]
        lines.append(
            f"| {arm} | {d['excess_nat']:+.3f} (≤ {d['limit']:.3f}) {_fmt(d['refused'])} | "
            f"{cap} | {o['frac_early']:.1%} / {o['frac_late']:.1%} {_fmt(o['refused'])} | "
            f"{sel['median_proposed_per_place']} , {sel['frac_kept_all']} {_fmt(sel['refused'])} | "
            f"{g['spectrum']['median_decades']} {_fmt(g['spectrum']['refused'])} | "
            f"{g['target']['frac_reached']:.1%} {_fmt(g['target']['refused'])} | "
            f"{g['calendar']['frac_all_early']} {_fmt(g['calendar']['refused'])} | "
            f"{g['censoring']['frac_top']} {_fmt(g['censoring']['refused'])} | "
            f"**{'passe' if e['passes'] else ('REFUS' if e['passes'] is False else '—')}** |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
