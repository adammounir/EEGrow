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
    a = ap.parse_args(argv)
    out = evaluate(a.root)
    Path(a.out + ".json").write_text(json.dumps(out, indent=2, default=float))
    md = markdown(out)
    Path(a.out + ".md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
