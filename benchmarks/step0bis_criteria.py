"""Step 0-bis reading: the BEHAVIOUR criteria of step 0 on shin2017a and physionetmi.

WHY THIS EXISTS
---------------
Step 0 froze its protocol on bnci2014_001, and addendum 2 chose its line-search grid
on bnci2014_001 subjects 1-3. The cadrage (§ 5, decision 2) requires the behaviour
criteria to hold on ``shin2017a`` and ``physionetmi`` too: the two datasets where the
previous campaign scored 59 % / 39 % of fits on a barely trained epoch, with
validation and growth held-out splits of a few trials. A criterion that only holds
where it was tuned is not a property of the protocol.

Committed BEFORE any 0-bis cell is read (PREREG_step0bis_transfer.md). Nothing here
reads an accuracy.

Per (arm, dataset), over seeds 0-2, all thresholds those of step 0 / addendum 1:
  gate        step0_gate on every cell (dataset structure, frozen config, addendum-2
              grid and normalisation on growing arms); a refused cell is relaunched,
              its criteria are NOT read
  optimum     > 20 % of fits with selected_epoch <= 5, or > 20 % >= 195 -> refuse
  growth      (growing arms) selection, spectrum, target, calendar, censoring at
              s = 64 -> refuse, exactly ``step0_criteria.growth_criteria``
  départ      median(first_batch_loss) - ln(2), REPORTED only (the refusing limit of
              addendum 1 is relative to a 4-class reference measured on bnci)
An (arm, dataset) passes if no criterion refuses it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step0_criteria as crit  # noqa: E402
import step0_gate as gate  # noqa: E402

DATASETS = ("shin2017a", "physionetmi")
N_CLASSES = {"shin2017a": 2, "physionetmi": 2}
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
NORMALIZATION = "match_extending_layer"
#: arm -> (lr, weight decay) of config B, target width (None: not growing).
ARMS = {
    "bd_shallow": (crit.CANDIDATES["B"]["bd_shallow"], None),
    "bd_deep4": (crit.CANDIDATES["B"]["bd_deep4"], None),
    "grow_shallow": (crit.GROWING["grow_shallow"][1], crit.GROWING["grow_shallow"][2]),
    "grow_deep4": (crit.GROWING["grow_deep4"][1], crit.GROWING["grow_deep4"][2]),
}


def evaluate_cell_set(root, dataset, arm):
    (lr, wd), target = ARMS[arm]
    growing = target is not None
    cells, recs = {}, []
    for seed in crit.SEEDS:
        p = gate.cell_path(Path(root) / "B" / "results", "within_session", dataset,
                           arm, seed)
        if not p.exists():
            cells[seed] = {"refused": True, "reasons": [f"missing {p}"], "n_spikes": 0}
            continue
        try:
            rs = gate.read_records(p)
        except ValueError as exc:
            cells[seed] = {"refused": True, "reasons": [str(exc)], "n_spikes": 0}
            continue
        cells[seed] = gate.gate_cell(
            rs, dataset=dataset, lr=lr, weight_decay=wd, growing=growing,
            grid=list(GRID) if growing else None,
            normalization=NORMALIZATION if growing else None)
        recs += rs
    e = {"gate": cells, "n_spikes": sum(c.get("n_spikes", 0) for c in cells.values())}
    if any(c["refused"] for c in cells.values()):
        e.update(status="gate refused: relaunch, criteria NOT read", passes=None)
        return e
    fb = np.array([r["first_batch_loss"] for r in recs], float)
    e["departure"] = {"median_first_batch_loss": float(np.median(fb)),
                      "excess_nat": float(np.median(fb) - math.log(N_CLASSES[dataset])),
                      "refusing": False}
    e["optimum"] = crit.optimum(recs)
    verdicts = [e["optimum"]["refused"]]
    if growing:
        e["growth"] = crit.growth_criteria(recs, target, grid_top=max(GRID))
        verdicts += [g["refused"] for g in e["growth"].values()]
    e["status"] = "read"
    e["passes"] = None if any(v is None for v in verdicts) else not any(verdicts)
    return e


def evaluate(root):
    return {ds: {arm: evaluate_cell_set(root, ds, arm) for arm in ARMS}
            for ds in DATASETS}


def _f(x, fmt="{:.1%}"):
    return "—" if x is None else fmt.format(x)


def markdown(out):
    lines = ["| dataset | bras | optimum ≤5 / ≥195 | prop./place | spectre (déc.) | cible "
             "| tout tôt | censure (s=64) | départ (excès ln 2) | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, arms in out.items():
        for arm, e in arms.items():
            if e["status"] != "read":
                lines.append(f"| {ds} | {arm} | {e['status']} ||||||||")
                continue
            o, g = e["optimum"], e.get("growth")
            cells = [f"{_f(o['frac_early'])} / {_f(o['frac_late'])}"]
            if g:
                cells += [_f(g["selection"]["median_proposed_per_place"], "{:.1f}"),
                          _f(g["spectrum"]["median_decades"], "{:.2f}"),
                          _f(g["target"]["frac_reached"]),
                          _f(g["calendar"]["frac_all_early"]),
                          _f(g["censoring"]["frac_top"], "{:.2%}")]
            else:
                cells += ["n/a"] * 5
            verdict = {True: "passe", False: "REFUS", None: "—"}[e["passes"]]
            lines.append(f"| {ds} | {arm} | " + " | ".join(cells)
                         + f" | {e['departure']['excess_nat']:+.3f} | **{verdict}** |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="…/step0_results (holds B/results)")
    ap.add_argument("--out", required=True, help="prefix: writes .json and .md")
    a = ap.parse_args(argv)
    out = evaluate(a.root)
    Path(a.out + ".json").write_text(json.dumps(out, indent=2, default=float))
    md = markdown(out)
    Path(a.out + ".md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
