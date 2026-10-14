"""Step-0 gates: G-lancement, G-complétude, G-pic. They refuse; they never warn.

WHY THIS EXISTS
---------------
Every correction of the step-0 protocol is a *launch override* whose config default
is the historical, losing value (``patience``, ``selection_monitor``,
``smooth_window``, ``restore_selected``, ``growth_holdout``, ``scaling_grid``,
``weight_decay``). A cell launched without one of them writes well-formed records that
look exactly like a correct cell. The only defence is to read what *ran* -- the stamps
``FitRecorder`` writes off the live callbacks and the live optimizer -- and to refuse
the cell on a single mismatch, before anything else reads it.

The previous campaign also taught two silent failure modes this gate closes:

* a cancelled-then-relaunched cell appends its fits to the same JSONL, so a cell can
  hold phantom fits that a fit count alone does not reveal (duplicate
  ``(subject, session, cv_ind)``);
* gradient spikes up to 1e16 that tracked the execution context, not the model; the
  threshold (1e6) sits in an empty band of 167 076 fits (healthy <= 5.8e4, spikes
  >= 5.6e6; ``analysis/step0_catalogue/spike_threshold.py``).

HOW
---
One JSONL file is one cell: ``<root>/<eval>/<dataset>/<arm>__seed<k>__fits.jsonl``.
``gate_cell`` returns the list of reasons the cell is refused (empty = admitted).
The CLI exits 1 if any cell is refused, and writes the verdict as JSON so the
relaunch list is a file, not a reading of a log.

Usage::

    python step0_gate.py --root RESULTS --eval within_session --dataset bnci2014_001 \\
        --arm bd_deep4 --seeds 0 1 2 --lr 6.25e-4 --weight-decay 0.01 \\
        --out verdict.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

#: The frozen step-0 selection stamp (PREREG § 3). Exact equality, field by field.
SELECTION = {
    "selection_monitor": "valid_acc",
    "lower_is_better": False,
    "tie_break": "earliest",
    "smooth_window": 5,
    "swap": False,
    "stop_monitor": "valid_loss",
    "patience": 200,
}
MAX_EPOCHS = 200
SPIKE_THRESHOLD = 1e6
#: Datasets the step-0 battery runs on, with the session names MOABB serves.
#: shin2017a / physionetmi (step 0-bis, 14/10): MOABB ``Shin2017A`` serves 3 MI sessions
#: (``sessions_per_subject=3``, the even-numbered recordings) and ``PhysionetMI`` one;
#: names and subject counts read off the complete historical cells
#: ``results_final/within_session/{shin2017a,physionetmi}/grow_shallow__seed0__fits.jsonl``
#: (435 fits = 29 x 3 x 5, 545 fits = 109 x 1 x 5, no subject missing).
SESSIONS = {"bnci2014_001": ("0train", "1test"),
            "shin2017a": ("0imagery", "2imagery", "4imagery"),
            "physionetmi": ("0",)}
N_SUBJECTS = {"bnci2014_001": 9, "shin2017a": 29, "physionetmi": 109}
N_FOLDS = 5


def cell_path(root, evaluation, dataset, arm, seed) -> Path:
    return Path(root) / evaluation / dataset / f"{arm}__seed{seed}__fits.jsonl"


def read_records(path) -> list[dict]:
    """All records of a cell. A truncated last line is a refusal, not a skip."""
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {i + 1} is not JSON ({exc}); "
                                 "a cell still being written cannot be gated") from exc
    return out


def _close(a, b) -> bool:
    return a is not None and b is not None and math.isclose(float(a), float(b),
                                                             rel_tol=1e-9)


def launch_violations(rec: dict, *, lr: float, weight_decay: float,
                      growing: bool, grid=None, normalization=None) -> list[str]:
    """G-lancement on one fit record: did this fit run the frozen protocol?"""
    bad = []
    sel = rec.get("selection") or {}
    for key, want in SELECTION.items():
        if sel.get(key) != want:
            bad.append(f"selection.{key}={sel.get(key)!r} != {want!r}")
    if rec.get("max_epochs") != MAX_EPOCHS:
        bad.append(f"max_epochs={rec.get('max_epochs')!r} != {MAX_EPOCHS}")
    if rec.get("epochs") != MAX_EPOCHS:
        # patience 200 >= max_epochs: a shorter fit is a crash or a wrong patience.
        bad.append(f"epochs={rec.get('epochs')!r} != {MAX_EPOCHS}")
    opt = rec.get("optimizer") or {}
    if opt.get("class") != "AdamW":
        bad.append(f"optimizer.class={opt.get('class')!r}")
    if not _close(opt.get("lr"), lr):
        bad.append(f"optimizer.lr={opt.get('lr')!r} != {lr}")
    if not _close(opt.get("weight_decay"), weight_decay):
        bad.append(f"optimizer.weight_decay={opt.get('weight_decay')!r} != {weight_decay}")
    # Constant schedule (train.lr_schedule=null): every epoch ran at the frozen lr.
    lrs = {h.get("lr") for h in rec.get("history", [])}
    if not lrs or any(not _close(v, lr) for v in lrs):
        bad.append(f"per-epoch lr not constant at {lr}: {sorted(map(str, lrs))[:3]}")
    if not (rec.get("n_holdout") or 0) > 0:
        bad.append(f"n_holdout={rec.get('n_holdout')!r}: growth held-out missing")
    if not rec.get("selected_path"):
        bad.append("selected weights not saved (save_selected)")
    if rec.get("first_batch_loss") is None:
        bad.append("first_batch_loss missing")
    if growing:
        events = [h for h in rec.get("history", []) if "grow_s" in h]
        kinds = Counter(h.get("grow_holdout") for h in events)
        if set(kinds) - {"disjoint"}:
            bad.append(f"grow_holdout not all disjoint: {dict(kinds)}")
        # Addendum 2: the line-search grid and the normalisation that actually ran.
        if grid is not None:
            want = sorted(float(g) for g in grid)
            off = [h["epoch"] for h in events
                   if sorted(float(k) for k in (h.get("grow_losses") or {})) != want]
            if off:
                bad.append(f"line-search grid != {want} on {len(off)} event(s)")
        norms = {h.get("grow_normalization") for h in events}
        if events and norms != {normalization}:
            bad.append(f"grow_normalization {sorted(map(str, norms))} != {normalization!r}")
    return bad


def completeness_violations(recs: list[dict], *, dataset: str) -> list[str]:
    """G-complétude: every (subject, session, fold) exactly once, nothing else."""
    bad = []
    keys = Counter((str(r.get("subject")), str(r.get("session")), r.get("cv_ind"))
                   for r in recs)
    dup = {k: n for k, n in keys.items() if n > 1}
    if dup:
        bad.append(f"{sum(dup.values()) - len(dup)} phantom fit(s): duplicated "
                   f"(subject, session, cv_ind) {sorted(dup)[:3]}")
    subjects = sorted({k[0] for k in keys}, key=lambda s: (len(s), s))
    if len(subjects) != N_SUBJECTS[dataset]:
        bad.append(f"{len(subjects)} subjects != {N_SUBJECTS[dataset]}")
    sessions = SESSIONS[dataset]
    want = N_SUBJECTS[dataset] * len(sessions) * N_FOLDS
    # MOABB numbers cv_ind across the WHOLE cell, in an order that varies (read off
    # the 13/10 smoke: 0..29 over 3 subjects, subject 3 first). So the check is: each
    # of 0..want-1 exactly once, and N_FOLDS fits per (subject, session).
    folds = Counter(k[2] for k in keys.elements())
    if sorted(folds) != list(range(want)) or any(n != 1 for n in folds.values()):
        missing = sorted(set(range(want)) - set(folds))
        bad.append(f"cv_ind not exactly 0..{want - 1} once each "
                   f"(missing {missing[:5]}{'...' if len(missing) > 5 else ''})")
    per_ss = Counter((k[0], k[1]) for k in keys.elements())
    for s in subjects:
        sess = {k[1] for k in per_ss if k[0] == s}
        if sess != set(sessions):
            bad.append(f"subject {s}: sessions {sorted(sess)} != {list(sessions)}")
    off = {k: n for k, n in per_ss.items() if n != N_FOLDS}
    if off:
        bad.append(f"(subject, session) without exactly {N_FOLDS} fits: "
                   f"{sorted(off.items())[:3]}")
    per_subject = Counter(k[0] for k in keys.elements())
    if len(set(per_subject.values())) > 1:
        bad.append(f"fits/subject not constant: {dict(per_subject)}")
    if len(recs) != want:
        bad.append(f"{len(recs)} fits != {want}")
    return bad


def spiked(rec: dict) -> bool:
    v = rec.get("grad_norm_max_fit")
    return v is None or not math.isfinite(float(v)) or float(v) > SPIKE_THRESHOLD


def gate_cell(recs: list[dict], *, dataset: str, lr: float, weight_decay: float,
              growing: bool, grid=None, normalization=None) -> dict:
    """Verdict for one cell. ``refused`` is True on any violation of any gate."""
    reasons = completeness_violations(recs, dataset=dataset)
    launch = Counter()
    for r in recs:
        for v in launch_violations(r, lr=lr, weight_decay=weight_decay,
                                   growing=growing, grid=grid,
                                   normalization=normalization):
            launch[v] += 1
    reasons += [f"G-lancement: {v} ({n} fit(s))" for v, n in launch.items()]
    spikes = [r.get("fit_uid") for r in recs if spiked(r)]
    if spikes:
        reasons.append(f"G-pic: {len(spikes)} fit(s) with grad_norm_max_fit > "
                       f"{SPIKE_THRESHOLD:g} or missing")
    return {"refused": bool(reasons), "reasons": reasons, "n_fits": len(recs),
            "n_spikes": len(spikes), "spiked_fit_uids": spikes,
            "gpu_uuids": sorted({str(r.get("gpu_uuid")) for r in recs})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval", dest="evaluation", default="within_session")
    ap.add_argument("--dataset", default="bnci2014_001", choices=sorted(SESSIONS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--growing", action="store_true",
                    help="also require every growth event to use the disjoint held-out")
    ap.add_argument("--grid", type=float, nargs="+", default=None,
                    help="expected line-search grid on every growth event (addendum 2)")
    ap.add_argument("--normalization", default=None,
                    help="expected grow_normalization (addendum 2); default: none")
    ap.add_argument("--out", required=True, help="verdict JSON")
    a = ap.parse_args(argv)

    verdict = {"arm": a.arm, "dataset": a.dataset, "eval": a.evaluation,
               "lr": a.lr, "weight_decay": a.weight_decay, "cells": {}}
    for seed in a.seeds:
        path = cell_path(a.root, a.evaluation, a.dataset, a.arm, seed)
        if not path.exists():
            cell = {"refused": True, "reasons": [f"missing {path}"], "n_fits": 0,
                    "n_spikes": 0, "spiked_fit_uids": [], "gpu_uuids": []}
        else:
            try:
                recs = read_records(path)
            except ValueError as exc:
                recs, cell = None, {"refused": True, "reasons": [str(exc)],
                                    "n_fits": 0, "n_spikes": 0,
                                    "spiked_fit_uids": [], "gpu_uuids": []}
            if recs is not None:
                cell = gate_cell(recs, dataset=a.dataset, lr=a.lr,
                                 weight_decay=a.weight_decay, growing=a.growing,
                                 grid=a.grid, normalization=a.normalization)
        verdict["cells"][f"seed{seed}"] = cell
    verdict["refused_cells"] = [k for k, c in verdict["cells"].items() if c["refused"]]
    verdict["n_spikes"] = sum(c["n_spikes"] for c in verdict["cells"].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(verdict, indent=2))
    for k, c in verdict["cells"].items():
        print(f"{a.arm} {k}: {'REFUSED' if c['refused'] else 'admitted'} "
              f"({c['n_fits']} fits, {c['n_spikes']} spikes)")
        for r in c["reasons"]:
            print(f"    - {r}")
    return 1 if verdict["refused_cells"] else 0


if __name__ == "__main__":
    sys.exit(main())
