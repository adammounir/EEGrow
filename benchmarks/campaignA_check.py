"""Integrity check of finished campaign-A cells. Reads files, never a contrast.

For every cell of ``--cells`` whose CSV exists (unfinished cells are counted, not
checked), against what the cell line and the manifest say it must contain:

  rows     one CSV row per session of the subject (within: the 5 folds of a session
           are averaged by MOABB; cross_session / cross_subject: one row per test session)
  fits     JSONL lines: within 5 x sessions, cross_session = sessions, cross_subject 1;
           every line parses; ``subject`` == the cell's subject
  identity within / cross_session train_subjects == {subj}; cross_subject subj not in
           train_subjects; test_sessions all of subj; cross_session train and test
           sessions disjoint
  files    init / growth / final / selected weights and >= 1 logits file per fit;
           ckpt with epochs [10, 25, 50, 100] on cross_session / cross_subject
  growth   bd: no growth event; tiny / random: every growth row carries the arm's
           direction, growth file events == applied epochs, init width != final width
           when grown
  score    logits reproduce the scorer on the float32 softmax that skorch's
           predict_proba returns: accuracy = its argmax, ROC-AUC (2 classes) on p[:, 1].
           Not l1 - l0: the float32 softmax saturates at 1.0 for large logits and the
           scorer ranks those trials as ties (up to 43/160 in the canary, AUC -0.05).
           Non-finite logits
           (a diverged net) the constant predictor, flagged ``diverged``; per-session
           mean of the logit-file scores == CSV row within 1e-6 (CSV is float32)

Stability (non-finite loss, grad > 1e6) and growth (width_end > width_start) are
COUNTED here, decided by the reader. Exit 1 on any integrity problem.

    python campaignA_check.py --cells campaignA_cells.txt --manifest ../manifest [--only-ids 0-35]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("init", "growth", "final", "selected")
CKPT = [10, 25, 50, 100]


def parse_cells(path, only=None):
    for line in open(path):
        f = line.split()
        cid = int(f[0])
        if only is not None and cid not in only:
            continue
        ov = f[9:]
        yield dict(id=cid, prot=f[1], ds=f[2], model=f[3], arm=f[4], seed=int(f[5]), subj=f[6],
                   rdir=f[7], csv=f[8], ckpt=any(o.startswith("train.save_checkpoints=") for o in ov),
                   max_epochs=next((int(o.split("=")[1]) for o in ov if o.startswith("train.max_epochs=")), None))


def score_of(z, n_classes):
    lg, y = z["logits"], z["y"]
    classes = np.unique(y)
    if not np.isfinite(lg).all():
        # a diverged net is scored as the constant predictor (subject_stamp.constant_score)
        return 0.5 if n_classes == 2 else float((y == classes[0]).mean())
    import torch
    # what the scorer saw: skorch predict_proba = float32 softmax of the forward
    p = torch.softmax(torch.as_tensor(lg, dtype=torch.float32), dim=1).numpy()
    if n_classes == 2:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y == classes[1], p[:, 1]))
    idx = {c: i for i, c in enumerate(classes)}
    return float((p.argmax(1) == np.array([idx[c] for c in y])).mean())


def check_cell(c, man, sizes, counters, ckpt_epochs=CKPT):
    bad = []
    sub = man[c["ds"]]["subjects"][c["subj"]]
    sessions = sorted(sub["sessions"])
    n_classes = len(man[c["ds"]]["classes"])
    stem = c["csv"][:-4]
    csv = pd.read_csv(c["csv"])
    if sorted(map(str, csv["session"])) != sessions:
        bad.append(f"csv sessions {sorted(map(str, csv['session']))} != {sessions}")
    if set(map(str, csv["subject"])) != {c["subj"]}:
        bad.append(f"csv subjects {set(csv['subject'])}")
    exp_fits = {"within_session": 5 * len(sessions), "cross_session": len(sessions),
                "cross_subject": 1}[c["prot"]]
    recs = []
    for i, line in enumerate(open(stem + "__fits.jsonl")):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(f"jsonl line {i} does not parse")
    sizes["jsonl"].append(os.path.getsize(stem + "__fits.jsonl"))
    if len(recs) != exp_fits:
        bad.append(f"{len(recs)} fits, expected {exp_fits}")
    per_session = collections.defaultdict(list)
    for r in recs:
        uid = r["fit_uid"]
        if str(r.get("subject")) != c["subj"]:
            bad.append(f"{uid[:8]} subject {r.get('subject')}")
        tr, te = set(r.get("train_subjects") or []), r.get("test_sessions") or []
        if c["prot"] == "cross_subject":
            if c["subj"] in tr or not tr:
                bad.append(f"{uid[:8]} train_subjects {sorted(tr)[:5]} holds the test subject")
        elif tr != {c["subj"]}:
            bad.append(f"{uid[:8]} train_subjects {sorted(tr)}")
        if not te or any(not s.startswith(f"{c['subj']}:") for s in te):
            bad.append(f"{uid[:8]} test_sessions {te}")
        if c["prot"] == "cross_session" and set(r.get("train_sessions") or []) & set(te):
            bad.append(f"{uid[:8]} train and test sessions overlap")
        if c["max_epochs"] and len(r["history"]) != c["max_epochs"]:
            bad.append(f"{uid[:8]} {len(r['history'])} epochs, expected {c['max_epochs']}")
        for k in KINDS:
            p = f"{stem}__fits__{k}/{uid}.pt"
            if not os.path.exists(p):
                bad.append(f"{uid[:8]} missing {k}")
            else:
                sizes[k].append(os.path.getsize(p))
        import torch
        if c["ckpt"]:
            p = f"{stem}__fits__ckpt/{uid}.pt"
            if not os.path.exists(p):
                bad.append(f"{uid[:8]} missing ckpt")
            else:
                sizes["ckpt"].append(os.path.getsize(p))
                got = sorted(torch.load(p, weights_only=False)["checkpoints"])
                want = [e for e in ckpt_epochs if e <= len(r["history"])]
                if got != want or r.get("ckpt_epochs") != want:
                    bad.append(f"{uid[:8]} ckpt epochs {got}, expected {want}")
        elif os.path.exists(f"{stem}__fits__ckpt/{uid}.pt"):
            bad.append(f"{uid[:8]} ckpt written on a within_session cell (option B)")
        applied = [h["epoch"] for h in r["history"] if h.get("grow_applied")]
        dirs = {h.get("grow_direction") for h in r["history"] if "grow_s" in h}
        if c["arm"] == "bd":
            if applied or dirs:
                bad.append(f"{uid[:8]} bd arm with growth rows {applied}")
        else:
            if dirs != {c["arm"]}:
                bad.append(f"{uid[:8]} directions {dirs} != {{{c['arm']}}}")
            g = torch.load(f"{stem}__fits__growth/{uid}.pt", weights_only=False)["events"]
            if [e["epoch"] for e in g] != applied or r.get("n_growth_snapshots") != len(applied):
                bad.append(f"{uid[:8]} growth events {[e['epoch'] for e in g]} != applied {applied}")
            if any(not e["tensors"] for e in g):
                bad.append(f"{uid[:8]} growth event without tensors")
            counters["grew"] += bool(r.get("width_end") and r.get("width_start")
                                     and r["width_end"] > r["width_start"])
            counters["growing_fits"] += 1
        fin = all(isinstance(h.get(k), (int, float)) and math.isfinite(h[k])
                  for h in r["history"] for k in ("train_loss", "valid_loss"))
        gm = r.get("grad_norm_max_fit")
        counters["fits"] += 1
        counters["nonfinite"] += not fin
        counters["spikes"] += not (isinstance(gm, (int, float)) and math.isfinite(gm) and gm <= 1e6)
        counters[f"gpu:{r.get('gpu_name')}"] += 1
        lg = sorted(glob.glob(f"{stem}__fits__logits/{uid}*.npz"))
        if not lg:
            bad.append(f"{uid[:8]} no logits")
        n_te = 0
        for p in lg:
            z = np.load(p)
            sizes["logits"].append(os.path.getsize(p))
            s = score_of(z, n_classes)
            if abs(s - float(z["score"])) > 1e-6:
                bad.append(f"{uid[:8]} logits give {s:.6f}, scorer {float(z['score']):.6f}")
            div = not np.isfinite(z["logits"]).all()
            if "diverged" not in z or bool(z["diverged"]) != div:
                bad.append(f"{uid[:8]} diverged flag {z.get('diverged')} != non-finite logits {div}")
            counters["diverged_logit_files"] += div
            if len({len(z[k]) for k in ("logits", "y", "test_idx", "subject", "session")}) != 1:
                bad.append(f"{uid[:8]} logits arrays of unequal length")
            if set(z["subject"]) != {c["subj"]}:
                bad.append(f"{uid[:8]} logits subjects {set(z['subject'])}")
            if len(set(z["session"])) != 1:
                bad.append(f"{uid[:8]} one logits file spans sessions {set(z['session'])}")
            per_session[str(z["session"][0])].append(float(z["score"]))
            n_te += len(z["y"])
        if r.get("n_test") is not None and n_te != r["n_test"]:
            bad.append(f"{uid[:8]} n_test {r['n_test']} != {n_te} logit trials")
    for _, row in csv.iterrows():
        got = per_session.get(str(row["session"]))
        if not got or abs(np.mean(got) - float(row["score"])) > 1e-6:
            bad.append(f"session {row['session']}: logits {got} != csv {row['score']}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--only-ids", default=None, help="a-b range of cell ids")
    ap.add_argument("--epochs", type=int, default=None,
                    help="expected epochs when the cells ran with an EXTRA max_epochs override")
    a = ap.parse_args()
    only = None
    if a.only_ids:
        lo, hi = map(int, a.only_ids.split("-"))
        only = set(range(lo, hi + 1))
    man = {Path(f).stem: json.load(open(f)) for f in glob.glob(f"{a.manifest}/*.json")}
    sizes, counters = collections.defaultdict(list), collections.Counter()
    n_done = n_todo = 0
    problems = {}
    for c in parse_cells(a.cells, only):
        if a.epochs:
            c["max_epochs"] = a.epochs
        if not os.path.exists(c["csv"]):
            n_todo += 1
            continue
        n_done += 1
        try:
            bad = check_cell(c, man, sizes, counters)
        except Exception as e:  # a crash of the checker on a cell is a problem of that cell
            bad = [f"checker crashed: {type(e).__name__}: {e}"]
        if bad:
            problems[c["id"]] = (c, bad)
    print(f"cells done {n_done}, not done {n_todo}")
    print("fits:", {k: v for k, v in counters.items() if not k.startswith("gpu:")})
    print("cards:", {k[4:]: v for k, v in counters.items() if k.startswith("gpu:")})
    print("mean file size (KB):", {k: round(np.mean(v) / 1024, 1) for k, v in sizes.items() if v})
    tot = sum(sum(v) for v in sizes.values())
    print(f"bytes on disk for the checked cells: {tot / 2**30:.2f} GiB")
    if problems:
        print(f"\nPROBLEMS in {len(problems)} cells:")
        for cid, (c, bad) in sorted(problems.items()):
            print(f" - cell {cid} {c['prot']} {c['ds']} {c['model']}__{c['arm']} seed{c['seed']} subj{c['subj']}:")
            for b in bad[:8]:
                print("     ", b)
        return 1
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
