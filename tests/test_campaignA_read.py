"""campaignA_read.py on a synthetic grid: Q1 / Q2 and the Shallow init control (C1-C3).

Scores are built with known per-subject deltas, so the reader must recover them, keep
C1 + C2 = Q1 subject by subject, apply the PREREG_shallow_init_control § 3 reading,
and refuse a control arm whose growth rows do not match it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
PROTS = ["within_session", "cross_session", "cross_subject"]
DIRECTION = {"bd": None, "fix": None, "tiny": "tiny", "random": "random", "tiny_bdinit": "tiny"}
N_SUBJ = 12


def write_fit(stem, subj, arm):
    hist = [{"epoch": 1, "train_loss": 1.0, "valid_loss": 1.0}]
    if DIRECTION[arm] is not None:
        hist.append({"epoch": 5, "train_loss": 0.9, "valid_loss": 0.9, "grow_s": 1.0,
                     "grow_applied": True, "grow_direction": DIRECTION[arm]})
    rec = {"fit_uid": f"{arm}{subj}", "subject": subj, "history": hist, "grad_norm_max_fit": 3.0,
           "width_start": 8, "width_end": 40 if DIRECTION[arm] else 8, "gpu_name": "NVIDIA A40"}
    Path(stem + "__fits.jsonl").write_text(json.dumps(rec) + "\n")


def build(tmp, offsets, models=("shallow", "eegnex")):
    """offsets[arm] = delta added to bd's score; control arms only for shallow."""
    rng = np.random.default_rng(1)
    man = tmp / "manifest"
    man.mkdir()
    (man / "ds4.json").write_text(json.dumps({"classes": ["a", "b", "c", "d"]}))
    lines = {"A": [], "S": []}
    cid = 0
    for prot in PROTS:
        for m in models:
            base = 0.55 + 0.05 * rng.standard_normal(N_SUBJ)
            arms = ["bd", "tiny", "random"] + (["tiny_bdinit", "fix"] if m == "shallow" else [])
            for arm in arms:
                noise = 0.01 * rng.standard_normal(N_SUBJ)
                for i in range(N_SUBJ):
                    subj = str(i + 1)
                    d = tmp / "res" / f"{m}__{arm}" / prot
                    d.mkdir(parents=True, exist_ok=True)
                    csv = d / f"{arm}__seed0__subj{subj}.csv"
                    score = base[i] + offsets[arm] + (0 if arm == "bd" else noise[i])
                    csv.write_text(f"session,subject,score\n0,{subj},{score:.9f}\n")
                    write_fit(str(csv)[:-4], subj, arm)
                    grid = "S" if arm in ("tiny_bdinit", "fix") else "A"
                    lines[grid].append(f"{cid} {prot} ds4 {m} {arm} 0 {subj} {d} {csv} model=x")
                    cid += 1
    for g, ls in lines.items():
        (tmp / f"cells_{g}.txt").write_text("\n".join(ls) + "\n")
    return man


def read(tmp, man):
    out = tmp / "read" / "campaignA"
    r = subprocess.run([sys.executable, str(BENCH / "campaignA_read.py"),
                        "--cells", str(tmp / "cells_A.txt"), "--cells", str(tmp / "cells_S.txt"),
                        "--manifest", str(man), "--out", str(out)],
                       cwd=BENCH, capture_output=True, text=True)
    return r, (json.load(open(str(out) + ".json")) if r.returncode == 0 else None)


def test_growth_effect_survives_the_init(tmp_path):
    tmp = tmp_path
    man = build(tmp, {"bd": 0, "tiny": 0.05, "random": 0.04, "tiny_bdinit": 0.03, "fix": 0.01})
    r, out = read(tmp, man)
    assert r.returncode == 0, r.stdout + r.stderr
    c1 = out["pooled"]["C1"]
    assert len(c1) == 3 and {x["model"] for x in c1} == {"shallow"}
    for q, want in (("Q1", 0.05), ("C1", 0.03), ("C2", 0.02), ("C3", 0.01)):
        for row in out["pooled"][q]:
            if row["model"] == "shallow":
                assert abs(row["D"] - want) < 0.01, (q, row)
    assert all(x["verdict"] == "SUPÉRIEUR" for x in c1)
    assert all(x["verdict"] == "RAPPORTÉ" for x in out["pooled"]["C2"] + out["pooled"]["C3"])
    # Q1 family = the readable (model, protocol) endpoints, not the control's
    assert sum(1 for x in out["pooled"]["Q1"] if x["n_datasets"]) == 6
    for prot, x in out["shallow_q1_reading"].items():
        assert "attribuable à la croissance" in x["reading"], x
    idn = [c["identity_C1_C2_Q1"] for c in out["cases"].values() if "identity_C1_C2_Q1" in c]
    assert len(idn) == 3 and all(i["n"] == N_SUBJ and i["max_abs_error"] < 1e-12 for i in idn)
    assert not any("C1" in c for k, c in out["cases"].items() if "/eegnex/" in k)


def test_q1_confounded_by_init(tmp_path):
    tmp = tmp_path
    man = build(tmp, {"bd": 0, "tiny": 0.05, "random": 0.04, "tiny_bdinit": -0.03, "fix": 0.06})
    r, out = read(tmp, man)
    assert r.returncode == 0, r.stdout + r.stderr
    for prot, x in out["shallow_q1_reading"].items():
        assert x["Q1"] == "SUPÉRIEUR" and x["C1"] != "SUPÉRIEUR"
        assert "confondu par l'init" in x["reading"] and x["C1"] in x["reading"]


@pytest.mark.parametrize("arm,bad", [("tiny_bdinit", "random"), ("fix", "tiny")])
def test_refuses_wrong_growth_rows(tmp_path, arm, bad):
    tmp = tmp_path
    man = build(tmp, {"bd": 0, "tiny": 0.05, "random": 0.04, "tiny_bdinit": 0.03, "fix": 0.01})
    f = next((tmp / "res" / f"shallow__{arm}" / "cross_session").glob("*__fits.jsonl"))
    rec = json.loads(f.read_text())
    rec["history"] = [h for h in rec["history"] if "grow_s" not in h] + [
        {"epoch": 5, "train_loss": 1.0, "valid_loss": 1.0, "grow_s": 1.0, "grow_direction": bad}]
    f.write_text(json.dumps(rec) + "\n")
    r, _ = read(tmp, man)
    assert r.returncode == 2 and "REFUSED" in r.stdout, r.stdout
