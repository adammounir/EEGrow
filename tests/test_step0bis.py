"""Step 0-bis: the gate knows the shin2017a / physionetmi structure, and the reading
script refuses to read a refused cell and reads a correct one."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_gate as g  # noqa: E402
import step0bis_criteria as sb  # noqa: E402
from test_step0_gate import _record  # noqa: E402


def _cell(dataset, arm, *, growing=False):
    (lr, wd), target = sb.ARMS[arm]
    recs, i = [], 0
    for s in range(g.N_SUBJECTS[dataset], 0, -1):
        for sess in g.SESSIONS[dataset]:
            for _ in range(g.N_FOLDS):
                r = _record(s, sess, i)
                r["model"] = arm
                r["optimizer"].update(lr=lr, weight_decay=wd)
                r["selected_epoch"] = 80
                for h in r["history"]:
                    h["lr"] = lr
                if growing:
                    r["width_start"], r["width_end"] = 8, target
                    width = 8
                    for ep, after in ((10, 8 + (target - 8) // 2), (20, target)):
                        h = r["history"][ep - 1]
                        h.update(grow_s=2.0, grow_applied=True, grow_width_after=after,
                                 grow_n_proposed=3 * (after - width),
                                 grow_n_kept=after - width, grow_holdout="disjoint",
                                 grow_eig_proposed=list(np.logspace(0, -3, 6)),
                                 grow_losses={str(x): 1.0 for x in sb.GRID},
                                 grow_normalization=sb.NORMALIZATION)
                        width = after
                recs.append(r)
                i += 1
    return recs


def _write(root, dataset, arm, recs, seed):
    p = g.cell_path(root / "B" / "results", "within_session", dataset, arm, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_gate_admits_a_complete_shin_and_physionet_cell():
    for ds in sb.DATASETS:
        (lr, wd), _ = sb.ARMS["bd_shallow"]
        v = g.gate_cell(_cell(ds, "bd_shallow"), dataset=ds, lr=lr, weight_decay=wd,
                        growing=False)
        assert not v["refused"], (ds, v["reasons"])
        assert v["n_fits"] == g.N_SUBJECTS[ds] * len(g.SESSIONS[ds]) * g.N_FOLDS


def test_gate_refuses_a_missing_shin_session():
    (lr, wd), _ = sb.ARMS["bd_shallow"]
    recs = [r for r in _cell("shin2017a", "bd_shallow") if not
            (r["subject"] == "12" and r["session"] == "4imagery")]
    assert g.gate_cell(recs, dataset="shin2017a", lr=lr, weight_decay=wd,
                       growing=False)["refused"]


def test_reads_every_arm_on_both_datasets(tmp_path):
    for ds in sb.DATASETS:
        for arm, (_, target) in sb.ARMS.items():
            for seed in (0, 1, 2):
                _write(tmp_path, ds, arm, _cell(ds, arm, growing=target is not None),
                       seed)
    out = sb.evaluate(tmp_path)
    for ds in sb.DATASETS:
        for arm in sb.ARMS:
            assert out[ds][arm]["status"] == "read", (ds, arm, out[ds][arm])
            assert out[ds][arm]["passes"] is True, (ds, arm, out[ds][arm])
    assert "physionetmi" in sb.markdown(out)


def test_a_refused_cell_is_not_read(tmp_path):
    for seed in (0, 1, 2):
        recs = _cell("physionetmi", "grow_deep4", growing=True)
        if seed == 1:
            recs = copy.deepcopy(recs)
            recs[3]["history"][9]["grow_normalization"] = None   # ran without addendum 2
        _write(tmp_path, "physionetmi", "grow_deep4", recs, seed)
    e = sb.evaluate_cell_set(tmp_path, "physionetmi", "grow_deep4")
    assert e["passes"] is None and "NOT read" in e["status"]


def test_early_optimum_refuses_on_a_dataset(tmp_path):
    for seed in (0, 1, 2):
        recs = _cell("shin2017a", "bd_deep4")
        for r in recs[: len(recs) // 4]:
            r["selected_epoch"] = 3                     # 25 % scored before epoch 5
        _write(tmp_path, "shin2017a", "bd_deep4", recs, seed)
    assert sb.evaluate_cell_set(tmp_path, "shin2017a", "bd_deep4")["passes"] is False
