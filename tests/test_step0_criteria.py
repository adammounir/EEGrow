"""Step-0 criteria and the B-then-H config rule, on synthetic results trees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_criteria as cr  # noqa: E402
import step0_gate as g  # noqa: E402
from test_step0_gate import _cell  # noqa: E402

import pytest  # noqa: E402


def _write_tree(root, *, late_frac=0.0, cap_loss=0.001, ref_acc=0.72, skip=None):
    for cand, arms in cr.CANDIDATES.items():
        for arm, (lr, wd) in arms.items():
            for seed in cr.SEEDS:
                recs = _cell()
                n_late = int(round(late_frac * len(recs)))
                for i, r in enumerate(recs):
                    r["optimizer"].update(lr=lr, weight_decay=wd)
                    for h in r["history"]:
                        h["lr"] = lr
                    r["selected_epoch"] = 199 if i < n_late else 80
                if skip == (cand, arm, seed):
                    continue
                p = g.cell_path(root / cand / "results", "within_session",
                                "bnci2014_001", arm, seed)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    aux = root / "aux"
    aux.mkdir(parents=True, exist_ok=True)
    with open(aux / "capacity.jsonl", "w") as fh:
        for cand, arms in cr.CANDIDATES.items():
            for arm in arms:
                for s in cr.SEEDS:
                    loss = cap_loss(cand, arm) if callable(cap_loss) else cap_loss
                    fh.write(json.dumps({"arm": arm, "candidate": cand, "seed": s,
                                         "final_loss_eval": loss}) + "\n")
    for m in ("shallow", "deep"):
        for subj in range(1, 10):
            for s in cr.SEEDS:
                (aux / f"reference_{m}_S{subj}_seed{s}.jsonl").write_text(json.dumps(
                    {"model": m, "subject": subj, "seed": s, "max_epochs": 800,
                     "max_increase": 80, "test_acc": ref_acc + 0.02 * (subj - 5)}) + "\n")


def test_everything_passes_freezes_b(tmp_path):
    _write_tree(tmp_path)
    out = cr.evaluate(tmp_path)
    assert {a: v["frozen_config"] for a, v in out["arms"].items()} == {
        "bd_shallow": "B", "bd_deep4": "B"}
    assert not out["reference"]["shallow"]["refused"]


def test_b_failing_capacity_falls_back_to_h(tmp_path):
    _write_tree(tmp_path, cap_loss=lambda c, a: 0.4 if (c, a) == ("B", "bd_deep4") else 0.001)
    out = cr.evaluate(tmp_path)
    assert out["arms"]["bd_deep4"]["frozen_config"] == "H"
    assert out["arms"]["bd_shallow"]["frozen_config"] == "B"


def test_late_edge_refuses_both_configs(tmp_path):
    _write_tree(tmp_path, late_frac=0.25)
    out = cr.evaluate(tmp_path)
    assert out["arms"]["bd_deep4"]["frozen_config"] == "REFUSED"


def test_reference_far_from_published_refuses(tmp_path):
    _write_tree(tmp_path, ref_acc=0.45)
    out = cr.evaluate(tmp_path)
    assert out["reference"]["deep"]["refused"]
    assert out["arms"]["bd_deep4"]["frozen_config"] == "REFUSED"


def test_a_missing_cell_blocks_reading_that_config(tmp_path):
    _write_tree(tmp_path, skip=("B", "bd_shallow", 1))
    out = cr.evaluate(tmp_path)
    b = out["arms"]["bd_shallow"]["candidates"]["B"]
    assert b["passes"] is None and "NOT read" in b["status"]
    assert out["arms"]["bd_shallow"]["frozen_config"].startswith("undecided")
    assert cr.markdown(out)


def test_departure_is_reported_not_refusing(tmp_path):
    _write_tree(tmp_path)
    d = cr.evaluate(tmp_path)["arms"]["bd_deep4"]["candidates"]["B"]["departure"]
    assert d["refusing"] is False and abs(d["excess_nat"] - (1.9 - 1.3863)) < 1e-3


def _grow_rec(*, epochs=(5, 10, 15, 20), proposed=40, s=1.0, keep_all=False,
              reach=True, start=8, target=40):
    hist, width = [], start
    step = (target - start) // len(epochs)
    for ep in range(1, 201):
        h = {"epoch": ep}
        if ep in epochs:
            after = width + step
            h.update(grow_s=s, grow_applied=True, grow_width_after=after,
                     grow_n_proposed=proposed, grow_n_kept=proposed if keep_all else step,
                     grow_eig_proposed=list(np.logspace(0, -3, proposed)),
                     grow_holdout="disjoint")
            width = after
        hist.append(h)
    return {"width_start": start, "width_end": target if reach else width - 1,
            "history": hist}


import numpy as np  # noqa: E402


def test_growth_criteria_pass_on_a_real_selection():
    g = cr.growth_criteria([_grow_rec() for _ in range(10)], 40)
    assert not any(v["refused"] for v in g.values()), g


@pytest.mark.parametrize("kw,crit", [
    (dict(s=4.0), "censoring"),
    (dict(keep_all=True), "selection"),
    (dict(proposed=10), "selection"),          # 10 candidates for 8 places
    (dict(reach=False), "target"),
    (dict(epochs=(2, 4, 6, 8)), "calendar"),
])
def test_each_growth_criterion_refuses(kw, crit):
    g = cr.growth_criteria([_grow_rec(**kw) for _ in range(10)], 40)
    assert g[crit]["refused"], (crit, g[crit])


def test_flat_spectrum_is_refused():
    recs = [_grow_rec() for _ in range(4)]
    for r in recs:
        for h in r["history"]:
            if "grow_eig_proposed" in h:
                h["grow_eig_proposed"] = [1.0, 0.9, 0.8]
    assert cr.growth_criteria(recs, 40)["spectrum"]["refused"]
