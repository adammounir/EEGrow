"""embed_growth.py's pure parts (EXPLO_growth_trajectory_A.md): tie-aware identification
and the width gate on growth events / checkpoints."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import embed_growth  # noqa: E402

N = 30


def items(fn, seed=0):
    rng = np.random.default_rng(seed)
    return {(str(s), se): fn(rng, s) for s in range(N) for se in ("0", "1")}


def top1(h):
    return float(np.mean(list(h.values())))


def test_ties_give_exact_chance_and_clones():
    same = items(lambda r, s: np.ones(10) + np.arange(10))
    h, clone = embed_growth.identify_ties(same)
    assert abs(top1(h) - 1 / N) < 1e-12 and clone == 1.0


def test_planted_and_noise():
    subj = np.random.default_rng(1).standard_normal((N, 20))
    h, clone = embed_growth.identify_ties(items(lambda r, s: subj[s] + 0.3 * r.standard_normal(20)))
    assert top1(h) > 0.9 and clone == 0.0
    h, _ = embed_growth.identify_ties(items(lambda r, s: r.standard_normal(20)))
    assert top1(h) < 0.2


def sd(w):
    return {"conv_time.layer.weight": torch.randn(w, 1, 25, 1),
            "conv_spat.layer.weight": torch.randn(40, w, 1, 22)}


def fit(widths=(15, 19, 22, 26, 29, 32, 37, 40), ck=None):
    ev, b = [], 8
    for i, a in enumerate(widths):
        ev.append(((i + 1) * 5, b, a, sd(a)))
        b = a
    ck = ck or {10: 19, 25: 29, 50: 40, 100: 40}
    d = {"ins": ev, 0: sd(8), 200: sd(40)}
    d.update({t: sd(w) for t, w in ck.items()})
    return d


def test_width_gate():
    assert embed_growth.check(fit(), "tiny") is None
    assert embed_growth.check(fit(ck={10: 15, 25: 29, 50: 40, 100: 40}), "tiny") == "checkpoint width"
    assert embed_growth.check(fit(widths=(15, 19, 22, 26, 29, 32, 37, 39)), "random") == "growth events"


def test_sets_and_zero_norm_neurons():
    d = fit()
    s = embed_growth.sets_at(d, "tiny")
    assert sum(len(i) for _, i in s[("added", "ins")]) == 32 and s[("G1", 10)][0][1] == list(range(8, 15))
    assert ("added", 25) not in s and ("G1", 0) not in s and ("init8", 0) in s
    z = sd(40)
    z["conv_spat.layer.weight"][:, 8:12] = 0
    e, n0 = embed_growth.embed([(z, list(range(8, 40)))])
    assert n0 == 4 and np.isfinite(np.concatenate(e)).all()
