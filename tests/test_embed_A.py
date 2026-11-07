"""embed_A.py's pure functions on synthetic data (PREREG_growth_embedding_A.md).

Identification must find planted subject structure and sit at chance without it; the
residual test (P2) must be at chance when the embedding is a linear function of the
covariance plus noise, and above chance when it carries a subject term the covariance
does not; the leave-one-out signature must find a planted pool shift.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import embed_A  # noqa: E402

N = 40


def items(fn, n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {(str(s), se): fn(rng, s, se) for s in range(n) for se in ("0", "1")}


def top1(h):
    return float(np.mean(list(h.values())))


def test_identify_planted_and_chance():
    subj = np.random.default_rng(1).standard_normal((N, 30))
    planted = items(lambda r, s, se: subj[s] + 0.3 * r.standard_normal(30))
    noise = items(lambda r, s, se: r.standard_normal(30))
    h, auc = embed_A.identify(planted)
    assert top1(h) > 0.9 and auc > 0.95
    h, auc = embed_A.identify(noise)
    assert top1(h) < 0.2 and 0.35 < auc < 0.65


def test_ridge_residual_is_not_a_valid_null():
    """Why "beyond the covariance" is not a criterion (PREREG § 9): an embedding that is a
    linear function of the covariance plus noise keeps a ridge residual that still
    identifies the subjects, so the residual test of the first draft would have passed
    with nothing beyond the covariance."""
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(2)
    cov_subj = rng.standard_normal((N, 50))
    W = rng.standard_normal((50, 20))
    cov = items(lambda r, s, se: cov_subj[s] + 0.1 * r.standard_normal(50), seed=3)
    linear = {k: cov[k] @ W + 0.5 * rng.standard_normal(20) for k in cov}
    res = {}
    for s in {k[0] for k in cov}:
        tr = [k for k in cov if k[0] != s]
        X, Y = np.array([cov[k] for k in tr]), np.array([linear[k] for k in tr])
        mx, sx, my, sy = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X - mx) / sx, (Y - my) / sy)
        for k in (k for k in cov if k[0] == s):
            res[k] = (linear[k] - my) / sy - m.predict(((cov[k] - mx) / sx)[None])[0]
    assert top1(embed_A.identify(res)[0]) > 0.8


def test_loo_signature_finds_the_left_out_subject():
    rng = np.random.default_rng(4)
    own = {str(s): rng.standard_normal(30) for s in range(N)}
    mean = np.mean(list(own.values()), 0)
    # fit i is trained on everyone but i: its neurons are the pool mean minus i's share
    fit = {s: mean - own[s] / (N - 1) + 0.002 * rng.standard_normal(30) for s in own}
    h = embed_A.loo_signature(fit, own)
    assert top1(h) > 0.9
    shuffled = dict(zip(own, rng.permutation(list(fit.values()))))
    assert top1(embed_A.loo_signature(shuffled, own)) < 0.2


def test_describe_and_zblocks():
    sd = {"conv_time.layer.weight": torch.randn(40, 1, 25, 1),
          "conv_spat.layer.weight": torch.randn(40, 40, 1, 22)}
    spec, topo = embed_A.shallow_describe(sd, list(range(8, 40)))
    assert spec.shape == (33,) and topo.shape == (22,)
    assert abs(spec.sum() - 1) < 1e-9 and abs(topo.sum() - 1) < 1e-9
    bd = {"conv_time_spat.conv_time.weight": sd["conv_time.layer.weight"],
          "conv_time_spat.conv_spat.weight": sd["conv_spat.layer.weight"]}
    assert np.allclose(embed_A.shallow_describe(bd, list(range(8, 40)))[0], spec)
    nx = {"net.block_4.0.first.layer.weight": torch.randn(32, 64, 1, 16)}
    assert embed_A.eegnex_describe(nx, list(range(4, 32)))[0].shape == (33,)
    z = embed_A.zblocks({("1", "0"): [np.ones(3), np.zeros(2)], ("2", "0"): [np.zeros(3), np.ones(2)]})
    assert z[("1", "0")].shape == (5,) and np.allclose(z[("1", "0")] + z[("2", "0")], 0)


def test_width_gate_excludes_wrong_geometry():
    good = {"conv_time.layer.weight": torch.randn(40, 1, 25, 1),
            "conv_spat.layer.weight": torch.randn(40, 40, 1, 22)}
    fits = {("1", "0"): (good, 40, 8), ("2", "0"): (good, 40, 40), ("3", "0"): (good, 32, 8)}
    emb, excl = embed_A.shallow_sets(fits, "tiny", list(range(8, 40)))
    assert list(emb) == [("1", "0")] and excl == 2
    emb, excl = embed_A.shallow_sets(fits, "bd", list(range(40)))
    assert list(emb) == [("2", "0")] and excl == 2


def test_cache_of_reads_the_cell_lines(tmp_path):
    """Amendment 1: the covariance cache is the one the dataset's cells read, never a
    hard-coded path; two caches for one dataset, or none, refuse."""
    import pytest
    head = "{} cross_session {} shallow bd 0 1 /r /r/x.csv model=bd_shallow cache.enabled=true"
    lines = [head.format(0, "lee2019_mi") + " cache.path=/c/old",
             head.format(1, "cho2017") + " cache.path=/c/A",
             head.format(2, "cho2017") + " cache.path=/c/A",
             head.format(3, "zhou2016") + " cache.path=/c/A",
             head.format(4, "zhou2016") + " cache.path=/c/B"]
    p = tmp_path / "cells.txt"
    p.write_text("\n".join(lines) + "\n")
    assert embed_A.cache_of([p], "lee2019_mi") == "/c/old"
    assert embed_A.cache_of([p], "cho2017") == "/c/A"
    for ds in ("zhou2016", "weibo2014"):
        with pytest.raises(SystemExit, match="REFUSED"):
            embed_A.cache_of([p], ds)
