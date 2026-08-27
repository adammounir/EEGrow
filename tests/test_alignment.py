"""Tests for Euclidean Alignment.

The point of EA is a mathematical guarantee, not a heuristic, so the tests check the
guarantee itself: whitening (mean covariance becomes identity), per-group isolation,
and above all *invariance to a per-subject linear mixing up to a rotation* -- the
property that makes EA work across subjects at all. The rotation is invisible in the
temporal Gram matrix ``XᵀX``, which gives an exact equality to assert on.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegrow.alignment import euclidean_align, euclidean_reference, inverse_sqrtm

C, T, N = 6, 128, 40


def _trials(seed: int, n: int = N, mixing: np.ndarray | None = None) -> np.ndarray:
    """``n`` trials of coloured noise, optionally passed through a mixing matrix."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, C, T))
    # colour the sensor space so the covariance is not already the identity
    A = mixing if mixing is not None else np.eye(C) + 0.4 * rng.standard_normal((C, C))
    return np.einsum("cd,ndt->nct", A, X)


def test_reference_is_spd_and_symmetric():
    R = euclidean_reference(_trials(0))
    assert R.shape == (C, C)
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    assert np.linalg.eigvalsh(R).min() > 0


def test_inverse_sqrtm_inverts():
    R = euclidean_reference(_trials(1))
    P = inverse_sqrtm(R)
    np.testing.assert_allclose(P @ R @ P, np.eye(C), atol=1e-8)


def test_alignment_whitens():
    """The defining property: after EA the group's mean covariance is the identity."""
    X = _trials(2)
    Xa = euclidean_align(X, preserve_scale=False)
    np.testing.assert_allclose(euclidean_reference(Xa), np.eye(C), atol=1e-8)


def test_alignment_is_per_group():
    """Two subjects with different mixings both end up whitened, independently."""
    rng = np.random.default_rng(3)
    Xa_ = _trials(4, mixing=np.eye(C) + 0.6 * rng.standard_normal((C, C)))
    Xb_ = _trials(5, mixing=np.eye(C) + 0.6 * rng.standard_normal((C, C)))
    X = np.concatenate([Xa_, Xb_])
    groups = np.array(["s1"] * len(Xa_) + ["s2"] * len(Xb_))

    Xal = euclidean_align(X, groups, preserve_scale=False)
    for g in ("s1", "s2"):
        np.testing.assert_allclose(
            euclidean_reference(Xal[groups == g]), np.eye(C), atol=1e-8
        )
    # ...and grouping actually changed something: one shared reference would not
    # whiten either subject.
    pooled = euclidean_align(X, preserve_scale=False)
    assert not np.allclose(euclidean_reference(pooled[groups == "s1"]), np.eye(C),
                           atol=1e-3)


def test_invariant_to_subject_mixing_up_to_rotation():
    """``align(A X) = O align(X)``: the nuisance mixing is cancelled exactly.

    An orthogonal ``O`` acts on channels only, so the temporal Gram ``XᵀX`` -- and
    hence anything a rotation-equivariant decoder could learn -- is identical.
    """
    rng = np.random.default_rng(6)
    X = _trials(7)
    A = np.eye(C) + 0.8 * rng.standard_normal((C, C))
    X_mixed = np.einsum("cd,ndt->nct", A, X)

    Ga = euclidean_align(X, preserve_scale=False)
    Gb = euclidean_align(X_mixed, preserve_scale=False)

    gram_a = np.einsum("nct,ncs->nts", Ga, Ga)
    gram_b = np.einsum("nct,ncs->nts", Gb, Gb)
    np.testing.assert_allclose(gram_a, gram_b, atol=1e-6, rtol=1e-6)


def test_idempotent():
    X = _trials(8)
    once = euclidean_align(X, preserve_scale=False)
    twice = euclidean_align(once, preserve_scale=False)
    np.testing.assert_allclose(once, twice, atol=1e-8)


def test_preserve_scale_keeps_global_rms_but_not_per_group_amplitude():
    """Global amplitude is kept (clean ablation); per-subject amplitude is not."""
    rng = np.random.default_rng(9)
    loud = 1000.0 * _trials(10, mixing=np.eye(C) + 0.2 * rng.standard_normal((C, C)))
    quiet = _trials(11, mixing=np.eye(C) + 0.2 * rng.standard_normal((C, C)))
    X = np.concatenate([loud, quiet])
    groups = np.array([0] * len(loud) + [1] * len(quiet))

    Xa = euclidean_align(X, groups, preserve_scale=True)
    np.testing.assert_allclose(
        np.sqrt(np.mean(Xa**2)), np.sqrt(np.mean(X**2)), rtol=1e-10
    )
    # the 1000x amplitude gap between the two subjects is gone
    ratio = np.sqrt(np.mean(Xa[groups == 0] ** 2)) / np.sqrt(np.mean(Xa[groups == 1] ** 2))
    assert 0.9 < ratio < 1.1


def test_rank_deficient_reference_warns_and_stays_finite():
    """A collinear channel (e.g. after average reference) must not blow up."""
    X = _trials(12)
    X[:, -1, :] = X[:, 0, :]
    with pytest.warns(RuntimeWarning, match="rank-deficient"):
        Xa = euclidean_align(X, preserve_scale=False)
    assert np.isfinite(Xa).all()


def test_shape_and_group_validation():
    with pytest.raises(ValueError, match="n_trials"):
        euclidean_align(np.zeros((4, 5)))
    with pytest.raises(ValueError, match="groups has length"):
        euclidean_align(_trials(13), groups=np.zeros(3))
