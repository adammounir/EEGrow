"""Euclidean Alignment (EA) of EEG trials.

Why
---
Two subjects wearing the same cap still produce very different signals: skull
thickness, electrode impedance, cap placement and cortical folding all rescale and
mix the sensor space. Formally, a large part of the between-subject shift is well
modelled by a subject-specific *linear mixing* ``X_s = A_s X``. Any decoder trained
across subjects therefore spends capacity on ``A_s`` rather than on the neural
signal.

EA removes that mixing without labels. For one subject, take the mean trial
covariance ``R`` and whiten every trial with ``R^{-1/2}``. After the transform the
subject's mean covariance is the identity, so the sensor space is put in a common
frame. Crucially the whitening cancels the nuisance mixing *exactly up to a
rotation*: if ``X' = A X`` then ``(A R Aᵀ)^{-1/2} A = O R^{-1/2}`` for some
orthogonal ``O``, hence aligned trials from two subjects differ only by ``O``.

Reference: Junqueira, Aristimunha, Chevallier, de Camargo, *A Systematic Evaluation
of Euclidean Alignment with Deep Learning for EEG Decoding* (arXiv:2401.10746) —
+4.33 % accuracy and −70 % convergence time on cross-subject motor imagery.

Leakage
-------
The reference matrix of the *test* subject is estimated on that subject's own trials,
which are available at inference time but **unlabelled**. This is unsupervised domain
adaptation, not label leakage. It is however transductive: it assumes a batch of test
trials, not a single online trial. Any paper using this must say so explicitly.

Scale convention
----------------
``R^{-1/2}`` also changes the amplitude of the data by several orders of magnitude
(MOABB serves volts, ~1e-5, while whitened data has unit variance). In an ablation
that would confound "whitening" with "rescaling" -- the aligned arm would differ from
the raw arm on two counts at once. ``preserve_scale=True`` (the default) therefore
rescales the whole array by a *single global* factor so its RMS matches the input's.
One global factor, not one per subject: a per-subject factor would put back exactly
the amplitude differences EA is meant to remove.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["euclidean_reference", "inverse_sqrtm", "euclidean_align"]


def euclidean_reference(X: np.ndarray) -> np.ndarray:
    """Mean spatial covariance of a set of trials.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)

    Returns
    -------
    R : ndarray, shape (n_channels, n_channels)

    Notes
    -----
    Normalised by ``n_times`` so ``R`` is a covariance (per-sample) and not a sum of
    squares; this only fixes a global scale but keeps the numbers interpretable.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got {X.shape}")
    n_times = X.shape[2]
    # einsum over trials avoids materialising the (n_trials, C, C) stack.
    return np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * n_times)


def inverse_sqrtm(R: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Inverse square root of a symmetric positive semi-definite matrix.

    ``R`` is often rank-deficient in practice (average reference removes one degree of
    freedom, interpolated channels remove more). Eigenvalues below ``rcond * max`` are
    floored rather than zeroed: zeroing them would project the data onto a subspace and
    silently drop channels, whereas flooring keeps the transform full-rank and merely
    caps how much those directions get amplified.
    """
    R = np.asarray(R, dtype=np.float64)
    w, V = np.linalg.eigh((R + R.T) / 2.0)
    w_max = float(w.max())
    if w_max <= 0:
        raise ValueError("reference covariance is not positive definite")
    floor = rcond * w_max
    n_floored = int((w < floor).sum())
    if n_floored:
        warnings.warn(
            f"reference covariance is rank-deficient: {n_floored}/{len(w)} eigenvalues "
            f"below {rcond:g} x max were floored",
            RuntimeWarning,
            stacklevel=2,
        )
        w = np.maximum(w, floor)
    return (V * w**-0.5) @ V.T


def euclidean_align(
    X: np.ndarray,
    groups=None,
    *,
    preserve_scale: bool = True,
    rcond: float = 1e-10,
) -> np.ndarray:
    """Whiten each group of trials by its own mean covariance.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
    groups : array-like of length n_trials, optional
        Alignment unit -- typically the subject id, or ``(subject, session)`` for
        session-level alignment. ``None`` aligns everything as one group, which is
        only meaningful on single-subject data.
    preserve_scale : bool
        Rescale the output by one global factor so its RMS matches the input's. See
        the module docstring: this is what makes an aligned/raw ablation clean.
    rcond : float
        Relative eigenvalue floor, see :func:`inverse_sqrtm`.

    Returns
    -------
    Xa : ndarray, same shape as ``X``, dtype float64
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got {X.shape}")
    if groups is None:
        groups = np.zeros(len(X), dtype=int)
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError(f"groups has length {len(groups)}, expected {len(X)}")

    Xa = np.empty(X.shape, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        P = inverse_sqrtm(euclidean_reference(X[idx]), rcond=rcond)
        Xa[idx] = np.einsum("cd,ndt->nct", P, X[idx])

    if preserve_scale:
        rms_in = float(np.sqrt(np.mean(np.square(X, dtype=np.float64))))
        rms_out = float(np.sqrt(np.mean(np.square(Xa))))
        if rms_out > 0:
            Xa *= rms_in / rms_out
    return Xa
