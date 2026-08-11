"""Project EEG datasets onto a common electrode montage.

Why
---
A network has a fixed input width, so training one decoder across datasets requires a
single channel space. There is no free one: over the 11 MOABB motor-imagery datasets
with known electrode positions, the set of electrodes *every* dataset records is one
channel (Cz). Keeping the intersection is therefore not an option -- the common space
has to be constructed.

Spherical-spline interpolation constructs it. The scalp potential is a smooth field
sampled at the electrodes; given the samples, the field can be estimated anywhere on
the sphere (Perrin et al. 1989, the method behind ``mne.io.Raw.interpolate_bads``).
Missing electrodes of a target montage are exactly bad channels whose value we never
observed, so the machinery is the same.

What it cannot do
-----------------
Interpolation is *not* extrapolation. The estimate is trustworthy inside the region the
source electrodes cover and degrades outside it, because splines fitted to a handful of
points near the vertex say nothing about the occiput. Reconstructing 19 target channels
from the 3 electrodes of BNCI2014_004 produces an array of the right shape and no
information -- three spatial degrees of freedom smeared over 22 outputs. That is worse
than useless in a benchmark: it looks like data.

The guard is therefore **geometric**, not a count. Counting is measurably the wrong
criterion: Shin2017A has to reconstruct 20 of 22 target electrodes, yet every one of
them has a recorded electrode within 3.9 cm, because its 30-channel cap covers the same
scalp under the 10-05 intermediate names -- it is the best-supported dataset of the
pool. BNCI2014_004 reconstructs fewer (19) with a worst-case gap of 10.1 cm. So what is
checked is the distance from each reconstructed electrode to its nearest *recorded*
one, against :data:`MAX_GAP_CM`. The reference point for that threshold is the target
montage's own density: the 22 electrodes of ``SENSORIMOTOR_22`` sit a median 3.45 cm and
at most 4.22 cm from their nearest neighbour, so a reconstruction with a source inside
4.5 cm is no more of a stretch than the montage it is joining. The measured gaps come
back in the diagnostics, so a paper can report them per dataset.

A dataset with no coordinates at all (BNCI2014_002 ships ``EEG1``..``EEG15``) cannot be
interpolated on any terms and raises.

Rank
----
Spline interpolation is a *linear* operator on the recorded channels, so the projected
array has rank at most ``min(n_support, len(target))`` -- the reported ``rank``, which is
the rank of the projection itself and therefore a ceiling on the data's. Real EEG is full
rank in its own channels, so the two coincide in practice. Reconstructing 13 channels
from 14 electrodes yields 22 columns spanning a 14-dimensional space, and the covariance
of that array is singular by construction. This is not a numerical detail:

* CSP and unregularised Riemannian pipelines fail outright (``scipy.linalg.eigh`` on a
  non-positive-definite matrix). Shrunk estimators (``Covariances("oas")``) survive.
* Euclidean alignment inverts the mean covariance, so on rank-deficient input it whitens
  inside a subspace and discards the rest -- i.e. interpolation and alignment interact,
  and a pool that mixes ranks confounds the two.
* Convolutional nets are indifferent to it; they invert nothing.

``rank`` is therefore reported in the diagnostics, and a pool should keep it constant
across datasets unless the analysis is known to tolerate the variation.

Naming
------
The same electrode is spelled several ways across datasets: ``FP1``/``Fp1``, ``Fc3.``
with the PhysioNet EDF trailing dot, ``EEG-C3``, and the pre-1991 ``T3``/``T4``/``T5``/
``T6`` for ``T7``/``T8``/``P7``/``P8``. Comparing raw strings makes the intersection
empty for typographic reasons alone, so every name goes through :func:`canonical`
first. Shin2017A is the interesting case: 2 of its 30 electrodes match the target by
name, yet its cap covers the scalp densely in the 10-05 intermediate nomenclature
(``AFF5h``, ``FCC3h``). Interpolation recovers the target there with genuine support --
which is precisely the case that justifies the method.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "canonical",
    "SENSORIMOTOR_22",
    "MAX_GAP_CM",
    "resolve_positions",
    "nearest_source_gaps",
    "interpolate_to_montage",
]

#: Largest distance, in cm along the scalp, allowed between a reconstructed electrode
#: and the nearest recorded one. 4.5 cm is read off the target montage rather than
#: picked: ``SENSORIMOTOR_22``'s own nearest-neighbour spacing is a median 3.45 cm with
#: a maximum of 4.22 cm, so anything inside 4.5 cm is interpolated at the density the
#: montage already assumes. Measured over the MOABB motor-imagery pool this admits
#: Shin2017A (3.87 cm worst case), Lee2019_MI (3.86) and Zhou2016 (4.44, marginal), and
#: refuses BNCI2015_001 (6.61), AlexMI (5.50) and BNCI2014_004 (10.11).
MAX_GAP_CM = 4.5

#: The 22 electrodes of BNCI2014_001 (BCI IV-2a): a sensorimotor grid from Fz to Pz.
#: Chosen as the default target because 5 of the pooled datasets record all 22 natively
#: and a 6th misses one, so the pool's core needs no interpolation at all -- the arms
#: that do need it can then be added as an explicit ablation instead of being baked in.
SENSORIMOTOR_22 = (
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
)

#: Electrodes renamed by the 1991 10-10 revision. Datasets predating it (and some that
#: simply kept the old labels) use the left-hand spelling for the same scalp position.
_RENAMED = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_DEFAULT_MONTAGE = "standard_1005"


def canonical(name: str) -> str:
    """Comparison key for an electrode name: upper case, no decoration, 10-10 spelling.

    Upper case because case is the most common difference (``FP1`` vs ``Fp1``) and
    carries no information; the montage spelling is recovered separately by
    :func:`resolve_positions`, so nothing is lost.
    """
    n = str(name).strip().replace(".", "")
    for prefix in ("EEG-", "EEG "):
        if n.upper().startswith(prefix.upper()):
            n = n[len(prefix):]
            break
    upper = n.strip().upper()
    return _RENAMED.get(upper, upper)


def _montage_lookup(montage_name: str) -> dict[str, str]:
    """canonical key -> the montage's own spelling, which ``set_montage`` requires."""
    import mne

    m = mne.channels.make_standard_montage(montage_name)
    return {canonical(c): c for c in m.ch_names}


def resolve_positions(
    ch_names: list[str], *, montage: str = _DEFAULT_MONTAGE
) -> tuple[list[int], list[str], list[str]]:
    """Split channel names into those the montage places and those it does not.

    Returns
    -------
    keep : list of int
        Indices into ``ch_names`` of channels with a known position.
    renamed : list of str
        Those channels under the montage's own spelling, same order as ``keep``.
    unknown : list of str
        Original names the montage cannot place. A channel here is unusable for
        interpolation -- it has no scalp coordinates to interpolate from.
    """
    lut = _montage_lookup(montage)
    keep: list[int] = []
    renamed: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for i, name in enumerate(ch_names):
        key = canonical(name)
        std = lut.get(key)
        if std is None:
            unknown.append(name)
            continue
        if key in seen:
            # two source channels canonicalising to one electrode: keep the first, a
            # duplicate position makes the spline system singular
            continue
        seen.add(key)
        keep.append(i)
        renamed.append(std)
    return keep, renamed, unknown


def nearest_source_gaps(
    missing, source, *, montage: str = _DEFAULT_MONTAGE
) -> dict[str, float]:
    """Scalp distance, in cm, from each ``missing`` electrode to its nearest ``source``.

    Measured along the sphere the montage's coordinates lie on, not through the head:
    the spline is fitted on the sphere, so the distance that governs how well a point is
    supported is the arc, not the chord. The two differ by a few percent at these
    separations, but the arc is the honest one to quote.
    """
    import mne

    m = mne.channels.make_standard_montage(montage)
    pos = {canonical(k): np.asarray(v)
           for k, v in m.get_positions()["ch_pos"].items()}
    radius = float(np.mean([np.linalg.norm(v) for v in pos.values()]))
    src = [pos[canonical(s)] for s in source if canonical(s) in pos]
    if not src:
        raise ValueError("no source electrode has a position")
    unit_src = np.stack([v / np.linalg.norm(v) for v in src])

    out = {}
    for name in missing:
        p = pos[canonical(name)]
        u = p / np.linalg.norm(p)
        arc = radius * np.arccos(np.clip(unit_src @ u, -1.0, 1.0))
        out[str(name)] = float(arc.min() * 100.0)
    return out


def interpolate_to_montage(
    X: np.ndarray,
    ch_names,
    target=SENSORIMOTOR_22,
    *,
    sfreq: float = 250.0,
    montage: str = _DEFAULT_MONTAGE,
    max_gap_cm: float | None = MAX_GAP_CM,
    mode: str = "accurate",
):
    """Re-express trials on ``target``, interpolating the electrodes that are missing.

    Channels of ``X`` that are *not* in ``target`` are not discarded first: they are
    kept as support for the spline fit and dropped only afterwards. Schirrmeister2017's
    128 electrodes therefore constrain the 22 outputs far better than 22 would.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
    ch_names : sequence of str
        Names of ``X``'s channels, in order, as the dataset spells them.
    target : sequence of str
        Electrodes to produce, in the returned order.
    sfreq : float
        Only used to build the temporary :class:`mne.Info`; interpolation is purely
        spatial, so the value does not affect the result.
    max_gap_cm : float or None
        Refuse if any reconstructed electrode is farther than this from the nearest
        recorded one -- the difference between interpolating and inventing. ``None``
        forces the projection through, for a deliberate negative control.
    mode : {"accurate", "fast"}
        Passed to MNE's spline solver.

    Returns
    -------
    Xt : ndarray, shape (n_trials, len(target), n_times)
    info : dict
        ``present`` / ``interpolated`` / ``unknown`` / ``n_support`` / ``gaps_cm`` /
        ``max_gap_cm`` / ``median_gap_cm`` / ``rank``, so a caller can record in its
        results exactly how much of the input was reconstructed, how well supported it
        was, and how many dimensions it actually spans.
    """
    import mne

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got {X.shape}")
    ch_names = [str(c) for c in ch_names]
    if len(ch_names) != X.shape[1]:
        raise ValueError(
            f"{len(ch_names)} names for {X.shape[1]} channels")
    target = list(target)

    keep, renamed, unknown = resolve_positions(ch_names, montage=montage)
    if not keep:
        raise ValueError(
            f"none of the {len(ch_names)} channels has a known position in "
            f"{montage!r} (first few: {ch_names[:5]}); this dataset cannot be "
            "interpolated -- exclude it or supply a montage explicitly")

    lut = _montage_lookup(montage)
    # validate the target before translating it: a name the montage cannot place has no
    # entry to look up, and a KeyError would say nothing useful about which one
    unresolved_target = [t for t in target if canonical(t) not in lut]
    if unresolved_target:
        raise ValueError(f"target electrodes absent from {montage!r}: "
                         f"{unresolved_target}")
    have = {canonical(r) for r in renamed}
    missing_std = [lut[canonical(t)] for t in target if canonical(t) not in have]

    gaps = (nearest_source_gaps(missing_std, renamed, montage=montage)
            if missing_std else {})
    if gaps and max_gap_cm is not None:
        far = {k: v for k, v in gaps.items() if v > max_gap_cm}
        if far:
            worst = ", ".join(f"{k} {v:.1f} cm" for k, v in
                              sorted(far.items(), key=lambda kv: -kv[1])[:5])
            raise ValueError(
                f"{len(far)} target electrode(s) have no recorded electrode within "
                f"{max_gap_cm} cm ({worst}). Spherical splines interpolate, they do "
                f"not extrapolate: those channels would be invented, not measured. "
                f"Pass max_gap_cm=None only as a deliberate negative control.")

    order = {canonical(t): i for i, t in enumerate(target)}

    if not missing_std:
        # Nothing to invent: a permutation, and no spline solver in the hot path. This
        # is the majority of the pool, so keep it exact and cheap.
        idx = [-1] * len(target)
        for pos, name in zip(keep, renamed):
            j = order.get(canonical(name))
            if j is not None:
                idx[j] = pos
        return X[:, idx, :], {
            "present": list(target), "interpolated": [], "unknown": unknown,
            "n_support": len(keep), "gaps_cm": {},
            "max_gap_cm": 0.0, "median_gap_cm": 0.0,
            "rank": len(target)}

    data = np.concatenate(
        [X[:, keep, :], np.zeros((X.shape[0], len(missing_std), X.shape[2]),
                                dtype=X.dtype)],
        axis=1)
    info = mne.create_info(list(renamed) + list(missing_std), float(sfreq), "eeg")
    info.set_montage(mne.channels.make_standard_montage(montage))
    # The zeros above are placeholders; marking them bad is what makes MNE solve for
    # them instead of reading them.
    info["bads"] = list(missing_std)
    epochs = mne.EpochsArray(data, info, verbose="ERROR")
    epochs.interpolate_bads(reset_bads=True, mode=mode, verbose="ERROR")

    got = epochs.get_data(copy=False)
    names_out = epochs.ch_names
    idx = [None] * len(target)
    for pos, name in enumerate(names_out):
        j = order.get(canonical(name))
        if j is not None:
            idx[j] = pos
    if any(i is None for i in idx):
        gap = [t for t, i in zip(target, idx) if i is None]
        raise RuntimeError(f"interpolation did not produce {gap}")
    Xt = got[:, idx, :]
    return Xt, {
        "present": [t for t in target if canonical(t) in have],
        "interpolated": [t for t in target if canonical(t) not in have],
        "unknown": unknown,
        "n_support": len(keep),
        "gaps_cm": gaps,
        "max_gap_cm": max(gaps.values()),
        "median_gap_cm": float(np.median(list(gaps.values()))),
        # linear operator: the 22 columns span at most as many dimensions as there were
        # recorded electrodes. Carried in the results because it decides which estimators
        # are even applicable (see the module docstring).
        "rank": int(min(len(keep), len(target))),
    }
