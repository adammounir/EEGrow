"""The montage projection decides what a cross-dataset benchmark is even measuring, so
the tests have to answer two different questions: does the plumbing put the right
channel in the right column, and does the interpolation actually recover a field it did
not see. A permutation bug and a silently useless spline both produce an array of the
correct shape.
"""

import numpy as np
import pytest

from eegrow.montage import (
    MAX_GAP_CM,
    SENSORIMOTOR_22,
    canonical,
    interpolate_to_montage,
    nearest_source_gaps,
    resolve_positions,
)

MONTAGE = "standard_1005"


def _positions(names):
    import mne

    m = mne.channels.make_standard_montage(MONTAGE)
    pos = m.get_positions()["ch_pos"]
    return np.array([pos[n] for n in names])


def _smooth_field(names, n_trials=4, n_times=32, seed=0):
    """A scalp field that is smooth by construction, sampled at ``names``.

    A low-order function of position is exactly what spherical splines are built to
    represent, so recovery of a hidden electrode should be near-exact. If the test used
    white noise per channel instead, no interpolator could pass it and the test would
    only be measuring that the field is unpredictable.
    """
    rng = np.random.default_rng(seed)
    p = _positions(names)
    p = p / np.linalg.norm(p, axis=1, keepdims=True).mean()
    # three linear terms + three quadratic ones: enough spatial structure to make
    # ordering mistakes visible, still band-limited on the sphere
    basis = np.concatenate([p, p[:, [0]] * p[:, [1, 2]], p[:, [1]] * p[:, [2]]], axis=1)
    w = rng.normal(size=(basis.shape[1], n_trials, n_times))
    return np.einsum("cb,btk->tck", basis, w)


# a realistic 64-channel cap (the PhysionetMI / Cho2017 layout, in 10-10 spelling)
CAP64 = [
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    "Iz", "FT9", "FT10",
]


def test_canonical_folds_the_spellings_datasets_actually_use():
    # case, the PhysioNet EDF trailing dot, the BNCI gdf prefix, pre-1991 labels
    assert canonical("FP1") == canonical("Fp1") == "FP1"
    assert canonical("Fc3.") == "FC3"
    assert canonical("EEG-C3") == canonical("C3") == "C3"
    assert canonical("T3") == "T7" and canonical("T6") == "P8"


def test_a_positionless_dataset_is_refused_not_guessed():
    """BNCI2014_002 ships ``EEG1``..``EEG15``. There is no scalp to interpolate on."""
    names = [f"EEG{i}" for i in range(1, 16)]
    _, _, unknown = resolve_positions(names)
    assert unknown == names
    X = np.zeros((2, 15, 16))
    with pytest.raises(ValueError, match="known position"):
        interpolate_to_montage(X, names, SENSORIMOTOR_22)


def test_a_hidden_electrode_is_recovered_from_its_neighbours():
    """The only test that distinguishes interpolation from shape juggling.

    C3 is removed from the source, so the value returned in its column was never seen.
    Compared against the true field at C3's position.
    """
    X = _smooth_field(CAP64)
    truth = X[:, CAP64.index("C3"), :]
    src = [c for c in CAP64 if c != "C3"]
    Xs = X[:, [CAP64.index(c) for c in src], :]

    Xt, meta = interpolate_to_montage(Xs, src, SENSORIMOTOR_22)
    got = Xt[:, list(SENSORIMOTOR_22).index("C3"), :]
    assert meta["interpolated"] == ["C3"]
    err = np.linalg.norm(got - truth) / np.linalg.norm(truth)
    assert err < 0.05, f"relative error {err:.3f} -- the spline is not recovering C3"


def test_channels_outside_the_target_still_constrain_the_fit():
    """Dropping non-target channels first would throw away most of the support.

    Schirrmeister2017 has 128 electrodes for a 22-electrode target; the extra 106 are
    what make the reconstruction good. Same hidden channel as above, but with the
    source pre-restricted to the target: the error must be visibly worse.
    """
    X = _smooth_field(CAP64)
    truth = X[:, CAP64.index("C3"), :]

    def err_from(src):
        Xs = X[:, [CAP64.index(c) for c in src], :]
        Xt, meta = interpolate_to_montage(Xs, src, SENSORIMOTOR_22)
        got = Xt[:, list(SENSORIMOTOR_22).index("C3"), :]
        return np.linalg.norm(got - truth) / np.linalg.norm(truth), meta

    wide, meta_wide = err_from([c for c in CAP64 if c != "C3"])
    narrow, _ = err_from([c for c in SENSORIMOTOR_22 if c != "C3"])
    assert meta_wide["n_support"] == 63
    assert wide < narrow


def test_present_channels_are_passed_through_bit_exact():
    """The 5 datasets that record all 22 natively must not be perturbed at all.

    They are the core of the pool; running them through a spline solver would make the
    interpolation ablation compare interpolated-vs-interpolated.
    """
    X = _smooth_field(CAP64)
    src = list(SENSORIMOTOR_22)
    Xs = X[:, [CAP64.index(c) for c in src], :]
    Xt, meta = interpolate_to_montage(Xs, src, SENSORIMOTOR_22)
    assert meta["interpolated"] == []
    np.testing.assert_array_equal(Xt, Xs)


def test_the_output_column_order_is_the_target_order():
    """A permutation bug is invisible downstream: the net just learns worse."""
    X = _smooth_field(CAP64)
    src = list(reversed(SENSORIMOTOR_22))
    Xs = X[:, [CAP64.index(c) for c in src], :]
    Xt, _ = interpolate_to_montage(Xs, src, SENSORIMOTOR_22)
    for j, name in enumerate(SENSORIMOTOR_22):
        np.testing.assert_array_equal(Xt[:, j, :], X[:, CAP64.index(name), :])


def test_odd_spellings_are_matched_not_reinterpolated():
    """PhysionetMI's ``Fc3.`` is FC3. Failing to see that would interpolate a channel
    that was recorded -- a silent quality loss, and the interpolation ablation would
    then measure a naming bug."""
    X = _smooth_field(CAP64)
    src = [c.upper() + "." for c in SENSORIMOTOR_22]
    Xs = X[:, [CAP64.index(c) for c in SENSORIMOTOR_22], :]
    Xt, meta = interpolate_to_montage(Xs, src, SENSORIMOTOR_22)
    assert meta["interpolated"] == []
    np.testing.assert_array_equal(Xt, Xs)


def test_extrapolation_from_three_electrodes_is_refused():
    """BNCI2014_004 has C3/Cz/C4. 19 of 22 target channels would be invented from 3
    spatial degrees of freedom: shaped like data, empty of information. Its worst gap
    to a recorded electrode is 10.1 cm, more than twice the target montage's own
    nearest-neighbour spacing."""
    X = _smooth_field(CAP64)
    src = ["C3", "Cz", "C4"]
    Xs = X[:, [CAP64.index(c) for c in src], :]
    with pytest.raises(ValueError, match="within 4.5 cm"):
        interpolate_to_montage(Xs, src, SENSORIMOTOR_22)


def test_a_dense_cap_that_names_nothing_is_accepted():
    """The regression that a count-based guard gets backwards, and the reason the guard
    is geometric.

    Shin2017A must reconstruct 20 of 22 target electrodes -- 91 %, which any fraction
    threshold rejects -- yet its 30-channel cap covers the same scalp under the 10-05
    intermediate names, so every reconstruction has a recorded neighbour within 3.9 cm.
    It is the *best*-supported dataset of the pool. Emulated here by a cap of 10-05
    intermediate positions, which share almost no names with the target.
    """
    dense = ["AFF5h", "AFF1h", "AFF2h", "AFF6h", "FFC3h", "FFC1h", "FFC2h", "FFC4h",
             "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h",
             "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
             "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h",
             "PPO1h", "PPO2h", "Cz", "Pz"]
    X = _smooth_field(dense)
    named = {canonical(c) for c in dense} & {canonical(t) for t in SENSORIMOTOR_22}
    assert len(named) == 2, "the point of the test is that the names do not match"

    Xt, meta = interpolate_to_montage(X, dense, SENSORIMOTOR_22)
    assert len(meta["interpolated"]) == 20
    assert meta["max_gap_cm"] < 4.5
    assert Xt.shape == (X.shape[0], len(SENSORIMOTOR_22), X.shape[2])


def test_extrapolation_can_be_forced_but_is_reported():
    """The guard is a default, not a prohibition -- a negative-control run must stay
    possible, and must record in its results how far the invention reached."""
    X = _smooth_field(CAP64)
    src = ["C3", "Cz", "C4"]
    Xs = X[:, [CAP64.index(c) for c in src], :]
    Xt, meta = interpolate_to_montage(Xs, src, SENSORIMOTOR_22, max_gap_cm=None)
    assert Xt.shape[1] == len(SENSORIMOTOR_22)
    assert len(meta["interpolated"]) == 19 and meta["n_support"] == 3
    assert meta["max_gap_cm"] > 9.0


def test_the_threshold_is_calibrated_on_the_target_montage_density():
    """4.5 cm is read off the target, not chosen: SENSORIMOTOR_22's own electrodes sit
    at most 4.22 cm from their nearest neighbour. If that ever stops being true the
    threshold has lost its justification and should be recomputed."""
    p = _positions([c for c in SENSORIMOTOR_22])
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    radius = np.linalg.norm(_positions(list(SENSORIMOTOR_22)), axis=1).mean()
    arc = radius * np.arccos(np.clip(p @ p.T, -1, 1)) * 100
    np.fill_diagonal(arc, np.inf)
    nn = arc.min(axis=1)
    assert np.median(nn) == pytest.approx(3.45, abs=0.1)
    assert nn.max() == pytest.approx(4.22, abs=0.1)
    assert nn.max() < MAX_GAP_CM


def test_a_target_electrode_the_montage_cannot_place_is_an_error():
    X = _smooth_field(CAP64)
    src = list(SENSORIMOTOR_22)
    Xs = X[:, [CAP64.index(c) for c in src], :]
    with pytest.raises(ValueError, match="absent from"):
        interpolate_to_montage(Xs, src, list(SENSORIMOTOR_22) + ["NOSUCH"])


def test_two_channels_that_canonicalise_to_one_do_not_make_the_system_singular():
    """Duplicated positions make the spline normal equations singular; the second copy
    is dropped rather than allowed to blow up the solve."""
    X = _smooth_field(CAP64)
    src = [c for c in CAP64 if c != "C3"] + ["cz"]  # Cz twice, second lower case
    Xs = np.concatenate(
        [X[:, [CAP64.index(c) for c in CAP64 if c != "C3"], :],
         X[:, [CAP64.index("Cz")], :]], axis=1)
    keep, renamed, _ = resolve_positions(src)
    assert len(keep) == len(set(renamed)) == 63
    Xt, meta = interpolate_to_montage(Xs, src, SENSORIMOTOR_22)
    assert np.isfinite(Xt).all() and meta["n_support"] == 63


def test_the_gap_is_measured_along_the_scalp_and_is_symmetric_in_scale():
    """Sanity on the metric itself, against head anatomy rather than against the code.

    One 10-10 step is ~4 cm on an adult scalp (the coronal arc T7-Cz spans ~18 cm in
    four steps), and Fpz to Oz over the vertex is ~29 cm (80 % of a ~36 cm nasion-inion
    arc). A chord-for-arc or metre-for-cm slip would move the threshold by enough to
    flip every dataset's verdict, so pin both ends of the scale.
    """
    g = nearest_source_gaps(["C3"], ["C1", "Oz"])
    assert 3.0 < g["C3"] < 5.0          # C1 is the immediate neighbour, one 10-10 step
    g = nearest_source_gaps(["Fpz"], ["Oz"])
    assert 25.0 < g["Fpz"] < 35.0       # front to back, over the vertex


def test_the_reported_rank_is_the_rank_the_array_actually_has():
    """Interpolation is linear, so 22 columns built from 14 electrodes span 14 dimensions
    and their covariance is singular. Discovered the hard way: CSP died with a
    non-positive-definite matrix on exactly this configuration. The number has to be in
    the diagnostics because it decides which estimators are applicable at all.

    Deliberately *not* the smooth field the other tests use: that field is built from a
    6-term spatial basis, so it would cap the array's rank at 6 and the test would pass
    for the wrong reason. Rank is a linear-algebra fact about the projection, independent
    of whether the input is physically plausible, so full-rank noise is the right input.
    """
    rng = np.random.default_rng(1)
    src = ["Fp1", "Fp2", "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4",
           "O1", "Oz", "O2"]  # Zhou2016's real cap
    X = rng.normal(size=(6, len(src), 64))
    Xt, meta = interpolate_to_montage(X, src, SENSORIMOTOR_22, max_gap_cm=None)
    assert meta["rank"] == 14
    flat = Xt.transpose(1, 0, 2).reshape(len(SENSORIMOTOR_22), -1)
    assert np.linalg.matrix_rank(flat) == 14

    # and a dataset that records the target outright stays full rank, so the two tiers of
    # the pool are not silently on different footings
    Xf = rng.normal(size=(6, len(CAP64), 64))
    Ft, mf = interpolate_to_montage(Xf, CAP64, SENSORIMOTOR_22)
    assert mf["rank"] == len(SENSORIMOTOR_22)
    flat = Ft.transpose(1, 0, 2).reshape(len(SENSORIMOTOR_22), -1)
    assert np.linalg.matrix_rank(flat) == len(SENSORIMOTOR_22)


def test_the_shape_and_name_count_must_agree():
    with pytest.raises(ValueError, match="names for"):
        interpolate_to_montage(np.zeros((2, 5, 8)), ["C3", "Cz"], SENSORIMOTOR_22)
