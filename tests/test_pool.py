"""The pool decides which trials each arm trains on, so its failure modes are the ones
that would invalidate a whole grid silently: a held-out subject leaking into training, an
arm getting a different amplitude than the arm it is compared to, or a tier selection
quietly returning the wrong datasets. Nothing here needs MOABB or downloaded data -- the
cache format is written by hand.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import pool as poolmod  # noqa: E402


@pytest.fixture
def fake_pool(tmp_path, monkeypatch):
    """Two datasets, three subjects each, distinguishable amplitudes.

    The amplitudes differ by three orders of magnitude on purpose: that is the real
    situation across amplifiers, and it is what the ``scale`` arm exists to neutralise.
    """
    monkeypatch.setattr(poolmod, "pool_root", lambda: tmp_path)
    rng = np.random.default_rng(0)
    for name, amp in (("ds_a", 1e-5), ("ds_b", 1e-2)):
        d = tmp_path / name
        d.mkdir()
        for s in (1, 2, 3):
            n = 8
            X = (rng.normal(size=(n, 4, 16)) * amp).astype(np.float32)
            np.savez_compressed(
                d / f"sub-{s}.npz", X=X, y=np.arange(n) % 2,
                session=np.asarray(["0"] * n), subject=np.asarray([str(s)] * n),
                dataset=np.asarray([name] * n),
                diag=json.dumps({"present": [], "interpolated": [], "unknown": [],
                                 "n_support": 4, "gaps_cm": {}, "max_gap_cm": 0.0,
                                 "median_gap_cm": 0.0}))
    return tmp_path


def test_excluded_subjects_are_absent_from_the_training_groups(fake_pool):
    """The one bug that would invalidate every number without failing anything."""
    X, y, g = poolmod.load(["ds_a", "ds_b"], exclude=[("ds_a", "2"), ("ds_b", "3")])
    assert "ds_a|2" not in set(g) and "ds_b|3" not in set(g)
    assert len(set(g)) == 4 and len(y) == 32


def test_exclusion_matches_on_string_form_not_object_identity(fake_pool):
    """Subject ids arrive as ints from MOABB and as strings from filenames; an exclusion
    that silently fails to match is a leak, so both spellings must work."""
    _, _, g = poolmod.load(["ds_a"], exclude=[("ds_a", 2)])
    assert "ds_a|2" not in set(g)


def test_restricting_to_one_subject_returns_only_that_subject(fake_pool):
    _, y, g = poolmod.load(["ds_a"], subjects={"ds_a": ["3"]})
    assert set(g) == {"ds_a|3"} and len(y) == 8


def test_loading_nothing_is_an_error_not_an_empty_array(fake_pool):
    """An arm whose training set came out empty must stop, not train on zero trials."""
    with pytest.raises(ValueError, match="nothing loaded"):
        poolmod.load(["ds_a"], exclude=[("ds_a", s) for s in ("1", "2", "3")])


def test_an_unbuilt_dataset_is_named_in_the_error(fake_pool):
    with pytest.raises(FileNotFoundError, match="ds_missing"):
        poolmod.load(["ds_missing"])


def test_scale_equalises_the_subjects_and_keeps_the_overall_amplitude(fake_pool):
    """Two things at once, because they are the two halves of the control's purpose.

    Per subject: the 1000x amplitude gap between the two datasets must be gone, else
    pooling lets one dataset dominate the loss for non-neural reasons. Globally: the arm
    must sit at the same RMS as the raw arm, else it also differs from it by an effective
    learning rate and stops being a clean control.
    """
    Xr, _, g = poolmod.load(["ds_a", "ds_b"])
    Xs, _, gs = poolmod.load(["ds_a", "ds_b"], align="scale")

    stds = [float(Xs[gs == grp].std()) for grp in np.unique(gs)]
    assert max(stds) / min(stds) < 1.05, "subjects are still on different scales"
    raw_ratio = max(float(Xr[g == grp].std()) for grp in np.unique(g)) / \
        min(float(Xr[g == grp].std()) for grp in np.unique(g))
    assert raw_ratio > 100, "the fixture no longer poses the problem"

    rms = lambda a: float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))  # noqa: E731
    assert rms(Xs) == pytest.approx(rms(Xr), rel=1e-3)


def test_euclidean_alignment_also_lands_on_the_raw_amplitude(fake_pool):
    """The three arms must be comparable on amplitude, which is the whole reason
    ``preserve_scale`` exists. If EA and scale ended up at different RMS, the
    ``euclidean - scale`` contrast would not isolate whitening."""
    Xr, _, _ = poolmod.load(["ds_a", "ds_b"])
    Xe, _, _ = poolmod.load(["ds_a", "ds_b"], align="euclidean")
    Xs, _, _ = poolmod.load(["ds_a", "ds_b"], align="scale")
    rms = lambda a: float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))  # noqa: E731
    assert rms(Xe) == pytest.approx(rms(Xr), rel=1e-2)
    assert rms(Xs) == pytest.approx(rms(Xe), rel=1e-2)


def test_every_arm_arrives_at_an_O1_amplitude(fake_pool):
    """The bug this pins cost a whole 90-point grid.

    MOABB serves volts, so the fixture's arrays sit at 1e-5 and 1e-2. Handed to a convnet
    at that scale, ShallowFBCSPNet predicted one class for 99.7% of trials -- accuracy
    exactly 0.5000 in every cell of the grid. Rescaling the same fold to unit RMS took it
    to 0.649. The earlier code rescaled all three arms to the *raw* RMS in the name of
    comparability, which preserved the pathology instead of the comparison.
    """
    for align in ("none", "scale", "euclidean"):
        X, _, _ = poolmod.load(["ds_a", "ds_b"], align=align)
        rms = float(np.sqrt(np.mean(X.astype(np.float64) ** 2)))
        assert rms == pytest.approx(poolmod.TARGET_RMS, rel=1e-4), align
        assert 1e-3 < np.abs(X).max() < 1e3, f"{align}: still in a dead numeric regime"


def test_the_raw_arm_keeps_the_amplitude_gap_the_scale_arm_exists_to_remove(fake_pool):
    """The global rescale must not quietly turn ``none`` into ``scale``.

    ``none`` is the arm that still carries the cross-amplifier confound; if one global
    factor had been applied per subject rather than once overall, ``none`` and ``scale``
    would coincide and the alignment axis would lose its control.
    """
    X, _, g = poolmod.load(["ds_a", "ds_b"], align="none")
    stds = {grp: float(X[g == grp].std()) for grp in np.unique(g)}
    assert max(stds.values()) / min(stds.values()) > 100


def test_alignment_is_per_subject_not_per_dataset(fake_pool):
    """A per-dataset reference would leave the between-subject mixing EA removes."""
    Xe, _, g = poolmod.load(["ds_a", "ds_b"], align="euclidean")
    # after whitening, each subject's mean covariance is proportional to the identity;
    # its off-diagonal mass must be far below the diagonal's
    for grp in np.unique(g):
        block = Xe[g == grp].astype(np.float64)
        C = np.einsum("nct,ndt->cd", block, block) / (block.shape[0] * block.shape[2])
        off = np.abs(C - np.diag(np.diag(C))).mean()
        assert off < 0.1 * np.abs(np.diag(C)).mean()


def test_an_unknown_alignment_is_refused(fake_pool):
    with pytest.raises(ValueError, match="unknown align"):
        poolmod.load(["ds_a"], align="riemannian")


def test_tiers_select_the_datasets_the_ablation_needs():
    """``core`` is the no-interpolation control and must not silently acquire a dataset
    that needs interpolation, or the ablation loses its baseline."""
    core = set(poolmod.tier("core"))
    interp = set(poolmod.tier("interp"))
    assert core == {"bnci2014_001", "cho2017", "lee2019_mi", "physionetmi",
                    "schirrmeister2017", "weibo2014"}
    assert interp == {"shin2017a"}, (
        "zhou2016 belongs to `lowrank`: 14 electrodes make the 22-channel projection "
        "rank-deficient, which would confound the interpolation axis with the "
        "alignment one")
    assert poolmod.tier("lowrank") == ["zhou2016"]
    assert core & interp == set()
    assert set(poolmod.tier("core+interp")) == core | interp
    assert poolmod.tier("extrap") == ["bnci2014_004"]
    # an explicit list must stay an explicit list, not be read as a tier
    assert poolmod.tier("cho2017,physionetmi") == ["cho2017", "physionetmi"]


def test_the_epoch_window_is_the_floor_of_the_pool():
    """3.0 s is imposed by Cho2017 and PhysionetMI, whose MOABB intervals are exactly
    3 s. If WINDOW ever exceeds that, those datasets can no longer be epoched and the
    pool loses 161 of its 290 subjects."""
    assert poolmod.WINDOW == 3.0
    assert poolmod.N_TIMES == 750
