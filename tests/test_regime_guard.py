"""The guard decides whether a grow/bd delta is a measurement or an artefact, so its
failure modes are the ones worth pinning down. Every test below is a bug that actually
happened on the production grid, not a hypothetical.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import regime_guard as rg  # noqa: E402

PAIRS = [("grow_shallow", "bd_shallow")]


def _hydra_dir(root, tree, name, ev, ds, model, seed, resample, label=None,
               align=None, mtime=None):
    """One hydra run record. ``resample=None`` writes ``null``, i.e. the native rate.

    The two trees nest at different depths, and that asymmetry is itself a source of
    bugs, so reproduce it faithfully: a sweep is ``multirun/<date>/<overrides>/`` while
    a single run is ``outputs/<date_time>/``.
    """
    rel = f"2026-01-01/{name}" if tree == "multirun" else name
    d = root / tree / rel / ".hydra"
    d.mkdir(parents=True, exist_ok=True)
    ov = [f"eval={ev}", f"dataset={ds}", f"model={model}", f"seed={seed}"]
    if align:
        ov.append(f"align={align}")
    (d / "overrides.yaml").write_text("".join(f"- {o}\n" for o in ov))
    (d / "config.yaml").write_text(
        f"dataset:\n  name: {ds}\n  resample: {'null' if resample is None else resample}\n"
        f"model:\n  name: {model}\n  label: {label or model}\n"
        f"eval:\n  name: {ev}\n")
    if mtime is not None:
        for f in ("overrides.yaml", "config.yaml"):
            os.utime(d / f, (mtime, mtime))
    return d.parent


def _csv(root, ev, ds, stem, mtime=None, sfreq=None):
    d = root / "results" / ev / ds
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.csv"
    cols, vals = ["score", "model"], ["0.5", stem.split("__")[0]]
    if sfreq is not None:
        cols.append("sfreq")
        vals.append(str(sfreq))
    p.write_text(",".join(cols) + "\n" + ",".join(vals) + "\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _log(root, name, csv_path, sfreq, when):
    d = root / "slurm_logs"
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))
    (d / name).write_text(
        f"[{stamp},100][eegrow.benchmark][INFO] - dims: chans=22 times=1000 "
        f"outputs=4 sfreq={sfreq}\n"
        f"[{stamp},400][eegrow.benchmark][INFO] - saved 48 rows -> {csv_path}\n")


T0 = 1_700_000_000.0


def test_csv_sfreq_is_authoritative(tmp_path):
    """A run that recorded its own rate needs no reconstruction at all."""
    _hydra_dir(tmp_path, "outputs", "d1", "within_session", "alexmi", "bd_shallow", "0",
               None, mtime=T0)
    _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 10,
         sfreq=250.0)
    ev = rg.cell_evidence(str(tmp_path))
    assert ev[("within_session", "alexmi", "bd_shallow", "0", "none")] == \
        ("250", rg.CSV_SFREQ)


def test_log_beats_hydra_when_the_record_was_clobbered(tmp_path):
    """The real false negative: hydra says 250, the run actually trained at 512.

    Seven cells of the grid looked clean this way. Their true record lived in an
    ``outputs/`` directory that a concurrent array task overwrote, leaving only an
    older 250 Hz launch for the guard to find.
    """
    _hydra_dir(tmp_path, "multirun", "old", "within_session", "alexmi", "bd_shallow",
               "0", 250, mtime=T0)
    p = _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 100)
    _log(tmp_path, "j_0.out", p, "512.0", T0 + 100)
    rate, tier = rg.cell_evidence(str(tmp_path))[
        ("within_session", "alexmi", "bd_shallow", "0", "none")]
    assert (rate, tier) == ("512", rg.LOG_CERT)


def test_log_of_a_superseded_run_is_ignored(tmp_path):
    """A log only certifies the bytes on disk if its save second is the file's mtime.

    Without that check a log from an earlier run of the same cell -- later overwritten
    -- is read as describing the current file. That is what made a first pass report
    77 contaminated cells where there are 7.
    """
    _hydra_dir(tmp_path, "multirun", "cur", "within_session", "alexmi", "bd_shallow",
               "0", 250, mtime=T0)
    p = _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 500)
    _log(tmp_path, "old_0.out", p, "512.0", T0 + 100)  # ran, then was overwritten
    rate, tier = rg.cell_evidence(str(tmp_path))[
        ("within_session", "alexmi", "bd_shallow", "0", "none")]
    assert (rate, tier) == ("250", rg.HYDRA_MULTIRUN)


def test_latest_launch_wins_across_the_two_hydra_trees(tmp_path):
    """Regression: 'most recent' was decided on a path component, not a date.

    The two trees nest at different depths (``multirun/<date>/<overrides>/`` vs
    ``outputs/<date>/``), so a fixed negative index picked the tree *name* -- a
    constant. max() then fell through to the rate string, where 'NATIVE' > '250'
    lexicographically, and every cell that had ever run at the native rate stayed
    native forever, including the ones a 250 Hz relaunch had already fixed.
    """
    _hydra_dir(tmp_path, "multirun", "a", "within_session", "alexmi", "bd_shallow", "0",
               None, mtime=T0)
    _hydra_dir(tmp_path, "outputs", "b", "within_session", "alexmi", "bd_shallow", "0",
               250, mtime=T0 + 50)
    _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 100)
    assert rg.cell_regimes(str(tmp_path))[
        ("within_session", "alexmi", "bd_shallow", "0", "none")] == "250"


def test_a_launch_that_never_wrote_the_csv_is_not_credited(tmp_path):
    """A relaunch that crashed leaves a config but no result; the CSV is still old."""
    _hydra_dir(tmp_path, "multirun", "ok", "within_session", "alexmi", "bd_shallow",
               "0", None, mtime=T0)
    _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 10)
    _hydra_dir(tmp_path, "outputs", "crashed", "within_session", "alexmi", "bd_shallow",
               "0", 250, mtime=T0 + 900)
    assert rg.cell_regimes(str(tmp_path))[
        ("within_session", "alexmi", "bd_shallow", "0", "none")] == "NATIVE"


def test_classic_ml_cells_are_covered(tmp_path):
    """``ml_csp_lda`` is the config key, ``csp_lda`` the CSV stem.

    Keying on the override name found no CSV, so 900 ML cells left the guard's
    coverage without a word -- silent loss of coverage, worse than a false alarm.
    """
    _hydra_dir(tmp_path, "multirun", "m", "within_session", "alexmi", "ml_csp_lda", "0",
               250, label="csp_lda", mtime=T0)
    _csv(tmp_path, "within_session", "alexmi", "csp_lda__seed0", mtime=T0 + 10)
    assert ("within_session", "alexmi", "csp_lda", "0", "none") in \
        rg.cell_regimes(str(tmp_path))


def test_rates_are_compared_canonically(tmp_path):
    """``250.0`` from the yaml and ``250`` from the CLI are one regime, not two."""
    _hydra_dir(tmp_path, "multirun", "g", "within_session", "alexmi", "grow_shallow",
               "0", "250.0", mtime=T0)
    _hydra_dir(tmp_path, "multirun", "b", "within_session", "alexmi", "bd_shallow", "0",
               250, mtime=T0)
    _csv(tmp_path, "within_session", "alexmi", "grow_shallow__seed0", mtime=T0 + 10)
    _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 10)
    r = rg.cell_regimes(str(tmp_path))
    # both arms resolved, and to the same spelling -- "no mixed pair" would also hold
    # vacuously if the guard had simply failed to find either of them
    assert set(r.values()) == {"250"}
    assert rg.mixed_pairs(r, PAIRS) == []


def test_the_aligned_arm_does_not_borrow_the_raw_arm_rate(tmp_path):
    """Both arms are the same (eval, dataset, model, seed); only ``align`` separates."""
    _hydra_dir(tmp_path, "multirun", "raw", "within_session", "alexmi", "grow_shallow",
               "0", 250, mtime=T0)
    _hydra_dir(tmp_path, "multirun", "ea", "within_session", "alexmi", "grow_shallow",
               "0", None, align="euclidean", mtime=T0 + 5)
    _csv(tmp_path, "within_session", "alexmi", "grow_shallow__seed0", mtime=T0 + 10)
    _csv(tmp_path, "within_session", "alexmi", "grow_shallow__easubject__seed0",
         mtime=T0 + 10)
    r = rg.cell_regimes(str(tmp_path))
    assert r[("within_session", "alexmi", "grow_shallow", "0", "none")] == "250"
    assert r[("within_session", "alexmi", "grow_shallow", "0", "euclidean")] == "NATIVE"


def test_a_mixed_pair_aborts_the_analysis(tmp_path, monkeypatch):
    _hydra_dir(tmp_path, "multirun", "g", "within_session", "alexmi", "grow_shallow",
               "0", 250, mtime=T0)
    _hydra_dir(tmp_path, "multirun", "b", "within_session", "alexmi", "bd_shallow", "0",
               None, mtime=T0)
    _csv(tmp_path, "within_session", "alexmi", "grow_shallow__seed0", mtime=T0 + 10)
    _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0 + 10)
    monkeypatch.delenv("EEGROW_ALLOW_MIXED", raising=False)
    with pytest.raises(SystemExit):
        rg.assert_paired(PAIRS, bench_root=str(tmp_path))


def test_a_csv_without_any_record_is_unknown_not_guessed(tmp_path):
    """No hydra directory, no log: say so rather than inherit a neighbour's rate."""
    _csv(tmp_path, "within_session", "alexmi", "bd_shallow__seed0", mtime=T0)
    rate, tier = rg.cell_evidence(str(tmp_path))[
        ("within_session", "alexmi", "bd_shallow", "0", "none")]
    assert (rate, tier) == ("UNKNOWN", "aucune")
def test_the_scope_narrows_the_question_without_weakening_it(tmp_path, monkeypatch):
    """A cell contaminated elsewhere must not force an analysis that never reads it to
    disable the check for the cells it does read."""
    for ds, rate in (("alexmi", None), ("cho2017", 250)):
        _hydra_dir(tmp_path, "multirun", f"g_{ds}", "cross_subject", ds, "grow_shallow",
                   "0", 250, mtime=T0)
        _hydra_dir(tmp_path, "multirun", f"b_{ds}", "cross_subject", ds, "bd_shallow",
                   "0", rate, mtime=T0)
        _csv(tmp_path, "cross_subject", ds, "grow_shallow__seed0", mtime=T0 + 10)
        _csv(tmp_path, "cross_subject", ds, "bd_shallow__seed0", mtime=T0 + 10)
    monkeypatch.delenv("EEGROW_ALLOW_MIXED", raising=False)

    # unscoped: alexmi is mixed, everything stops
    with pytest.raises(SystemExit):
        rg.assert_paired(PAIRS, bench_root=str(tmp_path))
    # scoped to cho2017 alone: passes, and still on a real check, not a bypass
    _, bad = rg.assert_paired(PAIRS, bench_root=str(tmp_path),
                              scope={("cross_subject", "cho2017")})
    assert bad == []
    # scoped to alexmi: still stops
    with pytest.raises(SystemExit):
        rg.assert_paired(PAIRS, bench_root=str(tmp_path),
                         scope={("cross_subject", "alexmi")})

