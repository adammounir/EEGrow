"""The scaling step that guards the deep arms against the unit their input arrives in.

A convnet handed volt-scale EEG collapses to a constant prediction; ``pool.py`` reaches for
Epochs and so bypasses MOABB's ``unit_factor``, which is where that actually bit us. On the
MOABB array path the step is nearly a no-op (measured: -0.002 over 8871 paired folds), and
it is kept so the arms do not silently depend on an upstream default. These tests pin the
three properties that make it correct rather than merely present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from pipelines import RMSScaler  # noqa: E402


def test_volt_scale_input_comes_out_at_unit_rms():
    X = (np.random.default_rng(0).normal(size=(32, 8, 64)) * 5e-6).astype("float32")
    Xt = RMSScaler().fit(X).transform(X)
    assert float(np.sqrt(np.mean(Xt.astype(np.float64) ** 2))) == pytest.approx(1.0,
                                                                               rel=1e-5)


def test_the_test_set_is_scaled_by_the_constant_fitted_on_train():
    """A scaler that refitted on test would leak a test statistic, however scalar.

    It would also break the comparison it exists to serve: two folds whose test sets
    happened to differ in amplitude would be handed to the net at the same scale, hiding
    exactly the between-recording variation the alignment axis is measuring.
    """
    rng = np.random.default_rng(1)
    Xtr = (rng.normal(size=(32, 8, 64)) * 1e-5).astype("float32")
    Xte = (rng.normal(size=(16, 8, 64)) * 1e-2).astype("float32")  # 1000x louder
    s = RMSScaler().fit(Xtr)
    ratio_before = float(Xte.std()) / float(Xtr.std())
    ratio_after = float(s.transform(Xte).std()) / float(s.transform(Xtr).std())
    assert ratio_after == pytest.approx(ratio_before, rel=1e-3)


def test_the_deep_pipeline_carries_the_scaler_and_the_classic_ones_do_not():
    """The classic arms must stay untouched: they are scale-invariant by construction, so
    adding the step there could only introduce noise into a fixed reference."""
    import yaml

    cfg = Path(__file__).resolve().parent.parent / "benchmarks" / "config"
    train = {"lr": 6.25e-4, "max_epochs": 2, "batch_size": 64}
    from pipelines import build_pipeline

    deep = build_pipeline(yaml.safe_load((cfg / "model" / "bd_shallow.yaml").read_text()),
                          train, n_chans=8, n_times=128, n_outputs=2, sfreq=250.0,
                          device="cpu", seed=0)
    assert "rms" in dict(deep.named_steps)

    ml = build_pipeline(yaml.safe_load((cfg / "model" / "ml_csp_lda.yaml").read_text()),
                        train, n_chans=8, n_times=128, n_outputs=2, sfreq=250.0,
                        device="cpu", seed=0)
    assert not any(isinstance(s, RMSScaler) for _, s in ml.steps)
