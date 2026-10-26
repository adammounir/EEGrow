"""The optimizer the benchmark builds is the one the config says, weight decay included.

`weight_decay` was never passed before 14/10: every fit ran under AdamW's own default
while the step-0 PREREG draft said 0. These tests pin the explicit value and fail if a
torch upgrade moves the default the historical records were produced under.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "config"


def _train_cfg(**over):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["train"]
    cfg.update(over)
    return cfg


def _clf(train_cfg, model="bd_deep4"):
    from pipelines import build_pipeline

    model_cfg = yaml.safe_load((ROOT / "model" / f"{model}.yaml").read_text())
    pipe = build_pipeline(model_cfg, train_cfg, n_chans=4, n_times=500, n_outputs=4,
                          sfreq=250.0, device="cpu", seed=0)
    return pipe.named_steps["clf"]


def test_config_default_is_the_historical_adamw_default():
    """0.01 is what every historical record's `optimizer` stamp shows."""
    default = inspect.signature(torch.optim.AdamW.__init__).parameters["weight_decay"]
    assert default.default == 0.01
    assert _train_cfg()["weight_decay"] == 0.01


def test_weight_decay_is_passed_explicitly():
    clf = _clf(_train_cfg())
    assert clf.get_params()["optimizer__weight_decay"] == 0.01


def test_braindecode_deep4_recipe_reaches_the_live_optimizer():
    clf = _clf(_train_cfg(lr=1e-2, weight_decay=5e-4))
    clf.initialize()
    group = clf.optimizer_.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (1e-2, 5e-4)


WN_V = "*.parametrizations.weight.original1"


def test_no_decay_params_default_is_one_group():
    """null = every historical cell: a single group, no `param_groups` in the stamp."""
    from eegrow.training.recording import _optimizer_stamp

    assert _train_cfg()["no_decay_params"] is None
    clf = _clf(_train_cfg(), model="bd_tidnet")
    clf.initialize()
    assert len(clf.optimizer_.param_groups) == 1
    assert "param_groups" not in _optimizer_stamp(clf)


def test_tidnet_weight_norm_direction_is_exempt_from_decay():
    """v (original1) of every weight-normed conv at 0; g and the rest keep 0.01."""
    from eegrow.training.recording import _optimizer_stamp

    clf = _clf(_train_cfg(optimizer="adam", amsgrad=True, weight_decay=0.01,
                          no_decay_params=[WN_V]), model="bd_tidnet")
    clf.initialize()
    v = [n for n, _ in clf.module_.named_parameters() if n.endswith("weight.original1")]
    g = [n for n, _ in clf.module_.named_parameters() if n.endswith("weight.original0")]
    assert v and len(v) == len(g)
    exempt, default = clf.optimizer_.param_groups
    assert exempt["weight_decay"] == 0.0 and len(exempt["params"]) == len(v)
    assert default["weight_decay"] == 0.01
    ids = {id(p) for p in exempt["params"]}
    named = dict(clf.module_.named_parameters())
    assert ids == {id(named[n]) for n in v}
    stamp = _optimizer_stamp(clf)
    assert stamp["weight_decay"] == 0.01 and stamp["amsgrad"] is True
    assert [grp["weight_decay"] for grp in stamp["param_groups"]] == [0.0, 0.01]


def test_no_decay_pattern_matching_nothing_refuses_the_fit():
    import numpy as np
    import pytest

    clf = _clf(_train_cfg(max_epochs=1, no_decay_params=["*.no_such_param"]),
               model="bd_tidnet")
    X = np.random.default_rng(0).standard_normal((16, 4, 500)).astype("float32")
    y = np.arange(16) % 4
    with pytest.raises(ValueError, match="match no parameter"):
        clf.fit(X, y)


def test_no_decay_pipeline_survives_sklearn_clone():
    # MOABB clones the pipeline before each fit; the 26/10 TIDNet test died there.
    import numpy as np
    from sklearn.base import clone

    from eegrow.training.recording import _optimizer_stamp

    clf = clone(_clf(_train_cfg(max_epochs=1,
                                no_decay_params=["*.parametrizations.weight.original1"]),
                     model="bd_tidnet"))
    X = np.random.default_rng(0).standard_normal((16, 4, 500)).astype("float32")
    y = np.arange(16) % 4
    clf.fit(X, y)
    stamp = _optimizer_stamp(clf)
    assert [grp["weight_decay"] for grp in stamp["param_groups"]] == [0.0, 0.01]
