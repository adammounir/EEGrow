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
