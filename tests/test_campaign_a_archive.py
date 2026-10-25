"""Campaign A additions (25/10/2026): the random-direction growth control and the
recorder's new archive flags.

``direction="random"`` is only a valid control if it changes the DIRECTION of the new
neurons and nothing else: same junction, same count, same widths, same amplitude (the
TINY paper's mean squared norm per neuron), and the existing weights untouched.
"""

from __future__ import annotations

import copy

import pytest
import torch
from sklearn.base import clone

from eegrow import FitRecorder, GrowingShallowFBCSPNet
from eegrow.training import loop

C, T, N_CLASSES = 8, 256, 4
DEVICE = "cpu"
TINY = dict(min_singular_ratio=0.1, normalization="tiny_paper", fixed_s=1.0)


def _model():
    torch.manual_seed(0)
    return GrowingShallowFBCSPNet(n_chans=C, n_outputs=N_CLASSES, n_times=T, n_filters_time=4,
                                  n_filters_spat=8, target_n_filters_time=12, device=DEVICE)


def _batches(seed=1, n=4):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(16, C, T, generator=g),
             torch.randint(0, N_CLASSES, (16,), generator=g)) for _ in range(n)]


def _step(model, direction, seed=5):
    torch.manual_seed(seed)  # the random draw comes from the global generator
    return loop.grow_step(model, _batches(), DEVICE, direction=direction, **TINY)


def test_random_direction_changes_only_the_new_neurons():
    a, b = _model(), _model()
    before = {k: v.clone() for k, v in a.state_dict().items()}
    ra, rb = _step(a, "tiny"), _step(b, "random")
    assert ra["applied"] and rb["applied"]
    assert (ra["direction"], rb["direction"]) == ("tiny", "random")
    for key in ("width_before", "width_after", "n_candidates", "n_proposed", "s"):
        assert ra[key] == rb[key], key
    sa, sb = a.state_dict(), b.state_dict()
    assert {k: v.shape for k, v in sa.items()} == {k: v.shape for k, v in sb.items()}
    grown = [k for k in sa if sa[k].shape != before[k].shape]
    assert grown, "the step resized nothing"
    for k in sa:
        if "running" in k or "num_batches" in k:
            continue
        if k in grown:
            # old slice identical in both arms (Algorithm 1: no update of existing
            # weights), new slice different
            old = tuple(slice(0, n) for n in before[k].shape)
            torch.testing.assert_close(sa[k][old], before[k], rtol=0, atol=0)
            torch.testing.assert_close(sb[k][old], before[k], rtol=0, atol=0)
            assert not torch.allclose(sa[k], sb[k])
        else:
            torch.testing.assert_close(sb[k], sa[k], rtol=0, atol=0)


def test_random_direction_keeps_the_tiny_amplitude():
    """Mean squared norm per new neuron = 1e-3 on both sides, in both arms."""
    sq = {}
    for direction in ("tiny", "random"):
        m = _model()
        layer_state = {}
        orig = loop._tiny_paper_normalize

        def spy(layer, sq_norm=loop.TINY_PAPER_SQ_NORM):
            orig(layer, sq_norm)
            om, al = layer.extended_input_layer, layer.previous_module.extended_output_layer
            k = om.weight.shape[1]
            a2 = al.weight.double().pow(2).sum() + (al.bias.double().pow(2).sum()
                                                    if al.bias is not None else 0)
            layer_state.update(omega=float(om.weight.double().pow(2).sum()) / k,
                               alpha=float(a2) / k)
        loop._tiny_paper_normalize = spy
        try:
            _step(m, direction)
        finally:
            loop._tiny_paper_normalize = orig
        sq[direction] = layer_state
    for d in sq.values():
        assert d["omega"] == pytest.approx(1e-3, rel=1e-6)
        assert d["alpha"] == pytest.approx(1e-3, rel=1e-6)


def test_random_direction_depends_on_the_seed_only():
    a, b, c = _model(), _model(), _model()
    _step(a, "random", seed=5); _step(b, "random", seed=5); _step(c, "random", seed=6)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)


@pytest.mark.parametrize("kw", [dict(normalization=None, fixed_s=None),
                                dict(normalization="match_extending_layer", fixed_s=1.0)])
def test_random_direction_refuses_without_the_paper_normalisation(kw):
    with pytest.raises(ValueError, match="tiny_paper"):
        loop.grow_step(_model(), _batches(), DEVICE, direction="random",
                       min_singular_ratio=0.1, **kw)


def test_recorder_new_flags_survive_clone(tmp_path):
    rec = FitRecorder(tmp_path / "x__fits.jsonl", meta={"a": 1}, save_init=True,
                      save_growth=True, save_checkpoints=[10, 25, 50, 100],
                      save_test_logits=True)
    c = clone(rec)
    for k in ("save_init", "save_growth", "save_checkpoints", "save_test_logits"):
        assert getattr(c, k) == getattr(rec, k)


def test_cache_readonly_never_saves():
    """Campaign A reads a cache built and verified beforehand: with readonly, MOABB
    loads every cached level and never writes (base.py
    _get_single_subject_data_using_cache saves only under save_epochs / save_array)."""
    import sys
    from pathlib import Path

    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from utils import cache_config

    rw = cache_config(OmegaConf.create({"enabled": True, "path": "/x"}))
    ro = cache_config(OmegaConf.create({"enabled": True, "path": "/x", "readonly": True}))
    assert rw["save_epochs"] and rw["save_array"] and rw["use"]
    assert not ro["save_epochs"] and not ro["save_array"] and not ro["save_raw"] and ro["use"]
    assert not any(ro[k] for k in ("overwrite_raw", "overwrite_epochs", "overwrite_array"))
    assert {k: v for k, v in ro.items() if not k.startswith("save")} == \
        {k: v for k, v in rw.items() if not k.startswith("save")}


@pytest.mark.parametrize("scoring,y,want", [
    ("roc_auc", ["a", "b", "b", "a"], 0.5),
    ("accuracy", ["l", "r", "f", "f", "t", "l", "l", "r"], 2 / 8),   # share of "f", first sorted
])
def test_logit_scorer_scores_a_diverged_net_as_constant(tmp_path, scoring, y, want):
    """Dry-run 818169: a grown TIDNet turned NaN at epoch 160 and sklearn's ROC-AUC
    refused its probabilities, so the cell died with no CSV, while accuracy would have
    argmaxed the NaN to class 0 in silence. PREREG § 3.1 counts an unstable fit with its
    score: both metrics give the constant predictor, and the file says it diverged."""
    import sys
    from pathlib import Path

    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from subject_stamp import LogitScorer

    class Rec:
        meta, path, last_fit_uid_ = {}, str(tmp_path / "x__fits.jsonl"), "uid0"

    class Net:
        callbacks_ = [("rec", Rec())]

        def forward(self, X, training, device):
            out = torch.zeros(len(X), 4)
            out[0, 0] = float("nan")          # one non-finite logit is enough
            return out

    n = len(y)
    cols = {k: np.array(["1"] * n) for k in ("subject", "session", "run")}
    sc = LogitScorer(scoring, [(np.arange(n), cols)], tmp_path / "logits")
    assert sc(Net(), np.zeros((n, 2, 3), dtype="float32"), np.array(y)) == pytest.approx(want)
    z = np.load(tmp_path / "logits" / "uid0.npz")
    assert bool(z["diverged"]) and float(z["score"]) == pytest.approx(want)
