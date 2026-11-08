"""adapt_C.py on synthetic inputs (PREREG_growth_adaptive_size.md, PREREG_growth_signature.md):
inner folds, the size choice, partners, adapter save / re-apply, and both readers on
fabricated runs whose answer is known."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import adapt_B  # noqa: E402
import adapt_C  # noqa: E402
import braindecode.models as bdm  # noqa: E402

C, T = 8, 1000
adapt_B._fastpath_off()


def model(name, seed=0):
    torch.manual_seed(seed)
    return getattr(bdm, adapt_B.ARCH[name])(n_chans=C, n_outputs=2, n_times=T).eval()


def test_inner_folds_disjoint_balanced():
    yc = np.array([0] * 10 + [1] * 10)
    folds = adapt_C.inner_folds(yc, seed=3)
    (a, b), (c, d) = folds
    assert not (a & b).any() and (a | b).all() and (c == b).all() and (d == a).all()
    assert np.bincount(yc[a]).tolist() == [5, 5]
    assert (adapt_C.inner_folds(yc, 3)[0][0] == a).all()          # deterministic


def test_choose_ties_smaller_and_zero():
    assert adapt_C.choose({0: 0.7, 1: 0.8, 2: 0.8, 4: 0.6}) == 1
    assert adapt_C.choose({0: 0.9, 1: 0.8}) == 0


def test_partners_derangement():
    subs = [str(i) for i in range(1, 12)]
    p = adapt_C.partners(subs)
    assert all(s not in v and len(v) == 4 and len(set(v)) == 4 for s, v in p.items())
    assert p == adapt_C.partners(subs)
    assert all(len(v) == 2 for v in adapt_C.partners(["1", "2", "3"]).values())
    one = adapt_C.partners(subs, 1)
    assert sorted(v[0] for v in one.values()) == sorted(subs)      # a permutation


@pytest.mark.parametrize("name", adapt_C.MODELS)
@pytest.mark.parametrize("arm", adapt_C.SAVED)
def test_saved_adapter_reapplies_exactly(name, arm):
    sd = model(name).state_dict()
    Xc, yc = torch.randn(8, C, T), np.array([0, 1] * 4)
    Xt = torch.randn(6, C, T)
    shape, dev = (C, 2, T), torch.device("cpu")
    lg, n, par = adapt_C.adapt(sd, name, arm, 1e-2, Xc, yc, Xt, 5, shape, dev, keep=True)
    assert par is not None and sum(p.numel() for p in par) == n
    again = adapt_C.apply_adapter(sd, name, arm, par, Xc, Xt, 5, shape, dev)
    assert np.allclose(lg, again, atol=1e-5)
    other = adapt_C.apply_adapter(sd, name, arm, [p * 0 for p in par], Xc, Xt, 5, shape, dev)
    z, _ = adapt_B.adapt(sd, name, "adabn", 0.0, Xc, yc, Xt, 5, shape, dev)
    if arm.startswith("lora"):                                       # B = 0 -> AdaBN
        assert np.allclose(other, z, atol=1e-5)
    m = model(name)                                                  # nothing left attached
    with torch.no_grad():
        assert torch.equal(m(Xt), model(name)(Xt))


@pytest.mark.parametrize("name", adapt_C.MODELS)
def test_every_level_trains(name):
    sd = model(name).state_dict()
    Xc, yc, Xt = torch.randn(8, C, T), np.array([0, 1] * 4), torch.randn(4, C, T)
    shape, dev = (C, 2, T), torch.device("cpu")
    for arm in adapt_C.TRAINED:
        lg, n, _ = adapt_C.adapt(sd, name, arm, 1e-2, Xc, yc, Xt, 1, shape, dev)
        assert n > 0 and np.isfinite(lg).all()


# ---------------------------------------------------------------- fabricated readings
DS = ("lee2019_mi", "cho2017")


def fake_run(m, ds, s, opt, informative, rng):
    """Outer score peaks at the subject's level ``opt``; the inner CV sees it or not."""
    out = {"model": m, "dataset": ds, "subject": s, "g0": {"0": {"ok": True}}, "refused": None,
           "k_missing": [], "outer": {}, "inner": {}, "n_params": {}}
    base = 0.7 + 0.05 * rng.standard_normal()
    for k in adapt_C.KS:
        for r in range(3):
            key = f"{k}/0/{r}"
            for b in (0,) + adapt_C.LEVELS:
                for f in adapt_C.FAMILIES:
                    arm = "adabn" if b == 0 else f"{f}{b}"
                    sc = base - 0.02 * abs(np.log2(b + 1) - np.log2(opt + 1)) + (0.005 if f == "grow" else 0)
                    out["outer"].setdefault(arm, {})[key] = sc
                    val = sc if informative else 0.5
                    out["inner"].setdefault(arm, {})[key] = [val, val]
            out["outer"].setdefault("zero", {})[key] = base
    return out


def write_fake(tmp, informative, sig_own_bonus):
    rng = np.random.default_rng(1)
    for m in adapt_C.MODELS:
        for ds in DS:
            for i in range(1, 25):
                s = str(i)
                opt = (0,) + adapt_C.LEVELS
                r = fake_run(m, ds, s, opt[i % 6], informative, rng)
                f = tmp / "runs" / m / ds / f"{s}.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(r))
                if ds == "lee2019_mi":
                    sig = {"model": m, "dataset": ds, "subject": s, "partners": [], "own": {}, "other": {},
                           "own_run": {}}
                    for arm in adapt_C.SAVED:
                        for k in adapt_C.KS:
                            for rr in range(3):
                                key = f"{k}/0/{rr}"
                                own = 0.7 + 0.01 * rng.standard_normal()
                                bonus = sig_own_bonus if arm == "grow8" else sig_own_bonus / 2
                                sig["own"].setdefault(arm, {})[key] = own
                                sig["own_run"].setdefault(arm, {})[key] = own
                                sig["other"].setdefault(arm, {})[key] = {"x": own - bonus}
                    g = tmp / "sig" / m / ds / f"{s}.json"
                    g.parent.mkdir(parents=True, exist_ok=True)
                    g.write_text(json.dumps(sig))
    (tmp / "tune.json").write_text(json.dumps({"lr": {}, "fix": {m: {"grow": 2, "lora": 2} for m in adapt_C.MODELS}}))
    (tmp / "b.json").write_text(json.dumps({"ok_models": list(adapt_B.MODELS)}))


def test_read_informative_cv_beats_permuted_size(tmp_path):
    write_fake(tmp_path, True, 0.02)
    adapt_C.read(str(tmp_path), str(tmp_path / "b.json"))
    res = json.load(open(tmp_path / "adapt_C.json"))
    for k in adapt_C.KS:
        s1 = res["primary"][f"S1 taille du sujet − taille d'un autre (croissance) k={k}"]
        s2 = res["primary"][f"S2 croissance − LoRA, tailles choisies k={k}"]
        s3 = res["primary"][f"S3 croissance : taille choisie − meilleure taille fixe k={k}"]
        assert s1["verdict"] == "SUPÉRIEUR" and s1["mean"] > 0
        assert np.isclose(s2["mean"], 0.005 * 5 / 6) and s2["verdict"] == "SUPÉRIEUR"   # b = 0 for 1/6
        assert s3["mean"] > 0
    sig = json.load(open(tmp_path / "signature_C.json"))
    assert sig["gate"]["ok"]
    t1 = sig["primary"]["T1 croissance (b=8) : son adaptateur − celui d'un autre k=10"]
    t2 = sig["primary"]["T2 T1 croissance − T1 LoRA (b=8) k=10"]
    assert np.isclose(t1["mean"], 0.02) and t1["verdict"] == "SUPÉRIEUR"
    assert np.isclose(t2["mean"], 0.01) and t2["verdict"] == "SUPÉRIEUR"


def test_read_uninformative_cv_has_no_size_effect(tmp_path):
    write_fake(tmp_path, False, 0.0)
    adapt_C.read(str(tmp_path), str(tmp_path / "b.json"))
    res = json.load(open(tmp_path / "adapt_C.json"))
    for k in adapt_C.KS:
        s1 = res["primary"][f"S1 taille du sujet − taille d'un autre (croissance) k={k}"]
        assert np.isclose(s1["mean"], 0.0) and s1["verdict"] != "SUPÉRIEUR"
        assert res["levels"][f"shallow {k} grow"][0] == 1.0       # flat inner -> always b = 0
    sig = json.load(open(tmp_path / "signature_C.json"))
    t1 = sig["primary"]["T1 croissance (b=8) : son adaptateur − celui d'un autre k=50"]
    assert np.isclose(t1["mean"], 0.0) and t1["verdict"] != "SUPÉRIEUR"


def test_sig_gate_refuses_when_reapplication_drifts(tmp_path):
    write_fake(tmp_path, True, 0.02)
    for f in (tmp_path / "sig").rglob("*.json"):
        r = json.loads(f.read_text())
        for arm in r["own_run"]:
            r["own_run"][arm] = {k: v + 0.1 for k, v in r["own_run"][arm].items()}
        f.write_text(json.dumps(r))
    adapt_C.read(str(tmp_path), str(tmp_path / "b.json"))
    assert not (tmp_path / "signature_C.json").exists()
    assert "REFUSE" in (tmp_path / "signature_C.md").read_text()


def test_ctnet_is_out():
    assert "ctnet" not in adapt_C.MODELS
