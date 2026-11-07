"""Reading 3: adapt a shared model to a new subject by adding neurons, against LoRA, the
head and AdaBN, on the 5 models and the 11 datasets of campaign A
(analysis/PREREG_growth_adaptation_B.md). Committed with the PREREG BEFORE any adaptation
outside the tuning dataset, and before reading 2's report is opened.

The shared model of subject i is campaign A's ``<model>__bd`` cross_subject fit that left
i out. The neurons are added where campaign A grew each model, and LoRA sits on the layer
those neurons write into:

    shallow    conv_time -> conv_spat                       (as reading 2, adapt_A.Delta)
    eegnex     block_1 conv -> BN -> block_2 conv           (new BN for the new filters)
    conformer  every FFN: Linear(40,160) -> GELU -> Dropout -> Linear(160,40)   (6)
    ctnet      every FFN: same                                                  (6)
    msvtnet    every FFN: linear1(72,72) -> ReLU -> Dropout -> linear2(72,72)   (2)

LoRA rank r on the second layer; growth adds, per junction, the number of neurons whose
parameter count is closest to that LoRA's (at least 1).

    python adapt_B.py tune  --cells CELLS --model M --subject S --out DIR
    python adapt_B.py merge --out DIR
    python adapt_B.py run   --cells CELLS --tasks FILE --line L --out DIR
    python adapt_B.py read  --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import adapt_A
from adapt_A import G0_MAX_FAIL, NI, STEPS, holm, verdict
from embed_A import N_BOOT, cache_of

ARCH = {"shallow": "ShallowFBCSPNet", "eegnex": "EEGNeX", "conformer": "EEGConformer",
        "ctnet": "CTNet", "msvtnet": "MSVTNet"}
MODELS = tuple(ARCH)
TUNE_DS = "bnci2014_001"
MAIN_DS = ("alexmi", "bnci2014_002", "bnci2014_004", "bnci2015_001", "cho2017", "lee2019_mi",
           "physionetmi", "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016")
KS, K_MAIN, K_POS = (5, 10, 25, 50), 10, 50
PS, P_MAIN = (2, 8), 2
LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
TRAINED = ("head", "lora2", "lora8", "grow2", "grow8", "full")
ARMS = ("zero", "adabn") + TRAINED
DRAWS_TWO, DRAWS_ONE = 3, 6     # per direction: 2 x 3 or 1 x 6 adaptations per (subject, k)
CHUNK = 50                      # gradient accumulation; BatchNorm is in eval, so exact
# G0 (rebuild = campaign logits) with the campaign's TF32 in mind: |dlogit| <= 1e-2 x the
# logit scale, argmax equal except on near-ties (saved top-2 margin <= 2 max|dlogit|),
# |dscore| <= 5e-3
G0_REL, G0_SCORE = 1e-2, 5e-3


def _fastpath_off():
    # nn.TransformerEncoderLayer's eval fast path skips linear1/linear2 entirely; PyTorch
    # already refuses it when hooks are attached, this makes it explicit for every arm.
    # TF32 off: every card computes in full float32 (cuDNN allows TF32 by default on
    # ampere/hopper, which is why the campaign's A40 logits differ from a 2080's by ~5e-3)
    import torch
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------- deltas
def _uniform(shape, bound, g):
    import torch
    return (torch.rand(*shape, generator=g) * 2 - 1) * bound


class EEGNeXDelta:
    """block_1 conv (1 -> 8, (1, 64), no bias) -> BN -> block_2 conv (8 -> 32, (1, 64)).
    grow: m new temporal filters, their own BN, and their block_2 weights at 0.
    lora: dW = B A on block_2 conv seen as 32 x (8 * 64), B = 0."""

    def __init__(self, model, kind, p, seed):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.c1, self.c2, bn1 = model.block_1[1], model.block_2[0], model.block_1[2]
        assert self.c1.bias is None and self.c2.bias is None
        n_out, n_in, kh, kw = self.c2.weight.shape
        self.kind, self.bns, self.handles = kind, [], []
        if kind == "grow":
            k1 = self.c1.weight.shape[1:]
            fan = int(np.prod(k1))
            self.wt = torch.nn.Parameter(_uniform((p, *k1), 1 / np.sqrt(fan), g))
            self.bn = torch.nn.BatchNorm2d(p, eps=bn1.eps, momentum=bn1.momentum)
            self.ws = torch.nn.Parameter(torch.zeros(n_out, p, kh, kw))
            self.params = [self.wt, self.ws, *self.bn.parameters()]
            self.bns = [self.bn]
            self.handles.append(self.c1.register_forward_pre_hook(self.keep))
        elif kind == "lora":
            fan = n_in * kh * kw
            self.A = torch.nn.Parameter(_uniform((p, fan), 1 / np.sqrt(fan), g))
            self.B = torch.nn.Parameter(torch.zeros(n_out, p))
            self.params = [self.A, self.B]
        else:
            raise ValueError(kind)
        self.handles.append(self.c2.register_forward_hook(self.hook))

    def keep(self, mod, inp):
        self.x0 = inp[0]

    def hook(self, mod, inp, out):
        import torch.nn.functional as F
        c1, c2 = self.c1, self.c2
        if self.kind == "grow":
            h = self.bn(F.conv2d(self.x0, self.wt, None, c1.stride, c1.padding, c1.dilation))
            w = self.ws
        else:
            h, w = inp[0], (self.B @ self.A).reshape(c2.weight.shape)
        return out + F.conv2d(h, w, None, c2.stride, c2.padding, c2.dilation)


class FFNDelta:
    """Every transformer FFN lin1 -> act -> drop -> lin2. grow: m new hidden units per
    FFN (PyTorch Linear init), outgoing weights at 0. lora: dW = B A on lin2, B = 0."""

    def __init__(self, ffns, kind, p, seed):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.kind, self.bns, self.handles, self.params, self.parts = kind, [], [], [], []
        for lin1, act, drop, lin2 in ffns:
            d_in, d_h, d_out = lin1.in_features, lin1.out_features, lin2.out_features
            part = {"act": act, "drop": drop}
            if kind == "grow":
                part["w1"] = torch.nn.Parameter(_uniform((p, d_in), 1 / np.sqrt(d_in), g))
                part["b1"] = torch.nn.Parameter(_uniform((p,), 1 / np.sqrt(d_in), g))
                part["w2"] = torch.nn.Parameter(torch.zeros(d_out, p))
                self.params += [part["w1"], part["b1"], part["w2"]]
                self.handles.append(lin1.register_forward_pre_hook(self._keep(part)))
            elif kind == "lora":
                part["A"] = torch.nn.Parameter(_uniform((p, d_h), 1 / np.sqrt(d_h), g))
                part["B"] = torch.nn.Parameter(torch.zeros(d_out, p))
                self.params += [part["A"], part["B"]]
            else:
                raise ValueError(kind)
            self.handles.append(lin2.register_forward_hook(self._hook(part)))
            self.parts.append(part)

    @staticmethod
    def _keep(part):
        def f(mod, inp):
            part["x"] = inp[0]
        return f

    def _hook(self, part):
        def f(mod, inp, out):
            import torch.nn.functional as F
            if self.kind == "grow":
                h = part["drop"](part["act"](F.linear(part["x"], part["w1"], part["b1"])))
                return out + F.linear(h, part["w2"])
            return out + F.linear(inp[0], part["B"] @ part["A"])
        return f


class ShallowDelta(adapt_A.Delta):
    """Reading 2's delta, with the interface of the others."""

    def __init__(self, model, kind, p, seed):
        super().__init__(model, kind, p, seed)
        self.bns, self.handles = [], [self.handle]


def ffns(model, name):
    if name == "conformer":
        out = [blk[1].fn[1] for blk in model.transformer]
    elif name == "ctnet":
        out = [blk.feed_forward.module[0] for blk in model.trans.layers]
    elif name == "msvtnet":
        return [(l.linear1, l.activation, l.dropout, l.linear2)
                for l in model.transformer.trans.layers]
    else:
        raise ValueError(name)
    return [(f[0], f[1], f[2], f[3]) for f in out]


def unit_costs(model, name):
    """(parameters of one added neuron, parameters of one LoRA rank), per junction."""
    if name == "shallow":
        cts = model.conv_time_spat
        n_spat, n_time, _, C = cts.conv_spat.weight.shape
        K = cts.conv_time.weight.shape[2]
        return K + 1 + n_spat * C, n_spat + n_time * C
    if name == "eegnex":
        c1, c2 = model.block_1[1], model.block_2[0]
        n_out, n_in, kh, kw = c2.weight.shape
        return int(np.prod(c1.weight.shape[1:])) + 2 + n_out * kh * kw, n_out + n_in * kh * kw
    lin1, _, _, lin2 = ffns(model, name)[0]
    return lin1.in_features + 1 + lin2.out_features, lin2.out_features + lin2.in_features


def n_grow(model, name, p):
    """Neurons per junction for the growth arm of budget p: closest to LoRA rank p."""
    unit, rank = unit_costs(model, name)
    return max(1, int(round(p * rank / unit)))


def make_delta(model, name, kind, p, seed):
    m = n_grow(model, name, p) if kind == "grow" else p
    if name == "shallow":
        return ShallowDelta(model, kind, m, seed)
    if name == "eegnex":
        return EEGNeXDelta(model, kind, m, seed)
    return FFNDelta(ffns(model, name), kind, m, seed)


def remove(delta):
    for h in delta.handles:
        h.remove()


# ---------------------------------------------------------------- model and adaptation
def base_model(name, sd, n_chans, n_outputs, n_times):
    import braindecode.models as bdm
    m = getattr(bdm, ARCH[name])(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    m.load_state_dict(sd)
    return m.eval()


def bns_of(model, extra=()):
    import torch
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)] + list(extra)


def adabn(model, X, extra=(), extra_params=()):
    """Every BatchNorm's running stats = the calibration batch's (one full-batch pass,
    momentum None), as reading 2 did for Shallow's single BatchNorm."""
    import torch
    bns = bns_of(model, extra)
    if not bns:
        return
    moms = []
    for bn in bns:
        bn.reset_running_stats()
        moms.append(bn.momentum)
        bn.momentum = None
    model.eval()
    for bn in bns:
        bn.train()
    with torch.no_grad():
        try:
            model(X)
        except torch.OutOfMemoryError:
            # same full-batch pass on the CPU (the 128-channel datasets on 11 GB cards)
            dev = X.device
            torch.cuda.empty_cache()
            mods, ps = [model, *extra], list(extra_params)
            for mod in mods:
                mod.cpu()
            for q in ps:
                q.data = q.data.cpu()
            for bn in bns:
                bn.reset_running_stats()
            model(X.cpu())
            for mod in mods:
                mod.to(dev)
            for q in ps:
                q.data = q.data.to(dev)
    for bn, mom in zip(bns, moms):
        bn.momentum = mom
        bn.eval()
    model.eval()


def logits_of(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + CHUNK]).float().cpu() for i in range(0, len(X), CHUNK)]).numpy()


def adapt(sd, name, arm, lr, Xc, yc, Xt, seed, shape, dev):
    """(logits on Xt, n trained params) after adapting the shared model with ``arm``."""
    import torch
    torch.manual_seed(seed)
    model = base_model(name, sd, *shape).to(dev)
    if arm == "zero":
        return logits_of(model, Xt), 0
    delta = None
    if arm[:4] in ("grow", "lora"):
        delta = make_delta(model, name, arm[:4], int(arm[4:]), seed)
        for q in delta.params:
            q.data = q.data.to(dev)
        for bn in delta.bns:
            bn.to(dev)
    extra = delta.bns if delta else ()
    adabn(model, Xc, extra, delta.params if delta else ())
    if arm == "adabn":
        return logits_of(model, Xt), 0
    for q in model.parameters():
        q.requires_grad_(False)
    if arm == "head":
        params = list(model.final_layer.parameters())
    elif arm == "full":
        params = list(model.parameters())
    else:
        params = delta.params
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    yc_t = torch.as_tensor(yc, device=dev)
    n = len(yc_t)
    for _ in range(STEPS):
        model.train()
        for bn in bns_of(model, extra):
            bn.eval()
        opt.zero_grad()
        for i in range(0, n, CHUNK):
            loss = torch.nn.functional.cross_entropy(model(Xc[i:i + CHUNK]), yc_t[i:i + CHUNK],
                                                     reduction="sum") / n
            loss.backward()
        opt.step()
    adabn(model, Xc, extra, delta.params if delta else ())
    out = logits_of(model, Xt)
    n_par = sum(q.numel() for q in params)
    if delta is not None:
        remove(delta)
    return out, n_par


# ---------------------------------------------------------------- data
def subject_data(ds, subj, cells_path):
    """X, y, session, run (in the order paradigm.get_data returns them), read from the
    cache the campaign's cells read; a cache miss refuses (adapt_A.subject_data)."""
    from omegaconf import OmegaConf
    from sklearn.preprocessing import LabelEncoder

    from check_cache import build_campaign
    from utils import cache_config
    cache = cache_of([cells_path], ds)
    dataset, par, _ = build_campaign(ds)
    cc = cache_config(OmegaConf.create({"enabled": True, "path": cache, "readonly": True}))

    def miss(subject, *a, **k):
        raise SystemExit(f"REFUSED: cache miss for {ds} subject {subject} in {cache}")
    dataset._get_single_subject_data = miss
    X, labels, meta = par.get_data(dataset=dataset, subjects=[int(subj)], cache_config=cc)
    y = LabelEncoder().fit(sorted(set(labels))).transform(labels)
    return (X.astype(np.float32), y.astype(np.int64), meta["session"].astype(str).to_numpy(),
            meta["run"].astype(str).to_numpy())


def splits(sess, run, y):
    """(rule, [(direction, calibration mask, test mask)], draws per direction).
    >= 2 sessions: first vs last session, both ways. One session: train runs -> test runs
    when runs are named so; else all runs but the last -> the last run; else (one run)
    first half -> second half of each class's trials, in the order get_data returns them."""
    S = sorted(set(sess))
    if len(S) >= 2:
        a, b = sess == S[0], sess == S[-1]
        return "sessions", [(0, a, b), (1, b, a)], DRAWS_TWO
    R = sorted(set(run))
    if any("train" in r for r in R) and any("test" in r for r in R):
        cal = np.array(["train" in r for r in run])
        return "train/test runs", [(0, cal, ~cal & np.array(["test" in r for r in run]))], DRAWS_ONE
    if len(R) >= 2:
        last = run == R[-1]
        return "runs: all but last -> last", [(0, ~last, last)], DRAWS_ONE
    # per class: cho2017 returns all trials of one class, then the other (measured 07/11),
    # so a global first half would hold one class only
    half = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        half[idx[:len(idx) // 2]] = True
    return "per class: first half -> second half", [(0, half, ~half)], DRAWS_ONE


def draws(y, k, subj, d, n):
    out = []
    for r in range(n):
        rng = np.random.default_rng(1000 * int(subj) + 10 * d + r)
        out.append(np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False)
                                           for c in np.unique(y)])))
    return out


def shared_fit(cells_path, ds, name, subj):
    """(state_dict, saved test logits by session) of the <name>__bd cross_subject fit
    that left ``subj`` out."""
    import torch
    from campaignA_check import parse_cells
    c = [c for c in parse_cells(cells_path)
         if (c["prot"], c["ds"], c["model"], c["arm"], c["subj"]) ==
         ("cross_subject", ds, name, "bd", str(subj))]
    assert len(c) == 1, (ds, name, subj, len(c))
    stem = c[0]["csv"][:-4]
    fs = glob.glob(f"{stem}__fits__final/*.pt")
    assert len(fs) == 1, fs
    p = torch.load(fs[0], map_location="cpu", weights_only=False)
    assert str(subj) not in [str(s) for s in p["meta"]["train_subjects"]], p["meta"]
    uid = os.path.basename(fs[0])[:-3]
    saved = {}
    for f in glob.glob(f"{stem}__fits__logits/{uid}*.npz"):
        z = np.load(f)
        se = set(z["session"].tolist())
        assert len(se) == 1 and not (se & set(saved)), (f, se)
        saved[se.pop()] = {"logits": z["logits"], "y": z["y"]}
    assert saved, stem
    return p["state_dict"], saved


def g0(model, saved, X, y, sess, dev, n_classes):
    import torch
    out = {}
    for se, z in saved.items():
        sel = sess == se
        lg, ref = logits_of(model, torch.as_tensor(X[sel], device=dev)), z["logits"]
        ok = len(lg) == len(ref) and np.array_equal(y[sel], z["y"])
        r = {"ok": False, "n": int(sel.sum())}
        if ok:
            top2 = np.sort(ref, 1)[:, -2:]
            dmax = float(np.abs(lg - ref).max())
            mism = lg.argmax(1) != ref.argmax(1)
            ds = abs(adapt_A.score(lg, y[sel], n_classes) - adapt_A.score(ref, y[sel], n_classes))
            r.update(max_abs=dmax, scale=float(max(1.0, np.abs(ref).max())), n_mism=int(mism.sum()),
                     d_score=float(ds))
            ok = (dmax <= G0_REL * r["scale"] and bool(((top2[:, 1] - top2[:, 0])[mism] <= 2 * dmax).all())
                  and ds <= G0_SCORE)
        r["ok"] = bool(ok)
        out[se] = r
    return out


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_subject(ds, name, cells_path, subj, ks, arms_lrs):
    """{arm: {lr: {k: score averaged over the draws}}}, G0, n_params, availability."""
    import torch
    _fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    dev = device()
    X, y, sess, run = subject_data(ds, subj, cells_path)
    sd, saved = shared_fit(cells_path, ds, name, subj)
    n_classes = len(np.unique(y))
    shape = (X.shape[1], n_classes, X.shape[2])
    gate = g0(base_model(name, sd, *shape).to(dev), saved, X, y, sess, dev, n_classes)
    rule, dirs, n_draws = splits(sess, run, y)
    out = {"subject": str(subj), "dataset": ds, "model": name, "g0": gate, "rule": rule,
           "n_classes": n_classes, "scores": {}, "n_params": {}, "k_missing": [], "refused": None}
    if any(len(np.unique(y[t])) < n_classes for _, _, t in dirs):
        out["refused"] = "a test part lacks a class"
        return out
    ok_k = [k for k in ks if all(np.bincount(y[c], minlength=n_classes).min() >= k for _, c, _ in dirs)]
    out["k_missing"] = [k for k in ks if k not in ok_k]
    res = {}
    for d, cal, test in dirs:
        Xc_all, yc_all = X[cal], y[cal]
        Xt, yt = torch.as_tensor(X[test], device=dev), y[test]
        for k in ok_k:
            for r, idx in enumerate(draws(yc_all, k, subj, d, n_draws)):
                Xc, yc = torch.as_tensor(Xc_all[idx], device=dev), yc_all[idx]
                seed = 1000 * int(subj) + 10 * d + r
                for arm, lrs in arms_lrs.items():
                    for lr in lrs:
                        lg, n = adapt(sd, name, arm, lr, Xc, yc, Xt, seed, shape, dev)
                        s = adapt_A.score(lg, yt, n_classes)
                        res.setdefault(arm, {}).setdefault(str(lr), {}).setdefault(str(k), []).append(s)
                        out["n_params"][arm] = n
    out["scores"] = {a: {lr: {k: float(np.mean(v)) for k, v in by.items()} for lr, by in r_.items()}
                     for a, r_ in res.items()}
    return out


# ---------------------------------------------------------------- reading
def sboot(groups):
    """Pooled contrast over datasets: weighted mean of dataset means, weights sqrt(n_d)
    (MOABB's convention); bootstrap resamples subjects within each dataset."""
    groups = {k: np.asarray(v, float) for k, v in groups.items() if len(v)}
    if not groups:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_sup": 1.0,
                "n": 0, "n_ds": 0, "mde": float("nan")}
    w ={k: np.sqrt(len(v)) for k, v in groups.items()}
    tot = sum(w.values())
    rng = np.random.default_rng(0)
    est, b = 0.0, np.zeros(N_BOOT)
    for k in sorted(groups):
        v = groups[k]
        est += w[k] / tot * v.mean()
        b += w[k] / tot * v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return {"mean": float(est), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "p_sup": float((b <= 0).mean()), "n": int(sum(len(v) for v in groups.values())),
            "n_ds": len(groups), "mde": float(2.8016 * b.std(ddof=1))}


def ci(c):
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def load_runs(out):
    R = {}
    for f in glob.glob(f"{out}/runs/*/*/*.json"):
        r = json.load(open(f))
        R.setdefault((r["model"], r["dataset"]), {})[r["subject"]] = r
    return R


def read(out):
    tune = json.load(open(f"{out}/tune.json"))
    R = load_runs(out)
    L = ["# Adapter un modèle partagé en lui ajoutant des neurones — 5 modèles × 11 datasets "
         "(PREREG_growth_adaptation_B.md)", ""]
    excl, cells_refused, sc = {}, [], {}
    for (m, ds), subs in sorted(R.items()):
        failed = [s for s, r in subs.items() if not all(g["ok"] for g in r["g0"].values())]
        if len(failed) > G0_MAX_FAIL:
            cells_refused.append((m, ds, len(failed)))
            continue
        for s, r in subs.items():
            why = "G0" if s in failed else r["refused"]
            if why:
                excl.setdefault((m, ds), {}).setdefault(why, []).append(s)
                continue
            for arm, by in r["scores"].items():
                for k, v in next(iter(by.values())).items():
                    sc.setdefault((m, ds, arm, int(k)), {})[s] = v

    def diff(m, ds, x, y, k):
        a, b = sc.get((m, ds, x, k), {}), sc.get((m, ds, y, k), {})
        return {s: a[s] - b[s] for s in a if s in b}

    L += [f"- taux d'apprentissage (réglés sur {TUNE_DS}) : {json.dumps(tune['lr'])}",
          f"- cellules (modèle, dataset) refusées par G0 (> {G0_MAX_FAIL} sujets) : {cells_refused}",
          f"- sujets exclus : { {f'{m}/{d}': {w: len(v) for w, v in e.items()} for (m, d), e in excl.items()} }", ""]
    # G1 per model
    G1, ok_models = {}, []
    for m in MODELS:
        G1[m] = sboot({ds: list(diff(m, ds, "full", "zero", K_POS).values()) for ds in MAIN_DS})
        if G1[m]["n"] and G1[m]["lo"] > 0:
            ok_models.append(m)
    L += ["## G1 — fine-tuning complet − zéro-shot, k = 50, par modèle", "",
          "| modèle | écart [IC95] | n sujets / datasets | G1 |", "|---|---|---|---|"]
    L += [f"| {m} | {ci(c) if c['n'] else '—'} | {c['n']} / {c.get('n_ds', 0)} | "
          f"{'PASSE' if m in ok_models else 'ÉCHOUE'} |" for m, c in G1.items()]
    crit = {"E1 croissance − LoRA": "lora2", "E2 croissance − tête": "head", "E3 croissance − AdaBN": "adabn"}

    def pooled(models, y, k=K_MAIN, x="grow2"):
        groups = {}
        for ds in MAIN_DS:
            ds_d = [diff(m, ds, x, y, k) for m in models]
            common = set.intersection(*[set(d) for d in ds_d]) if ds_d else set()
            groups[ds] = [np.mean([d[s] for d in ds_d]) for s in sorted(common)]
        return sboot(groups)

    P = {name: pooled(ok_models, y) for name, y in crit.items()} if ok_models else {}
    rej = holm([c["p_sup"] for c in P.values()]) if P else []
    for (name, c), r_ in zip(P.items(), rej):
        c["verdict"] = verdict(c, r_)
    L += ["", f"## Critère principal — groupé (modèles qui passent G1 : {ok_models}), k = {K_MAIN}, "
              f"p = {P_MAIN}, Holm sur 3, marge NI {NI}", "",
          "| critère | écart [IC95] | p sup. | n sujets / datasets | MDE | verdict |", "|---|---|---|---|---|---|"]
    L += [f"| {name} | {ci(c)} | {c['p_sup']:.4f} | {c['n']} / {c['n_ds']} | {c['mde']:.3f} | **{c['verdict']}** |"
          for name, c in P.items()] or ["| — | aucun modèle ne passe G1 : pas de critère | | | | |"]
    S = {(m, name): pooled([m], y) for m in MODELS for name, y in crit.items()}
    rej = holm([c["p_sup"] for c in S.values()])
    for (key, c), r_ in zip(S.items(), rej):
        c["verdict"] = verdict(c, r_) if key[0] in ok_models else "SANS VERDICT (G1)"
    L += ["", "## Secondaire — par modèle (groupé sur les datasets, Holm sur 15)", "",
          "| modèle | " + " | ".join(crit) + " |", "|---|" + "---|" * len(crit)]
    for m in MODELS:
        L.append(f"| {m} | " + " | ".join(f"{ci(S[(m, n)])} **{S[(m, n)]['verdict']}**" for n in crit) + " |")
    L += ["", f"## Rapporté — score moyen par modèle et dataset, k = {K_MAIN}", ""]
    for m in MODELS:
        L += [f"### {m}", "", "| dataset | n | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
        for ds in MAIN_DS:
            z = sc.get((m, ds, "zero", K_MAIN), {})
            L.append(f"| {ds} | {len(z)} | " + " | ".join(
                f"{np.mean(list(sc[(m, ds, a, K_MAIN)].values())):.3f}" if sc.get((m, ds, a, K_MAIN)) else "—"
                for a in ARMS) + " |")
        L.append("")
    rep = {}
    L += ["## Rapporté — contrastes groupés par k (modèles qui passent G1)", "",
          "| contraste | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for x, y in (("grow2", "lora2"), ("grow8", "lora8"), ("grow2", "head"), ("grow8", "head"),
                 ("lora2", "head"), ("grow2", "adabn"), ("adabn", "zero"), ("full", "zero"), ("grow2", "full")):
        row = {k: pooled(ok_models, y, k, x) for k in KS} if ok_models else {}
        rep[f"{x} - {y}"] = row
        L.append(f"| {x} − {y} | " + " | ".join(ci(c) if c["n"] else "—" for c in row.values()) + " |")
    npar = {m: {ds: next(iter(R[(m, ds)].values()))["n_params"] for ds in MAIN_DS if (m, ds) in R} for m in MODELS}
    json.dump({"lr": tune["lr"], "cells_refused": cells_refused, "G1": G1, "ok_models": ok_models,
               "primary": P, "secondary": {f"{k[0]} {k[1]}": v for k, v in S.items()}, "reported": rep,
               "n_params": npar}, open(f"{out}/adapt_B.json", "w"), indent=1, default=str)
    open(f"{out}/adapt_B.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- task lists
PER_LINE = {"shallow": 12, "eegnex": 2, "conformer": 6, "ctnet": 6, "msvtnet": 6}


def write_tasks(cells_path, out):
    """tune_tasks.txt: one 'model subject' per line (bnci2014_001); run_tasks.txt: comma
    lists of 'model dataset subject', one (model, dataset) per line, PER_LINE subjects at
    most. Subjects = those of the campaign's cross_subject bd cells."""
    from campaignA_check import parse_cells
    subs = {}
    for c in parse_cells(cells_path):
        if c["prot"] == "cross_subject" and c["arm"] == "bd" and c["model"] in MODELS:
            subs.setdefault((c["model"], c["ds"]), set()).add(c["subj"])
    tune = [f"{m} {s}" for m in MODELS for s in sorted(subs[(m, TUNE_DS)], key=int)]
    run = []
    for m in MODELS:
        for ds in MAIN_DS:
            ss = sorted(subs[(m, ds)], key=int)
            for i in range(0, len(ss), PER_LINE[m]):
                run.append(",".join(f"{m} {ds} {s}" for s in ss[i:i + PER_LINE[m]]))
    open(f"{out}/tune_tasks.txt", "w").write("\n".join(tune) + "\n")
    open(f"{out}/run_tasks.txt", "w").write("\n".join(run) + "\n")
    print(len(tune), "tuning tasks,", len(run), "run tasks,",
          sum(len(r.split(",")) for r in run), "(model, subject) units")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tasks", "tune", "merge", "run", "read"])
    ap.add_argument("--cells")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--subject")
    ap.add_argument("--tasks")
    ap.add_argument("--line", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "tasks":
        write_tasks(a.cells, a.out)
    elif a.mode == "tune":
        f = f"{a.out}/tune/{a.model}/{a.subject}.json"
        if not os.path.exists(f):
            r = run_subject(TUNE_DS, a.model, a.cells, a.subject, (K_MAIN,), {arm: LRS for arm in TRAINED})
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump(r, open(f + ".tmp", "w"), indent=1)
            os.replace(f + ".tmp", f)
    elif a.mode == "merge":
        best, curve = {}, {}
        for m in MODELS:
            rs = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/tune/{m}/*.json"))]
            bad = [r["subject"] for r in rs if not all(g["ok"] for g in r["g0"].values())]
            if len(rs) != 9 or bad:
                raise SystemExit(f"REFUSED: {m}: {len(rs)} tuning subjects, G0 fails on {bad}")
            curve[m] = {arm: {str(lr): float(np.mean([r["scores"][arm][str(lr)][str(K_MAIN)] for r in rs]))
                              for lr in LRS} for arm in TRAINED}
            best[m] = {arm: float(max(c, key=c.get)) for arm, c in curve[m].items()}
        json.dump({"lr": best, "curve": curve}, open(f"{a.out}/tune.json", "w"), indent=1)
        print(json.dumps(best))
    elif a.mode == "run":
        lr = json.load(open(f"{a.out}/tune.json"))["lr"]
        line = open(a.tasks).read().split("\n")[a.line - 1]
        for item in line.split(","):
            m, ds, s = item.split()
            f = f"{a.out}/runs/{m}/{ds}/{s}.json"
            if os.path.exists(f):
                continue
            arms_lrs = {"zero": [0.0], "adabn": [0.0], **{arm: [lr[m][arm]] for arm in TRAINED}}
            r = run_subject(ds, m, a.cells, s, KS, arms_lrs)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump(r, open(f + ".tmp", "w"), indent=1)
            os.replace(f + ".tmp", f)
            print(f"done {m} {ds} {s}", flush=True)
    else:
        read(a.out)


if __name__ == "__main__":
    main()
