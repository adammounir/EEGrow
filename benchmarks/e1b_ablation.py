"""E1b -- does the full transformer need the FFNs that growth leaves narrow?
Test-time ablation and pruning of every transformer FFN on campaign A's archived final
weights (analysis/PREREG_E1b_ffn_ablation.md). Committed with the PREREG BEFORE any
ablation score is read. CPU, read-only on the archive, no training.

    python e1b_ablation.py selftest --cells A.txt --out DIR
    python e1b_ablation.py tasks    --cells A.txt --out DIR
    python e1b_ablation.py run      --cells A.txt --tasks FILE --line L --out DIR
    python e1b_ablation.py read     --cells A.txt --tasks FILE --out DIR

Every FFN is ``lin1 -> act -> dropout -> lin2`` and its output is added back by the
block's residual (Conformer ``x + FFN(LN(x))``, CTNet ``LN(x + FFN(x))``, MSVTNet
``nn.TransformerEncoderLayer``; braindecode 1.5.2 source). One forward hook on ``lin2``
does every condition:

    mean  hidden units outside ``keep`` -> their mean activation on the scored test
          trials (all positions); keep = nothing gives the FFN's mean output
    zero  the FFN's output -> 0 (the branch is removed)

A grown fit is rebuilt as the braindecode model with its FFN Linears resized to the
grown widths (a GrowingPair computes ``second(drop(act(first(x))))``, the same function);
the rebuild must give back the campaign's own test logits (G0) or the fit is excluded.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_BOOT = 10_000
ALPHA = 0.025                   # one-sided, Holm over the 3 protocols (campaign A § 3.2)
NI = -0.02                      # campaign A's non-inferiority margin
G0_REL, G0_SCORE = 1e-2, 5e-3   # adapt_B.g0's rule
G0_MAX_FAIL = 0.05              # > 5 % of a (model, arm, protocol)'s fits fail G0 -> refused
CHUNK = 50
MODELS = ("conformer", "ctnet", "msvtnet")
ARMS = ("bd", "tiny")
PROTS = ("within_session", "cross_session", "cross_subject")
N_BLOCKS = {"conformer": 6, "ctnet": 6, "msvtnet": 2}
PRIMARY = "conformer"


# ---------------------------------------------------------------- rebuild
def ffn_slots(name, net):
    """[(container, key of lin1, key of lin2)] per block; int key = Sequential index."""
    if name == "conformer":
        return [(blk[1].fn[1], 0, 3) for blk in net.transformer]
    if name == "ctnet":
        return [(lay.feed_forward.module[0], 0, 3) for lay in net.trans.layers]
    if name == "msvtnet":
        return [(lay, "linear1", "linear2") for lay in net.transformer.trans.layers]
    raise ValueError(name)


def _get(c, k):
    return c[k] if isinstance(k, int) else getattr(c, k)


def _set(c, k, m):
    if isinstance(k, int):
        c[k] = m
    else:
        setattr(c, k, m)


def rename(name, sd):
    """Growing wrapper keys -> braindecode keys (names read on the archive, 09/11)."""
    out, alias = {}, {}
    for k, v in sd.items():
        if k.startswith("pairs."):
            # MultiJunctionGrowingModel.pairs (a ModuleList) holds the SAME modules as the
            # net, so its keys are aliases: checked equal below, then dropped
            alias[k] = v
            continue
        if not k.startswith("net."):
            raise SystemExit(f"REFUSED: grown state_dict key outside .net: {k}")
        k = k[4:]
        if name == "msvtnet":
            k = k.replace(".linear1.first.layer.", ".linear1.").replace(".linear1.second.layer.", ".linear2.")
        else:
            k = k.replace(".first.layer.", ".0.").replace(".second.layer.", ".3.")
        out[k] = v
    for k, v in alias.items():
        _, i, side, rest = k.split(".", 3)          # pairs.<i>.<first|second>.layer.<w|b>
        P = lin1_key(name, int(i))[:-len("weight")]  # ...linear1. / ...0.
        if side == "second":
            P = P[:-len("linear1.")] + "linear2." if name == "msvtnet" else P[:-2] + "3."
        t = P + rest.split(".", 1)[1]
        if t not in out or not torch_equal(out[t], v):
            raise SystemExit(f"REFUSED: alias {k} differs from {t}")
    return out


def torch_equal(a, b):
    import torch
    return tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a, b))


def lin1_key(name, i):
    return {"conformer": f"transformer.{i}.1.fn.1.0.weight",
            "ctnet": f"trans.layers.{i}.feed_forward.module.0.0.weight",
            "msvtnet": f"transformer.trans.layers.{i}.linear1.weight"}[name]


def build(name, sd, n_chans, n_outputs, n_times):
    """(model in eval, FFN hidden widths). strict load: any unmatched key raises."""
    import torch
    import braindecode.models as bdm
    from adapt_B import ARCH
    if any(k.startswith("net.") for k in sd):
        sd = rename(name, sd)
    net = getattr(bdm, ARCH[name])(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    widths = []
    for i, (c, k1, k2) in enumerate(ffn_slots(name, net)):
        h = int(sd[lin1_key(name, i)].shape[0])
        l1, l2 = _get(c, k1), _get(c, k2)
        if h != l1.out_features:
            _set(c, k1, torch.nn.Linear(l1.in_features, h, bias=l1.bias is not None))
            _set(c, k2, torch.nn.Linear(h, l2.out_features, bias=l2.bias is not None))
        widths.append(h)
    net.load_state_dict(sd, strict=True)
    return net.eval(), widths


def importance(name, net):
    """s_j = ||W1[j]|| ||W2[:, j]|| per hidden unit, per block (E1's measure)."""
    out = []
    for c, k1, k2 in ffn_slots(name, net):
        W1, W2 = _get(c, k1).weight.detach().double(), _get(c, k2).weight.detach().double()
        out.append((W1.norm(dim=1) * W2.norm(dim=0)).numpy())
    return out


# ---------------------------------------------------------------- ablation
class Ablator:
    """One forward hook per lin2. ``spec[j]`` = None (untouched), ("zero",) or
    ("mean", keep bool array). ``collect`` sums lin2's input for the means."""

    def __init__(self, name, net):
        self.lin2 = [_get(c, k2) for c, _, k2 in ffn_slots(name, net)]
        self.B = len(self.lin2)
        self.spec, self.mu, self.acc = [None] * self.B, [None] * self.B, None
        self.handles = [l.register_forward_hook(self._hook(j)) for j, l in enumerate(self.lin2)]

    def _hook(self, j):
        def f(mod, inp, out):
            import torch
            import torch.nn.functional as F
            h = inp[0]
            if self.acc is not None:
                flat = h.reshape(-1, h.shape[-1]).double()
                self.acc[j][0] += flat.sum(0)
                self.acc[j][1] += flat.shape[0]
            s = self.spec[j]
            if s is None:
                return None
            if s[0] == "zero":
                return torch.zeros_like(out)
            keep = torch.as_tensor(s[1], device=h.device)
            return F.linear(torch.where(keep, h, self.mu[j].to(h.dtype)), mod.weight, mod.bias)
        return f

    def start_collect(self):
        import torch
        self.acc = [[torch.zeros(l.in_features, dtype=torch.float64), 0] for l in self.lin2]

    def stop_collect(self):
        self.mu = [(s / max(n, 1)).float() for s, n in self.acc]
        self.acc = None

    def set(self, spec):
        assert len(spec) == self.B
        self.spec = list(spec)


def top_keep(s, k):
    """Boolean mask keeping the k units of largest importance (ties: lower index)."""
    keep = np.zeros(len(s), bool)
    keep[np.argsort(-s, kind="stable")[:k]] = True
    return keep


def uniform_alloc(total, widths):
    """``total`` units split equally over the blocks, remainder one each from the LAST
    block (so the remainder never favours block 0), capped at each block's width."""
    B = len(widths)
    a = [total // B] * B
    for j in range(total - sum(a)):
        a[B - 1 - j] += 1
    return [min(x, w) for x, w in zip(a, widths)]


def conditions(B, widths, imp, alloc_tiny=None):
    """{name: spec}. m = mean ablation, z = zero ablation, p = pruning (mean-replaced)."""
    off = lambda j: ("mean", np.zeros(widths[j], bool))
    C = {"full": [None] * B}
    for j in range(B):
        C[f"m{j}"] = [off(i) if i == j else None for i in range(B)]
        C[f"z{j}"] = [("zero",) if i == j else None for i in range(B)]
    C["m_deep"] = [None] + [off(i) for i in range(1, B)]
    C["z_deep"] = [None] + [("zero",)] * (B - 1)
    C["m_all"] = [off(i) for i in range(B)]
    C["z_all"] = [("zero",)] * B
    if alloc_tiny is not None:
        for tag, a in (("p_tiny", alloc_tiny), ("p_unif", uniform_alloc(sum(alloc_tiny), widths))):
            C[tag] = [None if a[j] >= widths[j] else ("mean", top_keep(imp[j], a[j])) for j in range(B)]
    return C


# ---------------------------------------------------------------- data and fits
def logits_of(model, X):
    import torch
    with torch.no_grad():
        return torch.cat([model(torch.as_tensor(X[i:i + CHUNK])).float()
                          for i in range(0, len(X), CHUNK)]).numpy()


def score(lg, y, n_classes):
    from campaignA_check import score_of
    return score_of({"logits": lg, "y": y}, n_classes)


def g0(lg, ref, y, n_classes):
    """adapt_B.g0's rule on one scored group: |dlogit| <= 1e-2 x logit scale, argmax equal
    except on near-ties, |dscore| <= 5e-3."""
    if lg.shape != ref.shape:
        return {"ok": False, "why": "shape"}
    dmax = float(np.abs(lg - ref).max())
    mism = lg.argmax(1) != ref.argmax(1)
    top2 = np.sort(ref, 1)[:, -2:]
    ds = abs(score(lg, y, n_classes) - score(ref, y, n_classes))
    scale = float(max(1.0, np.abs(ref).max()))
    ok = dmax <= G0_REL * scale and bool(((top2[:, 1] - top2[:, 0])[mism] <= 2 * dmax).all()) and ds <= G0_SCORE
    return {"ok": bool(ok), "max_abs": dmax, "scale": scale, "n_mism": int(mism.sum()), "d_score": float(ds)}


def subject_cells(cells_path, ds, subj):
    from campaignA_check import parse_cells
    return [c for c in parse_cells(cells_path)
            if c["ds"] == ds and c["subj"] == str(subj) and c["model"] in MODELS and c["arm"] in ARMS]


def fit_files(c, counts):
    """[(uid, final .pt, [logits .npz])] of one cell; a cell whose final weights do not
    match its fits.jsonl is excluded and counted."""
    stem = c["csv"][:-4]
    try:
        uids = [json.loads(l)["fit_uid"] for l in open(stem + "__fits.jsonl")]
    except Exception as e:
        counts[f"cell_unreadable:{type(e).__name__}"] += 1
        return []
    out = []
    for u in uids:
        pt = f"{stem}__fits__final/{u}.pt"
        lg = sorted(glob.glob(f"{stem}__fits__logits/{u}.npz") + glob.glob(f"{stem}__fits__logits/{u}__g*.npz"))
        if not os.path.exists(pt) or not lg:
            counts["fit_missing_files"] += 1
            continue
        out.append((u, pt, lg))
    return out


def groups_of(c, npzs, X, y, sess, subj):
    """[(session, X_g, y_g, saved logits, test_idx)] or None (refused). cross_subject:
    test_idx indexes the whole dataset, the subject's session rows are taken instead
    (adapt_B.shared_fit); otherwise test_idx indexes the subject's get_data rows."""
    out = []
    for f in npzs:
        z = np.load(f)
        if set(np.asarray(z["subject"]).astype(str).tolist()) != {str(subj)}:
            return None
        se = set(np.asarray(z["session"]).astype(str).tolist())
        if len(se) != 1:
            return None
        se = se.pop()
        if c["prot"] == "cross_subject":
            sel = np.flatnonzero(sess == se)
        else:
            sel = np.asarray(z["test_idx"])
        if len(sel) != len(z["y"]) or not np.array_equal(y[sel], z["y"]):
            return None
        out.append((se, X[sel], y[sel], z["logits"], np.asarray(z["test_idx"])))
    return out


def run_subject(cells_path, ds, subj, out, quiet=False):
    import torch
    import adapt_B
    adapt_B._fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    counts = collections.Counter()
    t0 = time.time()
    X, y, sess, _ = adapt_B.subject_data(ds, subj, cells_path)
    t_data = time.time() - t0
    cells = subject_cells(cells_path, ds, subj)
    # load every fit first: bd's pruning needs the matched tiny fit's widths and test set
    fits = []
    for c in cells:
        for uid, pt, npzs in fit_files(c, counts):
            p = torch.load(pt, map_location="cpu", weights_only=False)
            m = p["meta"]
            key = (c["model"], c["prot"], m.get("cv_ind"), str(m.get("session")), tuple(sorted(m.get("test_sessions") or [])))
            fits.append({"c": c, "uid": uid, "p": p, "npzs": npzs, "key": key})
    tiny = {f["key"]: f for f in fits if f["c"]["arm"] == "tiny"}
    records = []
    for f in fits:
        c, p = f["c"], f["p"]
        name, B = c["model"], N_BLOCKS[c["model"]]
        rec = {"model": name, "arm": c["arm"], "prot": c["prot"], "ds": ds, "subj": str(subj),
               "uid": f["uid"], "key": [str(k) for k in f["key"][2:]], "excluded": None, "groups": []}
        records.append(rec)
        tf = time.time()
        sd = p["state_dict"]
        if not all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()):
            rec["excluded"] = "nonfinite_weights"
            continue
        G = groups_of(c, f["npzs"], X, y, sess, subj)
        if G is None:
            rec["excluded"] = "test_set_mismatch"
            continue
        if any(not np.isfinite(g[3]).all() for g in G):
            rec["excluded"] = "diverged_logits"
            continue
        n_classes = int(G[0][3].shape[1])
        if n_classes == 2 and any(len(np.unique(g[2])) < 2 for g in G):
            rec["excluded"] = "one_class_test"
            continue
        model, widths = build(name, sd, X.shape[1], n_classes, X.shape[2])
        rec["widths"] = widths
        imp = importance(name, model)
        alloc = None
        if c["arm"] == "bd":
            t = tiny.get(f["key"])
            if t is None:
                rec["no_tiny_match"] = "missing"
            else:
                Gt = groups_of(t["c"], t["npzs"], X, y, sess, subj)
                same = Gt is not None and len(Gt) == len(G) and all(
                    a[0] == b[0] and np.array_equal(a[4], b[4]) for a, b in zip(G, Gt))
                jw = t["p"].get("junction_widths")
                if not same or jw is None or len(jw) != B:
                    rec["no_tiny_match"] = "test_set_or_widths"
                else:
                    alloc = [int(w) for w in jw]
                    rec["alloc_tiny"], rec["alloc_unif"] = alloc, uniform_alloc(sum(alloc), widths)
        C = conditions(B, widths, imp, alloc)
        ab = Ablator(name, model)
        for se, Xg, yg, ref, _ in G:
            ab.set(C["full"])
            ab.start_collect()
            lg_full = logits_of(model, Xg)
            ab.stop_collect()
            gr = {"session": se, "n": int(len(yg)), "g0": g0(lg_full, ref, yg, n_classes), "scores": {}}
            for tag, spec in C.items():
                ab.set(spec)
                lg = lg_full if tag == "full" else logits_of(model, Xg)
                gr["scores"][tag] = score(lg, yg, n_classes) if np.isfinite(lg).all() else None
            rec["groups"].append(gr)
        rec["sec"] = round(time.time() - tf, 2)
        if not all(g["g0"]["ok"] for g in rec["groups"]):
            rec["excluded"] = "g0"
    res = {"ds": ds, "subj": str(subj), "n_cells": len(cells), "n_fits": len(fits), "counts": dict(counts),
           "t_data": round(t_data, 1), "t_total": round(time.time() - t0, 1), "fits": records}
    os.makedirs(f"{out}/runs/{ds}", exist_ok=True)
    tmp = f"{out}/runs/{ds}/{subj}.json.tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, f"{out}/runs/{ds}/{subj}.json")
    if not quiet:
        print(ds, subj, "cells", len(cells), "fits", len(fits), dict(counts), "t", res["t_total"])
    return res


# ---------------------------------------------------------------- selftest
def selftest(cells_path, out):
    """Synthetic checks of the hook, then timing and G0 on two real subjects. Prints NO
    ablation score (the PREREG forbids reading one before the freeze)."""
    import torch
    import adapt_B
    adapt_B._fastpath_off()
    torch.manual_seed(0)
    n_ok = 0
    import braindecode.models as bdm
    from adapt_B import ARCH
    for name in MODELS:
        net = getattr(bdm, ARCH[name])(n_chans=8, n_outputs=3, n_times=1000).eval()
        X = torch.randn(7, 8, 1000).numpy()
        B = N_BLOCKS[name]
        ab = Ablator(name, net)
        ab.set([None] * B)
        ab.start_collect()
        base = logits_of(net, X)
        ab.stop_collect()
        # 1. mean mode keeping every unit = the untouched model, exactly
        ab.set([("mean", np.ones(_get(c, k1).out_features, bool)) for c, k1, _ in ffn_slots(name, net)])
        assert np.array_equal(logits_of(net, X), base), name
        # 2. mean mode keeping nothing: lin2's output is constant over tokens = W2 mu + b2
        seen = []
        h = ab.lin2[0].register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
        ab.set([("mean", np.zeros(_get(c, k1).out_features, bool)) for c, k1, _ in ffn_slots(name, net)])
        logits_of(net, X)
        h.remove()
        o = seen[0].reshape(-1, seen[0].shape[-1])
        want = torch.nn.functional.linear(ab.mu[0], ab.lin2[0].weight, ab.lin2[0].bias)
        assert torch.allclose(o, want.expand_as(o), atol=1e-5), name
        # 3. zero mode: lin2 outputs 0, and the logits change
        ab.set([("zero",)] * B)
        assert not np.allclose(logits_of(net, X), base), name
        # 4. rebuild through the grown key names at OTHER widths (strict load): block 0
        # cut to its first 5 units must equal the full net with units 5.. silenced
        sd = {}
        for k, v in net.state_dict().items():
            for i in range(B):
                P = lin1_key(name, i)[:-len("linear1.weight" if name == "msvtnet" else "0.weight")]
                a, b = (P + "linear1.", P + "linear2.") if name == "msvtnet" else (P + "0.", P + "3.")
                ga, gb = (P + "linear1.first.layer.", P + "linear1.second.layer.") if name == "msvtnet" \
                    else (P + "first.layer.", P + "second.layer.")
                if k.startswith(a):
                    k, v = ga + k[len(a):], (v[:5] if i == 0 else v)
                elif k.startswith(b):
                    k, v = gb + k[len(b):], (v[:, :5] if i == 0 and k.endswith("weight") else v)
            sd["net." + k] = v
        for side in ("first", "second"):   # the archive's pairs.* aliases (block 0 is the first)
            k0 = next(k for k in sd if k.endswith(f"{side}.layer.weight"))
            sd[f"pairs.0.{side}.layer.weight"] = sd[k0].clone()
        m2, w2 = build(name, sd, 8, 3, 1000)
        h0 = _get(*ffn_slots(name, net)[0][:2]).out_features
        assert w2 == [5] + [h0] * (B - 1), (name, w2)
        ab.mu = [torch.zeros_like(m) for m in ab.mu]
        ab.set([("mean", np.arange(h0) < 5)] + [None] * (B - 1))
        assert np.allclose(logits_of(m2, X), logits_of(net, X), atol=1e-5), name
        n_ok += 4
    # 5. allocation helpers
    assert uniform_alloc(232, [160] * 6) == [38, 38, 39, 39, 39, 39]
    assert sum(uniform_alloc(232, [160] * 6)) == 232
    assert top_keep(np.array([1., 3., 2., 3.]), 2).tolist() == [False, True, False, True]
    C = conditions(6, [160] * 6, [np.arange(160.)] * 6, [152] + [16] * 5)
    assert C["p_tiny"][0][1].sum() == 152 and C["p_unif"][0][1].sum() == 38 and C["p_tiny"][1][1].sum() == 16
    n_ok += 5
    print(f"synthetic checks: {n_ok} passed")
    for ds, subj in (("bnci2014_001", "1"), ("lee2019_mi", "1")):
        r = run_subject(cells_path, ds, subj, out, quiet=True)
        by = collections.Counter((f["model"], f["arm"], f["excluded"]) for f in r["fits"])
        secs = collections.defaultdict(list)
        for f in r["fits"]:
            if "sec" in f:
                secs[f["model"]].append(f["sec"])
        print(ds, subj, "cells", r["n_cells"], "fits", r["n_fits"], r["counts"], "t_data", r["t_data"],
              "t_total", r["t_total"])
        print("  fits by (model, arm, excluded):", dict(by))
        print("  seconds per fit:", {m: round(float(np.mean(v)), 2) for m, v in secs.items()})
        print("  G0 max |dlogit|/scale:", max(g["g0"].get("max_abs", np.inf) / g["g0"].get("scale", 1)
                                             for f in r["fits"] for g in f["groups"]))


# ---------------------------------------------------------------- tasks
def tasks(cells_path, out):
    from campaignA_check import parse_cells
    T = sorted({(c["ds"], c["subj"]) for c in parse_cells(cells_path) if c["model"] in MODELS and c["arm"] in ARMS},
               key=lambda t: (t[0], int(t[1])))
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/tasks.txt", "w") as fh:
        for ds, s in T:
            fh.write(f"{ds} {s}\n")
    print(len(T), "tasks")


# ---------------------------------------------------------------- reading
def sboot(groups):
    """adapt_B.sboot (sqrt(n_d)-weighted mean of dataset means, subjects resampled within
    each dataset, default_rng(0)), plus the non-inferiority p."""
    groups = {k: np.asarray(v, float) for k, v in groups.items() if len(v)}
    if not groups:
        return None
    w = {k: np.sqrt(len(v)) for k, v in groups.items()}
    tot = sum(w.values())
    rng = np.random.default_rng(0)
    est, b = 0.0, np.zeros(N_BOOT)
    for k in sorted(groups):
        v = groups[k]
        est += w[k] / tot * v.mean()
        b += w[k] / tot * v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return {"mean": float(est), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "p_sup": float((b <= 0).mean()), "p_ni": float((b <= NI).mean()),
            "n": int(sum(len(v) for v in groups.values())), "n_ds": len(groups),
            "mde": float(2.8016 * b.std(ddof=1)),
            "by_ds": {k: [float(v.mean()), len(v)] for k, v in sorted(groups.items())}}


def holm(ps, alpha=ALPHA):
    order = np.argsort(ps)
    rej, m = [False] * len(ps), len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def contrast(fits, a, b):
    """{ds: [subject means of (score_a - score_b), averaged over the fit's groups]}."""
    acc = collections.defaultdict(list)
    for f in fits:
        v = [g["scores"][a] - g["scores"][b] for g in f["groups"]
             if g["scores"].get(a) is not None and g["scores"].get(b) is not None]
        if v:
            acc[(f["ds"], f["subj"])].append(float(np.mean(v)))
    g = collections.defaultdict(list)
    for (ds, _), v in sorted(acc.items()):
        g[ds].append(float(np.mean(v)))
    return g


def spearman_cost_alloc(fits):
    """Per bd fit with a tiny match: Spearman across blocks between the ablation cost
    s(full) - s(m_j) and the tiny width fraction; subject mean, by dataset."""
    from scipy.stats import spearmanr
    acc = collections.defaultdict(list)
    for f in fits:
        if "alloc_tiny" not in f:
            continue
        B = len(f["widths"])
        if any(g["scores"].get(t) is None for g in f["groups"] for t in ["full"] + [f"m{j}" for j in range(B)]):
            continue
        cost = [np.mean([g["scores"]["full"] - g["scores"][f"m{j}"] for g in f["groups"]]) for j in range(B)]
        frac = [a / w for a, w in zip(f["alloc_tiny"], f["widths"])]
        if np.std(cost) == 0 or np.std(frac) == 0:
            continue
        acc[(f["ds"], f["subj"])].append(float(spearmanr(cost, frac)[0]))
    g = collections.defaultdict(list)
    for (ds, _), v in sorted(acc.items()):
        g[ds].append(float(np.mean(v)))
    return g


def ci(c):
    return "—" if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']}, {c['n_ds']} ds, MDE {c['mde']:.3f})"


def read(cells_path, tasks_file, out):
    T = [tuple(l.split()) for l in open(tasks_file) if l.strip()]
    missing = [t for t in T if not os.path.exists(f"{out}/runs/{t[0]}/{t[1]}.json")]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} of {len(T)} tasks have no result, e.g. {missing[:5]}")
    F, counts = [], collections.Counter()
    for ds, s in T:
        r = json.load(open(f"{out}/runs/{ds}/{s}.json"))
        for k, v in r["counts"].items():
            counts[k] += v
        F += r["fits"]
    # exclusions and G0, by (model, arm, protocol)
    excl = collections.Counter((f["model"], f["arm"], f["prot"], f["excluded"]) for f in F)
    tot = collections.Counter((f["model"], f["arm"], f["prot"]) for f in F)
    refused = {k for k in tot if sum(n for (m, a, p, e), n in excl.items() if (m, a, p) == k and e == "g0") > G0_MAX_FAIL * tot[k]}
    ok = collections.defaultdict(list)
    for f in F:
        if f["excluded"] is None:
            ok[(f["model"], f["arm"], f["prot"])].append(f)
    R, L = {"counts": dict(counts), "refused_g0": sorted(map(list, refused)),
            "exclusions": {"|".join(map(str, k)): n for k, n in sorted(excl.items(), key=str)}}, []
    L.append("# E1b — ablation des FFN au test (campagne A, poids finaux)\n")
    L.append(f"Tâches : {len(T)} sujets, toutes présentes. Fits lus : {len(F)}. Compteurs : {dict(counts)}.\n")
    L.append("## Exclusions par (modèle, bras, protocole)\n")
    for k in sorted(tot):
        ex = {e: n for (m, a, p, e), n in excl.items() if (m, a, p) == k and e is not None}
        L.append(f"- {' / '.join(k)} : {tot[k]} fits, exclus {ex or 0}{' — **REFUSÉ (G0)**' if k in refused else ''}")
    L.append("")
    # primary
    L.append(f"## Endpoints primaires ({PRIMARY}, bras bd ; Holm à {ALPHA} unilatéral sur les 3 protocoles)\n")
    prim = {}
    for h, (a, b, kind) in {"H1": ("m_deep", "full", "ni"), "H2": ("m_deep", "m0", "sup"),
                            "H3": ("p_tiny", "p_unif", "sup")}.items():
        cs = {}
        for p in PROTS:
            k = (PRIMARY, "bd", p)
            cs[p] = None if k in refused else sboot(contrast(ok[k], a, b))
        ps = [(1.0 if cs[p] is None else cs[p]["p_ni" if kind == "ni" else "p_sup"]) for p in PROTS]
        rej = holm(ps)
        prim[h] = {}
        for p, rj in zip(PROTS, rej):
            c = cs[p]
            if c is None:
                v = "REFUSÉ"
            elif rj:
                v = "NON INFÉRIEUR" if kind == "ni" else "SUPÉRIEUR"
            elif c["hi"] < 0 and (kind == "sup" or c["lo"] <= NI):
                v = "INFÉRIEUR"
            else:
                v = "NON CONCLUANT"
            prim[h][p] = {"contrast": f"{a} − {b}", "ci": c, "verdict": v}
            L.append(f"- **{h}** {p} : {a} − {b} = {ci(c)} → **{v}**")
        L.append("")
    R["primary"] = prim
    # secondaries
    L.append("## Secondaires (ne décident rien)\n")
    sec = {}
    for m in MODELS:
        B = N_BLOCKS[m]
        for arm in ARMS:
            for p in PROTS:
                k = (m, arm, p)
                if k in refused or not ok[k]:
                    continue
                row = {f"m{j}": sboot(contrast(ok[k], f"m{j}", "full")) for j in range(B)}
                for t in ("m_deep", "m_all", "z_deep", "z_all"):
                    row[t] = sboot(contrast(ok[k], t, "full"))
                row["z_deep-z0"] = sboot(contrast(ok[k], "z_deep", "z0"))
                row["m_deep-m0"] = sboot(contrast(ok[k], "m_deep", "m0"))
                if arm == "bd":
                    row["p_tiny-full"] = sboot(contrast(ok[k], "p_tiny", "full"))
                    row["p_unif-full"] = sboot(contrast(ok[k], "p_unif", "full"))
                    row["p_tiny-p_unif"] = sboot(contrast(ok[k], "p_tiny", "p_unif"))
                    row["rho_cost_alloc"] = sboot(spearman_cost_alloc(ok[k]))
                sec["|".join(k)] = row
                L.append(f"### {m} / {arm} / {p}\n")
                for t, c in row.items():
                    L.append(f"- {t}{' − full' if t[0] in 'mz' and '-' not in t else ''} : {ci(c)}")
                L.append("")
    R["secondary"] = sec
    os.makedirs(f"{out}", exist_ok=True)
    json.dump(R, open(f"{out}/E1b.json", "w"), indent=1)
    open(f"{out}/E1b.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "tasks", "run", "read"))
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks")
    ap.add_argument("--line", type=int)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest(a.cells, a.out)
    elif a.cmd == "tasks":
        tasks(a.cells, a.out)
    elif a.cmd == "run":
        ds, s = open(a.tasks).read().split("\n")[a.line].split()
        if os.path.exists(f"{a.out}/runs/{ds}/{s}.json"):
            print("done already", ds, s)
            return
        run_subject(a.cells, ds, s, a.out)
    else:
        read(a.cells, a.tasks, a.out)


if __name__ == "__main__":
    main()
