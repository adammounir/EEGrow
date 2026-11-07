"""Adapt a shared Shallow to a new subject by adding neurons, against LoRA, the head and
AdaBN (analysis/PREREG_growth_adaptation_A.md). Committed with the PREREG BEFORE any
adaptation on lee2019_mi.

The shared model of subject i is campaign A's ``shallow__bd`` cross_subject fit that left
i out. Calibrate on k trials per class of one session, test on the whole other session,
both directions, 3 draws.

    python adapt_A.py tune  --cells CELLS --subject S --out DIR   # bnci2014_001, lr grid
    python adapt_A.py merge --out DIR                              # -> DIR/tune.json
    python adapt_A.py run   --cells CELLS --subject S --out DIR   # lee2019_mi, reads tune.json
    python adapt_A.py read  --out DIR                              # gates, criteria, report
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from embed_A import N_BOOT, cache_of

TUNE_DS, MAIN_DS = "bnci2014_001", "lee2019_mi"
KS, K_MAIN, K_POS = (5, 10, 25, 50), 10, 50
PS, P_MAIN = (2, 8), 2
LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
TRAINED = ("head", "lora2", "lora8", "grow2", "grow8", "full")
ARMS = ("zero", "adabn") + TRAINED
STEPS, DRAWS = 100, 3
NI = -0.02
G0_TOL, G0_MAX_FAIL = 1e-3, 2


# ---------------------------------------------------------------- model and arms
def combined(wt, bt, ws):
    """braindecode CombinedConv's merge (modules/layers.py), for any number of temporal
    filters: conv_time (m,1,K,1) + bias (m,) followed by conv_spat (40,m,1,C) as one conv."""
    w = (wt * ws.permute(1, 0, 2, 3)).sum(0).unsqueeze(1)
    b = ws.reshape(ws.shape[0], ws.shape[1], -1).sum(-1) @ bt
    return w, b


class Delta:
    """A trainable additive branch on the output of ``conv_time_spat`` (before the
    BatchNorm). ``grow``: m new temporal filters with their spatial weights, spatial
    weights at 0. ``lora``: ΔW = B A on conv_spat seen as 40 x (40 C), B = 0."""

    def __init__(self, model, kind, p, seed):
        import torch
        cts = model.conv_time_spat
        g = torch.Generator().manual_seed(seed)
        n_spat, n_time, _, C = cts.conv_spat.weight.shape
        K = cts.conv_time.weight.shape[2]
        self.kind, self.cts = kind, cts
        if kind == "grow":
            fresh = torch.nn.Conv2d(1, p, (K, 1))  # PyTorch default init, seeded below
            with torch.no_grad():
                bound = 1 / np.sqrt(K)
                fresh.weight.copy_((torch.rand(p, 1, K, 1, generator=g) * 2 - 1) * bound)
                fresh.bias.copy_((torch.rand(p, generator=g) * 2 - 1) * bound)
            self.wt = torch.nn.Parameter(fresh.weight.detach().clone())
            self.bt = torch.nn.Parameter(fresh.bias.detach().clone())
            self.ws = torch.nn.Parameter(torch.zeros(n_spat, p, 1, C))
            self.params = [self.wt, self.bt, self.ws]
        elif kind == "lora":
            fan_in = n_time * C
            bound = 1 / np.sqrt(fan_in)  # kaiming_uniform(a=sqrt(5)), as peft's lora_A
            self.A = torch.nn.Parameter((torch.rand(p, fan_in, generator=g) * 2 - 1) * bound)
            self.B = torch.nn.Parameter(torch.zeros(n_spat, p))
            self.params = [self.A, self.B]
        else:
            raise ValueError(kind)
        self.handle = cts.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        import torch.nn.functional as F
        x = inp[0]
        if self.kind == "grow":
            w, b = combined(self.wt, self.bt, self.ws)
        else:
            ws = (self.B @ self.A).reshape(self.cts.conv_spat.weight.shape)
            w, b = combined(self.cts.conv_time.weight, self.cts.conv_time.bias, ws)
        return out + F.conv2d(x, w, b)

    def n_params(self):
        return sum(q.numel() for q in self.params)


def base_model(sd, n_chans, n_outputs, n_times):
    from braindecode.models import ShallowFBCSPNet
    m = ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    m.load_state_dict(sd)
    return m.eval()


def adabn(model, X):
    """BatchNorm running stats = the calibration batch's (momentum None, one pass)."""
    import torch
    bn = model.bnorm
    bn.reset_running_stats()
    mom, bn.momentum = bn.momentum, None
    model.eval()
    bn.train()
    with torch.no_grad():
        model(X)
    bn.momentum = mom
    model.eval()


def logits_of(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 50]) for i in range(0, len(X), 50)]).numpy()


def adapt(sd, arm, lr, Xc, yc, Xt, seed, shape):
    """Logits on Xt after adapting the shared model on (Xc, yc) with ``arm``."""
    import torch
    torch.manual_seed(seed)
    model = base_model(sd, *shape)
    if arm == "zero":
        return logits_of(model, Xt), 0
    adabn(model, Xc)
    if arm == "adabn":
        return logits_of(model, Xt), 0
    for q in model.parameters():
        q.requires_grad_(False)
    delta = None
    if arm == "head":
        params = list(model.final_layer.conv_classifier.parameters())
    elif arm == "full":
        params = list(model.parameters())
    else:
        delta = Delta(model, arm[:4], int(arm[4:]), seed)
        params = delta.params
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    yc_t = torch.as_tensor(yc)
    for _ in range(STEPS):
        model.train()
        model.bnorm.eval()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xc), yc_t)
        loss.backward()
        opt.step()
    adabn(model, Xc)
    out = logits_of(model, Xt)
    n = sum(q.numel() for q in params)
    if delta is not None:
        delta.handle.remove()
    return out, n


def score(lg, y, n_classes):
    from campaignA_check import score_of
    return score_of({"logits": lg, "y": y}, n_classes)


# ---------------------------------------------------------------- data
def subject_data(ds, subj, cells_path):
    """X (float32), y (campaign label codes), session per trial, for one subject, from
    the campaign's own paradigm and cache; a cache miss refuses (embed_A.covariances)."""
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
    # MOABB's evaluations encode labels with a LabelEncoder fitted on the strings (sorted)
    y = LabelEncoder().fit(sorted(set(labels))).transform(labels)
    return X.astype(np.float32), y.astype(np.int64), meta["session"].astype(str).to_numpy()


def shared_fit(cells_path, ds, subj):
    """(state_dict, saved logits by session) of the shallow__bd cross_subject fit that
    left ``subj`` out."""
    import torch
    from campaignA_check import parse_cells
    c = [c for c in parse_cells(cells_path)
         if (c["prot"], c["ds"], c["model"], c["arm"], c["subj"]) ==
         ("cross_subject", ds, "shallow", "bd", str(subj))]
    assert len(c) == 1, (ds, subj, len(c))
    stem = c[0]["csv"][:-4]
    fs = glob.glob(f"{stem}__fits__final/*.pt")
    assert len(fs) == 1, fs
    p = torch.load(fs[0], map_location="cpu", weights_only=False)
    assert str(subj) not in [str(s) for s in p["meta"]["train_subjects"]], p["meta"]
    uid = os.path.basename(fs[0])[:-3]
    saved = {}
    for f in glob.glob(f"{stem}__fits__logits/{uid}__g*.npz"):
        z = np.load(f)
        se = set(z["session"].tolist())
        assert len(se) == 1, se
        saved[se.pop()] = {"logits": z["logits"], "y": z["y"]}
    return p["state_dict"], saved


def g0(sd, saved, X, y, sess, shape):
    """Does the rebuilt model give back the campaign's test logits, session by session?"""
    import torch
    model = base_model(sd, *shape)
    out = {}
    for se, z in saved.items():
        sel = sess == se
        lg = logits_of(model, torch.as_tensor(X[sel]))
        ok = (len(lg) == len(z["logits"]) and np.array_equal(y[sel], z["y"]))
        dmax = float(np.abs(lg - z["logits"]).max()) if ok else None
        ok = ok and dmax <= G0_TOL and np.array_equal(lg.argmax(1), z["logits"].argmax(1))
        out[se] = {"ok": bool(ok), "max_abs": dmax, "n": int(sel.sum())}
    return out


def draws(y, sess_cal, k, subj, d):
    """Calibration indices (within the calibration session) for the 3 draws."""
    out = []
    for r in range(DRAWS):
        rng = np.random.default_rng(1000 * int(subj) + 10 * d + r)
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False)
                              for c in np.unique(y)])
        out.append(np.sort(idx))
    return out


def run_subject(ds, cells_path, subj, ks, arms_lrs, n_classes_hint=None):
    """{arm: {lr: {k: score averaged over 2 directions x 3 draws}}}, plus G0 and n_params."""
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    X, y, sess = subject_data(ds, subj, cells_path)
    sd, saved = shared_fit(cells_path, ds, subj)
    shape = (X.shape[1], int(sd["final_layer.conv_classifier.bias"].shape[0]), X.shape[2])
    n_classes = shape[1]
    gate = g0(sd, saved, X, y, sess, shape)
    sessions = sorted(set(sess))
    assert len(sessions) == 2, sessions
    res, nparams = {}, {}
    for d, (cal, test) in enumerate([(sessions[0], sessions[1]), (sessions[1], sessions[0])]):
        Xc_all, yc_all = X[sess == cal], y[sess == cal]
        Xt, yt = torch.as_tensor(X[sess == test]), y[sess == test]
        for k in ks:
            for r, idx in enumerate(draws(yc_all, cal, k, subj, d)):
                Xc, yc = torch.as_tensor(Xc_all[idx]), yc_all[idx]
                seed = 1000 * int(subj) + 10 * d + r
                for arm, lrs in arms_lrs.items():
                    for lr in lrs:
                        lg, n = adapt(sd, arm, lr, Xc, yc, Xt, seed, shape)
                        s = score(lg, yt, n_classes)
                        res.setdefault(arm, {}).setdefault(str(lr), {}).setdefault(str(k), []).append(s)
                        nparams[arm] = n
    avg = {a: {lr: {k: float(np.mean(v)) for k, v in by.items()} for lr, by in r_.items()}
           for a, r_ in res.items()}
    return {"subject": str(subj), "g0": gate, "scores": avg, "n_params": nparams,
            "n_classes": n_classes}


# ---------------------------------------------------------------- reading
def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "p_sup": float((m <= 0).mean()),
            "n": len(d), "mde": float(2.8016 * d.std(ddof=1) / np.sqrt(len(d)))}


def holm(ps):
    order = np.argsort(ps)
    rej, m = [False] * len(ps), len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= 0.05 / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def verdict(c, sup):
    if sup:
        return "SUPÉRIEUR"
    if c["lo"] > NI:
        return "NON INFÉRIEUR"
    if c["hi"] < 0:
        return "INFÉRIEUR"
    return "NON CONCLUANT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "merge", "run", "read"])
    ap.add_argument("--cells")
    ap.add_argument("--subject")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "tune":
        r = run_subject(TUNE_DS, a.cells, a.subject, (K_MAIN,), {arm: LRS for arm in TRAINED})
        os.makedirs(f"{a.out}/tune", exist_ok=True)
        json.dump(r, open(f"{a.out}/tune/{a.subject}.json", "w"), indent=1)
    elif a.mode == "merge":
        rs = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/tune/*.json"))]
        bad = [r["subject"] for r in rs if not all(g["ok"] for g in r["g0"].values())]
        if bad:
            raise SystemExit(f"REFUSED: G0 fails on the tuning dataset for subjects {bad}")
        best, curve = {}, {}
        for arm in TRAINED:
            curve[arm] = {str(lr): float(np.mean([r["scores"][arm][str(lr)][str(K_MAIN)] for r in rs]))
                          for lr in LRS}
            best[arm] = float(max(curve[arm], key=curve[arm].get))
        json.dump({"n_subjects": len(rs), "lr": best, "curve": curve},
                  open(f"{a.out}/tune.json", "w"), indent=1)
        print(json.dumps(best))
    elif a.mode == "run":
        lr = json.load(open(f"{a.out}/tune.json"))["lr"]
        arms_lrs = {"zero": [0.0], "adabn": [0.0], **{arm: [lr[arm]] for arm in TRAINED}}
        r = run_subject(MAIN_DS, a.cells, a.subject, KS, arms_lrs)
        os.makedirs(f"{a.out}/runs", exist_ok=True)
        json.dump(r, open(f"{a.out}/runs/{a.subject}.json", "w"), indent=1)
    else:
        read(a.out)


def read(out):
    tune = json.load(open(f"{out}/tune.json"))
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/runs/*.json"), key=lambda f: int(os.path.basename(f)[:-5]))]
    failed = [r["subject"] for r in rs if not all(g["ok"] for g in r["g0"].values())]
    L = ["# Adapter un modèle partagé en lui ajoutant des neurones — lee2019_mi "
         "(PREREG_growth_adaptation_A.md)", "",
         f"- sujets lus : {len(rs)} ; G0 (reconstruction = logits de la campagne) échoue sur "
         f"{len(failed)} : {failed}",
         f"- taux d'apprentissage choisis sur {TUNE_DS} : {tune['lr']}", ""]
    out_json = {"g0_failed": failed, "lr": tune["lr"]}
    if len(failed) > G0_MAX_FAIL:
        L.append(f"**REFUSÉ : G0 échoue sur plus de {G0_MAX_FAIL} sujets, aucun critère produit.**")
        open(f"{out}/adapt_A.md", "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        raise SystemExit(2)
    rs = [r for r in rs if r["subject"] not in failed]
    n = len(rs)

    def s(arm, k):
        return np.array([next(iter(r["scores"][arm].values()))[str(k)] for r in rs])

    pos = boot(s("full", K_POS) - s("zero", K_POS))
    g1 = pos["lo"] > 0
    crit = {"E1 croissance - LoRA": ("grow2", "lora2"), "E2 croissance - tête": ("grow2", "head"),
            "E3 croissance - AdaBN": ("grow2", "adabn")}
    C = {name: boot(s(x, K_MAIN) - s(y, K_MAIN)) for name, (x, y) in crit.items()}
    rej = holm([c["p_sup"] for c in C.values()])
    for (name, c), r_ in zip(C.items(), rej):
        c["verdict"] = verdict(c, r_) if g1 else "SANS VERDICT (G1 échoue)"
    L += [f"## G1 — contrôle positif (k = {K_POS}) : fine-tuning complet − zéro-shot = "
          f"{pos['mean']:+.3f} [{pos['lo']:+.3f}, {pos['hi']:+.3f}] → **{'PASSE' if g1 else 'ÉCHOUE'}**", "",
          f"## Critères (n = {n}, k = {K_MAIN}, p = {P_MAIN}, Holm sur 3, marge NI {NI})", "",
          "| critère | écart [IC95] | p sup. | MDE | verdict |", "|---|---|---|---|---|"]
    L += [f"| {name} | {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p_sup']:.4f} | "
          f"{c['mde']:.3f} | **{c['verdict']}** |" for name, c in C.items()]
    L += ["", "## Score moyen par bras et par k (rapporté)", "",
          "| bras | params | " + " | ".join(f"k={k}" for k in KS) + " |",
          "|---|---|" + "---|" * len(KS)]
    for arm in ARMS:
        L.append(f"| {arm} | {rs[0]['n_params'].get(arm, 0)} | " +
                 " | ".join(f"{s(arm, k).mean():.3f}" for k in KS) + " |")
    L += ["", "## Contrastes rapportés, par k (IC95)", "", "| contraste | " +
          " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---|" * len(KS)]
    rep = {}
    for x, y in (("grow2", "lora2"), ("grow8", "lora8"), ("grow2", "head"), ("grow8", "head"),
                 ("lora2", "head"), ("grow2", "adabn"), ("adabn", "zero"), ("full", "zero"),
                 ("grow2", "full"), ("grow8", "grow2")):
        row = {k: boot(s(x, k) - s(y, k)) for k in KS}
        rep[f"{x} - {y}"] = row
        L.append(f"| {x} − {y} | " + " | ".join(
            f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" for c in row.values()) + " |")
    out_json.update({"n": n, "G1": pos, "criteria": C, "reported": rep,
                     "mean": {arm: {k: float(s(arm, k).mean()) for k in KS} for arm in ARMS}})
    json.dump(out_json, open(f"{out}/adapt_A.json", "w"), indent=1, default=str)
    open(f"{out}/adapt_A.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
