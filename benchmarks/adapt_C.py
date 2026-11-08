"""Reading 4: a size that adapts to the subject (axis 1, PREREG_growth_adaptive_size.md)
and the subject's signature in the adapter (axis 2, PREREG_growth_signature.md).
Committed with both PREREGs BEFORE any adaptation outside the tuning dataset.

Same shared models, junctions, data, splits, draws and training as reading 3 (adapt_B.py,
imported, not copied). What is new:

- budgets b in LEVELS for both families (grow<b> = the neurons closest to LoRA rank b,
  as reading 3; lora<b> = rank b); b = 0 is AdaBN;
- for each draw, the size is chosen ON THE CALIBRATION TRIALS ONLY by a 2-fold inner
  cross-validation over {0} + LEVELS, separately for growth and LoRA;
- adapters of the 2-session datasets are saved, and the signature pass applies subject
  j's adapter to subject i's other session, with i's own AdaBN.

    python adapt_C.py tasks --cells CELLS --out DIR
    python adapt_C.py tune  --cells CELLS --model M --subject S --out DIR
    python adapt_C.py merge --out DIR
    python adapt_C.py run   --cells CELLS --tasks FILE --line L --out DIR
    python adapt_C.py sig   --cells CELLS --tasks FILE --line L --out DIR
    python adapt_C.py read  --out DIR --adapt-b FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import adapt_A
import adapt_B
from adapt_A import G0_MAX_FAIL, NI, STEPS
from adapt_B import MAIN_DS, TUNE_DS, ci, sboot

# CTNet is out: its shared models' transformer output is constant across trials (exploratory
# probe of 08/11, JOURNAL), so any FFN adapter only adds a bias and cannot move the AUC
MODELS = ("shallow", "eegnex", "conformer", "msvtnet")
LEVELS = (1, 2, 4, 8, 16)
KS = (10, 50)
FAMILIES = ("grow", "lora")
LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
TRAINED = tuple(f"{f}{b}" for f in FAMILIES for b in LEVELS)
TWO_SESSION = ("bnci2014_004", "bnci2015_001", "lee2019_mi", "shin2017a", "zhou2016")
SAVED = ("grow2", "grow8", "lora2", "lora8")      # adapters kept for the signature pass
N_PARTNERS = 4
ALPHA = 0.025                                      # one-sided, Holm
SIG_TOL = 5e-3                                     # own adapter re-evaluated vs run score


def holm(ps, alpha=ALPHA):
    order = np.argsort(ps)
    rej, m = [False] * len(ps), len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def verdict(c, sup):
    return adapt_A.verdict(c, sup)


# ---------------------------------------------------------------- adaptation
def adapt(sd, name, arm, lr, Xc, yc, Xt, seed, shape, dev, keep=False):
    """adapt_B.adapt, and the trained adapter's tensors when ``keep``."""
    import torch
    if not keep or arm[:4] not in ("grow", "lora"):
        lg, n = adapt_B.adapt(sd, name, arm, lr, Xc, yc, Xt, seed, shape, dev)
        return lg, n, None
    box = {}
    orig = adapt_B.remove

    def grab(delta):
        box["params"] = [q.detach().cpu().clone() for q in delta.params]
        orig(delta)
    adapt_B.remove = grab
    try:
        lg, n = adapt_B.adapt(sd, name, arm, lr, Xc, yc, Xt, seed, shape, dev)
    finally:
        adapt_B.remove = orig
    return lg, n, box["params"]


def apply_adapter(sd, name, arm, params, Xc, Xt, seed, shape, dev):
    """Logits on Xt of the shared model + an adapter with the given tensors, after AdaBN on
    Xc (the last two steps of adapt_B.adapt, with the trained tensors swapped in)."""
    model = adapt_B.base_model(name, sd, *shape).to(dev)
    delta = adapt_B.make_delta(model, name, arm[:4], int(arm[4:]), seed)
    for q, v in zip(delta.params, params):
        q.data = v.to(dev).clone()
    for bn in delta.bns:
        bn.to(dev)
    adapt_B.adabn(model, Xc, delta.bns, delta.params)
    out = adapt_B.logits_of(model, Xt)
    adapt_B.remove(delta)
    return out


def inner_folds(yc, seed):
    """Two disjoint halves of a calibration draw, balanced per class (floor/ceil)."""
    rng = np.random.default_rng(seed)
    a = np.zeros(len(yc), bool)
    for c in np.unique(yc):
        idx = rng.permutation(np.flatnonzero(yc == c))
        a[idx[:len(idx) // 2]] = True
    return [(a, ~a), (~a, a)]


def choose(val):
    """Level with the best mean inner score; ties -> the smaller level. ``val``: {b: score}."""
    best = max(val.values())
    return min(b for b, v in val.items() if v >= best - 1e-12)


def run_subject(ds, name, cells_path, subj, ks, lrs, save_dir=None):
    """Per draw: outer score of every arm, inner-CV score of every (family, level)."""
    import torch
    adapt_B._fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    dev = adapt_B.device()
    X, y, sess, run = adapt_B.subject_data(ds, subj, cells_path)
    sd, saved = adapt_B.shared_fit(cells_path, ds, name, subj)
    nc = len(np.unique(y))
    shape = (X.shape[1], nc, X.shape[2])
    gate = adapt_B.g0(adapt_B.base_model(name, sd, *shape).to(dev), saved, X, y, sess, dev, nc)
    rule, dirs, n_draws = adapt_B.splits(sess, run, y)
    out = {"subject": str(subj), "dataset": ds, "model": name, "g0": gate, "rule": rule,
           "n_classes": nc, "outer": {}, "inner": {}, "n_params": {}, "k_missing": [], "refused": None}
    if any(len(np.unique(y[t])) < nc for _, _, t in dirs):
        out["refused"] = "a test part lacks a class"
        return out
    ok_k = [k for k in ks if all(np.bincount(y[c], minlength=nc).min() >= k for _, c, _ in dirs)]
    out["k_missing"] = [k for k in ks if k not in ok_k]
    keep = save_dir is not None and ds in TWO_SESSION
    for d, cal, test in dirs:
        Xc_all, yc_all = X[cal], y[cal]
        Xt, yt = torch.as_tensor(X[test], device=dev), y[test]
        for k in ok_k:
            for r, idx in enumerate(adapt_B.draws(yc_all, k, subj, d, n_draws)):
                Xc, yc = torch.as_tensor(Xc_all[idx], device=dev), yc_all[idx]
                seed = 1000 * int(subj) + 10 * d + r
                key = f"{k}/{d}/{r}"
                for arm in ("zero", "adabn") + TRAINED:
                    lg, n, par = adapt(sd, name, arm, lrs.get(arm, 0.0), Xc, yc, Xt, seed, shape, dev,
                                       keep=keep and arm in SAVED)
                    out["outer"].setdefault(arm, {})[key] = adapt_A.score(lg, yt, nc)
                    out["n_params"][arm] = n
                    if par is not None:
                        f = f"{save_dir}/{name}/{ds}/{subj}/{arm}_{k}_{d}_{r}.pt"
                        os.makedirs(os.path.dirname(f), exist_ok=True)
                        torch.save(par, f)
                # inner CV: train on one half, score on the other, both ways
                for fold, (tr, va) in enumerate(inner_folds(yc, seed)):
                    Xa, ya, Xv, yv = Xc[torch.as_tensor(tr)], yc[tr], Xc[torch.as_tensor(va)], yc[va]
                    for arm in ("adabn",) + TRAINED:
                        lg, _, _ = adapt(sd, name, arm, lrs.get(arm, 0.0), Xa, ya, Xv, seed, shape, dev)
                        out["inner"].setdefault(arm, {}).setdefault(key, []).append(adapt_A.score(lg, yv, nc))
    return out


def partners(subs, n=N_PARTNERS):
    """Subject -> the next n subjects in a fixed random cyclic order (never itself)."""
    order = [str(s) for s in np.random.default_rng(0).permutation(sorted(subs, key=int))]
    m = len(order)
    return {s: [order[(i + j) % m] for j in range(1, min(n, m - 1) + 1)] for i, s in enumerate(order)}


def sig_subject(ds, name, cells_path, subj, save_dir, run_json, parts):
    """Own and partners' adapters on subject i's test part, with i's AdaBN."""
    import torch
    adapt_B._fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    dev = adapt_B.device()
    X, y, sess, run = adapt_B.subject_data(ds, subj, cells_path)
    sd, _ = adapt_B.shared_fit(cells_path, ds, name, subj)
    nc = len(np.unique(y))
    shape = (X.shape[1], nc, X.shape[2])
    rule, dirs, n_draws = adapt_B.splits(sess, run, y)
    R = json.load(open(run_json))
    out = {"subject": str(subj), "dataset": ds, "model": name, "partners": parts, "own": {}, "other": {},
           "own_run": {}}
    for d, cal, test in dirs:
        Xc_all, yc_all = X[cal], y[cal]
        Xt, yt = torch.as_tensor(X[test], device=dev), y[test]
        for k in KS:
            if k in R["k_missing"]:
                continue
            for r, idx in enumerate(adapt_B.draws(yc_all, k, subj, d, n_draws)):
                Xc = torch.as_tensor(Xc_all[idx], device=dev)
                seed = 1000 * int(subj) + 10 * d + r
                key = f"{k}/{d}/{r}"
                for arm in SAVED:
                    own = torch.load(f"{save_dir}/{name}/{ds}/{subj}/{arm}_{k}_{d}_{r}.pt")
                    lg = apply_adapter(sd, name, arm, own, Xc, Xt, seed, shape, dev)
                    out["own"].setdefault(arm, {})[key] = adapt_A.score(lg, yt, nc)
                    out["own_run"].setdefault(arm, {})[key] = R["outer"][arm][key]
                    for j in parts:
                        f = f"{save_dir}/{name}/{ds}/{j}/{arm}_{k}_{d}_{r}.pt"
                        if not os.path.exists(f):
                            continue
                        lg = apply_adapter(sd, name, arm, torch.load(f), Xc, Xt, seed, shape, dev)
                        out["other"].setdefault(arm, {}).setdefault(key, {})[j] = adapt_A.score(lg, yt, nc)
    return out


# ---------------------------------------------------------------- reading helpers
def subject_values(r, fix, perm_r=None):
    """Per-draw quantities of one subject (dict key -> value) for the axis-1 contrasts.
    ``fix``: {family: level}; ``perm_r``: the partner's run (its inner choices)."""
    out = {}
    for key in r["outer"]["adabn"]:
        sc = {0: r["outer"]["adabn"][key]}
        row = {}
        for f in FAMILIES:
            s = {**sc, **{b: r["outer"][f"{f}{b}"][key] for b in LEVELS}}
            val = {0: np.mean(r["inner"]["adabn"][key]),
                   **{b: np.mean(r["inner"][f"{f}{b}"][key]) for b in LEVELS}}
            b = choose(val)
            row[f"{f}_cv"] = s[b]
            row[f"{f}_level"] = b
            row[f"{f}_fix"] = s[fix[f]]
            row[f"{f}_oracle"] = max(s.values())
            if perm_r is not None and key in perm_r["inner"]["adabn"]:
                pv = {0: np.mean(perm_r["inner"]["adabn"][key]),
                      **{bb: np.mean(perm_r["inner"][f"{f}{bb}"][key]) for bb in LEVELS}}
                row[f"{f}_perm"] = s[choose(pv)]
        out[key] = row
    return out


def read(out, adapt_b_json):
    tune = json.load(open(f"{out}/tune.json"))
    fix = tune["fix"]
    b_ok = json.load(open(adapt_b_json))["ok_models"]          # reading 3's G1, frozen
    models = [m for m in MODELS if m in b_ok]
    R = adapt_B.load_runs(out)
    excl, refused_cells, V = {}, [], {}
    for (m, ds), subs in sorted(R.items()):
        failed = [s for s, r in subs.items() if not all(g["ok"] for g in r["g0"].values())]
        if len(failed) > G0_MAX_FAIL:
            refused_cells.append((m, ds, len(failed)))
            continue
        good = {s: r for s, r in subs.items() if s not in failed and not r["refused"]}
        for s in subs:
            if s not in good:
                excl.setdefault(f"{m}/{ds}", []).append(s)
        pmap = {s: (p[0] if p else None) for s, p in partners(good, 1).items()}
        for s, r in good.items():
            V[(m, ds, s)] = subject_values(r, fix[m], good.get(pmap[s]))

    def per_subject(m, ds, k, fn):
        vals = {}
        for (mm, dd, s), rows in V.items():
            if (mm, dd) != (m, ds):
                continue
            xs = [fn(row) for key, row in rows.items() if int(key.split("/")[0]) == k]
            xs = [x for x in xs if x is not None]
            if xs:
                vals[s] = float(np.mean(xs))
        return vals

    def pooled(ms, k, fn):
        groups = {}
        for ds in MAIN_DS:
            per = [per_subject(m, ds, k, fn) for m in ms]
            common = set.intersection(*[set(p) for p in per]) if per else set()
            groups[ds] = [np.mean([p[s] for p in per]) for s in sorted(common)]
        return sboot(groups)

    diff = lambda a, b: (lambda row: row[a] - row[b] if a in row and b in row else None)  # noqa: E731
    crit = {"S1 taille du sujet − taille d'un autre (croissance)": diff("grow_cv", "grow_perm"),
            "S2 croissance − LoRA, tailles choisies": diff("grow_cv", "lora_cv"),
            "S3 croissance : taille choisie − meilleure taille fixe": diff("grow_cv", "grow_fix")}
    L = ["# Une taille qui s'adapte au sujet — 4 modèles × 11 datasets (PREREG_growth_adaptive_size.md)", "",
         f"- modèles retenus (G1 de la lecture 3, CTNet exclu) : {models}",
         f"- taux d'apprentissage (réglés sur {TUNE_DS}) : {json.dumps(tune['lr'])}",
         f"- meilleure taille fixe (réglée sur {TUNE_DS}, k = 10) : {json.dumps(fix)}",
         f"- cellules refusées par G0 : {refused_cells}",
         f"- sujets exclus : { {k: len(v) for k, v in excl.items()} }", ""]
    P = {(n, k): pooled(models, k, fn) for k in KS for n, fn in crit.items()}
    for (key, c), rj in zip(P.items(), holm([c["p_sup"] for c in P.values()])):
        c["verdict"] = verdict(c, rj)
    L += [f"## Critère principal (groupé, Holm sur 6, α = {ALPHA} unilatéral, marge NI {NI})", "",
          "| critère | k | écart [IC95] | p sup. | n sujets / datasets | MDE | verdict |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {n} | {k} | {ci(c)} | {c['p_sup']:.4f} | {c['n']} / {c['n_ds']} | {c['mde']:.3f} | **{c['verdict']}** |"
          for (n, k), c in P.items()]
    S = {(m, n, k): pooled([m], k, fn) for m in models for k in KS for n, fn in crit.items()}
    for c, rj in zip(S.values(), holm([c["p_sup"] for c in S.values()])):
        c["verdict"] = verdict(c, rj)
    L += ["", f"## Secondaire — par modèle (Holm sur {len(S)})", "",
          "| modèle | k | " + " | ".join(crit) + " |", "|---|---|" + "---|" * len(crit)]
    for m in models:
        for k in KS:
            L.append(f"| {m} | {k} | " + " | ".join(f"{ci(S[(m, n, k)])} **{S[(m, n, k)]['verdict']}**"
                                                 for n in crit) + " |")
    rep = {"lora_cv − lora_perm": diff("lora_cv", "lora_perm"), "lora_cv − lora_fix": diff("lora_cv", "lora_fix"),
           "grow_oracle − grow_fix": diff("grow_oracle", "grow_fix"),
           "lora_oracle − lora_fix": diff("lora_oracle", "lora_fix"),
           "grow_fix − lora_fix": diff("grow_fix", "lora_fix")}
    L += ["", "## Rapporté (sans valeur décisionnelle)", "", "| contraste | " + " | ".join(f"k={k}" for k in KS) + " |",
          "|---|" + "---|" * len(KS)]
    REP = {n: {k: pooled(models, k, fn) for k in KS} for n, fn in rep.items()}
    L += [f"| {n} | " + " | ".join(ci(c) if c["n"] else "—" for c in row.values()) + " |" for n, row in REP.items()]
    L += ["", "### Tailles choisies (part des tirages, toutes cellules)", "",
          "| modèle | k | famille | " + " | ".join(f"b={b}" for b in (0,) + LEVELS) + " |",
          "|---|---|---|" + "---|" * (len(LEVELS) + 1)]
    lv = {}
    for m in models:
        for k in KS:
            for f in FAMILIES:
                xs = [row[f"{f}_level"] for (mm, _, _), rows in V.items() if mm == m
                      for key, row in rows.items() if int(key.split("/")[0]) == k]
                if xs:
                    h = [float(np.mean(np.array(xs) == b)) for b in (0,) + LEVELS]
                    lv[f"{m} {k} {f}"] = h
                    L.append(f"| {m} | {k} | {f} | " + " | ".join(f"{v:.2f}" for v in h) + " |")
    json.dump({"models": models, "fix": fix, "lr": tune["lr"], "refused": refused_cells, "excluded": excl,
               "primary": {f"{n} k={k}": c for (n, k), c in P.items()},
               "secondary": {f"{m} {n} k={k}": c for (m, n, k), c in S.items()},
               "reported": {n: {str(k): c for k, c in row.items()} for n, row in REP.items()},
               "levels": lv}, open(f"{out}/adapt_C.json", "w"), indent=1, default=str)
    open(f"{out}/adapt_C.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    read_sig(out, models, excl)


def read_sig(out, models, excl):
    """Axis 2: own − other adapter on the other session, with the subject's own AdaBN."""
    G = {}
    bad, tot = 0, 0
    for f in glob.glob(f"{out}/sig/*/*/*.json"):
        r = json.load(open(f))
        m, ds, s = r["model"], r["dataset"], r["subject"]
        if m not in models or s in excl.get(f"{m}/{ds}", []):
            continue
        for arm, by in r["own"].items():
            for key, v in by.items():
                tot += 1
                bad += abs(v - r["own_run"][arm][key]) > SIG_TOL
                oth = r["other"].get(arm, {}).get(key, {})
                if oth:
                    G.setdefault((m, ds, s), {}).setdefault(arm, {})[key] = (v, float(np.mean(list(oth.values()))))
    L = ["# La signature du sujet dans l'adaptateur — 4 modèles × 5 datasets à 2 sessions "
         "(PREREG_growth_signature.md)", "",
         f"- adaptateurs ré-évalués : {tot}, écart au score du passage d'adaptation > {SIG_TOL} : {bad}"]
    gate = tot > 0 and bad <= 0.05 * tot
    L.append(f"- garde-fou G-sig (≤ 5 % d'écarts) : **{'PASSE' if gate else 'REFUSE'}**")
    if not gate:
        open(f"{out}/signature_C.md", "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        return

    def per_subject(m, ds, k, fn):
        vals = {}
        for (mm, dd, s), arms in G.items():
            if (mm, dd) != (m, ds):
                continue
            xs = [fn(arms, key) for key in next(iter(arms.values())) if int(key.split("/")[0]) == k]
            xs = [x for x in xs if x is not None]
            if xs:
                vals[s] = float(np.mean(xs))
        return vals

    def pooled(k, fn, ms=models):
        groups = {}
        for ds in TWO_SESSION:
            per = [per_subject(m, ds, k, fn) for m in ms]
            common = set.intersection(*[set(p) for p in per]) if per else set()
            groups[ds] = [np.mean([p[s] for p in per]) for s in sorted(common)]
        return sboot(groups)

    def t1(arm):
        return lambda arms, key: (arms[arm][key][0] - arms[arm][key][1]) if key in arms.get(arm, {}) else None

    def t1diff(a, b):
        return lambda arms, key: ((arms[a][key][0] - arms[a][key][1]) - (arms[b][key][0] - arms[b][key][1])
                                  if key in arms.get(a, {}) and key in arms.get(b, {}) else None)
    crit = {"T1 croissance (b=8) : son adaptateur − celui d'un autre": t1("grow8"),
            "T2 T1 croissance − T1 LoRA (b=8)": t1diff("grow8", "lora8")}
    P = {(n, k): pooled(k, fn) for k in KS for n, fn in crit.items()}
    for c, rj in zip(P.values(), holm([c["p_sup"] for c in P.values()])):
        c["verdict"] = "SUPÉRIEUR" if rj else ("INFÉRIEUR" if c["hi"] < 0 else "NON SIGNIFICATIF")
    L += ["", f"## Critère principal (groupé sur les datasets à 2 sessions, Holm sur 4, α = {ALPHA})", "",
          "| critère | k | écart [IC95] | p sup. | n sujets / datasets | MDE | verdict |", "|---|---|---|---|---|---|---|"]
    L += [f"| {n} | {k} | {ci(c)} | {c['p_sup']:.4f} | {c['n']} / {c['n_ds']} | {c['mde']:.3f} | **{c['verdict']}** |"
          for (n, k), c in P.items()]
    rep = {"T1 croissance b=2": t1("grow2"), "T1 LoRA b=8": t1("lora8"), "T1 LoRA b=2": t1("lora2"),
           "T1 croissance − LoRA, b=2": t1diff("grow2", "lora2")}
    L += ["", "## Rapporté", "", "| contraste | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---|" * len(KS)]
    REP = {n: {k: pooled(k, fn) for k in KS} for n, fn in rep.items()}
    L += [f"| {n} | " + " | ".join(ci(c) if c["n"] else "—" for c in row.values()) + " |" for n, row in REP.items()]
    L += ["", "### T1 croissance b=8 par modèle", "", "| modèle | " + " | ".join(f"k={k}" for k in KS) + " |",
          "|---|" + "---|" * len(KS)]
    PM = {m: {k: pooled(k, t1("grow8"), [m]) for k in KS} for m in models}
    L += [f"| {m} | " + " | ".join(ci(c) if c["n"] else "—" for c in row.values()) + " |" for m, row in PM.items()]
    json.dump({"gate": {"total": tot, "off": bad, "ok": gate},
               "primary": {f"{n} k={k}": c for (n, k), c in P.items()},
               "reported": {n: {str(k): c for k, c in row.items()} for n, row in REP.items()},
               "per_model": {m: {str(k): c for k, c in row.items()} for m, row in PM.items()}},
              open(f"{out}/signature_C.json", "w"), indent=1, default=str)
    open(f"{out}/signature_C.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- tasks and main
PER_LINE = {"shallow": 4, "eegnex": 1, "conformer": 2, "msvtnet": 2}


def write_tasks(cells_path, out):
    from campaignA_check import parse_cells
    subs = {}
    for c in parse_cells(cells_path):
        if c["prot"] == "cross_subject" and c["arm"] == "bd" and c["model"] in MODELS:
            subs.setdefault((c["model"], c["ds"]), set()).add(c["subj"])
    tune = [f"{m} {s}" for m in MODELS for s in sorted(subs[(m, TUNE_DS)], key=int)]
    run, sig = [], []
    for m in MODELS:
        for ds in MAIN_DS:
            ss = sorted(subs[(m, ds)], key=int)
            for i in range(0, len(ss), PER_LINE[m]):
                run.append(",".join(f"{m} {ds} {s}" for s in ss[i:i + PER_LINE[m]]))
            if ds in TWO_SESSION:
                for i in range(0, len(ss), 6):
                    sig.append(",".join(f"{m} {ds} {s}" for s in ss[i:i + 6]))
    for n, rows in (("tune", tune), ("run", run), ("sig", sig)):
        open(f"{out}/{n}_tasks.txt", "w").write("\n".join(rows) + "\n")
    print(len(tune), "tune,", len(run), "run,", len(sig), "sig lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tasks", "tune", "merge", "run", "sig", "read"])
    ap.add_argument("--cells")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--subject")
    ap.add_argument("--tasks")
    ap.add_argument("--line", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapt-b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "tasks":
        write_tasks(a.cells, a.out)
    elif a.mode == "tune":
        f = f"{a.out}/tune/{a.model}/{a.subject}.json"
        if not os.path.exists(f):
            res = {}
            for lr in LRS:
                r = run_subject_tune(a.model, a.cells, a.subject, lr)
                res[str(lr)] = r
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump(res, open(f + ".tmp", "w"), indent=1)
            os.replace(f + ".tmp", f)
    elif a.mode == "merge":
        merge(a.out)
    elif a.mode in ("run", "sig"):
        line = open(a.tasks).read().split("\n")[a.line - 1]
        lrs = json.load(open(f"{a.out}/tune.json"))["lr"]
        for item in line.split(","):
            m, ds, s = item.split()
            f = f"{a.out}/{'runs' if a.mode == 'run' else 'sig'}/{m}/{ds}/{s}.json"
            if os.path.exists(f):
                continue
            if a.mode == "run":
                r = run_subject(ds, m, a.cells, s, KS, lrs[m], save_dir=f"{a.out}/adapters")
            else:
                subs = [os.path.basename(p)[:-5] for p in glob.glob(f"{a.out}/runs/{m}/{ds}/*.json")]
                r = sig_subject(ds, m, a.cells, s, f"{a.out}/adapters", f"{a.out}/runs/{m}/{ds}/{s}.json",
                                partners(subs)[s])
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump(r, open(f + ".tmp", "w"), indent=1)
            os.replace(f + ".tmp", f)
            print(f"done {a.mode} {m} {ds} {s}", flush=True)
    else:
        read(a.out, a.adapt_b)


def run_subject_tune(name, cells_path, subj, lr):
    """Tuning dataset, k = 10, every trained arm at one lr: mean outer score over draws."""
    lrs = {arm: lr for arm in TRAINED}
    r = run_subject_outer_only(TUNE_DS, name, cells_path, subj, lrs)
    return r


def run_subject_outer_only(ds, name, cells_path, subj, lrs):
    import torch
    adapt_B._fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    dev = adapt_B.device()
    X, y, sess, run = adapt_B.subject_data(ds, subj, cells_path)
    sd, saved = adapt_B.shared_fit(cells_path, ds, name, subj)
    nc = len(np.unique(y))
    shape = (X.shape[1], nc, X.shape[2])
    gate = adapt_B.g0(adapt_B.base_model(name, sd, *shape).to(dev), saved, X, y, sess, dev, nc)
    rule, dirs, n_draws = adapt_B.splits(sess, run, y)
    sc = {}
    for d, cal, test in dirs:
        Xt, yt = torch.as_tensor(X[test], device=dev), y[test]
        for r_, idx in enumerate(adapt_B.draws(y[cal], 10, subj, d, n_draws)):
            Xc, yc = torch.as_tensor(X[cal][idx], device=dev), y[cal][idx]
            seed = 1000 * int(subj) + 10 * d + r_
            for arm in ("adabn",) + TRAINED:
                lg, _ = adapt_B.adapt(sd, name, arm, lrs.get(arm, 0.0), Xc, yc, Xt, seed, shape, dev)
                sc.setdefault(arm, []).append(adapt_A.score(lg, yt, nc))
    return {"g0": gate, "scores": {a: float(np.mean(v)) for a, v in sc.items()}}


def merge(out):
    """lr per (model, arm) = best mean score over the 9 tuning subjects; best fixed level per
    (model, family) = the level whose tuned score is highest (ties -> smaller)."""
    best, fix, curve = {}, {}, {}
    for m in MODELS:
        rs = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/tune/{m}/*.json"))]
        bad = [i for i, r in enumerate(rs) if not all(g["ok"] for g in r[str(LRS[0])]["g0"].values())]
        if len(rs) != 9 or bad:
            raise SystemExit(f"REFUSED: {m}: {len(rs)} tuning subjects, G0 fails on {bad}")
        curve[m] = {arm: {str(lr): float(np.mean([r[str(lr)]["scores"][arm] for r in rs])) for lr in LRS}
                    for arm in TRAINED}
        best[m] = {arm: float(max(c, key=c.get)) for arm, c in curve[m].items()}
        fix[m] = {}
        for f in FAMILIES:
            top = {b: max(curve[m][f"{f}{b}"].values()) for b in LEVELS}
            fix[m][f] = choose(top)
    json.dump({"lr": best, "fix": fix, "curve": curve}, open(f"{out}/tune.json", "w"), indent=1)
    print(json.dumps({"lr": best, "fix": fix}))


if __name__ == "__main__":
    main()
