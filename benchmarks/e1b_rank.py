"""E1b-rank (EXPLORATORY, analysis/EXPLO_E1b_rank.md) -- does removing every transformer
FFN (E1b's z_all) lose the ranking of the trials, or only the scale/shape of the logits?

E1b-bias showed that z_all costs Conformer accuracy on every dataset, while 2-class AUC
barely moves. This script re-runs E1b's ``full`` and ``z_all`` forwards on the same frozen
fits and, per scored group, adds a ranking score that is blind to how the logits are
scaled or offset:

    pw_*    pairwise AUC on logit differences, float64: mean over class pairs j < k of the
            AUC separating the trials of class k from those of class j with the score
            l_k - l_j (Hand & Till 2001's averaging, on logit differences instead of
            probabilities). Unchanged by any per-class constant offset, by any positive
            scaling of all logits and by a per-trial constant (softmax gauge). With 2
            classes it is the AUC of l1 - l0.
    ovr_*   one-vs-rest macro AUC on the float64 log-odds l_k - logsumexp(l_{j != k})
            (not offset-invariant beyond 2 classes, recorded only)
    acc_*   argmax accuracy (same as E1b-bias)

Imports the frozen E1b copy (same rebuild, hook and G0). CPU, read-only, no training.

    python e1b_rank.py selftest --e1b DIR --bias DIR --out OUT
    python e1b_rank.py run      --e1b DIR --bias DIR --line L --out OUT
    python e1b_rank.py read     --e1b DIR --bias DIR --out OUT
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

G1_TOL = 1e-6        # |score - stored score| (E1b nat_*, E1b-bias acc_*)
G2_TOL = 1e-9        # pw is unchanged by a per-class offset and a positive scaling
K_MARGIN = -0.010    # verdict threshold on pw_z - pw_full


def load_e1b(e1b):
    sys.path.insert(0, e1b)
    import e1b_ablation as E
    return E


def col_of(y):
    """Column index of each trial's label, the scorer's convention (sorted labels of the
    group = logit columns); None when a class is missing from the group."""
    classes = np.unique(y)
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[c] for c in y]), len(classes)


def acc(lg, y):
    t, _ = col_of(y)
    return float((lg.argmax(1) == t).mean())


def pairwise_auc(lg, y):
    from sklearn.metrics import roc_auc_score
    t, k = col_of(y)
    if k != lg.shape[1]:
        return float("nan")
    lg = lg.astype(np.float64)
    a = []
    for j in range(k):
        for i in range(j + 1, k):
            m = (t == i) | (t == j)
            a.append(roc_auc_score(t[m] == i, lg[m, i] - lg[m, j]))
    return float(np.mean(a))


def ovr_auc(lg, y):
    from scipy.special import logsumexp
    from sklearn.metrics import roc_auc_score
    t, k = col_of(y)
    if k != lg.shape[1]:
        return float("nan")
    lg = lg.astype(np.float64)
    if k == 2:
        return float(roc_auc_score(t == 1, lg[:, 1] - lg[:, 0]))
    a = []
    for i in range(k):
        s = lg[:, i] - logsumexp(np.delete(lg, i, 1), axis=1)
        a.append(roc_auc_score(t == i, s))
    return float(np.mean(a))


def g2_ok(lg, y, rng):
    """The premise: pw is blind to a per-class offset and a positive scaling."""
    p = pairwise_auc(lg, y)
    if not np.isfinite(p):
        return True
    q = pairwise_auc(2.5 * lg.astype(np.float64) + rng.normal(0, 3, lg.shape[1]), y)
    return abs(p - q) <= G2_TOL


def run_subject(E, e1b, bias, ds, subj, out, quiet=False):
    import torch
    import adapt_B
    adapt_B._fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rng = np.random.default_rng(0)
    cells_path = os.path.join(e1b, "grids/A/campaignA_cells_planned_av5.txt")
    ref = {f["uid"]: f for f in json.load(open(f"{e1b}/out/runs/{ds}/{subj}.json"))["fits"]}
    refb = {f["uid"]: f for f in json.load(open(f"{bias}/out/runs/{ds}/{subj}.json"))["fits"]}
    counts = collections.Counter()
    t0 = time.time()
    X, y, sess, _ = adapt_B.subject_data(ds, subj, cells_path)
    t_data = time.time() - t0
    records = []
    for c in E.subject_cells(cells_path, ds, subj):
        for uid, pt, npzs in E.fit_files(c, counts):
            tf = time.time()
            r, rb = ref.get(uid), refb.get(uid)
            rec = {"model": c["model"], "arm": c["arm"], "prot": c["prot"], "ds": ds, "subj": str(subj),
                   "uid": uid, "excluded": None, "groups": []}
            records.append(rec)
            if r is None or rb is None:
                rec["excluded"] = "not_in_e1b"
                continue
            if r["excluded"] is not None:
                rec["excluded"] = "e1b:" + r["excluded"]
                continue
            p = torch.load(pt, map_location="cpu", weights_only=False)
            G = E.groups_of(c, npzs, X, y, sess, subj)
            if G is None or len(G) != len(r["groups"]) or len(G) != len(rb["groups"]):
                rec["excluded"] = "test_set_mismatch"
                continue
            n_classes = int(G[0][3].shape[1])
            rec["n_classes"] = n_classes
            model, _ = E.build(c["model"], p["state_dict"], X.shape[1], n_classes, X.shape[2])
            B = E.N_BLOCKS[c["model"]]
            ab = E.Ablator(c["model"], model)
            for (se, Xg, yg, saved, _), rg, rbg in zip(G, r["groups"], rb["groups"]):
                ab.set([None] * B)
                lf = E.logits_of(model, Xg)
                ab.set([("zero",)] * B)
                lz = E.logits_of(model, Xg)
                g = {"session": se, "n": int(len(yg)), "g0": E.g0(lf, saved, yg, n_classes),
                     "acc_full": acc(lf, yg), "acc_z": acc(lz, yg),
                     "pw_full": pairwise_auc(lf, yg), "pw_z": pairwise_auc(lz, yg),
                     "ovr_full": ovr_auc(lf, yg), "ovr_z": ovr_auc(lz, yg),
                     "nat_full": E.score(lf, yg, n_classes), "nat_z": E.score(lz, yg, n_classes)}
                # G1: same scores as the frozen E1b run (nat) and the frozen E1b-bias run (acc)
                g["g1"] = (str(rg["session"]) == str(se) == str(rbg["session"])
                           and abs(g["nat_full"] - rg["scores"]["full"]) <= G1_TOL
                           and abs(g["nat_z"] - rg["scores"]["z_all"]) <= G1_TOL
                           and abs(g["acc_full"] - rbg["acc_full"]) <= G1_TOL
                           and abs(g["acc_z"] - rbg["acc_z"]) <= G1_TOL)
                # G2: pw is blind to offset and scale, on the real logits of both conditions
                g["g2"] = g2_ok(lf, yg, rng) and g2_ok(lz, yg, rng)
                rec["groups"].append(g)
            rec["sec"] = round(time.time() - tf, 2)
            if not all(g["g0"]["ok"] for g in rec["groups"]):
                rec["excluded"] = "g0"
    res = {"ds": ds, "subj": str(subj), "counts": dict(counts), "t_data": round(t_data, 1),
           "t_total": round(time.time() - t0, 1), "fits": records}
    os.makedirs(f"{out}/runs/{ds}", exist_ok=True)
    tmp = f"{out}/runs/{ds}/{subj}.json.tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, f"{out}/runs/{ds}/{subj}.json")
    if not quiet:
        print(ds, subj, "fits", len(records), dict(counts), "t", res["t_total"])
    return res


def gates(F):
    n = collections.Counter()
    for f in F:
        n["fits"] += 1
        if f["excluded"] is not None:
            n[f"excluded:{f['excluded']}"] += 1
        for g in f["groups"]:
            n["groups"] += 1
            n["g1_fail"] += not g["g1"]
            n["g2_fail"] += not g["g2"]
            n["pw_nan"] += not (np.isfinite(g["pw_full"]) and np.isfinite(g["pw_z"]))
    return dict(n)


def selftest(E, e1b, bias, out):
    """Synthetic checks, then timing and gates on two real subjects. Prints NO score."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y3 = np.repeat([0, 1, 2], 40)
    lg = rng.normal(0, 1, (120, 3))
    lg[np.arange(120), y3] += 1.5
    p = pairwise_auc(lg, y3)
    assert abs(pairwise_auc(0.3 * lg + np.array([5., -2., 1.]) + rng.normal(0, 9, (120, 1)), y3) - p) < 1e-12
    y2 = np.repeat([0, 1], 30)
    l2 = rng.normal(0, 1, (60, 2))
    assert abs(pairwise_auc(l2, y2) - roc_auc_score(y2 == 1, l2[:, 1] - l2[:, 0])) < 1e-12
    assert abs(ovr_auc(l2, y2) - pairwise_auc(l2, y2)) < 1e-12
    noisy = lg + rng.normal(0, 3, lg.shape)                   # information destroyed: pw must drop
    assert pairwise_auc(noisy, y3) < p - 0.05
    assert abs(pairwise_auc(np.zeros((120, 3)), y3) - 0.5) < 1e-12   # constant output = chance
    assert np.isnan(pairwise_auc(lg[:80], y3[:80]))           # class 2 missing from the group
    shifted = lg + np.array([0., 0., -4.])                    # offset: accuracy drops, rank does not
    assert acc(shifted, y3) < acc(lg, y3) - 0.1 and abs(pairwise_auc(shifted, y3) - p) < 1e-12
    print("synthetic checks: 7 passed")
    for ds, subj in (("bnci2014_001", "1"), ("lee2019_mi", "1")):
        r = run_subject(E, e1b, bias, ds, subj, out, quiet=True)
        secs = collections.defaultdict(list)
        for f in r["fits"]:
            if "sec" in f:
                secs[f["model"]].append(f["sec"])
        print(ds, subj, "t_data", r["t_data"], "t_total", r["t_total"], "gates", gates(r["fits"]))
        print("  seconds per fit:", {m: round(float(np.mean(v)), 2) for m, v in secs.items()})


def contrast(fits, a, b):
    """{ds: [subject means of (a - b), averaged over the fit's groups]} (E1b's contrast)."""
    accu = collections.defaultdict(list)
    for f in fits:
        v = [g[a] - g[b] for g in f["groups"]]
        v = [x for x in v if np.isfinite(x)]
        if v:
            accu[(f["ds"], f["subj"])].append(float(np.mean(v)))
    out = collections.defaultdict(list)
    for (ds, _), v in sorted(accu.items()):
        out[ds].append(float(np.mean(v)))
    return out


def read(E, e1b, bias, out):
    T = [tuple(l.split()) for l in open(f"{e1b}/out/tasks.txt") if l.strip()]
    miss = [t for t in T if not os.path.exists(f"{out}/runs/{t[0]}/{t[1]}.json")]
    if miss:
        raise SystemExit(f"REFUSED: {len(miss)} of {len(T)} tasks have no result, e.g. {miss[:5]}")
    F = []
    for ds, s in T:
        F += json.load(open(f"{out}/runs/{ds}/{s}.json"))["fits"]
    G = gates(F)
    if G.get("g1_fail", 0) or G.get("g2_fail", 0):
        raise SystemExit(f"REFUSED: gates {G}")
    ok = collections.defaultdict(list)
    for f in F:
        if f["excluded"] is None:
            ok[(f["model"], f["arm"], f["prot"], "multi" if f["n_classes"] > 2 else "bin")].append(f)
    ci = lambda c: "—" if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']}, {c['n_ds']} ds, MDE {c['mde']:.3f})"
    L = ["# E1b-rank — supprimer les FFN abîme-t-il le classement des essais ? (EXPLORATOIRE)\n",
         f"Fits lus : {len(F)}. Portes : {G}.\n",
         "A = acc(z_all) − acc(full) ; K = pw(z_all) − pw(full), pw = AUC par paires de classes sur les écarts de logits (float64).",
         f"Verdict (note gelée) : « rang intact » si borne basse de K > {K_MARGIN} ; « rang perdu » si borne haute de K < {K_MARGIN} ; sinon non concluant.\n",
         "## Primaire : Conformer bd, datasets à 3-4 classes\n"]
    R = {"gates": G, "primary": {}, "secondary": {}}
    verdicts = []
    for p in E.PROTS:
        fs = ok[("conformer", "bd", p, "multi")]
        A, K = E.sboot(contrast(fs, "acc_z", "acc_full")), E.sboot(contrast(fs, "pw_z", "pw_full"))
        if K is None:
            v = "AUCUN DATASET"
        elif K["lo"] > K_MARGIN:
            v = "RANG INTACT"
        elif K["hi"] < K_MARGIN:
            v = "RANG PERDU"
        else:
            v = "NON CONCLUANT"
        verdicts.append(v)
        R["primary"][p] = {"A": A, "K": K, "verdict": v}
        L.append(f"- **{p}** : A = {ci(A)} ; K = {ci(K)} → **{v}**")
    ni, nl = verdicts.count("RANG INTACT"), verdicts.count("RANG PERDU")
    concl = ("le coût est un changement d'échelle/forme des logits, pas de classement (≥ 2/3 protocoles) : axe clos" if ni >= 2 else
             "les FFN de Conformer portent de l'information de classe (≥ 2/3 protocoles) : proposer la suite à Adam" if nl >= 2 else
             "non concluant")
    R["conclusion"] = concl
    L += ["", f"**Conclusion écrite avant lecture : {concl}.**\n", "## Secondaires (ne décident rien)\n",
          "| modèle | bras | protocole | classes | A (accuracy) | K (pw) | OvR z − full | pw full médian |",
          "|---|---|---|---|---|---|---|---|"]
    for m in E.MODELS:
        for arm in E.ARMS:
            for p in E.PROTS:
                for k in ("multi", "bin"):
                    fs = ok[(m, arm, p, k)]
                    if not fs:
                        continue
                    A = E.sboot(contrast(fs, "acc_z", "acc_full"))
                    K = E.sboot(contrast(fs, "pw_z", "pw_full"))
                    O = E.sboot(contrast(fs, "ovr_z", "ovr_full"))
                    pf = float(np.nanmedian([g["pw_full"] for f in fs for g in f["groups"]]))
                    R["secondary"]["|".join((m, arm, p, k))] = {"A": A, "K": K, "ovr": O, "pw_full": pf}
                    L.append(f"| {m} | {arm} | {p} | {k} | {ci(A)} | {ci(K)} | {ci(O)} | {pf:.3f} |")
    L.append("\n## Par dataset (Conformer bd)\n")
    for p in E.PROTS:
        for k in ("multi", "bin"):
            fs = ok[("conformer", "bd", p, k)]
            if not fs:
                continue
            a, kk = contrast(fs, "acc_z", "acc_full"), contrast(fs, "pw_z", "pw_full")
            L.append(f"- {p} / {k} : " + " ; ".join(f"{d} A {np.mean(a[d]):+.3f} K {np.mean(kk[d]):+.3f}" for d in sorted(a)))
    json.dump(R, open(f"{out}/E1b_rank.json", "w"), indent=1)
    open(f"{out}/E1b_rank.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "run", "read"))
    ap.add_argument("--e1b", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--line", type=int)
    a = ap.parse_args()
    E = load_e1b(a.e1b)
    if a.mode == "selftest":
        selftest(E, a.e1b, a.bias, a.out)
    elif a.mode == "run":
        T = [tuple(l.split()) for l in open(f"{a.e1b}/out/tasks.txt") if l.strip()]
        run_subject(E, a.e1b, a.bias, *T[a.line], a.out)
    else:
        read(E, a.e1b, a.bias, a.out)


if __name__ == "__main__":
    main()
