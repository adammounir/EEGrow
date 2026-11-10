"""E1b-bias (EXPLORATORY, analysis/EXPLO_E1b_bias.md) -- is the cost of removing every
transformer FFN (E1b's z_all) only a constant shift of the logits?

E1b scores 2-class datasets with ROC-AUC, which a constant shift of l1 - l0 leaves
unchanged, and 3-4-class datasets with accuracy, which such a shift can flip. This script
re-runs E1b's ``full`` and ``z_all`` forwards on the same frozen fits and, per scored group:

    acc_*   argmax accuracy (every dataset, 2-class included)
    nat_*   E1b's own score (ROC-AUC if 2 classes, accuracy otherwise)
    *_zrc   z_all logits recentred: lg_z - mean_test(lg_z) + mean_test(lg_full), per class
            (label-free, transductive: the constant correction that matches the full
            model's mean output on the same trials)
    bias_share  n ||mean(d)||^2 / ||d||_F^2 with d = lg_z - lg_full, each row centred
            over classes (softmax gauge removed): share of the logit change that is a
            constant shift

Imports the frozen E1b copy (same rebuild, hook and G0). CPU, read-only, no training.

    python e1b_bias.py selftest --e1b DIR --out OUT
    python e1b_bias.py run      --e1b DIR --line L --out OUT
    python e1b_bias.py read     --e1b DIR --out OUT
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

G1_TOL = 1e-6        # |score - E1b's stored score| for full and z_all
G2_TOL = 1e-9        # 2-class: AUC of recentred z_all == AUC of z_all (shift invariance)
R_MARGIN = -0.010    # verdict threshold on acc_zrc - acc_full


def load_e1b(e1b):
    sys.path.insert(0, e1b)
    import e1b_ablation as E
    return E


def acc(lg, y):
    classes = np.unique(y)
    idx = {c: i for i, c in enumerate(classes)}
    return float((lg.argmax(1) == np.array([idx[c] for c in y])).mean())


def auc64(lg, y):
    from sklearn.metrics import roc_auc_score
    lg = lg.astype(np.float64)
    return float(roc_auc_score(y == np.unique(y)[1], lg[:, 1] - lg[:, 0]))


def bias_share(lz, lf):
    d = (lz - lf).astype(np.float64)
    d -= d.mean(1, keepdims=True)
    tot = float((d ** 2).sum())
    return float(len(d) * (d.mean(0) ** 2).sum() / tot) if tot > 0 else float("nan")


def run_subject(E, e1b, ds, subj, out, quiet=False):
    import torch
    import adapt_B
    adapt_B._fastpath_off()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    cells_path = os.path.join(e1b, "grids/A/campaignA_cells_planned_av5.txt")
    ref = {f["uid"]: f for f in json.load(open(f"{e1b}/out/runs/{ds}/{subj}.json"))["fits"]}
    counts = collections.Counter()
    t0 = time.time()
    X, y, sess, _ = adapt_B.subject_data(ds, subj, cells_path)
    t_data = time.time() - t0
    records = []
    for c in E.subject_cells(cells_path, ds, subj):
        for uid, pt, npzs in E.fit_files(c, counts):
            tf = time.time()
            r = ref.get(uid)
            rec = {"model": c["model"], "arm": c["arm"], "prot": c["prot"], "ds": ds, "subj": str(subj),
                   "uid": uid, "excluded": None, "groups": []}
            records.append(rec)
            if r is None:
                rec["excluded"] = "not_in_e1b"
                continue
            if r["excluded"] is not None:
                rec["excluded"] = "e1b:" + r["excluded"]
                continue
            p = torch.load(pt, map_location="cpu", weights_only=False)
            G = E.groups_of(c, npzs, X, y, sess, subj)
            if G is None or len(G) != len(r["groups"]):
                rec["excluded"] = "test_set_mismatch"
                continue
            n_classes = int(G[0][3].shape[1])
            rec["n_classes"] = n_classes
            model, _ = E.build(c["model"], p["state_dict"], X.shape[1], n_classes, X.shape[2])
            B = E.N_BLOCKS[c["model"]]
            ab = E.Ablator(c["model"], model)
            for (se, Xg, yg, saved, _), rg in zip(G, r["groups"]):
                ab.set([None] * B)
                lf = E.logits_of(model, Xg)
                ab.set([("zero",)] * B)
                lz = E.logits_of(model, Xg)
                lrc = lz - lz.mean(0, keepdims=True) + lf.mean(0, keepdims=True)
                g = {"session": se, "n": int(len(yg)), "g0": E.g0(lf, saved, yg, n_classes),
                     "acc_full": acc(lf, yg), "acc_z": acc(lz, yg), "acc_zrc": acc(lrc, yg),
                     "nat_full": E.score(lf, yg, n_classes), "nat_z": E.score(lz, yg, n_classes),
                     "nat_zrc": E.score(lrc, yg, n_classes), "bias_share": bias_share(lz, lf)}
                # G1: same scores as the frozen E1b run
                g["g1"] = (str(rg["session"]) == str(se)
                           and abs(g["nat_full"] - rg["scores"]["full"]) <= G1_TOL
                           and abs(g["nat_z"] - rg["scores"]["z_all"]) <= G1_TOL)
                # G2: 2-class AUC is blind to the recentring (the premise of this analysis), in
                # float64 on l1 - l0; the scorer's float32 softmax saturates and its ties can
                # move (checker_float32 note), so nat_zrc - nat_z is only recorded
                if n_classes == 2:
                    g["g2"] = abs(auc64(lrc, yg) - auc64(lz, yg)) <= G2_TOL
                    g["d_nat32"] = abs(g["nat_zrc"] - g["nat_z"])
                else:
                    g["g2"] = True
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
            if g.get("d_nat32", 0) > 0:
                n["nat32_moved"] += 1
                n["nat32_max_e6"] = max(n["nat32_max_e6"], round(g["d_nat32"] * 1e6))
    return dict(n)


def selftest(E, e1b, out):
    """Synthetic checks, then timing and gates on two real subjects. Prints NO score."""
    lf = np.array([[0., 1.], [2., 0.], [1., 1.5], [0., -1.]])
    y = np.array([1, 0, 1, 0])
    assert acc(lf, y) == 1.0
    assert abs(bias_share(lf + np.array([3., -1.]), lf) - 1.0) < 1e-12      # pure shift
    assert np.isnan(bias_share(lf, lf))                                     # no change: undefined
    d =np.array([[1., -1.], [-1., 1.], [1., -1.], [-1., 1.]])
    assert bias_share(lf + d, lf) < 1e-12                                   # zero-mean change
    lz = lf + np.array([0., -2.])
    lrc = lz - lz.mean(0) + lf.mean(0)
    assert np.allclose(lrc, lf)                                             # recentring undoes a shift
    print("synthetic checks: 5 passed")
    for ds, subj in (("bnci2014_001", "1"), ("lee2019_mi", "1")):
        r = run_subject(E, e1b, ds, subj, out, quiet=True)
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
        v = [g[a] - g[b] for g in f["groups"]] if b else [g[a] for g in f["groups"]]
        v = [x for x in v if np.isfinite(x)]
        if v:
            accu[(f["ds"], f["subj"])].append(float(np.mean(v)))
    out = collections.defaultdict(list)
    for (ds, _), v in sorted(accu.items()):
        out[ds].append(float(np.mean(v)))
    return out


def read(E, e1b, out):
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
    L = ["# E1b-bias — le coût de z_all est-il un décalage constant des logits ? (EXPLORATOIRE)\n",
         f"Fits lus : {len(F)}. Portes : {G}.\n",
         "A = acc(z_all) − acc(full) ; R = acc(z_all recentré) − acc(full) ; part récupérée = 1 − R/A (sur les moyennes).",
         f"Verdict (note gelée) : « biais » si borne basse de R > {R_MARGIN} ; « information » si borne haute de R < {R_MARGIN} ; sinon non concluant.\n",
         "## Primaire : Conformer bd, datasets à 3-4 classes\n"]
    R = {"gates": G, "primary": {}, "secondary": {}}
    verdicts = []
    for p in E.PROTS:
        fs = ok[("conformer", "bd", p, "multi")]
        A, Rc = E.sboot(contrast(fs, "acc_z", "acc_full")), E.sboot(contrast(fs, "acc_zrc", "acc_full"))
        if Rc is None:
            v = "AUCUN DATASET"
        elif Rc["lo"] > R_MARGIN:
            v = "BIAIS"
        elif Rc["hi"] < R_MARGIN:
            v = "INFORMATION"
        else:
            v = "NON CONCLUANT"
        verdicts.append(v)
        share = (1 - Rc["mean"] / A["mean"]) if A and Rc and A["mean"] < 0 else float("nan")
        R["primary"][p] = {"A": A, "R": Rc, "share": share, "verdict": v}
        L.append(f"- **{p}** : A = {ci(A)} ; R = {ci(Rc)} ; part récupérée {share:.2f} → **{v}**")
    nb, ni = verdicts.count("BIAIS"), verdicts.count("INFORMATION")
    concl = ("FFN = un biais de sortie (≥ 2/3 protocoles) : axe clos" if nb >= 2 else
             "FFN = de l'information de classe (≥ 2/3 protocoles) : proposer la suite à Adam" if ni >= 2 else
             "non concluant")
    R["conclusion"] = concl
    L += ["", f"**Conclusion écrite avant lecture : {concl}.**\n", "## Secondaires (ne décident rien)\n",
          "| modèle | bras | protocole | classes | A (accuracy) | R (recentré) | nat z − full | bias_share médian |",
          "|---|---|---|---|---|---|---|---|"]
    for m in E.MODELS:
        for arm in E.ARMS:
            for p in E.PROTS:
                for k in ("multi", "bin"):
                    fs = ok[(m, arm, p, k)]
                    if not fs:
                        continue
                    A = E.sboot(contrast(fs, "acc_z", "acc_full"))
                    Rc = E.sboot(contrast(fs, "acc_zrc", "acc_full"))
                    N = E.sboot(contrast(fs, "nat_z", "nat_full"))
                    bs = float(np.nanmedian([g["bias_share"] for f in fs for g in f["groups"]]))
                    R["secondary"]["|".join((m, arm, p, k))] = {"A": A, "R": Rc, "nat": N, "bias_share": bs}
                    L.append(f"| {m} | {arm} | {p} | {k} | {ci(A)} | {ci(Rc)} | {ci(N)} | {bs:.2f} |")
    L.append("\n## Par dataset (Conformer bd)\n")
    for p in E.PROTS:
        for k in ("multi", "bin"):
            fs = ok[("conformer", "bd", p, k)]
            if not fs:
                continue
            a, r = contrast(fs, "acc_z", "acc_full"), contrast(fs, "acc_zrc", "acc_full")
            L.append(f"- {p} / {k} : " + " ; ".join(f"{d} A {np.mean(a[d]):+.3f} R {np.mean(r[d]):+.3f}" for d in sorted(a)))
    json.dump(R, open(f"{out}/E1b_bias.json", "w"), indent=1)
    open(f"{out}/E1b_bias.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "run", "read"))
    ap.add_argument("--e1b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--line", type=int)
    a = ap.parse_args()
    E = load_e1b(a.e1b)
    if a.mode == "selftest":
        selftest(E, a.e1b, a.out)
    elif a.mode == "run":
        T = [tuple(l.split()) for l in open(f"{a.e1b}/out/tasks.txt") if l.strip()]
        run_subject(E, a.e1b, *T[a.line], a.out)
    else:
        read(E, a.e1b, a.out)


if __name__ == "__main__":
    main()
