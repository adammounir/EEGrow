"""restored_epoch de la sonde (grow_shallow within_session) sous 3 règles de sélection.

orig  : RestoreBestModel tel qu'il a tourné (1er argmax de valid_acc, égalité -> plus tôt)
late  : même courbe, égalité -> époque la plus TARDIVE
cv    : par (sujet, seed), courbe valid_acc moyennée sur tous les plis/sessions
        (pondérée par n_valid = tous les essais du sujet), puis dernier argmax

Porte : `orig` recalculé depuis l'historique doit reproduire `restored_epoch` enregistré
fit par fit, et sa moyenne par sujet doit reproduire `restored_probe` de ranking_<ds>.csv.
Sinon on ne lit rien.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # JSONL rapatriés de /scratch/amounir/results_{final,fresh} sous HERE/
A = Path("/Users/adammounir/Desktop/Inria/Exploration/eegrow/benchmarks/analysis")
sys.path.insert(0, str(A))
import donor_matrix as dm  # noqa: E402
import donor_predictor as dp  # noqa: E402

DATASETS = {  # nom -> (arbre, porte d'étage C ?)
    "physionetmi": ("results_final", True), "lee2019_mi": ("results_final", True),
    "cho2017": ("results_final", True), "bnci2014_001": ("results_final", False),
    "shin2017a": ("results_final", False), "dreyer2023a": ("results_fresh", False),
    "liu2024": ("results_fresh", False),
}


def n_valid_of(acc):
    vals = np.unique(np.round(acc, 6))
    for k in range(1, 2000):
        if np.allclose(vals * k, np.round(vals * k), atol=1e-4):
            return k
    return np.nan


def reduce(ds, tree):
    fits, curves = [], []
    for seed in (0, 1, 2):
        p = HERE / tree / "within_session" / ds / f"grow_shallow__seed{seed}__fits.jsonl"
        for line in p.open():
            r = json.loads(line)
            h = r["history"]
            acc = np.array([x["valid_acc"] for x in h], float)
            ep = np.array([x["epoch"] for x in h])
            best = acc.max()
            first, last = ep[np.argmax(acc)], ep[np.where(acc == best)[0][-1]]
            # RestoreBestModel écrit None quand la meilleure époque est la dernière
            orig = np.nan if first == ep[-1] else first
            fits.append(dict(seed=seed, subject=int(r["subject"]), session=r["session"],
                             cv=r["cv_ind"], rec=r["restored_epoch"], orig=orig,
                             late=last, nv=n_valid_of(acc), n_epochs=len(ep)))
            curves.append((seed, int(r["subject"]), n_valid_of(acc), acc))
    f = pd.DataFrame(fits)
    f["rec"] = pd.to_numeric(f["rec"], errors="coerce")
    # porte 1 : fit par fit
    both = f.rec.isna() == f.orig.isna()
    same = both & ((f.rec == f.orig) | f.rec.isna())
    if not same.all():
        raise SystemExit(f"{ds}: orig != enregistré sur {(~same).sum()}/{len(f)} fits")
    # cv : courbe moyenne pondérée par n_valid, par (sujet, seed)
    rows = []
    by = {}
    for seed, s, nv, acc in curves:
        by.setdefault((s, seed), []).append((nv, acc))
    for (s, seed), lst in by.items():
        L = min(len(a) for _, a in lst)
        w = np.array([nv for nv, _ in lst], float)
        m = (np.stack([a[:L] for _, a in lst]) * w[:, None]).sum(0) / w.sum()
        last = np.where(np.isclose(m, m.max()))[0][-1] + 1
        rows.append(dict(subject=s, seed=seed, cv_ep=last, nv_tot=w.sum()))
    c = pd.DataFrame(rows)
    return f, c


def main():
    out = []
    pooled = {k: [] for k in ("orig", "late", "cv")}
    for ds, (tree, stage_c) in DATASETS.items():
        f, c = reduce(ds, tree)
        subj = f.groupby("subject").agg(orig=("orig", "mean"), late=("late", "mean"),
                                        nv=("nv", "median"),
                                        early_orig=("orig", lambda v: np.mean(v.fillna(999) < 5)),
                                        early_late=("late", lambda v: np.mean(v < 5)))
        subj = subj.join(c.groupby("subject").agg(cv=("cv_ep", "mean"),
                                                  nv_tot=("nv_tot", "mean"),
                                                  early_cv=("cv_ep", lambda v: np.mean(v < 5))))
        subj = subj.reset_index()
        # porte 2 : reproduit restored_probe
        rk = pd.read_csv(A / f"ranking_{ds}.csv")
        chk = subj.merge(rk[["subject", "restored_probe", "acc_probe"]], on="subject")
        gap = np.nanmax(np.abs(chk.orig - chk.restored_probe))
        if not gap < 1e-9:
            raise SystemExit(f"{ds}: restored_probe non reproduit (écart {gap})")
        # fiabilité
        icc = {}
        for col, src, key in (("orig", f, "orig"), ("late", f, "late"), ("cv", c, "cv_ep")):
            g = [v.dropna().to_numpy(float) for _, v in src.groupby("subject")[key]]
            i, _, k = dp.icc1(g)
            icc[col] = (k * i / (1 + (k - 1) * i)) if i > 0 else 0.0
        # qualité de donneur
        d = dm.load_matrix(A / f"dxr_{ds}")
        q, _ = dm.donor_quality(d, dm.METRIC)
        m = chk.merge(q[["donor", "q_cent"]], left_on="subject", right_on="donor")
        m = m.dropna(subset=["orig", "late", "cv", "q_cent", "acc_probe"])
        line = dict(dataset=ds, stage_c=stage_c, n=len(m), nv_fit=f.nv.median(),
                    nv_cv=c.nv_tot.median(),
                    early_orig=(f.orig.fillna(999) < 5).mean(), early_late=(f.late < 5).mean(),
                    early_cv=(c.cv_ep < 5).mean())
        for k in ("orig", "late", "cv"):
            r = dp.partial_spearman(m[k].to_numpy(float), m.q_cent.to_numpy(float),
                                    m.acc_probe.to_numpy(float))
            line[f"par_{k}"] = r
            line[f"icck_{k}"] = icc[k]
            if stage_c:
                pooled[k].append((r, len(m)))
        line["rho_orig_cv"] = pd.Series(m.orig).rank().corr(pd.Series(m.cv).rank())
        out.append(line)
        subj.assign(dataset=ds).to_csv(HERE / f"subj_{ds}.csv", index=False)
    t = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print("\nÉtage C groupé (Fisher-z effets fixes, poids n-4) : physionetmi+lee2019_mi+cho2017")
    for k, lst in pooled.items():
        z = np.arctanh([r for r, _ in lst]); w = np.array([n - 4 for _, n in lst], float)
        zb = (w * z).sum() / w.sum(); se = 1 / np.sqrt(w.sum())
        print(f"  {k:5s} {np.tanh(zb):+.3f} [{np.tanh(zb-1.96*se):+.3f}, {np.tanh(zb+1.96*se):+.3f}]")
    t.to_csv(HERE / "re_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
