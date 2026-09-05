"""Quand le signal de croissance bat-il l'accuracy comme règle de tri des donneurs ?

Le face-à-face de la campagne d'allocation a tranché en défaveur de la croissance
(cho2017 12/12 pour `acc_probe`, physionetmi nul, lee2019_mi un léger avantage à
`params_probe`). L'hypothèse qui reste, et la seule qui sauverait l'angle du papier :
**l'accuracy cesse de discriminer quand les sujets se ressemblent**, et c'est là que la
dynamique de croissance prendrait le relais.

Ça se teste SANS GPU. Les matrices D×R contiennent le score de transfert mesuré de
chaque paire (donneur, receveur) -- la vérité terrain de « ce donneur vaut quoi pour ce
receveur ». Noter une règle de tri, c'est corréler le score qu'elle attribue aux donneurs
avec cette vérité. Aucun fit à refaire.

Deux lectures, et elles ne disent pas la même chose :

* NATURELLE -- la dispersion d'accuracy telle qu'elle est dans chaque dataset, n=7
  datasets. Descriptif, sous-puissant par construction, et rapporté comme tel.
* MANIPULÉE -- on FABRIQUE des viviers à dispersion contrôlée en échantillonnant dans
  une fenêtre d'accuracy de largeur variable, à taille de vivier CONSTANTE pour ne pas
  confondre dispersion et nombre de donneurs. C'est le test du mécanisme.

Réserve à porter avec le résultat manipulé : restreindre la fenêtre d'accuracy restreint
mécaniquement l'étendue d'`acc_probe`, donc atténue sa corrélation. C'est exactement le
mécanisme postulé, mais c'en est aussi la tautologie -- la manipulation montre que le
relais EXISTE, elle ne montre pas que les datasets réels se placent dans ce régime. Seule
la lecture naturelle peut le dire, et elle est à n=7.

Usage::

    ./.venv/bin/python benchmarks/analysis/rule_dispersion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260904)
N_BOOT = 20000
RULES = ["acc_probe", "params_probe", "restored_probe"]
# Les 5 datasets a n >= 50 donneurs. bnci2014_001 (9) et shin2017a (29) sont gardes pour
# la lecture naturelle seule : une fenetre a taille de vivier constante n'y laisse aucune
# marge de manoeuvre.
BIG = ["cho2017", "dreyer2023a", "lee2019_mi", "liu2024", "physionetmi"]
ALL = ["bnci2014_001", "shin2017a"] + BIG
POOL_M = 15                    # taille de vivier tenue CONSTANTE dans la manipulation
N_POOLS = 40                   # viviers tires par (receveur, largeur de fenetre)
WIDTHS = [0.30, 0.45, 0.60, 0.80, 1.00]


def truth(ds: str) -> pd.DataFrame:
    """Le score de transfert moyen (3 seeds) de chaque paire donneur -> receveur."""
    fr = pd.concat([pd.read_csv(f) for f in (A / f"dxr_{ds}").glob("*.csv")],
                   ignore_index=True)
    fr = fr[fr["self"] == 0]
    return fr.groupby(["receiver", "donor"])["roc_auc"].mean().reset_index()


def ranking(ds: str) -> pd.DataFrame:
    r = pd.read_csv(A / f"ranking_{ds}.csv")
    return r.set_index("subject")[RULES]


def rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman, mais `nan` si la regle est constante -- et non 0, qui serait un score."""
    if len(x) < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def natural(ds: str) -> dict:
    """Qualite de chaque regle sur le vivier ENTIER, moyennee sur les receveurs."""
    t, rk = truth(ds), ranking(ds)
    per = {m: [] for m in RULES}
    for recv, g in t.groupby("receiver"):
        d = g[g.donor.isin(rk.index)]
        y = d.roc_auc.to_numpy(float)
        for m in RULES:
            per[m].append(rho(rk.loc[d.donor, m].to_numpy(float), y))
    out = {"dataset": ds, "n_recv": t.receiver.nunique(),
           "sd_acc": float(rk["acc_probe"].std()),
           "cv_acc": float(rk["acc_probe"].std() / rk["acc_probe"].mean())}
    for m in RULES:
        out[m] = float(np.nanmean(per[m]))
    out["delta_par_acc"] = out["params_probe"] - out["acc_probe"]
    out["delta_res_acc"] = out["restored_probe"] - out["acc_probe"]
    out["_per"] = per
    return out


def manipulated(ds: str, window_on: str = "acc_probe") -> pd.DataFrame:
    """Viviers a dispersion contrôlée sur ``window_on``, taille constante a POOL_M.

    ``window_on`` est le CONTRÔLE decisif. Restreindre la fenetre d'accuracy restreint
    mecaniquement l'etendue d'`acc_probe`, donc attenue sa correlation : le gradient
    pourrait n'etre QUE de la restriction d'etendue, auquel cas il ne dit rien. Le test
    est la symetrie -- refaire la manipulation en fenetrant sur `params_probe`. Si les
    deux fenetrages se renvoient l'avantage de facon comparable, tout est artefact ; si
    l'echange est asymetrique, une des deux regles porte une information que l'autre
    n'a pas.
    """
    t, rk = truth(ds), ranking(ds)
    acc = rk[window_on]
    rows = []
    for recv, g in t.groupby("receiver"):
        d = g[g.donor.isin(rk.index)].set_index("donor")
        # Les donneurs de CE receveur, tries par accuracy : la fenetre se prend ici.
        order = acc.loc[d.index].sort_values().index.to_numpy()
        n = len(order)
        for w in WIDTHS:
            k = max(POOL_M, int(round(w * n)))
            if k > n:
                continue
            for _ in range(N_POOLS):
                s = RNG.integers(0, n - k + 1)
                block = order[s:s + k]
                pool = RNG.choice(block, POOL_M, replace=False)
                y = d.loc[pool, "roc_auc"].to_numpy(float)
                r = {m: rho(rk.loc[pool, m].to_numpy(float), y) for m in RULES}
                rows.append({"dataset": ds, "receiver": recv, "width": w,
                             "window_on": window_on,
                             "sd_win": float(acc.loc[pool].std()),
                             "sd_acc": float(rk.loc[pool, "acc_probe"].std()),
                             "sd_par": float(rk.loc[pool, "params_probe"].std()), **r})
                if w == 1.0:
                    break          # fenetre pleine : un seul tirage possible de bloc
    f = pd.DataFrame(rows)
    f["delta_par_acc"] = f["params_probe"] - f["acc_probe"]
    f["delta_res_acc"] = f["restored_probe"] - f["acc_probe"]
    return f


def boot_by_recv(f: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Bootstrap sur les RECEVEURS : les viviers d'un meme receveur sont correles."""
    per = f.groupby("receiver")[col].mean().to_numpy(float)
    per = per[~np.isnan(per)]
    if len(per) < 3:
        return float("nan"), float("nan"), float("nan")
    b = RNG.choice(per, (N_BOOT, len(per)), replace=True).mean(1)
    return float(per.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    print("=" * 78)
    print("A. LECTURE NATURELLE -- qualite de chaque regle sur le vivier entier")
    print("   rho de Spearman(score de la regle, transfert mesure), moyenne par receveur")
    print("=" * 78)
    nat = [natural(ds) for ds in ALL]
    hdr = f"{'dataset':<14}{'n':>4}{'sd(acc)':>9}{'acc':>8}{'params':>8}{'restor':>8}{'par-acc':>9}{'res-acc':>9}"
    print(hdr)
    for o in sorted(nat, key=lambda x: x["sd_acc"]):
        print(f"{o['dataset']:<14}{o['n_recv']:>4}{o['sd_acc']:>9.4f}"
              f"{o['acc_probe']:>8.3f}{o['params_probe']:>8.3f}{o['restored_probe']:>8.3f}"
              f"{o['delta_par_acc']:>+9.3f}{o['delta_res_acc']:>+9.3f}")
    nf = pd.DataFrame([{k: v for k, v in o.items() if k != "_per"} for o in nat])
    for lab, col in [("params-acc", "delta_par_acc"), ("restored-acc", "delta_res_acc")]:
        r = spearmanr(nf.sd_acc, nf[col])
        print(f"\n  Spearman(sd(acc), delta {lab}) sur n=7 datasets : "
              f"{r.statistic:+.3f} (p={r.pvalue:.3f}) -- DESCRIPTIF, n=7")

    print("\n" + "=" * 78)
    print(f"B. MANIPULATION -- viviers a dispersion controlee, taille fixe M={POOL_M}")
    print("=" * 78)
    man = pd.concat([manipulated(ds, w) for ds in BIG
                     for w in ("acc_probe", "params_probe")], ignore_index=True)
    man.to_csv(A / "rule_dispersion_pools.csv.gz", index=False)
    for ds in BIG:
        f = man[(man.dataset == ds) & (man.window_on == "acc_probe")]
        print(f"\n--- {ds} (n={f.receiver.nunique()} receveurs) ---")
        print(f"  {'fenetre':>8}{'sd(acc)':>9}{'acc':>8}{'params':>8}{'restor':>8}"
              f"   {'params - acc':>26}")
        for w in WIDTHS:
            s = f[f.width == w]
            if s.empty:
                continue
            mu, lo, hi = boot_by_recv(s, "delta_par_acc")
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {w:>8.2f}{s.sd_acc.mean():>9.4f}"
                  f"{np.nanmean(s.acc_probe):>8.3f}{np.nanmean(s.params_probe):>8.3f}"
                  f"{np.nanmean(s.restored_probe):>8.3f}"
                  f"   {mu:>+8.3f} [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\n" + "=" * 78)
    print("C. CONTRÔLE DE SYMÉTRIE -- fenetrer sur acc, puis sur params")
    print("   Si l'echange est symetrique, le gradient de B n'est que de la")
    print("   restriction d'etendue et ne dit rien. L'asymetrie est le resultat.")
    print("=" * 78)
    print(f"\n{'dataset':<14}{'fenetre sur acc':>26}{'fenetre sur params':>26}{'asym.':>9}")
    print(f"{'':14}{'-> params - acc':>26}{'-> acc - params':>26}")
    for ds in BIG:
        a = man[(man.dataset == ds) & (man.window_on == "acc_probe")
                & (man.width == 0.30)]
        p_ = man[(man.dataset == ds) & (man.window_on == "params_probe")
                 & (man.width == 0.30)]
        ma, la, ha = boot_by_recv(a, "delta_par_acc")
        p_ = p_.assign(delta_acc_par=-p_["delta_par_acc"])
        mp, lp, hp = boot_by_recv(p_, "delta_acc_par")
        sa = "*" if (la > 0 or ha < 0) else " "
        sp = "*" if (lp > 0 or hp < 0) else " "
        print(f"{ds:<14}{ma:>+9.3f} [{la:+.3f},{ha:+.3f}]{sa}"
              f"{mp:>+9.3f} [{lp:+.3f},{hp:+.3f}]{sp}{ma - mp:>+9.3f}")
    print("\nSorti : rule_dispersion_pools.csv.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
