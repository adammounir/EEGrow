"""XP 3 — la taille finale du réseau growing prédit-elle la qualité de donneur ?

Noyau partagé, rapatrié du scratchpad le 11/09/2026 pour que la figure et le chiffre
du point 3 soient reproductibles **depuis le dépôt**. Rien n'a changé dans le calcul :
l'implémentation reproduit les lignes publiées (`physionetmi` +0.520 / +0.430 / +0.385,
MDE 0.27) et le combiné +0.288 des 3 datasets pré-enregistrés.

XP 3 ne coûte aucun GPU : c'est une lecture croisée de XP 1 (la sonde, `ranking_*.csv`)
et XP 2 (la matrice donneur × receveur). Les deux existent sur 15 datasets ; seuls 4
avaient été lus, parce que XP 3 a tourné AVANT que l'XP 5 en ajoute 11.

LA QUANTITÉ QUI RÉPOND À LA QUESTION est la **rho partielle | accuracy** — ce que la
taille sait du sujet et que son accuracy ne sait pas. La marginale n'est pas lisible
sur `lee2019_mi` / `physionetmi`, où `params` corrèle avec l'accuracy (+0.52 / +0.65) :
une marginale positive peut n'y être qu'un relais de l'accuracy.

LA PORTE XP 2 n'est pas cosmétique. Sans transfert réel (> +0.01 au-dessus de la
chance) ET sans classement de donneurs stable (ICC_k ≥ 0.75), il n'y a pas de
« qualité de donneur » à prédire : corréler la taille à un classement de bruit ne veut
rien dire. 12 des 15 datasets la passent.

`yang2025` est ABSENT du tableau et doit le rester : sa matrice est partielle, et une
matrice à trous déplace la moyenne du receveur sans rien signaler
([[gates-must-refuse-not-warn]], [[dxr-scale-recount]]).

Usage::

    ./.venv/bin/python benchmarks/analysis/xp3_core.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

A = Path(__file__).resolve().parent
PULLED = A / "dxr_all"

# (nom, matrice D x R, ranking de la sonde, faisait-il partie des etages 1-4 ?)
DATASETS = [
    ("physionetmi",          A / "dxr_physionetmi",           A / "ranking_physionetmi.csv",              True),
    ("dreyer2023a",          A / "dxr_dreyer2023a",           A / "ranking_dreyer2023a.csv",              False),
    ("lee2019_mi",           A / "dxr_lee2019_mi",            A / "ranking_lee2019_mi.csv",               True),
    ("cho2017",              A / "dxr_cho2017",               A / "ranking_cho2017.csv",                  True),
    ("liu2024",              A / "dxr_liu2024",               A / "ranking_liu2024.csv",                  False),
    ("hefmiich2025",         PULLED / "hefmiich2025",         PULLED / "ranking_hefmiich2025.csv",        False),
    ("guttmannflury2025_mi", PULLED / "guttmannflury2025_mi", PULLED / "ranking_guttmannflury2025_mi.csv", False),
    ("guttmannflury2025_me", PULLED / "guttmannflury2025_me", PULLED / "ranking_guttmannflury2025_me.csv", False),
    ("shin2017a",            A / "dxr_shin2017a",             A / "ranking_shin2017a.csv",                False),
    ("chang2025",            PULLED / "chang2025",            PULLED / "ranking_chang2025.csv",           False),
    ("forenzo2023",          PULLED / "forenzo2023",          PULLED / "ranking_forenzo2023.csv",         False),
    ("dreyer2023b",          PULLED / "dreyer2023b",          PULLED / "ranking_dreyer2023b.csv",         False),
    ("kumar2024",            PULLED / "kumar2024",            PULLED / "ranking_kumar2024.csv",           False),
    ("brandl2020",           PULLED / "brandl2020",           PULLED / "ranking_brandl2020.csv",          False),
    ("bnci2014_001",         A / "dxr_bnci2014_001",          A / "ranking_bnci2014_001.csv",             True),
]

RNG = np.random.default_rng(20260908)
NBOOT = 4000


def rank(v):
    return pd.Series(v).rank().to_numpy()


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(x, y, z):
    """rho(x, y | z) : Spearman = Pearson sur les rangs, partialisé linéairement.

    Renvoie aussi les résidus. `ex` (la taille une fois l'accuracy retirée) ne dépend
    PAS de `y` : l'axe résiduel des figures n'est donc pas construit à partir de la
    variable qu'il sert à prédire.
    """
    rx, ry, rz = rank(x), rank(y), rank(z)

    def resid(a):
        A_ = np.c_[np.ones_like(rz), rz]
        return a - A_ @ np.linalg.lstsq(A_, a, rcond=None)[0]

    ex, ey = resid(rx), resid(ry)
    return float(np.corrcoef(ex, ey)[0, 1]), ex, ey


def boot_ci(fn, n, level=95):
    vals = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, n, n)
        try:
            v = fn(idx)
        except Exception:
            continue
        # un rebootage degenere (toutes les valeurs identiques) donne nan, pas une
        # exception : a n=9 ca arrive assez souvent pour vider l'IC si on ne filtre pas.
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < NBOOT // 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def icc1(vals):
    """ICC(1) one-way random sur une matrice sujets x seeds. Renvoie (ICC1, ICCk)."""
    n, k = vals.shape
    grand = vals.mean()
    msb = k * ((vals.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((vals - vals.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return max((msb - msw) / (msb + (k - 1) * msw), 0.0), max((msb - msw) / msb, 0.0)


def donor_quality(path):
    """Qualité de donneur + les deux portes de l'XP 2 (transfert moyen, ICC_k).

    La colonne est z-scorée PAR RECEVEUR avant d'être moyennée : sans ça, un donneur
    ne serait bien classé que parce qu'il a été évalué sur des receveurs faciles.
    """
    fr = pd.concat([pd.read_csv(f) for f in sorted(path.glob("donor*__seed*.csv"))],
                   ignore_index=True)
    fr = fr[fr["self"] == 0]
    mat = fr.pivot_table(index="donor", columns="receiver", values="roc_auc", aggfunc="mean")
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0)   # neutraliser la difficulte du receveur
    per_seed = {}
    for s, g in fr.groupby("seed"):
        m = g.pivot_table(index="donor", columns="receiver", values="roc_auc", aggfunc="mean")
        per_seed[s] = ((m - m.mean(axis=0)) / m.std(axis=0)).mean(axis=1)
    _, ik = icc1(pd.DataFrame(per_seed).dropna().to_numpy())
    return z.mean(axis=1), float(fr["roc_auc"].mean()) - 0.5, ik


def analyse(name, dxr_path, rank_path, core):
    q, transfer, icck = donor_quality(dxr_path)
    probe = pd.read_csv(rank_path).set_index("subject")
    common = [s for s in q.index if s in probe.index]
    d = pd.DataFrame({
        "quality": q.loc[common].to_numpy(),
        "params": probe.loc[common, "params_probe"].to_numpy(),
        "acc": probe.loc[common, "acc_probe"].to_numpy(),
        "ceil": probe.loc[common, "ceil_frac"].to_numpy(),
    }).dropna()
    n = len(d)
    rp = spearman(d["params"], d["quality"])
    ra = spearman(d["acc"], d["quality"])
    par, ex, ey = partial_spearman(d["params"], d["quality"], d["acc"])

    P, A_, Q = d["params"].to_numpy(), d["acc"].to_numpy(), d["quality"].to_numpy()
    lo, hi = boot_ci(lambda i: partial_spearman(P[i], Q[i], A_[i])[0], n)
    # difference appariee rho(params) - rho(accuracy), sur les MEMES sujets rebootes
    dif = rp - ra
    dlo, dhi = boot_ci(lambda i: spearman(P[i], Q[i]) - spearman(A_[i], Q[i]), n)
    mde = 2.802 / np.sqrt(n - 3) if n > 4 else np.nan
    gate = (transfer > 0.01) and (icck >= 0.75)
    return dict(name=name, core=core, n=n, rho_p=rp, rho_a=ra, par=par,
                par_lo=lo, par_hi=hi, dif=dif, dif_lo=dlo, dif_hi=dhi,
                mde=mde, ceil=float(d["ceil"].mean()), ex=ex, ey=ey,
                transfer=transfer, icck=icck, gate=gate)


def pool(rows_):
    """Fisher-z à effets fixes, poids n-4 (partielle à une covariable)."""
    z = np.arctanh([r["par"] for r in rows_])
    w = np.array([r["n"] - 4 for r in rows_], float)
    zb = float((w * z).sum() / w.sum())
    se = 1 / np.sqrt(w.sum())
    return np.tanh(zb), np.tanh(zb - 1.96 * se), np.tanh(zb + 1.96 * se)


def sign_p(k, n):
    """Test de signe binomial exact bilatéral, H0 : p = 1/2."""
    from math import comb
    def tail(j):
        return sum(comb(n, i) for i in range(j, n + 1)) / 2 ** n
    return float(min(1.0, 2 * min(tail(k), 1 - tail(k + 1) + comb(n, k) / 2 ** n)))


_ROWS: list | None = None


def rows() -> list:
    """Les 15 lignes, calculées une fois. Lazy : `analyse` reboote 2 x 4000 fois par
    dataset, et le script de figure n'a besoin que de `donor_quality` et de la porte."""
    global _ROWS
    if _ROWS is None:
        _ROWS = [analyse(*d) for d in DATASETS]
    return _ROWS


def gated_names() -> set[str]:
    """Les datasets sous porte XP 2, sans payer les bootstraps de `analyse`."""
    keep = set()
    for name, dxr_path, _, _ in DATASETS:
        _, transfer, icck = donor_quality(dxr_path)
        if transfer > 0.01 and icck >= 0.75:
            keep.add(name)
    return keep


def main() -> None:
    R = rows()
    print(f"{'dataset':<22}{'n':>4}{'rho_p':>8}{'rho_a':>8}{'partiel':>9}"
          f"{'IC95':>20}{'MDE':>7}{'porte':>7}{'diff':>8}{'IC diff':>20}")
    for r in R:
        print(f"{r['name']:<22}{r['n']:>4}{r['rho_p']:>+8.3f}{r['rho_a']:>+8.3f}"
              f"{r['par']:>+9.3f}   [{r['par_lo']:+.3f}, {r['par_hi']:+.3f}]"
              f"{r['mde']:>7.2f}{'OK' if r['gate'] else 'KO':>7}"
              f"{r['dif']:>+8.3f}   [{r['dif_lo']:+.3f}, {r['dif_hi']:+.3f}]")

    pub = [r for r in R if r["name"] in ("physionetmi", "lee2019_mi", "cho2017")]
    keep = [r for r in R if r["gate"]]
    for tag, sub in (("les 3 publies (validation)", pub), ("les 15", R),
                     ("les 12 sous porte XP 2", keep)):
        m, lo, hi = pool(sub)
        pos = sum(r["par"] > 0 for r in sub)
        print(f"\n{tag:<28} n={len(sub):>2}  rho combine = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"   positifs {pos}/{len(sub)}  p_signe={sign_p(pos, len(sub)):.4f}")
    dpos = sum(r["dif"] > 0 for r in R)
    print(f"\ndifference rho(taille) - rho(accuracy) positive sur {dpos}/15  "
          f"p_signe={sign_p(dpos, 15):.3f}")
    cc = np.corrcoef([r["ceil"] for r in R], [r["par"] for r in R])[0, 1]
    print(f"correlation (saturation de la sonde, rho partielle) sur les 15 : {cc:+.3f}")


if __name__ == "__main__":
    main()
