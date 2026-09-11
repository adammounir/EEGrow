"""Le chiffre de la figure XP 3, avec son incertitude et son nul.

CE QUE CE SCRIPT CORRIGE
------------------------
La figure « où tombent les bons donneurs le long de l'axe des tailles » affiche
« part de forte : 17 % à gauche → 54 % à droite (33 % = aucune relation) ». Trois
choses y manquent, et les trois sont exactement ce qu'un relecteur demandera :

1. **Aucun intervalle.** 17 % et 54 % sont des proportions sur des sous-ensembles de
   ~150 sujets ; sans IC on ne sait pas si l'écart tient à 10 points près.
2. **Le 33 % est nominal, pas mesuré.** Les terciles sont équilibrés DANS chaque
   dataset et l'axe est un percentile DANS chaque dataset, mais le tiers gauche de
   l'axe POOLÉ mélange des datasets de n différents ; la part attendue sous l'absence
   de relation n'est donc pas exactement 1/3. On la mesure par permutation au lieu de
   la supposer.
3. **Rien ne dit si l'effet est porté par un seul dataset.** `kumar2024` (n=18,
   partielle +0.681) pourrait le fabriquer à lui seul.

LE STATISTIQUE ET SON UNITÉ. L'unité est le SUJET, et les datasets sont FIXES (même
convention que la méta de claim 2, `donor_meta.py`) : le bootstrap rééchantillonne les
sujets À L'INTÉRIEUR de chaque dataset, jamais les datasets. Toute la chaîne est
refaite sur chaque rééchantillon — percentile de taille, résidualisation par
l'accuracy, terciles de qualité — parce que la chaîne EST l'estimateur ; figer les
rangs sur l'échantillon observé sous-estimerait l'incertitude.

LE NUL. Permuter la qualité de donneur À L'INTÉRIEUR de chaque dataset casse le lien
sujet ↔ qualité en gardant intactes les marges (les tailles, les accuracies, les n, le
déséquilibre entre datasets). C'est le nul exact de la question posée.

CE QUE LE CHIFFRE N'EST PAS. Le basculement est une version GROSSIÈRE de la rho
partielle +0.286 [+0.194, +0.373], pas un résultat indépendant : il découpe la même
association en trois classes pour la rendre lisible. L'endpoint pré-enregistré reste la
rho partielle sur les 3 datasets déclarés ([[xp3-extended-15-datasets]]) ; ceci est la
figure qui la montre.

Usage::

    ./.venv/bin/python benchmarks/analysis/xp3_tilt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import xp3_core as X  # noqa: E402

NBOOT = 4000
NPERM = 4000
RNG = np.random.default_rng(20260911)
BANDS = ["faible", "moyenne", "forte"]
OUT = A / "xp3_tilt.csv"


# --------------------------------------------------------------------------- chaine

def _pct(v: np.ndarray) -> np.ndarray:
    """Percentile intra-dataset, 0 = le plus petit modele du dataset, 100 le plus gros."""
    n = len(v)
    return 100.0 * (X.rank(v) - 0.5) / n


def _tercile(q: np.ndarray) -> np.ndarray:
    """Tercile de qualite DANS le dataset, par les rangs.

    Par les rangs et non par `pd.qcut` : sous bootstrap les valeurs se repetent, et
    `qcut` fusionne alors des bornes et rend des terciles desequilibres — le
    desequilibre se lirait ensuite comme un basculement.
    """
    n = len(q)
    return np.clip((3 * (X.rank(q) - 0.5) / n).astype(int), 0, 2)


def _axes(params: np.ndarray, acc: np.ndarray) -> dict[str, np.ndarray]:
    """Les deux axes de la figure, tous deux en percentile intra-dataset."""
    # `ex` = la taille une fois l'accuracy retiree (en rangs). Il ne depend pas de la
    # qualite : l'axe n'est pas construit avec la variable qu'il sert a predire.
    A_ = np.c_[np.ones(len(acc)), X.rank(acc)]
    rp = X.rank(params)
    ex = rp - A_ @ np.linalg.lstsq(A_, rp, rcond=None)[0]
    return {"p_size": _pct(params), "p_resid": _pct(ex)}


def shares(axis: np.ndarray, terc: np.ndarray) -> tuple[float, float]:
    """Part de « qualite forte » dans le tiers GAUCHE et le tiers DROIT de l'axe."""
    lo, hi = axis < 100 / 3, axis > 200 / 3
    f = terc == 2
    return (float(f[lo].mean()) if lo.any() else np.nan,
            float(f[hi].mean()) if hi.any() else np.nan)


# ---------------------------------------------------------------------------- donnees

def frame() -> pd.DataFrame:
    """Un sujet par ligne, sur les datasets qui passent la porte de l'XP 2."""
    keep = X.gated_names()
    out = []
    for name, dxr_path, rank_path, _ in X.DATASETS:
        if name not in keep:
            continue
        q, _, _ = X.donor_quality(dxr_path)
        probe = pd.read_csv(rank_path).set_index("subject")
        common = [s for s in q.index if s in probe.index]
        out.append(pd.DataFrame({
            "dataset": name,
            "quality": q.loc[common].to_numpy(),
            "params": probe.loc[common, "params_probe"].to_numpy(),
            "acc": probe.loc[common, "acc_probe"].to_numpy(),
        }).dropna())
    return pd.concat(out, ignore_index=True)


def _blocks(d: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Un bloc de tableaux numpy par dataset — le bootstrap ne touche plus a pandas."""
    return [(name, g["params"].to_numpy(), g["acc"].to_numpy(), g["quality"].to_numpy())
            for name, g in d.groupby("dataset", sort=False)]


def stat(blocks, picks=None, perm=False) -> dict[str, tuple[float, float]]:
    """Parts gauche/droite poolees, chaine entierement refaite par dataset.

    `picks` : indices de bootstrap par dataset. `perm` : la qualite est permutee dans
    chaque dataset (le nul).
    """
    acc_ax = {k: [] for k in ("p_size", "p_resid")}
    terc = []
    for j, (_, P, Acc, Q) in enumerate(blocks):
        i = picks[j] if picks is not None else np.arange(len(P))
        p, a, q = P[i], Acc[i], Q[i]
        if perm:
            q = RNG.permutation(q)
        ax = _axes(p, a)
        for k in acc_ax:
            acc_ax[k].append(ax[k])
        terc.append(_tercile(q))
    t = np.concatenate(terc)
    return {k: shares(np.concatenate(v), t) for k, v in acc_ax.items()}


def _ci(vals: np.ndarray) -> tuple[float, float]:
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


# ------------------------------------------------------------------------------ main

def main() -> None:
    d = frame()
    blocks = _blocks(d)
    n_s, n_ds = len(d), len(blocks)
    obs = stat(blocks)

    # bootstrap : sujets dans le dataset, datasets fixes
    keys = ("p_size", "p_resid")
    bl = {k: np.empty(NBOOT) for k in keys}
    bh = {k: np.empty(NBOOT) for k in keys}
    bt = {k: np.empty(NBOOT) for k in keys}
    for t in range(NBOOT):
        picks = [RNG.integers(0, len(P), len(P)) for _, P, _, _ in blocks]
        s = stat(blocks, picks)
        for k in keys:
            bl[k][t], bh[k][t] = s[k]
            bt[k][t] = s[k][1] - s[k][0]

    # nul par permutation de la qualite dans chaque dataset
    pl = {k: np.empty(NPERM) for k in keys}
    ph = {k: np.empty(NPERM) for k in keys}
    pt = {k: np.empty(NPERM) for k in keys}
    for t in range(NPERM):
        s = stat(blocks, perm=True)
        for k in keys:
            pl[k][t], ph[k][t] = s[k]
            pt[k][t] = s[k][1] - s[k][0]

    bar = "=" * 96
    print(bar)
    print("INTEGRITE")
    print(bar)
    print(f"{n_s} sujets, {n_ds} datasets sous porte XP 2, "
          f"{NBOOT} bootstraps (sujets dans dataset), {NPERM} permutations")
    for name, P, _, _ in blocks:
        print(f"  {name:<24} n={len(P):>4}")

    print()
    print(bar)
    print("1. BASCULEMENT POOLE — part de « qualite de donneur forte » le long de l'axe")
    print(bar)
    print(f"{'axe':<10}{'gauche':>22}{'droite':>22}{'droite - gauche':>26}{'nul':>10}{'p':>8}")
    for k, lab in (("p_size", "brut"), ("p_resid", "residuel")):
        lo, hi = obs[k]
        tilt = hi - lo
        p = float((np.abs(pt[k] - np.nanmean(pt[k])) >= abs(tilt - np.nanmean(pt[k]))).mean())
        print(f"{lab:<10}"
              f"{100 * lo:>9.1f} % [{100 * _ci(bl[k])[0]:.1f},{100 * _ci(bl[k])[1]:5.1f}]"
              f"{100 * hi:>9.1f} % [{100 * _ci(bh[k])[0]:.1f},{100 * _ci(bh[k])[1]:5.1f}]"
              f"{100 * tilt:>+13.1f} pts [{100 * _ci(bt[k])[0]:+.1f},{100 * _ci(bt[k])[1]:+6.1f}]"
              f"{100 * np.nanmean(pt[k]):>+9.1f}{p:>8.4f}")
    print()
    print("  « nul » = basculement moyen sous permutation de la qualite dans chaque")
    print("  dataset (marges intactes). La part attendue a gauche comme a droite y est")
    print(f"  {100 * np.nanmean(pl['p_size']):.1f} % / {100 * np.nanmean(ph['p_size']):.1f} %, "
          "et non exactement 33 % : les datasets n'ont pas le meme n.")

    print()
    print(bar)
    print("2. LE MEME BASCULEMENT, DATASET PAR DATASET (axe residuel)")
    print("   Un basculement porte par un seul dataset ne serait pas un resultat.")
    print(bar)
    print(f"{'dataset':<24}{'n':>5}{'gauche':>12}{'droite':>12}{'ecart':>12}")
    per = []
    for name, P, Acc, Q in blocks:
        ax = _axes(P, Acc)["p_resid"]
        lo, hi = shares(ax, _tercile(Q))
        per.append((name, len(P), lo, hi))
        print(f"{name:<24}{len(P):>5}{100 * lo:>11.0f} %{100 * hi:>11.0f} %"
              f"{100 * (hi - lo):>+11.0f} pts")
    # Convention du test de signe : les EGALITES EXACTES sont ecartees, pas comptees
    # comme des echecs. Elles n'en sont pas : a n=9 ou n=16 un tiers d'axe contient 3 a
    # 5 sujets, donc un ecart nul veut dire « meme compte de fortes des deux cotes »,
    # soit le seul resultat que ce grain puisse produire. La rho partielle, qui ne
    # decoupe rien, est positive sur 11 de ces 12 (cf. xp3_core).
    pos = sum(hi > lo for _, _, lo, hi in per)
    ties = sum(hi == lo for _, _, lo, hi in per)
    nz = len(per) - ties
    print(f"\n  basculement positif sur {pos}/{len(per)} datasets, dont {ties} egalites")
    print(f"  test de signe, egalites ecartees : {pos}/{nz}  p = {X.sign_p(pos, nz):.4f}")
    print(f"  egalites comptees comme echecs   : {pos}/{len(per)}  "
          f"p = {X.sign_p(pos, len(per)):.4f}  (conservateur)")

    print()
    print(bar)
    print("3. LAISSER UN DATASET DEHORS (axe residuel) — le pooled tient-il sans lui ?")
    print(bar)
    base = obs["p_resid"][1] - obs["p_resid"][0]
    print(f"{'retire':<24}{'basculement restant':>22}{'ecart au complet':>20}")
    for j, (name, _, _, _) in enumerate(blocks):
        sub = [b for i, b in enumerate(blocks) if i != j]
        s = stat(sub)["p_resid"]
        v = s[1] - s[0]
        print(f"{name:<24}{100 * v:>+19.1f} pts{100 * (v - base):>+17.1f} pts")
    print(f"{'(aucun)':<24}{100 * base:>+19.1f} pts")

    pd.DataFrame(per, columns=["dataset", "n", "share_left", "share_right"]).to_csv(
        OUT, index=False)
    print(f"\necrit {OUT}")


if __name__ == "__main__":
    main()
