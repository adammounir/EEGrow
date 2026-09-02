"""Étage 0 du chantier donneur-receveur : #params est-il une mesure du *sujet* ?

La narrative visée (réunion Sylvain du 01/09) est que la taille à laquelle un réseau
growing s'arrête sur un sujet mesure ce sujet, et prédit sa valeur comme donnée
d'entraînement mieux que son accuracy. Deux conditions doivent tenir AVANT qu'un
protocole donneur-receveur ait un sens, et aucune n'a jamais été mesurée :

1. FIABILITÉ. Si ``params_end`` varie autant entre réplicats du même sujet (seeds,
   folds, sessions) qu'entre sujets, ce n'est pas une mesure du sujet, c'est du bruit
   d'optimisation. On le quantifie par un ICC(1) : la part de la variance totale qui
   est inter-sujets. Un ICC bas ne réfute pas l'hypothèse -- il rend le prédicteur
   inutilisable, ce qui revient au même pour le papier.

2. NON-REDONDANCE. Si #params est une fonction de l'accuracy, alors « meilleur
   prédicteur que l'accuracy » est une phrase vide. On régresse #params sur
   l'accuracy et la taille du jeu d'entraînement, et on regarde ce qui reste.

La fiabilité borne la validité : une corrélation vraie ``rho`` entre #params et la
qualité de donneur ne peut s'observer qu'à ``rho * sqrt(ICC)`` (atténuation par erreur
de mesure, Spearman 1904). C'est ce facteur qui dit si la matrice donneur-receveur a
la moindre chance de sortir un signal, et donc si elle vaut le GPU.

L'unité d'analyse est le (dataset, sujet) -- cf. ``perf_io.by_subject``.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_predictor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import perf_io  # noqa: E402

#: Le bras sonde. Shallow parce que c'est le modèle dont Sylvain veut faire un
#: instrument bon marché, within_session parce que c'est la seule évaluation où le
#: sujet de la ligne est le sujet sur lequel le modèle a été *entraîné* -- en
#: cross_subject la colonne ``subject`` est le sujet de test, et lire son #params
#: comme « la taille que ce sujet appelle » serait un contresens.
PROBE_MODEL = "grow_shallow"
PROBE_EVAL = "within_session"

#: En dessous, un prédicteur n'est pas assez reproductible pour être utile : la
#: corrélation observable est plafonnée à sqrt(0.5) = 0.71 de la vraie.
ICC_FLOOR = 0.50


def icc1(groups: list[np.ndarray]) -> tuple[float, int, int]:
    """ICC(1) par ANOVA à un facteur : part inter-sujets de la variance.

    Renvoie ``(icc, n_groupes, k_moyen)``. Les groupes de taille 1 ne portent aucune
    information sur la variance intra et sont écartés plutôt que comptés comme
    parfaitement fiables -- l'erreur qui gonfle un ICC sans qu'on la voie.
    """
    groups = [g for g in groups if len(g) >= 2]
    n = len(groups)
    if n < 3:
        return float("nan"), n, 0
    k = float(np.mean([len(g) for g in groups]))
    grand = np.mean(np.concatenate(groups))
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (n - 1)
    dfw = sum(len(g) - 1 for g in groups)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / dfw
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return float(icc), n, int(round(k))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Corrélation de Spearman entre x et y, z contrôlé, sur les rangs.

    Un ``z`` constant n'a rien à contrôler et ferait diverger l'ajustement : on
    retombe alors sur la corrélation simple.
    """
    if np.ptp(z) == 0:
        return float(stats.spearmanr(x, y)[0])
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(stats.pearsonr(ex, ey)[0])


def main() -> int:
    fits = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    sc = perf_io.load(A / "perf_final" / "scores")
    sc = perf_io.attach_params(sc, fits)
    subj = perf_io.by_subject(sc)

    # ---------------------------------------------------------------- 1. fiabilité
    # Réplicats du même sujet : toutes les lignes de fits qui partagent
    # (dataset, sujet) à bras et alignement fixés -- seeds x folds x sessions.
    f = fits[(fits["eval"] == PROBE_EVAL) & (fits.model == PROBE_MODEL)
             & (fits.align_tag == "none")].copy()
    f["subject"] = pd.to_numeric(f["subject"], errors="coerce")

    print("=" * 78)
    print("1. FIABILITÉ — params_end est-il une mesure du sujet ou du bruit de seed ?")
    print("=" * 78)
    print(f"{'dataset':22s} {'n_suj':>5s} {'k':>3s} {'ICC(1)':>8s} {'ICC_k':>7s} "
          f"{'sqrt':>6s} {'CV_inter':>9s} {'CV_intra':>9s}  verdict")
    rel = {}
    for ds, d in f.groupby("dataset"):
        groups = [g.params_end.to_numpy(float)
                  for _, g in d.groupby("subject") if len(g) >= 2]
        icc, n, k = icc1(groups)
        if not np.isfinite(icc):
            continue
        means = np.array([g.mean() for g in groups])
        cv_inter = means.std(ddof=1) / means.mean()
        cv_intra = float(np.mean([g.std(ddof=1) / g.mean() for g in groups]))
        # Spearman-Brown. Le prédicteur n'est PAS un fit isolé : c'est la moyenne des
        # k réplicats du sujet, et moyenner divise la variance d'erreur par k. Juger
        # l'instrument sur l'ICC(1) d'un fit unique reviendrait à condamner un
        # thermomètre parce qu'une mesure isolée est bruitée -- un faux négatif.
        icc_k = k * max(icc, 0) / (1 + (k - 1) * max(icc, 0)) if icc > 0 else 0.0
        rel[ds] = icc_k
        verdict = ("utilisable" if icc_k >= ICC_FLOOR else
                   "TROP BRUITÉ" if icc_k > 0.2 else "INUTILISABLE")
        print(f"{ds:22s} {n:5d} {k:3d} {icc:8.3f} {icc_k:7.2f} {icc_k**0.5:6.2f} "
              f"{cv_inter:9.3f} {cv_intra:9.3f}  {verdict}")
    print(f"\n  k      = réplicats par sujet (seeds x folds x sessions).")
    print(f"  ICC(1) = fiabilité d'UN fit ; ICC_k = fiabilité de leur MOYENNE, qui")
    print(f"           est le prédicteur réel (Spearman-Brown).")
    print(f"  sqrt(ICC_k) plafonne toute corrélation observable avec la qualité de")
    print(f"           donneur : c'est l'atténuation par erreur de mesure.")

    # ------------------------------------------------------------ 2. non-redondance
    sel = subj[(subj["eval"] == PROBE_EVAL) & (subj.align_tag == "none")
               & (subj.model == PROBE_MODEL) & subj.n_params.notna()
               & subj.samples.notna()].copy()

    print()
    print("=" * 78)
    print("2. NON-REDONDANCE — #params dit-il autre chose que l'accuracy ?")
    print("=" * 78)
    print(f"{'dataset':22s} {'n':>4s} {'rho(p,acc)':>11s} {'partielle':>10s} "
          f"{'R2(acc+ess)':>12s} {'resid':>7s} {'plafond':>8s}")
    for ds, d in sel.groupby("dataset"):
        if len(d) < 8:
            continue
        p = d.n_params.to_numpy(float)
        acc = d.score.to_numpy(float)
        ess = np.log(d.samples.to_numpy(float))
        rho = stats.spearmanr(p, acc)[0]
        # #params ~ accuracy + log(n_essais), sur variables centrées-réduites. Une
        # covariable constante (jeu de taille fixe sur tout le dataset) rend la
        # matrice singulière et fait diverger la SVD : on l'écarte au lieu de la
        # laisser tomber le dataset entier.
        cols = [np.ones(len(d))] + [stats.zscore(v) for v in (acc, ess)
                                    if np.ptp(v) > 0]
        X = np.column_stack(cols)
        zp = stats.zscore(p)
        beta, *_ = np.linalg.lstsq(X, zp, rcond=None)
        resid = zp - X @ beta
        r2 = 1.0 - resid.var() / zp.var()
        # Ce qui reste de #params une fois l'accuracy et la taille du jeu retirées,
        # exprimé comme fraction de l'écart-type d'origine : c'est le budget
        # d'information neuve dont dispose la narrative.
        icc = rel.get(ds, float("nan"))
        ceil = (max(icc, 0) ** 0.5) * np.sqrt(1 - r2) if np.isfinite(icc) else np.nan
        print(f"{ds:22s} {len(d):4d} {rho:11.3f} "
              f"{partial_spearman(p, acc, ess):10.3f} {r2:12.3f} "
              f"{np.sqrt(1 - r2):7.2f} {ceil:8.2f}")
    print("\n  resid  = écart-type de #params restant après accuracy + log(n_essais).")
    print("  plafond= sqrt(ICC) x resid : corrélation maximale que #params peut")
    print("           encore avoir avec la qualité de donneur, en propre.")

    # --------------------------------------------------------------- 3. puissance
    print()
    print("=" * 78)
    print("3. PUISSANCE — combien de sujets pour voir ce plafond ?")
    print("=" * 78)
    for ds in ("cho2017", "lee2019_mi", "physionetmi", "bnci2014_001"):
        d = sel[sel.dataset == ds]
        n = len(d)
        if n == 0:
            print(f"{ds:22s} absent des scores (fits présents) — à resynchroniser")
            continue
        # n minimal pour détecter une corrélation r à 80 % de puissance, alpha .05
        # bilatéral : n = ((z_a + z_b) / atanh(r))^2 + 3.
        need = lambda r: int(np.ceil(((1.959964 + 0.841621)
                                      / np.arctanh(min(abs(r), .999))) ** 2 + 3))
        mde = np.tanh((1.959964 + 0.841621) / np.sqrt(max(n - 3, 1)))
        print(f"{ds:22s} n={n:4d}  MDE rho={mde:.2f}  "
              f"(il faudrait n={need(0.4)} pour rho=0.4, n={need(0.6)} pour rho=0.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
