"""Le bras `all_pool` lu comme une RÉFÉRENCE, pas comme un test.

Écrit avant les données du bras, mais APRÈS celles de claim 3 : c'est un ajout post-hoc
et il est déclaré tel quel dans ``PREREG_ADDENDUM_all_pool.md``. Aucune valeur p n'est
imprimée ici, et c'est délibéré -- un bras ajouté après lecture des résultats ne fournit
pas de test, il fournit un point de comparaison.

CE QUE CE FICHIER RÉPOND
------------------------
« Sélectionner K donneurs vaut-il mieux que ne pas sélectionner du tout ? » Sylvain
avait nommé « tout le monde dedans » comme l'un des trois comparateurs ; il manquait.

CE QU'IL NE RÉPOND PAS, ET IL FAUT L'IMPRIMER À CHAQUE FOIS
-----------------------------------------------------------
K n'est PAS apparié : `all_pool` s'entraîne sur 39 ou 82 sujets, les règles sur 3 à 20.
Chaque écart mélange donc « plus de données » et « meilleures données ». D'où les deux K
dans chaque ligne de sortie : sans eux la ligne est illisible et sera mal citée.

L'UNITÉ, ET POURQUOI L'IC EST ANTI-CONSERVATEUR
-----------------------------------------------
`all_pool` n'a pas de réplicat -- le pool est déterministe étant donné le pli. L'unité
de rééchantillonnage disponible est donc le SUJET DE TEST, or les sujets d'un même pli
partagent exactement le même pool d'entraînement : leurs écarts sont corrélés et l'IC
bootstrap sur les sujets est trop étroit. C'est le défaut de niveau d'analyse de
[[unit-of-analysis-subject]], et on ne peut pas le corriger ici (un bootstrap-cluster
sur 4 plis n'a aucune précision). On fait donc les deux et on les imprime ensemble :

  * l'IC sujet, ÉTIQUETÉ anti-conservateur, pour l'ordre de grandeur ;
  * les 4 écarts PLI PAR PLI en clair, qui sont la vraie évidence -- 4 plis concordants
    en signe disent plus qu'un IC dont on sait que la largeur est fausse.

LE CONTRÔLE DE SOUS-ENTRAÎNEMENT, QUI N'EST PAS OPTIONNEL
----------------------------------------------------------
`all_pool` voit 4 à 8 fois plus d'essais que la plus grosse cellule de règle, avec le
même `patience=200`. S'il perd, l'explication banale est qu'il n'a pas convergé, pas que
la sélection marche. On imprime donc `epochs` et `stop_reason` des deux côtés AVANT les
écarts : si `all_pool` sort majoritairement sur `max_epochs` alors que les règles sortent
sur `early_stopping`, la comparaison est nulle et non avenue, et c'est écrit à l'écran.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_all_analysis.py \
        --all /scratch/amounir/dsel_all/cho2017 --dsel /scratch/amounir/dsel2/cho2017
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRIC = "roc_auc"
N_BOOT = 20000
RNG = np.random.default_rng(20260902)


def load_all(p: Path) -> pd.DataFrame:
    """Les cellules du bras seulement : le nom de règle `all_pool` n'est dans aucun RULES."""
    files = sorted(p.glob("f*__all_pool__d*__seed*.csv"))
    if not files:
        raise SystemExit(f"aucune cellule all_pool dans {p}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d


def load_rules(p: Path) -> pd.DataFrame:
    files = sorted(p.glob("f*__r*__k*__*__seed*.csv"))
    if not files:
        raise SystemExit(f"aucune cellule v2 dans {p}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_subject(d: pd.DataFrame) -> pd.Series:
    """Un score par sujet de test, moyenné sur les fits qui l'ont vu.

    La moyenne se fait ICI et pas dans `donor_select.py`, qui écrit une ligne par sujet :
    c'est ce qui permet de la refaire appariée sujet par sujet.
    """
    return d.groupby("test_subject")[METRIC].mean()


def convergence(d: pd.DataFrame, tag: str) -> None:
    e = pd.to_numeric(d["epochs"], errors="coerce")
    reasons = d["stop_reason"].value_counts(normalize=True)
    top = ", ".join(f"{k} {100 * v:.0f} %" for k, v in reasons.head(3).items())
    print(f"  {tag:22s} epochs médiane {e.median():6.0f}  "
          f"[{e.quantile(.1):.0f}, {e.quantile(.9):.0f}]   {top}")


def boot_ci(delta: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(delta), size=(N_BOOT, len(delta)))
    m = delta[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", required=True, type=Path)
    p.add_argument("--dsel", required=True, type=Path)
    a = p.parse_args(argv)

    A = load_all(a.all)
    R = load_rules(a.dsel)
    ds = A["dataset"].iloc[0]
    n_pool = int(A["k"].iloc[0])

    print(f"\n{'=' * 78}\nBRAS all_pool — {ds} — RÉFÉRENCE DESCRIPTIVE, POST-HOC\n"
          f"{'=' * 78}")
    print(f"  {len(A) // A['test_subject'].nunique()} fits par sujet de test, "
          f"pool = {n_pool} donneurs, {A['fold'].nunique()} plis")
    print(f"  Statut : ajouté après lecture de claim 3, à la demande de S. Chevallier.")
    print(f"  K NON APPARIÉ aux règles : tout écart ci-dessous mélange «plus de "
          f"données» et «meilleures données».")

    print(f"\n  CONVERGENCE (à lire AVANT les écarts)")
    convergence(A, f"all_pool (K={n_pool})")
    for k in sorted(R["k"].unique()):
        convergence(R[R.k == k], f"règles (K={k})")

    sa = per_subject(A)
    print(f"\n  all_pool : {METRIC} moyen par sujet = {sa.mean():.4f} "
          f"(n={len(sa)} sujets)")

    fold_of = A.groupby("test_subject")["fold"].first()

    print(f"\n  ÉCART all_pool − règle, par sujet de test (positif = ne pas "
          f"sélectionner est MEILLEUR)")
    print(f"  {'règle':16s} {'K':>3s} {'écart':>9s} {'IC95 sujet*':>20s}  "
          f"{'par pli':>28s}")
    rows = []
    for k in sorted(R["k"].unique()):
        for rule in sorted(R["rule"].unique()):
            sub = R[(R.k == k) & (R.rule == rule)]
            sr = per_subject(sub)
            common = sa.index.intersection(sr.index)
            d = (sa.loc[common] - sr.loc[common])
            lo, hi = boot_ci(d.to_numpy(float))
            byf = d.groupby(fold_of.loc[common]).mean()
            fstr = " ".join(f"{v:+.3f}" for v in byf)
            print(f"  {rule:16s} {k:>3d} {d.mean():>+9.4f} "
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>20s}  {fstr:>28s}")
            rows.append(dict(dataset=ds, rule=rule, k=k, n_pool=n_pool,
                             delta=d.mean(), lo=lo, hi=hi,
                             folds_agree=int(np.sign(byf).abs().sum() == len(byf)
                                             and len(set(np.sign(byf))) == 1)))
    print("\n  * IC ANTI-CONSERVATEUR : les sujets d'un même pli partagent le pool "
          "d'entraînement,\n    donc leurs écarts sont corrélés. La colonne «par pli» "
          "est la vraie évidence :\n    4 signes concordants valent plus que la largeur "
          "de cet intervalle.")

    out = a.dsel.parent / f"all_pool_{ds}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  écrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
