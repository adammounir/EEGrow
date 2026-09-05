"""Pourquoi `restored_probe` bat-il `acc_probe` sur certains datasets et pas d'autres ?

Post-hoc et assume comme tel : rien ici n'est pre-enregistre. Le but n'est pas de
produire un p mais de comprendre un resultat deja acquis (`dalloc_restored`, 531833-35),
et chaque hypothese est ecrite pour pouvoir ECHOUER. Trois des cinq echouent.

Ordre des sections, qui est aussi l'ordre dans lequel il faut les lire :

A. Le nul est-il MECANIQUE ? Si les deux regles designaient les memes donneurs, un delta
   nul ne dirait rien sur la croissance. Reponse : non, TVD 0.56-0.72, et a budget 3 /
   beta 8 les ensembles sont disjoints (Jaccard 0.000).

B. Y a-t-il seulement quelque chose a gagner ? L'oracle top-5 est le PLAFOND de toute
   regle de tri. Reponse : oui partout, +0.109 a +0.152, et physionetmi a le PLUS gros
   gisement tout en ne repondant pas. L'hypothese « rien a gagner » est refutee.

C. L'endpoint pre-enregistre a-t-il dilue un effet reel ? Oui pour cho2017 : la moitie de
   la grille (beta=1, gros budget) fait converger les deux allocations et n'apporte que
   des zeros. Dose-reponse sur la divergence : +0.043/TVD sur cho2017, +0.024 sur
   lee2019_mi, -0.053 sur physionetmi. Le clivage n'est donc PAS lee contre les autres.

D. Est-ce une affaire de FIABILITE ? Oui a l'interieur d'un dataset -- jeter des replicats
   detruit la regle, Spearman(ICC, capture) = +1.00 / +1.00 / +0.886. Non entre datasets :
   a fiabilite appariee (ICC ~ 0.43) cho2017 capture 0.314 et physionetmi 0.125. Et
   l'accuracy est BEAUCOUP plus fiable que la croissance partout (ICC(1,1) 0.96 vs 0.13
   sur cho2017) tout en etant une bien plus mauvaise regle : fiabilite != validite.

E. Ce qui reste, et qui explique cho2017 : l'accuracy n'echoue pas globalement
   (Spearman +0.414 sur tout le vivier) mais AU SOMMET, la ou l'allocation concentre.
   Les 7 sujets a accuracy >= 0.90 sont mediocres comme donneurs (rang 22.8/52) et
   `acc_probe` leur donne ses 5 places de top-5 pour chaque receveur. Les retirer fait
   passer sa capture de 0.123 a 0.427 et reduit l'avantage de la croissance de +0.330 a
   +0.076. L'avantage de cho2017 est donc de la ROBUSTESSE, pas de l'information
   complementaire -- et c'est une autre revendication, plus faible.

   Reserve a porter : « artefact » est une INFERENCE a partir d'une accuracy implausible
   en MI 2 classes, pas un fait documente. Les sujets que Cho et al. excluent (32, 46, 49)
   ne sont PAS ceux-la. La lecture prudente est : sur cho2017 les meilleurs performeurs
   within-session ne sont pas les meilleurs donneurs, qu'on l'explique par un artefact ou
   par une separabilite idiosyncrasique non transferable.

   lee2019_mi ne suit PAS ce motif : y retirer les hauts performeurs DEGRADE l'accuracy
   (0.389 -> 0.163), donc ses hauts performeurs sont de vrais bons donneurs. C'est le
   seul dataset ou l'avantage de la croissance survit au controle, et c'est aussi le seul
   qui etait significatif en pre-enregistre.

Usage::

    ./.venv/bin/python benchmarks/analysis/why_lee_responds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import donor_matrix as dm  # noqa: E402

RNG = np.random.default_rng(20260905)
K = 5                       # top-k : l'allocation concentre, elle n'exploite que la tete
SUSPECT_ACC = 0.90          # accuracy within-session implausible en MI 2 classes
DS = ["shin2017a", "cho2017", "dreyer2023a", "lee2019_mi", "liu2024", "physionetmi"]
END2END = ["cho2017", "lee2019_mi", "physionetmi"]


def truth(ds: str) -> pd.DataFrame:
    fr = pd.concat([pd.read_csv(f) for f in (A / f"dxr_{ds}").glob("*.csv")],
                   ignore_index=True)
    return (fr[fr["self"] == 0].groupby(["receiver", "donor"])["roc_auc"]
            .mean().reset_index())


def ranking(ds: str) -> pd.DataFrame:
    return pd.read_csv(A / f"ranking_{ds}.csv").set_index("subject")


def capture(t: pd.DataFrame, rk: pd.DataFrame, col: str,
            drop: tuple = (), score: pd.Series | None = None) -> float:
    """Fraction de l'oracle top-K captee par la regle, moyennee sur les receveurs.

    Denominateur = ce que le MEILLEUR top-K possible gagne sur l'allocation uniforme.
    Normaliser est indispensable : le gisement varie d'un facteur 1.4 entre datasets et
    comparer des gains bruts confondrait « bonne regle » et « dataset genereux ».
    """
    s = rk[col] if score is None else score
    keep = [x for x in s.dropna().index if x not in drop]
    num, den = [], []
    for _, g in t.groupby("receiver"):
        d = g[g.donor.isin(keep)]
        if len(d) < 2 * K:
            continue
        y = d.roc_auc.to_numpy(float)
        mu = y.mean()
        num.append(y[np.argsort(-s.loc[d.donor].to_numpy(float))[:K]].mean() - mu)
        den.append(np.sort(y)[-K:].mean() - mu)
    return float(np.mean(num) / np.mean(den))


def icc11(d: pd.DataFrame, col: str) -> tuple[float, float]:
    """ICC(1,1) -- fiabilite d'UN replicat. C'est la seule forme comparable entre
    datasets : l'ICC(1,k) monte mecaniquement avec k (Spearman-Brown), et les datasets
    n'ont pas le meme nombre de replicats (15 ici, 30 sur lee2019_mi, 45 sur shin2017a).
    Confondre les deux fait dire « lee est plus fiable » quand il a juste plus de fits.
    """
    d = d[["subject", col]].dropna()
    g = d.groupby("subject")[col]
    k = g.size().mean()
    msb, msw = g.mean().var(ddof=1) * k, g.var(ddof=1).mean()
    return float((msb - msw) / (msb + (k - 1) * msw)), float(k)


def probe_fits() -> pd.DataFrame:
    f = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz", low_memory=False)
    f["align_tag"] = f["align_tag"].fillna("none")
    f = f[(f["eval"] == dm.PROBE_EVAL) & (f.model == dm.PROBE_MODEL)
          & (f.align_tag == "none")]
    f["subject"] = pd.to_numeric(f["subject"], errors="coerce")
    return f[["dataset", "subject", "restored_epoch"]].dropna()


def main() -> int:
    T = {ds: truth(ds) for ds in DS}
    R = {ds: ranking(ds) for ds in DS}

    print("=" * 78)
    print("D. FIABILITE -- degradation controlee, tout le reste tenu fixe")
    print("=" * 78)
    fits = probe_fits()
    print(f"{'dataset':<13}{'m':>4}{'ICC(1,m)':>10}{'capture':>10}{'barre acc':>11}")
    for ds in END2END:
        g = fits[fits.dataset == ds]
        i1, ktot = icc11(g, "restored_epoch")
        bar = capture(T[ds], R[ds], "acc_probe")
        arr = {s: v["restored_epoch"].to_numpy(float) for s, v in g.groupby("subject")}
        for m in sorted({1, 2, 3, 5, 8, int(ktot)}):
            if m > ktot:
                continue
            caps = [capture(T[ds], R[ds], "restored_probe",
                            score=pd.Series({s: RNG.choice(v, m, replace=False).mean()
                                             for s, v in arr.items()}))
                    for _ in range(30)]
            icc_m = m * i1 / (1 + (m - 1) * i1)
            print(f"{ds:<13}{m:>4}{icc_m:>10.3f}{np.mean(caps):>10.3f}{bar:>11.3f}")
        print()

    print("=" * 78)
    print("E. OU l'accuracy echoue-t-elle : sur tout le vivier, ou au SOMMET ?")
    print("=" * 78)
    print(f"{'dataset':<13}{'n':>4}{'rho global acc':>16}{'capture acc':>13}"
          f"{'suspects':>10}{'/5 top-5':>10}")
    for ds in DS:
        t, rk = T[ds], R[ds]
        gv = t.groupby("donor")["roc_auc"].mean()
        i = gv.index.intersection(rk.index)
        rho = spearmanr(rk.loc[i, "acc_probe"], gv.loc[i]).statistic
        susp = set(rk.index[rk["acc_probe"] >= SUSPECT_ACC])
        hits = []
        for _, g in t.groupby("receiver"):
            d = g[g.donor.isin(rk.index)]
            if len(d) < 2 * K:
                continue
            top = d.donor.to_numpy()[
                np.argsort(-rk.loc[d.donor, "acc_probe"].to_numpy(float))[:K]]
            hits.append(sum(s in susp for s in top))
        print(f"{ds:<13}{len(i):>4}{rho:>+16.3f}{capture(t, rk, 'acc_probe'):>13.3f}"
              f"{len(susp):>10}{np.mean(hits) if hits else float('nan'):>10.2f}")

    print("\n  Retirer les sujets a accuracy >= 0.90 :")
    print(f"  {'dataset':<13}{'acc avant':>11}{'acc apres':>11}"
          f"{'ecart avant':>13}{'ecart apres':>13}")
    for ds in END2END:
        t, rk = T[ds], R[ds]
        susp = tuple(rk.index[rk["acc_probe"] >= SUSPECT_ACC])
        a0, r0 = capture(t, rk, "acc_probe"), capture(t, rk, "restored_probe")
        a1 = capture(t, rk, "acc_probe", susp)
        r1 = capture(t, rk, "restored_probe", susp)
        print(f"  {ds:<13}{a0:>11.3f}{a1:>11.3f}{r0 - a0:>+13.3f}{r1 - a1:>+13.3f}")

    print("\n  Lecture : un ecart qui s'effondre = avantage de ROBUSTESSE (cho2017).")
    print("  Un ecart qui tient = information complementaire (lee2019_mi).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
