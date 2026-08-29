"""Test de falsification du mecanisme « amplitude » + gate tout-contre-rien.

Hypothese testee
----------------
Sur le pool, l'interaction croissance x alignement vaut +1.29 pp contre `none` mais est
nulle contre `scale` : elle porte donc sur la normalisation d'amplitude, pas sur le
blanchiment. Mecanisme propose : la line search de `grow_step` compare des magnitudes de
gradient de part et d'autre d'une jonction ; dans un pool ou un dataset domine en
amplitude, c'est le gain de l'amplificateur qui decide *ou* la capacite est allouee.

Prediction falsifiable : `within` n'a qu'un seul dataset, donc un seul amplificateur.
L'interaction doit **disparaitre**. Si elle survit sur `within`, le mecanisme est faux.

Unite d'analyse = le sujet tenu a l'ecart. Les seeds partagent donnees et decoupe : elles
sont moyennees *dans* le sujet avant toute statistique.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
B = 20000
RNG = np.random.default_rng(0)

df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "*.csv"))])

g = df.groupby(["arm", "model", "align", "seed"]).subject.nunique()
assert (g == 52).all(), f"cellules incompletes :\n{g[g != 52]}"
print(f"{len(g)} cellules x 52 sujets, seeds {sorted(df.seed.unique())}")

per = (
    df.groupby(["arm", "model", "align", "subject"]).score.mean()
    .unstack(["arm", "model", "align"])
)


def stat(d, label):
    """d = vecteur de differences appariees par sujet, en points de proportion."""
    d = np.asarray(d.dropna() if hasattr(d, "dropna") else d, dtype=float)
    n = len(d)
    boot = RNG.choice(d, size=(B, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_t = stats.ttest_1samp(d, 0).pvalue
    p_w = stats.wilcoxon(d).pvalue
    mde = d.std(ddof=1) * (stats.t.ppf(.975, n - 1) + stats.t.ppf(.80, n - 1)) / np.sqrt(n)
    return dict(contraste=label, delta_pp=100 * d.mean(), ic_lo=100 * lo, ic_hi=100 * hi,
                p_t=p_t, p_wilcox=p_w, MDE_pp=100 * mde, gagnants=f"{int((d > 0).sum())}/{n}")


def inter(arm, base):
    """(grow EA - grow base) - (bd EA - bd base), apparie par sujet."""
    return (per[(arm, "grow_shallow", "euclidean")] - per[(arm, "grow_shallow", base)]) - (
        per[(arm, "bd_shallow", "euclidean")] - per[(arm, "bd_shallow", base)]
    )


print("\n=== scores moyens (52 sujets) ===")
print((per.mean() * 100).round(2).unstack("align").to_string())

rows = [stat(inter(a, b), f"interaction croissance x EA @ {a}, base {b}")
        for a in ["pooled", "within"] for b in ["none", "scale"]]
# le test decisif : l'interaction est-elle plus faible sur within que sur pooled ?
rows.append(stat(inter("pooled", "none") - inter("within", "none"),
                 "interaction pooled - interaction within (base none)"))
res = pd.DataFrame(rows)

# Holm sur la famille des 5
order = np.argsort(res.p_t.values)
holm, run = np.empty(len(res)), 0.0
for rank, i in enumerate(order):
    run = max(run, (len(res) - rank) * res.p_t.values[i])
    holm[i] = min(run, 1.0)
res["holm"] = holm
print("\n=== test de falsification ===")
print(res.round(4).to_string(index=False))

print("\n=== gate tout-allume contre rien (debloque par 504342) ===")
gate = [
    stat(per[("pooled", "grow_shallow", "euclidean")] - per[("within", "bd_shallow", "none")],
         "grow+pooled+EA  -  bd+within+aucun alignement"),
    stat(per[("pooled", "grow_shallow", "euclidean")] - per[("within", "bd_shallow", "euclidean")],
         "grow+pooled+EA  -  bd+within+EA  (ancien gate, baseline deja alignee)"),
    stat(per[("within", "bd_shallow", "euclidean")] - per[("within", "bd_shallow", "none")],
         "EA seul @ within/fixe"),
    stat(per[("within", "grow_shallow", "euclidean")] - per[("within", "bd_shallow", "euclidean")],
         "croissance seule @ within/EA"),
]
print(pd.DataFrame(gate).round(4).to_string(index=False))
