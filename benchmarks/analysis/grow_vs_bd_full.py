"""grow_deep vs bd_deep4 a budget plein -- plan COMPLET 2 modeles x 3 graines x 2 alignements.

La question ouverte : l'inversion mesuree sur align=none tient-elle sous alignement
euclidien, ou est-ce l'alignement qui portait le gain publie ?

Unite d'analyse = le SUJET partout. On moyenne les sessions puis les graines avant de
comparer : sessions et graines d'un meme sujet sont correlees, les traiter comme
independantes gonfle les p (cf. unit-of-analysis-subject). Les IC sont des bootstrap
APPARIES sur les sujets -- le meme sujet porte les deux bras, jeter l'appariement
reviendrait a payer la variance inter-sujets (0.15) pour mesurer un effet de 0.07.
"""
import glob, os
import numpy as np, pandas as pd

D = "/scratch/amounir/eegrow_budget/benchmarks/results_growvsbd/within_session/bnci2014_001"
EXPECTED_ROWS = 18   # 9 sujets x 2 sessions

frames = []
for f in sorted(glob.glob(os.path.join(D, "*.csv"))):
    df = pd.read_csv(f)
    df["src"] = os.path.basename(f)
    frames.append(df)
d = pd.concat(frames, ignore_index=True)
if "align" not in d.columns:
    d["align"] = np.where(d.src.str.contains("easubject"), "euclidean", "none")

# ---------------------------------------------------------------- integrite
print("=" * 70)
print("INTEGRITE")
print("=" * 70)
sizes = d.groupby("src").size()
print(f"cellules : {len(sizes)}/12 attendues")
bad = sizes[sizes != EXPECTED_ROWS]
print("lignes par cellule : "
      + (f"toutes a {EXPECTED_ROWS} OK" if bad.empty else f"ANOMALIE\n{bad.to_string()}"))
print("sha        :", d.eegrow_sha.unique())
print("device     :", d.device.unique())
print("sujets     :", sorted(d.subject.unique()))
print("sessions   :", sorted(d.session.unique()))
missing = [f"{m}/{al}/s{s}" for m in ("bd_deep4", "grow_deep")
           for al in ("none", "euclidean") for s in (1, 2, 3)
           if not len(d[(d.model == m) & (d["align"] == al) & (d.seed == s)])]
print("manquantes :", missing or "aucune")


def subject_table(sub):
    """une ligne par sujet, une colonne par bras : moyenne sessions puis graines."""
    return (sub.groupby(["model", "subject", "seed"]).score.mean()
               .groupby(["model", "subject"]).mean().unstack(0))


def paired(x, label, n_boot=20000, seed=0):
    """bootstrap apparie sur les sujets + MDE. x = vecteur des deltas par sujet."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = x.std(ddof=1)
    mde = 2.9 * sd / np.sqrt(n)
    verdict = "significatif" if (lo > 0 or hi < 0) else "NON significatif"
    print(f"{label:<34} {x.mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  n={n}  "
          f"sd={sd:.4f}  MDE={mde:.4f}  {verdict}")
    if not (lo > 0 or hi < 0) and abs(x.mean()) < mde:
        print(f"{'':34}   |effet| < MDE : mesure vide, pas un nul.")
    return x.mean(), lo, hi, mde


# ------------------------------------------------- contraste par alignement
print("\n" + "=" * 70)
print("CONTRASTE grow_deep - bd_deep4, PAR ALIGNEMENT")
print("=" * 70)
tables = {}
for al in ("none", "euclidean"):
    sub = d[d["align"] == al]
    if not len(sub):
        continue
    print(f"\n---------------- align = {al} ----------------")
    print(sub.groupby("model").score.agg(["mean", "std", "count"]).round(4).to_string())
    print("\npar bras x graine :")
    print(sub.groupby(["model", "seed"]).score.mean().unstack(1).round(4).to_string())
    per = subject_table(sub)
    if not {"grow_deep", "bd_deep4"} <= set(per.columns):
        print("bras non apparie -- rien de conclusif.")
        continue
    per["delta"] = per["grow_deep"] - per["bd_deep4"]
    tables[al] = per
    print("\npar sujet :")
    print(per.round(4).to_string())
    print()
    paired(per["delta"].values, f"delta ({al})")
    print(f"{'':34} grow_deep gagne sur "
          f"{(per['delta'] > 0).sum()}/{len(per)} sujets")

# ------------------------------------------- l'alignement portait-il le gain ?
print("\n" + "=" * 70)
print("EFFET DE L'ALIGNEMENT EUCLIDIEN, PAR BRAS  (euclidean - none, apparie sujet)")
print("=" * 70)
if {"none", "euclidean"} <= set(tables):
    arm = {}
    for m in ("bd_deep4", "grow_deep"):
        gain = tables["euclidean"][m] - tables["none"][m]
        arm[m] = paired(gain.values, f"align gain sur {m}")
    print()
    # interaction : est-ce que l'alignement change le CONTRASTE ?
    inter = tables["euclidean"]["delta"] - tables["none"]["delta"]
    i_mean, i_lo, i_hi, i_mde = paired(inter.values,
                                       "interaction (delta_eucl - delta_none)")
    print("\nLECTURE")
    if i_lo > 0 or i_hi < 0:
        print("  Interaction significative : l'alignement ne deplace PAS les deux bras du")
        print("  meme montant, il change le contraste lui-meme. Reste a dire QUI bouge --")
        print("  c'est la decomposition par bras ci-dessus qui le dit, et il faut la lire")
        print("  avant de conclure quoi que ce soit sur la croissance :")
        for m in ("bd_deep4", "grow_deep"):
            mu, lo, hi, mde = arm[m]
            if lo > 0 or hi < 0:
                sens = "monte" if mu > 0 else "descend"
                print(f"    {m:<10} {sens} de {abs(mu):.4f} (significatif)")
            elif abs(mu) < mde:
                print(f"    {m:<10} {mu:+.4f} sous sa MDE {mde:.4f} -- mesure vide")
            else:
                print(f"    {m:<10} {mu:+.4f} non significatif")
        print("  Un contraste qui remonte parce que la BASELINE se degrade n'est pas un")
        print("  gain de la croissance : c'est une perte de la reference.")
    else:
        print("  Interaction non significative : l'alignement deplace les deux bras du")
        print("  meme montant, il ne peut donc pas expliquer un ECART entre bras.")
        if abs(i_mean) < i_mde:
            print(f"  Attention : |{i_mean:+.4f}| < MDE {i_mde:.4f}, mesure vide et non nul.")
else:
    print("plan incomplet : les deux alignements ne sont pas encore apparies.")

print("\nrappel publie (budget coupe) : bd_deep4 0.275  grow_deep 0.408  delta +0.133")
print("                               0.275 est SOUS la chance unilaterale 95% = 0.295")
