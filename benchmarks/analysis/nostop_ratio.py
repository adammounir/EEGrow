"""Plancher de valeurs singulieres : ratio 0.10 (defaut) contre 1e-6 (quasi desactive).

L'hypothese testee : en abaissant le plancher, la croissance garde plus de directions
candidates, atteint sa largeur cible, et le modele apprend mieux.

Choix des bras. Le plancher vit dans `MIN_SINGULAR_RATIO` -- c'est un seuil RELATIF sur
le spectre des candidats, il n'existe pas de seuil absolu 1e-5 dans ce code. Sur
`grow_deep` (w2=8) il est inerte par construction : le fan-out fournit exactement 8
candidats pour 8 slots, il ne coupe rien dans 100 % des etapes, et les deux bras
seraient bit-identiques (cf. gromo-candidate-rank-fanout). L'axe n'existe donc que sur
`grow_shallow` (coupe dans 71 % des etapes) et `grow_deep_w32`.

Unite d'analyse = le SUJET : moyenne des sessions puis des graines avant de comparer.
IC = bootstrap APPARIE sur les sujets, le meme sujet portant les deux ratios.

Deux etages, et il faut lire les deux dans l'ordre :
  1. MECANISME -- le plancher mord-il ? (largeur finale, cible atteinte)
  2. ENDPOINT  -- l'accuracy suit-elle ?
Un mecanisme qui marche sans effet sur l'endpoint n'est pas un echec de mesure : c'est
le resultat, et il dit que la largeur bloquee n'etait pas de la largeur utile.
"""
import glob, json, os
import numpy as np, pandas as pd

ROOT = "/scratch/amounir/eegrow_budget/benchmarks/results_nostop"
RATIOS = (("r010", 0.10), ("r1em6", 1e-6))
MODELS = ("grow_shallow", "grow_deep_w32")
EXPECTED_ROWS = 18   # 9 sujets x 2 sessions

frames = []
for ratio_dir, ratio in RATIOS:
    for f in sorted(glob.glob(os.path.join(ROOT, ratio_dir,
                                           "within_session/bnci2014_001/*.csv"))):
        df = pd.read_csv(f)
        df["ratio"] = ratio
        df["src"] = f"{ratio_dir}/{os.path.basename(f)}"
        frames.append(df)
d = pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------- integrite
print("=" * 74)
print("INTEGRITE")
print("=" * 74)
sizes = d.groupby("src").size()
print(f"cellules : {len(sizes)}/8 attendues")
bad = sizes[sizes != EXPECTED_ROWS]
print("lignes/cellule : "
      + (f"toutes a {EXPECTED_ROWS} OK" if bad.empty else f"ANOMALIE\n{bad.to_string()}"))
print("sha      :", d.eegrow_sha.unique())
print("device   :", d.device.unique())
print("sujets   :", sorted(d.subject.unique()))
print("sessions :", sorted(d.session.unique()))
print("graines  :", sorted(d.seed.unique()))
missing = [f"{m}/r={r}/s{s}" for m in MODELS for _, r in RATIOS
           for s in sorted(d.seed.unique())
           if not len(d[(d.model == m) & (d.ratio == r) & (d.seed == s)])]
print("manquantes :", missing or "aucune")

# ------------------------------------------------------------ 1. mecanisme
print("\n" + "=" * 74)
print("1. MECANISME -- le plancher mord-il vraiment ?")
print("=" * 74)
print(f"{'bras':<15}{'ratio':<7}{'fits':>5}{'width_end moy':>15}{'cible atteinte':>16}"
      f"{'restored_epoch med':>21}")
for m in MODELS:
    for ratio_dir, ratio in RATIOS:
        we, re_, hit, n = [], [], 0, 0
        pat = os.path.join(ROOT, ratio_dir, "within_session/bnci2014_001",
                           f"{m}__*__fits.jsonl")
        for f in glob.glob(pat):
            for line in open(f):
                o = json.loads(line)
                n += 1
                we.append(o["width_end"])
                re_.append(o["restored_epoch"])
                hit += o["width_end"] >= o["target_width"]
        print(f"{m:<15}{ratio_dir:<7}{n:>5}{np.mean(we):>15.2f}"
              f"{f'{hit}/{n}':>16}{np.median(re_):>21.1f}")
print("\nSi la cible est atteinte bien plus souvent a 1e-6, l'intervention fait ce qu'on")
print("lui demande et l'axe est vivant : ce qui suit mesure un effet reel, pas un no-op.")


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
    print(f"{label:<30} {x.mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  n={n}  "
          f"sd={sd:.4f}  MDE={mde:.4f}  {verdict}")
    return x.mean(), lo, hi, mde


# -------------------------------------------------------------- 2. endpoint
print("\n" + "=" * 74)
print("2. ENDPOINT -- accuracy, delta = (ratio 1e-6) moins (ratio 0.10), apparie sujet")
print("=" * 74)
bounds = {}
for m in MODELS:
    sub = d[d.model == m]
    print(f"\n---------------- {m} ----------------")
    print(sub.groupby("ratio").score.agg(["mean", "std", "count"]).round(4).to_string())
    print("\npar ratio x graine :")
    print(sub.groupby(["ratio", "seed"]).score.mean().unstack(1).round(4).to_string())
    per = (sub.groupby(["ratio", "subject", "seed"]).score.mean()
              .groupby(["ratio", "subject"]).mean().unstack(0))
    per["delta"] = per[1e-6] - per[0.10]
    print("\npar sujet :")
    print(per.round(4).to_string())
    print()
    bounds[m] = paired(per["delta"].values, f"delta {m}")
    print(f"{'':30} 1e-6 gagne sur {(per['delta'] > 0).sum()}/{len(per)} sujets")

# ---------------------------------------------------------------- lecture
print("\n" + "=" * 74)
print("LECTURE")
print("=" * 74)
for m in MODELS:
    mu, lo, hi, mde = bounds[m]
    if lo > 0 or hi < 0:
        print(f"  {m:<15} effet significatif {mu:+.4f}.")
    else:
        # Non significatif : la question utile n'est pas "est-ce nul ?" mais "de combien
        # peut-on encore esperer ?". C'est la borne HAUTE de l'IC qui repond, et elle
        # dit quelque chose meme quand |effet| < MDE.
        print(f"  {m:<15} non significatif. Le gain compatible avec les donnees est")
        print(f"  {'':15} borne a {hi:+.4f} au plus (borne haute de l'IC95).")
        if abs(mu) < mde:
            print(f"  {'':15} |effet| {abs(mu):.4f} < MDE {mde:.4f} : on ne peut pas")
            print(f"  {'':15} declarer un NUL, seulement exclure tout gain > {hi:+.4f}.")
print("\n  Le plancher mord (etage 1) sans deplacer l'accuracy (etage 2) : les directions")
print("  qu'il refusait n'etaient pas des directions utiles. L'abstention ne coute rien.")
print("\nprior 1 sujet (grow_shallow.yaml) : ratio 0.10 -> 0.690, ratio 0.01 -> 0.552.")
print("Ce prior annoncait une CHUTE massive a plancher bas ; sur 9 sujets elle n'existe")
print("pas non plus. Un seul sujet ne mesurait que du bruit inter-sujets (sd ~ 0.17).")
