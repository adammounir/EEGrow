"""Claim 3 : le verdict. Écrit AVANT les données, exprès.

Ce fichier existe avant que le premier CSV soit tombé, et c'est délibéré : les seuils,
la statistique de test et la règle de décision sont fixés avant de voir les scores. Sans
ça, « quelle règle gagne » devient une question à laquelle on répond en regardant, et un
protocole à cinq règles offre assez de comparaisons pour que quelque chose ressorte par
hasard (cf. [[unit-of-analysis-subject]] : les p appariés se gonflent vite quand on
multiplie les contrastes sur les mêmes sujets).

LA QUANTITÉ TESTÉE
------------------
Pour chaque sujet de test i et chaque règle r, on a un score (moyenne sur les seeds ou
sur les tirages, selon la règle). La quantité qui décide est la DIFFÉRENCE APPARIÉE
``s_i(r) - s_i(random)``, moyennée sur les 109 sujets, avec un IC bootstrap sur les
sujets. L'appariement est ce qui donne la puissance : la variance inter-sujets de
physionetmi est énorme (des sujets à 0.33 et d'autres à 0.91 d'accuracy sonde) et elle
disparaît entièrement dans la différence.

LA RÈGLE DE DÉCISION, POSÉE MAINTENANT
--------------------------------------
Claim 3 passe si les TROIS conditions tiennent ensemble :

  (a) `resid_top` ou `params_top` bat `random` -- IC de la différence appariée
      strictement au-dessus de 0 après correction de Holm sur les 4 contrastes vs
      random ;
  (b) le contrôle négatif s'inverse -- `params_bottom` ne bat pas random, et son point
      estimé est du signe opposé. Un effet qui ne change pas de signe quand on inverse
      le tri n'est pas un effet de tri ;
  (c) la règle gagnante n'est pas battue par `acc_top`. Ce n'est PAS « acc_top est non
      significatif » -- deux tests séparés ne font pas une comparaison -- mais l'IC de
      la différence directe ``s_i(params) - s_i(acc)``, appariée elle aussi.

Si (a) tient mais pas (c), le résultat honnête est « sélectionner aide, mais la taille
n'apporte rien de plus que l'accuracy » : c'est un résultat d'atelier, pas un ICLR, et
il faut l'écrire tel quel.

CE QUE LA PUISSANCE PERMET DE DÉTECTER
--------------------------------------
Le MDE est calculé et imprimé à côté de chaque contraste, sur l'écart-type OBSERVÉ des
différences appariées. Un contraste non significatif dont le MDE dépasse l'effet
plausible n'est pas un nul : c'est un manque de puissance, et il est étiqueté comme tel
([[underpowered-not-null]]).

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_select_analysis.py \
        --dsel benchmarks/analysis/dsel_physionetmi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))

METRIC = "roc_auc"
N_BOOT = 10000
RNG = np.random.default_rng(20260902)
RULES = ["params_top", "resid_top", "acc_top", "params_bottom"]


def load(dsel: Path) -> pd.DataFrame:
    files = sorted(dsel.glob("f*__*__d*__seed*.csv"))
    if not files:
        raise SystemExit(f"no cell CSV under {dsel}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(files)} cellules, {len(d)} lignes, "
          f"{d.test_subject.nunique()} sujets de test, règles {sorted(d.rule.unique())}")
    return d


def per_subject(d: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par sujet de test, une colonne par règle.

    La moyenne interne (sur les seeds pour les règles déterministes, sur les tirages
    pour `random`) est prise AVANT l'appariement, jamais après : ce qui doit être
    apparié est le sujet, et un sujet dont on garderait les seeds séparées entrerait
    trois fois dans un bootstrap qui croirait avoir 3n observations indépendantes.

    `random` moyenne sur ses R tirages, ce qui en fait l'espérance de la règle aléatoire
    plutôt qu'un tirage particulier -- c'est la bonne ligne de base : la question n'est
    pas « bat-on ce tirage-ci » mais « bat-on l'aléatoire en moyenne ».
    """
    w = (d.groupby(["test_subject", "rule"])[METRIC].mean()
         .unstack("rule").sort_index())
    n_before = len(w)
    w = w.dropna()
    if len(w) < n_before:
        print(f"  {n_before - len(w)} sujets écartés : toutes les règles ne les ont pas "
              "encore couverts (array incomplet)")
    return w


def boot_ci(x: np.ndarray, reps: int = N_BOOT) -> tuple[float, float]:
    n = len(x)
    idx = RNG.integers(0, n, size=(reps, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni : on teste 4 règles contre le même random, sur les mêmes sujets.

    Bonferroni simple serait conservateur au point de rendre le protocole aveugle ;
    Holm garde le contrôle du taux d'erreur familial sans payer ce prix. La correction
    porte sur la famille « une règle bat-elle random », pas sur le contraste direct
    params/acc qui répond à une autre question et est déclaré à part.
    """
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * pvals[k])
        adj[k] = min(running, 1.0)
    return adj


def contrast(w: pd.DataFrame, a: str, b: str) -> dict:
    """Une différence appariée a - b, avec IC bootstrap, p de permutation et MDE.

    Le p vient d'une permutation de SIGNE (on retourne au hasard le signe de chaque
    différence), pas d'un t-test : les scores sont des ROC-AUC bornés, les différences
    sont loin d'être gaussiennes sur 45 essais de test, et la permutation ne suppose que
    la symétrie sous l'hypothèse nulle.
    """
    x = (w[a] - w[b]).to_numpy(float)
    n = len(x)
    lo, hi = boot_ci(x)
    signs = RNG.choice([-1.0, 1.0], size=(N_BOOT, n))
    null = (signs * np.abs(x)).mean(axis=1)
    p = float((np.abs(null) >= abs(x.mean())).mean())
    return {"pair": f"{a} - {b}", "delta": float(x.mean()), "lo": lo, "hi": hi,
            "p": p, "n": n, "mde": float(2.8 * x.std(ddof=1) / np.sqrt(n))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dsel", required=True, type=Path)
    a = p.parse_args(argv)

    d = load(a.dsel)
    w = per_subject(d)
    rules = [r for r in RULES if r in w.columns]
    if "random" not in w.columns:
        raise SystemExit("pas de règle `random` : il n'y a pas de ligne de base")

    print("\n" + "=" * 78)
    print("SCORES PAR RÈGLE (moyenne sur les sujets de test, pour mémoire)")
    print("=" * 78)
    chance = float(d["chance"].iloc[0])
    for r in ["random"] + rules:
        v = w[r].to_numpy(float)
        print(f"  {r:15s} {v.mean():.4f}  (±{v.std(ddof=1)/np.sqrt(len(v)):.4f} sem, "
              f"chance {chance:.2f})")
    print("  Ces moyennes ne décident rien : elles ne sont pas appariées et la variance")
    print("  inter-sujets de physionetmi les noie. Le tableau qui décide est en dessous.")

    print("\n" + "=" * 78)
    print("(a) CHAQUE RÈGLE CONTRE L'ALÉATOIRE — différences appariées par sujet")
    print("=" * 78)
    res = {r: contrast(w, r, "random") for r in rules}
    adj = holm({r: res[r]["p"] for r in rules})
    for r in rules:
        c = res[r]
        star = "  *" if (c["lo"] > 0 or c["hi"] < 0) and adj[r] < 0.05 else ""
        print(f"  {c['pair']:28s} {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
              f"p={c['p']:.4f}  Holm={adj[r]:.4f}  MDE={c['mde']:.4f}{star}")
        if not star and c["mde"] > abs(c["delta"]) * 2:
            print(f"    → non significatif AVEC un MDE de {c['mde']:.4f} : "
                  "sous-puissant, pas nul.")

    print("\n" + "=" * 78)
    print("(b) CONTRÔLE NÉGATIF — le tri inversé doit perdre")
    print("=" * 78)
    if "params_bottom" in res and "params_top" in res:
        top, bot = res["params_top"]["delta"], res["params_bottom"]["delta"]
        ok = top > 0 > bot
        print(f"  params_top vs random  {top:+.4f}   params_bottom vs random {bot:+.4f}")
        print(f"  → signes {'OPPOSÉS, le contrôle passe' if ok else 'NON opposés'}. "
              + ("" if ok else "Un effet qui ne s'inverse pas avec le tri n'est pas un "
                               "effet de tri : chercher ce qui bouge en même temps "
                               "(nombre d'essais, sessions, classes)."))
        print("  " + str(contrast(w, "params_top", "params_bottom")))

    print("\n" + "=" * 78)
    print("(c) LA COMPARAISON QUI FAIT LE PAPIER — la taille contre l'accuracy")
    print("=" * 78)
    for r in ["params_top", "resid_top"]:
        if r in w.columns and "acc_top" in w.columns:
            c = contrast(w, r, "acc_top")
            star = "  *" if c["lo"] > 0 or c["hi"] < 0 else ""
            print(f"  {c['pair']:28s} {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
                  f"  p={c['p']:.4f}  MDE={c['mde']:.4f}{star}")
    print("  Un IC qui contient 0 ici veut dire « indiscernables », PAS « la taille "
          "gagne »\n  ni « l'accuracy gagne ». C'est l'erreur qui a failli être commise "
          "sur claim 2.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    win = [r for r in ("resid_top", "params_top")
           if r in res and res[r]["lo"] > 0 and adj[r] < 0.05]
    neg_ok = ("params_bottom" not in res) or res["params_bottom"]["delta"] < 0
    if win and neg_ok:
        print(f"  (a) OUI ({', '.join(win)} bat random)  (b) contrôle négatif OK")
        print("  Reste (c) : lire l'IC contre acc_top ci-dessus avant d'écrire quoi que "
              "ce soit.")
    elif win:
        print(f"  (a) OUI ({', '.join(win)}) mais (b) le contrôle négatif NE S'INVERSE "
              "PAS.\n  Ne pas publier : l'effet n'est pas attribuable au tri.")
    else:
        print("  (a) NON. Aucune règle fondée sur #params ne bat l'aléatoire à ce n.")
        print("  Vérifier le MDE de chaque ligne avant de conclure au nul.")
    w.to_csv(a.dsel.parent / f"{a.dsel.name}_per_subject.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
