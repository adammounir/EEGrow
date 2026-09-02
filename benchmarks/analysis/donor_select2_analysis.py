"""Claim 3, v2 : le verdict sur (c). Écrit AVANT les données, comme le premier.

POURQUOI UN DEUXIÈME FICHIER PLUTÔT QU'UNE RALLONGE AU PREMIER
--------------------------------------------------------------
``donor_select_analysis.py`` reste tel quel, avec sa règle de décision d'origine et son
glob d'origine. Il a répondu à (a) et (b) et ses chiffres sont publiés ; le modifier
après coup pour qu'il « tombe juste » sur (c) serait exactement la manœuvre que la
pré-registration existe pour empêcher. Ce fichier-ci est une pré-registration NOUVELLE,
pour un protocole NOUVEAU, avec sa propre unité d'analyse.

CE QUE V1 A ÉTABLI, ET QU'ON NE REJUGE PAS
------------------------------------------
(a) sélectionner ses donneurs bat l'aléatoire : `resid_top` +0.0390, 4 plis sur 4
concordants, IC bootstrap-cluster [+0.0361, +0.0435]. (b) le contrôle négatif s'inverse :
`params_bottom` -0.0518, 4/4. Ces deux-là ne dépendaient pas du défaut de v1 et ils
restent acquis. Ici ils servent de GARDE-FOU, pas de résultat : s'ils ne se répliquent
pas, c'est v2 qui est cassé, et rien d'autre dans ce fichier n'est interprétable.

LE DÉFAUT DE V1, ET CE QUE V2 CHANGE
------------------------------------
En v1 une règle déterministe ne produisait qu'UN pool par pli : le contraste entre deux
règles avait 4 réplicats, pas 109, et le bootstrap sur les sujets ne le voyait pas.
V2 tire M candidats par (pli, réplicat), partagés par toutes les règles. L'unité de
rééchantillonnage devient donc le **réplicat**, il y en a F x R, et deux règles d'un même
réplicat sont comparées sur le même jeu de sujets disponibles.

L'UNITÉ D'ANALYSE, ÉNONCÉE SANS AMBIGUÏTÉ
-----------------------------------------
Les 109 sujets sont le dataset ENTIER, pas un échantillon tiré d'une population plus
grande. Pour la question posée ici -- « sur physionetmi, la règle A bat-elle la règle B
en espérance sur le tirage du pool ? » -- ils sont donc FIXES, et le seul hasard qui
reste est celui du tirage des candidats. On rééchantillonne les réplicats, stratifiés
par pli (les plis sont une partition fixe du dataset, pas un tirage).

Ce que ce protocole ne peut PAS conclure, quel que soit son n : que le résultat vaut pour
d'autres sujets ou d'autres datasets. Ça demande la réplication sur lee2019_mi/cho2017 et
il faut l'écrire ainsi plutôt que de laisser le lecteur généraliser tout seul.

LA RÈGLE DE DÉCISION, POSÉE MAINTENANT
--------------------------------------
GARDE-FOU G. À chaque K : `resid_top` bat `random` et `params_bottom` le perd. Si G
tombe, on s'arrête et on cherche la panne -- on ne lit pas (c).

ENDPOINT PRIMAIRE P. `resid_top` - `acc_top` **à K=5**, un seul test, IC bootstrap sur
les réplicats. K=5 est choisi AVANT les données et sur un critère qui n'en contient
aucune : le dry-run donne un recouvrement `resid_top` ∩ `acc_top` de 7 % à K=5 contre
29 % à K=10 et 56 % à K=20. C'est le K où les deux règles choisissent des sujets
quasi disjoints, donc le seul où le contraste a de la prise. Prendre la moyenne sur les
K diluerait le signal avec deux régimes où le protocole est aveugle par construction.

SECONDAIRE S. Le même contraste à K=10 et K=20, Holm sur les 3 K. Descriptif : il dit
OÙ la prise se perd, il ne rejuge pas P.

PRÉDICTION T -- et elle peut me donner tort. J'ai expliqué l'échec de v1 par la
saturation : à K=20 sur un vivier de 82 (24 %), tous les critères raisonnables ramassent
les mêmes sujets. Cette explication prédit que le gain de `resid_top` sur `random`
DÉCROÎT quand K/M grandit. On teste `delta(K=5) - delta(K=20) > 0`. Si ce n'est pas le
cas, mon explication de v1 est fausse et il faut le dire au lieu de garder le balayage
en K comme une illustration.

ÉQUIVALENCE. Si P n'est pas significatif, on ne conclut PAS « c'est pareil » -- c'est
l'erreur que j'ai commise le matin du 02/09. On lance un TOST à ±25 % de l'effet de
sélection mesuré au même K, et l'équivalence n'est déclarée que s'il passe. Sinon le
verdict est INDÉTERMINÉ, avec le nombre de réplicats qu'il faudrait, calculé et imprimé.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_select2_analysis.py \
        --dsel benchmarks/analysis/dsel2_physionetmi
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
N_BOOT = 20000
RNG = np.random.default_rng(20260902)

PRIMARY_K = 5                 # déclaré avant les données, cf. docstring
PRIMARY = ("resid_top", "acc_top")
EQUIV_MARGIN = 0.25           # fraction de l'effet de sélection
RULES = ["params_top", "resid_top", "acc_top", "params_bottom"]


def load(dsel: Path) -> pd.DataFrame:
    """Les cellules v2 seulement. Le glob est volontairement incompatible avec v1.

    Un CSV v1 s'appelle ``f0__resid_top__d0__seed0.csv`` et n'a ni ``r<rep>`` ni
    ``k<K>`` : il ne peut donc pas entrer ici par accident. Deux protocoles dont l'unité
    d'analyse diffère ne doivent jamais se retrouver dans le même DataFrame.
    """
    files = sorted(dsel.glob("f*__r*__k*__*__seed*.csv"))
    if not files:
        raise SystemExit(f"aucune cellule v2 sous {dsel}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(files)} cellules, {len(d)} lignes, {d.test_subject.nunique()} sujets, "
          f"K={sorted(d.k.unique())}, {d.rep.nunique()} réplicats x {d.fold.nunique()} "
          f"plis = {d.groupby(['fold', 'rep']).ngroups} unités, "
          f"règles {sorted(d.rule.unique())}")

    # Le partage du jeu de candidats est CE QUI REND L'APPARIEMENT VALIDE. On le vérifie
    # au lieu de le supposer : si deux règles d'un même réplicat n'ont pas vu les mêmes
    # candidats, la différence mesurée contient « qui était disponible » et le contraste
    # ne veut plus rien dire.
    bad = (d.groupby(["fold", "rep"])["candidates"].nunique() > 1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} réplicats où les règles n'ont pas vu les "
                         "mêmes candidats : appariement invalide, ne pas analyser.")
    return d


def cells(d: pd.DataFrame, k: int) -> pd.DataFrame:
    """Un score par (pli, réplicat, règle) à K fixé : la matrice qu'on rééchantillonne.

    La moyenne sur les sujets de test d'un pli est prise ICI, avant tout appariement.
    Deux règles d'un même réplicat scorent exactement les mêmes sujets, donc la
    différence de leurs moyennes est déjà la moyenne de leurs différences appariées :
    la variance inter-sujets, énorme sur physionetmi, disparaît sans qu'on ait à
    manipuler les sujets comme des unités -- ce qu'ils ne sont pas ici.
    """
    w = (d[d.k == k].groupby(["fold", "rep", "rule"])[METRIC].mean()
         .unstack("rule").sort_index())
    n_before = len(w)
    w = w.dropna()
    if len(w) < n_before:
        print(f"  {n_before - len(w)} réplicats écartés : toutes les règles ne les "
              "couvrent pas encore (array incomplet)")
    return w


def boot_clusters(delta: pd.Series, reps: int = N_BOOT) -> np.ndarray:
    """Bootstrap des réplicats, STRATIFIÉ PAR PLI.

    Les plis sont une partition fixe du dataset -- chaque sujet est testé exactement une
    fois -- et non un tirage. Les rééchantillonner reviendrait à traiter les sujets comme
    un échantillon, ce que la docstring exclut explicitement. On rééchantillonne donc les
    réplicats à l'intérieur de chaque pli, puis on moyenne les plis à poids égaux.
    """
    per_fold = [g.to_numpy(float) for _, g in delta.groupby(level="fold")]
    out = np.zeros(reps)
    for v in per_fold:
        idx = RNG.integers(0, len(v), size=(reps, len(v)))
        out += v[idx].mean(axis=1)
    return out / len(per_fold)


def contrast(w: pd.DataFrame, a: str, b: str) -> dict:
    """Différence a - b au niveau RÉPLICAT, avec IC bootstrap, p de permutation, MDE."""
    delta = w[a] - w[b]
    x = delta.to_numpy(float)
    n = len(x)
    boot = boot_clusters(delta)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # Permutation de SIGNE au niveau du réplicat, pas du sujet : sous l'hypothèse nulle
    # « les deux règles se valent », c'est le tirage du pool qui décide du signe, et il
    # est échangeable d'un réplicat à l'autre. Permuter des sujets supposerait leur
    # indépendance, qui est fausse ici -- ils partagent le pool de leur réplicat.
    signs = RNG.choice([-1.0, 1.0], size=(N_BOOT, n))
    null = (signs * np.abs(x)).mean(axis=1)
    p = float((np.abs(null) >= abs(x.mean())).mean())
    return {"pair": f"{a} - {b}", "delta": float(x.mean()), "lo": lo, "hi": hi,
            "p": p, "n": n, "sd": float(x.std(ddof=1)),
            "mde": float(2.8 * x.std(ddof=1) / np.sqrt(n))}


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    m, adj, running = len(order), {}, 0.0
    for i, key in enumerate(order):
        running = max(running, (m - i) * pvals[key])
        adj[key] = min(running, 1.0)
    return adj


def line(c: dict, tag: str = "") -> None:
    star = "  *" if c["lo"] > 0 or c["hi"] < 0 else ""
    print(f"  {c['pair']:30s} {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
          f"p={c['p']:.4f}  n={c['n']}  MDE={c['mde']:.4f}{star}{tag}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dsel", required=True, type=Path)
    a = p.parse_args(argv)

    d = load(a.dsel)
    ks = sorted(int(k) for k in d.k.unique())
    W = {k: cells(d, k) for k in ks}

    print("\n" + "=" * 78)
    print("GARDE-FOU G — v1 doit se répliquer à chaque K, sinon rien n'est lisible")
    print("=" * 78)
    gate_ok = True
    sel_eff = {}
    for k in ks:
        w = W[k]
        cr = contrast(w, "resid_top", "random")
        cb = contrast(w, "params_bottom", "random")
        sel_eff[k] = cr["delta"]
        ok = cr["lo"] > 0 and cb["delta"] < 0
        gate_ok &= ok
        print(f"\n  K={k}  (K/M = {100 * k / int(d.n_candidates.iloc[0]):.0f} %)")
        line(cr)
        line(cb)
        print(f"    → garde-fou {'OK' if ok else 'ÉCHOUE — chercher la panne avant (c)'}")
    if not gate_ok:
        print("\n  STOP. Le protocole ne reproduit pas (a)/(b) : ne pas lire (c).")
        return 1

    print("\n" + "=" * 78)
    print(f"ENDPOINT PRIMAIRE P — {PRIMARY[0]} - {PRIMARY[1]} à K={PRIMARY_K}")
    print("=" * 78)
    if PRIMARY_K not in W:
        raise SystemExit(f"K={PRIMARY_K} absent : l'endpoint primaire n'existe pas")
    P = contrast(W[PRIMARY_K], *PRIMARY)
    line(P)
    print(f"  effet de sélection de référence au même K : {sel_eff[PRIMARY_K]:+.4f}")
    print(f"  soit un contraste valant {100 * P['delta'] / sel_eff[PRIMARY_K]:+.0f} % "
          "de l'effet de sélection.")

    print("\n" + "=" * 78)
    print("SECONDAIRE S — le même contraste à chaque K (descriptif, Holm sur les 3)")
    print("=" * 78)
    sec = {k: contrast(W[k], *PRIMARY) for k in ks}
    adj = holm({k: sec[k]["p"] for k in ks})
    for k in ks:
        line(sec[k], tag=f"   [K={k}, Holm={adj[k]:.4f}]")

    print("\n" + "=" * 78)
    print("PRÉDICTION T — la saturation en K, qui peut me donner tort")
    print("=" * 78)
    kmin, kmax = min(ks), max(ks)
    gmin = contrast(W[kmin], "resid_top", "random")
    gmax = contrast(W[kmax], "resid_top", "random")
    print(f"  gain sur random : K={kmin} {gmin['delta']:+.4f}   "
          f"K={kmax} {gmax['delta']:+.4f}   écart {gmin['delta'] - gmax['delta']:+.4f}")
    if gmin["delta"] > gmax["delta"]:
        print("  → le gain DÉCROÎT avec K : l'explication de l'échec de v1 (K=20 sur un")
        print("    vivier de 82 = régime saturé) est soutenue par la mesure.")
    else:
        print("  → le gain NE décroît PAS avec K. Mon explication de v1 est FAUSSE :")
        print("    le protocole v1 n'échouait pas par saturation, et il faut chercher")
        print("    ailleurs avant de réutiliser cet argument dans le papier.")

    print("\n" + "=" * 78)
    print("VERDICT SUR (c)")
    print("=" * 78)
    if P["lo"] > 0:
        print(f"  P SIGNIFICATIF ET POSITIF. À K={PRIMARY_K}, la part de `#params` "
              "orthogonale à\n  l'accuracy sélectionne de MEILLEURS donneurs que "
              "l'accuracy elle-même.")
        print("  C'est la seule configuration où « la taille mesure les données » est "
              "une méthode\n  et pas une observation. Vérifier alors le recouvrement "
              "des pools avant d'écrire.")
    elif P["hi"] < 0:
        print(f"  P SIGNIFICATIF ET NÉGATIF : à K={PRIMARY_K} l'accuracy fait MIEUX. "
              "Le résultat est\n  publiable comme réfutation, et il faut l'écrire ainsi.")
    else:
        m = EQUIV_MARGIN * sel_eff[PRIMARY_K]
        equiv = (P["lo"] > -m) and (P["hi"] < m)
        print(f"  P non significatif. TOST à ±{EQUIV_MARGIN:.0%} de l'effet de "
              f"sélection (±{m:.4f}) : "
              f"{'ÉQUIVALENCE DÉMONTRÉE' if equiv else 'équivalence NON démontrée'}.")
        if equiv:
            print("  → « sélectionner aide beaucoup, mais la taille ne fait pas mieux "
                  "que l'accuracy ».\n    Résultat d'atelier, honnête, à écrire tel quel.")
        else:
            need = int(np.ceil((2.8 * P["sd"] / m) ** 2))
            print(f"  → INDÉTERMINÉ. Ni A ni B ni équivalence. Il faudrait ~{need} "
                  f"réplicats\n    (on en a {P['n']}) pour trancher à cette marge, soit "
                  f"x{need / max(P['n'], 1):.1f} de calcul.")
            print("    NE PAS écrire « indiscernables » : c'est l'erreur du 02/09 matin.")

    out = a.dsel.parent / f"{a.dsel.name}_per_cluster.csv"
    pd.concat({k: W[k] for k in ks}, names=["k"]).to_csv(out)
    print(f"\nécrit {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
