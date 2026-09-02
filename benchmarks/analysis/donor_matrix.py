"""Claim 2 : la taille où la croissance s'arrête prédit-elle la qualité de donneur ?

Lit la matrice produite par ``benchmarks/donor_receiver.py`` et répond dans l'ordre où
les réponses peuvent tuer les suivantes. Chaque section est une porte : si elle ne
passe pas, celles d'en dessous ne veulent rien dire et il faut le dire plutôt que
publier le rho.

PORTE 0 -- Y A-T-IL DU TRANSFERT ?
    Si un modèle entraîné sur un sujet et testé sur un autre est à la chance, il n'y a
    pas de « qualité de donneur » à prédire, et toute corrélation trouvée ensuite est
    une corrélation avec du bruit. C'est le contrôle négatif du protocole entier, et il
    se mesure avant tout le reste.

PORTE 1 -- LA QUALITÉ DE DONNEUR EST-ELLE UNE PROPRIÉTÉ DU DONNEUR ?
    Même question que l'étage 0 posait au prédicteur, posée cette fois au *critère*.
    Si la ligne d'un donneur change autant d'une seed à l'autre qu'elle change d'un
    donneur à l'autre, il n'y a rien de stable à prédire. On mesure donc la fiabilité
    de la sortie (ICC sur les seeds, corrigé par Spearman-Brown pour la moyenne des 3),
    et son sqrt plafonne toute corrélation observable -- exactement comme sqrt(ICC_k)
    du prédicteur la plafonne de l'autre côté. Le plafond réel est le produit des deux.

LA CORRECTION QUI N'EST PAS OPTIONNELLE : CENTRER PAR RECEVEUR
    Un receveur facile donne un score élevé à tous ses donneurs. La moyenne brute d'une
    ligne mélange donc « ce donneur est bon » avec « ce donneur est tombé sur des
    receveurs faciles » -- et la diagonale exclue fait que chaque donneur est moyenné
    sur un ensemble de receveurs LÉGÈREMENT DIFFÉRENT (les N-1 autres, jamais les mêmes
    N-1). C'est petit, mais c'est un biais systématique et non du bruit : le donneur
    difficile est absent de sa propre colonne, donc sa ligne est prise sur une
    population en moyenne plus facile que celle du donneur facile. Z-scorer chaque
    COLONNE avant de moyenner les lignes retire les deux d'un coup. Les deux versions
    sont reportées ; c'est la centrée qui est le critère.

LA COMPARAISON QUI FAIT LE PAPIER
    « #params prédit mieux que l'accuracy » n'est pas « rho(params) est significatif et
    rho(acc) ne l'est pas » -- deux tests séparés ne font pas une comparaison, et c'est
    l'erreur classique. On teste la DIFFÉRENCE des deux corrélations sur les mêmes
    sujets (bootstrap apparié sur les donneurs), et on donne la rho partielle de
    #params à accuracy contrôlée.

L'unité d'analyse est le donneur. n = 52 sur cho2017.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_matrix.py --dxr /path/to/dxr/cho2017
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import perf_io  # noqa: E402
from donor_predictor import icc1  # noqa: E402

PROBE_MODEL = "grow_shallow"
PROBE_EVAL = "within_session"
METRIC = "roc_auc"
N_BOOT = 10000
RNG = np.random.default_rng(20260901)


def load_matrix(dxr: Path) -> pd.DataFrame:
    """Les CSV par (donneur, seed) concaténés, diagonale marquée mais conservée.

    La diagonale est le modèle appliqué à ses PROPRES données d'entraînement : elle est
    optimiste par construction et n'est pas un score de transfert. Elle reste dans la
    frame parce qu'elle diagnostique le fit (un donneur qui n'apprend même pas ses
    propres données ne peut rien donner à personne), et elle est écartée partout où on
    parle de transfert.
    """
    files = sorted(dxr.glob("donor*__seed*.csv"))
    if not files:
        raise SystemExit(f"no donor CSV under {dxr}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(files)} (donor, seed) cells, {len(d)} rows, "
          f"{d.donor.nunique()} donors x {d.receiver.nunique()} receivers x "
          f"{d.seed.nunique()} seeds")
    return d


def donor_quality(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Une ligne par donneur : qualité brute, qualité centrée par receveur, fiabilité.

    ``q_raw``  moyenne du donneur sur les receveurs (hors soi), seeds moyennées.
    ``q_cent`` même chose après z-score de chaque colonne receveur -- le critère.
    ``q_seed_*`` la qualité centrée calculée seed par seed, d'où sort l'ICC.
    """
    off = d[d["self"] == 0].copy()
    per_seed = []
    for s, g in off.groupby("seed"):
        m = g.pivot_table(index="donor", columns="receiver", values=metric)
        # z-score PAR COLONNE : on retire la difficulté du receveur, pas la qualité du
        # donneur. std sur les donneurs présents dans la colonne ; une colonne
        # constante (aucun contraste entre donneurs sur ce receveur) n'apporte rien et
        # produirait des inf.
        sd = m.std(axis=0, ddof=1)
        z = (m - m.mean(axis=0)).div(sd.where(sd > 0))
        per_seed.append(pd.DataFrame({
            "donor": m.index, "seed": s,
            "q_raw": m.mean(axis=1).to_numpy(),
            "q_cent": z.mean(axis=1).to_numpy(),
        }))
    ps = pd.concat(per_seed, ignore_index=True)
    q = ps.groupby("donor")[["q_raw", "q_cent"]].mean().reset_index()
    q["n_seeds"] = ps.groupby("donor").size().to_numpy()
    return q, ps


def boot_ci(fn, n: int, reps: int = N_BOOT) -> tuple[float, float]:
    """IC 95 % percentile par ré-échantillonnage DES DONNEURS.

    Le donneur est l'unité : ré-échantillonner les cases (d, r) traiterait les 51 cases
    d'une même ligne comme 51 observations indépendantes, alors qu'elles partagent le
    même fit -- c'est la faute qui gonfle un p de quatre ordres de grandeur
    ([[unit-of-analysis-subject]]).
    """
    vals = []
    for _ in range(reps):
        idx = RNG.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman(x, y) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def partial_spearman(x, y, z) -> float:
    """rho(x, y) à z contrôlé, sur les rangs."""
    if np.ptp(z) == 0:
        return spearman(x, y)
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(stats.pearsonr(ex, ey)[0])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dxr", required=True, type=Path)
    p.add_argument("--metric", default=METRIC, choices=["roc_auc", "acc"])
    a = p.parse_args(argv)

    d = load_matrix(a.dxr)
    dataset = str(d.dataset.iloc[0])
    chance = float(d.chance.iloc[0])
    metric = a.metric

    # ------------------------------------------------------- porte 0 : du transfert ?
    print("\n" + "=" * 78)
    print("PORTE 0 — y a-t-il seulement du transfert entre sujets ?")
    print("=" * 78)
    off = d[d["self"] == 0]
    diag = d[d["self"] == 1]
    cell = off.groupby(["donor", "receiver"])[metric].mean()
    per_donor = off.groupby("donor")[metric].mean()
    t = perf_io.test((per_donor - chance).to_numpy())
    print(f"{dataset} · {metric} · chance {chance:.3f}")
    print(f"  hors-diagonale : {cell.mean():.4f}  (min {cell.min():.4f}, "
          f"max {cell.max():.4f}, {100 * (cell > chance).mean():.0f} % au-dessus "
          "de la chance)")
    print(f"  diagonale (données d'entraînement, PAS du transfert) : "
          f"{diag[metric].mean():.4f}")
    print(f"  au-dessus de la chance, unité = donneur (n={len(per_donor)}) : "
          f"{t['mean']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]")
    if t["lo"] <= 0:
        print("  → NON. Sans transfert il n'y a pas de qualité de donneur à prédire ; "
              "tout ce qui suit est une corrélation avec du bruit.")

    # ------------------------------------ porte 1 : la qualité est-elle du donneur ?
    q, per_seed = donor_quality(d, metric)
    print("\n" + "=" * 78)
    print("PORTE 1 — la qualité de donneur est-elle une propriété du donneur ?")
    print("=" * 78)
    groups = [g.q_cent.to_numpy(float) for _, g in per_seed.groupby("donor")
              if len(g) >= 2]
    icc, n_g, k = icc1(groups)
    icc_k = k * max(icc, 0) / (1 + (k - 1) * max(icc, 0)) if icc > 0 else 0.0
    print(f"  ICC(1) sur les seeds : {icc:.3f}  (n={n_g} donneurs, k={k} seeds)")
    print(f"  ICC_k (fiabilité de la moyenne des {k} seeds, Spearman-Brown) : "
          f"{icc_k:.2f}   sqrt = {icc_k ** 0.5:.2f}")
    print("  sqrt(ICC_k) plafonne la corrélation observable DU CÔTÉ DU CRITÈRE ;")
    print("  le plafond réel est ce facteur multiplié par celui du prédicteur")
    print("  (0.93 sur cho2017, étage 0) et par la part non redondante (0.98).")

    # ------------------------------------------------------------- les prédicteurs
    fits = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    f = fits[(fits["eval"] == PROBE_EVAL) & (fits.model == PROBE_MODEL)
             & (fits.align_tag == "none") & (fits.dataset == dataset)].copy()
    f["subject"] = pd.to_numeric(f["subject"], errors="coerce")
    probe = f.groupby("subject").agg(params_probe=("params_end", "mean"),
                                     k_probe=("params_end", "size")).reset_index()

    sc = perf_io.load(A / "perf_final" / "scores")
    sc = perf_io.attach_params(sc, fits)
    subj = perf_io.by_subject(sc)
    s = subj[(subj["eval"] == PROBE_EVAL) & (subj.align_tag == "none")
             & (subj.model == PROBE_MODEL) & (subj.dataset == dataset)]
    acc = s[["subject", "score", "samples"]].rename(columns={"score": "acc_probe"})
    acc["subject"] = pd.to_numeric(acc["subject"], errors="coerce")

    dxr_params = (d[d["self"] == 0].groupby("donor")
                  .agg(params_dxr=("params_end", "mean"),
                       width_dxr=("width_end", "mean"),
                       epochs_dxr=("epochs", "mean")).reset_index())

    # Le fit donneur SATURE-T-IL la cible de croissance ? Il voit 100 % des essais là
    # où la sonde en voit 80 % sur un fold, et `grow_every=5` sur 200 époques laisse 40
    # événements pour 32 neurones à ajouter : il peut atteindre `target_n_filters_time`
    # et s'y arrêter. Une cellule à la cible n'est plus une mesure de « jusqu'où ce
    # sujet fait grandir le réseau », c'est un plafond. Mesuré ici plutôt que supposé,
    # parce que la fraction censurée décide de la lecture des deux blocs suivants.
    cells = d[d["self"] == 0].drop_duplicates(["donor", "seed"])
    w_max = float(cells["width_end"].max())
    frac_cap = float((cells["width_end"] >= w_max).mean())

    q = (q.merge(probe, left_on="donor", right_on="subject", how="left")
          .merge(acc, on="subject", how="left")
          .merge(dxr_params, on="donor", how="left"))

    print("\n" + "=" * 78)
    print("CONTRÔLE — la sonde (80 % des essais, 5 folds) et le fit donneur (100 %, "
          "1 fit)")
    print("=" * 78)
    print(f"  fits donneur à la cible de croissance (largeur {w_max:.0f}) : "
          f"{100 * frac_cap:.0f} % des {len(cells)} cellules")
    if frac_cap > 0.25:
        print("  → LE FIT DONNEUR EST CENSURÉ. Au-dessus de la cible il n'y a plus de")
        print("    variance à mesurer : les cellules au plafond sont toutes à la même")
        print("    taille quel que soit le sujet. Deux conséquences, dans cet ordre :")
        print("    1. `#params (fit donneur)` ne peut pas être le prédicteur, et le")
        print("       rho qui lui est associé plus bas est un plancher, pas sa valeur.")
        print("    2. le rho sonde/donneur juste en dessous est ATTÉNUÉ PAR LA CENSURE.")
        print("       Un rho bas n'y dit donc pas que les deux mesures sont en")
        print("       désaccord ; il dit que l'une des deux a arrêté de varier.")
        print("    La sonde, elle, voit 80 % des essais sur un fold et reste sous la")
        print("    cible : c'est elle qui porte l'information, et c'est déjà le")
        print("    prédicteur déclaré (k=15 réplicats contre 3).")
    ok = q.params_probe.notna() & q.params_dxr.notna()
    if ok.sum() >= 8:
        print(f"  rho(params_sonde, params_donneur) = "
              f"{spearman(q.params_probe[ok], q.params_dxr[ok]):+.3f} sur "
              f"{int(ok.sum())} donneurs")
        print(f"  taille moyenne : sonde {q.params_probe[ok].mean():.0f}, "
              f"donneur {q.params_dxr[ok].mean():.0f} "
              f"(x{q.params_dxr[ok].mean() / q.params_probe[ok].mean():.2f})")
        print("  Un rho bas ici ne réfute pas claim 2 : ce sont deux régimes")
        print("  d'entraînement différents. Mais il dit lequel des deux est le")
        print("  prédicteur, et c'est la sonde (k=15 réplicats contre 3).")
    else:
        print("  pas assez de donneurs appariés avec la sonde de la campagne")

    # -------------------------------------------------------- claim 2, la question
    print("\n" + "=" * 78)
    print("CLAIM 2 — #params prédit-il la qualité de donneur, et mieux que l'accuracy ?")
    print("=" * 78)
    for crit, label in (("q_cent", "qualité centrée par receveur (le critère)"),
                        ("q_raw", "qualité brute (pour mémoire)")):
        m = q[q[crit].notna() & q.params_probe.notna() & q.acc_probe.notna()]
        n = len(m)
        if n < 8:
            print(f"\n  {label}: n={n}, insuffisant")
            continue
        y = m[crit].to_numpy(float)
        pp = m.params_probe.to_numpy(float)
        aa = m.acc_probe.to_numpy(float)
        print(f"\n  {label} — n = {n} donneurs")
        for name, x in (("#params (sonde)", pp), ("accuracy (sonde)", aa),
                        ("#params (fit donneur)", m.params_dxr.to_numpy(float))):
            if not np.isfinite(x).all():
                continue
            r = spearman(x, y)
            lo, hi = boot_ci(lambda i, x=x, y=y: spearman(x[i], y[i]), n)
            star = "  *" if (lo > 0 or hi < 0) else ""
            print(f"    rho({name:22s}, critère) = {r:+.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]{star}")
        rp = partial_spearman(pp, y, aa)
        lo, hi = boot_ci(
            lambda i: partial_spearman(pp[i], y[i], aa[i]), n)
        print(f"    rho partielle (#params, critère | accuracy) = {rp:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]")
        # Même question pour le #params du FIT DONNEUR. Elle se pose parce que c'est
        # la seule des trois marginales qui sorte significative : il faut savoir si
        # elle porte une information propre ou si elle relaie l'accuracy. Ce fit est
        # celui qui a produit la ligne de transfert -- sa taille finale est donc
        # solidaire de sa qualité d'ajustement, là où la sonde est mesurée à part.
        pd_ = m.params_dxr.to_numpy(float)
        if np.isfinite(pd_).all():
            rpd = partial_spearman(pd_, y, aa)
            lo, hi = boot_ci(
                lambda i: partial_spearman(pd_[i], y[i], aa[i]), n)
            star = "  *" if (lo > 0 or hi < 0) else ""
            print(f"    rho partielle (#params fit donneur, critère | accuracy) = "
                  f"{rpd:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
        # La différence des deux corrélations, sur les MÊMES donneurs. Deux tests
        # séparés ne font pas une comparaison : « l'un est significatif, l'autre non »
        # est compatible avec une différence nulle.
        dd = abs(spearman(pp, y)) - abs(spearman(aa, y))
        lo, hi = boot_ci(
            lambda i: abs(spearman(pp[i], y[i])) - abs(spearman(aa[i], y[i])), n)
        verdict = ("#params GAGNE" if lo > 0 else
                   "accuracy gagne" if hi < 0 else "indiscernables")
        print(f"    |rho(#params)| - |rho(accuracy)| = {dd:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  → {verdict}")
        # MDE : ce que ce n permet de détecter, à écrire à côté de tout nul.
        mde = np.tanh((1.959964 + 0.841621) / np.sqrt(max(n - 3, 1)))
        print(f"    MDE rho à 80 % de puissance, n={n} : {mde:.2f}")

    print("\n" + "=" * 78)
    print("À ÉCRIRE À CÔTÉ DE TOUT RÉSULTAT DE CE TABLEAU")
    print("=" * 78)
    print("  Une corrélation, même forte, ne fait pas claim 3. Elle dit que #params")
    print("  classe les donneurs ; elle ne dit pas que SÉLECTIONNER par #params bat")
    print("  l'aléatoire. Le test interventionnel est un protocole séparé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
