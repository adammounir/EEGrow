"""« La croissance aide les mauvais sujets » : effet reel, ou regression vers la moyenne ?

LA FIGURE QUI POSE LA QUESTION. `perf_figures` trie les sujets par le score de `bd_deep4`
puis dessine, le long de cet axe, le segment `grow_deep - bd_deep4`. Sur `lee2019_mi` et
`shin2017a` les segments « croissance en tete » sont visiblement concentres EN BAS.

POURQUOI CETTE LECTURE EST PIEGEE. Le score de bd apparait DEUX FOIS dans la figure : une
fois comme position verticale, une fois avec un signe moins dans le delta. Un sujet est en
bas soit parce qu'il est vraiment difficile, soit parce que ce fit-la a eu de la malchance
-- et pour les seconds `grow - bd` est mecaniquement positif. Formellement, la pente
naive vaut

    pente(delta ~ b) = [cov(g, b) - var(b)] / var(b)

et var(b) = var_vrai + var_bruit : le bruit de mesure de bd tire la pente vers le negatif
meme si les deux bras sont strictement echangeables. C'est le paradoxe de Galton, et il
produit exactement le motif qu'on croit lire.

LE TEST. Les 3 graines donnent des replicats independants du meme sujet. On classe avec la
graine k, on mesure le delta avec la graine m != k : le bruit de l'axe n'est alors plus
celui du delta, et la pente devient non biaisee.

    pente NAIVE  : delta = g[m] - b[m]   sur l'axe b[m]   (meme graine -> biaisee)
    pente INDEP  : delta = g[m] - b[m]   sur l'axe b[k]   (graine disjointe -> propre)

Les deux utilisent le MEME delta et un axe a UNE graine des deux cotes : l'attenuation par
erreur de mesure sur l'axe est identique, seule l'independance change. On moyenne sur les
6 paires ordonnees (k, m). L'ecart NAIVE - INDEP est, par construction, l'artefact.

Convention de signe : delta = grow - bd, donc « la croissance aide les sujets faibles »
= pente NEGATIVE. La pente est en points de score par ecart-type de bd.

LE SECOND CONFONDANT. Sur `lee2019_mi` et `shin2017a` (2 classes, chance 0.5) les sujets
du bas sont SOUS la chance : battre un classifieur qui fait pire que le hasard n'est pas
un resultat (cf. l'audit v5). On refait donc tout en ne gardant que les sujets dont bd est
au-dessus du seuil exact, et on regarde si le motif survit.

Unite d'analyse = le SUJET. Sessions moyennees ; IC = bootstrap sur les sujets, stratifie
par dataset pour le poole.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_io  # noqa: E402

ROOT = Path(__file__).resolve().parent / "perf_final" / "scores"
EVAL = "cross_session"
ALIGN = "none"
BD, GROW = "bd_deep4", "grow_deep"
#: ordre des panneaux de la figure d'origine, recopie ici pour ne pas importer
#: `perf_figures` (et matplotlib avec) juste pour une liste
ORDER = ["bnci2014_001", "bnci2014_004", "bnci2015_001", "lee2019_mi",
         "shin2017a", "zhou2016"]
BOOT = 10000
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- donnees

def load_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores par dataset x sujet x graine, drapeau au-dessus de la chance par sujet)."""
    sc = perf_io.load(ROOT)
    sc = sc[(sc["eval"] == EVAL) & (sc.align_tag == ALIGN)
            & sc.model.isin([BD, GROW])].copy()

    # sessions moyennees, graine CONSERVEE : c'est la graine qui porte le replicat
    cell = (sc.groupby(["dataset", "model", "subject", "seed"], as_index=False)
              .agg(score=("score", "mean"), n_rows=("score", "size")))
    wide = cell.pivot_table(index=["dataset", "subject", "seed"],
                            columns="model", values="score").reset_index()

    # « bd a appris sur ce sujet », deux criteres, et le choix compte :
    #   STRICT  -- chacune des 6 lignes (3 graines x 2 sessions) passe le seuil exact.
    #              Une seule ligne malchanceuse disqualifie un sujet qui apprend, donc ce
    #              critere selectionne lui-meme sur le bruit : il sert de sensibilite.
    #   MOYEN   -- le score MOYEN du sujet passe le seuil calcule sur le nombre total
    #              d'essais de test accumules. C'est le critere de reference : il teste
    #              l'affirmation qui nous interesse (« ce sujet a un signal decodable »)
    #              avec toute la puissance disponible.
    bd = sc[sc.model == BD]
    ac = (bd.groupby(["dataset", "subject"], as_index=False)
            .agg(frac_above=("above_chance", "mean"), mean_score=("score", "mean"),
                 n_eff=("n_eff", "sum"), metric=("metric", "first"),
                 chance=("chance", "first")))
    ac["thr_mean"] = [perf_io.auc_threshold(n) if m == "roc_auc"
                      else perf_io.binom_threshold(n, c)
                      for n, m, c in zip(ac.n_eff, ac.metric, ac.chance)]
    ac["bd_learned"] = ac.mean_score > ac.thr_mean          # critere de reference
    ac["bd_learned_strict"] = ac.frac_above >= 1.0          # sensibilite
    return wide, ac


# ------------------------------------------------------------------------- estimateur

def _slope(delta: np.ndarray, axis: np.ndarray) -> float:
    """OLS de delta sur l'axe z-score : points de score par ecart-type de bd."""
    a = (axis - axis.mean())
    sd = a.std(ddof=0)
    if sd == 0:
        return np.nan
    z = a / sd
    return float(np.dot(delta - delta.mean(), z) / np.dot(z, z))


def slopes_for(mat_b: np.ndarray, mat_g: np.ndarray) -> tuple[float, float]:
    """Pentes (naive, independante), moyennees sur les paires ordonnees de graines.

    `mat_*` : (n_sujets, n_graines). Pour chaque (k, m) avec k != m, le delta est
    toujours g[m] - b[m] ; seul l'axe change (b[m] contre b[k]).
    """
    n_seed = mat_b.shape[1]
    naive, indep = [], []
    for m in range(n_seed):
        delta = mat_g[:, m] - mat_b[:, m]
        naive.append(_slope(delta, mat_b[:, m]))
        for k in range(n_seed):
            if k != m:
                indep.append(_slope(delta, mat_b[:, k]))
    return float(np.nanmean(naive)), float(np.nanmean(indep))


def boot_ci(fn, n_subj: int, boot: int = BOOT) -> tuple[float, float]:
    """Percentile bootstrap en reechantillonnant les SUJETS (fn prend les indices)."""
    vals = [fn(RNG.integers(0, n_subj, n_subj)) for _ in range(boot)]
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return (np.nan, np.nan)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def win_halves(mat_b: np.ndarray, mat_g: np.ndarray) -> dict:
    """Part de sujets ou la croissance gagne, moitie basse contre moitie haute de l'axe."""
    out = {}
    n_seed = mat_b.shape[1]
    for tag, same in (("naive", True), ("indep", False)):
        lo_w, lo_n, hi_w, hi_n = 0, 0, 0, 0
        for m in range(n_seed):
            delta = mat_g[:, m] - mat_b[:, m]
            for k in range(n_seed):
                if same and k != m:
                    continue
                if not same and k == m:
                    continue
                order = np.argsort(mat_b[:, k])
                half = len(order) // 2
                lo, hi = order[:half], order[len(order) - half:]
                lo_w += int((delta[lo] > 0).sum()); lo_n += len(lo)
                hi_w += int((delta[hi] > 0).sum()); hi_n += len(hi)
        out[tag] = (lo_w, lo_n, hi_w, hi_n)
    return out


# ------------------------------------------------------------------------------- main

def main() -> None:
    wide, ac = load_grid()
    seeds = sorted(wide.seed.unique())
    datasets = [d for d in ORDER if d in set(wide.dataset)] \
        + sorted(set(wide.dataset) - set(ORDER))

    print("=" * 96)
    print("INTEGRITE")
    print("=" * 96)
    print(f"eval={EVAL}  align={ALIGN}  arms={BD} / {GROW}  graines={seeds}")
    for ds in datasets:
        sub = wide[wide.dataset == ds]
        cnt = sub.groupby("subject").size()
        a = ac[ac.dataset == ds]
        print(f"  {ds:<16} sujets={cnt.size:>3}  cellules/sujet={sorted(cnt.unique())}"
              f"  NaN={int(sub[[BD, GROW]].isna().sum().sum()):>2}"
              f"  bd a appris : {int(a.bd_learned.sum())}/{len(a)} (reference)"
              f", {int(a.bd_learned_strict.sum())}/{len(a)} (strict)")

    # ------------------------------------------------------------------ pentes
    print("\n" + "=" * 96)
    print("1. PENTE DE delta = grow - bd LE LONG DE L'AXE DE TRI")
    print("   (points de score par ecart-type de bd ; NEGATIF = la croissance aide les faibles)")
    print("=" * 96)
    print(f"{'dataset':<16}{'n':>4}{'NAIVE (axe=meme graine)':>28}"
          f"{'INDEP (axe=graine disjointe)':>32}{'artefact':>11}")

    pooled = []                     # (delta_matrices) pour la version poolee
    for ds in datasets:
        sub = wide[wide.dataset == ds].dropna(subset=[BD, GROW])
        piv_b = sub.pivot_table(index="subject", columns="seed", values=BD)
        piv_g = sub.pivot_table(index="subject", columns="seed", values=GROW)
        keep = piv_b.notna().all(axis=1) & piv_g.notna().all(axis=1)
        mb, mg = piv_b[keep].to_numpy(), piv_g[keep].to_numpy()
        n = len(mb)
        if n < 4:
            print(f"{ds:<16}{n:>4}   (trop peu de sujets pour une pente)")
            continue
        sn, si = slopes_for(mb, mg)
        lo_n, hi_n = boot_ci(lambda idx: slopes_for(mb[idx], mg[idx])[0], n)
        lo_i, hi_i = boot_ci(lambda idx: slopes_for(mb[idx], mg[idx])[1], n)
        print(f"{ds:<16}{n:>4}  {sn:+.4f} [{lo_n:+.4f},{hi_n:+.4f}]"
              f"   {si:+.4f} [{lo_i:+.4f},{hi_i:+.4f}]{sn - si:>+11.4f}")
        pooled.append((ds, mb, mg))

    # poole : moyenne des pentes par dataset, ponderee par le nombre de sujets. Le
    # bootstrap reechantillonne les sujets DANS chaque dataset (stratifie), jamais les
    # datasets : c'est le sujet qui est l'unite, et les 6 datasets sont le plan, pas un
    # tirage.
    def pooled_slopes(picks=None) -> tuple[float, float]:
        nai = ind = wtot = 0.0
        for j, (_, mb, mg) in enumerate(pooled):
            idx = np.arange(len(mb)) if picks is None else picks[j]
            a, b = slopes_for(mb[idx], mg[idx])
            w = len(idx)
            nai += a * w; ind += b * w; wtot += w
        return nai / wtot, ind / wtot

    pn, pi = pooled_slopes()
    bn, bi = np.empty(BOOT), np.empty(BOOT)
    for t in range(BOOT):
        picks = [RNG.integers(0, len(mb), len(mb)) for _, mb, _ in pooled]
        bn[t], bi[t] = pooled_slopes(picks)
    print("-" * 96)
    print(f"{'POOLE (pondere n)':<16}{sum(len(m) for _, m, _ in pooled):>4}"
          f"  {pn:+.4f} [{np.nanquantile(bn,.025):+.4f},{np.nanquantile(bn,.975):+.4f}]"
          f"   {pi:+.4f} [{np.nanquantile(bi,.025):+.4f},{np.nanquantile(bi,.975):+.4f}]"
          f"{pn - pi:>+11.4f}")

    # ------------------------------------------------- moities basse / haute de l'axe
    print("\n" + "=" * 96)
    print("2. LE MEME MOTIF EN COMPTE DE VICTOIRES : moitie BASSE contre moitie HAUTE")
    print("=" * 96)
    print(f"{'dataset':<16}{'axe':<8}{'grow gagne, moitie basse':>28}"
          f"{'moitie haute':>18}{'ecart':>9}")
    for ds, mb, mg in pooled:
        w = win_halves(mb, mg)
        for tag in ("naive", "indep"):
            lw, ln, hw, hn = w[tag]
            gap = lw / ln - hw / hn
            print(f"{ds:<16}{tag:<8}{f'{lw}/{ln} = {lw/ln:.0%}':>28}"
                  f"{f'{hw}/{hn} = {hw/hn:.0%}':>18}{gap:>+9.0%}")

    # --------------------------------------------- restriction au-dessus de la chance
    print("\n" + "=" * 96)
    print("3. MEME ANALYSE, SUJETS OU bd A REELLEMENT APPRIS (toutes lignes > seuil exact)")
    print("=" * 96)
    for crit, label in (("bd_learned", "reference : score moyen > seuil"),
                        ("bd_learned_strict", "strict : chaque ligne > seuil")):
        print(f"\n  -- critere {label} --")
        print(f"  {'dataset':<16}{'n gardes':>9}{'NAIVE':>26}{'INDEP':>28}")
        kept = []
        for ds in datasets:
            ok = set(ac[(ac.dataset == ds) & ac[crit]].subject)
            sub = wide[(wide.dataset == ds) & wide.subject.isin(ok)].dropna(
                subset=[BD, GROW])
            piv_b = sub.pivot_table(index="subject", columns="seed", values=BD)
            piv_g = sub.pivot_table(index="subject", columns="seed", values=GROW)
            keep = piv_b.notna().all(axis=1) & piv_g.notna().all(axis=1)
            mb, mg = piv_b[keep].to_numpy(), piv_g[keep].to_numpy()
            n_all = wide[wide.dataset == ds].subject.nunique()
            if len(mb) < 4:
                print(f"  {ds:<16}{f'{len(mb)}/{n_all}':>9}   (trop peu apres filtre)")
                continue
            sn, si = slopes_for(mb, mg)
            lo_n, hi_n = boot_ci(lambda idx: slopes_for(mb[idx], mg[idx])[0], len(mb))
            lo_i, hi_i = boot_ci(lambda idx: slopes_for(mb[idx], mg[idx])[1], len(mb))
            print(f"  {ds:<16}{f'{len(mb)}/{n_all}':>9}  {sn:+.4f} [{lo_n:+.4f},{hi_n:+.4f}]"
                  f"   {si:+.4f} [{lo_i:+.4f},{hi_i:+.4f}]")
            kept.append((ds, mb, mg))
        if len(kept) > 1:
            def ps(picks=None):
                a = b = w = 0.0
                for j, (_, mb, mg) in enumerate(kept):
                    idx = np.arange(len(mb)) if picks is None else picks[j]
                    x, y = slopes_for(mb[idx], mg[idx])
                    a += x * len(idx); b += y * len(idx); w += len(idx)
                return a / w, b / w
            kn, ki = ps()
            bkn, bki = np.empty(BOOT), np.empty(BOOT)
            for t in range(BOOT):
                pk = [RNG.integers(0, len(mb), len(mb)) for _, mb, _ in kept]
                bkn[t], bki[t] = ps(pk)
            ntot = sum(len(mb) for _, mb, _ in kept)
            print(f"  {'POOLE':<16}{ntot:>9}  {kn:+.4f} "
                  f"[{np.nanquantile(bkn,.025):+.4f},{np.nanquantile(bkn,.975):+.4f}]"
                  f"   {ki:+.4f} "
                  f"[{np.nanquantile(bki,.025):+.4f},{np.nanquantile(bki,.975):+.4f}]")

    # -------------------------------------- l'endpoint global, avec et sans le filtre
    print("\n" + "=" * 96)
    print("4. ENDPOINT GLOBAL delta = grow - bd, TOUS SUJETS contre SUJETS OU bd A APPRIS")
    print("   Si le delta se degrade en retirant les sujets sous la chance, le 'gain sur")
    print("   les faibles' etait du plancher de chance, pas de la croissance.")
    print("=" * 96)
    print(f"{'dataset':<16}{'tous sujets':>34}{'bd a appris':>34}")
    for ds in datasets:
        sub = wide[wide.dataset == ds].dropna(subset=[BD, GROW])
        per = sub.groupby("subject")[[BD, GROW]].mean()
        ok = set(ac[(ac.dataset == ds) & ac.bd_learned].subject)
        cells = []
        for who in (per.index, per.index.intersection(list(ok))):
            d = (per.loc[who, GROW] - per.loc[who, BD]).to_numpy()
            if len(d) < 2:
                cells.append(f"{'n<2':>34}"); continue
            r = perf_io.test(d)
            cells.append(f"{r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] n={r['n']:>3}"
                         .rjust(34))
        print(f"{ds:<16}{cells[0]}{cells[1]}")

    print("\n" + "=" * 96)
    print("LECTURE")
    print("=" * 96)
    print("  L'ecart NAIVE - INDEP est la part de pente fabriquee par le tri lui-meme.")
    print("  Si INDEP revient a zero, « la croissance aide les mauvais sujets » est un")
    print("  artefact de regression vers la moyenne et ne peut pas porter un papier.")
    print("  Si INDEP reste franchement negative -- et le reste au bloc 3, sur les seuls")
    print("  sujets ou la baseline a appris -- alors l'effet est reel et publiable.")


if __name__ == "__main__":
    main()
