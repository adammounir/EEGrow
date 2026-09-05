"""Lecture de la campagne « prévalence du motif », suivant `PREREG_motif_prevalence.md`.

La question, en une phrase : sur quelle FRACTION des datasets d'imagerie motrice la
dynamique de croissance classe-t-elle les donneurs mieux que l'accuracy ? À n=6 on ne
pouvait pas répondre (`lee2019_mi` seul contre `cho2017` seul, avec deux lectures
incompatibles et une puissance ~0.3 sur tout modérateur). Cette lecture est écrite pour
n=17 et son endpoint primaire est un simple test de signe, parce que c'est la seule
statistique que 17 datasets soutiennent honnêtement.

CE QUI EST MESURÉ ET CE QUI NE L'EST PAS. `capture` est calculée sur du TRANSFERT RÉEL
(la matrice donneur×receveur), pas sur un proxy : c'est la fraction de ce que le
meilleur top-5 possible gagnait sur l'allocation uniforme. Mais ce n'est PAS le gain
end-to-end d'une allocation de budget — `dalloc_restored` a montré que le lien entre les
deux est faible (P(les 3 signes prédits sont les vrais) = 0.751). Aucune sortie de ce
script ne doit être convertie en prévalence de gain end-to-end.

TROIS PIÈGES, TRAITÉS ICI PLUTÔT QUE SIGNALÉS
---------------------------------------------
1. **Le nombre de réplicats.** La fiabilité pilote la capture (Spearman(ICC, capture) =
   +1.00 / +1.00 / +0.886) et les datasets ajoutés ont de 1 à 11 sessions, donc k va de
   15 à 165. L'endpoint primaire égalise à k=15 par sous-échantillonnage. Sans ça,
   « les nouveaux répondent mieux » pourrait n'être que « ils ont plus de fits ».
2. **L'angle mort du contrôle aberrants.** Sur un dataset sans sujet suspect,
   `gap_clean == gap` par construction et « le motif survit » est une phrase vide. Ces
   datasets sont comptés dans le primaire (qui ne porte que sur le signe) et EXCLUS du
   secondaire B.
3. **Une matrice partielle.** Un dataset déclaré dont la matrice a des trous fait
   ÉCHOUER la lecture. Un receveur manquant déplace la moyenne sans rien signaler
   ([[gates-must-refuse-not-warn]]).

Usage::

    ./.venv/bin/python benchmarks/analysis/motif_readout.py
    ./.venv/bin/python benchmarks/analysis/motif_readout.py --datasets cho2017 lee2019_mi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import donor_matrix as dm  # noqa: E402

RNG = np.random.default_rng(20260905)
N_BOOT = 20000
K = 5             # top-k : l'allocation concentre, elle n'exploite que la tete
SUSPECT_ACC = 0.90   # accuracy within-session implausible en MI 2 classes
K_COMMON = 15     # replicats apres egalisation = le k du plus pauvre des existants
N_SUB = 30        # tirages de sous-echantillonnage
MIN_SUBJECTS = 16    # G1
MIN_REPLICATES = 15  # G1
# G2 porte sur l'ICC(1, K_COMMON) -- la fiabilite de la quantite REELLEMENT utilisee
# (une moyenne de K_COMMON replicats), pas celle d'un replicat isole. Corrige le 05/09
# avant toute soumission ; voir la note datee dans PREREG_motif_prevalence.md §5.
MIN_ICC_K = 0.30     # G2

# Les exports de sonde ne vivent pas tous au meme endroit : la replication fraiche
# (`dreyer2023a`, `liu2024`) a le sien. Router au mauvais arbre rend un ICC `nan` et
# excluait silencieusement deux datasets -- dont un des deux echecs.
DYNAMICS_OF = {"dreyer2023a": "dynamics_fresh", "liu2024": "dynamics_fresh"}

# Les 6 mesures avant cette campagne. Leur classement est dans la prereg, calcule avant
# le premier fit des 11 ajoutes -- il ne peut donc pas etre reconstruit apres coup.
BASELINE = ["shin2017a", "cho2017", "dreyer2023a", "lee2019_mi", "liu2024", "physionetmi"]
ADDED = ["stieger2021", "yang2025", "hefmiich2025", "guttmannflury2025_mi",
         "guttmannflury2025_me", "chang2025", "forenzo2023", "zhou2020",
         "kumar2024", "dreyer2023b", "brandl2020"]

# Modeles geles dans la prereg §8. Ne rien ajouter, ne rien retirer apres coup.
MODERATORS = ["n_subjects", "n_channels", "epoch_s", "trials_per_subject",
              "acc_mean", "acc_sd", "icc_restored", "icc_acc",
              "rho_acc_restored", "suspect_frac", "oracle_headroom"]


# --------------------------------------------------------------------------- chargement

def truth(ds: str, root: Path) -> pd.DataFrame:
    """La matrice de transfert, moyennee sur les seeds. `self=1` (le donneur teste sur
    lui-meme) est retire : ce n'est pas du transfert et il gonflerait toute regle qui
    correle avec la performance propre du sujet -- c'est-a-dire `acc_probe`."""
    fs = sorted((root / f"dxr_{ds}").glob("*.csv"))
    if not fs:
        return pd.DataFrame()
    fr = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return (fr[fr["self"] == 0].groupby(["receiver", "donor"])["roc_auc"]
            .mean().reset_index())


def capture(t: pd.DataFrame, score: pd.Series, drop: tuple = ()) -> float:
    """Fraction de l'oracle top-K captee par la regle, moyennee sur les receveurs.

    Denominateur = ce que le MEILLEUR top-K possible gagne sur l'allocation uniforme.
    Normaliser est indispensable : le gisement varie d'un facteur 1.4 entre datasets et
    comparer des gains bruts confondrait « bonne regle » et « dataset genereux ».
    """
    s = score.dropna()
    keep = [x for x in s.index if x not in drop]
    num, den = [], []
    for _, g in t.groupby("receiver"):
        d = g[g.donor.isin(keep)]
        if len(d) < 2 * K:
            continue
        y = d.roc_auc.to_numpy(float)
        mu = y.mean()
        num.append(y[np.argsort(-s.loc[d.donor].to_numpy(float))[:K]].mean() - mu)
        den.append(np.sort(y)[-K:].mean() - mu)
    if not num:
        return float("nan")
    return float(np.mean(num) / np.mean(den))


def icc11(d: pd.DataFrame, col: str) -> tuple[float, float]:
    """ICC(1,1) -- fiabilite d'UN replicat, la seule forme comparable entre datasets.
    L'ICC(1,k) monte mecaniquement avec k (Spearman-Brown) et k vaut ici de 15 a 165 ;
    les confondre ferait dire « stieger est plus fiable » quand il a juste 11 sessions.
    """
    d = d[["subject", col]].dropna()
    g = d.groupby("subject")[col]
    k = g.size().mean()
    if len(g) < 3 or k < 2:
        return float("nan"), float(k)
    msb, msw = g.mean().var(ddof=1) * k, g.var(ddof=1).mean()
    denom = msb + (k - 1) * msw
    return (float((msb - msw) / denom) if denom > 0 else float("nan")), float(k)


def probe_fits(dynamics: Path) -> pd.DataFrame:
    f = pd.read_csv(dynamics / "gd_fits.csv.gz", low_memory=False)
    f["align_tag"] = f["align_tag"].fillna("none")
    f = f[(f["eval"] == dm.PROBE_EVAL) & (f.model == dm.PROBE_MODEL)
          & (f["align_tag"] == "none")]
    f["subject"] = pd.to_numeric(f["subject"], errors="coerce")
    return f[["dataset", "subject", "restored_epoch"]].dropna()


def scores(ds: str, root: Path) -> pd.DataFrame:
    """Les scores PAR FOLD de la sonde non alignee -- la source de `acc_probe`, de son
    ICC, et des trois moderateurs de geometrie (`n_chans_in`, `n_times_in`, essais).

    Trois arbres, un seul format : la campagne finale range sous `perf_final/scores`, la
    replication fraiche sous `results_fresh`, les 11 ajoutes sous `results_motif`. Le
    seul niveau qui differe est le prefixe.
    """
    for base in (root / "perf_final" / "scores", root / "results_fresh",
                 root / "results_motif"):
        d = base / "within_session" / ds
        fs = sorted(d.glob(f"{dm.PROBE_MODEL}__seed*.csv"))
        if fs:
            return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return pd.DataFrame()


def equalized(fits: pd.DataFrame) -> pd.Series:
    """`restored_probe` sous-echantillonne a K_COMMON replicats par sujet.

    Un sujet qui a moins de K_COMMON replicats garde les siens -- le tronquer serait
    pire que l'inegalite qu'on corrige. La moyenne sur N_SUB tirages est ce qui rend la
    valeur stable ; un tirage unique reintroduirait le bruit qu'on cherche a egaliser.
    """
    arr = {s: v["restored_epoch"].to_numpy(float)
           for s, v in fits.groupby("subject") if len(v) > 0}
    out = {}
    for s, v in arr.items():
        m = min(K_COMMON, len(v))
        out[s] = float(np.mean([RNG.choice(v, m, replace=False).mean()
                                for _ in range(N_SUB)]))
    return pd.Series(out)


# ----------------------------------------------------------------------------- dataset

def one(ds: str, root: Path, dynamics: Path) -> dict | None:
    """Une ligne de resultat, ou None si le dataset n'a pas de donnees du tout."""
    rk_path = root / f"ranking_{ds}.csv"
    t = truth(ds, root)
    if not rk_path.exists() or t.empty:
        return None
    rk = pd.read_csv(rk_path).set_index("subject")
    fits = probe_fits(dynamics)
    fits = fits[fits.dataset == ds]

    icc_r, k_r = icc11(fits, "restored_epoch")
    n_ok = int((rk["k_probe"] >= MIN_REPLICATES).sum()) if "k_probe" in rk else 0

    sc = scores(ds, root)
    if sc.empty:
        icc_acc, n_ch, ep_s, tr = np.nan, np.nan, np.nan, np.nan
    else:
        sc["subject"] = pd.to_numeric(sc["subject"], errors="coerce")
        icc_acc = icc11(sc.rename(columns={"score": "acc"}), "acc")[0]
        n_ch = float(sc["n_chans_in"].iloc[0])
        ep_s = float(sc["n_times_in"].iloc[0]) / float(sc["sfreq"].iloc[0])
        # Essais par sujet : un fold voit 80 % des essais d'une session, et un sujet a
        # `n_sessions` sessions. On reconstitue le total plutot que de rapporter la
        # taille d'un fold, qui varierait avec le protocole et non avec le dataset.
        tr = float((sc["samples"] + sc["samples_test"]).median()
                   * sc["n_sessions"].iloc[0])

    # ---- G3 : la matrice doit etre pleine. Refus, pas avertissement.
    donors, receivers = set(t.donor), set(t.receiver)
    pool = donors | receivers
    expected = len(pool) * (len(pool) - 1)
    if len(t) != expected:
        raise SystemExit(
            f"{ds} : matrice D×R incomplete -- {len(t)} cases pour {expected} "
            f"attendues sur un vivier de {len(pool)} sujets. Un receveur manquant "
            f"deplace la moyenne sans rien signaler ; refus de lire une matrice a trous.")

    # ---- G1 / G2 : lus sur la SONDE seule, donc avant d'avoir vu le moindre gap.
    # Spearman-Brown : la fiabilite d'une moyenne de m replicats, a partir de celle d'un
    # seul. C'est cette valeur-la que la regle utilise, donc c'est elle que la porte doit
    # regarder -- un ICC(1,1) de 0.05 sur 15 replicats vaut 0.44 une fois moyenne.
    icc_k = (K_COMMON * icc_r / (1 + (K_COMMON - 1) * icc_r)
             if np.isfinite(icc_r) and icc_r > 0 else icc_r)
    gate, why = True, []
    if n_ok < MIN_SUBJECTS:
        gate, _ = False, why.append(f"G1 {n_ok}<{MIN_SUBJECTS} sujets a k>={MIN_REPLICATES}")
    if not (icc_k > MIN_ICC_K):
        gate, _ = False, why.append(
            f"G2 ICC(1,{K_COMMON})={icc_k:.3f}<={MIN_ICC_K}")

    rest = equalized(fits).reindex(rk.index)
    acc = rk["acc_probe"]
    susp = tuple(rk.index[acc >= SUSPECT_ACC])

    gap = capture(t, rest) - capture(t, acc)
    gap_clean = capture(t, rest, susp) - capture(t, acc, susp)

    if not gate:
        cls = "EXCLU"
    elif gap <= 0:
        cls = "ABSENT"
    elif len(susp) < 3:
        cls = "MOTIF_NON_TESTE"
    elif gap_clean >= gap / 2:
        cls = "MOTIF"
    else:
        cls = "ROBUSTESSE"

    gv = t.groupby("donor")["roc_auc"].mean()
    i = gv.index.intersection(rk.index)
    # Gisement : ce que le meilleur top-K gagne sur l'uniforme, en AUC brute. C'est le
    # denominateur de `capture`, sorti ici parce qu'il est un modérateur candidat.
    head = np.mean([np.sort(g.roc_auc.to_numpy(float))[-K:].mean()
                    - g.roc_auc.mean() for _, g in t.groupby("receiver")
                    if len(g) >= 2 * K])

    return {
        "dataset": ds, "classe": cls, "gate": gate,
        "why": "; ".join(why), "gap": gap, "gap_clean": gap_clean,
        "n_subjects": len(rk), "n_suspects": len(susp),
        "k_probe_median": float(rk["k_probe"].median()) if "k_probe" in rk else np.nan,
        "icc_restored": icc_r, "icc_restored_k": icc_k, "icc_acc": icc_acc,
        "acc_mean": float(acc.mean()), "acc_sd": float(acc.std(ddof=1)),
        "rho_acc_restored": float(spearmanr(acc.loc[i].to_numpy(float),
                                            rest.loc[i].to_numpy(float),
                                            nan_policy="omit").statistic),
        "suspect_frac": len(susp) / len(rk), "oracle_headroom": float(head),
        "n_channels": n_ch, "epoch_s": ep_s, "trials_per_subject": tr,
    }


# ----------------------------------------------------------------------------- endpoints

def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--root", type=Path, default=A)
    p.add_argument("--dynamics", type=Path, default=None)
    a = p.parse_args(argv)
    todo = a.datasets if a.datasets else BASELINE + ADDED

    rows = []
    for ds in todo:
        dyn = a.dynamics or A / DYNAMICS_OF.get(
            ds, "dynamics_final" if ds in BASELINE else "dynamics_motif")
        if not (dyn / "gd_fits.csv.gz").exists():
            print(f"  (pas encore mesure : {ds})")
            continue
        r = one(ds, a.root, dyn)
        if r is None:
            print(f"  (pas encore mesure : {ds})")
            continue
        r["vague"] = "existant" if ds in BASELINE else "ajoute"
        rows.append(r)

    if not rows:
        raise SystemExit("aucun dataset lisible.")
    df = pd.DataFrame(rows)

    print("=" * 92)
    print("PAR DATASET -- gap = capture(restored, k egalise) - capture(acc)")
    print("=" * 92)
    cols = ["dataset", "vague", "n_subjects", "k_probe_median", "icc_restored",
            "icc_restored_k", "n_suspects", "gap", "gap_clean", "classe"]
    print(df[cols].to_string(index=False,
                             formatters={"gap": "{:+.3f}".format,
                                         "gap_clean": "{:+.3f}".format,
                                         "icc_restored": "{:.3f}".format,
                                         "icc_restored_k": "{:.3f}".format}))
    for _, r in df[~df.gate].iterrows():
        print(f"  EXCLU {r.dataset} : {r.why}")

    adm = df[df.gate]
    if adm.empty:
        raise SystemExit("aucun dataset admis par la porte.")

    print("\n" + "=" * 92)
    print("PRIMAIRE -- prevalence du signe (unite d'analyse = le dataset)")
    print("=" * 92)
    k, n = int((adm.gap > 0).sum()), len(adm)
    bt = binomtest(k, n, 0.5, alternative="greater")
    lo, hi = clopper_pearson(k, n)
    print(f"  {k}/{n} datasets a gap > 0   proportion {k / n:.3f}  "
          f"IC95 Clopper-Pearson [{lo:.3f}, {hi:.3f}]   p={bt.pvalue:.4f} (binomial "
          f"exact unilateral vs 0.5)")
    if bt.pvalue >= 0.05:
        # La MDE, pas un « pas d'effet » : [[underpowered-not-null]].
        need = next((x for x in range(n + 1)
                     if binomtest(x, n, 0.5, alternative="greater").pvalue < 0.05), n)
        print(f"  NON RESOLU a n={n} : il aurait fallu {need}/{n} positifs. Rapporter "
              f"la MDE, jamais « pas d'effet ».")

    # SENSIBILITE SANS PORTE. Elle existe parce que la porte G2, dans sa premiere
    # version, excluait exactement les datasets ou la regle ECHOUE -- ce qui aurait
    # biaise la prevalence vers le haut sans que rien ne le signale. Publier les deux
    # chiffres cote a cote enleve la possibilite de regler la reponse par le seuil.
    k2, n2 = int((df.gap > 0).sum()), int(df.gap.notna().sum())
    b2 = binomtest(k2, n2, 0.5, alternative="greater")
    lo2, hi2 = clopper_pearson(k2, n2)
    print(f"  SANS PORTE (tous matrices completes) : {k2}/{n2} = {k2 / n2:.3f} "
          f"IC95 [{lo2:.3f}, {hi2:.3f}] p={b2.pvalue:.4f}")
    if (k / n > 0.5) != (k2 / n2 > 0.5):
        print("  ATTENTION : la porte change le SENS de la conclusion. Rapporter les "
              "deux et discuter, ne pas choisir.")

    print("\n" + "=" * 92)
    print("SECONDAIRE A -- magnitude (bootstrap sur les DATASETS)")
    print("=" * 92)
    g = adm.gap.to_numpy(float)
    b = RNG.choice(g, (N_BOOT, len(g)), replace=True).mean(1)
    pb = 2 * min((b <= 0).mean(), (b >= 0).mean())
    print(f"  gap moyen {g.mean():+.4f}  IC95 [{np.percentile(b, 2.5):+.4f}, "
          f"{np.percentile(b, 97.5):+.4f}]  p={max(pb, 1 / N_BOOT):.4f}")

    print("\n" + "=" * 92)
    print("SECONDAIRE B -- motif ou robustesse, sur les seuls datasets testables")
    print("=" * 92)
    tst = adm[adm.n_suspects >= 3]
    nm = int((tst.classe == "MOTIF").sum())
    nr = int((tst.classe == "ROBUSTESSE").sum())
    print(f"  {nm} MOTIF / {nm + nr} testables (le reste : {nr} ROBUSTESSE). "
          f"{int((adm.classe == 'MOTIF_NON_TESTE').sum())} datasets a <3 suspects sont "
          f"exclus d'ici -- leur controle aberrants est vide par construction.")
    print("  Descriptif, sans test : le prior est 1/2 et aucun n atteignable ne le tranche.")

    print("\n" + "=" * 92)
    print(f"MODERATEURS -- liste gelee prereg §8, Holm sur {len(MODERATORS)}")
    print("=" * 92)
    ps = {}
    for m in MODERATORS:
        if m not in adm or adm[m].notna().sum() < 5:
            continue
        r = spearmanr(adm[m], adm.gap, nan_policy="omit")
        ps[m] = (float(r.statistic), float(r.pvalue))
    items = sorted(ps.items(), key=lambda kv: kv[1][1])
    run = 0.0
    print(f"  {'moderateur':<22}{'rho':>8}{'p brut':>10}{'p Holm':>10}")
    for i, (m, (rho, pv)) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * pv))
        print(f"  {m:<22}{rho:>+8.3f}{pv:>10.4f}{run:>10.4f}"
              f"{'*' if run < 0.05 else ''}")
    print(f"\n  Exploratoire. A n={len(adm)} et alpha Holm ~ {0.05 / max(len(items), 1):.4f}, "
          f"la puissance est ~0.70 pour |rho|=0.7 et ~0.35 pour |rho|=0.5.")

    out = A / "motif_prevalence.csv"
    df.to_csv(out, index=False)
    print(f"\nSorti : {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
