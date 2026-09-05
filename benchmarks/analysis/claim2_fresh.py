"""La lecture confirmatoire de claim 2 sur `dreyer2023a` + `liu2024`, dans l'ordre imposé.

Ce fichier n'a aucune liberté : il exécute `PREREG_claim2_fresh_dreyer_liu.md`, écrite et
horodatée le 03/09/2026 avant le premier fit sur ces deux datasets. L'ordre des étapes
est lui-même pré-spécifié et il est portant — la porte de fiabilité et le garde-fou G se
lisent AVANT H1, parce qu'un garde-fou qu'on consulte après avoir vu le résultat n'est
plus un garde-fou ([[gates-must-refuse-not-warn]]).

    0. intégrité      k_probe constant, sinon on ne lit rien
    1. porte ICC(1,k) sur la SONDE SEULE, jamais sur q_cent, seuil 0.50
    2. garde-fou G    acc_probe doit prédire q_cent sur CHAQUE dataset
    3. H1 primaire    restored_epoch -> q_cent | acc_probe, GROUPÉ, unilatéral
    4. H2 secondaire  params_probe   -> q_cent | acc_probe, Holm sur la famille de 2
    5. H3 descriptif  aucune inférence, la préreg l'interdit explicitement

L'ENDPOINT PRIMAIRE EST GROUPÉ, ET CE N'EST PAS UN CHOIX DE CONFORT. MDE à puissance
0.80 : +0.321 sur dreyer2023a seul, +0.351 sur liu2024 seul, tous deux AU-DESSUS de la
cible +0.288 ; +0.241 sur les 110 groupés. Les deux estimations par dataset sont
descriptives, et un rho non significatif sur un dataset seul n'est pas un échec de
réplication — il n'a pas la puissance de l'être ([[underpowered-not-null]]).

Usage::

    ./.venv/bin/python benchmarks/analysis/claim2_fresh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import donor_matrix as dm  # noqa: E402
import donor_predictor as dp  # noqa: E402

DATASETS = ["dreyer2023a", "liu2024"]
TARGET = 0.288                    # l'effet à répliquer, gelé dans la préreg
ALPHA = 0.05                      # unilatéral
ICC_FLOOR = dp.ICC_FLOOR          # 0.50, la constante qui existait déjà
N_BOOT = 20000
RNG = np.random.default_rng(20260904)

#: Les deux prédicteurs de la famille testée, dans l'ordre H1 puis H2. `restored_probe`
#: est primaire : l'étage C a trouvé POST-HOC que la durée d'apprentissage productif
#: supplante la taille, et une préreg qui testerait `params_probe` en primaire
#: re-testerait ce qui est déjà établi au lieu de ce qui ne l'est pas.
PREDICTORS = [("H1", "restored_probe", "restored_epoch"),
              ("H2", "params_probe", "params_end")]


# ----------------------------------------------------------------- frames

def dataset_frame(ds: str) -> pd.DataFrame:
    """Une ligne par donneur : les deux prédicteurs, la covariable, la qualité."""
    d = dm.load_matrix(A / f"dxr_{ds}")
    q, _ = dm.donor_quality(d, dm.METRIC)
    rank = pd.read_csv(A / f"ranking_{ds}.csv")
    m = q.merge(rank, left_on="donor", right_on="subject", how="inner")
    keep = ["donor", "q_cent", "q_raw", "restored_probe", "params_probe",
            "acc_probe", "ceil_frac", "k_probe"]
    m = m[keep].dropna(subset=["q_cent", "restored_probe", "params_probe",
                               "acc_probe"]).reset_index(drop=True)
    m["dataset"] = ds
    return m


def mde(n: int, alpha: float = ALPHA, power: float = 0.80) -> float:
    """MDE d'une partielle du premier ordre : SE = 1/sqrt(n-4), unilatéral."""
    z = stats.norm.isf(alpha) + stats.norm.isf(1 - power)
    return float(np.tanh(z / np.sqrt(max(n - 4, 1))))


# ------------------------------------------------------- groupage intra-dataset

def within_ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rangs centrés-réduits INTRA-dataset, pour que l'empilement reste intra.

    `params_probe` n'a ni la même échelle ni le même plafond d'un dataset à l'autre
    (34 k–50 k sur dreyer2023a, 19 k–39 k sur liu2024) : empiler les valeurs brutes
    créerait entre datasets une corrélation qui n'existe dans aucun des deux.
    """
    out = []
    for _, g in df.groupby("dataset", sort=False):
        g = g.copy()
        for c in cols:
            r = stats.rankdata(g[c].to_numpy(float))
            sd = r.std(ddof=1)
            g[f"z_{c}"] = (r - r.mean()) / (sd if sd > 0 else 1.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def pooled_partial(df: pd.DataFrame, pred: str) -> float:
    x = df[f"z_{pred}"].to_numpy(float)
    y = df["z_q_cent"].to_numpy(float)
    z = df["z_acc_probe"].to_numpy(float)
    ex = x - np.polyval(np.polyfit(z, x, 1), z)
    ey = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(stats.pearsonr(ex, ey)[0])


def pooled_test(df: pd.DataFrame, pred: str) -> dict:
    """Bootstrap des DONNEURS stratifié par dataset ; les datasets sont FIXES.

    Les deux datasets ne sont pas un échantillon aléatoire de datasets : ce sont ceux
    qu'on a. On ne ré-échantillonne donc que ce qui est réellement échantillonné, les
    donneurs, et le p unilatéral s'obtient par inversion (la part de la distribution
    bootstrap au-delà de zéro du mauvais côté).
    """
    cols = [pred, "q_cent", "acc_probe"]
    est = pooled_partial(within_ranks(df, cols), pred)
    groups = [g.index.to_numpy() for _, g in df.groupby("dataset", sort=False)]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([RNG.choice(g, size=len(g), replace=True)
                              for g in groups])
        b = within_ranks(df.loc[idx].reset_index(drop=True), cols)
        v = pooled_partial(b, pred)
        if np.isfinite(v):
            vals.append(v)
    v = np.asarray(vals)
    p_one = float(max((v <= 0).mean(), 1.0 / len(v)))
    return dict(est=est, lo=float(np.percentile(v, 2.5)),
                hi=float(np.percentile(v, 97.5)), p=p_one, n=len(df))


def fisher(rows: list[dict]) -> dict:
    """Effets fixes inverse-variance sur z=atanh(rho), SE=1/sqrt(n-4)."""
    z = np.array([np.arctanh(r["rho"]) for r in rows])
    se = np.array([1.0 / np.sqrt(r["n"] - 4) for r in rows])
    w = 1.0 / se ** 2
    zf = float((w * z).sum() / w.sum())
    sef = float(np.sqrt(1.0 / w.sum()))
    q = float((w * (z - zf) ** 2).sum())
    k = len(rows)
    return dict(rho=float(np.tanh(zf)),
                lo=float(np.tanh(zf - 1.96 * sef)),
                hi=float(np.tanh(zf + 1.96 * sef)),
                p=float(stats.norm.sf(zf / sef)),          # unilatéral
                q=q, df=k - 1, p_q=float(stats.chi2.sf(q, k - 1)),
                i2=float(max(0.0, (q - (k - 1)) / q) * 100) if q > 0 else 0.0)


# ------------------------------------------------------------------- rapport

def main(argv=None) -> int:
    frames = {ds: dataset_frame(ds) for ds in DATASETS}
    fits = pd.read_csv(A / "dynamics_fresh" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    fits = fits[(fits["eval"] == dm.PROBE_EVAL) & (fits.model == dm.PROBE_MODEL)
                & (fits.align_tag == "none")].copy()
    fits["subject"] = pd.to_numeric(fits["subject"], errors="coerce")

    # ------------------------------------------------------------ 0. intégrité
    print("=" * 78)
    print("0. INTÉGRITÉ — k_probe constant, sinon on ne lit rien")
    print("=" * 78)
    ok = True
    for ds, m in frames.items():
        ks = sorted(m.k_probe.unique())
        n_fits = int((fits.dataset == ds).sum())
        good = len(ks) == 1 and ks[0] == 15
        ok &= good
        print(f"  {ds:14s} n={len(m):3d} donneurs   k_probe={ks}   "
              f"fits sonde={n_fits:5d}   {'OK' if good else 'ANOMALIE'}")
    if not ok:
        print("\n  REFUS : k_probe n'est pas constant à 15. La sonde est polluée")
        print("  ([[cancelled-cell-pollutes-fits-jsonl]]) — rien n'est lisible.")
        return 1

    # ------------------------------------------------------- 1. porte de fiabilité
    print()
    print("=" * 78)
    print(f"1. PORTE ICC(1,k) — sur la SONDE SEULE, seuil {ICC_FLOOR:.2f}")
    print("=" * 78)
    print(f"  {'dataset':14s} {'variable':16s} {'ICC(1)':>8s} {'ICC(1,k)':>9s} "
          f"{'sqrt':>6s}  verdict")
    gate = {}
    for ds in DATASETS:
        d = fits[fits.dataset == ds]
        for _, _col, raw in PREDICTORS:
            # `.dropna()` AVANT le groupage, et ce n'est pas cosmétique : 4 fits de
            # dreyer2023a n'ont pas de `restored_epoch` (le sujet 42 deux fois, 34 et
            # 40 une fois), et une seule valeur manquante dans un groupe propage un
            # `nan` jusqu'à l'ICC du dataset entier — la porte aurait alors REFUSÉ pour
            # cause de nan et non pour cause de bruit, ce qui est un refus faux.
            groups = [v for v in (g[raw].dropna().to_numpy(float)
                                  for _, g in d.groupby("subject")) if len(v) >= 2]
            n_nan = int(d[raw].isna().sum())
            icc, _n, k = dp.icc1(groups)
            icc_k = k * max(icc, 0) / (1 + (k - 1) * max(icc, 0)) if icc > 0 else 0.0
            gate[(ds, raw)] = icc_k
            print(f"  {ds:14s} {raw:16s} {icc:8.3f} {icc_k:9.3f} {icc_k ** 0.5:6.2f}"
                  f"  {'PASSE' if icc_k >= ICC_FLOOR else 'REFUSÉ'}"
                  f"{f'   ({n_nan} fits sans valeur, écartés)' if n_nan else ''}")
    print("\n  ICC(1) = fiabilité d'UN fit ; ICC(1,k) = celle de leur MOYENNE sur les")
    print("  k=15 réplicats, qui est le prédicteur réel (Spearman-Brown). sqrt(ICC(1,k))")
    print("  plafonne toute corrélation observable : c'est l'atténuation de Spearman.")

    admis = [ds for ds in DATASETS
             if all(gate[(ds, raw)] >= ICC_FLOOR for _, _c, raw in PREDICTORS)]
    refuses = [ds for ds in DATASETS if ds not in admis]

    # Sur un dataset REFUSÉ on ne lit ni H1 ni H2 — mais on a le droit de dire POURQUOI
    # la sonde n'est pas fiable, parce que ce diagnostic porte sur la sonde seule et
    # jamais sur `q_cent`. C'est exactement le périmètre que la préreg autorise.
    for ds in refuses:
        m = frames[ds]
        d = fits[fits.dataset == ds]
        n_ep1 = int((d.restored_epoch <= 1).sum())
        print(f"\n  POURQUOI {ds} EST REFUSÉ (diagnostic sur la sonde seule) :")
        print(f"    acc_probe médiane = {m.acc_probe.median():.3f} "
              f"(ROC-AUC, hasard 0.500), max = {m.acc_probe.max():.3f}")
        print(f"    essais d'entraînement par sujet = "
              f"{int(d.samples.median()) if 'samples' in d else '?'}")
        print(f"    {n_ep1}/{len(d)} fits restaurés à l'époque <= 1 : le meilleur")
        print(f"    valid_acc est atteint AVANT tout apprentissage. Une sonde qui")
        print(f"    n'apprend pas ne classe pas les sujets — l'ICC le mesure, il ne")
        print(f"    l'invente pas.")

    if not admis:
        print("\n  REFUS : aucun dataset ne passe la porte. H1 et H2 ne se lisent pas.")
        return 1
    if refuses:
        print()
        print("  " + "!" * 70)
        print("  L'ENDPOINT PRIMAIRE PRÉ-ENREGISTRÉ N'EST PAS EXÉCUTABLE.")
        print(f"  Il était le test GROUPÉ sur les 110 sujets ; {', '.join(refuses)} est")
        print("  écarté par la porte de fiabilité, donc il ne reste que "
              f"{', '.join(admis)}.")
        print("  Ce qui suit est lu à n réduit, avec le MDE que ça implique. La préreg")
        print("  a calculé ce MDE À L'AVANCE précisément pour ce cas, et il est")
        print("  AU-DESSUS de la cible : un non-rejet ici ne dit rien contre l'effet.")
        print("  " + "!" * 70)

    # ------------------------------------------------------------ 2. garde-fou G
    print()
    print("=" * 78)
    print("2. GARDE-FOU G — acc_probe doit prédire q_cent sur CHAQUE dataset")
    print("=" * 78)
    print(f"  {'dataset':14s} {'n':>4s} {'rho(acc, q_cent)':>18s}  verdict")
    passG = []
    for ds in DATASETS:
        m = frames[ds]
        rho = dm.spearman(m.acc_probe.to_numpy(float), m.q_cent.to_numpy(float))
        n = len(m)
        p = float(stats.norm.sf(np.arctanh(rho) * np.sqrt(n - 3)))
        good = rho > 0 and p < ALPHA
        if good:
            passG.append(ds)
        print(f"  {ds:14s} {n:4d} {rho:+18.3f}   p={p:.4f}  "
              f"{'transporte' if good else 'NE TRANSPORTE PAS'}")
    print("\n  Référence au niveau du pool sur les datasets déjà mesurés : +0.56 /")
    print("  +0.67 / +0.67. Si G échoue, ce n'est pas H1 qui est réfutée : c'est la")
    print("  matrice D×R du dataset qui est cassée, et on ne lit ni H1 ni H2 dessus,")
    print("  ni pour ni contre.")

    lisibles = [ds for ds in admis if ds in passG]
    print(f"\n  DATASETS LISIBLES : {lisibles or 'aucun'}")
    if not lisibles:
        print("  REFUS : la chaîne s'arrête ici. Aucun test n'est exécuté.")
        return 1

    # --------------------------------------------------------- 3-4. H1 puis H2
    df = pd.concat([frames[ds] for ds in lisibles], ignore_index=True)
    print()
    print("=" * 78)
    print(f"3-4. H1 (primaire) et H2 (secondaire) — GROUPÉ sur {lisibles}")
    print("=" * 78)
    print(f"  n groupé = {len(df)} donneurs   MDE = {mde(len(df)):+.3f}   "
          f"cible gelée = {TARGET:+.3f}")

    res = {}
    for tag, col, _raw in PREDICTORS:
        rows = [dict(dataset=ds, n=len(frames[ds]),
                     rho=dm.partial_spearman(frames[ds][col].to_numpy(float),
                                             frames[ds]["q_cent"].to_numpy(float),
                                             frames[ds]["acc_probe"].to_numpy(float)))
                for ds in lisibles]
        pooled = pooled_test(df, col)
        res[tag] = dict(col=col, rows=rows, pooled=pooled,
                        fisher=fisher(rows) if len(rows) > 1 else None)

    # Holm sur la famille des 2, ordre croissant des p.
    order = sorted(res, key=lambda t: res[t]["pooled"]["p"])
    thr = {t: ALPHA / (len(order) - i) for i, t in enumerate(order)}
    for t in order:
        res[t]["holm"] = res[t]["pooled"]["p"] < thr[t]

    for tag, _col, _raw in PREDICTORS:
        r = res[tag]
        p = r["pooled"]
        print(f"\n  {tag} — {r['col']} -> q_cent, acc_probe contrôlé")
        print(f"    {'dataset':14s} {'n':>4s} {'rho_p':>8s} {'MDE':>8s}  (descriptif)")
        for row in r["rows"]:
            print(f"    {row['dataset']:14s} {row['n']:4d} {row['rho']:+8.3f} "
                  f"{mde(row['n']):+8.3f}"
                  f"  {'' if abs(row['rho']) >= mde(row['n']) else 'sous-puissant'}")
        print(f"    GROUPÉ  rho = {p['est']:+.3f}  IC95 [{p['lo']:+.3f}, "
              f"{p['hi']:+.3f}]  p unilatéral = {p['p']:.4f}")
        print(f"    seuil Holm = {thr[tag]:.4f}  ->  "
              f"{'REJETÉ H0' if r['holm'] else 'NON REJETÉ'}")
        if r["fisher"]:
            f = r["fisher"]
            print(f"    Fisher-z (secondaire) rho = {f['rho']:+.3f} "
                  f"[{f['lo']:+.3f}, {f['hi']:+.3f}]  p = {f['p']:.4f}   "
                  f"hétérogénéité Q({f['df']})={f['q']:.2f} p={f['p_q']:.3f} "
                  f"I2={f['i2']:.0f} %")

    # ----------------------------------------------------------- 5. H3 descriptif
    print()
    print("=" * 78)
    print("5. H3 — DESCRIPTIF, aucun test (la préreg l'interdit)")
    print("=" * 78)
    w = within_ranks(df, ["restored_probe", "params_probe", "q_cent", "acc_probe"])

    def second_order(target: str, other: str) -> float:
        """rho(target, q_cent) sachant (acc_probe, other), sur les rangs empilés."""
        Z = np.column_stack([np.ones(len(w)), w["z_acc_probe"], w[f"z_{other}"]])
        ex = w[f"z_{target}"].to_numpy(float)
        ey = w["z_q_cent"].to_numpy(float)
        rx = ex - Z @ np.linalg.lstsq(Z, ex, rcond=None)[0]
        ry = ey - Z @ np.linalg.lstsq(Z, ey, rcond=None)[0]
        return float(stats.pearsonr(rx, ry)[0])

    a = second_order("restored_probe", "params_probe")
    b = second_order("params_probe", "restored_probe")
    print(f"  restored_epoch | (acc_probe, params_probe) = {a:+.3f}")
    print(f"  params_probe   | (acc_probe, restored_epoch) = {b:+.3f}")
    print(f"  -> « supplante » {'tient' if a > b else 'NE TIENT PAS'} "
          f"sur ces données (écart {a - b:+.3f}), DESCRIPTIF.")
    col = {ds: dm.spearman(frames[ds].restored_probe.to_numpy(float),
                           frames[ds].params_probe.to_numpy(float))
           for ds in lisibles}
    print("  rho(restored_epoch, params_probe) intra-dataset : " +
          ", ".join(f"{ds} {v:+.3f}" for ds, v in col.items()))
    print("\n  ET C'EST LA RÉSERVE QUI COMPTE SUR H3. À rho = "
          f"{max(col.values()):+.3f} entre les deux prédicteurs, une partielle du")
    print("  SECOND ordre les sépare mal : chacune répond à « que reste-t-il de l'un")
    print("  quand l'autre a déjà tout pris », sur une variance résiduelle réduite de")
    print(f"  {(1 - max(col.values()) ** 2) * 100:.0f} %. Le SIGNE est informatif — il")
    print("  contredit l'étage C, où `restored_epoch` supplantait `params` — mais")
    print("  l'ÉCART ne l'est pas, et H3 était déjà déclaré sans test pour cette")
    print("  raison exacte ([[restored-epoch-supersedes-params]]).")

    # ------------------------------------------------------------------ verdict
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    # LE STATUT AVANT LE CHIFFRE. La préreg fixe l'endpoint confirmatoire au test
    # GROUPÉ sur les 110 sujets et qualifie explicitement les estimations par dataset
    # de « descriptives uniquement ». La porte de fiabilité ayant écarté liu2024, ce
    # qui reste EST une estimation mono-dataset : elle ne peut donc pas être promue
    # confirmatoire par le fait qu'elle est sortie positive. Écrire « répliqué » ici
    # serait redéfinir l'endpoint après avoir vu le résultat.
    confirmatoire = len(lisibles) == len(DATASETS)
    print(f"  Statut : {'CONFIRMATOIRE' if confirmatoire else 'DESCRIPTIF'} — "
          f"{len(lisibles)}/{len(DATASETS)} datasets lisibles"
          f"{'' if confirmatoire else ', endpoint groupé pré-enregistré non exécutable'}")
    print()
    for tag, _c, _r in PREDICTORS:
        r, p = res[tag], res[tag]["pooled"]
        signe = ("positif" if p["lo"] > 0 else "indéterminé (l'IC contient zéro)")
        holm = "seuil Holm franchi" if r["holm"] else "seuil Holm non franchi"
        print(f"  {tag} {r['col']:16s} {p['est']:+.3f} "
              f"[{p['lo']:+.3f}, {p['hi']:+.3f}]  MDE {mde(p['n']):+.3f}  "
              f"{signe}, {holm}")
    print("\n  À n=60 le MDE (+0.321) est AU-DESSUS de la cible gelée (+0.288) : ce")
    print("  dispositif n'avait pas, par construction, la puissance de trancher. Un")
    print("  résultat positif y reste donc une observation encourageante et non une")
    print("  réplication — le seul énoncé défendable est le chiffre avec son IC.")
    print("\n  Interdits de la préreg, mot pour mot : sur un nul, interdit d'écrire")
    print("  « ne prédit pas », « ne se réplique pas », « c'est du bruit ». Le seul")
    print("  énoncé autorisé est le chiffre avec son IC et son MDE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
