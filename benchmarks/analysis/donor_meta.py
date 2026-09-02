"""Claim 2 combinée sur les quatre matrices D×R : l'estimation qui décide ICLR.

`donor_summary.py` met les quatre rho partielles côte à côte et s'arrête là. Ce
fichier fait le pas d'après, celui qui manquait pour trancher : les COMBINER. C'est
nécessaire parce que la lecture dataset par dataset est trompeuse dans les deux sens.

    physionetmi   n=109   rho_p = +0.385   MDE 0.272   -> significatif
    lee2019_mi    n= 54   rho_p = +0.229   MDE 0.392   -> sous-puissant
    cho2017       n= 52   rho_p = +0.150   MDE 0.400   -> sous-puissant
    bnci2014_001  n=  9   rho_p = +0.239   MDE 1.143   -> ininterprétable

Annoncer « ça réplique sur 1 dataset sur 4 » serait faux : trois des quatre datasets
n'avaient pas la puissance de détecter l'effet mesuré sur le quatrième, donc leur
non-significativité n'est pas une information contre lui ([[underpowered-not-null]]).
Annoncer « les quatre sont positives » serait tout aussi creux sans intervalle joint.
La quantité qui répond est l'estimation COMBINÉE et son intervalle.

L'ANALYSE PRIMAIRE : DATASETS FIXES, DONNEURS RÉ-ÉCHANTILLONNÉS
    Les quatre datasets ne sont pas un échantillon aléatoire de datasets : ce sont
    ceux qu'on a. Un modèle à effets aléatoires les traiterait comme tirés d'une
    population de datasets et estimerait une variance inter-datasets sur k=4 — un
    tau^2 sur 4 points n'a aucune précision, et l'intervalle qui en sort est une
    fiction. On fixe donc les datasets et on ne ré-échantillonne que ce qui est
    réellement échantillonné : les DONNEURS, stratifiés par dataset.

    Conséquence à écrire dans le papier, pas à cacher : cette estimation ne
    généralise pas à un cinquième dataset. Elle répond à « l'effet est-il présent sur
    ces quatre-là, vu conjointement », ce qui est exactement la question de claim 2.

    L'agrégation elle-même est faite sur les rangs INTRA-dataset : `#params` n'a ni la
    même échelle ni le même plafond d'un dataset à l'autre (40 k–104 k sur cho2017,
    41 k–46 k sur bnci2014_001), donc empiler les valeurs brutes créerait une
    corrélation entre datasets qui n'existe dans aucun. On empile des rangs centrés,
    la corrélation reste intra-dataset par construction.

LA SECONDAIRE : FISHER-Z, POUR ÊTRE COMPARABLE À LA LITTÉRATURE
    Effets fixes par inverse-variance sur z = atanh(rho), SE = 1/sqrt(n-4) (le -4 et
    non -3 parce que c'est une partielle du premier ordre : une covariable consommée).
    On imprime Q et I^2 pour que l'hétérogénéité soit visible, et l'effet aléatoire
    DerSimonian-Laird À TITRE INDICATIF UNIQUEMENT, avec son avertissement k=4.

L'EXCLUSION DE bnci2014_001, DÉCIDÉE SUR UN CRITÈRE QUI N'EST PAS LE RÉSULTAT
    93 % des fits de la sonde y finissent collés à `target_n_filters_time = 40`. Le
    prédicteur n'est donc pas une mesure sur ce dataset, c'est une borne : son
    étendue est 41 k–46 k contre 40 k–104 k sur cho2017. Une variable censurée à 93 %
    n'a plus la variance nécessaire pour corréler avec quoi que ce soit.

    Le critère est `ceil_probe >= 0.50`, il porte sur la SONDE et pas sur la qualité
    des donneurs, et il aurait été le même si la rho de bnci2014_001 avait été la plus
    forte des quatre (elle est à +0.239, au milieu). Les deux analyses sont imprimées,
    avec et sans, pour que l'exclusion ne travaille pas en douce.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_meta.py
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

DATASETS = ["physionetmi", "lee2019_mi", "cho2017", "bnci2014_001"]
CEIL_MAX = 0.50           # critère d'exclusion, sur la sonde, déclaré ci-dessus
N_BOOT = 20000
RNG = np.random.default_rng(20260902)


def dataset_frame(ds: str) -> pd.DataFrame:
    """Une ligne par donneur : le prédicteur, la covariable, la qualité de donneur."""
    d = dm.load_matrix(A / f"dxr_{ds}")
    q, _ = dm.donor_quality(d, dm.METRIC)
    rank = pd.read_csv(A / f"ranking_{ds}.csv")
    m = q.merge(rank, left_on="donor", right_on="subject", how="inner")
    keep = ["donor", "q_cent", "params_probe", "acc_probe", "ceil_frac"]
    m = m[keep].dropna().reset_index(drop=True)
    m["dataset"] = ds
    return m


def partial(df: pd.DataFrame) -> float:
    return dm.partial_spearman(
        df["params_probe"].to_numpy(float),
        df["q_cent"].to_numpy(float),
        df["acc_probe"].to_numpy(float))


def within_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rangs centrés-réduits INTRA-dataset, pour que l'empilement reste intra."""
    out = []
    for ds, g in df.groupby("dataset", sort=False):
        g = g.copy()
        for c in ("params_probe", "q_cent", "acc_probe"):
            r = stats.rankdata(g[c].to_numpy(float))
            out_c = (r - r.mean()) / (r.std(ddof=1) if r.std(ddof=1) > 0 else 1.0)
            g[f"z_{c}"] = out_c
        out.append(g)
    return pd.concat(out, ignore_index=True)


def pooled_partial(df: pd.DataFrame) -> float:
    """rho partielle sur les rangs empilés : résidus de z_params et z_q sur z_acc."""
    x = df["z_params_probe"].to_numpy(float)
    y = df["z_q_cent"].to_numpy(float)
    z = df["z_acc_probe"].to_numpy(float)
    ex = x - np.polyval(np.polyfit(z, x, 1), z)
    ey = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(stats.pearsonr(ex, ey)[0])


def pooled_ci(df: pd.DataFrame) -> tuple[float, float, float]:
    """Bootstrap des DONNEURS stratifié par dataset (les datasets sont fixes)."""
    groups = [g.index.to_numpy() for _, g in df.groupby("dataset", sort=False)]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([RNG.choice(g, size=len(g), replace=True)
                              for g in groups])
        b = within_ranks(df.loc[idx].reset_index(drop=True))
        v = pooled_partial(b)
        if np.isfinite(v):
            vals.append(v)
    v = np.asarray(vals)
    # p bilatéral par inversion : la part de la distribution de l'autre côté de zéro.
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
            float(max(p, 1.0 / len(v))))


def fisher(rows: list[dict]) -> dict:
    """Effets fixes inverse-variance sur z=atanh(rho), SE=1/sqrt(n-4)."""
    z = np.array([np.arctanh(r["rho_p"]) for r in rows])
    se = np.array([1.0 / np.sqrt(r["n"] - 4) for r in rows])
    w = 1.0 / se ** 2
    zf = float((w * z).sum() / w.sum())
    sef = float(np.sqrt(1.0 / w.sum()))
    q = float((w * (z - zf) ** 2).sum())
    k = len(rows)
    i2 = float(max(0.0, (q - (k - 1)) / q) * 100) if q > 0 else 0.0
    # DerSimonian-Laird, imprimé mais non décisionnel (k=4).
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1.0 / (se ** 2 + tau2)
    zr = float((wr * z).sum() / wr.sum())
    ser = float(np.sqrt(1.0 / wr.sum()))
    return dict(
        rho_fe=float(np.tanh(zf)),
        lo_fe=float(np.tanh(zf - 1.96 * sef)), hi_fe=float(np.tanh(zf + 1.96 * sef)),
        p_fe=float(2 * stats.norm.sf(abs(zf / sef))),
        q=q, df=k - 1, p_q=float(stats.chi2.sf(q, k - 1)), i2=i2, tau2=tau2,
        rho_re=float(np.tanh(zr)),
        lo_re=float(np.tanh(zr - 1.96 * ser)), hi_re=float(np.tanh(zr + 1.96 * ser)))


def report(df: pd.DataFrame, rows: list[dict], label: str) -> None:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print(f"{'dataset':>14} {'n':>4} {'rho_p':>8} {'ceil':>7}")
    for r in rows:
        print(f"{r['dataset']:>14} {r['n']:>4} {r['rho_p']:>+8.3f} {r['ceil']:>7.2f}")

    w = within_ranks(df)
    est = pooled_partial(w)
    lo, hi, p = pooled_ci(df.reset_index(drop=True))
    print(f"\n  PRIMAIRE  datasets fixes, bootstrap des donneurs stratifié")
    print(f"    n = {len(df)} donneurs sur {df.dataset.nunique()} datasets")
    print(f"    rho partielle groupée = {est:+.3f}   IC95 [{lo:+.3f}, {hi:+.3f}]"
          f"   p = {p:.4f}")
    verdict = ("EFFET NON NUL" if lo > 0 else
               "INDÉTERMINÉ (l'IC contient zéro)")
    print(f"    -> {verdict}")

    f = fisher(rows)
    print(f"\n  SECONDAIRE  Fisher-z, effets fixes")
    print(f"    rho = {f['rho_fe']:+.3f}  IC95 [{f['lo_fe']:+.3f}, {f['hi_fe']:+.3f}]"
          f"  p = {f['p_fe']:.4f}")
    print(f"    hétérogénéité : Q({f['df']}) = {f['q']:.2f}, p = {f['p_q']:.3f}, "
          f"I2 = {f['i2']:.0f} %")
    print(f"    DL aléatoire (INDICATIF, k={len(rows)}) : {f['rho_re']:+.3f} "
          f"[{f['lo_re']:+.3f}, {f['hi_re']:+.3f}]  tau2 = {f['tau2']:.4f}")


def main(argv=None) -> int:
    frames, rows = [], []
    for ds in DATASETS:
        m = dataset_frame(ds)
        frames.append(m)
        rows.append(dict(dataset=ds, n=len(m), rho_p=partial(m),
                         ceil=float(m["ceil_frac"].mean())))
    all_df = pd.concat(frames, ignore_index=True)

    kept = [r["dataset"] for r in rows if r["ceil"] < CEIL_MAX]
    dropped = [r["dataset"] for r in rows if r["ceil"] >= CEIL_MAX]

    report(all_df, rows, "A. LES QUATRE DATASETS (aucune exclusion)")
    if dropped:
        sub = all_df[all_df.dataset.isin(kept)].reset_index(drop=True)
        srows = [r for r in rows if r["dataset"] in kept]
        report(sub, srows,
               f"B. PRÉ-SPÉCIFIÉE : ceil_probe < {CEIL_MAX:.2f} "
               f"(exclu : {', '.join(dropped)})")

    print(f"\n{'=' * 74}")
    print("B est l'analyse à rapporter ; A est là pour montrer que l'exclusion ne")
    print("change pas le sens de la conclusion. Si A et B divergent en VERDICT, ne")
    print("rapporter ni l'une ni l'autre sans dire laquelle a été choisie et quand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
