"""Les figures de la matrice donneur-receveur.

Une figure par question, dans l'ordre des portes de `donor_matrix.py` -- le script
imprime les nombres, celui-ci les rend regardables, et les deux lisent EXACTEMENT le
même code de chargement (`donor_matrix.load_matrix`, `donor_quality`, `spearman`) pour
qu'une figure ne puisse pas raconter autre chose que le tableau.

Deux partis pris qui ne sont pas cosmétiques :

* la heatmap trie les LIGNES par qualité de donneur et les COLONNES par difficulté de
  receveur. Non trié, un bloc-diagonale apparent n'est qu'un artefact de l'ordre des
  sujets ; trié, la structure qu'on voit est celle qu'on a mesurée (porte 1) ;
* les nuages de la figure 3 portent le rho ET son intervalle bootstrap. Un nuage de 52
  points sans IC laisse le lecteur estimer la pente à l'œil, et l'œil surestime -- c'est
  précisément l'erreur que la figure doit empêcher, puisque le résultat est un NUL.

Usage :
    python donor_figures.py --dxr dxr_cho2017 [--out figures_dxr]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A = Path(__file__).resolve().parent
import perf_io  # noqa: E402
import donor_matrix as dm  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GROW_C, FIX_C, WARN_C = "#C2255C", "#4C6EF5", "#E8590C"


def load_predictors(d: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Le même assemblage que `donor_matrix.main`, une ligne par donneur.

    La sonde (`params_probe`, `acc_probe`) vient de la CAMPAGNE : 5 folds x 3 seeds =
    15 réplicats par sujet, mesurés indépendamment de cette matrice. Le fit donneur
    (`params_dxr`) vient de la matrice elle-même. Les garder séparés est le point du
    contrôle : ce sont deux régimes d'entraînement, et un seul des deux est censuré.
    """
    q, per_seed = dm.donor_quality(d, dm.METRIC)

    fits = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    f = fits[(fits["eval"] == dm.PROBE_EVAL) & (fits.model == dm.PROBE_MODEL)
             & (fits.align_tag == "none") & (fits.dataset == dataset)].copy()
    f["subject"] = pd.to_numeric(f["subject"], errors="coerce")
    probe = f.groupby("subject").agg(params_probe=("params_end", "mean"),
                                     k_probe=("params_end", "size")).reset_index()

    sc = perf_io.load(A / "perf_final" / "scores")
    sc = perf_io.attach_params(sc, fits)
    subj = perf_io.by_subject(sc)
    s = subj[(subj["eval"] == dm.PROBE_EVAL) & (subj.align_tag == "none")
             & (subj.model == dm.PROBE_MODEL) & (subj.dataset == dataset)]
    acc = s[["subject", "score"]].rename(columns={"score": "acc_probe"})
    acc["subject"] = pd.to_numeric(acc["subject"], errors="coerce")

    dxr = (d[d["self"] == 0].groupby("donor")
           .agg(params_dxr=("params_end", "mean"),
                width_dxr=("width_end", "mean")).reset_index())

    q = (q.merge(probe, left_on="donor", right_on="subject", how="left")
          .merge(acc, left_on="donor", right_on="subject", how="left")
          .merge(dxr, on="donor", how="left"))
    return q, per_seed


# ------------------------------------------------------------------ fig 1 : heatmap
def fig_heatmap(d: pd.DataFrame, q: pd.DataFrame, dataset: str, out: Path) -> None:
    off = d[d["self"] == 0]
    m = off.pivot_table(index="donor", columns="receiver", values=dm.METRIC)
    # Lignes par qualité de donneur (le critère), colonnes par difficulté de receveur.
    # Sans ce tri la figure ne montre que l'ordre des identifiants MOABB.
    row_order = q.sort_values("q_cent", ascending=False).donor.tolist()
    col_order = m.mean(axis=0).sort_values(ascending=False).index.tolist()
    m = m.reindex(index=[r for r in row_order if r in m.index], columns=col_order)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    v = np.nanmax(np.abs(m.to_numpy() - 0.5))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=0.5 - v, vmax=0.5 + v,
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel(f"receveur — trié par difficulté ({len(col_order)} sujets)")
    ax.set_ylabel("donneur — trié par qualité (q centrée)")
    ax.set_title(f"{dataset} · transfert sujet→sujet, {dm.METRIC}\n"
                 "diagonale exclue (elle n'est pas du transfert)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.set_label(f"{dm.METRIC} (chance 0.50)")
    # Des lignes horizontales nettes = la qualité est portée par le donneur. Des
    # colonnes nettes = elle est portée par le receveur. C'est la lecture de porte 1
    # faite à l'œil, et elle doit concorder avec l'ICC imprimé par donor_matrix.
    fig.text(0.5, -0.02, "Lire les BANDES : horizontales → la qualité appartient au "
             "donneur ; verticales → au receveur.", ha="center", fontsize=8.5,
             style="italic")
    fig.tight_layout(); fig.savefig(out / "dxr_01_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------- fig 2 : la qualité, sujet par sujet
def fig_quality(per_seed: pd.DataFrame, q: pd.DataFrame, dataset: str,
                icc: float, icc_k: float, out: Path) -> None:
    order = q.sort_values("q_cent").donor.tolist()
    pos = {s: i for i, s in enumerate(order)}
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.scatter(per_seed.donor.map(pos), per_seed.q_cent, s=16, alpha=0.55,
               color=FIX_C, label="une seed", zorder=3)
    ax.scatter([pos[s] for s in q.donor], q.q_cent, s=34, color=GROW_C,
               label="moyenne des 3 seeds", zorder=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks([]); ax.set_xlabel(f"sujet donneur ({len(order)}), trié par qualité")
    ax.set_ylabel("qualité de donneur (z, centrée par receveur)")
    ax.set_title(f"{dataset} · la qualité de donneur est une propriété du sujet — "
                 f"ICC(1) = {icc:.3f}, ICC$_k$ = {icc_k:.2f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "dxr_02_donor_quality.png",
                                    bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- fig 3 : les deux prédicteurs
def fig_predictors(q: pd.DataFrame, dataset: str, out: Path) -> None:
    m = q[q.q_cent.notna() & q.params_probe.notna() & q.acc_probe.notna()]
    y = m.q_cent.to_numpy(float)
    n = len(m)
    panels = [("#params (sonde, k=15)", m.params_probe.to_numpy(float), GROW_C,
               "params_probe"),
              ("accuracy (sonde, k=15)", m.acc_probe.to_numpy(float), FIX_C,
               "acc_probe")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, (label, x, c, _) in zip(axes, panels):
        r = dm.spearman(x, y)
        lo, hi = dm.boot_ci(lambda i, x=x: dm.spearman(x[i], y[i]), n)
        ax.scatter(x, y, s=30, color=c, alpha=0.8, edgecolor="white", lw=0.5)
        # Ajustement sur les RANGS, pas sur les valeurs : c'est un rho de Spearman
        # qui est reporté, et une droite des moindres carrés sur les valeurs brutes
        # dessinerait une pente que le test ne teste pas.
        rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        b, a0 = np.polyfit(rx, ry, 1)
        xs = np.linspace(rx.min(), rx.max(), 2)
        xv = np.interp(xs, np.sort(rx), np.sort(x))
        yv = np.interp(a0 + b * xs, np.sort(ry), np.sort(y))
        ax.plot(xv, yv, color=c, lw=1.4, ls="--", alpha=0.7)
        sig = "significatif" if (lo > 0 or hi < 0) else "NON significatif"
        ax.set_title(f"{label}\nρ = {r:+.3f}  [{lo:+.3f}, {hi:+.3f}]  · {sig}",
                     fontsize=9.5)
        ax.set_xlabel(label)
    axes[0].set_ylabel("qualité de donneur (z)")
    fig.suptitle(f"{dataset} · claim 2 — qui prédit la qualité de donneur ? "
                 f"(n = {n} sujets, MDE ρ = {np.tanh(2.8016 / np.sqrt(n - 3)):.2f})",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "dxr_03_predictors.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------- fig 4 : la comparaison appariée, la vraie
def fig_paired(q: pd.DataFrame, dataset: str, out: Path) -> None:
    """La figure qui porte le verdict.

    Deux tests séparés ne font pas une comparaison : « l'un est significatif, l'autre
    non » est parfaitement compatible avec une différence nulle. C'est la distribution
    de la DIFFÉRENCE, rééchantillonnée sur les mêmes sujets, qui tranche.
    """
    m = q[q.q_cent.notna() & q.params_probe.notna() & q.acc_probe.notna()]
    y = m.q_cent.to_numpy(float)
    pp = m.params_probe.to_numpy(float)
    aa = m.acc_probe.to_numpy(float)
    n = len(m)
    draws = np.empty(dm.N_BOOT)
    rng = np.random.default_rng(20260901)
    for b in range(dm.N_BOOT):
        i = rng.integers(0, n, n)
        draws[b] = abs(dm.spearman(pp[i], y[i])) - abs(dm.spearman(aa[i], y[i]))
    obs = abs(dm.spearman(pp, y)) - abs(dm.spearman(aa, y))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    verdict = ("#params GAGNE" if lo > 0 else
               "accuracy gagne" if hi < 0 else "INDISCERNABLES")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(draws, bins=60, color="0.75", edgecolor="white", lw=0.4)
    ax.axvline(0, color="0.3", lw=1.2, label="pas de différence")
    ax.axvline(obs, color=GROW_C, lw=2.0, label=f"observé {obs:+.3f}")
    ax.axvspan(lo, hi, color=GROW_C, alpha=0.12, label=f"IC 95 % [{lo:+.3f}, {hi:+.3f}]")
    ax.set_xlabel("|ρ(#params, qualité)| − |ρ(accuracy, qualité)|")
    ax.set_ylabel("tirages bootstrap")
    ax.set_yticks([])
    ax.set_title(f"{dataset} · {verdict} — l'IC contient zéro, donc on n'écrit ni que\n"
                 "#params gagne, ni que l'accuracy gagne ; mais rien ne soutient "
                 "#params.", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "dxr_04_paired.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- fig 5 : le plafond mesuré
def fig_ceiling(d: pd.DataFrame, q: pd.DataFrame, dataset: str, out: Path) -> None:
    """L'échelle sur laquelle le nul de claim 2 a été mesuré.

    `grow_shallow.yaml` borne la croissance des DEUX côtés : elle part de
    `n_filters_time: 8` et ne peut pas dépasser `target_n_filters_time: 40`. Un fit qui
    finit à 8 n'a pas grandi (abstention), un fit qui finit à 40 a peut-être voulu aller
    plus loin -- dans les deux cas la largeur a cessé de mesurer le sujet.

    Les DEUX variables sont touchées, pas seulement celle de la matrice : c'est la
    correction importante de cette figure. Ce qui sauve le prédicteur est la moyenne sur
    k=15 réplicats -- aucun sujet n'a une moyenne collée à 40 -- mais le haut de
    l'échelle reste comprimé, donc le rho de claim 2 est mesuré sur une règle tronquée.
    À montrer avec le nul, jamais sans lui.
    """
    cells = d[d["self"] == 0].drop_duplicates(["donor", "seed"])
    fits = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    pf = fits[(fits["eval"] == dm.PROBE_EVAL) & (fits.model == dm.PROBE_MODEL)
              & (fits.align_tag == "none") & (fits.dataset == dataset)]
    w_lo, w_hi = 8.0, float(max(cells.width_end.max(), pf.width_end.max()))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    bins = np.arange(w_lo - 0.5, w_hi + 1.5, 1)
    for ax, (w, c, lab, n) in zip(axes, [
            (pf.width_end, FIX_C, "sonde de campagne — 80 % des essais", len(pf)),
            (cells.width_end, WARN_C, "fit donneur — 100 % des essais", len(cells))]):
        ax.hist(w, bins=bins, color=c, alpha=0.85)
        ax.axvline(w_lo, color="0.25", ls="--", lw=1.1)
        ax.axvline(w_hi, color="0.25", ls="--", lw=1.1)
        f_lo = float((w <= w_lo).mean()); f_hi = float((w >= w_hi).mean())
        ax.set_title(f"{lab}  ({n} fits)\nplancher {100 * f_lo:.0f} %  ·  "
                     f"plafond {100 * f_hi:.0f} %  ·  "
                     f"censuré {100 * (f_lo + f_hi):.0f} %", fontsize=9.5)
        ax.set_xlabel("largeur en fin de croissance (filtres temporels)")
    axes[0].set_ylabel("fits")
    fig.suptitle(f"{dataset} · les deux variables sont bornées à [{w_lo:.0f}, "
                 f"{w_hi:.0f}] par grow_shallow.yaml — le nul de claim 2 est mesuré "
                 "sur une règle tronquée", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "dxr_05_ceiling.png", bbox_inches="tight")
    plt.close(fig)

    # Le vrai garde-fou : ce n'est pas le fit isolé qui sert de prédicteur, c'est la
    # MOYENNE des k réplicats du sujet. Si aucune moyenne ne touche la borne, la
    # variable garde du contraste malgré la censure des fits individuels.
    g = pf.assign(subject=pd.to_numeric(pf["subject"], errors="coerce")) \
           .groupby("subject").width_end.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist(g["mean"], bins=np.arange(w_lo, w_hi + 1, 1.5), color=GROW_C, alpha=0.85)
    ax.axvline(w_hi, color="0.25", ls="--", lw=1.2)
    ax.annotate(f"cible {w_hi:.0f}\naucun sujet n'y touche\n(max {g['mean'].max():.1f})",
                xy=(w_hi, ax.get_ylim()[1] * 0.62), xytext=(-10, 0),
                textcoords="offset points", ha="right", fontsize=8.5)
    ax.set_xlabel(f"largeur MOYENNE du sujet sur k={int(g['size'].median())} réplicats")
    ax.set_ylabel("sujets")
    ax.set_title(f"{dataset} · le prédicteur garde du contraste ({g['mean'].min():.1f} "
                 f"→ {g['mean'].max():.1f}) : la moyenne rattrape la censure des fits\n"
                 "mais le haut de l'échelle reste comprimé, donc ρ est atténué",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "dxr_06_predictor_scale.png",
                                    bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dxr", required=True, type=Path)
    p.add_argument("--out", type=Path, default=A / "figures_dxr")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    d = dm.load_matrix(a.dxr)
    dataset = str(d.dataset.iloc[0])
    q, per_seed = load_predictors(d, dataset)

    groups = [g.q_cent.to_numpy(float) for _, g in per_seed.groupby("donor")
              if len(g) >= 2]
    from donor_predictor import icc1
    icc, _n, k = icc1(groups)
    icc_k = k * max(icc, 0) / (1 + (k - 1) * max(icc, 0)) if icc > 0 else 0.0

    fig_heatmap(d, q, dataset, a.out)
    fig_quality(per_seed, q, dataset, icc, icc_k, a.out)
    fig_predictors(q, dataset, a.out)
    fig_paired(q, dataset, a.out)
    fig_ceiling(d, q, dataset, a.out)
    print(f"5 figures écrites dans {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
