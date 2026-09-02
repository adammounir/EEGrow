"""Les quatre matrices D×R côte à côte — la figure qui décide de la narrative.

`donor_figures.py` regarde UNE matrice à la fois. Ce fichier fait la seule chose qu'on
ne peut pas faire dataset par dataset : mettre les quatre estimations de claim 2 sur le
même axe, avec leurs intervalles, leur n et leur MDE. C'est nécessaire parce que le
résultat de cho2017 (rho partielle +0.150, IC qui touche zéro) a été lu au départ comme
une réfutation, et qu'il ne l'est pas : son IC [-0.106, +0.421] CONTIENT l'estimation de
physionetmi (+0.385). Un nul dont l'IC couvre l'effet mesuré ailleurs est un manque de
puissance, pas un désaccord ([[underpowered-not-null]]).

CE QUI EST TRACÉ, ET POURQUOI CETTE QUANTITÉ-LÀ
    La marginale rho(#params, qualité) n'est pas lisible sur lee2019_mi et physionetmi :
    `#params` y corrèle avec l'accuracy du sujet (+0.52 / +0.65 à l'étage 0), donc une
    marginale positive peut n'être qu'un relais de l'accuracy. La quantité qui répond à
    claim 2 sur les quatre datasets est la RHO PARTIELLE à accuracy contrôlée : ce que
    la taille finale du réseau sait du sujet et que son accuracy ne sait pas. C'est elle
    qui est en gras, les marginales sont là pour le contexte.

LA CO-VARIABLE À NE PAS CACHER
    Le panneau de droite porte le taux de saturation de la sonde (part des fits qui
    finissent à `target_n_filters_time = 40`). Il va de 18 % sur physionetmi à 93 % sur
    bnci2014_001. Une variable censurée à 93 % n'a plus de variance à corréler : le
    dataset où claim 2 « échoue » le plus fort est aussi celui où le prédicteur n'existe
    quasiment plus. Publier la forêt sans ce panneau serait présenter comme un résultat
    ce qui est en partie un artefact de plafond.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import donor_matrix as dm  # noqa: E402
import donor_figures as df_  # noqa: E402
from donor_predictor import icc1  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GROW_C, FIX_C, WARN_C = "#C2255C", "#4C6EF5", "#E8590C"

DATASETS = ["bnci2014_001", "cho2017", "lee2019_mi", "physionetmi"]


def collect(dataset: str) -> dict:
    """Un dataset → une ligne de la forêt. Rien n'est recalculé à la main ici.

    Toutes les statistiques passent par `donor_matrix` (même centrage par receveur,
    même bootstrap apparié sur les donneurs, même graine) : la figure doit être une
    vue du tableau, pas une seconde implémentation qui pourrait en diverger.
    """
    d = dm.load_matrix(A / f"dxr_{dataset}")
    q, per_seed = df_.load_predictors(d, dataset)
    q = q.dropna(subset=["q_cent", "params_probe", "acc_probe"])
    n = len(q)
    y = q["q_cent"].to_numpy(float)
    xp = q["params_probe"].to_numpy(float)
    xa = q["acc_probe"].to_numpy(float)

    def ci(fn):
        return dm.boot_ci(fn, n)

    out = {"dataset": dataset, "n": n}
    out["rho_p"] = dm.spearman(xp, y)
    out["rho_p_ci"] = ci(lambda i: dm.spearman(xp[i], y[i]))
    out["rho_a"] = dm.spearman(xa, y)
    out["rho_a_ci"] = ci(lambda i: dm.spearman(xa[i], y[i]))
    out["par"] = dm.partial_spearman(xp, y, xa)
    out["par_ci"] = ci(lambda i: dm.partial_spearman(xp[i], y[i], xa[i]))
    out["mde"] = 2.8 / np.sqrt(max(n - 3, 1))  # z de puissance 80 % / alpha 5 %

    # Portes 0 et 1, pour pouvoir dire à quelles lignes de la forêt on a le droit de
    # croire : une corrélation avec un critère non fiable est bornée par sqrt(ICC_k).
    off = d[d["self"] == 0]
    out["above"] = float((off[dm.METRIC] - off["chance"]).mean())
    groups = [g.q_cent.to_numpy(float) for _, g in per_seed.groupby("donor")
              if len(g) >= 2]
    icc, _, k = icc1(groups)
    icc = max(icc, 0.0)
    out["icc_k"] = k * icc / (1 + (k - 1) * icc) if icc > 0 else 0.0

    fits = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    f = fits[(fits["eval"] == dm.PROBE_EVAL) & (fits.model == dm.PROBE_MODEL)
             & (fits.align_tag == "none") & (fits.dataset == dataset)]
    out["ceil_probe"] = float((f.width_end >= 40).mean())
    out["floor_probe"] = float((f.width_end <= 8).mean())
    out["ceil_dxr"] = float((off.drop_duplicates(["donor", "seed"]).width_end >= 40).mean())
    out["q"] = q
    return out


def fig_forest(rows: list[dict], out: Path) -> None:
    """Les quatre datasets sur un axe, la censure de la sonde à côté.

    La lecture voulue est verticale : la partielle monte quand la saturation descend.
    Quatre points ne font pas une tendance, et la figure ne prétend pas le contraire —
    elle interdit seulement de lire le +0.150 de cho2017 comme un fait sur la croissance
    alors qu'il est aussi un fait sur l'échelle de mesure.
    """
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [2.6, 1]})
    ys = np.arange(len(rows))[::-1]
    off = {"rho_p": +0.22, "rho_a": 0.0, "par": -0.22}
    style = {"rho_p": (GROW_C, "o", r"$\rho$(#params, qualité)"),
             "rho_a": (FIX_C, "s", r"$\rho$(accuracy, qualité)"),
             "par": ("#212529", "D", r"$\rho$ partielle (#params | accuracy)")}
    for key, (c, mk, lab) in style.items():
        for yi, r in zip(ys, rows):
            v, (lo, hi) = r[key], r[f"{key}_ci"]
            if not np.isfinite(v):
                continue
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [yi + off[key]] * 2, color=c, lw=1.4, alpha=.85)
            ax.plot(v, yi + off[key], mk, color=c, ms=7 if key == "par" else 5.5,
                    mfc=c if sig else "white", mew=1.4, zorder=3,
                    label=lab if yi == ys[0] else None)
    ax.axvline(0, color="#adb5bd", lw=1, zorder=0)
    for yi, r in zip(ys, rows):
        ax.axhline(yi - 0.5, color="#e9ecef", lw=.8, zorder=0)
        ax.plot([-r["mde"], r["mde"]], [yi - 0.42] * 2, color="#adb5bd", lw=5,
                alpha=.55, solid_capstyle="butt", zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['dataset']}\nn={r['n']}  MDE {r['mde']:.2f}" for r in rows])
    ax.set_xlabel(r"$\rho$ de Spearman avec la qualité de donneur (centrée par receveur)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_title("Claim 2 sur les quatre matrices D×R\n"
                 "plein = IC exclut 0 ; barre grise = zone indétectable (MDE)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=.9)

    for yi, r in zip(ys, rows):
        ax2.barh(yi, r["ceil_probe"] * 100, color=WARN_C, alpha=.85, height=.5)
        ax2.barh(yi, -r["floor_probe"] * 100, color="#868e96", alpha=.6, height=.5)
        ax2.text(r["ceil_probe"] * 100 + 2, yi, f"{r['ceil_probe']*100:.0f} %",
                 va="center", fontsize=7.5, color=WARN_C)
    ax2.axvline(0, color="#495057", lw=1)
    ax2.set_yticks(ys); ax2.set_yticklabels([])
    ax2.set_ylim(-0.75, len(rows) - 0.25)
    ax2.set_xlabel("% des fits de la sonde\n← plancher (8)     plafond (40) →")
    ax2.set_title("Le prédicteur a-t-il encore\nde la variance ?", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "dxr_summary_01_forest.png", bbox_inches="tight")
    plt.close(fig)


def fig_scatter(rows: list[dict], out: Path) -> None:
    """Un sujet = un point, sur les quatre datasets — la décomposition demandée.

    L'axe x est le résidu de `#params` après régression sur l'accuracy (rangs), pas
    `#params` brut : c'est la variable dont la partielle mesure la pente, et la tracer
    telle quelle évite de montrer une pente que l'accuracy explique déjà.
    """
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, r in zip(axes, rows):
        q = r["q"]
        if len(q) < 4 or not np.isfinite(r["par"]):
            ax.text(.5, .5, "sonde indisponible", ha="center", va="center",
                    transform=ax.transAxes, color="#868e96")
            ax.set_title(r["dataset"], fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
            continue
        rp = pd.Series(q["params_probe"].to_numpy(float)).rank().to_numpy()
        ra = pd.Series(q["acc_probe"].to_numpy(float)).rank().to_numpy()
        ry = pd.Series(q["q_cent"].to_numpy(float)).rank().to_numpy()
        ex = rp - np.polyval(np.polyfit(ra, rp, 1), ra)
        ey = ry - np.polyval(np.polyfit(ra, ry, 1), ra)
        sig = r["par_ci"][0] > 0 or r["par_ci"][1] < 0
        ax.axhline(0, color="#dee2e6", lw=.8); ax.axvline(0, color="#dee2e6", lw=.8)
        ax.scatter(ex, ey, s=22, color=GROW_C if sig else "#868e96",
                   alpha=.75, edgecolor="white", lw=.5)
        b = np.polyfit(ex, ey, 1)
        xs = np.linspace(ex.min(), ex.max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=GROW_C if sig else "#868e96",
                lw=1.8, ls="-" if sig else "--")
        lo, hi = r["par_ci"]
        ax.set_title(f"{r['dataset']}  (n={r['n']})\n"
                     rf"$\rho$ part. = {r['par']:+.3f} [{lo:+.2f}, {hi:+.2f}]"
                     + ("  *" if sig else ""), fontsize=8.5)
        ax.set_xlabel("#params, résidu | accuracy (rangs)", fontsize=8)
    axes[0].set_ylabel("qualité de donneur,\nrésidu | accuracy (rangs)", fontsize=8)
    fig.suptitle("Ce que la taille finale du réseau sait du sujet et que son accuracy "
                 "ne sait pas — un point = un sujet", fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out / "dxr_summary_02_partial_scatter.png", bbox_inches="tight")
    plt.close(fig)


def fig_gates(rows: list[dict], out: Path) -> None:
    """Les portes 0 et 1, sans lesquelles la forêt ne veut rien dire.

    Un rho élevé sur un dataset où il n'y a pas de transfert (porte 0) ou dont la
    qualité de donneur n'est pas reproductible d'une seed à l'autre (porte 1) serait une
    corrélation avec du bruit. La figure les met en regard pour qu'on ne cite jamais une
    ligne de la forêt sans savoir sur quel socle elle repose.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ys = np.arange(len(rows))[::-1]
    names = [r["dataset"] for r in rows]
    a1.barh(ys, [r["above"] for r in rows], color=FIX_C, alpha=.85, height=.55)
    for yi, r in zip(ys, rows):
        a1.text(r["above"] + .004, yi, f"{r['above']:+.3f}", va="center", fontsize=7.5)
    a1.set_yticks(ys); a1.set_yticklabels(names, fontsize=8)
    a1.set_xlabel("ROC-AUC de transfert au-dessus de la chance")
    a1.set_title("Porte 0 — y a-t-il du transfert ?", fontsize=9.5, loc="left")
    a2.barh(ys, [r["icc_k"] for r in rows], color=GROW_C, alpha=.85, height=.55)
    a2.axvline(.75, color="#495057", ls=":", lw=1.2)
    for yi, r in zip(ys, rows):
        a2.text(r["icc_k"] + .01, yi, f"{r['icc_k']:.2f}", va="center", fontsize=7.5)
    a2.set_yticks(ys); a2.set_yticklabels([])
    a2.set_xlim(0, 1.12); a2.set_xlabel("ICC_k (moyenne des 3 seeds)")
    a2.set_title("Porte 1 — le critère est-il stable ?", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "dxr_summary_03_gates.png", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    out = A / "figures_dxr_summary"
    out.mkdir(parents=True, exist_ok=True)
    rows = [collect(d) for d in DATASETS]
    fig_forest(rows, out)
    fig_scatter(rows, out)
    fig_gates(rows, out)

    cols = ["dataset", "n", "rho_p", "rho_a", "par", "mde", "icc_k",
            "ceil_probe", "ceil_dxr", "above"]
    t = pd.DataFrame([{c: r[c] for c in cols} for r in rows])
    t.to_csv(out / "dxr_summary.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n3 figures + table dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
