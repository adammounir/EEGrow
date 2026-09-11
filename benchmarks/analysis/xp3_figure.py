"""La figure XP 3, version soumettable : le basculement, avec ses IC et son nul mesuré.

Reprend la mise en forme de la version du 09/09 — celle qu'Adam a validée — et lui
ajoute ce qui lui manquait pour être publiable : les intervalles de confiance sur les
deux parts, la valeur du nul MESURÉE par permutation au lieu du « 33 % » nominal, et
une seconde planche qui répond d'avance à la question du relecteur (« quel dataset
porte l'effet ? »).

DEUX DÉCISIONS DE MISE EN FORME, toutes deux nécessaires
--------------------------------------------------------
1. l'axe x est un PERCENTILE INTRA-DATASET, pas un nombre de paramètres. Un
   `params_end` de 30 000 est énorme sur un dataset et médian sur un autre ; empiler
   les 12 en unités brutes mélangerait « gros modèle » et « dataset qui fait pousser
   gros ».
2. la couleur est un TERCILE DE QUALITÉ DE DONNEUR calculé DANS son dataset. Les
   terciles étant équilibrés par construction et l'axe étant un percentile,
   l'histogramme TOTAL est plat : tout ce qui se lit est la COMPOSITION de chaque
   barre. Sans relation, chaque barre serait un tiers de chaque couleur d'un bout à
   l'autre — et la permutation confirme que c'est bien 32.9 %, pas une autre valeur.

Le panneau de DROITE porte le résultat : la taille RÉSIDUELLE, ce qui reste de la
taille une fois l'accuracy retirée. Le gauche est confondu par l'accuracy et sa pente y
est en partie triviale — il est là pour montrer de combien la confusion compte.

Écrit trois PNG dans `figures_xp3/`.

Usage::

    ./.venv/bin/python benchmarks/analysis/xp3_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import xp3_core as X  # noqa: E402
import xp3_tilt as T  # noqa: E402

OUT = A / "figures_xp3"
INK, MUT, RULE = "#12171c", "#5b6672", "#dde3e9"
# gris -> rouge : la couleur EST le rang de qualite, donc elle doit etre ordonnee.
# Trois teintes d'une meme famille, pas trois teintes categorielles.
BANDS = ["faible", "moyenne", "forte"]
PAL = {"faible": "#c9ced6", "moyenne": "#c98a86", "forte": "#a51f18"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})

NBOOT = 4000


def prepare() -> tuple[pd.DataFrame, dict]:
    """Le tableau sujet-par-sujet, et les intervalles qui vont dans le bandeau."""
    d = T.frame()
    blocks = T._blocks(d)
    obs = T.stat(blocks)

    keys = ("p_size", "p_resid")
    bl = {k: np.empty(NBOOT) for k in keys}
    bh = {k: np.empty(NBOOT) for k in keys}
    for t in range(NBOOT):
        picks = [T.RNG.integers(0, len(P), len(P)) for _, P, _, _ in blocks]
        s = T.stat(blocks, picks)
        for k in keys:
            bl[k][t], bh[k][t] = s[k]
    null = float(np.nanmean([T.stat(blocks, perm=True)["p_resid"][0] for _ in range(400)]))

    # les colonnes que la figure dessine, calculees dataset par dataset
    out = []
    for name, g in d.groupby("dataset", sort=False):
        P, Acc, Q = (g[c].to_numpy() for c in ("params", "acc", "quality"))
        ax = T._axes(P, Acc)
        out.append(pd.DataFrame({
            "dataset": name, "p_size": ax["p_size"], "p_resid": ax["p_resid"],
            "band": [BANDS[i] for i in T._tercile(Q)]}))
    stats = {k: dict(lo=obs[k][0], hi=obs[k][1],
                     lo_ci=T._ci(bl[k]), hi_ci=T._ci(bh[k])) for k in keys}
    return pd.concat(out, ignore_index=True), {"stats": stats, "null": null}


def _panels(d: pd.DataFrame, S: dict, kind: str, fname: str,
            title: str, sub: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=(kind == "kde"))
    specs = [("p_size", "taille finale — percentile dans son dataset",
              "brut : l'accuracy n'est pas retirée"),
             ("p_resid", "taille RÉSIDUELLE — percentile dans son dataset",
              "l'accuracy est retirée : c'est le panneau qui porte le résultat")]
    for ax, (col, xlab, note) in zip(axes, specs):
        if kind == "hist":
            sns.histplot(data=d, x=col, hue="band", hue_order=BANDS, multiple="stack",
                         bins=12, binrange=(0, 100), palette=PAL, ax=ax,
                         edgecolor="white", linewidth=0.6, legend=False)
            ax.set_ylabel("sujets")
        else:
            sns.kdeplot(data=d, x=col, hue="band", hue_order=BANDS, multiple="stack",
                        palette=PAL, ax=ax, bw_adjust=0.9, clip=(0, 100),
                        linewidth=0.7, legend=False)
            ax.set_ylabel("densité")
        s = S["stats"][col]
        ax.set_xlim(0, 100)
        # Le bandeau de chiffres est dessine DANS l'axe : il lui faut sa place, sinon
        # il se pose sur la courbe empilee la plus haute.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.24)
        ax.set_xlabel(xlab, fontsize=9.5)
        ax.set_title(note, fontsize=10, color=MUT, pad=6)
        ax.text(0.5, 0.975,
                f"part de « forte » :  {100 * s['lo']:.0f} % "
                f"[{100 * s['lo_ci'][0]:.0f}, {100 * s['lo_ci'][1]:.0f}]  à gauche"
                f"   →   {100 * s['hi']:.0f} % "
                f"[{100 * s['hi_ci'][0]:.0f}, {100 * s['hi_ci'][1]:.0f}]  à droite\n"
                f"nul mesuré par permutation : {100 * S['null']:.1f} %  "
                "(IC95 bootstrap sur les sujets, datasets fixes)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color=INK, linespacing=1.5,
                bbox=dict(facecolor="white", edgecolor=RULE, pad=4))
        for x in (100 / 3, 200 / 3):
            ax.axvline(x, color=MUT, lw=0.7, ls=(0, (4, 3)), alpha=0.55)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # Legende au niveau de la FIGURE, en bandeau : dans l'axe elle se posait sur les
    # barres du tiers gauche, qui sont justement celles que la figure fait lire.
    fig.legend(handles=[Line2D([], [], color=PAL[b], lw=7,
                               label=f"qualité de donneur {b}") for b in BANDS[::-1]],
               fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0.055, 0.885),
               frameon=False, handlelength=1.1, columnspacing=1.8)
    fig.suptitle(title, fontsize=13.5, y=0.978, x=0.008, ha="left", fontweight="bold")
    fig.text(0.008, 0.935, sub, fontsize=9.5, color=MUT, ha="left", va="top")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.77, bottom=0.10, wspace=0.17)
    fig.savefig(OUT / f"{fname}.png", dpi=155)
    plt.close(fig)


def _per_dataset(d: pd.DataFrame, S: dict, fname: str) -> None:
    """La planche qui répond d'avance au relecteur : qui porte l'effet ?

    À gauche, le basculement dataset par dataset. À droite, ce qu'il reste du poolé
    quand on retire chaque dataset — si une seule barre y faisait tout le travail,
    une des lignes s'effondrerait.
    """
    blocks = T._blocks(T.frame())
    per = []
    for name, P, Acc, Q in blocks:
        lo, hi = T.shares(T._axes(P, Acc)["p_resid"], T._tercile(Q))
        per.append((name, len(P), 100 * (hi - lo)))
    per.sort(key=lambda r: r[2])
    base = 100 * (S["stats"]["p_resid"]["hi"] - S["stats"]["p_resid"]["lo"])

    loo = []
    for j, (name, *_) in enumerate(blocks):
        s = T.stat([b for i, b in enumerate(blocks) if i != j])["p_resid"]
        loo.append((name, 100 * (s[1] - s[0])))
    loo.sort(key=lambda r: r[1])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    ax = axes[0]
    y = np.arange(len(per))
    ax.barh(y, [r[2] for r in per], color=PAL["forte"], height=0.62, alpha=0.88)
    ax.set_yticks(y, [f"{r[0]}  (n={r[1]})" for r in per], fontsize=8.5)
    ax.axvline(0, color=MUT, lw=0.8)
    ax.axvline(base, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.text(base, len(per) - 0.3, f" poolé {base:+.0f}", fontsize=8.5, color=INK)
    ax.set_xlabel("basculement : part de « forte » à droite − à gauche (points)")
    ax.set_title("dataset par dataset, sur l'axe résiduel", fontsize=10, color=MUT, pad=6)

    ax = axes[1]
    y = np.arange(len(loo))
    ax.scatter([r[1] for r in loo], y, s=34, color=PAL["forte"], zorder=3)
    ax.set_yticks(y, [r[0] for r in loo], fontsize=8.5)
    ax.axvline(base, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.set_xlim(0, max(base * 1.6, max(r[1] for r in loo) * 1.15))
    ax.set_xlabel("basculement poolé quand ce dataset est RETIRÉ (points)")
    ax.set_title("aucun dataset ne porte l'effet à lui seul",
                 fontsize=10, color=MUT, pad=6)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(axis="x", color=RULE, lw=0.6)
        a.set_axisbelow(True)
    fig.suptitle("Le basculement n'est pas l'affaire d'un dataset",
                 fontsize=13.5, y=0.975, x=0.008, ha="left", fontweight="bold")
    fig.text(0.008, 0.915,
             "trois datasets rendent un écart exactement nul : à n=9 à n=25, un tiers "
             "d'axe ne contient que 3 à 8 sujets et le découpage en terciles n'a plus "
             "le grain\npour montrer autre chose. La rho partielle, qui ne découpe rien, "
             "est positive sur 11 de ces 12 (p = 0.006).",
             fontsize=9, color=MUT, ha="left", va="top")
    fig.subplots_adjust(left=0.175, right=0.985, top=0.775, bottom=0.115, wspace=0.42)
    fig.savefig(OUT / f"{fname}.png", dpi=155)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d, S = prepare()
    n_s, n_ds = len(d), d.dataset.nunique()
    sub = (f"{n_s} sujets, {n_ds} datasets sous porte XP 2 · un tercile de qualité de "
           "donneur par couleur, calculé dans son propre dataset · les terciles étant "
           "équilibrés\net l'axe étant un percentile, le total est presque plat : ce "
           "qui se lit est la composition de chaque barre, pas sa hauteur")
    title = "Où tombent les bons donneurs le long de l'axe des tailles"
    _panels(d, S, "hist", "xp3_01_hist", title, sub)
    _panels(d, S, "kde", "xp3_02_kde",
            "La même chose en densité — la version lissée de la figure précédente", sub)
    _per_dataset(d, S, "xp3_03_per_dataset")
    for k in ("p_size", "p_resid"):
        s = S["stats"][k]
        print(f"{k:8s} gauche {100 * s['lo']:5.1f} % "
              f"[{100 * s['lo_ci'][0]:.1f}, {100 * s['lo_ci'][1]:.1f}]"
              f"   droite {100 * s['hi']:5.1f} % "
              f"[{100 * s['hi_ci'][0]:.1f}, {100 * s['hi_ci'][1]:.1f}]")
    print(f"nul mesure : {100 * S['null']:.2f} %")
    print(f"{n_s} sujets / {n_ds} datasets -> {OUT}/")


if __name__ == "__main__":
    main()
