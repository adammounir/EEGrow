"""La chaîne donneur→receveur en une page : les quatre étages, leurs verdicts, leurs figures.

Le troisième jumeau de ``build_perf_report`` et ``build_growth_dynamics`` -- même
stylesheet importée et non copiée, même règle « la docstring de la figure EST sa
légende », même palette. Un lecteur qui passe d'une page à l'autre lit la même couleur
comme le même bras.

POURQUOI UNE PAGE À PART ET NON UNE SECTION DE ``perf_report``. Les deux pages
répondent à des questions disjointes et pour des lecteurs disjoints : ``perf_report``
demande « la croissance paie-t-elle », celle-ci demande « la taille du modèle qui a
poussé dit-elle quelque chose du sujet ». Elles ne partagent ni les données (900 CSV de
score contre 4 matrices D×R et 3 800 cellules de sélection) ni la cadence de
regénération. Les fusionner enterrerait la première sous les figures de la seconde --
c'est l'argument que ``build_perf_report`` fait déjà contre sa fusion avec
``build_growth_dynamics``, et il vaut à l'identique ici. Les trois pages se citent
mutuellement en en-tête.

CE QUE CETTE PAGE DOIT IMPRIMER MÊME QUAND C'EST GÊNANT. Deux des quatre étages
concluent CONTRE l'hypothèse de départ, et un troisième a été ajouté après avoir vu les
résultats. Une page qui ne montrerait que l'étage qui marche serait une page fausse.
Chaque section porte donc son statut (pré-enregistré / post-hoc) et son verdict en
clair, avant la figure.

    python benchmarks/analysis/build_dxr_report.py [<out_dir>]
"""

from __future__ import annotations

import base64
import html
import io
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))

import donor_select2_analysis as ds2  # noqa: E402
from build_growth_dynamics import _CSS  # noqa: E402

DPI = 110

#: Palette, reprise de `_CSS` pour que la page et les tracés soient un seul objet.
INK, MUT, RULE = "#12171c", "#5b6672", "#dde3e9"
ACCENT, TEAL = "#a51f18", "#00695c"
#: Les cinq règles de sélection. L'ordre est celui de la lecture voulue : les deux bras
#: « taille » encadrent l'aléatoire, l'accuracy est le comparateur qui gagne.
RULE_COLOR = {"params_top": ACCENT, "resid_top": "#d4645c", "acc_top": TEAL,
              "random": "#8892a0", "params_bottom": "#c9a227"}
RULE_LABEL = {"params_top": "les plus gros", "resid_top": "les plus gros, à accuracy égale",
              "acc_top": "les meilleurs en accuracy", "random": "au hasard",
              "params_bottom": "les plus petits (contrôle négatif)"}

#: Les jumeaux. Écrits ici pour que chaque page nomme les deux autres.
PERF_URL = "https://claude.ai/code/artifact/44b84ba1-d91c-418a-a223-6c0b923cc6ac"
DYNAMICS_URL = "https://claude.ai/code/artifact/1c817e93-e1a0-49e4-ab1f-0c2419c96c7c"

#: Les trois datasets où l'étage 3 a tourné, et l'ordre d'affichage : par taille de
#: vivier croissante, parce que c'est la variable qui explique la saturation en K.
SEL_DATASETS = ["cho2017", "lee2019_mi", "physionetmi"]
ALL_DATASETS = ["cho2017", "physionetmi"]


# --------------------------------------------------------------------------- données

def load_selection() -> dict[str, dict]:
    """Les cellules de l'étage 3, relues par le MÊME code qui a produit le verdict.

    ``donor_select2_analysis`` est importé et non recopié : la figure doit être une vue
    du test, pas une seconde implémentation qui pourrait en diverger d'un bootstrap ou
    d'une graine. Les contrastes calculés ici sont donc, aux arrondis près, les lignes
    imprimées par ce script en console.
    """
    out = {}
    for name in SEL_DATASETS:
        d = ds2.load(A / "dsel2" / name)
        ks = sorted(int(k) for k in d.k.unique())
        W = {k: ds2.cells(d, k) for k in ks}
        rows = []
        for k in ks:
            for pair in [("resid_top", "random"), ("params_bottom", "random"),
                         ("resid_top", "acc_top")]:
                c = ds2.contrast(W[k], *pair)
                c.update(dataset=name, k=k, a=pair[0], b=pair[1])
                rows.append(c)
        # `candidates` est le jeu M partagé, sérialisé en `9;10;11;...` : c'est sa
        # CARDINALITÉ qui nous intéresse, pas sa valeur.
        m = len(str(d.candidates.iloc[0]).split(";"))
        out[name] = {"k": ks, "m": m, "n_subj": int(d.test_subject.nunique()),
                     "rows": pd.DataFrame(rows)}
    return out


def load_all_pool() -> pd.DataFrame:
    """Les écarts `all_pool − règle`, tels que `donor_all_analysis` les a écrits."""
    return pd.concat([pd.read_csv(A / "dsel2" / f"all_pool_{n}.csv")
                      for n in ALL_DATASETS], ignore_index=True)


def load_summary() -> pd.DataFrame:
    return pd.read_csv(A / "figures_dxr_summary" / "dxr_summary.csv")


# --------------------------------------------------------------------------- figures

def _ax(fig):
    for ax in np.atleast_1d(fig.axes).ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(RULE)
        ax.tick_params(colors=MUT, labelsize=9)
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lab.set_color(INK)
    return fig


def fig_chain(sel: dict, summ: pd.DataFrame) -> plt.Figure:
    """La chaîne entière sur une image, avec son verdict par étage.

    C'est la seule figure de la page qui ne montre pas de données : elle existe parce
    que les trois suivantes sont illisibles sans savoir laquelle réfute laquelle. Un
    étage vert n'autorise pas à citer l'étage suivant -- il autorise seulement à le
    lancer, et c'est exactement ce que les flèches disent.

    Le cadre rouge est un résultat NÉGATIF, pas une panne : l'étage 3 a tourné comme
    prévu et il répond non. Le cadre gris est le bras ajouté après coup, qui ne teste
    rien et sert de repère.
    """
    fig, ax = plt.subplots(figsize=(13.4, 4.9))
    ax.set_xlim(0, 102), ax.set_ylim(0, 42), ax.axis("off")
    W, GAP = 22.0, 4.0
    boxes = [
        ("Étage 0", "la sonde", "#params d'un shallow\nqui pousse sur un sujet,\n"
         "15 réplicats", "FIABLE\nICC$_k$ = 0.81 – 0.98", TEAL),
        ("Étage 1", "la matrice", "un modèle par donneur,\ntesté sur chacun\n"
         "des receveurs", "TRANSFERT\n4 datasets, 224 donneurs", TEAL),
        ("Étage 2", "le prédicteur", "ρ partielle de #params,\naccuracy contrôlée\n ",
         "POSITIF\nρ = +0.288\n[+0.16, +0.41]", TEAL),
        ("Étage 3", "l'intervention", "sélectionner K donneurs,\nentraîner sur leur "
         "union,\nscorer les autres", "RÉFUTÉ\nl'accuracy fait mieux", ACCENT),
    ]
    for i, (num, name, what, verdict, col) in enumerate(boxes):
        x = 1 + i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, 10), W, 26, fill=True, facecolor="white",
                                   edgecolor=col, lw=1.7, zorder=2))
        ax.text(x + W / 2, 33.6, num, ha="center", va="top", fontsize=8,
                color=col, family="monospace", zorder=3)
        ax.text(x + W / 2, 30.6, name, ha="center", va="top", fontsize=11,
                weight="bold", color=INK, zorder=3)
        ax.text(x + W / 2, 26.3, what, ha="center", va="top", fontsize=8.2,
                color=MUT, linespacing=1.5, zorder=3)
        ax.text(x + W / 2, 14.5, verdict, ha="center", va="center", fontsize=8.8,
                weight="bold", color=col, linespacing=1.5, zorder=3)
        if i < 3:
            ax.annotate("", xy=(x + W + GAP - 0.4, 23), xytext=(x + W + 0.4, 23),
                        arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.5,
                                        mutation_scale=14))
    x3 = 1 + 3 * (W + GAP)
    ax.add_patch(plt.Rectangle((x3, 1.5), W, 6.4, fill=True, facecolor="#f0f1f3",
                               edgecolor=MUT, lw=1.1, ls=(0, (3, 2)), zorder=2))
    ax.text(x3 + W / 2, 4.7, "bras « tout le monde dedans »\nrepère post-hoc, aucun test",
            ha="center", va="center", fontsize=8, color=MUT, style="italic",
            linespacing=1.5, zorder=3)
    ax.annotate("", xy=(x3 + W / 2, 8.2), xytext=(x3 + W / 2, 9.6),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2,
                                mutation_scale=12))
    ax.text(1, 40, "chaque étage est une condition de LECTURE du suivant, "
            "pas une preuve du suivant", fontsize=9.5, color=MUT, style="italic")
    return _ax(fig)


def fig_selection_forest(sel: dict) -> plt.Figure:
    """L'étage 3 en entier : le garde-fou d'abord, l'endpoint ensuite.

    Les deux lignes grises du haut de chaque bloc sont le GARDE-FOU -- « sélectionner
    par la taille bat le hasard » et « prendre les plus petits fait pire ». Elles ne
    sont pas le résultat : elles disent seulement si le protocole a de la prise sur ce
    dataset. La ligne colorée du bas est l'endpoint pré-enregistré.

    Sur cho2017 le garde-fou tombe aux trois K -- le contrôle négatif y bat le hasard,
    ce qui est le signe INVERSE du prédit -- donc l'endpoint n'y est pas tracé. Ne pas
    lire un résultat sous un garde-fou rouge est la règle qui rend les deux autres
    datasets citables.

    Sur lee2019_mi et physionetmi, où le garde-fou tient, l'endpoint est négatif et son
    intervalle exclut zéro : à K=5 sélectionner sur la taille fait MOINS BIEN que
    sélectionner sur l'accuracy. C'est la réfutation, et elle réplique sur deux
    datasets indépendants.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharex=True)
    for ax, name in zip(axes, SEL_DATASETS):
        r = sel[name]["rows"]
        gate_ok = {}
        for k in sel[name]["k"]:
            g = r[(r.k == k) & (r.b == "random")]
            gate_ok[k] = bool((g.iloc[0].lo > 0) and (g.iloc[1].hi < 0))
        y, labels = 0, []
        for k in sel[name]["k"][::-1]:
            for _, row in r[(r.k == k)].iloc[::-1].iterrows():
                is_end = row.b == "acc_top"
                if is_end and not gate_ok[k]:
                    continue
                col = (ACCENT if is_end else
                       (MUT if gate_ok[k] else "#c9a227"))
                ax.plot([row.lo, row.hi], [y, y], color=col, lw=2.6,
                        solid_capstyle="butt", alpha=.85)
                ax.plot([row.delta], [y], "o", color=col, ms=6.5, mec="white", mew=1.1)
                lab = ("taille − accuracy" if is_end else
                       ("gros − hasard" if row.a == "resid_top"
                        else "petits − hasard"))
                labels.append((y, f"K={k}  ·  {lab}", is_end))
                y += 1
        ax.axvline(0, color=RULE, lw=1, zorder=0)
        ax.set_yticks([p for p, _, _ in labels])
        ax.set_yticklabels([t for _, t, _ in labels], fontsize=8)
        for tick, (_, _, is_end) in zip(ax.get_yticklabels(), labels):
            if is_end:
                tick.set_color(ACCENT), tick.set_fontweight("bold")
        ax.set_ylim(-0.7, y - 0.3)
        ok = all(gate_ok.values())
        ax.set_title(f"{name}\n{sel[name]['n_subj']} sujets · "
                     + ("garde-fou OK" if ok else "GARDE-FOU EN ÉCHEC"),
                     fontsize=10, color=INK if ok else "#c9a227", weight="bold")
        ax.set_xlabel("Δ ROC-AUC", fontsize=9, color=MUT)
    fig.tight_layout()
    return _ax(fig)


def fig_saturation(sel: dict) -> plt.Figure:
    """La prédiction qui pouvait me donner tort, et qui tient.

    L'échec de la première version de l'étage 3 avait été expliqué par la saturation :
    à K=20 sur un vivier de 82, on prend un quart du vivier et toutes les règles
    finissent par se ressembler. Cette explication est falsifiable -- elle exige que le
    gain sur le hasard soit plus PETIT au plus grand K qu'au plus petit. C'est le cas
    sur les deux datasets où le garde-fou tient, d'un facteur 2 : +0.043 → +0.022 sur
    physionetmi, +0.031 → +0.017 sur lee2019_mi.

    Ce que la figure montre aussi et qu'il faut dire : sur physionetmi la décroissance
    n'est PAS monotone, le gain culmine à K=10 avant de retomber. La prédiction portait
    sur les extrémités et elle tient ; la forme intermédiaire n'était pas prédite et
    n'est pas expliquée.

    L'axe des x est K/M et non K, pour que trois viviers de tailles différentes soient
    comparables. cho2017 est tracé pour montrer à quoi ressemble un dataset où il n'y a
    rien à saturer : la courbe part déjà près de zéro.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    color = {"cho2017": "#c9a227", "lee2019_mi": "#4a7fb5", "physionetmi": TEAL}
    for name in SEL_DATASETS:
        r = sel[name]["rows"]
        g = r[(r.a == "resid_top") & (r.b == "random")].sort_values("k")
        ks = g.k.to_numpy(float)
        ratio = 100 * ks / sel[name]["m"]  # K/M, pour comparer des viviers inégaux
        ok = (g.lo > 0).all()
        ax.errorbar(ratio, g.delta, yerr=[g.delta - g.lo, g.hi - g.delta],
                    marker="o", ms=6, lw=2 if ok else 1.3, capsize=3,
                    color=color[name], ls="-" if ok else (0, (3, 2)),
                    label=f"{name} — {'garde-fou OK' if ok else 'GARDE-FOU EN ÉCHEC'}")
    ax.axhline(0, color=RULE, lw=1)
    ax.set_xlabel("K en % du jeu de candidats M", fontsize=9.5, color=MUT)
    ax.set_ylabel("gain de la sélection par taille sur le hasard\n(Δ ROC-AUC)",
                  fontsize=9.5, color=MUT)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return _ax(fig)


def fig_all_pool(ap: pd.DataFrame) -> plt.Figure:
    """Le comparateur manquant : ne pas sélectionner du tout.

    Chaque point est `all_pool − règle` en ROC-AUC moyen par sujet de test ; positif
    veut dire que ne rien sélectionner est MEILLEUR. Les 30 comparaisons sont positives,
    et les 4 plis sont concordants en signe dans chacune -- c'est cette concordance qui
    porte l'évidence, pas la largeur des intervalles, qui est anti-conservatrice
    (les sujets d'un même pli partagent le pool d'entraînement).

    Ce que la figure ne dit PAS : K n'est pas apparié. `all_pool` s'entraîne sur 39 ou
    81 donneurs, les règles sur 3 à 20. Chaque écart mélange donc « plus de données » et
    « meilleures données », et c'est pour ça que le bras est rapporté comme repère
    descriptif et non comme un test.

    La lecture qui compte est la PENTE : l'écart décroît quand K monte, ce qui est
    attendu si la part « plus de données » domine. Il ne s'annule jamais sur la plage
    testée.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, name in zip(axes, ALL_DATASETS):
        g = ap[ap.dataset == name]
        ks = sorted(g.k.unique())
        for rule in ["acc_top", "resid_top", "params_top", "random", "params_bottom"]:
            s = g[g.rule == rule].sort_values("k")
            ax.plot(range(len(ks)), s.delta, "o-", color=RULE_COLOR[rule], ms=6,
                    lw=1.8, label=RULE_LABEL[rule], mec="white", mew=1)
            for x, (_, row) in enumerate(s.iterrows()):
                ax.plot([x, x], [row.lo, row.hi], color=RULE_COLOR[rule], lw=1.2,
                        alpha=.55)
        ax.axhline(0, color=RULE, lw=1)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_ylim(bottom=0)
        n_pool = int(g.n_pool.iloc[0])
        ax.set_title(f"{name} — vivier de {n_pool} donneurs", fontsize=10,
                     color=INK, weight="bold")
        ax.set_ylabel("Δ ROC-AUC   (all_pool − règle)", fontsize=9, color=MUT)
    axes[0].legend(frameon=False, fontsize=8.2, loc="upper right")
    fig.tight_layout()
    return _ax(fig)


def fig_recovery(ap: pd.DataFrame) -> plt.Figure:
    """Le recadrage : la sélection ne récupère qu'une fraction de ce qu'elle a coûté.

    Le déficit est `all_pool − random` : ce que coûte le fait de restreindre le vivier à
    K sujets tirés au hasard. Chaque barre est la part de ce déficit qu'une règle
    récupère, `(règle − hasard) / (all_pool − hasard)`. Une barre à 100 % voudrait dire
    « cette règle annule le coût de la restriction » ; aucune n'y arrive.

    Deux choses à lire. L'accuracy récupère plus que la taille dans 5 lignes sur 6 --
    c'est le troisième site indépendant, après les deux datasets de l'étage 3, où
    l'accuracy bat #params. Et l'échelle : le meilleur récupérateur plafonne autour de
    40 %, donc la question « quel critère de sélection » est dominée d'un ordre de
    grandeur par la question « pourquoi restreindre ».
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), sharey=True)
    rules = ["acc_top", "resid_top", "params_top", "params_bottom"]
    for ax, name in zip(axes, ALL_DATASETS):
        g = ap[ap.dataset == name]
        ks = sorted(g.k.unique())
        w, xs = 0.19, np.arange(len(ks))
        for i, rule in enumerate(rules):
            rec = []
            for k in ks:
                d_rand = float(g[(g.k == k) & (g.rule == "random")].delta.iloc[0])
                d_rule = float(g[(g.k == k) & (g.rule == rule)].delta.iloc[0])
                rec.append(100 * (d_rand - d_rule) / d_rand)
            ax.bar(xs + (i - 1.5) * w, rec, w, color=RULE_COLOR[rule],
                   label=RULE_LABEL[rule], edgecolor="white", lw=.6)
            for x, v in zip(xs + (i - 1.5) * w, rec):
                ax.text(x, v + (1.5 if v >= 0 else -4), f"{v:.0f}", ha="center",
                        fontsize=7.6, color=MUT)
        ax.axhline(0, color=INK, lw=.9)
        ax.set_xticks(xs), ax.set_xticklabels([f"K={k}" for k in ks])
        deficits = [float(g[(g.k == k) & (g.rule == "random")].delta.iloc[0])
                    for k in ks]
        ax.set_title(f"{name}\ndéficit de l'aléatoire : "
                     + ", ".join(f"{d:+.3f}" for d in deficits),
                     fontsize=9.6, color=INK, weight="bold")
    axes[0].set_ylabel("part du déficit récupérée (%)", fontsize=9.5, color=MUT)
    # La légende va SOUS les axes : posée dedans elle recouvrait les étiquettes de
    # barre du premier panneau, qui sont la moitié de l'information de la figure.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.4, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _ax(fig)


# ----------------------------------------------------------------------- les tableaux

def _table(cols: list[str], rows: list[list[str]], note: str = "") -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    n = f'<p class="tnote">{note}</p>' if note else ""
    return (f'<div class="tbl"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>{n}</div>")


def table_stage12(summ: pd.DataFrame) -> str:
    rows = []
    for _, r in summ.sort_values("n", ascending=False).iterrows():
        excl = r.ceil_probe >= 0.50
        name = (f'<b>{r.dataset}</b>' if not excl
                else f'<span class="out">{r.dataset} — exclu</span>')
        rows.append([name, f"{int(r.n)}", f"{r.rho_p:+.3f}", f"{r.rho_a:+.3f}",
                     f"{r.par:+.3f}", f"{r.mde:.2f}", f"{r.icc_k:.2f}",
                     f"{100 * r.ceil_probe:.0f} %"])
    return _table(
        ["dataset", "donneurs", "ρ(#params)", "ρ(accuracy)", "ρ partielle",
         "MDE", "ICC des moyennes", "sondes au plafond"], rows,
        "ρ partielle = corrélation de #params avec la qualité de donneur, accuracy "
        "contrôlée. MDE = plus petit effet détectable à ce n. bnci2014_001 est exclu "
        "sur un critère qui porte sur la SONDE (93 % des fits collés au plafond de "
        "largeur), pas sur son résultat.")


def table_endpoints(sel: dict) -> str:
    rows = []
    for name in SEL_DATASETS:
        r = sel[name]["rows"]
        k = 5 if 5 in sel[name]["k"] else sel[name]["k"][1]
        gate = r[(r.k == k) & (r.b == "random")]
        ok = bool(gate.iloc[0].lo > 0 and gate.iloc[1].hi < 0)
        e = r[(r.k == k) & (r.b == "acc_top")].iloc[0]
        sel_eff = float(gate.iloc[0].delta)
        rows.append([
            f"<b>{name}</b>", f"{sel[name]['n_subj']}", f"K={k}",
            ('<span class="ok">tient</span>' if ok
             else '<span class="ko">échoue</span>'),
            (f"{e.delta:+.4f} [{e.lo:+.4f}, {e.hi:+.4f}]" if ok else "—"),
            (f"{100 * e.delta / sel_eff:+.0f} %" if ok else "—"),
        ])
    return _table(
        ["dataset", "sujets", "K primaire", "garde-fou",
         "taille − accuracy", "en % de l'effet de sélection"], rows,
        "L'endpoint n'est lu que sous un garde-fou qui tient. Négatif = sélectionner "
        "sur la taille fait MOINS BIEN que sélectionner sur l'accuracy.")


# --------------------------------------------------------------------------- la page

def _para(text: str) -> str:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{html.escape(' '.join(p.split()))}</p>" for p in parts)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


#: Les figures de l'étage 1 sont déjà sur disque, produites par `donor_figures` et
#: `donor_summary` : on les embarque telles quelles plutôt que de les redessiner, avec
#: une légende écrite ici parce que leur docstring d'origine ne l'est pas.
DISK = [
    ("summary_forest", "figures_dxr_summary/dxr_summary_01_forest.png",
     "Étage 2 — les quatre datasets sur un axe",
     "La partielle par dataset, avec à côté la part des sondes collées au plafond de "
     "largeur. La lecture voulue est verticale : la partielle monte quand la "
     "saturation descend. Quatre points ne font pas une tendance et la figure ne le "
     "prétend pas — elle interdit seulement de lire le +0.150 de cho2017 comme un "
     "fait sur la croissance alors qu'il est d'abord un fait sur l'échelle de la sonde."),
    ("summary_scatter", "figures_dxr_summary/dxr_summary_02_partial_scatter.png",
     "Étage 2 — un sujet, un point",
     "L'axe des x est le RÉSIDU de #params après régression sur l'accuracy, en rangs : "
     "c'est la variable dont la partielle mesure la pente. Tracer #params brut "
     "montrerait une pente que l'accuracy explique déjà."),
    ("summary_gates", "figures_dxr_summary/dxr_summary_03_gates.png",
     "Les deux portes sans lesquelles la forêt ne veut rien dire",
     "Porte 0 : y a-t-il du transfert sur ce dataset ? Porte 1 : la qualité de donneur "
     "est-elle reproductible d'une graine à l'autre ? Un ρ élevé sur un dataset qui "
     "échoue à l'une des deux serait une corrélation avec du bruit."),
    ("dxr_heatmap", "figures_dxr_physionetmi/dxr_01_heatmap.png",
     "Étage 1 — la matrice elle-même (physionetmi)",
     "Une case = un modèle entraîné sur le seul donneur D, testé sur le seul receveur "
     "R. Les colonnes sont z-scorées avant tout résumé, pour retirer la difficulté "
     "propre du receveur : sans ça la « qualité de donneur » d'une ligne serait "
     "surtout une mesure de qui elle a eu la chance de croiser."),
    ("dxr_quality", "figures_dxr_physionetmi/dxr_02_donor_quality.png",
     "Étage 1 — la qualité de donneur est-elle une propriété du sujet ?",
     "Le classement des donneurs est refait indépendamment sur chaque graine. S'il ne "
     "survivait pas au changement de graine, la suite de la chaîne mesurerait du bruit."),
    ("dxr_predictors", "figures_dxr_physionetmi/dxr_03_predictors.png",
     "Étage 2 — les deux prédicteurs côte à côte (physionetmi)",
     "#params et accuracy de la sonde, chacun contre la qualité de donneur. Les deux "
     "montent : toute la question de l'étage 2 est de savoir ce qui reste au premier "
     "quand on a retiré le second."),
    ("dxr_paired", "figures_dxr_physionetmi/dxr_04_paired.png",
     "Étage 2 — la figure qui porte le verdict",
     "Deux tests séparés ne font pas une comparaison : « l'un est significatif, l'autre "
     "non » est parfaitement compatible avec une différence nulle. C'est la "
     "distribution de la DIFFÉRENCE, rééchantillonnée sur les mêmes sujets, qui tranche."),
    ("dxr_ceiling", "figures_dxr_physionetmi/dxr_05_ceiling.png",
     "L'échelle sur laquelle tout ceci est mesuré",
     "<code>grow_shallow.yaml</code> borne la croissance des deux côtés : départ à "
     "<code>n_filters_time: 8</code>, plafond à <code>target_n_filters_time: 40</code>. "
     "Un fit qui finit à 8 n'a pas grandi, un fit qui finit à 40 voulait peut-être "
     "aller plus loin — dans les deux cas la largeur a cessé de mesurer le sujet."),
    ("dxr_scale_bnci", "figures_dxr_bnci2014_001/dxr_06_predictor_scale.png",
     "Pourquoi bnci2014_001 ne peut pas répondre",
     "Le même graphique d'échelle sur bnci2014_001 : 93 % des sondes s'arrêtent sur le "
     "plafond, l'étendue totale de #params y est de 12 % contre 187 % sur physionetmi. "
     "#params y est une borne, pas une mesure. Avec 9 sujets le MDE sur une corrélation "
     "vaut 1.14 : même une corrélation parfaite n'y serait pas détectable."),
]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else A / "dxr_final"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    sel = load_selection()
    ap = load_all_pool()
    summ = load_summary()

    drawn = [
        ("chain", "La chaîne, et où elle casse", lambda: fig_chain(sel, summ)),
        ("selection_forest", "Étage 3 — l'intervention, garde-fou compris",
         lambda: fig_selection_forest(sel)),
        ("saturation", "Étage 3 — la prédiction qui pouvait me donner tort",
         lambda: fig_saturation(sel)),
        ("all_pool", "Le comparateur manquant — ne pas sélectionner du tout",
         lambda: fig_all_pool(ap)),
        ("recovery", "Ce que la sélection récupère du coût qu'elle a créé",
         lambda: fig_recovery(ap)),
    ]

    blocks: list[tuple[str, str, str, str]] = []
    for name, title, fn in drawn:
        fig = fn()
        fig.savefig(figs / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        doc = textwrap.dedent(fn.__doc__ or "").strip() if fn.__doc__ else ""
        if not doc:  # les lambdas n'en ont pas : on va la chercher sur la vraie fonction
            doc = textwrap.dedent(
                globals()[f"fig_{name}"].__doc__ or "").strip()
        blocks.append((name, title, _para(doc),
                       base64.b64encode(buf.getvalue()).decode()))
        print(f"  + {name}: {len(buf.getvalue()) / 1e6:.2f} MB")

    for name, rel, title, cap in DISK:
        p = A / rel
        if not p.exists():
            print(f"  ! {name}: {rel} absent")
            continue
        blocks.append((name, title, f"<p>{cap}</p>", _b64(p)))
        print(f"  + {name}: {p.stat().st_size / 1e6:.2f} MB (disque)")

    _write_page(out, blocks, sel, ap, summ)


FINDINGS = [
    ("La taille du modèle qui a poussé porte de l'information sur le sujet, "
     "et ce n'est pas de l'accuracy déguisée.",
     "Corrélation partielle de <code>#params</code> avec la qualité de donneur, à "
     "accuracy contrôlée, combinée sur les trois datasets exploitables : "
     "<b>ρ = +0.288 [+0.159, +0.412]</b>, n = 215 donneurs. Hétérogénéité nulle "
     "(I² = 0 %). Le même calcul en gardant bnci2014_001 donne +0.284 [+0.155, +0.408] — "
     "l'exclusion ne change pas le verdict, elle change la lisibilité."),
    ("Mais elle ne sélectionne pas un bon jeu d'entraînement. C'est la réfutation, "
     "et elle réplique.",
     "Endpoint pré-enregistré, à K=5 : <b>−0.0377 [−0.0499, −0.0258]</b> sur "
     "physionetmi et <b>−0.0196 [−0.0346, −0.0060]</b> sur lee2019_mi. Sélectionner "
     "sur la taille fait moins bien que sélectionner sur l'accuracy, de 87 % et 82 % "
     "de l'effet de sélection lui-même. Deux datasets indépendants, même signe."),
    ("Sur cho2017 le protocole n'a pas de prise, et on ne lit donc pas son résultat.",
     "Le garde-fou échoue aux trois K : le contrôle négatif (prendre les plus petits) "
     "y <b>bat</b> le hasard, ce qui est le signe inverse du prédit. C'était pourtant "
     "le dataset choisi <i>parce que</i> les deux critères y sont les plus "
     "indépendants. Un endpoint lu sous un garde-fou rouge ne veut rien dire."),
    ("La saturation en K, qui pouvait me donner tort, tient.",
     "Le gain de la sélection sur le hasard est deux fois plus petit au plus grand K "
     "qu'au plus petit : <b>+0.043 → +0.022</b> sur physionetmi, "
     "<b>+0.031 → +0.017</b> sur lee2019_mi. C'est ce que prédisait l'explication de "
     "l'échec de la première version du protocole — à K=20 sur un vivier de 82, toutes "
     "les règles finissent par se ressembler. La prédiction portait sur les "
     "extrémités&nbsp;; entre les deux, physionetmi culmine à K=10, ce qui n'était pas "
     "prédit et n'est pas expliqué."),
    ("Le comparateur manquant change la façon de lire tout le reste.",
     "Ne rien sélectionner bat les cinq règles, aux trois K, sur les deux datasets : "
     "<b>30 comparaisons sur 30</b>, avec les 4 plis concordants en signe dans "
     "chacune. L'effet de sélection que mesure l'étage 3 (~0.04) est d'un ordre de "
     "grandeur plus petit que le coût de restreindre le vivier (0.06 à 0.28). "
     "Bras post-hoc, rapporté sans valeur p."),
    ("L'explication banale du bras all_pool est exclue, et ce qu'il en reste joue "
     "contre la conclusion.",
     "<code>all_pool</code> voit 4 à 8 fois plus d'essais que la plus grosse cellule de "
     "règle, au même budget d'époques : s'il gagnait par sous-entraînement des "
     "concurrents, ou perdait par le sien, la comparaison serait nulle. Les deux côtés "
     "sortent symétriquement sur le budget, et le biais résiduel <b>sous-estime</b> "
     "<code>all_pool</code>."),
]


def _write_page(out: Path, blocks, sel: dict, ap: pd.DataFrame,
                summ: pd.DataFrame) -> None:
    n_cells = sum(len(list((A / "dsel2" / n).glob("f*__r*__k*__*.csv")))
                  for n in SEL_DATASETS)
    n_all = sum(len(list((A / "dsel_all" / n).glob("f*.csv")))
                for n in ALL_DATASETS)
    n_donors = int(summ.n.sum())

    body = [f"""<header>
<p class="eyebrow">eegrow · chaîne donneur → receveur · claims ICLR</p>
<h1>La taille d'un modèle qui a poussé<br>mesure-t-elle son sujet&nbsp;?</h1>
<p class="standfirst">L'idée de départ, posée par S.&nbsp;Chevallier&nbsp;: quand on
laisse un réseau <i>pousser</i> sur un sujet, la taille à laquelle il s'arrête dit
peut-être quelque chose de ce sujet — et si oui, on tient un critère pour choisir sur
qui entraîner. Quatre étages ont été montés pour le vérifier. Deux répondent oui, le
troisième répond non, et un quatrième comparateur, ajouté après coup, déplace la
question.</p>
<dl class="stats">
<div><dt>donneurs</dt><dd>{n_donors}</dd></div>
<div><dt>cellules D×R</dt><dd>52&nbsp;746</dd></div>
<div><dt>cellules de sélection</dt><dd>{n_cells:,}</dd></div>
<div><dt>bras all_pool</dt><dd>{n_all}</dd></div>
<div><dt>datasets</dt><dd>4</dd></div>
</dl>
<p class="caveat">L'unité d'analyse n'est <b>jamais</b> le sujet dans l'étage 3 : deux
règles d'un même réplicat scorent exactement les mêmes sujets, donc l'unité est le
couple (pli, réplicat) et le bootstrap est stratifié par pli. Les endpoints et les
garde-fous ont été écrits <b>avant</b> les données ; le bras <code>all_pool</code> est
déclaré post-hoc et ne fournit aucun test. Rapports jumeaux&nbsp;:
<a href="{PERF_URL}"><i>Où la croissance gagne</i></a>, qui mesure ce que vaut la
croissance elle-même, et <a href="{DYNAMICS_URL}"><i>Growth dynamics</i></a>, qui décrit
le mécanisme dont la sonde de l'étage 0 lit la sortie.</p>
</header>"""]

    body.append('<section class="findings"><h2>Ce que ces expériences disent</h2><dl>')
    for head, text in FINDINGS:
        body.append(f"<div><dt>{html.escape(head)}</dt><dd>{text}</dd></div>")
    body.append("</dl></section>")

    body.append(f"""<section class="findings" id="tables">
<h2>Les deux tableaux à lire avant les figures</h2>
<h3>Étages 1 et 2 — le prédicteur, dataset par dataset</h3>
{table_stage12(summ)}
<h3>Étage 3 — l'intervention, au K primaire</h3>
{table_endpoints(sel)}
</section>""")

    body.append('<nav aria-label="Figures"><h2>Figures</h2><ol>')
    for i, (name, title, _, _) in enumerate(blocks, 1):
        body.append(f'<li><span class="num">{i:02d}</span>'
                    f'<a href="#{name}">{html.escape(title)}</a></li>')
    body.append("</ol></nav>")

    for i, (name, title, cap, b64) in enumerate(blocks, 1):
        body.append(f"""<section id="{name}">
<p class="eyebrow"><span class="num">{i:02d}</span></p>
<h2>{html.escape(title)}</h2>
<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(title)}"
 loading="lazy"></figure>
<div class="caption">{cap}</div>
</section>""")

    body.append("""<section id="suite">
<p class="eyebrow">ce qui reste ouvert</p>
<h2>Les trois questions que ces résultats posent</h2>
<div class="caption">
<p><b>Faut-il récupérer bnci2014_001&nbsp;?</b> La piste évidente est de relever le
plafond de largeur du growing sur ce dataset. Mais ça change la sonde, donc le
classement des donneurs, donc tous les étages en aval — c'est un arbitrage, pas un
réglage.</p>
<p><b>La redondance explique-t-elle la réfutation&nbsp;?</b> Cinq gros donneurs se
ressemblent peut-être trop, là où cinq donneurs divers couvriraient mieux l'espace.
C'est testable : il suffit d'ajouter une règle qui maximise la diversité à qualité
égale, dans le protocole déjà en place.</p>
<p><b>Et si la vraie question était « pourquoi restreindre »&nbsp;?</b> Le bras
<code>all_pool</code> suggère que le budget de recherche irait plus loin sur
« pondérer tous les donneurs » que sur « en choisir K ». Ce serait un changement de
sujet, et c'est celui que la mesure désigne.</p>
</div>
</section>""")

    inner = "".join(body)
    css = _CSS + _EXTRA_CSS
    (out / "dxr_report_body.html").write_text(f"<style>{css}</style>{inner}")
    (out / "dxr_report.html").write_text(
        f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Donneur → receveur — la chaîne et ses verdicts</title>"
        f"<style>{css}</style></head><body>{inner}</body></html>")
    size = (out / "dxr_report.html").stat().st_size / 1e6
    print(f"\n{len(blocks)} figures -> {out / 'dxr_report.html'} ({size:.1f} MB)")


#: Ce que `_CSS` ne couvre pas : cette page a des tableaux, l'autre n'en a pas.
_EXTRA_CSS = """
h3 { font-family:var(--sans); font-size:1rem; font-weight:600; letter-spacing:-.01em;
     margin:2rem 0 .7rem; color:var(--ink); }
.tbl { overflow-x:auto; margin:0 0 1.4rem; }
.tbl table { border-collapse:collapse; width:100%; font:400 .88rem/1.45 var(--sans);
     font-variant-numeric:tabular-nums; }
.tbl th { text-align:left; font:500 .7rem/1.3 var(--mono); letter-spacing:.06em;
     text-transform:uppercase; color:var(--mut); padding:.45rem .7rem .45rem 0;
     border-bottom:1px solid var(--rule); white-space:nowrap; }
.tbl td { padding:.42rem .7rem .42rem 0; border-bottom:1px solid var(--rule-soft);
     color:var(--ink); white-space:nowrap; }
.tbl td:first-child, .tbl th:first-child { white-space:normal; }
.tbl .out { color:var(--mut); text-decoration:line-through; }
.tbl .ok { color:var(--teal); font-weight:600; }
.tbl .ko { color:var(--accent); font-weight:600; }
.tnote { font-size:.85rem; color:var(--mut); margin:.6rem 0 0; max-width:70ch; }
.findings dd i { font-style:italic; }
"""


if __name__ == "__main__":
    main()
