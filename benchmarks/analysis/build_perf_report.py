"""Render every figure in ``perf_figures`` and assemble them into one HTML page.

The companion driver to ``build_growth_dynamics``, and deliberately its twin: same
docstring-as-caption rule, same skipped-figures list, same stylesheet -- imported from
it rather than copied, so the two reports are one visual artifact and a reader moving
between them reads the same colour as the same arm.

WHY TWO PAGES AND NOT ONE. Three reasons, in increasing order of how much they matter.
The assembled pages are 8 MB each and the host caps a rendered artifact at 16 MB, so one
page does not fit. They regenerate on different cadences -- this one off 900 score CSVs
in seconds, the other off 9 GB of JSONL through a streaming reducer. And they answer
different questions for different readers: "does it pay" is cited, "how does it behave"
is searched. Merging them buries the first under thirty figures of the second.

    python benchmarks/analysis/build_perf_report.py <scores_dir> <fits_csv> <out_dir>
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import perf_figures as pf  # noqa: E402
import perf_io  # noqa: E402
from build_growth_dynamics import _CSS  # noqa: E402

DPI = 110

#: The alignment arm every figure that has to fix one, fixes. Raw rather than EA because
#: it is the arm the decomposition question was posed on and the one the fixed controls
#: cover most completely; the EA sweep is drawn beside it wherever a figure can hold
#: both, and `alignment_effect` makes the choice itself the variable.
BASE_ALIGN = "none"

#: The twin report. Written down so each page names the other: a reader who arrives at
#: "growth is worth +0.005" needs the page that says what the mechanism was doing, and a
#: reader who arrives at "the line search refuses 65 % of its proposals" needs the page
#: that says whether that cost anything.
DYNAMICS_URL = "https://claude.ai/code/artifact/1c817e93-e1a0-49e4-ab1f-0c2419c96c7c"

#: Le troisième jumeau, ajouté le 02/09. Il ne partage NI les données NI la cadence de
#: cette page -- 4 matrices donneur/receveur et 3 840 cellules de sélection contre 900
#: CSV de score -- donc il est une page à part et non une section d'ici, pour la même
#: raison que `growth_dynamics` en est une. Ce qui se partage est la stylesheet et le
#: fait que chaque page nomme les deux autres.
DXR_URL = "https://claude.ai/code/artifact/d054d23b-25e8-4790-9994-069a7f632450"


def main() -> None:
    scores_dir, fits_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    figs_dir = out / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    sc = perf_io.load(scores_dir)
    if fits_csv.exists():
        sc = perf_io.attach_params(sc, pd.read_csv(fits_csv))
    else:
        print(f"  [!] {fits_csv} absent — the size figures will be skipped")
    subj = perf_io.by_subject(sc)
    print(f"{len(sc):,} scored folds -> {len(subj):,} subject-level units, "
          f"{sc.dataset.nunique()} datasets, {sc.model.nunique()} arms")

    aligns = [a for a in ("none", "easubject") if a in set(sc.align_tag)]
    evals = [e for e in perf_io.EVAL_ORDER if e in set(sc["eval"])]
    archs = [a for a in perf_io.TRIPLES if a in set(sc.arch)]

    jobs: list[tuple[str, str, callable]] = [
        ("coverage", "What the campaign measured", lambda: pf.coverage(sc)),
    ]
    for al in aligns:
        sfx = "_ea" if al != "none" else ""
        jobs.append((f"decomposition{sfx}",
                     f"grow − bd, decomposed — {pf.ALIGN_LABEL[al]}",
                     lambda al=al: pf.decomposition(subj, al)))
    jobs += [
        ("power", "Effect against minimum detectable effect",
         lambda: pf.power(subj, BASE_ALIGN)),
        ("seed_noise", "The noise floor", lambda: pf.seed_noise(sc, BASE_ALIGN)),
    ]
    for ev in evals:
        for arch in archs:
            jobs.append((f"decomp_ds__{arch}__{ev}",
                         f"{arch} by dataset — {pf.EVAL_LABEL[ev]}",
                         lambda a=arch, e=ev: pf.decomposition_by_dataset(
                             subj, a, e, BASE_ALIGN)))
    for ev in evals:
        for arch in archs:
            jobs.append((f"subject_delta__{arch}__{ev}",
                         f"{arch} subject by subject — {pf.EVAL_LABEL[ev]}",
                         lambda a=arch, e=ev: pf.subject_delta(subj, a, e, BASE_ALIGN)))
    for ev in evals:
        for arch in archs:
            jobs.append((f"dumbbell__{arch}__{ev}",
                         f"{arch} on the score axis — {pf.EVAL_LABEL[ev]}",
                         lambda a=arch, e=ev: pf.dumbbell(subj, a, e, BASE_ALIGN)))
    jobs.append(("chance_map", "Did this arm learn anything here?",
                 lambda: pf.chance_map(subj, BASE_ALIGN)))
    for ev in evals:
        jobs.append((f"levels__{ev}", f"Absolute level — {pf.EVAL_LABEL[ev]}",
                     lambda e=ev: pf.per_dataset_levels(subj, e, BASE_ALIGN)))
    for ev in evals:
        jobs.append((f"win_matrix__{ev}", f"Head to head — {pf.EVAL_LABEL[ev]}",
                     lambda e=ev: pf.win_matrix(subj, e, BASE_ALIGN)))
    jobs += [
        ("mean_rank", "Mean rank on the complete square",
         lambda: pf.mean_rank(subj, BASE_ALIGN)),
        ("champion_share", "Who wins a subject, and by how much",
         lambda: pf.champion_share(subj, BASE_ALIGN)),
    ]
    for ev in evals:
        jobs.append((f"pareto__{ev}", f"Accuracy against size — {pf.EVAL_LABEL[ev]}",
                     lambda e=ev: pf.pareto(subj, e, BASE_ALIGN)))
    for ev in evals:
        jobs.append((f"pareto_subj__{ev}",
                     f"Accuracy against size, subject by subject — {pf.EVAL_LABEL[ev]}",
                     lambda e=ev: pf.pareto_subjects(subj, e, BASE_ALIGN)))
    jobs += [
        ("width_reached", "Width matching", lambda: pf.width_reached(subj, BASE_ALIGN)),
        ("cost", "What a fold cost", lambda: pf.cost(subj, BASE_ALIGN)),
        ("alignment", "Euclidean alignment", lambda: pf.alignment_effect(subj)),
        ("protocol", "What generalisation costs",
         lambda: pf.protocol_penalty(subj, BASE_ALIGN)),
        ("train_size", "Score against training-set size",
         lambda: pf.train_size(subj, BASE_ALIGN)),
    ]

    blocks, skipped = [], []
    for name, title, fn in jobs:
        try:
            fig = fn()
        except Exception as exc:  # a figure that raises must not take the page with it
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        if fig is None:
            print(f"  - {name}: no data")
            skipped.append((name, "no data in the frames for this cut"))
            continue
        fig.savefig(figs_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        blocks.append((name, title, fn, base64.b64encode(buf.getvalue()).decode()))
        print(f"  + {name}: {len(buf.getvalue()) / 1e6:.2f} MB")

    _write_page(out, blocks, skipped, sc, subj)


# ------------------------------------------------------------------------- the page

#: Figure name -> the ``perf_figures`` function that drew it. Names carry a sweep suffix
#: (`decomp_ds__shallow__within_session`); the caption has to resolve back through it to
#: one docstring, so the mapping is written once here rather than parsed out of the name.
SRC = {"coverage": "coverage", "decomposition": "decomposition",
       "decomposition_ea": "decomposition", "power": "power",
       "seed_noise": "seed_noise", "decomp_ds": "decomposition_by_dataset",
       "subject_delta": "subject_delta", "dumbbell": "dumbbell",
       "pareto_subj": "pareto_subjects", "chance_map": "chance_map",
       "levels": "per_dataset_levels", "win_matrix": "win_matrix",
       "mean_rank": "mean_rank", "champion_share": "champion_share",
       "pareto": "pareto", "width_reached": "width_reached", "cost": "cost",
       "alignment": "alignment_effect", "protocol": "protocol_penalty",
       "train_size": "train_size"}


def _para(text: str) -> str:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{html.escape(' '.join(p.split()))}</p>" for p in parts)


def _fmt(r: dict) -> str:
    return (f"{r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}], "
            f"{r['n_win']}/{r['n']} subjects")


def _findings(sc: pd.DataFrame, subj: pd.DataFrame) -> list[tuple[str, str]]:
    """The numbers the reader needs before the figures, read off the frames.

    Computed, never written down: a re-render against a further-along campaign updates
    the prose it leads with instead of contradicting it. Every number here is at
    subject level with a bootstrap interval, because that is the only level at which
    this grid's units are independent.
    """
    out: list[tuple[str, str]] = []
    al = BASE_ALIGN

    # 1. The decomposition, everywhere it can be computed, Holm-corrected as one family.
    rows = []
    for ev in perf_io.EVAL_ORDER:
        for arch in perf_io.TRIPLES:
            d = perf_io.decompose(subj, arch, ev, al)
            if d and d["has_control"]:
                rows.append((ev, arch, d))
    if rows:
        adj = perf_io.holm([d["growth"]["p"] for _, _, d in rows])
        sig = [(ev, arch, d) for (ev, arch, d), p in zip(rows, adj) if p < 0.05]
        big = max(rows, key=lambda r: abs(r[2]["codebase"]["mean"]))
        out.append((
            "The growth term is the small one, everywhere it can be measured",
            "Across the "
            f"{len(rows)} (protocol × architecture) cells where the fixed control "
            "exists, <b>grow − fix</b> ranges "
            f"{min(d['growth']['mean'] for _, _, d in rows):+.4f} to "
            f"{max(d['growth']['mean'] for _, _, d in rows):+.4f}, and "
            f"<b>{len(sig)} of {len(rows)}</b> survive Holm correction. The codebase "
            f"term <b>fix − bd</b> is larger in "
            f"{sum(abs(d['codebase']['mean']) > abs(d['growth']['mean']) for _, _, d in rows)}"
            f" of them — largest on <code>{big[1]}</code> / {big[0].replace('_', '-')} "
            f"at {_fmt(big[2]['codebase'])}."))

        worst = min(rows, key=lambda r: r[2]["total"]["mean"])
        out.append((
            "Where the headline is negative, it is the re-implementation that is losing",
            f"<code>{worst[1]}</code> / {worst[0].replace('_', '-')}: "
            f"<b>grow − bd</b> = {_fmt(worst[2]['total'])}, of which growth contributes "
            f"{worst[2]['growth']['mean']:+.4f} and the codebase "
            f"{worst[2]['codebase']['mean']:+.4f}. Reported without the control, this "
            "reads as evidence that growth hurts. It is not."))

    # 2. Where growth is genuinely ahead, if anywhere.
    best = None
    for ev in perf_io.EVAL_ORDER:
        for arch in perf_io.TRIPLES:
            d = perf_io.decompose(subj, arch, ev, al)
            if d and d["has_control"] and d["growth"]["lo"] > 0:
                if best is None or d["growth"]["mean"] > best[2]["growth"]["mean"]:
                    best = (ev, arch, d)
    if best:
        g = best[2]["growth"]
        out.append((
            "Growth does win somewhere — by less than a seed",
            f"<code>{best[1]}</code> / {best[0].replace('_', '-')}: "
            f"<b>grow − fix</b> = {_fmt(g)}, interval clear of zero. Compare the "
            "median seed-to-seed standard deviation of "
            f"<b>{_seed_sd(sc, al):.4f}</b> on the same grid: the effect is real and "
            "smaller than re-running the same configuration."))
    else:
        out.append((
            "No protocol × architecture cell shows growth ahead of its own control",
            "Every <b>grow − fix</b> interval crosses zero at subject level. Read "
            "<a href=\"#power\">figure 03</a> before reading that as a null: several of "
            "these contrasts have a minimum detectable effect larger than the effect "
            "under discussion."))

    # 3. The chance audit.
    bad = subj[subj.above_chance < 0.5]
    if len(bad):
        by = (bad.groupby(["eval", "model"]).size().sort_values(ascending=False))
        out.append((
            "Some arms never left chance, and every delta over them is arithmetic",
            f"{len(bad):,} of {len(subj):,} subject-level cells "
            f"({100 * len(bad) / len(subj):.1f} %) sit at or below their dataset's "
            "chance threshold on the majority of their folds. Worst: "
            + ", ".join(f"<code>{m}</code>/{e.replace('_', '-')} ({n})"
                        for (e, m), n in by.head(3).items())
            + ". Those cells are hatched wherever they enter a comparison."))

    # 4. The structural gap.
    per_eval = sc.groupby(["model", "eval"]).dataset.nunique().unstack().fillna(0)
    if "cross_subject" in per_eval:
        fix = per_eval.loc[per_eval.index.str.startswith("fix_"), "cross_subject"]
        if len(fix) and (fix == 0).all():
            n_ls = sum(1 for arch in perf_io.TRIPLES
                       if perf_io.decompose(subj, arch, "cross_subject", al))
            out.append((
                "On cross-subject nothing is decomposable, by design",
                f"The {len(fix)} <code>fix_*</code> arms have zero datasets under "
                "<code>cross_subject</code> — zero claims in the scheduler, zero rows "
                "in the grid plan, so this is the grid as planned and no amount of "
                f"waiting closes it. The {n_ls} LOSO headlines below therefore cannot "
                "be split, and LOSO is the protocol a deployed decoder is judged on."))

    # 5. Coverage, so no number above is read as final.
    n_cell = len(sc.groupby(["eval", "dataset", "model", "align_tag", "seed"]))
    out.append((
        "This is a campaign in flight",
        f"{n_cell:,} cells scored across {sc.dataset.nunique()} datasets, "
        f"{sc.model.nunique()} arms and {sc['eval'].nunique()} protocols. Coverage is "
        "uneven between arms and protocols; <a href=\"#coverage\">figure 01</a> is the "
        "map, and every interval here is over the subjects actually present in both "
        "arms of its own contrast."))
    return out


def _seed_sd(sc: pd.DataFrame, align_tag: str) -> float:
    s = (sc[sc.align_tag == align_tag]
         .groupby(["eval", "dataset", "model", "subject", "session"])
         .score.agg(["std", "size"]))
    s = s[s["size"] >= 2]["std"].dropna()
    return float(s.median()) if len(s) else np.nan


def _write_page(out: Path, blocks, skipped, sc: pd.DataFrame,
                subj: pd.DataFrame) -> None:
    n_cell = len(sc.groupby(["eval", "dataset", "model", "align_tag", "seed"]))
    body = [f"""<header>
<p class="eyebrow">eegrow · campagne finale · <code>results_final</code></p>
<h1>Où la croissance gagne</h1>
<p class="standfirst">Le score, et ce qu'il vaut. <code>grow − bd</code> est le chiffre
qu'on citerait&nbsp;; il additionne deux choses — la croissance, et le fait que la classe
soit celle d'eegrow plutôt que celle de braindecode. Le bras de contrôle
<code>fix_*</code> les sépare, et c'est la seule mesure qui le permette.</p>
<dl class="stats">
<div><dt>cellules</dt><dd>{n_cell:,}</dd></div>
<div><dt>scores</dt><dd>{len(sc):,}</dd></div>
<div><dt>unités sujet</dt><dd>{len(subj):,}</dd></div>
<div><dt>datasets</dt><dd>{sc.dataset.nunique()}</dd></div>
</dl>
<p class="caveat">Campagne <b>en cours</b>. L'unité d'analyse est le <b>sujet</b> —
sessions et graines sont moyennées avant tout test, parce qu'elles sont corrélées.
Toutes les barres sont des bootstraps sur les sujets. Rapports jumeaux&nbsp;:
<a href="{DYNAMICS_URL}"><i>Growth dynamics</i></a>, qui décrit le mécanisme que ces
scores mesurent — même campagne, même palette, 32 figures&nbsp;; et
<a href="{DXR_URL}"><i>Donneur → receveur</i></a>, qui demande si la taille atteinte par
un modèle qui a poussé mesure le sujet sur lequel il a poussé — les quatre étages de la
chaîne et leurs verdicts, dont deux négatifs.</p>
</header>"""]

    body.append('<section class="findings"><h2>Ce que les figures disent</h2><dl>')
    for head, text in _findings(sc, subj):
        body.append(f"<div><dt>{html.escape(head)}</dt><dd>{text}</dd></div>")
    body.append("</dl></section>")

    body.append('<nav aria-label="Figures"><h2>Figures</h2><ol>')
    for i, (name, title, _, _) in enumerate(blocks, 1):
        body.append(f'<li><span class="num">{i:02d}</span>'
                    f'<a href="#{name}">{html.escape(title)}</a></li>')
    body.append("</ol></nav>")

    for i, (name, title, fn, b64) in enumerate(blocks, 1):
        src = SRC.get(name.split("__")[0], name.split("__")[0])
        doc = textwrap.dedent(getattr(pf, src).__doc__ or "").strip()
        body.append(f"""<section id="{name}">
<p class="eyebrow"><span class="num">{i:02d}</span>
<code>perf_figures.{html.escape(src)}</code></p>
<h2>{html.escape(title)}</h2>
<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(title)}"
 loading="lazy"></figure>
<div class="caption">{_para(doc)}</div>
</section>""")

    if skipped:
        body.append('<section class="skipped"><h2>Figures non produites</h2><ul>')
        for name, why in skipped:
            body.append(f"<li><code>{html.escape(name)}</code> — "
                        f"{html.escape(why)}</li>")
        body.append("</ul></section>")

    inner = "".join(body)
    (out / "perf_report_body.html").write_text(f"<style>{_CSS}</style>{inner}")
    (out / "perf_report.html").write_text(
        f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Où la croissance gagne — campagne finale</title>"
        f"<style>{_CSS}</style></head><body>{inner}</body></html>")
    size = (out / "perf_report.html").stat().st_size / 1e6
    print(f"\n{len(blocks)} figures, {len(skipped)} skipped -> "
          f"{out / 'perf_report.html'} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
