"""Render every figure in ``growth_dynamics`` and assemble them into one HTML page.

Deliberately dumb: it holds the *sweep* (which eval, which alignment, which showcase
cell) and nothing else. Every decision about what a figure means lives in the figure's
own docstring, which this script lifts verbatim into the page -- so a caption can never
drift from the code that drew it, which is the failure mode of a report written
separately from its figures.

A figure that returns ``None`` is skipped and *named* in the page's skipped list rather
than silently dropped: on a campaign that is 80 % done, "this figure is absent" and
"this figure is empty" are different facts and only one of them is about the data.

    python benchmarks/analysis/build_growth_dynamics.py <frames_dir> <out_dir>
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import growth_dynamics as gd  # noqa: E402

#: Rendering resolution. 110 keeps a 14-inch panel legible while holding the assembled
#: page under the size a browser will open without complaint; the PNGs on disk are the
#: same files, so there is one resolution to reason about rather than two.
DPI = 110

#: The cell every per-fold figure showcases. bnci2014_001 for the same reason the rest
#: of the eegrow report uses it: it is the dataset every other figure is drawn on, so a
#: trajectory here can be read next to a score there.
SHOWCASE = dict(eval_="within_session", dataset="bnci2014_001", align="")

#: The twin report: same campaign, same palette, the score side of the same question.
#: Each page names the other because neither is readable alone -- a mechanism finding
#: with no effect size is trivia, and an effect size with no mechanism is a table.
PERF_URL = "https://claude.ai/code/artifact/44b84ba1-d91c-418a-a223-6c0b923cc6ac"


def _sweep(fits: pd.DataFrame) -> list[tuple[str, str]]:
    """The (eval, align) pairs the campaign actually contains, in protocol order."""
    have = {(e, a if isinstance(a, str) else "")
            for e, a in zip(fits["eval"], fits.align_tag.fillna(""))}
    return [(e, a) for e in gd.EVAL_LABEL for a in ("", "easubject") if (e, a) in have]


def main() -> None:
    frames, out = Path(sys.argv[1]), Path(sys.argv[2])
    figs_dir = out / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fits = pd.read_csv(frames / "gd_fits.csv.gz")
    cm = pd.read_csv(frames / "gd_curves_mean.csv.gz")
    events = pd.read_csv(frames / "gd_events.csv.gz")
    for f in (fits, cm, events):
        f["align_tag"] = f.align_tag.fillna("")
    print(f"{len(fits):,} folds, {len(cm):,} curve rows, {len(events):,} growth events")

    sweep = _sweep(fits)
    jobs: list[tuple[str, str, callable]] = [
        ("coverage", "Campaign coverage", lambda: gd.coverage(fits)),
    ]

    for ev, al in sweep:
        tag = f"{ev}{'_ea' if al else ''}"
        jobs.append((f"grad_norm__{tag}", f"Gradient norm — {ev}{' + EA' if al else ''}",
                     lambda ev=ev, al=al: gd.gradient_norm_curves(cm, events, eval_=ev,
                                                                  align=al)))
    jobs += [
        ("grad_at_growth", "Gradient across a growth step",
         lambda: gd.gradient_norm_at_growth(events)),
        ("optimizer", "Learning rate and optimiser stamp",
         lambda: gd.learning_rate_and_optimizer(cm, fits, **{
             "eval_": SHOWCASE["eval_"], "align": SHOWCASE["align"]})),
        ("adam_eps", "AdamW eps attenuation",
         lambda: gd.adam_eps_attenuation(cm, events, eval_=SHOWCASE["eval_"],
                                         align=SHOWCASE["align"])),
        ("abstention", "The line search says no",
         lambda: gd.growth_abstention(events)),
        ("event_timeline", "Growth events, fold by fold",
         lambda: gd.growth_event_timeline(events, **SHOWCASE)),
        ("neurons_added", "Neurons proposed, kept, and what they cost",
         lambda: gd.neurons_added(events)),
    ]
    for ev, al in sweep:
        tag = f"{ev}{'_ea' if al else ''}"
        jobs.append((f"width__{tag}", f"Width trajectory — {ev}{' + EA' if al else ''}",
                     lambda ev=ev, al=al: gd.width_trajectory(cm, fits, eval_=ev,
                                                              align=al)))
    jobs += [
        ("width_reached", "Width reached against the target",
         lambda: gd.width_reached(fits)),
        ("eig_spectra", "Candidate spectra, one fold",
         lambda: gd.eigenvalue_spectra(events, **SHOWCASE)),
        ("eig_summary", "Candidate spectra, whole campaign",
         lambda: gd.eigenvalue_summary(events)),
        ("line_search", "The line-search factor s",
         lambda: gd.line_search_factor(events)),
        ("first_order", "Predicted gain against realised gain",
         lambda: gd.first_order_expected_vs_realised(events)),
        ("aftermath", "A growth step on the held-out split",
         lambda: gd.growth_step_aftermath(events)),
        ("epoch_cost", "Wall time per epoch as the network grows",
         lambda: gd.epoch_cost(cm, eval_=SHOWCASE["eval_"], align=SHOWCASE["align"])),
        ("stop_reason", "Why every fold ended",
         lambda: gd.stop_reason_breakdown(fits)),
        ("stopping", "Stopping and selection", lambda: gd.stopping_epochs(fits)),
        ("selected_width", "Which model actually got scored",
         lambda: gd.selected_model_width(fits, events)),
    ]
    for ev, al in sweep:
        if al:
            continue
        tag = f"{ev}"
        jobs.append((f"stop_subject__{tag}", f"Stopping per subject — {ev}",
                     lambda ev=ev: gd.stop_by_subject(fits, eval_=ev, align="")))

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
        doc = textwrap.dedent(fn.__doc__ or "") if fn.__doc__ else ""
        blocks.append((name, title, doc,
                       base64.b64encode(buf.getvalue()).decode()))
        print(f"  + {name}: {len(buf.getvalue()) / 1e6:.2f} MB")

    _write_page(out, blocks, skipped, fits, events)


def _doc_of(fn_name: str) -> str:
    fn = getattr(gd, fn_name, None)
    return textwrap.dedent(fn.__doc__ or "").strip() if fn else ""


def _para(text: str) -> str:
    """Docstring -> paragraphs, preserving the blank-line structure of the source."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{html.escape(' '.join(p.split()))}</p>" for p in parts)


def _findings(fits: pd.DataFrame, events: pd.DataFrame) -> list[tuple[str, str]]:
    """The numbers a reader needs before the figures, read off the frames.

    Computed rather than written down, so a re-render against a further-along campaign
    updates the prose it leads with instead of contradicting it.
    """
    out = []
    grow = fits[fits.model.str.startswith("grow")]
    ap = events[events.applied.fillna(False).astype(bool)] if "applied" in events else events

    out.append(("Every fold ran the full budget",
                f"{len(fits):,} folds, {fits.stop_reason.nunique()} distinct stop reason "
                f"(<code>{fits.stop_reason.mode()[0]}</code>), epochs "
                f"{int(fits.epochs.min())}–{int(fits.epochs.max())}. With "
                f"<code>patience=200</code> against <code>max_epochs=200</code>, early "
                f"stopping cannot fire. Nothing below is confounded by unequal "
                f"training length."))

    med = fits.groupby("model").restored_epoch.median()
    out.append(("...and ~90 % of it past its own optimum",
                "Median epoch of the model that actually gets scored: "
                + ", ".join(f"<b>{m}</b> {v:.0f}" for m, v in med.items())
                + " — out of 200."))

    if "applied" in events:
        rate = (1 - events.groupby("model").applied.mean()) * 100
        out.append(("The line search refuses most of what it is offered",
                    f"{len(events):,} growth opportunities, "
                    f"{100 * events.applied.mean():.1f} % applied. Refusal rate "
                    + ", ".join(f"<b>{m}</b> {v:.1f} %" for m, v in rate.items())
                    + ". A refusal is an abstention, not a cap — the arm tries again "
                      "five epochs later, and can refuse every time."))

    if "grow_s" in ap:
        out.append(("When it does accept, it takes the grid's ceiling",
                    f"<b>s = 1.0 on {100 * (ap.grow_s == 1.0).mean():.1f} %</b> of "
                    f"{len(ap):,} applied steps. <code>SCALING_GRID = "
                    f"(0.0, 0.1, 0.5, 1.0)</code> — 1.0 is its maximum, so the "
                    "loss-minimising amplitude is plausibly above it and the grid has "
                    "never been allowed to say so."))

    r = grow.groupby("model").agg(reached=("reached_target", "mean"),
                                  end=("width_end", "median"),
                                  tgt=("target_width", "median"))
    out.append(("A growing arm usually is not width-matched with its controls",
                "Folds reaching the target width: "
                + ", ".join(f"<b>{m}</b> {row.reached * 100:.0f} % "
                            f"(median {row.end:.0f} of {row.tgt:.0f})"
                            for m, row in r.iterrows())
                + ". Every parameter-efficiency claim reads <code>width_end</code>; "
                  "this is what that column contains."))

    miss = (fits.groupby(["model", "eval"]).dataset.nunique().unstack().fillna(0))
    if "cross_subject" in miss and (miss.loc[miss.index.str.startswith("fix_"),
                                             "cross_subject"] == 0).all():
        out.append(("There is no fixed control on cross-subject at all",
                    "The three <code>fix_*</code> arms have 0 datasets under "
                    "<code>cross_subject</code> — and 0 claims in the scheduler, so this "
                    "is the planned grid, not an incomplete one. On LOSO, "
                    "<code>grow − bd</code> cannot be decomposed into growth and "
                    "codebase. It is the protocol where that decomposition mattered most."))
    return out


def _write_page(out: Path, blocks, skipped, fits: pd.DataFrame,
                events: pd.DataFrame) -> None:
    # The figure name maps back to the function that drew it, so the caption is the
    # function's own docstring rather than a second description that can go stale.
    fn_for = {b[0]: b[0].split("__")[0] for b in blocks}
    alias = {"grad_norm": "gradient_norm_curves", "grad_at_growth":
             "gradient_norm_at_growth", "optimizer": "learning_rate_and_optimizer",
             "adam_eps": "adam_eps_attenuation", "event_timeline":
             "growth_event_timeline", "neurons_added": "neurons_added",
             "abstention": "growth_abstention",
             "width": "width_trajectory", "width_reached": "width_reached",
             "eig_spectra": "eigenvalue_spectra", "eig_summary": "eigenvalue_summary",
             "line_search": "line_search_factor", "first_order":
             "first_order_expected_vs_realised", "aftermath": "growth_step_aftermath",
             "epoch_cost": "epoch_cost", "stop_reason": "stop_reason_breakdown",
             "stopping": "stopping_epochs", "selected_width": "selected_model_width",
             "stop_subject": "stop_by_subject", "coverage": "coverage"}

    n_cell = len(fits.groupby(["eval", "dataset", "model", "align_tag", "seed"]))
    body = [f"""<header>
<p class="eyebrow">eegrow · campagne finale · <code>results_final</code></p>
<h1>Growth dynamics</h1>
<p class="standfirst">Ce qui se passe <em>à l'intérieur</em> d'un fit qui grandit :
l'optimiseur, la décision de croissance, l'arrêt. Toutes les colonnes lues ici sont
enregistrées depuis la première campagne et n'avaient jusqu'à présent été lues par
aucun code de tracé.</p>
<dl class="stats">
<div><dt>cellules</dt><dd>{n_cell:,}</dd></div>
<div><dt>folds</dt><dd>{len(fits):,}</dd></div>
<div><dt>opportunités de croissance</dt><dd>{len(events):,}</dd></div>
<div><dt>neurones ajoutés</dt><dd>{events.loc[events.applied.fillna(False).astype(bool), 'grow_n_kept'].sum():,.0f}</dd></div>
</dl>
<p class="caveat">Campagne <b>en cours</b>. Ces figures décrivent ce qui est écrit sur
disque à l'instant de l'export, pas la grille complète. Lire
<a href="#coverage">Campaign coverage</a> en premier. Rapport jumeau&nbsp;:
<a href="{PERF_URL}"><i>Où la croissance gagne</i></a> — les scores que ce mécanisme
produit, décomposés en terme de croissance et terme de codebase.</p>
</header>"""]

    body.append('<section class="findings"><h2>Ce que les 32 figures disent</h2><dl>')
    for head, text in _findings(fits, events):
        body.append(f"<div><dt>{html.escape(head)}</dt><dd>{text}</dd></div>")
    body.append("</dl></section>")

    body.append('<nav aria-label="Figures"><h2>Figures</h2><ol>')
    for i, (name, title, _, _) in enumerate(blocks, 1):
        body.append(f'<li><span class="num">{i:02d}</span>'
                    f'<a href="#{name}">{html.escape(title)}</a></li>')
    body.append("</ol></nav>")

    for i, (name, title, _, b64) in enumerate(blocks, 1):
        src = alias.get(fn_for[name], fn_for[name])
        doc = _doc_of(src)
        body.append(f"""<section id="{name}">
<p class="eyebrow"><span class="num">{i:02d}</span>
<code>growth_dynamics.{html.escape(src)}</code></p>
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
    (out / "growth_dynamics_body.html").write_text(f"<style>{_CSS}</style>{inner}")
    (out / "growth_dynamics.html").write_text(
        f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Growth dynamics — campagne finale</title><style>{_CSS}</style>"
        f"</head><body>{inner}</body></html>")
    size = (out / "growth_dynamics.html").stat().st_size / 1e6
    print(f"\n{len(blocks)} figures, {len(skipped)} skipped -> "
          f"{out / 'growth_dynamics.html'} ({size:.1f} MB)")


#: PALETTE. Neutrals biased cool -- the subject is instrumentation, not warmth -- and
#: the two accents are lifted straight from `growth_dynamics.MODEL_COLOR`: the growing
#: arm's red and the fixed control's teal. The page and the plots are one artifact, so
#: they share one palette rather than each having its own.
#:
#: TYPE. The journal-figure convention, which is this subject's own vernacular: a
#: grotesque for headings and labels, a serif for the caption prose that carries the
#: argument, a monospace for anything that names code. No webfont -- the CSP blocks font
#: CDNs and a 8 MB page has no room for an inlined face -- so each role is a deliberately
#: chosen system stack rather than one default sans doing all three jobs.
_CSS = """
:root {
  --ground:#f6f7f9; --surface:#ffffff; --ink:#12171c; --mut:#5b6672;
  --rule:#dde3e9; --rule-soft:#eaeef2;
  --accent:#a51f18; --accent-soft:#f3e3e2; --teal:#00695c;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --serif:ui-serif,Charter,"Bitstream Charter",Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0e1216; --surface:#151a20; --ink:#e3e8ed; --mut:#93a0ad;
    --rule:#242b33; --rule-soft:#1c222a;
    --accent:#f0776d; --accent-soft:#2a1a19; --teal:#4db6ac;
  }
}
:root[data-theme="dark"] {
  --ground:#0e1216; --surface:#151a20; --ink:#e3e8ed; --mut:#93a0ad;
  --rule:#242b33; --rule-soft:#1c222a;
  --accent:#f0776d; --accent-soft:#2a1a19; --teal:#4db6ac;
}
* { box-sizing:border-box; }
body {
  margin:0; padding:3.5rem 1.5rem 7rem; background:var(--ground); color:var(--ink);
  font:400 17px/1.65 var(--serif); -webkit-font-smoothing:antialiased;
}
body > * { max-width:74ch; margin-inline:auto; }
h1, h2, h3, .eyebrow, nav, dt, .stats, figure figcaption {
  font-family:var(--sans);
}
h1 {
  font-size:clamp(2.1rem,5vw,3rem); font-weight:640; letter-spacing:-.025em;
  line-height:1.08; margin:.35rem 0 .9rem; text-wrap:balance;
}
h2 {
  font-size:1.32rem; font-weight:620; letter-spacing:-.015em; line-height:1.25;
  margin:.15rem 0 1rem; text-wrap:balance;
}
p { margin:0 0 1rem; }
a { color:var(--accent); text-underline-offset:.18em; }
a:focus-visible, li a:focus-visible {
  outline:2px solid var(--accent); outline-offset:3px; border-radius:2px;
}
code {
  font:500 .84em/1.4 var(--mono); background:var(--rule-soft);
  padding:.12em .4em; border-radius:3px; word-break:break-word;
}
.eyebrow {
  font:500 .72rem/1.4 var(--mono); letter-spacing:.1em; text-transform:uppercase;
  color:var(--mut); margin:0 0 .5rem;
}
.eyebrow code { background:none; padding:0; text-transform:none; letter-spacing:0; }
.num { color:var(--accent); font-variant-numeric:tabular-nums; }

header { padding-bottom:2rem; }
.standfirst { font-size:1.12rem; color:var(--mut); max-width:60ch; }
.standfirst em { color:var(--ink); font-style:italic; }
.stats {
  display:flex; flex-wrap:wrap; gap:0 2.5rem; margin:1.6rem 0 1.2rem;
  padding:1rem 0; border-block:1px solid var(--rule);
}
.stats div { display:flex; flex-direction:column; gap:.15rem; }
.stats dt {
  font:500 .7rem/1.3 var(--mono); letter-spacing:.08em; text-transform:uppercase;
  color:var(--mut);
}
.stats dd {
  margin:0; font:600 1.35rem/1 var(--sans); font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;
}
.caveat {
  font-size:.95rem; color:var(--mut); border-left:3px solid var(--accent);
  padding-left:.9rem; margin:0;
}

.findings { margin-block:3rem; }
.findings dl { display:flex; flex-direction:column; gap:1.4rem; margin:0; }
.findings dt {
  font-size:1rem; font-weight:620; letter-spacing:-.01em; margin-bottom:.3rem;
}
.findings dd { margin:0; color:var(--mut); font-size:.98rem; }
.findings dd b { color:var(--ink); font-weight:600;
  font-variant-numeric:tabular-nums; }

nav { margin-block:3rem 3.5rem; }
nav ol { list-style:none; margin:0; padding:0; columns:2; column-gap:2.5rem; }
nav li {
  break-inside:avoid; display:flex; gap:.6rem; align-items:baseline;
  font-size:.9rem; padding:.22rem 0;
}
nav .num { font:500 .74rem var(--mono); }
nav a { color:var(--ink); text-decoration:none; }
nav a:hover { color:var(--accent); text-decoration:underline; }

section[id] {
  padding-top:2.6rem; margin-top:2.6rem; border-top:1px solid var(--rule);
}
figure {
  margin:1.2rem 0 1.4rem; overflow-x:auto; background:var(--surface);
  border:1px solid var(--rule-soft); border-radius:2px; padding:.75rem;
  /* Figures are the widest thing on the page and must be allowed to break the
     measure the prose is set to, without ever scrolling the body sideways. */
  max-width:min(1240px,94vw); margin-inline:auto;
  width:max-content; min-width:100%;
}
img { display:block; max-width:100%; height:auto; margin-inline:auto; }
.caption {
  color:var(--mut); font-size:.96rem; border-left:2px solid var(--rule);
  padding-left:1.1rem;
}
.caption p { margin:0 0 .7rem; }
.caption p:first-child { color:var(--ink); }
.caption p:last-child { margin-bottom:0; }

.skipped ul { color:var(--mut); font-size:.94rem; }
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
@media (max-width:640px) {
  body { padding:2rem 1rem 4rem; font-size:16px; }
  nav ol { columns:1; }
  .stats { gap:0 1.5rem; }
}
"""


if __name__ == "__main__":
    main()
