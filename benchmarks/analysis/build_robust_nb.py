"""Generate benchmarks/analysis/eegrow_robustness.ipynb (power, noise floor, stability)."""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path("/Users/adammounir/Desktop/Inria/Exploration/eegrow/benchmarks/analysis/"
           "eegrow_robustness.ipynb")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "../results_published"

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Robustness: what the benchmark can and cannot detect

The results notebook reports effects. This one reports the **resolution** of the
instrument that measured them, which is what decides whether a non-significant pair is
evidence of no effect or simply an experiment too small to see one. Three things,
in the order a sceptical reader needs them:

1. **A negative control.** Split the seeds of a single model in half and compare it to
   itself through the exact same pipeline. The true effect is zero by construction, so
   whatever this returns is the noise floor of the whole procedure. Any real effect has
   to clear it.
2. **Minimum detectable effect.** Given the observed subject-to-subject spread and the
   number of subjects actually available, the smallest true difference this design
   would catch 80 % of the time. A pair whose MDE is larger than the effect anyone
   would care about has not shown "no difference"; it has shown nothing.
3. **Stability across seeds.** Growth is an extra source of run-to-run variance. If a
   growing arm is markedly noisier than its fixed control, its mean advantage is worth
   less than the same number from a stable arm.
""")

code(r"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("%s")           # <- the only path to change for another campaign
FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GROW_C, FIX_C, WARN_C = "#C2255C", "#4C6EF5", "#E8590C"

# Pairs are width-matched architectures. grow_deep is paired with fix_deepeeg (the same
# 2-stage net built frozen at the geometry growth ends on), NOT with bd_deep4, which is
# a different network -- that contrast would measure architecture, not growth.
PAIRS = {"grow_shallow": "bd_shallow", "grow_sccnet": "bd_sccnet",
         "grow_eegnex": "bd_eegnex", "grow_deep": "fix_deepeeg"}
ML = {"csp_lda", "csp_svm", "fgmdm", "mdm", "ts_lr", "ts_svm"}

scores = pd.read_csv(ROOT / "eegrow_benchmark_all_scores.csv.gz")
scores = scores[scores.get("align", "none") == "none"]

# Drop the replicates that are bit-identical copies rather than independent runs: the
# ML arms shared one MOABB HDF5 store across seeds, so seeds 1-4 re-read seed 0's rows
# (proved by wall-clock times identical to the microsecond). Keeping them would inflate
# the effective n fivefold and shrink every interval that follows.
KEY = ["eval", "dataset", "model", "subject", "session"]
nuniq = scores.groupby(KEY)["score"].nunique()
degenerate = scores.set_index(KEY).index.isin(nuniq[nuniq == 1].index)
keep = (~degenerate) | (scores["seed"] == scores["seed"].min())
print(f"{len(scores)} rows -> {int(keep.sum())} after dropping redundant replicates")
df = scores[keep].copy()

# One value per (eval, dataset, subject): seeds and sessions averaged first.
# The subject is the pairing unit. Seeds are replicates of one measurement and
# sessions are repeated observations of one person, so both are averaged BEFORE any
# statistic -- otherwise the same subject enters the test several times and the
# effective n is wrong.
def per_subject(frame, model):
    return (frame[frame.model == model]
            .groupby(["eval", "dataset", "subject"], as_index=False)["score"].mean())
""" % ROOT)

md(r"""
## 1. Negative control — the noise floor of the procedure

A model against itself. The seeds of one arm are split into two disjoint halves and run
through exactly the pipeline used for a real contrast: average within subject, take
paired differences, summarise. The expected effect is exactly zero, so the spread of
what comes back **is** the floor. Reported per arm, since a growing arm is entitled to
be noisier than a fixed one.
""")

code(r"""
def split_half_delta(frame, model, rng):
    seeds = sorted(frame.loc[frame.model == model, "seed"].unique())
    if len(seeds) < 2:
        return None
    seeds = list(rng.permutation(seeds))
    a, b = seeds[: len(seeds) // 2], seeds[len(seeds) // 2:]
    sub = frame[frame.model == model]
    ga = (sub[sub.seed.isin(a)].groupby(["eval","dataset","subject"], as_index=False)
          ["score"].mean())
    gb = (sub[sub.seed.isin(b)].groupby(["eval","dataset","subject"], as_index=False)
          ["score"].mean())
    m = ga.merge(gb, on=["eval","dataset","subject"], suffixes=("_a","_b"))
    return m.assign(delta=m.score_a - m.score_b)

rng = np.random.default_rng(0)
rows = []
for model in sorted(set(PAIRS) | set(PAIRS.values()) | ML):
    if model not in set(df.model):
        continue
    m = split_half_delta(df, model, rng)
    if m is None or m.empty:
        continue
    for ev, g in m.groupby("eval"):
        d = g["delta"].to_numpy()
        rows.append(dict(model=model, eval=ev, n=len(d), mean=d.mean(),
                         sd=d.std(ddof=1), p95_abs=np.percentile(np.abs(d), 95)))
floor = pd.DataFrame(rows)
if floor.empty:
    print("not enough seeds for a split-half control")
else:
    floor["family"] = np.where(floor.model.str.startswith("grow"), "growing",
                        np.where(floor.model.isin(ML), "riemann/csp", "fixed deep"))
    display(floor.groupby(["family","eval"])[["mean","sd","p95_abs"]]
            .mean().round(4))
    print("\n`mean` should sit at ~0 -- it is the same model on both sides.")
    print("`p95_abs` is the size a difference has to beat to be distinguishable")
    print("from run-to-run noise on a single subject.")
""")

code(r"""
if not floor.empty:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    order = [f for f in ("riemann/csp", "fixed deep", "growing") if f in set(floor.family)]
    evs = [e for e in ("within_session","cross_session","cross_subject")
           if e in set(floor["eval"])]
    w = 0.8 / max(len(evs), 1)
    for j, ev in enumerate(evs):
        sub = floor[floor["eval"] == ev].groupby("family")["p95_abs"].mean().reindex(order)
        ax.bar(np.arange(len(order)) + j*w - 0.4 + w/2, sub.values, width=w*0.9,
               label=ev)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("noise floor  (95th pct of |self-vs-self delta|)")
    ax.set_title("What a difference has to beat to be more than run-to-run noise")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS/"10_noise_floor.png", bbox_inches="tight")
    plt.show()
""")

md(r"""
## 2. Minimum detectable effect

For each pair and protocol: the observed spread of paired subject-level differences and
the number of subjects give the smallest true effect a two-sided paired test at
alpha = 0.05 would detect with 80 % power,

$$\mathrm{MDE} = (z_{0.975} + z_{0.80})\,\frac{s_d}{\sqrt{n}}$$

Read the table with the observed delta next to it. A pair where |delta| < MDE and
p > 0.05 is **not** evidence of equivalence -- the experiment could not have resolved
an effect that size. The `n_for_observed` column says how many subjects it would take.

`resolvable = False` alongside `p < 0.05` is not a contradiction. The MDE is a normal
approximation built on the standard deviation, while the reported p comes from
Wilcoxon, a rank test. When the paired differences have heavier tails than a Gaussian
-- which is the normal state of affairs across subjects -- the rank test detects what
the parametric power calculation says it should not. The honest reading is that the
effect is real but small relative to the spread, so it should be reported with its
interval and never as a per-subject expectation.
""")

code(r"""
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
rows = []
for grow, fixed in PAIRS.items():
    if not {grow, fixed} <= set(df.model):
        continue
    a, b = per_subject(df, grow), per_subject(df, fixed)
    m = a.merge(b, on=["eval","dataset","subject"], suffixes=("_g","_f"))
    m["delta"] = m.score_g - m.score_f
    for ev, g in m.groupby("eval"):
        d = g["delta"].to_numpy()
        n, sd = len(d), g["delta"].std(ddof=1)
        mde = Z * sd / np.sqrt(n) if n > 1 and sd > 0 else np.nan
        p = stats.wilcoxon(d).pvalue if n >= 6 and np.ptp(d) > 0 else np.nan
        need = (Z * sd / abs(d.mean()))**2 if d.mean() != 0 and sd > 0 else np.nan
        rows.append(dict(pair=f"{grow} vs {fixed}", eval=ev, n_subjects=n,
                         delta=d.mean(), sd=sd, mde=mde, p=p,
                         resolvable=abs(d.mean()) >= mde if mde == mde else False,
                         n_for_observed=np.ceil(need) if need == need else np.nan))
power = pd.DataFrame(rows)
display(power.round(4).to_string(index=False))
if not power.empty:
    weak = power[(power.p > 0.05) & (~power.resolvable)]
    if len(weak):
        print("\nUNDERPOWERED, not null -- these pairs cannot support 'no difference':")
        for _, r in weak.iterrows():
            print(f"  {r['pair']:28s} {r['eval']:15s} delta={r.delta:+.4f} "
                  f"MDE={r.mde:.4f}  would need n≈{r.n_for_observed:.0f} "
                  f"(have {r.n_subjects})")
    else:
        print("\nNo pair is both non-significant and under-resolved: every null result")
        print("here is a well-powered null, and can be reported as an absence of")
        print("effect rather than an absence of evidence.")
""")

code(r"""
if not power.empty:
    p = power.dropna(subset=["mde"]).copy()
    p["label"] = p["pair"].str.replace(" vs ", "\nvs ") + "\n" + p["eval"]
    fig, ax = plt.subplots(figsize=(max(7, 0.9*len(p)), 4.2))
    x = np.arange(len(p))
    # Drawn as a symmetric band around zero, not a positive bar: the MDE is a
    # threshold on |effect|, and half of these effects are negative. A one-sided bar
    # would make "beyond the grey" unreadable for exactly the pairs where growth loses.
    ax.bar(x, 2 * p["mde"], bottom=-p["mde"], color="0.85", width=.75,
           label="undetectable zone (|effect| < MDE, 80 % power)")
    ax.scatter(x, p["delta"], color=np.where(p["delta"] > 0, GROW_C, WARN_C),
               zorder=3, s=46, label="observed effect")
    ax.axhline(0, color="0.4", lw=.8)
    for xi, r in zip(x, p.itertuples()):
        ax.plot([xi, xi], [0, r.delta], color="0.5", lw=.9, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(p["label"], fontsize=6.5)
    ax.set_ylabel("effect size (score units)")
    ax.set_title("A point inside the grey band could not have been detected either way")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS/"11_power.png", bbox_inches="tight")
    plt.show()
""")

md(r"""
## 3. Seed stability

Per (protocol, dataset, subject, session), the standard deviation across seeds. Growth
adds a data-dependent decision to training, so it is fair to ask whether it buys its
mean advantage at the cost of reproducibility. Only cells with genuinely distinct seeds
are counted -- the redundant ML replicates would otherwise report a spread of exactly
zero and flatter that family.
""")

code(r"""
var = (scores[~degenerate]
       .groupby(["eval","dataset","model","subject","session"], as_index=False)
       .agg(sd=("score","std"), n=("score","size")))
var = var[(var.n >= 3) & var.sd.notna()]
if var.empty:
    print("no cell with 3+ distinct seeds")
else:
    var["family"] = np.where(var.model.str.startswith("grow"), "growing",
                      np.where(var.model.isin(ML), "riemann/csp", "fixed deep"))
    display(var.groupby(["family","eval"])["sd"].describe()[["count","mean","50%"]]
            .round(4))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fams = [f for f in ("riemann/csp","fixed deep","growing") if f in set(var.family)]
    data = [var.loc[var.family == f, "sd"].to_numpy() for f in fams]
    bp = ax.boxplot(data, labels=fams, showfliers=False, patch_artist=True, widths=.55)
    for patch, f in zip(bp["boxes"], fams):
        patch.set_facecolor(GROW_C if f == "growing" else FIX_C); patch.set_alpha(.35)
    ax.set_ylabel("sd across seeds (within subject × session)")
    ax.set_title("Does growth cost reproducibility?")
    fig.tight_layout(); fig.savefig(FIGS/"12_seed_stability.png", bbox_inches="tight")
    plt.show()
    for ev, g in var.groupby("eval"):
        a = g.loc[g.family == "growing", "sd"]
        b = g.loc[g.family == "fixed deep", "sd"]
        if len(a) > 10 and len(b) > 10:
            u = stats.mannwhitneyu(a, b)
            print(f"{ev:15s} growing median {a.median():.4f} vs fixed "
                  f"{b.median():.4f}   Mann-Whitney p={u.pvalue:.2g}")
""")

md(r"""
## 4. Win / loss per dataset

The paired verdict in the least aggregated form there is: for every pair and every
dataset, the fraction of subjects on which the growing arm is ahead. 0.5 is a tie. This
is the figure that shows whether a mean advantage is broad or carried by one dataset --
the distinction between a method that works and an average that happens to be positive.
""")

code(r"""
recs = []
for grow, fixed in PAIRS.items():
    if not {grow, fixed} <= set(df.model):
        continue
    a, b = per_subject(df, grow), per_subject(df, fixed)
    m = a.merge(b, on=["eval","dataset","subject"], suffixes=("_g","_f"))
    m["win"] = m.score_g > m.score_f
    for (ev, ds), g in m.groupby(["eval","dataset"]):
        recs.append(dict(pair=grow, eval=ev, dataset=ds, frac=g.win.mean(), n=len(g)))
wins = pd.DataFrame(recs)
if wins.empty:
    print("no pair available")
else:
    evs = [e for e in ("within_session","cross_session","cross_subject")
           if e in set(wins["eval"])]
    fig, axes = plt.subplots(1, len(evs), figsize=(4.6*len(evs), 4.0), squeeze=False)
    for ax, ev in zip(axes[0], evs):
        piv = (wins[wins["eval"] == ev].pivot(index="dataset", columns="pair", values="frac"))
        im = ax.imshow(piv.values, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45,
                                                              ha="right", fontsize=7)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v-.5) > .28 else "black")
        ax.set_title(ev, fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=.7,
                 label="fraction of subjects where growing wins")
    fig.suptitle("Breadth of the paired advantage (0.5 = tie)", y=1.02)
    fig.savefig(FIGS/"13_win_matrix.png", bbox_inches="tight")
    plt.show()
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
