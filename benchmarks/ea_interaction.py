"""Does architecture growth benefit from Euclidean Alignment more than a fixed net?

The question is NOT whether EA helps -- Junqueira et al. settled that. It is whether
it helps the *growing* arm more, i.e. the interaction

    delta = (grow_EA - grow_raw) - (fixed_EA - fixed_raw)

Why that is the interesting quantity. gromo decides where and how much width to add
from the data's second-order statistics: the S tensor it accumulates is a variance. On
multi-subject EEG a large slice of that variance is subject nuisance -- skull, impedance,
cap placement -- so a growing net spends part of its budget modelling a nuisance a fixed
net cannot even try to model. Removing the nuisance should therefore pay the growing arm
more. That is a differential and falsifiable prediction, and it would *explain* the
currently negative fixed-vs-growing result rather than explain it away.

Design: a dose-response on subject count, because the hypothesis is about between-subject
nuisance and nothing else.

    bnci2014_001    9 subjects
    cho2017        52
    physionetmi   109

If the mechanism is real the interaction must GROW along that axis. Three ordered points
that move together are far stronger than one significant cell -- and far harder to get by
chance, since the sign of a trend cannot be fished for the way a single p-value can.

Pairing. One observation is a fully crossed quadruple on the same
(eval, dataset, subject, session, seed): grow_raw, grow_EA, fixed_raw, fixed_EA. Anything
less is not an interaction. Seeds are averaged inside each (subject, session) before
testing -- they are replicates of one measurement, not independent samples.

Usage:  python ea_interaction.py [results_root]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"

#: The two pairs bracket the effect: shallow is the only pair that does not lose on the
#: raw grid (+0.008), eegnex is the worst (-0.032). Measuring the interaction where
#: growth currently wins AND where it loses guards against reading a pair-specific quirk
#: as a general mechanism.
PAIRS = [("grow_shallow", "bd_shallow"), ("grow_eegnex", "bd_eegnex")]
MODELS = {m for p in PAIRS for m in p}
#: subject counts, for the dose-response ordering
N_SUBJ = {"bnci2014_001": 9, "cho2017": 52, "physionetmi": 109}
CELL = ["eval", "dataset", "subject", "session", "seed"]
AUC_DS = {"cho2017", "physionetmi"}
N_BOOT = 10000


def load(root: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty or "score" not in d.columns or "model" not in d.columns:
            continue
        if d["model"].iloc[0] not in MODELS:
            continue
        ev, ds, _ = f.split(os.sep)[-3:]
        if ds not in N_SUBJ:
            continue
        d["eval"], d["dataset"] = ev, ds
        # the raw grid predates the align axis, so its CSVs carry no such column
        d["align"] = d["align"].fillna("none") if "align" in d.columns else "none"
        rows.append(d)
    if not rows:
        raise SystemExit(f"aucun CSV exploitable sous {root!r}")
    df = pd.concat(rows, ignore_index=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    return df.drop_duplicates(subset=CELL + ["model", "align", "score"])


def quadruples(df: pd.DataFrame, grow: str, fixed: str) -> pd.DataFrame:
    """Fully crossed observations: the 4 arms present on the very same cell."""
    out = None
    for model, align, col in [(grow, "none", "g_raw"), (grow, "euclidean", "g_ea"),
                              (fixed, "none", "f_raw"), (fixed, "euclidean", "f_ea")]:
        s = df[(df["model"] == model) & (df["align"] == align)]
        s = s[CELL + ["score"]].rename(columns={"score": col})
        out = s if out is None else out.merge(s, on=CELL, how="inner")
    return out


def boot_ci(x: np.ndarray, rng: np.random.Generator, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean. n is small (9 subjects at the low end),
    so a normal-theory interval would be optimistic."""
    if len(x) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main() -> int:
    df = load(ROOT)
    have = df.groupby(["dataset", "align"]).size().unstack(fill_value=0)
    print("lignes chargees par (dataset, bras) :")
    print(have.to_string(), "\n")

    from regime_guard import assert_paired
    assert_paired(PAIRS, bench_root=os.path.dirname(os.path.abspath(ROOT)) or ".")

    rng = np.random.default_rng(0)
    recs = []
    for grow, fixed in PAIRS:
        Q = quadruples(df, grow, fixed)
        if Q.empty:
            print(f"[skip] {grow}: aucune cellule ou les 4 bras existent")
            continue
        for (ev, ds), sub in Q.groupby(["eval", "dataset"]):
            # one observation = one held-out subject/session, seeds averaged
            o = sub.groupby(["subject", "session"])[
                ["g_raw", "g_ea", "f_raw", "f_ea"]].mean()
            n_seeds = sub["seed"].nunique()
            d_grow = (o["g_ea"] - o["g_raw"]).to_numpy()
            d_fix = (o["f_ea"] - o["f_raw"]).to_numpy()
            inter = d_grow - d_fix
            p = np.nan
            if len(inter) >= 6 and not np.allclose(inter, 0):
                p = float(wilcoxon(d_grow, d_fix).pvalue)
            lo, hi = boot_ci(inter, rng)
            recs.append(dict(
                pair=grow, eval=ev, dataset=ds, n_subj=N_SUBJ[ds], n_obs=len(o),
                n_seeds=n_seeds, ea_grow=d_grow.mean(), ea_fixed=d_fix.mean(),
                interaction=inter.mean(), ci_lo=lo, ci_hi=hi, p=p,
                metric="AUC" if ds in AUC_DS else "acc"))
    if not recs:
        print("\nRien a analyser : le bras aligne n'a pas encore produit de cellule "
              "complete. Relancer apres ea_pilot_{light,heavy}.")
        return 1
    res = pd.DataFrame(recs).sort_values(["pair", "n_subj"])

    fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    print("\n" + "=" * 100)
    print("A) EFFETS SIMPLES — l'EA aide-t-elle chaque bras ? (controle de sanite)")
    print("=" * 100)
    print(res[["pair", "eval", "dataset", "n_subj", "n_obs", "ea_grow", "ea_fixed",
               "metric"]].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("B) INTERACTION — (grow_EA - grow_brut) - (fixe_EA - fixe_brut)")
    print("   > 0 : la croissance profite de l'alignement PLUS que le reseau fixe")
    print("=" * 100)
    print(res[["pair", "eval", "dataset", "n_subj", "n_obs", "interaction",
               "ci_lo", "ci_hi", "p"]].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("C) DOSE-REPONSE — l'interaction croit-elle avec le nombre de sujets ?")
    print("=" * 100)
    for pair, s in res.groupby("pair"):
        s = s.sort_values("n_subj")
        traj = "  ->  ".join(f"{r.dataset}({r.n_subj}) {r.interaction:+.4f}"
                             for r in s.itertuples())
        print(f"\n  {pair}\n    {traj}")
        if len(s) < 3:
            print(f"    [{len(s)}/3 points — dose-reponse incomplete, "
                  f"pas de conclusion de tendance]")
            continue
        v = s["interaction"].to_numpy()
        mono = bool(np.all(np.diff(v) > 0))
        print(f"    monotone croissante : {mono}   "
              f"amplitude {v[0]:+.4f} -> {v[-1]:+.4f}")
        # 3 ordered points: a rank correlation here has 6 possible orderings, so the
        # best one-sided p it can ever reach is 1/6 = 0.167. Reported as descriptive
        # only -- the evidence is the trajectory and its CIs, not a p-value.
        print("    (avec 3 points un test de tendance plafonne a p=0.167 : "
              "lire la trajectoire et les IC, pas une p-valeur)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
