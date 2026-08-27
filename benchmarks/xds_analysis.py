"""Read the cross-dataset grid and answer the four questions it was built for.

Everything is paired on the subject. A subject's decodability varies enormously (0.50 to
0.95 on motor imagery), and that variance is shared by every arm because every arm is
scored on the same held-out subjects. Pairing removes it; an unpaired test on 9 subjects
would drown a 2-point effect in between-subject spread.

The four contrasts, and what each one can and cannot conclude:

1. **pooled - within** -- the headline. Positive means training on 220 extra subjects
   from 5 other datasets beats training on the target's own 8, i.e. the harmonisation
   bought something. This is the only contrast that answers "better than the first
   benchmark", and it is a fair comparison because both arms are scored on the same
   subjects under the same 2-class 22-channel protocol.
2. **lodo - within** -- zero-shot transfer. Expected negative on motor imagery; a
   negative value here with a positive contrast 1 is the normal picture (the pool helps
   as extra data, not as a replacement).
3. **core+interp - core** -- what interpolation buys, at constant rank. Interpolation
   fidelity is separately validated, so a null here reads as "Shin2017A's 29 subjects
   carry nothing transferable", not "the spline destroyed the signal".
4. **euclidean - scale** and **scale - none** -- whitening versus mere rescaling. Their
   sum is the total alignment effect; reporting only the sum would attribute to Euclidean
   alignment a gain that per-subject rescaling already provides.

Plus the growth interaction, which is the eegrow question asked where extra data exists:
does a growable net convert a bigger, more heterogeneous pool into accuracy better than a
fixed one? ``(grow_pooled - grow_within) - (bd_pooled - bd_within)``.

    python benchmarks/xds_analysis.py --target bnci2014_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import default_results_root  # noqa: E402

METRICS = ("accuracy", "auc")


def load(target: str) -> pd.DataFrame:
    d = default_results_root().parent / "results_cross_dataset" / target
    files = sorted(d.glob("*.csv"))
    if not files:
        raise SystemExit(f"no results under {d}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # A run is one (arm, align, tier, model, seed); a row is one test subject inside it.
    # Averaging over seeds first makes the paired unit the subject rather than the
    # (subject, seed) pair, which would otherwise count the same subject three times and
    # shrink the confidence interval by sqrt(3) for free.
    keys = ["target", "arm", "align", "pool_tier", "model", "subject"]
    return df.groupby(keys, as_index=False)[list(METRICS)].mean()


def paired(a: pd.Series, b: pd.Series, metric: str) -> dict:
    """Paired contrast a - b over subjects, with a t interval and a sign test.

    Both are reported because they fail differently: the t-test assumes the differences
    are roughly normal, which 9 subjects cannot establish, while the sign test needs no
    such assumption but throws away magnitude. Agreement between them is what makes a
    claim on this sample size credible.
    """
    from scipy import stats

    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).dropna()
    n = len(d)
    if n < 2:
        return {"n": n}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    _, p_t = stats.ttest_1samp(d, 0.0)
    wins = int((d > 0).sum())
    p_sign = float(stats.binomtest(wins, n, 0.5).pvalue)
    return {"metric": metric, "n": n, "delta": mean,
            "lo": mean - tcrit * se, "hi": mean + tcrit * se,
            "p_t": float(p_t), "wins": wins, "p_sign": p_sign}


def _series(df, metric, **sel):
    q = df
    for k, v in sel.items():
        q = q[q[k] == v]
    return q.set_index("subject")[metric]


def contrast_table(df, metric, spec) -> pd.DataFrame:
    rows = []
    for label, left, right in spec:
        a, b = _series(df, metric, **left), _series(df, metric, **right)
        if a.empty or b.empty:
            continue
        r = paired(a, b, metric)
        r["contrast"] = label
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out[["contrast", "n", "delta", "lo", "hi", "p_t", "wins", "p_sign"]]


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values.

    This grid asks a dozen questions of one dataset. Reporting a bare p = 0.03 out of
    twelve tests would be the standard way to publish noise, and Holm controls the
    family-wise rate without assuming the tests are independent (they are not -- they
    share arms).
    """
    order = np.argsort(p.to_numpy())
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p.to_numpy()[i])
        adj[i] = min(1.0, running)
    return pd.Series(adj, index=p.index)


def show(title: str, tbl: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if tbl.empty:
        print("(aucune cellule disponible)")
        return
    t = tbl.copy()
    t["IC95"] = t.apply(lambda r: f"[{r.lo:+.4f}, {r.hi:+.4f}]", axis=1)
    t["signe"] = t.apply(lambda r: f"{r.wins}/{r.n}", axis=1)
    cols = ["contrast", "n", "delta", "IC95", "p_t", "signe", "p_sign"]
    if "p_holm" in t:
        cols.append("p_holm")
    print(t[cols].to_string(index=False,
                            float_format=lambda v: f"{v:.4f}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="bnci2014_001")
    ap.add_argument("--metric", default="accuracy", choices=METRICS)
    ap.add_argument("--align", default="euclidean",
                    help="alignment level at which the arm contrasts are read")
    ap.add_argument("--tier", default="core")
    a = ap.parse_args(argv)

    df = load(a.target)
    print(f"cible {a.target} — {len(df)} cellules (sujet x arm x align x tier x modele)")
    have = df.groupby(["arm", "align", "pool_tier", "model"]).size()
    missing = [k for k in have.index if have[k] == 0]
    print(f"{len(have)} combinaisons presentes"
          + (f", manquantes: {missing}" if missing else ""))

    models = sorted(df.model.unique())
    m = a.metric

    # 1 & 2 -- the arms, at one alignment level and one tier
    spec = []
    for mod in models:
        base = dict(align=a.align, pool_tier=a.tier, model=mod)
        spec.append((f"pooled - within [{mod}]",
                     {**base, "arm": "pooled"},
                     {**base, "arm": "within", "pool_tier": "core"}))
        spec.append((f"lodo - within [{mod}]",
                     {**base, "arm": "lodo"},
                     {**base, "arm": "within", "pool_tier": "core"}))
    arms = contrast_table(df, m, spec)

    # 3 -- interpolation, at constant rank
    spec = [(f"core+interp - core [{arm}, {mod}]",
             dict(arm=arm, align=a.align, pool_tier="core+interp", model=mod),
             dict(arm=arm, align=a.align, pool_tier="core", model=mod))
            for arm in ("lodo", "pooled") for mod in models]
    interp = contrast_table(df, m, spec)

    # 4 -- whitening vs rescaling, kept apart
    spec = []
    for arm in ("within", "lodo", "pooled"):
        tier = "core"
        for mod in models:
            spec.append((f"euclidean - scale [{arm}, {mod}]",
                         dict(arm=arm, align="euclidean", pool_tier=tier, model=mod),
                         dict(arm=arm, align="scale", pool_tier=tier, model=mod)))
            spec.append((f"scale - none [{arm}, {mod}]",
                         dict(arm=arm, align="scale", pool_tier=tier, model=mod),
                         dict(arm=arm, align="none", pool_tier=tier, model=mod)))
    align_tbl = contrast_table(df, m, spec)

    everything = pd.concat([t for t in (arms, interp, align_tbl) if not t.empty],
                           ignore_index=True)
    if not everything.empty:
        everything["p_holm"] = holm(everything["p_t"])
        adj = dict(zip(everything.contrast, everything.p_holm))
        for t in (arms, interp, align_tbl):
            if not t.empty:
                t["p_holm"] = t.contrast.map(adj)

    show(f"1-2. les bras ({m}, align={a.align}, pool={a.tier})", arms)
    show(f"3. ce que l'interpolation apporte ({m}, align={a.align})", interp)
    show(f"4. blanchiment vs remise a l'echelle ({m}, pool=core)", align_tbl)

    # growth interaction: a difference of differences, so it needs the two model arms of
    # the same contrast rather than a contrast of its own
    if {"grow_shallow", "bd_shallow"} <= set(models):
        print(f"\n=== 5. interaction croissance x pool ({m}) ===")
        rows = []
        for arm in ("lodo", "pooled"):
            g = (_series(df, m, arm=arm, align=a.align, pool_tier=a.tier,
                         model="grow_shallow")
                 - _series(df, m, arm="within", align=a.align, pool_tier="core",
                           model="grow_shallow"))
            b = (_series(df, m, arm=arm, align=a.align, pool_tier=a.tier,
                         model="bd_shallow")
                 - _series(df, m, arm="within", align=a.align, pool_tier="core",
                           model="bd_shallow"))
            r = paired(g, b, m)
            r["contrast"] = f"({arm}-within) grow - ({arm}-within) fixe"
            rows.append(r)
        show("difference de differences", pd.DataFrame(rows))
        print("Positif => la croissance convertit un pool plus grand et plus "
              "heterogene en precision mieux qu'un reseau fixe.")

    print("\nLes deux tests sont donnes exprès : le t suppose des differences a peu "
          "pres normales, ce que 9 sujets ne peuvent pas etablir ; le test des signes "
          "ne suppose rien mais jette la magnitude. C'est leur accord qui rend une "
          "conclusion credible a cette taille d'echantillon.")
    print("p_holm corrige sur les "
          f"{0 if everything.empty else len(everything)} tests de la famille.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
