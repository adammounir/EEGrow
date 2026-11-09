"""E1 -- what growth does inside the models, on campaign A's archived weights
(analysis/EXPLO_internal_E1.md). EXPLORATORY. Committed with its note BEFORE the bulk read.
Read-only on the archive; no forward pass, no training.

Usage::

    python explo_internal_E1.py selftest
    python explo_internal_E1.py extract --model conformer --cells A.txt S.txt --out out
    python explo_internal_E1.py read --out out
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_BOOT = 10_000
MODELS = ("shallow", "eegnex", "conformer", "ctnet", "msvtnet")
MULTI = ("conformer", "ctnet", "msvtnet")
ARMS = {"shallow": ("bd", "tiny", "random", "fix", "tiny_bdinit")}
PROTS = ("within_session", "cross_session", "cross_subject")
N_BLOCKS = {"conformer": 6, "ctnet": 6, "msvtnet": 2}


def arms_of(model):
    return ARMS.get(model, ("bd", "tiny", "random"))


# ---------------------------------------------------------------- junction tensors
def junctions(model, sd):
    """[(W1, W2)] per junction, W1 with the hidden units on dim 0, W2 on dim 1. Names read
    from the archive on 09/11 (one fit per arm); a missing name raises."""
    g = any(k.startswith("net.") for k in sd)   # growing wrappers keep the bd net under .net
    if model == "conformer":
        f = ("net.transformer.{}.1.fn.1.first.layer.weight", "net.transformer.{}.1.fn.1.second.layer.weight") \
            if g else ("transformer.{}.1.fn.1.0.weight", "transformer.{}.1.fn.1.3.weight")
        return [(sd[f[0].format(i)], sd[f[1].format(i)]) for i in range(6)]
    if model == "ctnet":
        f = ("net.trans.layers.{}.feed_forward.module.0.first.layer.weight",
             "net.trans.layers.{}.feed_forward.module.0.second.layer.weight") if g else \
            ("trans.layers.{}.feed_forward.module.0.0.weight", "trans.layers.{}.feed_forward.module.0.3.weight")
        return [(sd[f[0].format(i)], sd[f[1].format(i)]) for i in range(6)]
    if model == "msvtnet":
        f = ("net.transformer.trans.layers.{}.linear1.first.layer.weight",
             "net.transformer.trans.layers.{}.linear1.second.layer.weight") if g else \
            ("transformer.trans.layers.{}.linear1.weight", "transformer.trans.layers.{}.linear2.weight")
        return [(sd[f[0].format(i)], sd[f[1].format(i)]) for i in range(2)]
    if model == "eegnex":
        f = ("net.block_4.0.first.layer.weight", "net.block_4.0.second.layer.weight") if g else \
            ("block_4.0.weight", "block_5.0.weight")
        return [(sd[f[0]], sd[f[1]])]
    if model == "shallow":
        f = ("conv_time.layer.weight", "conv_spat.layer.weight") if "conv_time.layer.weight" in sd else \
            ("conv_time_spat.conv_time.weight", "conv_time_spat.conv_spat.weight")
        return [(sd[f[0]], sd[f[1]])]
    raise ValueError(model)


def _np(t):
    return t.detach().double().cpu().numpy()


def erank(A):
    """Roy & Vetterli effective rank: exp of the entropy of the normalised singular values."""
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 0]
    if not len(s):
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def srank(A):
    s = np.linalg.svd(A, compute_uv=False)
    return float((s ** 2).sum() / s[0] ** 2) if s[0] > 0 else 0.0


def junction_stats(W1, W2, linear, W1i=None, W2i=None):
    """Weights-only statistics of one junction (see the note, section 3)."""
    W1, W2 = _np(W1), _np(W2)
    h = W1.shape[0]
    assert W2.shape[1] == h, (W1.shape, W2.shape)
    A1 = W1.reshape(h, -1)
    A2 = np.moveaxis(W2, 1, 0).reshape(h, -1)       # one row per hidden unit
    s = np.linalg.norm(A1, axis=1) * np.linalg.norm(A2, axis=1)
    out = {"h": h, "n_eff": float(s.sum() ** 2 / (s ** 2).sum()) if (s ** 2).sum() > 0 else 0.0,
           "erank_W1": erank(A1), "erank_W2": erank(A2),
           "norm_W1": float(np.linalg.norm(A1)), "norm_W2": float(np.linalg.norm(A2))}
    out["n_eff_frac"] = out["n_eff"] / h
    if linear:
        M = W2 @ W1
        sv = np.linalg.svd(M, compute_uv=False)
        out.update(srank_M=srank(M), erank_M=erank(M), norm2_M=float(sv[0]))
    if W1i is not None and tuple(W1i.shape) == tuple(W1.shape) and tuple(W2i.shape) == tuple(W2.shape):
        W1i, W2i = _np(W1i), _np(W2i)
        out["drift_W1"] = float(np.linalg.norm(W1 - W1i) / np.linalg.norm(W1i))
        out["drift_W2"] = float(np.linalg.norm(W2 - W2i) / np.linalg.norm(W2i))
    return out


# ---------------------------------------------------------------- outputs
def output_stats(z):
    """nll, ece (15 bins, top-1), acc of one logits file, float32 softmax like the scorer."""
    import torch
    lg, y = z["logits"], z["y"]
    classes = np.unique(y)
    if lg.shape[1] != len(classes):
        # the scorer (campaignA_check.score_of) maps np.unique(y) onto the logit columns;
        # with a class missing from the test set that mapping is not defined: refuse
        return None, None
    idx = np.array([{c: i for i, c in enumerate(classes)}[c] for c in y])
    p = torch.softmax(torch.as_tensor(lg, dtype=torch.float32), dim=1).double().numpy()
    conf, pred = p.max(1), p.argmax(1)
    nll = float(-np.log(np.clip(p[np.arange(len(idx)), idx], 1e-12, None)).mean())
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs((pred[bins == b] == idx[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(15) if (bins == b).any())
    return {"nll": nll, "ece": float(ece), "acc": float((pred == idx).mean())}, pred


# ---------------------------------------------------------------- extraction
def cells_of(paths, model):
    from campaignA_check import parse_cells
    return [c for p in paths for c in parse_cells(p) if c["model"] == model]


def expected_fits(c, rec_sessions):
    return {"within_session": 5 * rec_sessions, "cross_session": rec_sessions, "cross_subject": 1}[c["prot"]]


def extract(model, cell_paths, out):
    import pandas as pd
    import torch
    os.makedirs(out, exist_ok=True)
    cells = cells_of(cell_paths, model)
    counts = collections.Counter()
    rows = []
    for c in cells:
        stem = c["csv"][:-4]
        try:
            n_sess = len(pd.read_csv(c["csv"])["session"].unique())
            recs = [json.loads(l) for l in open(stem + "__fits.jsonl")]
        except Exception as e:  # unreadable cell: counted, never silently skipped
            counts[f"cell_unreadable:{type(e).__name__}"] += 1
            continue
        if len(recs) != expected_fits(c, n_sess):
            counts["cell_bad_fit_count"] += 1
            continue
        counts["cells"] += 1
        for r in recs:
            uid = r["fit_uid"]
            fin, ini = f"{stem}__fits__final/{uid}.pt", f"{stem}__fits__init/{uid}.pt"
            if not (os.path.exists(fin) and os.path.exists(ini)):
                counts["fit_missing_weights"] += 1
                continue
            row = {"model": model, "arm": c["arm"], "prot": c["prot"], "ds": c["ds"], "subj": c["subj"],
                   "cv_ind": r.get("cv_ind"), "session": str(r.get("session")),
                   "test_sessions": sorted(r.get("test_sessions") or []),
                   "params_end": r.get("params_end")}
            p = torch.load(fin, map_location="cpu", weights_only=False)
            p0 = torch.load(ini, map_location="cpu", weights_only=False)
            sd = p["state_dict"]
            if not all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()):
                counts["fit_nonfinite_weights"] += 1
                row["diverged"] = True
            else:
                J, J0 = junctions(model, sd), junctions(model, p0["state_dict"])
                row["blocks"] = [junction_stats(W1, W2, model in MULTI, W1i, W2i)
                                 for (W1, W2), (W1i, W2i) in zip(J, J0)]
            # allocation, from the growth rows of the history (multi-junction models)
            g = [h for h in r["history"] if "grow_junction_foi" in h]
            if model in MULTI and c["arm"] != "bd":
                nb = N_BLOCKS[model]
                jw = p.get("junction_widths")
                tg = r.get("target_width")
                if jw is None or len(jw) != nb or not tg:
                    counts["fit_no_junction_widths"] += 1
                else:
                    row["width_frac"] = [w / (tg / nb) for w in jw]
                if g:
                    # share of the raw TINY gain among the junctions still competing (a
                    # junction at its target leaves `grow_junction_ids`): NaN when absent
                    sh = []
                    for h in g:
                        x = np.asarray(h["grow_junction_foi"], float)
                        if x.sum() <= 0:
                            counts["growth_rows_zero_gain"] += 1
                            continue
                        v = np.full(nb, np.nan)
                        v[h["grow_junction_ids"]] = x / x.sum()
                        sh.append(v)
                    if sh:
                        with np.errstate(all="ignore"):
                            m_ = np.nanmean(np.vstack(sh), 0)
                        row["foi_share"] = [None if np.isnan(v) else float(v) for v in m_]
                    # grow_junction is the junction id (loop._grow_step_best, `chosen["junction"]`)
                    row["pick_frac"] = [float(np.mean([h["grow_junction"] == j for h in g])) for j in range(nb)]
            # outputs
            lg = sorted(glob.glob(f"{stem}__fits__logits/{uid}*.npz"))
            row["out"] = []
            for f in lg:
                z = np.load(f)
                if not np.isfinite(z["logits"]).all():
                    counts["logits_nonfinite"] += 1
                    continue
                st, pred = output_stats(z)
                if st is None:
                    counts["logits_class_mismatch"] += 1
                    continue
                o = np.argsort(z["test_idx"])
                st.update(session=str(z["session"][0]), test_idx=z["test_idx"][o].tolist(),
                          pred=pred[o].tolist())
                row["out"].append(st)
            counts["fits"] += 1
            rows.append(row)
    with open(f"{out}/fits_{model}.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    json.dump(dict(counts), open(f"{out}/counts_{model}.json", "w"), indent=1)
    print(model, dict(counts))


# ---------------------------------------------------------------- reading
def sboot(groups):
    """adapt_B.sboot, copied (no torch/braindecode import): sqrt(n_d)-weighted mean of
    dataset means, bootstrap resamples subjects within each dataset."""
    groups = {k: np.asarray(v, float) for k, v in groups.items() if len(v)}
    if not groups:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_ds": 0}
    w = {k: np.sqrt(len(v)) for k, v in groups.items()}
    tot = sum(w.values())
    rng = np.random.default_rng(0)
    est, b = 0.0, np.zeros(N_BOOT)
    for k in sorted(groups):
        v = groups[k]
        est += w[k] / tot * v.mean()
        b += w[k] / tot * v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return {"mean": float(est), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "n": int(sum(len(v) for v in groups.values())), "n_ds": len(groups)}


def fmt(c, d=3):
    if not c["n"]:
        return "—"
    return f"{c['mean']:+.{d}f} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]"


def fkey(r):
    return (r["prot"], r["ds"], r["subj"], r["cv_ind"], r["session"], tuple(r["test_sessions"]))


def per_subject(rows, fn):
    """{ds: [subject means of fn(row)]}, fn returning None skips the fit."""
    acc = collections.defaultdict(list)
    for r in rows:
        v = fn(r)
        if v is not None and np.isfinite(v):
            acc[(r["ds"], r["subj"])].append(v)
    g = collections.defaultdict(list)
    for (ds, _), v in sorted(acc.items()):
        g[ds].append(float(np.mean(v)))
    return g


def paired(ra, rb, fn):
    """{ds: [subject means of fn(a) - fn(b)]} over fits present in both arms."""
    B = {fkey(r): r for r in rb}
    acc = collections.defaultdict(list)
    for r in ra:
        o = B.get(fkey(r))
        if o is None:
            continue
        x, y = fn(r), fn(o)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            acc[(r["ds"], r["subj"])].append(x - y)
    g = collections.defaultdict(list)
    for (ds, _), v in sorted(acc.items()):
        g[ds].append(float(np.mean(v)))
    return g


def agreement(ra, rb, counts, tag):
    B = {fkey(r): r for r in rb}
    acc = collections.defaultdict(list)
    for r in ra:
        o = B.get(fkey(r))
        if o is None:
            continue
        ob = {x["session"]: x for x in o["out"]}
        for x in r["out"]:
            y = ob.get(x["session"])
            if y is None or y["test_idx"] != x["test_idx"]:
                counts[f"agree_refused:{tag}"] += 1
                continue
            acc[(r["ds"], r["subj"])].append(float(np.mean(np.array(x["pred"]) == np.array(y["pred"]))))
    g = collections.defaultdict(list)
    for (ds, _), v in sorted(acc.items()):
        g[ds].append(float(np.mean(v)))
    return g


def blk(r, j, k):
    b = r.get("blocks")
    return None if b is None else b[j].get(k)


def out_mean(r, k):
    return float(np.mean([x[k] for x in r["out"]])) if r["out"] else None


def read(out):
    R = collections.defaultdict(list)
    counts = collections.Counter()
    for m in MODELS:
        f = f"{out}/fits_{m}.jsonl"
        if not os.path.exists(f):
            raise SystemExit(f"REFUSED: {f} missing")
        for l in open(f):
            r = json.loads(l)
            R[(m, r["arm"], r["prot"])].append(r)
        for k, v in json.load(open(f"{out}/counts_{m}.json")).items():
            counts[f"{m}:{k}"] += v
    res, L = {}, ["# E1 — ce que la croissance fait à l'intérieur des modèles (EXPLORATOIRE)", "",
                  "Note : `analysis/EXPLO_internal_E1.md`. Moyenne par sujet, groupée √n_d, IC95 bootstrap "
                  "stratifié 10 000. Poids seuls (aucune passe avant).", ""]

    # R1 -- allocation
    L += ["## R1 — Où la croissance met la capacité (FFN par bloc)", ""]
    for m in MULTI:
        nb = N_BLOCKS[m]
        L += [f"### {m}", "", "| bras | protocole | quantité | " + " | ".join(f"b{j}" for j in range(nb)) + " |",
              "|---|---|---|" + "---|" * nb]
        for arm in ("tiny", "random"):
            for prot in PROTS:
                rows = R[(m, arm, prot)]
                for q in ("width_frac", "foi_share", "pick_frac"):
                    cs = [sboot(per_subject(rows, lambda r, j=j: r[q][j] if q in r else None)) for j in range(nb)]
                    res[f"R1 {m} {arm} {prot} {q}"] = cs
                    L.append(f"| {arm} | {prot} | {q} | " + " | ".join(f"{c['mean']:.2f}" if c["n"] else "—" for c in cs) + " |")
                tot = sboot(per_subject(rows, lambda r: float(np.mean(r["width_frac"])) if "width_frac" in r else None))
                res[f"R1 {m} {arm} {prot} total_width_frac"] = tot
                L.append(f"| {arm} | {prot} | largeur FFN totale / bd | {fmt(tot, 2)} (n={tot['n']}) |" + " |" * (nb - 1))
        L += ["", f"Profil par dataset (tiny, within_session, width_frac) :", "",
              "| dataset | n | " + " | ".join(f"b{j}" for j in range(nb)) + " |", "|---|---|" + "---|" * nb]
        rows = R[(m, "tiny", "within_session")]
        for ds in sorted({r["ds"] for r in rows}):
            rr = [r for r in rows if r["ds"] == ds and "width_frac" in r]
            if rr:
                L.append(f"| {ds} | {len({r['subj'] for r in rr})} | " +
                         " | ".join(f"{np.mean([r['width_frac'][j] for r in rr]):.2f}" for j in range(nb)) + " |")
        L.append("")

    # R2 -- bd usage per block and correlation with the tiny allocation
    from scipy.stats import spearmanr
    L += ["## R2 — Le modèle complet (bd) utilise-t-il les blocs que la croissance élargit ?", ""]
    for m in MULTI:
        nb = N_BLOCKS[m]
        L += [f"### {m}", "", "| protocole | usage bd | " + " | ".join(f"b{j}" for j in range(nb)) + " |",
              "|---|---|" + "---|" * nb]
        for prot in PROTS:
            for k in ("n_eff_frac", "drift_W1", "drift_W2", "norm2_M", "srank_M"):
                cs = [sboot(per_subject(R[(m, "bd", prot)], lambda r, j=j, k=k: blk(r, j, k))) for j in range(nb)]
                res[f"R2 {m} bd {prot} {k}"] = cs
                L.append(f"| {prot} | {k} | " + " | ".join(f"{c['mean']:.3f}" if c["n"] else "—" for c in cs) + " |")
        L.append("")
        if nb < 3:
            L += ["(2 blocs : pas de corrélation entre blocs.)", ""]
            continue
        L += ["| protocole | allocation tiny | usage bd | Spearman moyen [IC95] | n |", "|---|---|---|---|---|"]
        for prot in PROTS:
            B = {fkey(r): r for r in R[(m, "bd", prot)]}
            for a in ("foi_share", "width_frac"):
                for u in ("n_eff_frac", "drift_W2", "norm2_M"):
                    acc = collections.defaultdict(list)
                    for r in R[(m, "tiny", prot)]:
                        o = B.get(fkey(r))
                        if o is None or a not in r or o.get("blocks") is None:
                            continue
                        x, y = r[a], [b[u] for b in o["blocks"]]
                        if any(v is None for v in x):
                            counts[f"R2_block_never_competing:{m}:{a}"] += 1
                            continue
                        if np.ptp(x) == 0 or np.ptp(y) == 0:
                            counts[f"R2_constant:{m}:{a}"] += 1
                            continue
                        acc[(r["ds"], r["subj"])].append(float(spearmanr(x, y)[0]))
                    g = collections.defaultdict(list)
                    for (ds, _), v in sorted(acc.items()):
                        g[ds].append(float(np.mean(v)))
                    c = sboot(g)
                    res[f"R2 {m} {prot} rho {a} {u}"] = c
                    L.append(f"| {prot} | {a} | {u} | {fmt(c, 2)} | {c['n']} |")
        L.append("")

    # R3 -- same accuracy, same internal solution?
    L += ["## R3 — Même accuracy, même solution interne ?", "",
          "| modèle | protocole | quantité | bd | tiny − bd | random − bd |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for prot in PROTS:
            for k, fn in (("Σ n_eff", lambda r: sum(b["n_eff"] for b in r["blocks"]) if r.get("blocks") else None),
                          ("Σ h", lambda r: sum(b["h"] for b in r["blocks"]) if r.get("blocks") else None),
                          ("erank_M moyen", lambda r: float(np.mean([b["erank_M"] for b in r["blocks"]]))
                           if r.get("blocks") and "erank_M" in r["blocks"][0] else None),
                          ("erank_W1 moyen", lambda r: float(np.mean([b["erank_W1"] for b in r["blocks"]]))
                           if r.get("blocks") else None)):
                base = sboot(per_subject(R[(m, "bd", prot)], fn))
                ct = sboot(paired(R[(m, "tiny", prot)], R[(m, "bd", prot)], fn))
                cr = sboot(paired(R[(m, "random", prot)], R[(m, "bd", prot)], fn))
                res[f"R3 {m} {prot} {k}"] = {"bd": base, "tiny-bd": ct, "random-bd": cr}
                if base["n"]:
                    L.append(f"| {m} | {prot} | {k} | {base['mean']:.2f} | {fmt(ct, 2)} | {fmt(cr, 2)} |")
    L.append("")

    # R4 -- outputs
    L += ["## R4 — Sorties : calibration et accord", "",
          "| modèle | protocole | nll tiny − bd | ece tiny − bd | nll random − bd | ece random − bd | "
          "accord tiny–bd | accord random–bd | accord tiny–random |", "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for prot in PROTS:
            t, b, rd = R[(m, "tiny", prot)], R[(m, "bd", prot)], R[(m, "random", prot)]
            cs = [sboot(paired(x, b, lambda r, k=k: out_mean(r, k))) for x in (t,) for k in ("nll", "ece")] + \
                 [sboot(paired(x, b, lambda r, k=k: out_mean(r, k))) for x in (rd,) for k in ("nll", "ece")] + \
                 [sboot(agreement(t, b, counts, f"{m}:tiny-bd")), sboot(agreement(rd, b, counts, f"{m}:random-bd")),
                  sboot(agreement(t, rd, counts, f"{m}:tiny-random"))]
            res[f"R4 {m} {prot}"] = cs
            L.append(f"| {m} | {prot} | " + " | ".join(fmt(c) for c in cs[:4]) + " | " +
                     " | ".join(f"{c['mean']:.3f}" if c["n"] else "—" for c in cs[4:]) + " |")
    L.append("")

    # R5 -- Shallow
    L += ["## R5 — Shallow : géométrie de conv_time → conv_spat", "",
          "| protocole | quantité | bd | tiny − bd | random − bd | fix − bd | tiny_bdinit − bd | tiny − tiny_bdinit |",
          "|---|---|---|---|---|---|---|---|"]
    for prot in PROTS:
        for k in ("erank_W1", "erank_W2", "n_eff", "norm_W1", "norm_W2"):
            fn = (lambda r, k=k: blk(r, 0, k))
            base = sboot(per_subject(R[("shallow", "bd", prot)], fn))
            cs = [sboot(paired(R[("shallow", a, prot)], R[("shallow", "bd", prot)], fn))
                  for a in ("tiny", "random", "fix", "tiny_bdinit")]
            cs.append(sboot(paired(R[("shallow", "tiny", prot)], R[("shallow", "tiny_bdinit", prot)], fn)))
            res[f"R5 {prot} {k}"] = {"bd": base, "contrasts": cs}
            L.append(f"| {prot} | {k} | {base['mean']:.2f} | " + " | ".join(fmt(c, 2) for c in cs) + " |")
    for a, b_ in (("fix", "bd"), ("tiny_bdinit", "bd")):
        for prot in PROTS:
            g = agreement(R[("shallow", a, prot)], R[("shallow", b_, prot)], counts, f"shallow:{a}-{b_}")
            c = sboot(g)
            res[f"R5 agree {a}-{b_} {prot}"] = c
            L.append(f"| {prot} | accord {a}–{b_} | {c['mean']:.3f} | | | | | |" if c["n"] else
                     f"| {prot} | accord {a}–{b_} | — | | | | | |")
    L += ["", "## Comptes", "", "```", json.dumps(dict(sorted(counts.items())), indent=1), "```"]
    open(f"{out}/E1.md", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{out}/E1.json", "w"), indent=1)
    print("\n".join(L))


# ---------------------------------------------------------------- self-test
def selftest():
    import torch
    rng = np.random.default_rng(0)
    # n_eff: h equal neurons -> h ; one live neuron -> 1
    W1, W2 = torch.ones(8, 4), torch.ones(4, 8)
    assert abs(junction_stats(W1, W2, True)["n_eff"] - 8) < 1e-9
    W1z = torch.zeros(8, 4); W1z[3] = 1.0
    assert abs(junction_stats(W1z, W2, True)["n_eff"] - 1) < 1e-9
    # erank of identity = n, of rank-1 = 1 ; srank of identity = n
    assert abs(erank(np.eye(5)) - 5) < 1e-9 and abs(erank(np.outer([1, 2, 3], [1, 1])) - 1) < 1e-9
    assert abs(srank(np.eye(5)) - 5) < 1e-9
    # conv layout: hidden on dim 0 of W1 and dim 1 of W2
    W1c, W2c = torch.randn(32, 64, 1, 16), torch.randn(8, 32, 1, 16)
    st = junction_stats(W1c, W2c, False)
    assert st["h"] == 32 and "erank_M" not in st and 1 <= st["n_eff"] <= 32
    # drift only when shapes match ; zero drift on identical weights
    A, B = torch.randn(16, 40), torch.randn(40, 16)
    assert junction_stats(A, B, True, A, B)["drift_W1"] == 0.0
    assert "drift_W1" not in junction_stats(A, B, True, torch.randn(8, 40), torch.randn(40, 8))
    # outputs: perfect confident predictor -> acc 1, ece ~ 0 ; chance logits -> ece = |acc - 1/k|
    y = np.repeat(np.arange(4), 25)
    lg = np.eye(4)[y] * 50
    st, _ = output_stats({"logits": lg, "y": y})
    assert st["acc"] == 1.0 and st["ece"] < 1e-6 and st["nll"] < 1e-6
    st, _ = output_stats({"logits": np.zeros((100, 4)), "y": y})
    assert abs(st["nll"] - math.log(4)) < 1e-6
    assert output_stats({"logits": np.zeros((50, 4)), "y": y[:50]})[0] is None   # 2 classes vs 4 columns
    # names: every model's junction keys resolve on both layouts (shapes only)
    sd_c = {f"transformer.{i}.1.fn.1.0.weight": torch.zeros(160, 40) for i in range(6)}
    sd_c.update({f"transformer.{i}.1.fn.1.3.weight": torch.zeros(40, 160) for i in range(6)})
    assert len(junctions("conformer", sd_c)) == 6
    # sboot: constant groups -> CI collapses on the constant
    c = sboot({"a": [1.0] * 5, "b": [1.0] * 9})
    assert c["mean"] == 1.0 and c["lo"] == 1.0 and c["hi"] == 1.0 and c["n"] == 14
    _ = rng
    print("selftest OK (8 checks)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    e = sp.add_parser("extract")
    e.add_argument("--model", required=True, choices=MODELS)
    e.add_argument("--cells", nargs="+", required=True)
    e.add_argument("--out", required=True)
    r = sp.add_parser("read")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "extract":
        extract(a.model, a.cells, a.out)
    else:
        read(a.out)


if __name__ == "__main__":
    main()
