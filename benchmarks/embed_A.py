"""Subject fingerprint of the neurons added by growth, on campaign A's weights
(analysis/PREREG_growth_embedding_A.md). Committed with the PREREG BEFORE any weight is read.

cross_session, subjects with exactly 2 sessions (their two fits train on disjoint
sessions), last-epoch weights (``__fits__final``). An item = (subject, training session).

  P1  top-1(added) - top-1(bd32), lee2019_mi
  P2  top-1(added) - top-1(added_random): same class, init, count and recipe, only the
      direction of the added neurons differs, lee2019_mi
"Beyond the covariance" is described, not tested (PREREG § 9: a ridge residual keeps
identifying a purely covariance-linear embedding, tests/test_embed_A.py).

Usage::

    python embed_A.py --cells-a campaignA_grid/campaignA_cells_planned_av5.txt \\
        --cells-s campaignS_grid/campaignA_cells_planned_av5.txt --out ../read/embed_A
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = ["lee2019_mi", "bnci2014_001", "bnci2015_001"]
PRIMARY = "lee2019_mi"
N_BOOT = 10_000
R_DRAWS = 20
SHALLOW = {"time": ("conv_time.layer.weight", "conv_time_spat.conv_time.weight"),
           "spat": ("conv_spat.layer.weight", "conv_time_spat.conv_spat.weight")}
EEGNEX_J2 = ("net.block_4.0.first.layer.weight", "block_4.0.weight")
# junction tensor and (start, end) width each arm must show (init file, final file);
# bd's files carry growable_width=None, so widths are read on the tensors themselves
JUNCTION = {"shallow": SHALLOW["time"], "eegnex": EEGNEX_J2}
WIDTHS = {("shallow", "bd"): (40, 40), ("shallow", "tiny"): (8, 40), ("shallow", "random"): (8, 40),
          ("shallow", "tiny_bdinit"): (8, 40), ("eegnex", "bd"): (32, 32), ("eegnex", "tiny"): (4, 32)}


# ---------------------------------------------------------------- embeddings
def key(sd, names):
    k = [n for n in names if n in sd]
    assert len(k) == 1, (names, [n for n in sd if "block_4" in n or "conv" in n][:8])
    return sd[k[0]]


def shallow_describe(sd, idx):
    """(spectrum, topography) of a neuron set: PREREG § 3 (= embed_offline.describe)."""
    t = key(sd, SHALLOW["time"])[idx, 0, :, 0].double().numpy()
    s = key(sd, SHALLOW["spat"])[:, idx, 0, :].double().numpy()
    spec = np.abs(np.fft.rfft(t, n=64, axis=1)) ** 2
    spec /= spec.sum(1, keepdims=True)
    topo = (s ** 2).sum(0)
    topo /= topo.sum(1, keepdims=True)
    return [spec.mean(0), topo.mean(0)]


def eegnex_describe(sd, idx):
    """Spectrum of the junction-2 kernels, averaged over their 64 input maps (§ 7.2)."""
    w = key(sd, EEGNEX_J2)[idx, :, 0, :].double().numpy()  # (k, 64, 16)
    spec = (np.abs(np.fft.rfft(w, n=64, axis=2)) ** 2).mean(1)
    spec /= spec.sum(1, keepdims=True)
    return [spec.mean(0)]


def zblocks(emb):
    """{item: [blocks]} -> {item: vector}, each block z-scored over the items."""
    ks = sorted(emb)
    parts = []
    for b in range(len(emb[ks[0]])):
        x = np.array([emb[k][b] for k in ks], float)
        parts.append((x - x.mean(0)) / (x.std(0) + 1e-12))
    X = np.concatenate(parts, 1)
    return {k: X[i] for i, k in enumerate(ks)}


# ---------------------------------------------------------------- identification
def pairs(items):
    """Subjects with exactly 2 items -> (subjects, first-item keys, second-item keys)."""
    by = collections.defaultdict(list)
    for s, se in items:
        by[s].append(se)
    subj = sorted(s for s, v in by.items() if len(v) == 2)
    return subj, [(s, sorted(by[s])[0]) for s in subj], [(s, sorted(by[s])[1]) for s in subj]


def corr_rows(A, B):
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)
    return A @ B.T / A.shape[1]


def identify(vec):
    """Cross-session NN identification on {item: vector} (correlation). {subj: 0/.5/1}, AUC."""
    subj, ka, kb = pairs(vec)
    n = len(subj)
    C = corr_rows(np.array([vec[k] for k in ka]), np.array([vec[k] for k in kb]))
    hit = 0.5 * (C.argmax(1) == np.arange(n)) + 0.5 * (C.argmax(0) == np.arange(n))
    same, diff = np.diag(C), C[~np.eye(n, dtype=bool)]
    auc = float((same[:, None] > diff[None, :]).mean() + 0.5 * (same[:, None] == diff[None, :]).mean())
    return dict(zip(subj, hit.tolist())), auc


def loo_signature(fit_emb, own_emb):
    """§ 7.1: signature of the left-out subject. fit_emb = {subj: vector} of the
    cross_subject fits (fit i = trained without i); own_emb = {subj: vector}, the
    subject's own cross_session embedding. Match signature_i = e_i - mean e against
    target_j = -(a_j - mean a). {subj: 0/1}."""
    subj = sorted(set(fit_emb) & set(own_emb))
    E = np.array([fit_emb[s] for s in subj])
    A = np.array([own_emb[s] for s in subj])
    C = corr_rows(E - E.mean(0), -(A - A.mean(0)))
    return {s: float(C[i].argmax() == i) for i, s in enumerate(subj)}


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "n": len(d),
            "mde": float(2.8016 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None}


def ci(c):
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


# ---------------------------------------------------------------- loading
def cells_of(paths):
    from campaignA_check import parse_cells
    return [c for p in paths for c in parse_cells(p)]


def load_fits(cells, prot, ds, model, arm, kind="final"):
    """{(subj, item): (state_dict, final width, init width)} for one case. item = training
    session in cross_session (single), the test subject in cross_subject."""
    import torch
    out = {}
    for c in cells:
        if (c["prot"], c["ds"], c["model"], c["arm"]) != (prot, ds, model, arm):
            continue
        stem = c["csv"][:-4]
        for f in sorted(glob.glob(f"{stem}__fits__{kind}/*.pt")):
            p = torch.load(f, map_location="cpu", weights_only=False)
            m = p["meta"]
            ini = torch.load(f.replace(f"__fits__{kind}/", "__fits__init/"), map_location="cpu",
                             weights_only=False)
            if prot == "cross_session":
                tr = m["train_sessions"]
                if len(tr) != 1:
                    continue  # > 2 sessions: overlapping training data (PREREG § 2.1)
                k = (str(m["subject"]), tr[0].split(":", 1)[1])
            else:
                k = (str(m["subject"]), "loo")
            assert k not in out, k
            w = key(p["state_dict"], JUNCTION[model]).shape[0]
            w0 = key(ini["state_dict"], JUNCTION[model]).shape[0]
            out[k] = (p["state_dict"], w, w0)
    return out


def covariances(ds, items):
    """{item: upper triangle of the log-Euclidean mean trace-normalised covariance} of the
    training session, from the campaign's own dataset/paradigm and read-only cache."""
    from omegaconf import OmegaConf

    from check_cache import build_campaign
    from utils import cache_config
    dataset, par, _ = build_campaign(ds)
    cc = cache_config(OmegaConf.create({"enabled": True, "path": "/scratch/amounir/moabb_cache_A",
                                        "readonly": True}))
    out = {}
    for subj in sorted({s for s, _ in items}, key=int):
        X, _, meta = par.get_data(dataset=dataset, subjects=[int(subj)], cache_config=cc)
        for s, se in [k for k in items if k[0] == subj]:
            sel = (meta["session"].astype(str) == se).to_numpy()
            assert sel.any(), (ds, subj, se, sorted(meta["session"].unique()))
            x = X[sel].astype(np.float64)
            x = x - x.mean(-1, keepdims=True)
            C = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
            C /= np.trace(C, axis1=1, axis2=2)[:, None, None]
            w, V = np.linalg.eigh(C)
            L = np.einsum("ncd,nd,ned->nce", V, np.log(np.maximum(w, 1e-12)), V).mean(0)
            out[(s, se)] = L[np.triu_indices(L.shape[0])]
    return out


# ---------------------------------------------------------------- main
def shallow_sets(fits, arm, idx, model="shallow", describe=shallow_describe):
    """Embeddings of the neuron set ``idx``; a fit whose (init, final) junction widths
    are not the arm's (WIDTHS) is excluded and counted (PREREG § 2.2)."""
    emb, excl = {}, 0
    for k, (sd, w, w0) in fits.items():
        if (w0, w) != WIDTHS[(model, arm)]:
            excl += 1
            continue
        emb[k] = describe(sd, idx)
    return emb, excl


def run_dataset(cells, ds, L, out):
    sets = {}
    fits = {arm: load_fits(cells, "cross_session", ds, "shallow", arm)
            for arm in ("tiny", "bd", "random", "tiny_bdinit")}
    counts = {arm: len(f) for arm, f in fits.items()}
    sets["added"], ex1 = shallow_sets(fits["tiny"], "tiny", list(range(8, 40)))
    sets["init8"], _ = shallow_sets(fits["tiny"], "tiny", list(range(8)))
    sets["added_random"], ex2 = shallow_sets(fits["random"], "random", list(range(8, 40)))
    sets["added_bdinit"], ex3 = shallow_sets(fits["tiny_bdinit"], "tiny_bdinit", list(range(8, 40)))
    sets["bd40"], ex4 = shallow_sets(fits["bd"], "bd", list(range(40)))
    excluded = {"tiny": ex1, "random": ex2, "tiny_bdinit": ex3, "bd": ex4}
    items = sorted(set.intersection(*[set(v) for v in sets.values()]))
    subj, _, _ = pairs({k: 0 for k in items})
    items = [k for k in items if k[0] in subj]
    cov = covariances(ds, items)
    vec = {n: zblocks({k: e[k] for k in items}) for n, e in sets.items()}
    hits, aucs = {}, {}
    for n, v in vec.items():
        hits[n], aucs[n] = identify(v)
    # bd32: 32 random of bd's 40 neurons, 20 draws, mean score over draws
    draws, draws_cov, bd_aucs = [], [], []
    for t in range(R_DRAWS):
        rng = np.random.default_rng(t)
        e = {k: shallow_describe(fits["bd"][k][0], sorted(rng.choice(40, 32, replace=False).tolist()))
             for k in items}
        h, a = identify(zblocks(e))
        draws.append(h)
        bd_aucs.append(a)
        draws_cov.append(identify(zblocks({k: [cov[k], *e[k]] for k in items}))[0])
    hits["bd32"] = {s: float(np.mean([d[s] for d in draws])) for s in subj}
    aucs["bd32"] = float(np.mean(bd_aucs))
    hits["cov+bd32"] = {s: float(np.mean([d[s] for d in draws_cov])) for s in subj}
    for b, name in ((0, "spectrum"), (1, "topo")):
        hits[f"added_{name}"], aucs[f"added_{name}"] = identify(
            zblocks({k: [sets["added"][k][b]] for k in items}))
    hits["cov"], aucs["cov"] = identify(zblocks({k: [cov[k]] for k in items}))
    hits["cov+added"], aucs["cov+added"] = identify(
        zblocks({k: [cov[k], *sets["added"][k]] for k in items}))
    n = len(subj)
    chance = 1 / n
    con = {}
    pair = {"P1 added - bd32": ("added", "bd32"), "P2 added - added_random": ("added", "added_random"),
            "added_bdinit - bd32": ("added_bdinit", "bd32"), "added - init8": ("added", "init8"),
            "added - bd40": ("added", "bd40"), "added_random - bd32": ("added_random", "bd32"),
            "cov+added - cov": ("cov+added", "cov"), "cov+bd32 - cov": ("cov+bd32", "cov")}
    for name, (x, y) in pair.items():
        con[name] = boot([hits[x][s] - hits[y][s] for s in subj])
    con["added - chance"] = boot([hits["added"][s] - chance for s in subj])
    out[ds] = {"n_subjects": n, "chance": chance, "fits": counts, "excluded": excluded,
               "top1": {k: float(np.mean([v[s] for s in subj])) for k, v in hits.items()},
               "auc": aucs, "contrasts": con}
    L += [f"## {ds} (n = {n} sujets à 2 sessions, hasard = {chance:.3f})", "",
          f"fits lus {counts}, exclus {excluded}", "",
          "| embedding | top-1 | AUC |", "|---|---|---|"]
    L += [f"| {k} | {out[ds]['top1'][k]:.3f} | {aucs[k]:.3f} |" if k in aucs else
          f"| {k} | {out[ds]['top1'][k]:.3f} | — |" for k in hits]
    L += ["", "| contraste (apparié par sujet) | IC95 | MDE |", "|---|---|---|"]
    L += [f"| {k} | {ci(v)} | {v['mde']:.3f} |" for k, v in con.items()]
    L.append("")
    return sets, items


def exploratory(cells, own_added, own_bd, L, out):
    """§ 7.1 (cross_subject LOO signature) and § 7.2 (EEGNeX spectrum), lee2019_mi."""
    ds = PRIMARY
    res = {}
    # own embedding of a subject = mean of its two cross_session items (z-scored per block)
    own = {}
    for arm, emb in (("tiny", own_added), ("bd", own_bd)):
        v = zblocks(emb)
        own[arm] = {s: np.mean([v[k] for k in v if k[0] == s], 0) for s in {k[0] for k in v}}
    for arm, idx in (("tiny", list(range(8, 40))), ("bd", list(range(40)))):
        f = load_fits(cells, "cross_subject", ds, "shallow", arm)
        e, ex = shallow_sets(f, arm, idx)
        fit = {k[0]: v for k, v in zblocks(e).items()}
        res[arm] = {"hits": loo_signature(fit, own[arm]), "excluded": ex, "n_fits": len(f)}
    subj = sorted(set(res["tiny"]["hits"]) & set(res["bd"]["hits"]))
    chance = 1 / len(subj)
    c = {f"{a} - chance": boot([res[a]["hits"][s] - chance for s in subj]) for a in ("tiny", "bd")}
    c["tiny - bd"] = boot([res["tiny"]["hits"][s] - res["bd"]["hits"][s] for s in subj])
    out["exploratory_loo"] = {"n": len(subj), "chance": chance, "contrasts": c,
                              "fits": {a: (res[a]["n_fits"], res[a]["excluded"]) for a in res}}
    L += ["## Exploratoire 7.1 — cross_subject, signature du sujet laissé de côté (lee2019_mi)", "",
          f"n = {len(subj)}, hasard = {chance:.3f}. Rien n'en est conclu (PREREG § 7.1).", "",
          "| contraste | IC95 |", "|---|---|"] + [f"| {k} | {ci(v)} |" for k, v in c.items()] + [""]
    # § 7.2 EEGNeX, junction 2 spectrum
    nx = {}
    for arm, idx in (("tiny", list(range(4, 32))), ("bd", list(range(32)))):
        f = load_fits(cells, "cross_session", ds, "eegnex", arm)
        e, ex = shallow_sets(f, arm, idx, model="eegnex", describe=eegnex_describe)
        nx[arm] = (e, ex, len(f))
    items = sorted(set(nx["tiny"][0]) & set(nx["bd"][0]))
    h = {a: identify(zblocks({k: nx[a][0][k] for k in items}))[0] for a in nx}
    subj = sorted(set(h["tiny"]) & set(h["bd"]))
    c = {"eegnex added - bd": boot([h["tiny"][s] - h["bd"][s] for s in subj]),
         "eegnex added - chance": boot([h["tiny"][s] - 1 / len(subj) for s in subj])}
    out["exploratory_eegnex"] = {"n": len(subj), "contrasts": c,
                                 "fits": {a: (nx[a][2], nx[a][1]) for a in nx}}
    L += ["## Exploratoire 7.2 — EEGNeX, spectre des noyaux ajoutés (jonction 2, lee2019_mi)", "",
          f"n = {len(subj)}, fits {out['exploratory_eegnex']['fits']}", "",
          "| contraste | IC95 |", "|---|---|"] + [f"| {k} | {ci(v)} |" for k, v in c.items()] + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-a", required=True)
    ap.add_argument("--cells-s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = cells_of([a.cells_a, a.cells_s])
    out, L = {}, ["# Empreinte du sujet dans les neurones ajoutés — campagne A (PREREG_growth_embedding_A.md)", ""]
    keep = {}
    for ds in DATASETS:
        keep[ds] = run_dataset(cells, ds, L, out)
    c = out[PRIMARY]["contrasts"]
    p1, p2 = c["P1 added - bd32"], c["P2 added - added_random"]
    v1 = ("EMPREINTE PROPRE À LA CROISSANCE" if p1["lo"] > 0 else
          "MOINS QU'UN RÉSEAU ENTRAÎNÉ" if p1["hi"] < 0 else "PAS AU-DELÀ D'UN RÉSEAU ENTRAÎNÉ")
    v2 = ("CRÉÉE PAR LA RÈGLE" if p2["lo"] > 0 else
          "PLUS FORTE SANS LA RÈGLE" if p2["hi"] < 0 else "PAS PAR LA RÈGLE")
    both = p1["lo"] > 0 and p2["lo"] > 0
    out["verdict"] = {"P1": v1, "P2": v2, "rule_writes_subject": both}
    L[2:2] = ["## Verdicts (lee2019_mi, PREREG § 5)", "",
              f"- P1 ajoutés − bd32 = {ci(p1)} : **{v1}**",
              f"- P2 ajoutés − ajoutés_random = {ci(p2)} : **{v2}**",
              f"- La règle TINY écrit le sujet dans ses neurones (P1 et P2) : **{'OUI' if both else 'NON'}**",
              "- « Au-delà de la covariance » : décrit, pas testé (PREREG § 9).", ""]
    sets, items = keep[PRIMARY]
    exploratory(cells, {k: sets["added"][k] for k in items}, {k: sets["bd40"][k] for k in items}, L, out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
