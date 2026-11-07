"""When does the subject fingerprint appear in the added neurons? (exploratory,
analysis/EXPLO_growth_trajectory_A.md). Committed with the plan BEFORE any growth snapshot
(``__fits__growth``) or checkpoint (``__fits__ckpt``) is read.

lee2019_mi, cross_session, 54 two-session subjects; same embedding and identification as
embed_A.py, with ties counted explicitly (seed 0 everywhere: inits, and maybe random
insertions, are identical across subjects).

Usage::

    python embed_growth.py --cells ../grids/A/campaignA_cells_planned_av5.txt --out ../read/embed_growth
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import embed_A
from embed_A import boot, ci, key, pairs, zblocks, SHALLOW

DS = "lee2019_mi"
ARMS = ("tiny", "random", "bd")
CKPT = (10, 25, 50, 100)
TOL = 1e-9
# amendment 1: the number of growth events varies by fit (up to 11, last one up to epoch
# 55 on the fits read), so all 40 neurons exist only from the end of growth; the added set
# is read at insertion, 100 and 200, and a fit still growing after epoch 100 is excluded
ADDED_FROM = 100


def describe(sd, idx):
    """embed_A.shallow_describe, but a zero-norm neuron (nothing to normalise) is counted
    and gives a zero row instead of NaN."""
    t = key(sd, SHALLOW["time"])[idx, 0, :, 0].double().numpy()
    s = key(sd, SHALLOW["spat"])[:, idx, 0, :].double().numpy()
    spec = np.abs(np.fft.rfft(t, n=64, axis=1)) ** 2
    topo = (s ** 2).sum(0)
    zs, zt = spec.sum(1, keepdims=True), topo.sum(1, keepdims=True)
    n_zero = int((zs == 0).sum() + (zt == 0).sum())
    spec = np.divide(spec, zs, out=np.zeros_like(spec), where=zs > 0)
    topo = np.divide(topo, zt, out=np.zeros_like(topo), where=zt > 0)
    return [spec.mean(0), topo.mean(0)], n_zero


def identify_ties(vec):
    """embed_A.identify with ties: a hit is 1/(number of candidates within TOL of the max)
    when the true subject is one of them. Also returns the fraction of items whose vector
    equals (within TOL) another subject's."""
    subj, ka, kb = pairs(vec)
    n = len(subj)
    A, B = np.array([vec[k] for k in ka]), np.array([vec[k] for k in kb])
    C = embed_A.corr_rows(A, B)
    hit = np.zeros(n)
    for M, w in ((C, 0.5), (C.T, 0.5)):
        top = M >= M.max(1, keepdims=True) - TOL
        hit += w * np.where(top[np.arange(n), np.arange(n)], 1.0 / top.sum(1), 0.0)
    X = np.concatenate([A, B])
    owner = np.repeat(np.arange(n)[None], 2, 0).ravel()
    D = np.abs(X[:, None, :] - X[None, :, :]).max(2)
    clone = ((D <= TOL) & (owner[:, None] != owner[None, :])).any(1)
    return dict(zip(subj, hit.tolist())), float(clone.mean())


def snapshots(stem):
    """{uid: {"ins": [(epoch, width_before, width_after, state_dict)], 0: sd, 10..100: sd,
    200: sd, "meta": meta}} for one cell; missing files leave the key out."""
    import torch
    out = {}
    for f in sorted(glob.glob(f"{stem}__fits__final/*.pt")):
        uid = os.path.basename(f)
        p = torch.load(f, map_location="cpu", weights_only=False)
        d = {"meta": p["meta"], 200: p["state_dict"]}
        g = f.replace("__fits__final/", "__fits__init/")
        if os.path.exists(g):
            d[0] = torch.load(g, map_location="cpu", weights_only=False)["state_dict"]
        g = f.replace("__fits__final/", "__fits__ckpt/")
        if os.path.exists(g):
            for ep, sd in torch.load(g, map_location="cpu", weights_only=False)["checkpoints"].items():
                d[int(ep)] = sd
        g = f.replace("__fits__final/", "__fits__growth/")
        if os.path.exists(g):
            ev = torch.load(g, map_location="cpu", weights_only=False)["events"]
            d["ins"] = [(e["epoch"], e["shapes_before"][SHALLOW["time"][0]][0],
                         e["width_after"], e["tensors"]) for e in ev]
        out[uid] = d
    return out


def width(sd):
    return key(sd, SHALLOW["time"]).shape[0]


def load(cells, arm):
    """{item: snapshots} for (lee2019_mi, cross_session, shallow, arm), items with one
    training session only (embed_A.load_fits)."""
    out = {}
    for c in cells:
        if (c["prot"], c["ds"], c["model"], c["arm"]) != ("cross_session", DS, "shallow", arm):
            continue
        for d in snapshots(c["csv"][:-4]).values():
            tr = d["meta"]["train_sessions"]
            if len(tr) != 1:
                continue
            k = (str(d["meta"]["subject"]), tr[0].split(":", 1)[1])
            assert k not in out, k
            out[k] = d
    return out


def check(d, arm):
    """Reason to exclude a fit, or None: widths must follow its own growth events."""
    if arm == "bd":
        return None if all(width(d[t]) == 40 for t in (0, *CKPT, 200) if t in d) else "bd width"
    ev = d.get("ins")
    if not ev or ev[0][0] != 5 or ev[0][1] != 8 or ev[-1][2] != 40 or width(d[200]) != 40:
        return "growth events"
    if any(b != a for (_, _, a, _), (_, b, _, _) in zip(ev, ev[1:])):
        return "non-contiguous events"
    if ev[-1][0] > ADDED_FROM:
        return "growth ends after epoch 100"
    w_at = {t: max([8] + [a for e, _, a, _ in ev if e <= t]) for t in (0, *CKPT)}
    if any(t in d and width(d[t]) != w_at[t] for t in (0, *CKPT)):
        return "checkpoint width"
    return None


def sets_at(d, arm):
    """{(set, instant): sd-or-None, idx}. 'ins' composes each neuron at its own event."""
    out = {}
    if arm == "bd":
        for t in (0, *CKPT, 200):
            if t in d:
                out[("bd40", t)] = [(d[t], list(range(40)))]
        return out
    ev = d["ins"]
    g1 = list(range(8, ev[0][2]))
    out[("added", "ins")] = [(sd, list(range(b, a))) for _, b, a, sd in ev]
    out[("G1", "ins")] = [(ev[0][3], g1)]
    for t in (0, *CKPT, 200):
        if t not in d:
            continue
        if t != 0:
            out[("init8", t)] = [(d[t], list(range(8)))]
        if t >= ADDED_FROM:
            out[("added", t)] = [(d[t], list(range(8, 40)))]
        if t >= 10:
            out[("G1", t)] = [(d[t], g1)]
    out[("init8", 0)] = [(d[0], list(range(8)))] if 0 in d else None
    return {k: v for k, v in out.items() if v}


def embed(parts):
    """Neuron-count weighted mean of (spectrum, topography) over the parts of a set."""
    tot, n, zeros = None, 0, 0
    for sd, idx in parts:
        e, z = describe(sd, idx)
        zeros += z
        tot = [x * len(idx) for x in e] if tot is None else [t + x * len(idx) for t, x in zip(tot, e)]
        n += len(idx)
    return [x / n for x in tot], zeros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from campaignA_check import parse_cells
    cells = list(parse_cells(a.cells))
    emb, excluded, zeros = {}, {}, {}
    for arm in ARMS:
        fits = load(cells, arm)
        excluded[arm] = {}
        for k, d in fits.items():
            why = check(d, arm)
            if why:
                excluded[arm][why] = excluded[arm].get(why, 0) + 1
                continue
            for (name, t), parts in sets_at(d, arm).items():
                e, z = embed(parts)
                emb.setdefault((arm, name, t), {})[k] = e
                zeros[(arm, name, t)] = zeros.get((arm, name, t), 0) + z
    # every set on the same items: two-session subjects present in all of them
    common = set.intersection(*[set(v) for v in emb.values()])
    subj, _, _ = pairs({k: 0 for k in common})
    items = sorted(k for k in common if k[0] in subj)
    n = len(subj)
    hits, clones = {}, {}
    for key_, e in emb.items():
        hits[key_], clones[key_] = identify_ties(zblocks({k: e[k] for k in items}))
    top1 = {k: float(np.mean(list(h.values()))) for k, h in hits.items()}

    def diff(x, y):
        return boot([hits[x][s] - hits[y][s] for s in subj])

    def vs_chance(x):
        return boot([hits[x][s] - 1 / n for s in subj])

    T = {"T1 random added ins - chance": vs_chance(("random", "added", "ins")),
         "T2 random added 200 - ins": diff(("random", "added", 200), ("random", "added", "ins")),
         "T4 tiny added ins - chance": vs_chance(("tiny", "added", "ins")),
         "tiny added 200 - ins": diff(("tiny", "added", 200), ("tiny", "added", "ins"))}
    for t in ("ins", 100, 200):
        T[f"T5 tiny - random, added, {t}"] = diff(("tiny", "added", t), ("random", "added", t))
    for t in ("ins", 10, 25, 50, 100, 200):
        T[f"T3 random G1 {t} - chance"] = vs_chance(("random", "G1", t))

    L = ["# Quand l'empreinte apparaît-elle ? — trajectoire des neurones ajoutés "
         "(EXPLO_growth_trajectory_A.md, exploratoire)", "",
         f"lee2019_mi, cross_session, n = {n} sujets, hasard = {1 / n:.3f}", "",
         f"exclus : {excluded}", "",
         "## Top-1 par ensemble et instant (ex æquo comptés ; « clones » = fraction d'items "
         "identiques à ceux d'un autre sujet ; « nuls » = neurones de norme nulle)", "",
         "| bras | ensemble | instant | top-1 | clones | nuls |", "|---|---|---|---|---|---|"]
    order = {"ins": -1}
    for k in sorted(top1, key=lambda k: (k[0], k[1], order.get(k[2], k[2]))):
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {top1[k]:.3f} | {clones[k]:.2f} | {zeros[k]} |")
    L += ["", "## Contrastes (appariés par sujet, IC95 bootstrap)", "",
          "| contraste | IC95 | MDE |", "|---|---|---|"]
    L += [f"| {name} | {ci(c)} | {c['mde']:.3f} |" for name, c in T.items()]
    t1, t2 = T["T1 random added ins - chance"], T["T2 random added 200 - ins"]
    at_chance = t1["lo"] <= 0 <= t1["hi"]
    rises = t2["lo"] > 0
    reading = ("l'empreinte est écrite par l'entraînement après l'insertion (exploratoire)"
               if at_chance and rises else
               "random dépend déjà des données à l'insertion : à comprendre avant d'aller plus loin"
               if t1["lo"] > 0 else
               "pas de montée après l'insertion : contradiction avec la lecture 1, chercher l'erreur")
    L += ["", f"**Lecture (§ 4)** : T1 {'≈ hasard' if at_chance else '≠ hasard'}, "
              f"T2 {'> 0' if rises else 'non > 0'} → {reading}."]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"n": n, "excluded": excluded, "top1": {str(k): v for k, v in top1.items()},
               "clones": {str(k): v for k, v in clones.items()},
               "zeros": {str(k): v for k, v in zeros.items()}, "contrasts": T, "reading": reading},
              open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
