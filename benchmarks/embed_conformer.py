"""Subject fingerprint in Conformer's grown FFN neurons (analysis/EXPLO_embed_conformer.md,
exploratory). Committed with the note BEFORE any Conformer weight is read for this purpose
(only one fit's key names and shapes were looked at, 09/11).

Same pipeline as reading 1 (embed_A: z-scored blocks, correlation, cross-session
nearest neighbour, bootstrap over subjects) on lee2019_mi cross_session, Conformer, arms
tiny / random / bd, seed 0. A neuron j of block 0's FFN is described by what it reads,
W1[j]**2 / ||W1[j]||**2 (40 token features), and what it writes, W2[:, j]**2 / ||W2[:, j]||**2;
a set of neurons is the mean of each block over the set.

    python embed_conformer.py --cells ../grids/A/campaignA_cells_planned_av5.txt --out ../read/embed_conformer
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

import embed_A
from embed_A import boot, ci, pairs, zblocks
from embed_growth import identify_ties

DS, MODEL = "lee2019_mi", "conformer"
G1, G2 = "pairs.0.first.layer.weight", "pairs.0.second.layer.weight"
B1, B2 = "transformer.0.1.fn.1.0.weight", "transformer.0.1.fn.1.3.weight"
START, BD_W, N_DRAWS, TOL = 16, 160, 20, 1e-6


def sk(sd, names):
    """The one key of ``names`` present in sd, with or without a leading 'net.'."""
    k = [n for n in sd for want in names if n == want or n == "net." + want]
    assert len(k) == 1, (names, k)
    return sd[k[0]]


def describe(W1, W2, idx):
    """[read block, write block] of the neurons idx (float64, each row normalised)."""
    r = W1[idx].double().numpy() ** 2
    w = W2[:, idx].double().numpy().T ** 2
    return [(r / r.sum(1, keepdims=True)).mean(0), (w / w.sum(1, keepdims=True)).mean(0)]


def load(cells, arm):
    """{(subject, train session): {"final": sd, "events": [...] or None}}."""
    import torch
    out = {}
    for c in cells:
        if (c["prot"], c["ds"], c["model"], c["arm"]) != ("cross_session", DS, MODEL, arm):
            continue
        for f in sorted(glob.glob(f"{c['csv'][:-4]}__fits__final/*.pt")):
            p = torch.load(f, map_location="cpu", weights_only=False)
            tr = p["meta"]["train_sessions"]
            if len(tr) != 1:
                continue
            k = (str(p["meta"]["subject"]), tr[0].split(":", 1)[1])
            assert k not in out, k
            g = f.replace("__fits__final/", "__fits__growth/")
            ev = torch.load(g, map_location="cpu", weights_only=False)["events"] if os.path.exists(g) else None
            out[k] = {"final": p["state_dict"], "events": ev}
    return out


def grown_sets(d):
    """(sets, reason, diag). sets = {name: [read, write]}; reason = exclusion or None."""
    ev = [e for e in (d["events"] or []) if e["junction"] == 0]
    if not ev:
        return None, "no junction-0 event", None
    if ev[0]["shapes_before"][G1][0] != START:
        return None, "start width", None
    if any(b["shapes_before"][G1][0] != a["tensors"][G1].shape[0] for a, b in zip(ev, ev[1:])):
        return None, "non-contiguous junction-0 events", None
    W1, W2 = sk(d["final"], [G1]), sk(d["final"], [G2])
    h = W1.shape[0]
    if h != ev[-1]["tensors"][G1].shape[0]:
        return None, "final width != last event", None
    sets = {"added": describe(W1, W2, list(range(START, h))),
            "start16": describe(W1, W2, list(range(START)))}
    # at insertion: each neuron at the event that created it (neuron-count weighted mean)
    parts, n, ratio = [None, None], 0, []
    for e in ev:
        b, a = e["shapes_before"][G1][0], e["tensors"][G1].shape[0]
        T1, T2 = e["tensors"][G1], e["tensors"][G2]
        x = describe(T1, T2, list(range(b, a)))
        parts = [x[i] * (a - b) if parts[i] is None else parts[i] + x[i] * (a - b) for i in range(2)]
        n += a - b
        new = float(T1[b:a].double().pow(2).sum(1).mean())
        old = float(T1[:b].double().pow(2).sum(1).mean())
        ratio.append(new / old)
    sets["added_ins"] = [p / n for p in parts]
    def added(e):
        k = f"pairs.{e['junction']}.first.layer.weight"
        return e["tensors"][k].shape[0] - e["shapes_before"][k][0]
    diag = {"n_added": h - START,
            "share_j0": sum(added(e) for e in ev) / sum(added(e) for e in d["events"]),
            "new_over_old_sqnorm_median": float(np.median(ratio))}
    return sets, None, diag


def neurons_ins(d):
    """[(read vec, write vec)] of every junction-0 neuron at its insertion (clone check)."""
    out = []
    for e in d["events"]:
        if e["junction"] != 0:
            continue
        b, a = e["shapes_before"][G1][0], e["tensors"][G1].shape[0]
        for j in range(b, a):
            out.append((e["tensors"][G1][j].double().numpy(), e["tensors"][G2][:, j].double().numpy()))
    return out


def clones(fits):
    """Fraction of random junction-0 neurons with an exact twin (cos >= 1 - TOL on both
    W1 row and W2 column) in another subject's fit / in the same subject's other fit."""
    rows = [(k, v) for k, d in sorted(fits.items()) for v in neurons_ins(d)]
    subj = np.array([k[0] for k, _ in rows])
    item = np.array(["|".join(k) for k, _ in rows])

    def unit(X):
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    A, B = unit(np.stack([v[0] for _, v in rows])), unit(np.stack([v[1] for _, v in rows]))
    same = ((A @ A.T) >= 1 - TOL) & ((B @ B.T) >= 1 - TOL)
    os_ = (same & (subj[:, None] != subj[None, :])).any(1)
    ss_ = (same & (subj[:, None] == subj[None, :]) & (item[:, None] != item[None, :])).any(1)
    return {"n": len(rows), "other_subject": float(os_.mean()), "same_subject": float(ss_.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from campaignA_check import parse_cells
    cells = list(parse_cells(a.cells))
    emb, excl, diag = collections.defaultdict(dict), collections.defaultdict(collections.Counter), {}
    fits = {arm: load(cells, arm) for arm in ("tiny", "random", "bd")}
    for arm in ("tiny", "random"):
        diag[arm] = []
        for k, d in fits[arm].items():
            sets, why, dg = grown_sets(d)
            if why:
                excl[arm][why] += 1
                continue
            diag[arm].append(dg)
            for name, e in sets.items():
                emb[f"{arm}_{name}"][k] = e
    bd_draw = {}
    for k, d in fits["bd"].items():
        W1, W2 = sk(d["final"], [B1]), sk(d["final"], [B2])
        if W1.shape[0] != BD_W:
            excl["bd"]["bd width"] += 1
            continue
        emb["bd_all"][k] = describe(W1, W2, list(range(BD_W)))
        bd_draw[k] = (W1, W2)
    # common items: two-session subjects present in every set
    common = set.intersection(*[set(v) for v in emb.values()], set(bd_draw))
    subj, _, _ = pairs({k: 0 for k in common})
    items = sorted(k for k in common if k[0] in subj)
    hits = {name: identify_ties(zblocks({k: e[k] for k in items}))[0] for name, e in emb.items()}
    # bd matched in count: per item, n = added count of the tiny fit of the same item
    m = {k: int(sk(fits["tiny"][k]["final"], [G1]).shape[0]) - START for k in items}
    acc = collections.defaultdict(float)
    for t in range(N_DRAWS):
        rng = np.random.default_rng(t)
        e = {k: describe(*bd_draw[k], sorted(rng.choice(BD_W, min(m[k], BD_W), replace=False).tolist()))
             for k in items}
        for s, h in identify_ties(zblocks(e))[0].items():
            acc[s] += h / N_DRAWS
    hits["bd_matched"] = dict(acc)
    n = len(subj)
    top1 = {k: float(np.mean([h[s] for s in subj])) for k, h in hits.items()}

    def diff(x, y):
        return boot([hits[x][s] - hits[y][s] for s in subj])

    def vs_chance(x):
        return boot([hits[x][s] - 1.0 / n for s in subj])
    C = {"E0 tiny_added - chance": vs_chance("tiny_added"),
         "E1 tiny_added - bd_matched": diff("tiny_added", "bd_matched"),
         "E2 tiny_added - random_added": diff("tiny_added", "random_added"),
         "random_added - bd_matched": diff("random_added", "bd_matched"),
         "tiny_added - tiny_start16": diff("tiny_added", "tiny_start16"),
         "tiny_added_ins - chance": vs_chance("tiny_added_ins"),
         "random_added_ins - chance": vs_chance("random_added_ins"),
         "tiny_added 200 - ins": diff("tiny_added", "tiny_added_ins"),
         "random_added 200 - ins": diff("random_added", "random_added_ins")}
    cl = clones({k: fits["random"][k] for k in items})
    dg = {arm: {key: float(np.median([x[key] for x in v])) for key in v[0]} for arm, v in diag.items()}
    L = ["# Empreinte du sujet dans les FFN grandis de Conformer (EXPLO_embed_conformer.md, exploratoire)", "",
         f"lee2019_mi, cross_session, n = {n} sujets, hasard = {1 / n:.3f}", "",
         f"fits lus { {a: len(v) for a, v in fits.items()} }, exclus { {a: dict(v) for a, v in excl.items()} }", "",
         f"diagnostics (médianes par fit) : {dg}", "",
         f"clones random à l'insertion : {cl}", "",
         "| ensemble | top-1 |", "|---|---|"]
    L += [f"| {k} | {v:.3f} |" for k, v in sorted(top1.items())]
    L += ["", "| contraste (apparié par sujet) | IC95 | MDE |", "|---|---|---|"]
    L += [f"| {k} | {ci(v)} | {v['mde']:.3f} |" for k, v in C.items()]
    e0, e1 = C["E0 tiny_added - chance"], C["E1 tiny_added - bd_matched"]
    verdict = ("EMPREINTE" if e0["lo"] > 0 else "PAS D'EMPREINTE DÉTECTÉE") + " ; " + (
        "PROPRE À LA CROISSANCE" if e1["lo"] > 0 else "PAS AU-DELÀ DE bd" if e1["hi"] >= 0 else "MOINS QUE bd")
    L += ["", f"**Lecture (§ 4)** : {verdict}"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    json.dump({"top1": top1, "contrasts": C, "clones": cl, "diag": dg,
               "excluded": {a: dict(v) for a, v in excl.items()}}, open(a.out + ".json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
