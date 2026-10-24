"""Offline test: do the neurons added by growth carry a subject fingerprint beyond a fixed
net's? (PREREG_growth_embedding_offline.md). Committed BEFORE any result is read.

Reads the selected-epoch state_dicts saved by the validation campaign (shallow__tiny,
shallow__bd), builds an order/sign/count-invariant embedding of a neuron set (mean
normalised spectrum of the conv_time filters + mean normalised topography of their
conv_spat weights), averages it per (subject, session) and scores cross-session
nearest-neighbour subject identification.

Usage::

    python embed_offline.py --root /scratch/amounir/eegrow_valid --out ../read/embed_offline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

DATASETS = {"lee2019_mi": [0], "bnci2014_001": [0, 1, 2]}
ARMS = {"added": ("shallow__tiny", "grow_shallow"), "init8": ("shallow__tiny", "grow_shallow"),
        "bd": ("shallow__bd", "bd_shallow")}
N_INIT = 8
BLOCKS = ("both", "spectrum", "topo")


def neuron_set(arm, width):
    if arm == "added":
        return list(range(N_INIT, width)) if width > N_INIT else None
    if arm == "init8":
        return list(range(N_INIT))
    return list(range(width))


KEYS = {"time": ("conv_time.layer.weight", "conv_time_spat.conv_time.weight"),  # grow, bd
        "spat": ("conv_spat.layer.weight", "conv_time_spat.conv_spat.weight")}


def weight(sd, name):
    k = [k for k in KEYS[name] if k in sd]
    assert len(k) == 1, (name, list(sd))
    return sd[k[0]]


def describe(sd, idx):
    t = weight(sd, "time")[idx, 0, :, 0].double().numpy()
    s = weight(sd, "spat")[:, idx, 0, :].double().numpy()
    spec = np.abs(np.fft.rfft(t, n=64, axis=1)) ** 2
    spec /= spec.sum(1, keepdims=True)
    topo = (s ** 2).sum(0)
    topo /= topo.sum(1, keepdims=True)
    return spec.mean(0), topo.mean(0)


def embeddings(root, arm, ds, seed):
    folder, model = ARMS[arm]
    pat = f"{root}/valid_results/{folder}/results/within_session/{ds}/{model}__seed{seed}__fits__selected/*.pt"
    acc, excluded, n = defaultdict(list), 0, 0
    for f in sorted(glob.glob(pat)):
        p = torch.load(f, map_location="cpu", weights_only=False)
        sd, m = p["state_dict"], p["meta"]
        width = weight(sd, "time").shape[0]
        idx = neuron_set(arm, width)
        n += 1
        if idx is None:
            excluded += 1
            continue
        acc[(m["subject"], m["session"])].append(describe(sd, idx))
    emb = {k: (np.mean([v[0] for v in vs], 0), np.mean([v[1] for v in vs], 0)) for k, vs in acc.items()}
    return emb, n, excluded


def identify(emb, block):
    subjects = sorted({s for s, _ in emb})
    sessions = sorted({se for _, se in emb})
    assert len(sessions) == 2, sessions
    subjects = [s for s in subjects if all((s, se) in emb for se in sessions)]
    keys = [(s, se) for se in sessions for s in subjects]

    def z(x):
        x = np.array(x)
        return (x - x.mean(0)) / (x.std(0) + 1e-12)
    parts = []
    if block in ("both", "spectrum"):
        parts.append(z([emb[k][0] for k in keys]))
    if block in ("both", "topo"):
        parts.append(z([emb[k][1] for k in keys]))
    X = np.concatenate(parts, 1)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    n = len(subjects)
    A, B = X[:n], X[n:]
    C = A @ B.T / X.shape[1]  # correlation, rows = session A subjects
    hit = 0.5 * (C.argmax(1) == np.arange(n)) + 0.5 * (C.argmax(0) == np.arange(n))
    same, diff = np.diag(C), C[~np.eye(n, dtype=bool)]
    auc = float((same[:, None] > diff[None, :]).mean() + 0.5 * (same[:, None] == diff[None, :]).mean())
    return dict(zip(subjects, hit.tolist())), auc


def boot(d, n_boot=10_000):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": len(d)}


def ci(c):
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, L = {}, ["# Empreinte du sujet dans les neurones ajoutés — lecture", ""]
    for ds, seeds in DATASETS.items():
        res = {}
        for arm in ARMS:
            per_seed = {b: [] for b in BLOCKS}
            aucs = {b: [] for b in BLOCKS}
            counts = []
            for seed in seeds:
                emb, n, ex = embeddings(a.root, arm, ds, seed)
                counts.append((n, ex))
                for b in BLOCKS:
                    h, auc = identify(emb, b)
                    per_seed[b].append(h)
                    aucs[b].append(auc)
            res[arm] = {b: ({s: float(np.mean([h[s] for h in per_seed[b] if s in h])) for s in per_seed[b][0]},
                            float(np.mean(aucs[b]))) for b in BLOCKS}
            res[arm]["counts"] = counts
        subj = sorted(set.intersection(*[set(res[arm]["both"][0]) for arm in ARMS]))
        chance = 1 / len(subj)
        L += [f"## {ds} (n = {len(subj)} sujets, hasard = {chance:.3f})", "",
              "| bras | bloc | top-1 | AUC même/autre | fits (exclus) |", "|---|---|---|---|---|"]
        out[ds] = {"n_subjects": len(subj), "chance": chance, "arms": {}}
        for arm in ARMS:
            for b in BLOCKS:
                h, auc = res[arm][b]
                top1 = float(np.mean([h[s] for s in subj]))
                out[ds]["arms"].setdefault(arm, {})[b] = {"top1": top1, "auc": auc}
                L.append(f"| {arm} | {b} | {top1:.3f} | {auc:.3f} | "
                         f"{sum(c[0] for c in res[arm]['counts'])} ({sum(c[1] for c in res[arm]['counts'])}) |")
        contrasts = {}
        for name, (x, y) in {"added - bd": ("added", "bd"), "added - init8": ("added", "init8")}.items():
            for b in BLOCKS:
                contrasts[f"{name} [{b}]"] = boot([res[x][b][0][s] - res[y][b][0][s] for s in subj])
        out[ds]["contrasts"] = contrasts
        L += ["", "| contraste (top-1, apparié par sujet) | IC95 |", "|---|---|"]
        L += [f"| {k} | {ci(v)} |" for k, v in contrasts.items()]
        L.append("")
    D = out["lee2019_mi"]["contrasts"]["added - bd [both]"]
    v = ("EMPREINTE PROPRE À LA CROISSANCE" if D["lo"] > 0 else
         "MOINS QU'UN RÉSEAU ENTRAÎNÉ" if D["hi"] < 0 else "PAS AU-DELÀ D'UN RÉSEAU ENTRAÎNÉ")
    out["primary"] = {"D": D, "verdict": v}
    L += ["## Critère principal (lee2019_mi) : top-1(ajoutés) − top-1(bd)", "", f"D = {ci(D)}", "",
          f"**Verdict : {v}**"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
