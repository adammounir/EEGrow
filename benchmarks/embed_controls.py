"""Three controls on the growth-neuron subject fingerprint (PREREG_growth_embedding_offline.md
§ 7). Committed BEFORE any of their results is read.

(a) count-matched bd: each bd fit keeps a random subset of its 40 neurons of the SAME size
    as the paired tiny fit's added set (pair = subject, session, cv_ind), 20 draws.
(b) width alone: the width at the selected epoch (a subject trait through the learning
    curve) as a 1-D embedding.
(c) raw-EEG covariance under the same preprocessing (paradigm, 8-32 Hz, 250 Hz, cache):
    log-Euclidean mean of the trace-normalised trial covariances of each (subject,
    session), upper triangle. Plus cov+added (concatenated z-blocks) - the question that
    matters: does the growth fingerprint add anything to a plain data statistic?

Usage::

    python embed_controls.py --root /scratch/amounir/eegrow_valid --out ../read/embed_controls
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from embed_offline import ARMS, DATASETS, N_INIT, boot, ci, describe, identify, weight

R_DRAWS = 20
MOABB = {"lee2019_mi": ("Lee2019_MI", "LeftRightImagery", True),
         "bnci2014_001": ("BNCI2014_001", "MotorImagery", False)}


def load(root, arm, ds, seed):
    folder, model = ARMS[arm]
    pat = f"{root}/valid_results/{folder}/results/within_session/{ds}/{model}__seed{seed}__fits__selected/*.pt"
    out = {}
    for f in sorted(glob.glob(pat)):
        p = torch.load(f, map_location="cpu", weights_only=False)
        m = p["meta"]
        out[(m["subject"], m["session"], m["cv_ind"])] = p["state_dict"]
    return out


def mean_by_session(items):
    acc = defaultdict(list)
    for (s, se, _), v in items.items():
        acc[(s, se)].append(v)
    return {k: tuple(np.mean([v[i] for v in vs], 0) for i in range(len(vs[0]))) for k, vs in acc.items()}


def identify_blocks(emb):
    """identify() on an embedding whose tuple holds any number of blocks (all used)."""
    k = next(iter(emb))
    nb = len(emb[k])
    if nb == 2:
        return identify(emb, "both")
    if nb == 1 and np.size(emb[k][0]) == 1:
        return identify_scalar(emb)
    if nb == 1:
        return identify(emb, "spectrum")  # the 'spectrum' slot = block 0
    flat = {kk: (np.concatenate([np.atleast_1d(b) for b in v[:-1]]), np.atleast_1d(v[-1])) for kk, v in emb.items()}
    return identify(flat, "both")


def identify_scalar(emb):
    """Same cross-session NN identification as identify(), for a 1-D embedding: the
    correlation distance is undefined on one number, so the distance is |a - b|."""
    subjects = sorted({s for s, _ in emb})
    sessions = sorted({se for _, se in emb})
    subjects = [s for s in subjects if all((s, se) in emb for se in sessions)]
    n = len(subjects)
    A = np.array([float(emb[(s, sessions[0])][0]) for s in subjects])
    B = np.array([float(emb[(s, sessions[1])][0]) for s in subjects])
    C = -np.abs(A[:, None] - B[None, :])
    rng = np.random.default_rng(0)
    C = C + 1e-9 * rng.standard_normal(C.shape)  # break exact ties at random, not by index
    hit = 0.5 * (C.argmax(1) == np.arange(n)) + 0.5 * (C.argmax(0) == np.arange(n))
    same, diff = np.diag(C), C[~np.eye(n, dtype=bool)]
    auc = float((same[:, None] > diff[None, :]).mean())
    return dict(zip(subjects, hit.tolist())), auc


def covariances(ds):
    import moabb.datasets as mds
    import moabb.paradigms as mpar
    from utils import cache_config
    cls, par, cached = MOABB[ds]
    dataset = getattr(mds, cls)()
    if getattr(dataset, "_selected_sessions", None) is not None:
        dataset._selected_sessions = None  # same off-by-one fix as run_moabb_hydra
    paradigm = getattr(mpar, par)(fmin=8.0, fmax=32.0, resample=250.0)
    cc = cache_config({"enabled": True, "path": "/scratch/amounir/moabb_cache"}) if cached else None
    emb = {}
    for subj in dataset.subject_list:
        X, _, meta = paradigm.get_data(dataset=dataset, subjects=[subj], **({"cache_config": cc} if cc else {}))
        for se in sorted(meta["session"].unique()):
            x = X[(meta["session"] == se).to_numpy()].astype(np.float64)
            x = x - x.mean(-1, keepdims=True)
            C = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
            C /= np.trace(C, axis1=1, axis2=2)[:, None, None]
            w, V = np.linalg.eigh(C)
            L = np.einsum("ncd,nd,ned->nce", V, np.log(np.maximum(w, 1e-12)), V).mean(0)
            iu = np.triu_indices(L.shape[0])
            emb[(str(subj), str(se))] = (L[iu],)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/amounir/eegrow_valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, L = {}, ["# Contrôles de l'empreinte des neurones ajoutés — lecture", ""]
    for ds, seeds in DATASETS.items():
        hits = defaultdict(list)  # name -> list over seeds of {subject: hit}
        auc = defaultdict(list)
        cov = covariances(ds)
        for seed in seeds:
            tiny, bd = load(a.root, "added", ds, seed), load(a.root, "bd", ds, seed)
            added, widths, n_add = {}, {}, {}
            for k, sd in tiny.items():
                w = weight(sd, "time").shape[0]
                widths[k] = (np.array([float(w)]),)
                if w > N_INIT:
                    added[k] = describe(sd, list(range(N_INIT, w)))
                    n_add[k] = w - N_INIT
            emb_added = mean_by_session(added)
            for name, emb in {"added": emb_added, "width_only": mean_by_session(widths)}.items():
                h, u = identify_blocks(emb)
                hits[name].append(h); auc[name].append(u)
            rng = np.random.default_rng(seed)
            draw_hits = []
            for _ in range(R_DRAWS):
                sub = {k: describe(bd[k], sorted(rng.choice(40, n_add[k], replace=False).tolist()))
                       for k in n_add if k in bd}
                h, u = identify_blocks(mean_by_session(sub))
                draw_hits.append(h); auc["bd_matched"].append(u)
            hits["bd_matched"].append({s: float(np.mean([d[s] for d in draw_hits])) for s in draw_hits[0]})
            keys = sorted(set(cov) & set(emb_added))
            for name, emb in {"cov": {k: cov[k] for k in keys},
                              "cov+added": {k: (cov[k][0], np.concatenate(emb_added[k])) for k in keys}}.items():
                h, u = identify_blocks(emb)
                hits[name].append(h); auc[name].append(u)
        subj = sorted(set.intersection(*[set(hs[0]) for hs in hits.values()]))
        per = {n: {s: float(np.mean([h[s] for h in hs if s in h])) for s in subj} for n, hs in hits.items()}
        out[ds] = {"n_subjects": len(subj), "chance": 1 / len(subj),
                   "top1": {n: float(np.mean(list(v.values()))) for n, v in per.items()},
                   "auc": {n: float(np.mean(v)) for n, v in auc.items()}, "contrasts": {}}
        L += [f"## {ds} (n = {len(subj)}, hasard = {1 / len(subj):.3f})", "", "| embedding | top-1 | AUC |",
              "|---|---|---|"]
        L += [f"| {n} | {out[ds]['top1'][n]:.3f} | {out[ds]['auc'][n]:.3f} |" for n in per]
        for name, (x, y) in {"(a) added - bd_matched": ("added", "bd_matched"),
                             "(b) added - width_only": ("added", "width_only"),
                             "(c) added - cov": ("added", "cov"),
                             "(c) cov+added - cov": ("cov+added", "cov")}.items():
            out[ds]["contrasts"][name] = boot([per[x][s] - per[y][s] for s in subj])
        L += ["", "| contraste (top-1, apparié par sujet) | IC95 |", "|---|---|"]
        L += [f"| {k} | {ci(v)} |" for k, v in out[ds]["contrasts"].items()]
        L.append("")
    c = out["lee2019_mi"]["contrasts"]
    nontrivial = c["(a) added - bd_matched"]["lo"] > 0 and c["(b) added - width_only"]["lo"] > 0
    beyond = c["(c) cov+added - cov"]["lo"] > 0
    v1 = "EMPREINTE NON TRIVIALE" if nontrivial else "EMPREINTE EXPLIQUÉE PAR (a) OU (b)"
    v2 = "AU-DELÀ DE LA COVARIANCE" if beyond else "REDONDANTE AVEC LA COVARIANCE"
    out["verdict"] = {"a_b": v1, "c": v2}
    L += ["## Verdicts (lee2019_mi, PREREG § 7)", "", f"- (a)+(b) : **{v1}**", f"- (c) : **{v2}**"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
