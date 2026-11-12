"""EC pilot (analysis/PREREG_EC_pilot.md) -- do EEG architectures fail on the SAME trials?

If two different architectures agree on their errors almost as much as one architecture
agrees with itself under two seeds, the trial decides, not the model. This pilot only
decides whether the axis goes on or stops.

Measure: error consistency of Geirhos et al. (NeurIPS 2020) between two correctness
vectors, computed inside each true class and averaged with the class counts as weights.
Population: campaign A, arm bd, within_session, cho2017 and lee2019_mi, models
eegnex / msvtnet / ctnet / conformer, seeds 0 / 1 / 2. CPU, read-only, no training.

    python ec_pilot.py --selftest      --out OUT
    python ec_pilot.py --check-mapping --cells CELLS --manifest MAN --out OUT
    python ec_pilot.py --read          --cells CELLS --manifest MAN --mapping MAPDIR --out OUT

--selftest      synthetic data only (no real file is opened).
--check-mapping reads y / test_idx / subject / session / run / test_fingerprint of the real
                npz (never logits, score or diverged) and the signals the nets saw; writes
                the G-art gate and the per-trial artefact features. Prints no score.
--read          the pre-registered read: gates G0, G1, G-art, then the verdict.

A gate refuses, it does not warn: a failed gate stops the script before any verdict.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = ("cho2017", "lee2019_mi")
MODELS = ("eegnex", "msvtnet", "ctnet", "conformer")
SEEDS = (0, 1, 2)
PROT, ARM = "within_session", "bd"
# MOABB 1.5.0 WithinSessionSplitter(n_folds=5) (moabb/evaluations/evaluations.py:90-95),
# = config/eval/within_session.yaml:3; expected fits = 5 x sessions (campaignA_check.py:93)
N_FOLDS = 5
MIN_SUBJ = 20          # a dataset that keeps fewer subjects after G0 refuses the read
G1_TOL = 1e-6          # |recomputed score - stored score|
N_BOOT, BOOT_SEED = 10_000, 0
MIN_ALL_WRONG = 30     # fewer trials missed by all 12 (model, seed) -> AUC_art = NaN
FEATURES = ("log_ptp_max", "log_var_mean", "kurt_max")
# decision rule (frozen, PREREG_EC_pilot.md)
GARDE_MEDX, GARDE_R, GARDE_RLO = 0.30, 0.85, 0.75
TUE_MEDX, TUE_R, TUE_RHI = 0.15, 0.60, 0.75
ART_MAX = 0.70


# ---------------------------------------------------------------- error consistency

def kappa(a, b):
    """Geirhos et al. 2020: (c_obs - c_exp) / (1 - c_exp), c_exp = pa pb + (1-pa)(1-pb).
    NaN when c_exp = 1 (both vectors constant and equal) or when empty."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return float("nan")
    pa, pb = a.mean(), b.mean()
    c_obs = float((a == b).mean())
    c_exp = float(pa * pb + (1 - pa) * (1 - pb))
    if c_exp >= 1.0:
        return float("nan")
    return (c_obs - c_exp) / (1 - c_exp)


STRATA = ("classe_x_session", "classe", "aucune")   # primary, then the two secondaries


def strata_of(y, sess, kind="classe_x_session"):
    """One stratum label per trial. Primary: true class x session (cho2017 has one session,
    so it is the class alone); 'classe': the class alone; 'aucune': one stratum."""
    y = np.asarray(y).astype(str)
    if kind == "aucune":
        return np.zeros(len(y), dtype="<U1")
    if kind == "classe":
        return y
    return np.char.add(np.char.add(y, "|"), np.asarray(sess).astype(str))


def valid_strata(vectors, strata):
    """A stratum where ANY of the vectors is constant (all right or all wrong) is dropped
    for EVERY couple, so that the same strata enter kappa_x, kappa_seed and kappa_same.
    vectors = {(model, seed): bool array}. -> (kept masks, [(stratum, n trials, [models
    whose vector is constant there])])."""
    kept, dropped = [], []
    for st in np.unique(strata):
        m = strata == st
        bad = sorted({k[0] for k, v in vectors.items() if v[m].all() or not v[m].any()})
        if bad:
            dropped.append((str(st), int(m.sum()), bad))
        else:
            kept.append(m)
    return kept, dropped


def kappa_strat(a, b, masks):
    """kappa inside each kept stratum, averaged with the stratum counts as weights. In a
    kept stratum no vector is constant, so 0 < p < 1 and c_exp < 1: kappa is finite."""
    if not masks:
        return float("nan")
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float(np.average([kappa(a[m], b[m]) for m in masks], weights=[int(m.sum()) for m in masks]))


def kappa_cond(a, b, strata):
    """kappa_strat on the strata valid for the two vectors alone. -> (kappa, n dropped)."""
    masks, dropped = valid_strata({("a", 0): np.asarray(a, bool), ("b", 0): np.asarray(b, bool)}, strata)
    return kappa_strat(a, b, masks), len(dropped)


def _nanmean(v):
    v = np.asarray(v, float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")


def subject_kappas(C, strata, models, seeds):
    """Per-subject quantities on the strata valid for ALL the vectors of C (12 in the
    primary). C[(m, s)] = correctness on the subject's trials (all its sessions), aligned.
    seed[m] = mean over the 3 unordered seed pairs of kappa(m_s, m_s'); x[m, n] = mean over
    the 6 ordered s != s' of kappa(m_s, n_s'); same[m, n] = mean over s of kappa(m_s, n_s).
    No valid stratum -> every quantity NaN, valid = False."""
    masks, dropped = valid_strata(C, strata)
    K = lambda a, b: kappa_strat(a, b, masks)
    seed = {m: _nanmean([K(C[m, s], C[m, t]) for s, t in itertools.combinations(seeds, 2)])
            for m in models}
    x, same = {}, {}
    for m, n in itertools.combinations(models, 2):
        x[m, n] = _nanmean([K(C[m, s], C[n, t]) for s in seeds for t in seeds if s != t])
        same[m, n] = _nanmean([K(C[m, s], C[n, s]) for s in seeds])
    return {"seed": seed, "x": x, "same": same, "valid": bool(masks),
            "n_strata": int(len(np.unique(strata))), "dropped": dropped,
            "kx": _nanmean(list(x.values())), "ks": _nanmean(list(seed.values())),
            "ksame": _nanmean(list(same.values()))}


def drop_summary(qs, n_trials):
    """Dropped strata and share of trials lost, in total and per responsible model."""
    n_st = sum(q["n_strata"] for q in qs)
    dr = [d for q in qs for d in q["dropped"]]
    by_m = collections.defaultdict(lambda: [0, 0])
    for _, n, ms in dr:
        for m in ms:
            by_m[m][0] += 1
            by_m[m][1] += n
    return {"n_strata": n_st, "n_dropped": len(dr), "trials_lost_share": (sum(n for _, n, _ in dr) / n_trials)
            if n_trials else float("nan"),
            "by_model": {m: {"strata": v[0], "trials_share": v[1] / n_trials} for m, v in sorted(by_m.items())}}


def ratio_boot(kx, ks, rng=None):
    """R = mean(kx) / mean(ks) over the subjects, 95 % percentile CI from 10 000 subject
    resamples (the same indices for numerator and denominator)."""
    kx, ks = np.asarray(kx, float), np.asarray(ks, float)
    rng = np.random.default_rng(BOOT_SEED) if rng is None else rng
    n = len(kx)
    den = ks.mean()
    R = float(kx.mean() / den) if den > 0 else float("nan")
    idx = rng.integers(0, n, (N_BOOT, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = kx[idx].mean(1) / ks[idx].mean(1)
    return {"R": R, "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "K_x": float(kx.mean()), "K_seed": float(den), "med_x": float(np.median(kx)), "n": int(n)}


def mean_boot(v, rng=None):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    rng = np.random.default_rng(BOOT_SEED) if rng is None else rng
    b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(b, 2.5)),
            "hi": float(np.percentile(b, 97.5)), "n": int(len(v))}


def sboot_ratio(num, den):
    """The pooled ratio on e1b_ablation.sboot's scheme (e1b_ablation.py:479-497, i.e.
    adapt_B.sboot): sqrt(n_d)-weighted mean of the dataset means, subjects resampled within
    each dataset, default_rng(0), datasets in sorted order; the same draws for the
    numerator and the denominator. Reported, never decides."""
    groups = {k: (np.asarray(num[k], float), np.asarray(den[k], float)) for k in num if len(num[k])}
    if not groups:
        return None
    w = {k: np.sqrt(len(v[0])) for k, v in groups.items()}
    tot = sum(w.values())
    rng = np.random.default_rng(0)
    en = ed = 0.0
    bn, bd = np.zeros(N_BOOT), np.zeros(N_BOOT)
    for k in sorted(groups):
        a, b = groups[k]
        idx = rng.integers(0, len(a), (N_BOOT, len(a)))
        en += w[k] / tot * a.mean()
        ed += w[k] / tot * b.mean()
        bn += w[k] / tot * a[idx].mean(1)
        bd += w[k] / tot * b[idx].mean(1)
    r = bn / bd
    return {"R": float(en / ed), "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5)),
            "K_x": float(en), "K_seed": float(ed), "n_ds": len(groups)}


def decide(medx, R, lo, hi, art):
    fin = lambda v: v is not None and np.isfinite(v)
    if (fin(medx) and medx >= GARDE_MEDX and fin(R) and R >= GARDE_R and fin(lo) and lo >= GARDE_RLO
            and fin(art) and art < ART_MAX):
        return "GARDE"
    if ((fin(medx) and medx <= TUE_MEDX) or (fin(R) and R <= TUE_R and fin(hi) and hi < TUE_RHI)
            or (fin(art) and art >= ART_MAX)):
        return "TUE"
    return "NON CONCLUANT"


def decide_global(verdicts, arts):
    if all(v == "GARDE" for v in verdicts):
        return "GARDE"
    if all(v == "TUE" for v in verdicts) or any(np.isfinite(a) and a >= ART_MAX for a in arts):
        return "TUE"
    return "NON CONCLUANT"


# ---------------------------------------------------------------- campaign files

def score_of(lg, y, n_classes):
    """The scorer's own path, imported (campaignA_check.py:66-79): float32 softmax, then
    ROC-AUC on p[:, 1] at 2 classes, accuracy of its argmax otherwise."""
    from campaignA_check import score_of as s
    return s({"logits": lg, "y": y}, n_classes)


def correct_of(lg, y):
    """Correctness on the scorer's path: argmax of the float32 softmax (campaignA_check.py:74,
    = skorch predict_proba; braindecode/classifier.py:196 predict = predict_proba(X).argmax(1)).
    Column k = class k (G0 requires the npz classes to be exactly 0..C-1). Also returns how
    many trials the raw-logit argmax labels differently, and how many have a non-unique
    softmax argmax (p0 == p1 at 2 classes; numpy then takes the first column)."""
    import torch
    p = torch.softmax(torch.as_tensor(lg, dtype=torch.float32), dim=1).numpy()
    pred = p.argmax(1)
    n_diff = int((pred != np.asarray(lg).argmax(1)).sum())
    n_tie = int(((p == p.max(1, keepdims=True)).sum(1) > 1).sum())
    return pred == np.asarray(y), n_diff, n_tie


def fingerprint(Xt, y):
    """subject_stamp.py:182-183: SHA-1 of the net's input bytes, then of y's bytes."""
    h = hashlib.sha1(np.ascontiguousarray(np.asarray(Xt)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y)).tobytes())
    return h.hexdigest()


class Ctx:
    """The cell lines (campaignA_check.parse_cells) and the manifest (campaignA_check.py:207)."""

    def __init__(self, cells, manifest):
        from campaignA_check import parse_cells
        self.cells_path = cells
        self.cells = {}
        for c in parse_cells(cells):
            if c["prot"] != PROT or c["arm"] != ARM:
                continue
            k = (c["ds"], c["model"], int(c["seed"]), str(c["subj"]))
            if k in self.cells:
                raise SystemExit(f"REFUSED: cell {k} listed twice in {cells}")
            self.cells[k] = c["csv"][:-4]          # the stem (e1b_ablation.py:256)
        self.man = {Path(f).stem: json.load(open(f)) for f in glob.glob(f"{manifest}/*.json")}

    def subjects(self, ds, models):
        return sorted({k[3] for k in self.cells if k[0] == ds and k[1] in models}, key=int)

    def n_expected(self, ds, subj):
        return N_FOLDS * len(self.man[ds]["subjects"][subj]["sessions"])

    def n_classes(self, ds):
        return len(self.man[ds]["classes"])


def load_couple(stem, n_exp, subj, n_classes, full):
    """One (model, seed) of one subject -> (data, None) or (None, (motif, detail)).
    full=False opens only y, test_idx, subject, session, run, test_fingerprint of each npz
    (np.load is lazy: an unopened member is never decompressed)."""
    jl = stem + "__fits.jsonl"
    if not os.path.exists(jl):
        return None, ("jsonl_absent", "")
    lines = open(jl).read().splitlines()
    try:
        recs = [json.loads(l) for l in lines]
    except Exception:
        return None, ("jsonl_illisible", "")
    uids = [r.get("fit_uid") for r in recs]
    files = sorted(glob.glob(stem + "__fits__logits/*.npz"))
    names = [os.path.basename(f)[:-4] for f in files]
    if not len(files) == len(recs) == n_exp:
        return None, ("compte", f"{len(files)} npz, {len(recs)} lignes jsonl, {n_exp} attendus")
    if len(set(uids)) != len(uids) or set(names) != set(uids):
        return None, ("npz_hors_jsonl", "les noms des npz ne sont pas les fit_uid du jsonl")
    if any(str(r.get("subject")) != subj for r in recs):
        return None, ("jsonl_sujet", "")
    keys, ys, runs, fps, npz, g1, corr, n_tie, n_diff, div = [], [], [], [], [], [], [], 0, 0, False
    for f in files:
        with np.load(f) as z:
            y = np.asarray(z["y"])
            ti = np.asarray(z["test_idx"]).astype(np.int64)
            su = np.asarray(z["subject"]).astype(str)
            se = np.asarray(z["session"]).astype(str)
            ru = np.asarray(z["run"]).astype(str)
            fp = str(z["test_fingerprint"])
            if full:
                lg = np.asarray(z["logits"])
                st = float(z["score"])
                div |= bool(z["diverged"])
        if set(su.tolist()) != {subj}:
            return None, ("npz_sujet", os.path.basename(f))
        if len(set(se.tolist())) != 1:
            return None, ("npz_multi_session", os.path.basename(f))
        if not len(y) == len(ti) == len(se) == len(ru):
            return None, ("npz_longueurs", os.path.basename(f))
        if not np.array_equal(np.unique(y), np.arange(n_classes)):
            return None, ("classes_npz", f"{os.path.basename(f)}: {np.unique(y).tolist()}")
        if full:
            if lg.shape != (len(y), n_classes):
                return None, ("logits_forme", os.path.basename(f))
            if not np.isfinite(lg).all():
                return None, ("logits_non_finis", os.path.basename(f))
            c, dd, t = correct_of(lg, y)
            corr.append(c)
            n_diff += dd
            n_tie += t
            g1.append((f, score_of(lg, y, n_classes), st))
        keys += list(zip(se.tolist(), ti.tolist()))
        ys.append(y)
        runs.append(ru)
        fps.append(fp)
        npz.append({"file": f, "session": se[0], "test_idx": ti, "y": y, "run": ru, "fp": fp})
    cnt = collections.Counter(keys)
    if any(v != 1 for v in cnt.values()):
        return None, ("essai_repete", f"{sum(v > 1 for v in cnt.values())} essais vus plus d'une fois")
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    d = {"keys": tuple(keys[i] for i in order), "y": np.concatenate(ys)[order],
         "run": np.concatenate(runs)[order], "fps": tuple(sorted(fps)), "npz": npz,
         "diverged": div, "g1": g1, "n_tie": n_tie, "n_argmax_diff": n_diff}
    if full:
        d["correct"] = np.concatenate(corr)[order]
    return d, None


def g0_subject(ctx, ds, subj, couples, full):
    """G0 on one subject: every couple complete and once per trial, same trials and same y
    for all couples, no diverged net (full only), same fingerprints across models at equal
    seed. -> (True, {couple: data}) or (False, (motif, detail))."""
    D = {}
    n_exp, n_cls = ctx.n_expected(ds, subj), ctx.n_classes(ds)
    for m, s in couples:
        stem = ctx.cells.get((ds, m, s, subj))
        if stem is None:
            return False, ("cellule_absente", f"{m} s{s}")
        d, why = load_couple(stem, n_exp, subj, n_cls, full)
        if d is None:
            return False, (why[0], f"{m} s{s}: {why[1]}")
        if full and d["diverged"]:
            return False, ("diverge", f"{m} s{s}")
        D[m, s] = d
    ref = D[couples[0]]
    for (m, s), d in D.items():
        if d["keys"] != ref["keys"]:
            return False, ("essais_differents", f"{m} s{s}")
        if not np.array_equal(d["y"], ref["y"]):
            return False, ("y_discordant", f"{m} s{s}")
    for s in sorted({s for _, s in couples}):
        if len({D[m, t]["fps"] for m, t in couples if t == s}) != 1:
            return False, ("empreintes_modeles", f"graine {s}")
    return True, D


def population(ctx, ds, models, seeds, full, subjects=None):
    couples = [(m, s) for m in models for s in seeds]
    adm, excl = {}, collections.defaultdict(list)
    for subj in (ctx.subjects(ds, models) if subjects is None else subjects):
        ok, p = g0_subject(ctx, ds, subj, couples, full)
        if ok:
            adm[subj] = p
        else:
            excl[p[0]].append([subj, p[1]])
    return adm, dict(excl)


def refuse_small(ds, adm, excl):
    if len(adm) < MIN_SUBJ:
        raise SystemExit(f"REFUSED: {ds} keeps {len(adm)} subjects after G0 (< {MIN_SUBJ}); "
                         f"excluded by motif: {{{', '.join(f'{k}: {len(v)}' for k, v in excl.items())}}}")


# ---------------------------------------------------------------- G-art (mapping)

def gart_subject(D, X, y_get, sess, run, n_man):
    """test_idx indexes the rows of the subject's paradigm.get_data (within_session: the
    runner loads this subject only, run_moabb_hydra.py:145-156; MOABB scores
    X[test_idx], moabb/evaluations/base.py:225; the npz stores that test_idx,
    subject_stamp.py:270-275 and 187). Per npz of every couple: indices in range, label of
    the indexed window == npz y on 100 % of trials, session and run equal, and the
    fingerprint of X[test_idx] (float32, pipelines.py:26-32 and 347) == test_fingerprint;
    the window count == manifest n == the number of distinct trials."""
    n_win = len(X)
    r = {"n_win": int(n_win), "n_manifest": int(n_man), "n_trials": len(next(iter(D.values()))["keys"]),
         "n_npz": 0, "idx_out": 0, "label_bad": 0, "session_bad": 0, "run_bad": 0, "fp_bad": 0, "n_checked": 0}
    for d in D.values():
        for z in d["npz"]:
            r["n_npz"] += 1
            ti = z["test_idx"]
            r["n_checked"] += len(ti)
            if (ti < 0).any() or (ti >= n_win).any():
                r["idx_out"] += 1
                continue
            r["label_bad"] += int((y_get[ti] != z["y"]).sum())
            r["session_bad"] += int((sess[ti] != z["session"]).sum())
            r["run_bad"] += int((run[ti] != z["run"]).sum())
            r["fp_bad"] += int(fingerprint(X[ti], z["y"]) != z["fp"])
    r["count_ok"] = bool(n_win == n_man == r["n_trials"])
    r["ok"] = bool(r["count_ok"] and r["idx_out"] == 0 and r["label_bad"] == 0 and r["session_bad"] == 0
                   and r["run_bad"] == 0 and r["fp_bad"] == 0)
    return r


def features(Xk):
    """(n, 3): log max-channel peak-to-peak, log mean-channel variance, max-channel
    kurtosis (Fisher, scipy default bias=True), on the float32 windows the net saw, in float64."""
    from scipy.stats import kurtosis
    x = np.asarray(Xk, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.log((x.max(2) - x.min(2)).max(1))
        f2 = np.log(x.var(2).mean(1))
        k = kurtosis(x, axis=2)
        f3 = np.where(np.isfinite(k).any(1), np.nanmax(np.where(np.isfinite(k), k, -np.inf), 1), np.nan)
    return np.stack([f1, f2, f3], 1)


def check_mapping(ctx, datasets, out):
    """G0 without diverged/logits/score, then G-art on every admitted subject, and the
    per-trial features (keyed by (session, test_idx), sorted). Prints no score."""
    import adapt_B
    os.makedirs(out, exist_ok=True)
    res = {"datasets": {}}
    for ds in datasets:
        if os.path.exists(f"{out}/{ds}.json") or os.path.exists(f"{out}/{ds}"):
            raise SystemExit(f"REFUSED: {out}/{ds} already exists; a mapping is never overwritten")
        adm, excl = population(ctx, ds, MODELS, SEEDS, full=False)
        os.makedirs(f"{out}/{ds}", exist_ok=True)
        subj_res = {}
        for subj, D in adm.items():
            X, y_get, sess, run = adapt_B.subject_data(ds, subj, ctx.cells_path)
            r = gart_subject(D, X, y_get, sess, run, ctx.man[ds]["subjects"][subj]["n"])
            subj_res[subj] = r
            if r["ok"]:
                keys = next(iter(D.values()))["keys"]
                se = np.array([k[0] for k in keys])
                ti = np.array([k[1] for k in keys], np.int64)
                np.savez_compressed(f"{out}/{ds}/{subj}.npz", session=se, test_idx=ti,
                                    feats=features(X[ti]), names=np.array(FEATURES))
            print(ds, subj, "G-art", "ok" if r["ok"] else "ECHEC",
                  {k: r[k] for k in ("n_win", "n_manifest", "n_trials", "n_npz", "label_bad",
                                     "session_bad", "run_bad", "fp_bad", "idx_out")}, flush=True)
        n_ok = sum(r["ok"] for r in subj_res.values())
        res["datasets"][ds] = {"n_population": len(ctx.subjects(ds, MODELS)), "n_g0_structural": len(adm),
                               "excluded": excl, "n_gart_ok": n_ok, "subjects": subj_res}
        print(f"== {ds}: population {len(ctx.subjects(ds, MODELS))}, G0 structurelle {len(adm)}, "
              f"G-art ok {n_ok}/{len(adm)}", flush=True)
        for k, v in excl.items():
            print(f"   exclus [{k}] {len(v)} : " + "; ".join(f"{s} ({d})" for s, d in v), flush=True)
        tmp = f"{out}/{ds}.json.tmp"
        json.dump(res["datasets"][ds], open(tmp, "w"), indent=1)
        os.replace(tmp, f"{out}/{ds}.json")
    return res


def artefact(ds, adm, mapping, mapdir):
    """AUC_art = max over the 3 features of max(AUC, 1 - AUC), 'missed by all 12' against
    'right for all 12', features rank-normalised within subject (over all its trials).
    NaN (with the reason) if G-art is not established for an admitted subject or < 30 trials
    are missed by all."""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    M = (mapping or {}).get("datasets", {}).get(ds)
    if M is None:
        return float("nan"), {}, "check-mapping absent pour ce dataset"
    lab, F = [], []
    for subj, D in adm.items():
        e = M["subjects"].get(subj)
        if not e or not e.get("ok") or not os.path.exists(f"{mapdir}/{ds}/{subj}.npz"):
            return float("nan"), {}, f"G-art non établie pour le sujet {subj}"
        f = np.load(f"{mapdir}/{ds}/{subj}.npz")
        keys = next(iter(D.values()))["keys"]
        if tuple(zip(f["session"].astype(str).tolist(), f["test_idx"].tolist())) != keys:
            return float("nan"), {}, f"essais du check-mapping != essais lus pour le sujet {subj}"
        R = np.stack([rankdata(v, nan_policy="omit") / np.isfinite(v).sum() for v in f["feats"].T], 1)
        c = np.stack([d["correct"] for d in D.values()], 1)
        aw, ar = ~c.any(1), c.all(1)
        lab.append(aw[aw | ar])
        F.append(R[aw | ar])
    lab, F = np.concatenate(lab), np.concatenate(F)
    info = {"n_all_wrong": int(lab.sum()), "n_all_right": int((~lab).sum())}
    if lab.sum() < MIN_ALL_WRONG or (~lab).sum() == 0:
        return float("nan"), info, f"{int(lab.sum())} essais ratés par tous (< {MIN_ALL_WRONG})"
    for j, name in enumerate(FEATURES):
        ok = np.isfinite(F[:, j])
        a = float(roc_auc_score(lab[ok], F[ok, j]))
        info[name] = a
    return float(max(max(info[n], 1 - info[n]) for n in FEATURES)), info, ""


# ---------------------------------------------------------------- read

def g1_failures(*pops):
    """(n npz, [(file, recomputed, stored)] that differ by more than 1e-6)."""
    n, bad = 0, []
    for adm in pops:
        for D in adm.values():
            for d in D.values():
                for f, s, st in d["g1"]:
                    n += 1
                    if not abs(s - st) <= G1_TOL:
                        bad.append((f, s, st))
    return n, bad


def g1_check(*pops):
    """G1 on the PRIMARY: one failure refuses the read."""
    n, bad = g1_failures(*pops)
    if bad:
        raise SystemExit(f"REFUSED: G1, {len(bad)} of {n} npz do not reproduce their score, e.g. {bad[:3]}")
    return n


def subject_strata(D, kind):
    d0 = next(iter(D.values()))
    return strata_of(d0["y"], [k[0] for k in d0["keys"]], kind)


def dataset_stats(adm, kind="classe_x_session", models=MODELS, seeds=SEEDS):
    """Per-subject kappas on the strata valid for all the vectors of (models x seeds).
    -> (rows, subjects without a valid stratum, drop summary)."""
    rows, drop, qs, n_tr = [], [], [], 0
    for subj, D in adm.items():
        C = {(m, s): D[m, s]["correct"] for m in models for s in seeds}
        q = subject_kappas(C, subject_strata(D, kind), models, seeds)
        q["subj"] = subj
        q["acc"] = float(np.mean([c.mean() for c in C.values()]))
        q["n"] = int(len(next(iter(C.values()))))
        qs.append(q)
        n_tr += q["n"]
        ok = q["valid"] and np.isfinite(q["ks"]) and (len(models) < 2 or np.isfinite(q["kx"]))
        (rows if ok else drop).append(q)
    return rows, [q["subj"] for q in drop], drop_summary(qs, n_tr)


def _subj_json(q):
    return {"subj": q["subj"], "n": q["n"], "acc": q["acc"], "kx": q["kx"], "ks": q["ks"], "ksame": q["ksame"],
            "seed": q["seed"], "x": {f"{m}|{n}": v for (m, n), v in q["x"].items()},
            "same": {f"{m}|{n}": v for (m, n), v in q["same"].items()}, "dropped_strata": q["dropped"]}


def fmt(v, d=3):
    return "NaN" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def read(ctx, mapdir, out):
    from scipy.stats import binom, spearmanr
    if os.path.exists(f"{out}/EC_pilot.json"):
        raise SystemExit(f"REFUSED: {out}/EC_pilot.json already exists; a read is never overwritten")
    # one file per dataset (--check-mapping); a missing one gives AUC_art = NaN, never GARDE
    mapping = {"datasets": {ds: json.load(open(f"{mapdir}/{ds}.json"))
                            for ds in DATASETS if os.path.exists(f"{mapdir}/{ds}.json")}}
    pops, excls = {}, {}
    for ds in DATASETS:
        pops[ds], excls[ds] = population(ctx, ds, MODELS, SEEDS, full=True)
        refuse_small(ds, pops[ds], excls[ds])
    # secondaries read the same way, so that G1 covers every npz this script opens
    phys, phys_excl = population(ctx, "physionetmi", ("eegnex",), SEEDS, full=True)
    shallow, shallow_excl = {}, collections.defaultdict(list)
    for ds in DATASETS:
        shallow[ds] = {}
        for subj, D in pops[ds].items():
            stem = ctx.cells.get((ds, "shallow", 0, subj))
            d, why = (None, ("cellule_absente", "")) if stem is None else load_couple(
                stem, ctx.n_expected(ds, subj), subj, ctx.n_classes(ds), True)
            if d is None:
                shallow_excl[f"{ds}:{why[0]}"].append(subj)
            elif d["diverged"] or d["keys"] != D[MODELS[0], 0]["keys"] or not np.array_equal(d["y"], D[MODELS[0], 0]["y"]) \
                    or d["fps"] != D[MODELS[0], 0]["fps"]:
                shallow_excl[f"{ds}:incompatible"].append(subj)
            else:
                shallow[ds][subj] = d
    # G1: the primary refuses; a failing secondary is excluded with its motif
    n_g1 = g1_check(*pops.values())
    sec_off = {}
    sh_as_pop = {f"{ds}:{s}": {("shallow", 0): d} for ds in DATASETS for s, d in shallow[ds].items()}
    for name, pop in (("shallow_s0", sh_as_pop), ("physionetmi_eegnex", phys)):
        n_s, bad = g1_failures(pop)
        if bad:
            sec_off[name] = f"G1 : {len(bad)} npz sur {n_s} ne reproduisent pas leur score, ex. {bad[:2]}"
    R = {"gates": {"G1_npz_primary": n_g1}, "datasets": {}, "secondary": {"excluded": sec_off}}
    L = ["# Pilote EC — les architectures se trompent-elles sur les mêmes essais ?\n",
         f"Note pré-enregistrée : `analysis/PREREG_EC_pilot.md`. G1 (primaire) : {n_g1} npz, 0 échec.\n",
         "Strates du primaire : vraie classe × session ; une strate où l'un des 12 vecteurs est constant est "
         "écartée pour tous les couples.\n",
         "| dataset | sujets | exclus (motif) | K_x | K_seed | R [IC 95 %] | med_x | AUC_art | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    verdicts, arts, kx_g, ks_g = [], [], {}, {}
    for ds in DATASETS:
        rows, drop, dsum = dataset_stats(pops[ds])
        if len(rows) < MIN_SUBJ:
            raise SystemExit(f"REFUSED: {ds} keeps {len(rows)} subjects with a valid stratum and a finite "
                             f"kappa (< {MIN_SUBJ})")
        kx = np.array([q["kx"] for q in rows])
        ks = np.array([q["ks"] for q in rows])
        kx_g[ds], ks_g[ds] = kx, ks
        rb = ratio_boot(kx, ks)
        art, ainfo, awhy = artefact(ds, {q["subj"]: pops[ds][q["subj"]] for q in rows}, mapping, mapdir)
        v = decide(rb["med_x"], rb["R"], rb["lo"], rb["hi"], art)
        verdicts.append(v)
        arts.append(art)
        ex = {k: len(x) for k, x in excls[ds].items()}
        R["datasets"][ds] = {"n_g0": len(pops[ds]), "excluded": excls[ds], "no_valid_stratum_subjects": drop,
                             "strata": dsum, **rb, "AUC_art": art, "art": ainfo, "art_nan_reason": awhy,
                             "verdict": v, "subjects": [_subj_json(q) for q in rows]}
        L.append(f"| {ds} | {len(rows)} | {ex} | {fmt(rb['K_x'])} | {fmt(rb['K_seed'])} | "
                 f"{fmt(rb['R'])} [{fmt(rb['lo'])}, {fmt(rb['hi'])}] | {fmt(rb['med_x'])} | "
                 f"{fmt(art)}{' (' + awhy + ')' if awhy else ''} | **{v}** |")
    g = decide_global(verdicts, arts)
    R["verdict"] = g
    from e1b_ablation import sboot          # imported unchanged (e1b_ablation.py:479-497)
    pooled = sboot_ratio(kx_g, ks_g)
    R["pooled"] = {**pooled, "sboot_K_x": sboot(kx_g), "sboot_K_seed": sboot(ks_g)}
    L += ["", f"**Verdict global (règle gelée) : {g}.**\n",
          f"Groupement √n (ne décide rien) : R = {fmt(pooled['R'])} [{fmt(pooled['lo'])}, {fmt(pooled['hi'])}], "
          f"K_x = {fmt(pooled['K_x'])}, K_seed = {fmt(pooled['K_seed'])}.\n",
          "## Exclusions G0 par motif\n"]
    for ds in DATASETS:
        for k, x in excls[ds].items():
            L.append(f"- {ds} [{k}] {len(x)} : " + "; ".join(f"{s} ({d})" for s, d in x))
        st = R["datasets"][ds]["strata"]
        L.append(f"- {ds} : strates écartées {st['n_dropped']} / {st['n_strata']}, part d'essais perdue "
                 f"{fmt(100 * st['trials_lost_share'], 1)} % ; par modèle responsable : "
                 + (", ".join(f"{m} {v['strata']} strates ({fmt(100 * v['trials_share'], 1)} %)"
                              for m, v in st["by_model"].items()) or "aucun")
                 + f" ; sujets sans strate valide : {R['datasets'][ds]['no_valid_stratum_subjects']}")
        L.append(f"- {ds} : artefact {R['datasets'][ds]['art']}")
    L += ["", "## Secondaires (ne décident rien)\n"]
    for ds in DATASETS:
        rows, _, _ = dataset_stats(pops[ds])
        kx = np.array([q["kx"] for q in rows])
        ksame = np.array([q["ksame"] for q in rows])
        acc = np.array([q["acc"] for q in rows])
        ks = np.array([q["ks"] for q in rows])
        dsame = mean_boot(ksame - kx)
        pair = {}
        for m, n in itertools.combinations(MODELS, 2):
            num = np.nanmean([q["x"][m, n] for q in rows])
            den = np.mean([np.nanmean([q["seed"][m] for q in rows]), np.nanmean([q["seed"][n] for q in rows])])
            pair[f"{m}|{n}"] = float(num / den)
        alt = {}
        for kind in ("classe", "aucune"):
            a_rows, a_drop, a_sum = dataset_stats(pops[ds], kind)
            alt[kind] = {**(ratio_boot([q["kx"] for q in a_rows], [q["ks"] for q in a_rows]) if a_rows else {}),
                         "no_valid_stratum_subjects": a_drop, "strata": a_sum}
        thr = np.array([binom.ppf(0.95, q["n"], 0.5) / q["n"] for q in rows])
        near = acc < thr
        sp_x, sp_s = spearmanr(acc, kx), spearmanr(acc, ks)
        n_diff = sum(d["n_argmax_diff"] for D in pops[ds].values() for d in D.values())
        n_tie = sum(d["n_tie"] for D in pops[ds].values() for d in D.values())
        sec = {"kappa_same": mean_boot(ksame), "same_minus_x": dsame, "R_pair": pair,
               "class_only": alt["classe"], "unconditional": alt["aucune"],
               "spearman_acc_kx": [float(sp_x.statistic), float(sp_x.pvalue)],
               "spearman_acc_kseed": [float(sp_s.statistic), float(sp_s.pvalue)],
               "near_chance_share": float(near.mean()),
               "near_chance_kx": _nanmean(kx[near]), "other_kx": _nanmean(kx[~near]),
               "near_chance_kseed": _nanmean(ks[near]), "other_kseed": _nanmean(ks[~near]),
               "argmax_logit_vs_softmax_diff": n_diff, "softmax_argmax_not_unique": n_tie}
        R["secondary"][ds] = sec
        L += [f"### {ds}\n",
              f"- κ_same = {fmt(sec['kappa_same']['mean'])} ; κ_same − κ_x = {fmt(dsame['mean'])} "
              f"[{fmt(dsame['lo'])}, {fmt(dsame['hi'])}] (effet du pli partagé)",
              "- R par paire : " + " ; ".join(f"{k} {fmt(v)}" for k, v in pair.items()),
              *[f"- κ strates « {kind} » : K_x {fmt(a.get('K_x'))}, K_seed {fmt(a.get('K_seed'))}, "
                f"R {fmt(a.get('R'))} [{fmt(a.get('lo'))}, {fmt(a.get('hi'))}], med_x {fmt(a.get('med_x'))} ; "
                f"strates écartées {a['strata']['n_dropped']}/{a['strata']['n_strata']} "
                f"({fmt(100 * a['strata']['trials_lost_share'], 1)} % des essais)"
                for kind, a in alt.items()],
              f"- Spearman(accuracy du sujet, κ_x) = {fmt(sp_x.statistic)} (p {fmt(sp_x.pvalue, 3)}) ; "
              f"(accuracy, κ_seed) = {fmt(sp_s.statistic)} (p {fmt(sp_s.pvalue, 3)})",
              f"- sujets proches du hasard (accuracy moyenne < quantile 95 % binomial à 0,5) : "
              f"{fmt(100 * near.mean(), 1)} % ; κ_x {fmt(sec['near_chance_kx'])} contre {fmt(sec['other_kx'])} ; "
              f"κ_seed {fmt(sec['near_chance_kseed'])} contre {fmt(sec['other_kseed'])}",
              f"- essais où l'argmax des logits ≠ l'argmax du softmax float32 : {n_diff} ; "
              f"argmax du softmax non unique (p0 == p1) : {n_tie}", ""]
        # Shallow, seed 0: kappa_same with the four models at seed 0, against the four among
        # themselves, on the strata valid for the 5 vectors (shallow s0 + the 4 at s0)
        if "shallow_s0" in sec_off:
            L.append(f"- Shallow graine 0 : secondaire EXCLU ({sec_off['shallow_s0']})\n")
            continue
        a, b, qs, n_tr, nov = [], [], [], 0, []
        for subj, d in shallow[ds].items():
            D = pops[ds][subj]
            V = {("shallow", 0): d["correct"], **{(m, 0): D[m, 0]["correct"] for m in MODELS}}
            st = subject_strata(D, "classe_x_session")
            masks, dropped = valid_strata(V, st)
            qs.append({"n_strata": int(len(np.unique(st))), "dropped": dropped})
            n_tr += len(st)
            if not masks:
                nov.append(subj)
                continue
            a.append(_nanmean([kappa_strat(V["shallow", 0], V[m, 0], masks) for m in MODELS]))
            b.append(_nanmean([kappa_strat(V[m, 0], V[n, 0], masks) for m, n in itertools.combinations(MODELS, 2)]))
        R["secondary"][ds]["shallow_s0"] = {"n": len(a), "kappa_same_shallow": _nanmean(a), "kappa_same_4": _nanmean(b),
                                            "no_valid_stratum_subjects": nov, "strata": drop_summary(qs, n_tr)}
        L.append(f"- Shallow graine 0 ({len(a)} sujets) : κ_same(shallow, autres) = {fmt(_nanmean(a))} ; "
                 f"κ_same entre les 4 à la graine 0 = {fmt(_nanmean(b))} ; strates "
                 f"{R['secondary'][ds]['shallow_s0']['strata']} ; sans strate valide {nov}\n")
    L.append(f"- exclusions Shallow : { {k: len(v) for k, v in shallow_excl.items()} }")
    if "physionetmi_eegnex" in sec_off:
        R["secondary"]["physionetmi_eegnex"] = {"excluded_secondary": sec_off["physionetmi_eegnex"]}
        L.append(f"- physionetmi, eegnex : secondaire EXCLU ({sec_off['physionetmi_eegnex']})")
    else:
        p_rows, p_nov, p_sum = dataset_stats(phys, "classe_x_session", models=("eegnex",))
        ps = [q["seed"]["eegnex"] for q in p_rows]
        mb = mean_boot(ps)
        R["secondary"]["physionetmi_eegnex"] = {"n": len(ps), "excluded": phys_excl, "kappa_seed": mb,
                                                "median": float(np.nanmedian(ps)) if ps else None,
                                                "no_valid_stratum_subjects": p_nov, "strata": p_sum}
        L.append(f"- physionetmi, eegnex, plafond de graine seul : {len(ps)} sujets, κ_seed = "
                 + (f"{fmt(mb['mean'])} [{fmt(mb['lo'])}, {fmt(mb['hi'])}]" if mb else "NaN")
                 + f" ; exclus G0 { {k: len(v) for k, v in phys_excl.items()} } ; sans strate valide {len(p_nov)} ; "
                 f"strates écartées {p_sum['n_dropped']}/{p_sum['n_strata']}")
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(f"{out}/EC_pilot.json", "w"), indent=1, default=str)
    open(f"{out}/EC_pilot.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- selftest

def _expit(x):
    return 1 / (1 + np.exp(-x))


def _simulate(rng, n_subj, n_tr, arch_sd):
    """Correctness = Bernoulli(expit(1 + d_i + g_{m,i} + e_{m,s,i})): d = trial difficulty
    shared by all, g = architecture x trial term shared by the seeds of m, e = seed noise."""
    kx, ks = [], []
    for _ in range(n_subj):
        y = np.repeat([0, 1], n_tr // 2)
        d = rng.normal(0, 1.5, n_tr)
        g = {m: rng.normal(0, arch_sd, n_tr) for m in MODELS}
        C = {(m, s): rng.random(n_tr) < _expit(1.0 + d + g[m] + rng.normal(0, 0.5, n_tr))
             for m in MODELS for s in SEEDS}
        q = subject_kappas(C, strata_of(y, np.zeros(n_tr, int)), MODELS, SEEDS)
        kx.append(q["kx"])
        ks.append(q["ks"])
    return ratio_boot(kx, ks)


def _old_rule_R(Cs, ys):
    """The rule before the audit, kept for the selftest only: per PAIR, a class whose
    c_exp = 1 for that pair is dropped for that pair alone."""
    def kc(a, b, y):
        ks, ws = [], []
        for c in np.unique(y):
            m = y == c
            k = kappa(a[m], b[m])
            if np.isfinite(k):
                ks.append(k)
                ws.append(m.sum())
        return np.average(ks, weights=ws) if ks else np.nan
    kx, ks = [], []
    for C, y in zip(Cs, ys):
        ks.append(np.nanmean([kc(C[m, s], C[m, t], y) for m in MODELS for s, t in itertools.combinations(SEEDS, 2)]))
        kx.append(np.nanmean([kc(C[m, s], C[n, t], y) for m, n in itertools.combinations(MODELS, 2)
                              for s in SEEDS for t in SEEDS if s != t]))
    return float(np.mean(kx) / np.mean(ks))


def _make_archive(root, rng, tamper=None, n_tr=40):
    """A fake campaign-A tree for one subject of a fake 2-class dataset 'synth', 12 couples,
    written the way subject_stamp.LogitScorer writes them (subject_stamp.py:182-190)."""
    from sklearn.model_selection import StratifiedKFold
    import shutil
    root = f"{root}_{tamper or 'propre'}"
    if os.path.exists(root):
        raise SystemExit(f"REFUSED: {root} exists; the selftest writes into a fresh --out, deletes nothing")
    os.makedirs(f"{root}/manifest")
    X = rng.normal(0, 1, (n_tr, 3, 32)).astype(np.float32)
    y = np.repeat([0, 1], n_tr // 2).astype(np.int64)
    sess, run = np.array(["0"] * n_tr), np.array(["0"] * n_tr)
    json.dump({"classes": ["a", "b"], "subjects": {"1": {"n": n_tr, "sessions": {"0": {"n": n_tr}}}}},
              open(f"{root}/manifest/synth.json", "w"))
    lines = []
    for i, (m, s) in enumerate((m, s) for m in MODELS for s in SEEDS):
        rdir = f"{root}/{m}__bd/results"
        stem = f"{rdir}/{PROT}/synth/bd_{m}__seed{s}__subj1"
        lines.append(f"{i} {PROT} synth {m} {ARM} {s} 1 {rdir} {stem}.csv")
        os.makedirs(stem + "__fits__logits")
        with open(stem + "__fits.jsonl", "w") as fh:
            for f, (_, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(X, y)):
                uid = f"{m}{s}f{f}"
                fh.write(json.dumps({"fit_uid": uid, "subject": "1"}) + "\n")
                lg = rng.normal(0, 2, (len(te), 2)).astype(np.float32)
                if tamper == "logits_nan" and (m, s, f) == ("msvtnet", 2, 3):
                    lg[0, 0] = np.nan
                yy = y[te].copy()
                if tamper == "y" and (m, s, f) == ("msvtnet", 1, 0):
                    yy[0] = 1 - yy[0]
                fp = fingerprint(X[te], y[te])
                if tamper == "empreinte" and (m, s, f) == ("ctnet", 0, 2):
                    fp = "0" * 40
                sc = score_of(lg, yy, 2) + (1e-3 if tamper == "score" and (m, s, f) == ("eegnex", 2, 1) else 0)
                np.savez_compressed(f"{stem}__fits__logits/{uid}.npz", logits=lg, y=yy, test_idx=te.astype(np.int64),
                                    subject=np.array(["1"] * len(te)), session=sess[te], run=run[te],
                                    test_fingerprint=np.array(fp), score=np.array(float(sc)),
                                    diverged=np.array(tamper == "diverge" and (m, s, f) == ("conformer", 0, 0)))
                if tamper in ("fantome", "fantome_ligne") and (m, s, f) == ("eegnex", 1, 0):
                    shutil.copy(f"{stem}__fits__logits/{uid}.npz", f"{stem}__fits__logits/{uid}x.npz")
                    if tamper == "fantome_ligne":
                        fh.write(json.dumps({"fit_uid": uid + "x", "subject": "1"}) + "\n")
    open(f"{root}/cells.txt", "w").write("\n".join(lines) + "\n")
    return Ctx(f"{root}/cells.txt", f"{root}/manifest"), X, y, sess, run


def selftest(out):
    rng = np.random.default_rng(1)
    n = 0
    # 1. two identical copies: kappa = 1
    a = rng.random(500) < 0.7
    y = np.repeat([0, 1], 250)
    assert kappa(a, a) == 1.0 and abs(kappa_cond(a, a, y)[0] - 1.0) < 1e-12
    n += 1
    # 2. two independent vectors: kappa ~ 0
    a, b = rng.random(200_000) < 0.7, rng.random(200_000) < 0.7
    assert abs(kappa(a, b)) < 0.01 and abs(kappa_cond(a, b, np.repeat([0, 1], 100_000))[0]) < 0.01
    n += 1
    # 3. shared difficulty: one difficulty per trial drives both models -> kappa > 0
    d = rng.normal(0, 1.5, 20_000)
    a, b = rng.random(20_000) < _expit(0.5 + 2 * d), rng.random(20_000) < _expit(0.5 + 2 * d)
    assert kappa_cond(a, b, np.repeat([0, 1], 10_000))[0] > 0.2
    n += 1
    # 4. class bias only: both good on class 0, bad on class 1, independent within class
    y = np.repeat([0, 1], 20_000)
    p = np.where(y == 0, 0.9, 0.3)
    a, b = rng.random(40_000) < p, rng.random(40_000) < p
    assert kappa(a, b) > 0.2 and abs(kappa_cond(a, b, y)[0]) < 0.02
    n += 1
    # 4b. degenerate stratum: ONE constant vector drops the stratum for EVERY couple, the
    # responsible model is named, and a subject without a valid stratum is left out
    assert np.isnan(kappa(np.ones(5, bool), np.ones(5, bool)))
    yy, ss = np.repeat([0, 1], 40), np.tile(np.repeat(["0", "1"], 20), 2)
    st = strata_of(yy, ss)
    assert len(np.unique(st)) == 4 and len(np.unique(strata_of(yy, ss, "classe"))) == 2
    C = {(m, s): rng.random(80) < 0.6 for m in MODELS for s in SEEDS}
    C["ctnet", 2][(yy == 1) & (ss == "0")] = True                      # ctnet s2 perfect on one stratum
    q = subject_kappas(C, st, MODELS, SEEDS)
    assert q["valid"] and len(q["dropped"]) == 1 and q["dropped"][0][1:] == (20, ["ctnet"])
    keep = st != "1|0"
    assert abs(q["seed"]["eegnex"] - _nanmean([kappa_strat(C["eegnex", s], C["eegnex", t],
                                                            [m & keep for m in [st == u for u in np.unique(st)] if (m & keep).any()])
                                               for s, t in itertools.combinations(SEEDS, 2)])) < 1e-12
    dsum = drop_summary([q], 80)
    assert dsum["n_dropped"] == 1 and dsum["by_model"] == {"ctnet": {"strata": 1, "trials_share": 0.25}}
    Cbad = {k: np.ones(80, bool) for k in C}
    qb = subject_kappas(Cbad, st, MODELS, SEEDS)
    assert not qb["valid"] and np.isnan(qb["kx"]) and len(qb["dropped"]) == 4
    n += 1
    # 4c. ties of the float32 softmax are counted (p0 == p1: numpy takes column 0)
    c, nd, nt = correct_of(np.array([[0.0, 1e-9], [3.0, 0.0]], np.float32), np.array([1, 0]))
    assert nt == 1 and nd == 1 and c.tolist() == [False, True]
    n += 1
    # 5. R ~ 1 when the architecture has no effect; 6. R < 1 when it has one
    r0 = _simulate(np.random.default_rng(2), 40, 200, 0.0)
    r1 = _simulate(np.random.default_rng(3), 40, 200, 1.5)
    print(f"  R sans effet d'architecture : {r0['R']:.3f} [{r0['lo']:.3f}, {r0['hi']:.3f}] "
          f"(K_x {r0['K_x']:.3f}, K_seed {r0['K_seed']:.3f})")
    print(f"  R avec effet d'architecture : {r1['R']:.3f} [{r1['lo']:.3f}, {r1['hi']:.3f}] "
          f"(K_x {r1['K_x']:.3f}, K_seed {r1['K_seed']:.3f})")
    assert abs(r0["R"] - 1) < 0.05 and r0["lo"] < 1 < r0["hi"]
    n += 1
    assert r1["R"] < 0.8 and r1["hi"] < 0.9
    n += 1
    # 6a. the auditor's scenario: one model perfect on one class under its 3 seeds, no
    # architecture effect elsewhere. Strata dropped for every couple -> R ~ 1; the old
    # per-pair rule pulls R down (kappa = 0 on that class for the cross pairs only)
    srng = np.random.default_rng(6)
    kx, ks, Cs, ys = [], [], [], []
    for _ in range(40):
        y = np.repeat([0, 1], 100)
        d = srng.normal(0, 1.5, 200)
        C = {(m, s): srng.random(200) < _expit(1.0 + d + srng.normal(0, 0.5, 200)) for m in MODELS for s in SEEDS}
        for s in SEEDS:
            C["eegnex", s] = C["eegnex", s] | (y == 0)
        q = subject_kappas(C, strata_of(y, np.zeros(200, int)), MODELS, SEEDS)
        kx.append(q["kx"])
        ks.append(q["ks"])
        Cs.append(C)
        ys.append(y)
    ra, r_old = ratio_boot(kx, ks), _old_rule_R(Cs, ys)
    print(f"  scénario de l'auditeur (eegnex parfait sur une classe) : R = {ra['R']:.3f} "
          f"[{ra['lo']:.3f}, {ra['hi']:.3f}] ; ancienne règle par paire : R = {r_old:.3f}")
    assert abs(ra["R"] - 1) < 0.05 and ra["lo"] < 1 < ra["hi"] and r_old < 0.9
    n += 1
    # 6b. one session harder for every model, no shared trial difficulty: class-only strata
    # see a positive kappa, the primary class x session strata do not
    srng = np.random.default_rng(7)
    k_cls, k_cs = [], []
    for _ in range(30):
        y = np.tile(np.repeat([0, 1], 50), 2)
        se = np.repeat(["0", "1"], 100)
        p = _expit(np.where(se == "1", -1.0, 1.5))
        C = {(m, s): srng.random(200) < p for m in MODELS for s in SEEDS}
        k_cls.append(subject_kappas(C, strata_of(y, se, "classe"), MODELS, SEEDS)["kx"])
        k_cs.append(subject_kappas(C, strata_of(y, se), MODELS, SEEDS)["kx"])
    print(f"  session plus dure pour tous : K_x strates classe = {np.mean(k_cls):.3f} ; "
          f"classe × session = {np.mean(k_cs):+.3f}")
    assert np.mean(k_cls) > 0.1 and abs(np.mean(k_cs)) < 0.03
    n += 1
    # 7. decision rule
    assert decide(0.35, 0.9, 0.8, 1.0, 0.6) == "GARDE"
    assert decide(0.35, 0.9, 0.8, 1.0, float("nan")) == "NON CONCLUANT"     # NaN artefact blocks GARDE
    assert decide(0.35, 0.9, 0.8, 1.0, 0.72) == "TUE"
    assert decide(0.10, 0.9, 0.8, 1.0, 0.5) == "TUE"
    assert decide(0.25, 0.55, 0.4, 0.70, 0.5) == "TUE"
    assert decide(0.25, 0.55, 0.4, 0.80, 0.5) == "NON CONCLUANT"
    assert decide_global(["GARDE", "GARDE"], [0.5, 0.6]) == "GARDE"
    assert decide_global(["GARDE", "NON CONCLUANT"], [0.5, 0.71]) == "TUE"
    assert decide_global(["TUE", "NON CONCLUANT"], [0.5, 0.6]) == "NON CONCLUANT"
    n += 1
    # 8. gates on a synthetic archive
    root = f"{out}/synth_archive"
    couples = [(m, s) for m in MODELS for s in SEEDS]
    clean = _make_archive(root, np.random.default_rng(4))
    ctx, X, y, sess, run = clean
    ok, D = g0_subject(ctx, "synth", "1", couples, True)
    assert ok, D
    assert g1_check({"1": D}) == 12 * N_FOLDS
    r = gart_subject(D, X, y, sess, run, len(X))
    assert r["ok"], r
    ok, Ds = g0_subject(ctx, "synth", "1", couples, False)
    assert ok and "correct" not in Ds[MODELS[0], 0]
    try:
        refuse_small("synth", {"1": D}, {})
        raise AssertionError("refuse_small did not refuse")
    except SystemExit as e:
        assert str(e).startswith("REFUSED")
    n += 1
    for tamper, motif in (("fantome", "compte"), ("fantome_ligne", "compte"), ("y", "y_discordant"),
                          ("diverge", "diverge"), ("empreinte", "empreintes_modeles"),
                          ("logits_nan", "logits_non_finis")):
        ctx, *_ = _make_archive(root, np.random.default_rng(4), tamper)
        ok, why = g0_subject(ctx, "synth", "1", couples, True)
        assert not ok and why[0] == motif, (tamper, why)
        print(f"  porte G0 refuse [{tamper}] : {why}")
        n += 1
    ctx, *_ = _make_archive(root, np.random.default_rng(4), "score")
    ok, D = g0_subject(ctx, "synth", "1", couples, True)
    try:
        g1_check({"1": D})
        raise AssertionError("G1 did not refuse")
    except SystemExit as e:
        assert str(e).startswith("REFUSED: G1, 1 of")
        print(f"  porte G1 refuse : {str(e)[:60]}...")
    n_s, bad = g1_failures({"1": D})          # a secondary: counted, never raises
    assert n_s == 12 * N_FOLDS and len(bad) == 1
    n += 1
    # --read never overwrites a read
    os.makedirs(f"{out}/read_existant")
    open(f"{out}/read_existant/EC_pilot.json", "w").write("{}")
    try:
        read(None, None, f"{out}/read_existant")
        raise AssertionError("read did not refuse")
    except SystemExit as e:
        assert "already exists" in str(e)
    n += 1
    # 9. G-art: a within-class permutation of the windows keeps every label but not the
    # signal (the fingerprint catches it); a wrong label is caught too
    ctx, X, y, sess, run = clean
    ok, D = g0_subject(ctx, "synth", "1", couples, True)
    perm = np.arange(len(y))
    for c in (0, 1):
        i = np.flatnonzero(y == c)
        perm[i] = rng.permutation(i)
    r = gart_subject(D, X[perm], y, sess, run, len(X))
    assert not r["ok"] and r["label_bad"] == 0 and r["fp_bad"] > 0, r
    y2 = y.copy()
    y2[0] = 1 - y2[0]
    r = gart_subject(D, X, y2, sess, run, len(X))
    assert not r["ok"] and r["label_bad"] > 0
    r = gart_subject(D, X, y, sess, run, len(X) + 1)
    assert not r["ok"] and not r["count_ok"]
    n += 1
    # 10. features, then artefact(): trials missed by all 12 that carry a large amplitude
    # give AUC_art ~ 1; missed trials drawn at random give ~ 0.5; < 30 missed -> NaN;
    # a subject without G-art -> NaN
    F = features(X)
    assert F.shape == (len(X), 3) and np.isfinite(F).all()
    Xa = X.copy()
    Xa[:5] *= 20
    assert (features(Xa)[:5, 0] > F[5:, 0].max()).all()
    n += 1
    arng = np.random.default_rng(5)

    def art_case(tag, n_subj, n_tr, p_wrong, loud):
        mdir = f"{out}/art_{tag}"
        os.makedirs(f"{mdir}/synth")
        adm, M = {}, {"datasets": {"synth": {"subjects": {}}}}
        for k in range(n_subj):
            s = str(k + 1)
            keys = tuple(("0", i) for i in range(n_tr))
            aw = arng.random(n_tr) < p_wrong                      # missed by all 12
            ar = ~aw & (arng.random(n_tr) < 0.5)                  # right for all 12
            Xs = arng.normal(0, 1, (n_tr, 3, 32))
            if loud:
                Xs[aw] *= 5
            adm[s] = {c: {"keys": keys, "correct": np.where(aw, False, np.where(ar, True, arng.random(n_tr) < 0.5))}
                      for c in couples}
            adm[s][couples[0]]["correct"] = np.where(aw, False, np.where(ar, True, False))
            adm[s][couples[1]]["correct"] = np.where(aw, False, np.where(ar, True, True))
            np.savez_compressed(f"{mdir}/synth/{s}.npz", session=np.array(["0"] * n_tr),
                                test_idx=np.arange(n_tr), feats=features(Xs), names=np.array(FEATURES))
            M["datasets"]["synth"]["subjects"][s] = {"ok": True}
        return adm, M, mdir
    adm, M, mdir = art_case("bruyant", 10, 60, 0.15, True)
    a_loud, info, why = artefact("synth", adm, M, mdir)
    adm, M, mdir = art_case("aleatoire", 10, 60, 0.15, False)
    a_rand, _, _ = artefact("synth", adm, M, mdir)
    print(f"  AUC_art amplitude liée aux ratés : {a_loud:.3f} ({info['n_all_wrong']} ratés par tous) ; "
          f"ratés au hasard : {a_rand:.3f}")
    assert a_loud > 0.9 and a_rand < 0.65 and not why
    adm, M, mdir = art_case("peu", 2, 60, 0.15, True)
    a_few, _, why = artefact("synth", adm, M, mdir)
    assert np.isnan(a_few) and "ratés par tous" in why
    M["datasets"]["synth"]["subjects"]["1"]["ok"] = False
    a_no, _, why = artefact("synth", adm, M, mdir)
    assert np.isnan(a_no) and "G-art" in why
    n += 1
    print(f"selftest : {n} contrôles passés")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--check-mapping", action="store_true")
    g.add_argument("--read", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells")
    ap.add_argument("--manifest")
    ap.add_argument("--mapping")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    if a.selftest:
        os.makedirs(a.out, exist_ok=True)
        selftest(a.out)
        return
    if not (a.cells and a.manifest):
        ap.error("--cells and --manifest are required")
    ctx = Ctx(a.cells, a.manifest)
    if a.check_mapping:
        bad = [d for d in a.datasets if d not in DATASETS]
        if bad:
            ap.error(f"--datasets must be within {DATASETS}")
        check_mapping(ctx, a.datasets, a.out)
    else:
        if not a.mapping:
            ap.error("--read needs --mapping (the --check-mapping output)")
        read(ctx, a.mapping, a.out)


if __name__ == "__main__":
    main()
