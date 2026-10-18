"""Read the étape-1 select_best_update measurement (grow_msvtnet_all, bnci2014_001).

EXPLORATORY, no PREREG: what it reports is a description of the mechanism on the dev
dataset, not a criterion value. Three risks of ranking junctions by gromo's
first_order_improvement (conversation 15-18/10/2026):

1. size bias -- moot here: MSVTNet's two FFNs have the same shape (72 -> 72 -> 72);
2. noise -- does the argmax on one half of the statistics batches agree with the
   other half? Compared with the agreement two independent draws would reach by
   chance at the observed choice rate, p^2 + (1-p)^2;
3. starvation -- final width of each junction; fits where a junction never grew.

Unit of analysis = the subject (sessions, folds and seeds are correlated): every rate
is computed per subject, then averaged, with a 95 % bootstrap CI over subjects.

Usage: python step1_select_read.py --root <…/step0_results> --out <prefix>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ARM = "grow_msvtnet_all"


def _boot(per_subject: dict, n_boot: int = 10000, seed: int = 0):
    vals = np.array([v for v in per_subject.values() if v is not None and np.isfinite(v)])
    if len(vals) == 0:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(boots, .025)), \
        float(np.quantile(boots, .975)), len(vals)


def _fmt(t):
    m, lo, hi, n = t
    return "n/a" if m is None else f"{m:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="…/step0_results (holds B/results)")
    ap.add_argument("--out", required=True, help="prefix: writes .json and .md")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.root}/B/results/within_session/bnci2014_001/"
                             f"{ARM}__seed*__fits.jsonl"))
    if not files:
        raise SystemExit(f"no fits under {a.root}")
    fits = [json.loads(l) for f in files for l in open(f)]

    events, by_subj = [], {}
    for r in fits:
        subj = r.get("subject")
        widths_end = None
        for h in r["history"]:
            if "grow_junction" not in h:
                continue
            e = {"subject": subj, "epoch": h["epoch"], "junction": h["grow_junction"],
                 "applied": bool(h.get("grow_applied")), "s": h.get("grow_s"),
                 "foi": h.get("grow_junction_foi"), "ids": h.get("grow_junction_ids"),
                 "a": h.get("grow_split_a"), "b": h.get("grow_split_b"),
                 "widths": h.get("grow_junction_widths_after"),
                 "fp64": h.get("grow_fp64_retry"), "split_failed": h.get("grow_split_failed"),
                 "split_fp64": h.get("grow_split_fp64_retry")}
            events.append(e)
            widths_end = e["widths"] or widths_end
        by_subj.setdefault(subj, []).append({"fit": r, "widths_end": widths_end})

    # --- risk 2: split-half agreement, events where both junctions competed --------
    def per_subject(pred, value):
        out = {}
        for s in by_subj:
            ev = [e for e in events if e["subject"] == s and pred(e)]
            out[s] = value(ev) if ev else None
        return out

    competing = lambda e: e["ids"] is not None and len(e["ids"]) == 2 and \
        e["a"] is not None and e["b"] is not None
    agree = per_subject(competing, lambda ev: np.mean([e["a"] == e["b"] for e in ev]))
    p0 = per_subject(competing, lambda ev: np.mean([e["junction"] == 0 for e in ev]))
    chance = {s: (None if p is None else p ** 2 + (1 - p) ** 2) for s, p in p0.items()}
    excess = {s: (None if agree[s] is None else agree[s] - chance[s]) for s in agree}
    half_vs_full = per_subject(competing, lambda ev: np.mean(
        [(e["a"] == e["junction"]) for e in ev] + [(e["b"] == e["junction"]) for e in ev]))

    # agreement by margin: relative gap between the two ranked gains
    rows = []
    for e in events:
        if competing(e) and e["foi"] and all(f is not None for f in e["foi"]):
            hi, lo = max(e["foi"]), min(e["foi"])
            rows.append(((hi - lo) / abs(hi) if hi else 0.0, e["a"] == e["b"]))
    margin_bins = {}
    if rows:
        gaps = np.array([g for g, _ in rows])
        agr = np.array([x for _, x in rows], dtype=float)
        for lo_q, hi_q in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
            lo_v, hi_v = np.quantile(gaps, lo_q), np.quantile(gaps, hi_q)
            m = (gaps >= lo_v) & (gaps <= hi_v)
            margin_bins[f"q{int(lo_q*100)}-{int(hi_q*100)}"] = {
                "gap_range": [float(lo_v), float(hi_v)], "agreement": float(agr[m].mean()),
                "n_events": int(m.sum())}

    # --- risk 3: starvation --------------------------------------------------------
    targets = [72, 72]
    starve = {}
    for s, lst in by_subj.items():
        w = [x["widths_end"] for x in lst if x["widths_end"] is not None]
        starve[s] = {
            "a_junction_never_grew": float(np.mean([min(v) <= 8 for v in w])) if w else None,
            "both_at_target": float(np.mean([v == targets for v in w])) if w else None,
            "share_width_j0": float(np.mean([v[0] / sum(v) for v in w])) if w else None,
        }
    applied = [e for e in events if e["applied"]]
    share_applied_j0 = per_subject(lambda e: e["applied"],
                                   lambda ev: np.mean([e["junction"] == 0 for e in ev]))
    # when does junction 1 first grow, per fit
    first_j1 = []
    for r in fits:
        ep = [h["epoch"] for h in r["history"] if h.get("grow_junction") == 1
              and h.get("grow_applied")]
        first_j1.append(min(ep) if ep else None)

    # --- step-0 style descriptors (no refusal here) -------------------------------
    s_app = [e["s"] for e in applied]
    censored = float(np.mean([s == 64.0 for s in s_app])) if s_app else None
    sel = [r.get("selected_epoch") for r in fits if r.get("selected_epoch") is not None]

    res = {
        "n_fits": len(fits), "n_subjects": len(by_subj), "n_events": len(events),
        "n_applied": len(applied),
        "n_competing_events": int(sum(competing(e) for e in events)),
        "risk2_split_agreement": _boot(agree),
        "risk2_chance_agreement": _boot(chance),
        "risk2_agreement_minus_chance": _boot(excess),
        "risk2_half_matches_full": _boot(half_vs_full),
        "risk2_agreement_by_margin": margin_bins,
        "share_chosen_j0_competing": _boot(p0),
        "risk3_share_applied_j0": _boot(share_applied_j0),
        "risk3_a_junction_never_grew": _boot({s: v["a_junction_never_grew"]
                                                for s, v in starve.items()}),
        "risk3_both_at_target": _boot({s: v["both_at_target"] for s, v in starve.items()}),
        "risk3_share_width_j0": _boot({s: v["share_width_j0"] for s, v in starve.items()}),
        "first_epoch_j1_grows": {
            "median": None if not [x for x in first_j1 if x] else
            float(np.median([x for x in first_j1 if x])),
            "never": float(np.mean([x is None for x in first_j1]))},
        "censoring_s64_applied": censored,
        "fp64_retry_events": int(sum(bool(e["fp64"]) for e in events)),
        "split_halves_failed": int(sum(e["split_failed"] or 0 for e in events)),
        "split_halves_fp64_retry": int(sum(e["split_fp64"] or 0 for e in events)),
        "selected_epoch_le5": float(np.mean([e <= 5 for e in sel])) if sel else None,
        "selected_epoch_ge195": float(np.mean([e >= 195 for e in sel])) if sel else None,
    }
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    lines = [f"# step1 select_best_update — {ARM}, bnci2014_001 (EXPLORATOIRE)", "",
             f"{res['n_fits']} fits, {res['n_subjects']} sujets, {res['n_events']} "
             f"événements ({res['n_applied']} appliqués, {res['n_competing_events']} "
             "avec 2 jonctions en compétition)", ""]
    for k in ("risk2_split_agreement", "risk2_chance_agreement",
              "risk2_agreement_minus_chance", "risk2_half_matches_full",
              "share_chosen_j0_competing", "risk3_share_applied_j0",
              "risk3_a_junction_never_grew", "risk3_both_at_target",
              "risk3_share_width_j0"):
        lines.append(f"- {k}: {_fmt(res[k])}")
    lines += [f"- agreement by margin: {json.dumps(margin_bins)}",
              f"- first epoch j1 grows: {res['first_epoch_j1_grows']}",
              f"- censoring s=64 (applied): {censored}",
              f"- fp64 retries (real step): {res['fp64_retry_events']} ; split halves "
              f"failed: {res['split_halves_failed']} ; split halves retried in fp64: "
              f"{res['split_halves_fp64_retry']}",
              f"- selected epoch <=5 / >=195: {res['selected_epoch_le5']} / "
              f"{res['selected_epoch_ge195']}"]
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
