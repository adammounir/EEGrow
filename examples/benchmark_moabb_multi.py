"""Multi-subject, multi-seed MOABB benchmark: growable vs fixed baselines.

Runs the same three arms as ``benchmark_eegclassifier.py`` (fixed-small / growable /
fixed-target) across several BNCI2014_001 subjects, within-subject (train/test split
per subject), and -- crucially -- across several **seeds** per subject.

Why multi-seed? A single split + single seed is a noisy estimator. Three sources of
variance stack here: the network init, the train/test split, and (for ``growable``)
the mini-batch gromo sees at each growth step. With one seed the ``growable`` vs
``fixed-target`` gap can be as much noise as signal. Averaging over seeds cancels the
init/split noise and lets us separate *run-to-run* variance (stability of an arm on a
subject) from *between-subject* variance (subject difficulty), so we can tell whether
the ranking of the three arms is robust.

For each (subject, seed): split with that seed, init the models with that seed, fit,
eval. We report, per subject, the mean +/- std **over seeds** (stability), and at the
bottom the mean +/- std **over subjects** (the headline result, N = subjects x seeds
runs). Writes a markdown report so it can be committed / shown in a PR.

Run:  python examples/benchmark_moabb_multi.py --subjects 1-9 --seeds 0-2
      python examples/benchmark_moabb_multi.py --subjects 1,2,3 --seeds 0,1,2,3
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

from benchmark_eegclassifier import build, load_moabb, split

from eegrow.training.skorch_integration import make_eeg_classifier

warnings.filterwarnings("ignore")

ARMS = ("fixed-small", "growable", "fixed-target")


def parse_ints(spec: str) -> list[int]:
    """'1-9' -> [1..9]; '1,3,5' -> [1,3,5]; mixes allowed."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_subject_seed(X, y, sf, *, model, epochs, grow_every, start, target, device,
                     seed):
    """One (subject, seed) run: seeded split + seeded init, all three arms."""
    C, T, K = X.shape[1], X.shape[2], int(y.max()) + 1
    Xtr, ytr, Xte, yte = split(X, y, seed=seed)
    accs = {}
    for kind in ARMS:
        m = build(model, kind, n_chans=C, n_classes=K, n_times=T, sfreq=sf,
                  start=start, target=target, device=device, seed=seed)
        clf = make_eeg_classifier(m, max_epochs=epochs, grow_every=grow_every,
                                  batch_size=32, device=device, verbose=False)
        clf.set_params(verbose=0)  # silence skorch's per-epoch print log
        clf.fit(Xtr, ytr)
        accs[kind] = float((clf.predict(Xte) == yte).mean())
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", default="1-9")
    p.add_argument("--seeds", default="0-2", help="e.g. '0-2' or '0,1,2'")
    p.add_argument("--model", choices=["sccnet", "eegnex"], default="sccnet")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--grow-every", type=int, default=5)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--target", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="examples/results_bnci2014_001.md")
    args = p.parse_args()

    subjects = parse_ints(args.subjects)
    seeds = parse_ints(args.seeds)

    # per_subject[s][arm] = list of acc over seeds
    per_subject: dict[int, dict[str, list[float]]] = {}
    t0 = time.time()
    for s in subjects:
        print(f"--- subject {s} ---", flush=True)
        X, y, sf = load_moabb(s)  # load once, reuse across seeds
        per_subject[s] = {arm: [] for arm in ARMS}
        for seed in seeds:
            accs = run_subject_seed(
                X, y, sf, model=args.model, epochs=args.epochs,
                grow_every=args.grow_every, start=args.start, target=args.target,
                device=args.device, seed=seed)
            for arm in ARMS:
                per_subject[s][arm].append(accs[arm])
            print(f"  S{s} seed={seed}: " +
                  "  ".join(f"{k}={accs[k]:.3f}" for k in ARMS), flush=True)

    def m_sd(vals):
        return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    # per-subject mean over seeds
    subj_mean = {s: {a: statistics.mean(per_subject[s][a]) for a in ARMS}
                 for s in subjects}
    # between-subject aggregate (of the per-subject means)
    overall = {a: m_sd([subj_mean[s][a] for s in subjects]) for a in ARMS}
    # average run-to-run (over-seeds) std, per arm -> stability
    seed_std = {a: statistics.mean([m_sd(per_subject[s][a])[1] for s in subjects])
                for a in ARMS}

    # ---- markdown report ----
    L = []
    L.append(f"# BNCI2014_001 within-subject benchmark ({args.model})\n")
    L.append(
        f"Growable vs fixed baselines via braindecode `EEGClassifier`. "
        f"Subjects {args.subjects}, seeds {args.seeds} "
        f"({len(subjects)}x{len(seeds)} = {len(subjects) * len(seeds)} runs/arm), "
        f"{args.epochs} epochs, grow_every={args.grow_every}, "
        f"width {args.start}->{args.target}.\n")
    L.append("Per-subject test accuracy (mean ± std **over seeds**):\n")
    L.append("| subject | " + " | ".join(ARMS) + " |")
    L.append("|" + "---|" * (len(ARMS) + 1))
    for s in subjects:
        cells = []
        for a in ARMS:
            mu, sd = m_sd(per_subject[s][a])
            cells.append(f"{mu:.3f} ± {sd:.3f}")
        L.append(f"| S{s} | " + " | ".join(cells) + " |")
    L.append("| **mean (over subjects)** | " +
             " | ".join(f"**{overall[a][0]:.3f} ± {overall[a][1]:.3f}**"
                        for a in ARMS) + " |")
    L.append("")
    L.append("Mean run-to-run std (over seeds, averaged over subjects) — lower is "
             "more stable:\n")
    L.append("| " + " | ".join(ARMS) + " |")
    L.append("|" + "---|" * len(ARMS))
    L.append("| " + " | ".join(f"{seed_std[a]:.3f}" for a in ARMS) + " |")
    report = "\n".join(L) + "\n"

    with open(args.out, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"(done in {time.time() - t0:.0f}s; written to {args.out})")


if __name__ == "__main__":
    main()
