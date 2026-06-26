"""Multi-subject MOABB benchmark: growable vs fixed baselines, via EEGClassifier.

Runs the same three arms as ``benchmark_eegclassifier.py`` (fixed-small / growable /
fixed-target) across several BNCI2014_001 subjects, within-subject (train/test split
per subject), and reports the per-subject test accuracy plus the mean +/- std across
subjects. Writes a markdown results table (so it can be committed / shown in a PR).

This is the standard MOABB within-subject protocol (subjects are the variation).
Single train/test split and single seed per subject -- not a cross-validated paper
result, but a solid sanity benchmark on real data.

Run:  python examples/benchmark_moabb_multi.py --subjects 1-9
      python examples/benchmark_moabb_multi.py --subjects 1,2,3 --epochs 30
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


def parse_subjects(spec: str) -> list[int]:
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


def run_subject(subject, *, model, epochs, grow_every, start, target, device):
    X, y, sf = load_moabb(subject)
    C, T, K = X.shape[1], X.shape[2], int(y.max()) + 1
    Xtr, ytr, Xte, yte = split(X, y)
    accs = {}
    for kind in ARMS:
        m = build(model, kind, n_chans=C, n_classes=K, n_times=T, sfreq=sf,
                  start=start, target=target, device=device)
        clf = make_eeg_classifier(m, max_epochs=epochs, grow_every=grow_every,
                                  batch_size=32, device=device, verbose=False)
        clf.set_params(verbose=0)  # silence skorch's per-epoch print log
        clf.fit(Xtr, ytr)
        accs[kind] = float((clf.predict(Xte) == yte).mean())
    return accs, (C, T, K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", default="1-9")
    p.add_argument("--model", choices=["sccnet", "eegnex"], default="sccnet")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--grow-every", type=int, default=5)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--target", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="examples/results_bnci2014_001.md")
    args = p.parse_args()

    subjects = parse_subjects(args.subjects)
    per_subject: dict[int, dict[str, float]] = {}
    t0 = time.time()
    for s in subjects:
        print(f"--- subject {s} ---", flush=True)
        accs, dims = run_subject(
            s, model=args.model, epochs=args.epochs, grow_every=args.grow_every,
            start=args.start, target=args.target, device=args.device)
        per_subject[s] = accs
        print(f"  S{s}: " + "  ".join(f"{k}={v:.3f}" for k, v in accs.items()),
              flush=True)

    # aggregate
    summary = {arm: [per_subject[s][arm] for s in subjects] for arm in ARMS}
    mean = {arm: statistics.mean(v) for arm, v in summary.items()}
    std = {arm: (statistics.pstdev(v) if len(v) > 1 else 0.0)
           for arm, v in summary.items()}

    # ---- markdown report ----
    lines = []
    lines.append(f"# BNCI2014_001 within-subject benchmark ({args.model})\n")
    lines.append(
        f"Growable vs fixed baselines via braindecode `EEGClassifier`. "
        f"Subjects {args.subjects}, {args.epochs} epochs, grow_every={args.grow_every}, "
        f"width {args.start}->{args.target}. Single split/seed per subject.\n")
    lines.append("Test accuracy per subject:\n")
    lines.append("| subject | " + " | ".join(ARMS) + " |")
    lines.append("|" + "---|" * (len(ARMS) + 1))
    for s in subjects:
        lines.append(f"| S{s} | " +
                     " | ".join(f"{per_subject[s][a]:.3f}" for a in ARMS) + " |")
    lines.append("| **mean** | " +
                 " | ".join(f"**{mean[a]:.3f} ± {std[a]:.3f}**" for a in ARMS) + " |")
    report = "\n".join(lines) + "\n"

    with open(args.out, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"(done in {time.time() - t0:.0f}s; written to {args.out})")


if __name__ == "__main__":
    main()
