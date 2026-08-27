"""Why does `bd_deep4` fail to fit its own training data? One cell, three arms.

THE FINDING THIS TESTS
----------------------
On the v5 curves, `bd_deep4` is the only net in the benchmark that cannot fit its
TRAINING set. On bnci2014_001 within_session its final train loss is 1.014 -- against a
chance loss of ln(4) = 1.386 -- while `fix_deepeeg`, a control 29x smaller, reaches
0.029. That is not capacity and not overfitting; a 283k-parameter net that cannot
memorise 230 examples has a descent problem.

The signature is instability. Fraction of epochs on which the TRAIN loss *rises*,
median over all v5 cells: `bd_deep4` 27.8% (worst single-epoch jump 0.066) against
`fix_deepeeg` 9.9% (0.002). Inside `bd_deep4`, the cells that fail are exactly the ones
with the largest jumps (bnci2014_001 within 0.17, zhou2016 within 0.18, alexmi within
0.14). `bd_deep4` and `bd_shallow` also start 0.40 and 0.34 nats ABOVE ln(k), i.e. the
stock braindecode Xavier init on the final classifier makes them confidently wrong at
epoch 1, where our nets start at +0.08.

Already refuted, so it is not the cause: gradient steps per epoch.
Spearman(batches/epoch, final train loss) = 0.095 for `bd_deep4`.

THE ARMS -- one factor each, which is the point
-----------------------------------------------
  constant  lr 6.25e-4, no schedule. This is HEAD, not v5: `drop_last=False`,
            `selection_monitor=valid_loss` and `RestoreBestModel` all landed after v5
            was published. So this arm doubles as the measurement of how much those
            committed fixes already recovered on their own -- v5 scored 0.272 here.
  lowlr     lr 6.25e-5, no schedule. The DIAGNOSTIC. If instability is a step-size
            problem, a 10x smaller step fixes it, and it does so without depending on
            the epoch budget the way an annealed schedule does.
  cosine    lr 6.25e-4, CosineAnnealingLR over max_epochs. The candidate production
            fix, and what braindecode's own Deep4Net recipe does. Caveat worth reading
            off `lr` in the fit record before interpreting: early stopping fires around
            epoch 60-90 of 200, at which point the cosine has decayed by only ~20%. A
            null result on this arm and a win on `lowlr` would mean the schedule is
            right in principle and mis-scaled in practice.

PRIMARY ENDPOINT is the final TRAIN loss, because that is the quantity the diagnosis is
about and it is measured 90 times per cell (9 subjects x 2 sessions x 5 folds).
Held-out accuracy is secondary and is paired across the 18 subject-sessions.

Seeds move the initialisation and nothing else; the paired test is over subjects, so 3
is enough. Runtime is ~35 min per cell on one CPU core, 9 cells.

Usage::

    python exp_deep4_lr.py                 # all 9 cells, resumable
    python exp_deep4_lr.py --subjects 1 2  # pilot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

# (name, lr, schedule). Order is the reading order of the write-up, not an execution
# constraint -- the cells are independent.
ARMS = [
    ("constant", "6.25e-4", "null"),
    ("lowlr", "6.25e-5", "null"),
    ("cosine", "6.25e-4", "cosine"),
]
SEEDS = [0, 1, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="root for results; one subdirectory per arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="restrict the subject list (pilot); default = all 9")
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--eval", dest="evaluation", default="within_session")
    args = ap.parse_args()

    cells = [(a, lr, sc, s) for (a, lr, sc) in ARMS for s in args.seeds]
    print(f"{len(cells)} cells: {[a for a, _, _ in ARMS]} x seeds {args.seeds}",
          flush=True)

    t_start = time.perf_counter()
    for i, (arm, lr, sched, seed) in enumerate(cells, 1):
        out = args.out / arm
        # Resumable: the runner writes one CSV per (cell, seed) at the very end, so its
        # presence means the cell finished. A 5-hour sequential run that cannot resume
        # is a 5-hour run that any interruption costs in full.
        done = out / "results" / args.evaluation / args.dataset / f"bd_deep4__seed{seed}.csv"
        if done.exists():
            print(f"[{i}/{len(cells)}] {arm} seed{seed}: already done, skipping",
                  flush=True)
            continue
        cmd = [PYTHON, str(HERE / "run_cell.py"),
               "model=bd_deep4", f"dataset={args.dataset}",
               f"eval={args.evaluation}", f"seed={seed}",
               f"train.lr={lr}", f"train.lr_schedule={sched}",
               f"results_dir={out / 'results'}"]
        if args.subjects:
            cmd.append(f"dataset.subjects=[{','.join(map(str, args.subjects))}]")
        print(f"[{i}/{len(cells)}] {arm} seed{seed} (lr={lr}, schedule={sched})",
              flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=HERE, env={**__import__("os").environ,
                                               "EEGROW_BENCH_ROOT": str(out)},
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            # Print and keep going: one arm dying on one seed should not cost the
            # other eight cells, and the resume check above makes the retry cheap.
            print(f"    FAILED rc={r.returncode} after {dt:.0f}s\n"
                  f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)
            continue
        tail = [l for l in r.stdout.splitlines() if "run_cell]" in l]
        print(f"    {dt / 60:.1f} min | {tail[-1] if tail else '(no summary line)'}",
              flush=True)

    print(f"\ntotal {(time.perf_counter() - t_start) / 60:.1f} min -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
