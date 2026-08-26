"""Is `bd_deep4` at chance because of its architecture, or because of when we stop it?

WHAT THE PREVIOUS EXPERIMENT SETTLED (benchmarks/exp_deep4_lr.py, 29 fits)
--------------------------------------------------------------------------
Not the learning rate, twice over. `cosine` scored 0.4203 against `constant`'s 0.4185
-- +0.0018, identical epochs (24) and identical restored epoch (4) -- because with
T_max=200 and a stop at epoch 24 the LR only moved 6.25e-4 -> 6.05e-4. And a 10x lower
CONSTANT lr, the diagnostic that does not depend on the budget, blows up just the same
(validation loss more than doubles on 6 of 10 fits) while ending at a train loss of
1.394, i.e. ln(4). Step size is not the mechanism.

The mechanism is in the fit record. On a representative fit::

     ep   train   vloss   vacc
      4  1.3963  1.1133  0.348   <- argmin of valid_loss
      7  1.1837  2.2200  0.261
     15  1.0147  3.2689  0.261
     24  0.8648  2.8051  0.304   <- EarlyStopping fires; train loss still descending

Validation loss bottoms at epoch 4 and then explodes, while the train loss keeps
falling (still falling at the last epoch on 80% of fits) and valid_acc freezes at
0.261. The net is not failing to learn, it is becoming confident and wrong on a
46-trial validation split, where a handful of confident errors dominate the mean
cross-entropy but move accuracy by 1/46. Valid loss explodes (max > 2x min) on 10/10
constant fits and 9/9 cosine fits.

Two config decisions then compound. `stop_monitor=valid_loss` with patience 20 ends
the fit 20 epochs after that epoch-4 minimum, and `RestoreBestModel` hands back the
model AT the minimum. The score this benchmark publishes for `bd_deep4` is the score of
a network trained for four epochs.

THE DESIGN -- a 2x2 on (budget, selection), one cell of which is free
--------------------------------------------------------------------
Two separable decisions, so two factors:

               selection=valid_loss        selection=valid_acc
  patience 20  p20_loss  (HEAD baseline)   p20_acc
  patience 200 [free, see below]           full_acc

  p20_loss   patience null (=20), sel valid_loss. HEAD as it stands; 0.4185 on
             subject 1, v5 scored 0.272 on the whole cell.
  p20_acc    same budget, sel valid_acc. Isolates SELECTION at a fixed trajectory:
             the training run is bit-identical to p20_loss for the same seed, only
             the restored epoch differs.
  full_acc   patience 200 = no early stopping (skorch needs `patience` consecutive
             misses and there are only max_epochs-1 epochs to miss on), sel
             valid_acc. The ceiling: what does this net reach if simply left alone.
  full_cos   full_acc plus CosineAnnealingLR. Cosine is only interpretable HERE --
             at the full budget T_max=200 matches what is actually traversed, so the
             LR genuinely anneals to ~0 instead of the 3% it moved last time.

The fourth cell -- patience 200 with selection on valid_loss -- is NOT run, and that
is a measurement rather than an omission. With the same seed its trajectory is
bit-identical to p20_loss's for the first 24 epochs, so it differs only if valid_loss
comes back BELOW its epoch-4 minimum somewhere in epochs 25-200. `analysis/deep4_budget.py`
reads argmin(valid_loss) off every full_acc history and checks exactly that: if the
argmin is < 25 on every fit, the cell is bit-identical to p20_loss by construction and
costs nothing to know. If it is not, the cell has to be run and the analysis says so.

PRIMARY ENDPOINT is held-out accuracy, paired on (subject, session, seed) against
p20_loss -- unlike the LR experiment, the question here is not "can it fit its training
data" but "does the model we hand to the test set get better", and the exact 95% chance
threshold for this cell (288 trials, 4 classes) is 0.295.

SECONDARY, and the one that generalises past `bd_deep4`: `restored_epoch` and the
epoch at which the run ends. `selection_monitor` is a GLOBAL knob, so if selection is
what matters here it is a benchmark-wide finding, not a per-model patch.

Usage::

    python exp_deep4_budget.py --out DIR                 # full grid, resumable
    python exp_deep4_budget.py --out DIR --subjects 1    # pilot
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

# (name, patience, selection_monitor, lr_schedule). `patience=None` -> config default.
ARMS = [
    ("p20_loss", "null", "valid_loss", "null"),
    ("p20_acc", "null", "valid_acc", "null"),
    ("full_acc", "200", "valid_acc", "null"),
    ("full_cos", "200", "valid_acc", "cosine"),
    # Run after the fact: the 2x2's fourth cell was predicted free and is not.
    # 58/360 full-budget fits have argmin(valid_loss) at epoch >= 25 (max 200),
    # so its trajectory does leave p20_loss's and the cell has to be measured.
    ("full_loss", "200", "valid_loss", "null"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arm names")
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--eval", dest="evaluation", default="within_session")
    args = ap.parse_args()

    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    cells = [(*a, s) for a in arms for s in args.seeds]
    print(f"{len(cells)} cells: {[a[0] for a in arms]} x seeds {args.seeds}", flush=True)

    t_start = time.perf_counter()
    for i, (arm, patience, sel, sched, seed) in enumerate(cells, 1):
        out = args.out / arm
        # Resumable: the runner writes the per-(cell, seed) CSV last, so its presence
        # means the cell finished. The full-budget arms are ~8x the length of the
        # patience-20 ones; losing one to an interruption is not a rerun worth eating.
        done = (out / "results" / args.evaluation / args.dataset
                / f"bd_deep4__seed{seed}.csv")
        if done.exists():
            print(f"[{i}/{len(cells)}] {arm} seed{seed}: done, skipping", flush=True)
            continue
        cmd = [PYTHON, str(HERE / "run_cell.py"),
               "model=bd_deep4", f"dataset={args.dataset}",
               f"eval={args.evaluation}", f"seed={seed}",
               f"train.patience={patience}", f"train.selection_monitor={sel}",
               f"train.lr_schedule={sched}", f"results_dir={out / 'results'}"]
        if args.subjects:
            cmd.append(f"dataset.subjects=[{','.join(map(str, args.subjects))}]")
        print(f"[{i}/{len(cells)}] {arm} seed{seed} "
              f"(patience={patience}, sel={sel}, sched={sched})", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=HERE, env={**os.environ,
                                               "EEGROW_BENCH_ROOT": str(out)},
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            print(f"    FAILED rc={r.returncode} after {dt:.0f}s\n"
                  f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)
            continue
        tail = [l for l in r.stdout.splitlines() if "run_cell]" in l]
        print(f"    {dt / 60:.1f} min | {tail[-1] if tail else '(no summary)'}",
              flush=True)

    print(f"\ntotal {(time.perf_counter() - t_start) / 60:.1f} min -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
