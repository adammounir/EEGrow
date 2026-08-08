"""One-off repair: bring results written to ``benchmarks/benchmarks/results`` home.

``results_dir`` used to be ``${hydra:runtime.cwd}/benchmarks/results``, which makes the
destination depend on the directory the process happens to be launched from. Under slurm
that is not something the sbatch script controls -- the same ``cd .../benchmarks`` before
``srun`` sent some jobs to ``benchmarks/results`` and others one level deeper. A whole
arm of the grid (the Euclidean-alignment pilot) landed in the second one, outside the
tree every analysis reads. The runs are fine; only the destination was wrong.

utils.default_results_root now anchors on the runner's own file, so nothing new can land
there. This script moves what already did.

Copy rather than move: the source tree stays as a fallback until the grid is closed.
Files touched in the last two minutes are skipped, since a run may still be writing.
Re-runnable, and must be re-run while jobs that predate the fix are still finishing.

    python consolidate_nested_results.py           # simulation
    python consolidate_nested_results.py --apply
"""

import os
import shutil
import sys
import time

from utils import default_results_root

QUIET_FOR = 120.0


def main(apply: bool) -> int:
    dst = default_results_root()
    src = dst.parent / "benchmarks" / "results"
    if not src.is_dir():
        print(f"rien a rapatrier : {src} n'existe pas")
        return 0
    print(f"source : {src}\ncible  : {dst}\n")

    now = time.time()
    copied = todo = fresh = same = 0
    for root, _dirs, files in os.walk(src):
        for name in files:
            # hdf5/lock are MOABB's own cache and are rebuilt on demand
            if not name.endswith((".csv", ".joblib")):
                continue
            s = os.path.join(root, name)
            rel = os.path.relpath(s, src)
            d = os.path.join(dst, rel)
            if now - os.path.getmtime(s) < QUIET_FOR:
                fresh += 1
                print(f"  [ecriture en cours, ignore] {rel}")
                continue
            if os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(s):
                same += 1
                continue
            if not apply:
                todo += 1
                print(f"  [copierait] {rel}")
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            copied += 1

    print(f"\ncopies={copied} a_copier={todo} deja_a_jour={same} trop_recents={fresh}")
    if not apply:
        print("(simulation -- relancer avec --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--apply" in sys.argv))
