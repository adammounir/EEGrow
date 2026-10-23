"""Write the cells of the 23/10 censoring campaign (PREREG_growth_ratio_censoring.md).

Question: under the validated rule (tiny), Conformer / CTNet never reach their target
width and MSVTNet often does not (39 growth events, ``min_singular_ratio=0.1`` keeps 2-3
neurons per event on Conformer). Does letting them grow further change performance?

Only ``model.min_singular_ratio`` moves. Everything else is the validation campaign's
line for the tiny arm, byte for byte (recipes imported from valid_make_cells), and the
reference arms (bd, tiny at the default 0.1) are the validation campaign's own cells:
same checkout, same commit, same protocol, same card model.
"""

from __future__ import annotations

import sys

from valid_make_cells import MODELS, RULES

RATIOS = {"r003": 0.03, "r001": 0.01, "r0001": 0.001, "r0": 0.0}
TARGETS = ["conformer", "ctnet", "msvtnet"]
DATASETS = {"lee2019_mi": [0], "bnci2014_001": [0, 1, 2]}


def cells():
    for ds, seeds in DATASETS.items():
        for name in TARGETS:
            grow, _, recipe, extra = MODELS[name]
            for seed in seeds:
                for tag, r in RATIOS.items():
                    yield (f"{name}__tiny_{tag} {grow} {ds} {seed} {recipe} {RULES['tiny']} "
                           f"++model.min_singular_ratio={r} {extra}").rstrip()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ratio_cells.txt"
    lines = list(cells())
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} cells -> {out}")
