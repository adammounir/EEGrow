"""Write the cells of the 21/10 validation campaign (PREREG_validation_growth_rule.md).

6 models x (braindecode twin + 3 growth rules) x 3 datasets. Seeds: 3 on bnci2014_001
(n = 9 subjects), 1 on cho2017 / lee2019_mi (n = 52 / 54: the subject is the unit, and
the seeds of one subject are averaged before any contrast anyway). Longest cells first
(large datasets, transformers), so the array's makespan is not set by a late giant.

Recipes = each model's optimiser read in its official repository (CATALOGUE § 5):
type, lr, betas, eps, wd, amsgrad, batch. Everything else is the common protocol of
step 0 (200 epochs, no schedule, no augmentation, score = last epoch).
"""

from __future__ import annotations

import sys

GRID = "train.scaling_grid=[0.0,0.25,0.5,1.0,2.0,4.0]"
RULES = {
    # gromo-style line search on a bounded grid, no normalisation (étape 1 variante A).
    "varA": f"{GRID} train.growth_normalization=null train.growth_fixed_s=null",
    # match_extending_layer + fixed s = 0.25 (best arm of the 20/10 attribution campaign).
    "m025": f"{GRID} train.growth_normalization=match_extending_layer train.growth_fixed_s=0.25",
    # TINY paper, App. E.3.1 + Algorithm 1 (loop.TINY_PAPER_SQ_NORM).
    "tiny": f"{GRID} train.growth_normalization=tiny_paper train.growth_fixed_s=1.0",
}
# name: (grow arm, bd arm, recipe, extra overrides of the growing arm only)
MODELS = {
    "conformer": ("grow_conformer_all", "bd_conformer",
                  "train.optimizer=adam train.lr=2e-4 train.betas=[0.5,0.999] "
                  "train.weight_decay=0.0 train.batch_size=72", ""),
    "ctnet": ("grow_ctnet_all", "bd_ctnet",
              "train.optimizer=adam train.lr=1e-3 train.betas=[0.5,0.999] "
              "train.weight_decay=0.0 train.batch_size=72", ""),
    "msvtnet": ("grow_msvtnet_all", "bd_msvtnet",
                "train.optimizer=adam train.lr=1e-3 train.weight_decay=0.0 "
                "train.batch_size=64", "model.growth_split_half=false"),
    "tidnet": ("grow_tidnet", "bd_tidnet",
               "train.optimizer=adam train.amsgrad=true train.lr=1e-3 "
               "train.weight_decay=0.01 train.batch_size=60", ""),
    "eegnex": ("grow_eegnex_j2", "bd_eegnex",
               "train.optimizer=adam train.lr=1e-3 train.optimizer_eps=1e-7 "
               "train.weight_decay=0.0 train.batch_size=128", ""),
    # Step-0 config B (lr 6.25e-4, wd 0), batch 64 = the attribution campaign's.
    "shallow": ("grow_shallow", "bd_shallow",
                "train.lr=6.25e-4 train.weight_decay=0.0 train.batch_size=64", ""),
}
DATASETS = {"lee2019_mi": [0], "cho2017": [0], "bnci2014_001": [0, 1, 2]}


def cells():
    for ds, seeds in DATASETS.items():
        for name, (grow, bd, recipe, extra) in MODELS.items():
            for seed in seeds:
                yield f"{name}__bd {bd} {ds} {seed} {recipe}"
                for rule, ov in RULES.items():
                    yield f"{name}__{rule} {grow} {ds} {seed} {recipe} {ov} {extra}".rstrip()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "valid_cells.txt"
    lines = list(cells())
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} cells -> {out}")
