"""Write the cells of the 23/10 Shallow learning-rate campaign (PREREG_shallow_lr_path.md).

Question: in the validation campaign grow_shallow (tiny) ends at EXACTLY bd_shallow's
architecture (8 -> 40, same parameter count) yet beats it by +0.024 / +0.033 / +0.059 on
the 3 datasets. Is that an effect of the growth path, or of bd_shallow being
under-optimised at lr 6.25e-4 under our protocol (the Deep4 trap of 14/10)?

Only ``train.lr`` moves, on both arms. Everything else is the validation campaign's line
byte for byte (recipe and tiny rule imported from valid_make_cells); the lr 6.25e-4 arms
are the validation campaign's own cells (same checkout, commit, protocol, card model).
"""

from __future__ import annotations

import sys

from valid_make_cells import DATASETS, MODELS, RULES

LRS = {"lr2e-4": "2e-4", "lr2e-3": "2e-3", "lr6.25e-3": "6.25e-3", "lr2e-2": "2e-2"}


def cells():
    grow, bd, recipe, extra = MODELS["shallow"]
    assert "train.lr=6.25e-4" in recipe, recipe
    for ds, seeds in DATASETS.items():
        for seed in seeds:
            for tag, lr in LRS.items():
                rec = recipe.replace("train.lr=6.25e-4", f"train.lr={lr}")
                yield f"shallow__bd_{tag} {bd} {ds} {seed} {rec}"
                yield f"shallow__tiny_{tag} {grow} {ds} {seed} {rec} {RULES['tiny']} {extra}".rstrip()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "lrshallow_cells.txt"
    lines = list(cells())
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} cells -> {out}")
