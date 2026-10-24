"""Cells of the campaign-A timing probe (timing_probe.sbatch).

probe_cells_datasets.txt  12 datasets x 6 models x {bd, tiny}, cross_subject fold 0
                          (the dominant cost), run on one card type.
probe_cells_cards.txt     lee2019_mi only, same 12 arms: the card-speed factors, run once
                          per card type.
Recipes and the tiny rule are the validation campaign's, imported.
"""

from __future__ import annotations

from valid_make_cells import MODELS, RULES

ICASSP = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004", "bnci2015_001", "cho2017",
          "lee2019_mi", "physionetmi", "schirrmeister2017", "shin2017a", "weibo2014", "zhou2016"]


def lines(datasets, ev="cross_subject"):
    for ds in datasets:
        for grow, bd, recipe, extra in MODELS.values():
            yield f"{ev} {bd} {ds} 0 {recipe}"
            yield f"{ev} {grow} {ds} 0 {recipe} {RULES['tiny']} {extra}".rstrip()


if __name__ == "__main__":
    for name, ds in {"probe_cells_datasets.txt": ICASSP, "probe_cells_cards.txt": ["lee2019_mi"]}.items():
        L = list(lines(ds))
        open(name, "w").write("\n".join(L) + "\n")
        print(f"{len(L)} cells -> {name}")
