"""Emit the work-queue grid consumed by ``pack_run.sh`` as a TSV.

One line = one cell = ``eval<TAB>dataset<TAB>model<TAB>seed``. Written out rather
than expanded inside the runner so that the exact set of cells a campaign covers
is a versioned artefact, greppable and diffable, instead of a shell expansion
nobody can reconstruct afterwards.

    python benchmarks/slurm/make_grid.py --eval cross_session > grid_xsess.tsv
    python benchmarks/slurm/make_grid.py --eval cross_session \
        --datasets bnci2014_001 bnci2014_004 bnci2015_001 lee2019_mi zhou2016 \
        --models grow_shallow bd_shallow grow_sccnet bd_sccnet grow_deep

Cells whose result CSV already exists are skipped by ``pack_run.sh``, not here:
the grid describes the campaign, the runner tracks what is left of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config"

# Only these have more than one session, so they are the only ones on which a
# cross-session protocol is defined at all. MOABB raises "Only one session
# available" on the rest, which shows up as an empty result rather than an error.
CROSS_SESSION_DATASETS = ["bnci2014_001", "bnci2014_004", "bnci2015_001",
                          "lee2019_mi", "shin2017a", "zhou2016"]

DEEP_PAIRS = ["bd_shallow", "grow_shallow", "bd_sccnet", "grow_sccnet",
              "bd_deep4", "grow_deep", "bd_eegnex", "grow_eegnex"]


def group(name: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG / name).glob("*.yaml"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", nargs="+", default=["cross_session"],
                    choices=group("eval"))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: the datasets on which the protocol is defined")
    ap.add_argument("--models", nargs="+", default=DEEP_PAIRS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    known_ds, known_m = group("dataset"), group("model")
    lines = []
    for ev in a.eval:
        if a.datasets is not None:
            datasets = a.datasets
        elif ev == "cross_session":
            datasets = CROSS_SESSION_DATASETS
        else:
            datasets = known_ds
        for ds in datasets:
            if ds not in known_ds:
                raise SystemExit(f"unknown dataset config: {ds}")
            for m in a.models:
                if m not in known_m:
                    raise SystemExit(f"unknown model config: {m}")
                for s in a.seeds:
                    lines.append(f"{ev}\t{ds}\t{m}\t{s}")

    text = "\n".join(lines) + "\n"
    if a.out == "-":
        print(text, end="")
    else:
        Path(a.out).write_text(text)
        print(f"{len(lines)} cells -> {a.out}")


if __name__ == "__main__":
    main()
