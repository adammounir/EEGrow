"""Emit the four point files of the classical-baseline arm of the v5 campaign.

The v5 grid declares the six ``ml_*`` configs and has run none of them, so on v5
"best of family" can only ever mean "best of four braindecode nets". That makes the
one comparison the whole exploration turns on -- deep learning against the Riemannian
/ CSP pipelines -- inexpressible on the campaign that has provenance. These files fix
that.

WHY THE SEED GRID IS NOT UNIFORM
--------------------------------
Every pipeline in this arm is deterministic given its split: ``CSP`` + shrinkage LDA,
``SVC(rbf)``, ``Covariances('oas')`` + ``TangentSpace`` + ``LogisticRegression``,
``MDM``, ``FgMDM`` -- not one of them takes a ``random_state``. The only thing
``seed`` reaches is MOABB's split, and only ``WithinSessionEvaluation`` draws a random
one (``KFold(random_state=seed)``); leave-one-subject-out and leave-one-session-out
are enumerations of the data, so the seed cannot move them.

That is a claim about the code, so it was checked against the 14-model campaign
before being spent: over the 7 848 (eval, dataset, model, subject, session) groups
that carry more than one seed,

    cross_session   seed-to-seed sd = 0.000000 on 100 % of groups
    cross_subject   seed-to-seed sd = 0.000000 on 100 % of groups
    within_session  seed-to-seed sd = 0.0277 mean, 0.233 max

(the deep arms, for contrast, sit at 0.037-0.062 mean sd in every mode). So five seeds
on ``within_session`` is a real replicate and five seeds on the other two modes is the
same number computed five times. Dropping them costs nothing and removes 80 % of the
cost of the only expensive mode -- ``cross_subject`` on schirrmeister2017 is 50 h of
CPU per cell.

It is still a claim about *this* stack, so the light pass carries a control: seed 1 on
five cheap (eval, dataset) combinations of the deterministic modes. If those come back
bit-identical to seed 0 the argument holds under moabb 1.5.0 too; if they do not, the
single-seed cells have to be topped up and we will know before the figures are drawn.

WHY FOUR FILES AND NOT ONE ARRAY
--------------------------------
Wall clock and memory span three orders of magnitude across this grid (measured on the
14-model campaign, summed over folds, per cell):

    within_session, all but schirrmeister2017   <= 26 min
    cross_subject, the 8 small datasets         <= 2 h
    cross_subject, cho2017/lee2019/physionetmi  4.6 h - 27 h, all subjects in RAM
    schirrmeister2017 within_session            2.4 h - 5.7 h
    schirrmeister2017 cross_subject             50 h - 95 h, 128 channels x 14 subjects

One array sized for the worst cell would hold a 450 GB reservation for 26-minute jobs;
one sized for the median would be killed on the tail. The July run of this arm did
exactly that and died with ``srun: error: marg026: task 0: Killed``.

    python benchmarks/slurm/make_ml_grid.py --out-dir /scratch/amounir/ml_v5
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = ["ml_csp_lda", "ml_csp_svm", "ml_fgmdm", "ml_mdm", "ml_ts_lr", "ml_ts_svm"]

ALL12 = ["alexmi", "bnci2014_001", "bnci2014_002", "bnci2014_004", "bnci2015_001",
         "cho2017", "lee2019_mi", "physionetmi", "schirrmeister2017", "shin2017a",
         "weibo2014", "zhou2016"]
# The only datasets with more than one session, so the only ones on which a
# cross-session protocol exists at all (same list as make_grid.py).
XSESS6 = ["bnci2014_001", "bnci2014_004", "bnci2015_001", "lee2019_mi", "shin2017a",
          "zhou2016"]
# All subjects live in memory at once under leave-one-subject-out, which is what makes
# these three the expensive cells rather than merely the big datasets.
XSUBJ_HEAVY = ["cho2017", "lee2019_mi", "physionetmi"]
XSUBJ_LIGHT = [d for d in ALL12 if d not in XSUBJ_HEAVY + ["schirrmeister2017"]]
# 128 channels x 14 subjects: its own pass in both protocols.
SCHIRR = "schirrmeister2017"

# Deterministic-mode control (see the docstring): cheap combinations where a second
# seed costs minutes, chosen to cover both protocols and both metrics.
CONTROL = [("cross_subject", d) for d in ("alexmi", "bnci2014_002", "zhou2016")] + \
          [("cross_session", d) for d in ("bnci2014_004", "zhou2016")]


def cells(ev: str, datasets: list[str], seeds: list[int]) -> list[str]:
    return [f"{ev}\t{ds}\t{m}\t{s}"
            for ds in datasets for m in MODELS for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    light = (
        cells("within_session", [d for d in ALL12 if d != SCHIRR], [0, 1, 2, 3, 4])
        + cells("cross_session", XSESS6, [0])
        + cells("cross_subject", XSUBJ_LIGHT, [0])
        + [f"{ev}\t{ds}\t{m}\t1" for ev, ds in CONTROL for m in MODELS]
    )
    files = {
        "ml_light.txt": light,
        "ml_mid.txt": cells("cross_subject", XSUBJ_HEAVY, [0]),
        "ml_schirr_within.txt": cells("within_session", [SCHIRR], [0, 1, 2, 3, 4]),
        "ml_schirr_xsubj.txt": cells("cross_subject", [SCHIRR], [0]),
    }
    for name, lines in files.items():
        (out / name).write_text("\n".join(lines) + "\n")
        print(f"{len(lines):4d} cells -> {out / name}")
    print(f"{sum(len(v) for v in files.values())} cells total")


if __name__ == "__main__":
    main()
