"""Le classement des sujets qui alimente claim 3, écrit une fois dans un CSV.

`donor_select.py` LIT ce fichier plutôt que de re-dériver la variable de sélection. La
raison n'est pas la commodité : la quantité dont claim 2 a mesuré la rho est très
précise -- la moyenne des 15 réplicats (5 folds x 3 seeds) de `params_end` de la sonde
`grow_shallow` within_session, relevée APRÈS `RestoreBestModel`, sur les fits non
alignés. Une deuxième dérivation dans le script d'entraînement pourrait diverger d'un
filtre sans que rien ne le signale, et l'intervention testerait alors une autre variable
que celle que la corrélation a établie.

Le fichier voyage vers le cluster avec le job : c'est aussi ce qui rend le protocole
auditable après coup, puisque le classement exact utilisé est sur le disque à côté des
scores.

Usage::

    ./.venv/bin/python benchmarks/analysis/donor_select_prep.py --dataset physionetmi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import perf_io  # noqa: E402
import donor_matrix as dm  # noqa: E402


def build(dataset: str) -> pd.DataFrame:
    fits = pd.read_csv(A / "dynamics_final" / "gd_fits.csv.gz")
    fits["align_tag"] = fits["align_tag"].fillna("none")
    f = fits[(fits["eval"] == dm.PROBE_EVAL) & (fits.model == dm.PROBE_MODEL)
             & (fits.align_tag == "none") & (fits.dataset == dataset)].copy()
    f["subject"] = pd.to_numeric(f["subject"], errors="coerce")
    probe = f.groupby("subject").agg(
        params_probe=("params_end", "mean"), width_probe=("width_end", "mean"),
        k_probe=("params_end", "size"),
        # La part des réplicats collés à la cible : c'est la censure, elle varie d'un
        # sujet à l'autre et un sujet à 100 % de plafond a une valeur qui n'est pas une
        # mesure mais une borne. On la transporte pour pouvoir la lire à l'analyse.
        ceil_frac=("width_end", lambda s: float((s >= 40).mean())),
    ).reset_index()

    sc = perf_io.attach_params(perf_io.load(A / "perf_final" / "scores"), fits)
    subj = perf_io.by_subject(sc)
    s = subj[(subj["eval"] == dm.PROBE_EVAL) & (subj.align_tag == "none")
             & (subj.model == dm.PROBE_MODEL) & (subj.dataset == dataset)]
    acc = s[["subject", "score"]].rename(columns={"score": "acc_probe"}).copy()
    acc["subject"] = pd.to_numeric(acc["subject"], errors="coerce")

    r = probe.merge(acc, on="subject", how="inner").dropna(
        subset=["params_probe", "acc_probe"])
    r["dataset"] = dataset
    return r.sort_values("subject").reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="physionetmi")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    r = build(a.dataset)
    out = a.out or A / f"ranking_{a.dataset}.csv"
    r.to_csv(out, index=False)
    print(f"{len(r)} sujets → {out}")
    print(r[["params_probe", "acc_probe", "k_probe", "ceil_frac"]]
          .describe().loc[["min", "50%", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
