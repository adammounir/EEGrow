"""Lecture de la campagne `dalloc_restored`, suivant `PREREG_alloc_restored.md`.

H_A (primaire) -- allouer le budget par `restored_probe` plutot que par `acc_probe`
change le score du receveur. Endpoint : delta apparie par SUJET DE TEST, a pli, seed,
budget et beta identiques, moyenne sur la grille, IC bootstrap 20 000 sur les sujets,
bilateral, Holm sur les 3 datasets.

H_B (le vrai enjeu) -- l'ecran hors ligne top-5 sur les matrices D*R predit le SIGNE du
resultat end-to-end. Trois signes geles avant le premier fit : cho2017 +, lee2019_mi +,
physionetmi -. Le succes est 3/3 ; 2/3 vaut p=0.5 sous binomiale et sera rapporte comme
un demi-echec.

L'appariement est verifie ici et non suppose : toute cellule sans son exacte contrepartie
dans l'autre bras fait ECHOUER le script plutot que d'etre ignoree en silence -- c'est le
mode de defaillance de [[gates-must-refuse-not-warn]] et de
[[cancelled-cell-pollutes-fits-jsonl]].

Usage::

    ./.venv/bin/python benchmarks/analysis/alloc_restored_readout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260905)
N_BOOT = 20000
DATASETS = ["cho2017", "lee2019_mi", "physionetmi"]
# Signes geles dans la prereg, AVANT le premier fit. Ne pas toucher.
PREDICTED = {"cho2017": +1, "lee2019_mi": +1, "physionetmi": -1}
# La cellule est identifiee par ces colonnes ; `measure` est le bras.
CELL = ["fold", "seed", "budget_tag", "beta"]
KEY = CELL + ["test_subject"]


def load(root: str, ds: str) -> pd.DataFrame:
    fr = pd.concat([pd.read_csv(f) for f in sorted((A / root / ds).glob("*.csv"))
                    if not f.name.endswith("__fits.jsonl")], ignore_index=True)
    return fr


def arm(ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Le bras `restored` et son bras de reference `acc`, restreints a la grille commune."""
    res = load("dalloc_restored", ds)
    acc = load("dalloc", ds)
    acc = acc[acc["measure"] == "acc_probe"]
    # beta=0 n'a pas ete relance (il ne depend d'aucune mesure) : il n'est pas dans
    # `restored`, donc l'intersection le retire des deux cotes. C'est voulu.
    grid = set(map(tuple, res[CELL].drop_duplicates().to_numpy()))
    acc = acc[[tuple(r) in grid for r in acc[CELL].to_numpy()]]
    return res, acc


def paired(ds: str) -> pd.DataFrame:
    res, acc = arm(ds)
    r = res.groupby(KEY, as_index=False)["roc_auc"].mean()
    a = acc.groupby(KEY, as_index=False)["roc_auc"].mean()
    m = r.merge(a, on=KEY, suffixes=("_res", "_acc"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        bad = m[m["_merge"] != "both"]
        raise SystemExit(
            f"{ds} : {len(bad)} lignes non appariees (ex. "
            f"{bad.iloc[0][KEY].to_dict()}). L'appariement est la condition de "
            f"validite du test ; refus de moyenner des grilles differentes.")
    m["delta"] = m["roc_auc_res"] - m["roc_auc_acc"]
    return m


def boot_by_subject(m: pd.DataFrame) -> tuple[float, float, float, float, int]:
    """Bootstrap sur les SUJETS DE TEST -- l'unite d'analyse ; les cellules d'un meme
    sujet sont correlees et les resampler gonflerait la precision d'un facteur 10^4."""
    per = m.groupby("test_subject")["delta"].mean().to_numpy(float)
    n = len(per)
    b = RNG.choice(per, (N_BOOT, n), replace=True).mean(1)
    mu = float(per.mean())
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    # p bilateral par inversion du bootstrap, borne a 1/N_BOOT (jamais annonce comme 0).
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return mu, lo, hi, max(float(p), 1.0 / N_BOOT), n


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    k, out, run = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        run = max(run, min(1.0, (k - i) * p))
        out[name] = run
    return out


def main() -> int:
    frames = {ds: paired(ds) for ds in DATASETS}

    print("=" * 78)
    print("H_A -- delta apparie par sujet de test, `restored_probe` - `acc_probe`")
    print("=" * 78)
    stats, ps = {}, {}
    for ds in DATASETS:
        mu, lo, hi, p, n = boot_by_subject(frames[ds])
        stats[ds] = (mu, lo, hi, n)
        ps[ds] = p
    adj = holm(ps)
    print(f"{'dataset':<14}{'n':>5}{'cellules':>10}{'delta':>10}{'IC95':>22}"
          f"{'p brut':>9}{'p Holm':>9}")
    for ds in DATASETS:
        mu, lo, hi, n = stats[ds]
        sig = "*" if adj[ds] < 0.05 else " "
        print(f"{ds:<14}{n:>5}{len(frames[ds]):>10}{mu:>+10.4f}"
              f"   [{lo:+.4f}, {hi:+.4f}]{ps[ds]:>9.4f}{adj[ds]:>8.4f}{sig}")

    print("\n" + "=" * 78)
    print("H_B -- l'ecran hors ligne (cout GPU nul) predit-il le signe end-to-end ?")
    print("=" * 78)
    print(f"{'dataset':<14}{'predit':>9}{'observe':>10}{'delta':>10}{'accord':>9}")
    hits = 0
    for ds in DATASETS:
        mu, lo, hi, _ = stats[ds]
        obs = int(np.sign(mu))
        ok = obs == PREDICTED[ds]
        hits += ok
        print(f"{ds:<14}{'+' if PREDICTED[ds] > 0 else '-':>9}"
              f"{'+' if obs > 0 else '-':>10}{mu:>+10.4f}{('OUI' if ok else 'NON'):>9}")
    print(f"\n  {hits}/3 signes corrects.", end=" ")
    if hits == 3:
        print("Succes tel que pre-enregistre (p=0.125 sous binomiale a 0.5).")
    elif hits == 2:
        print("DEMI-ECHEC : 2/3 vaut p=0.500 sous binomiale, ne prouve rien.")
    else:
        print("ECHEC : l'ecran hors ligne ne predit pas l'end-to-end.")

    print("\n" + "=" * 78)
    print("Detail par budget et beta -- descriptif, non pre-enregistre")
    print("=" * 78)
    for ds in DATASETS:
        f = frames[ds]
        print(f"\n--- {ds} ---")
        piv = (f.groupby(["budget_tag", "beta"])["delta"].mean()
                 .unstack("beta").round(4))
        print(piv.to_string())

    out = pd.concat([f.assign(dataset=ds) for ds, f in frames.items()],
                    ignore_index=True)
    out.drop(columns=["_merge"]).to_csv(A / "alloc_restored_paired.csv.gz", index=False)
    print("\nSorti : alloc_restored_paired.csv.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
