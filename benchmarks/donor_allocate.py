"""Étage 5 : sélectionner n'est pas un choix binaire, c'est le bout d'un continuum.

CE QUE LES ÉTAGES 3 ET 4 ONT LAISSÉ OUVERT
------------------------------------------
L'étage 3 compare cinq règles qui choisissent K donneurs et les entraînent à plein.
L'étage 4 ajoute `all_pool`, qui prend tout le vivier à plein. Ce sont **deux points
d'un même axe**, et l'axe n'a jamais été parcouru : entre « K sujets, tous leurs
essais » et « M sujets, tous leurs essais », personne n'a mesuré « M sujets, une part
de leurs essais chacun ».

Cette lacune n'est pas cosmétique, elle porte la seule conclusion large du dossier.
`all_pool` gagne 45 comparaisons sur 45, mais il a **plus de sujets ET plus d'essais**
que les règles. Le docstring de `donor_select.py` le dit lui-même : « la comparaison
mélange plus de données et meilleures données ». Tant que les deux bougent ensemble,
« la diversité des sujets paie » et « le volume de données paie » sont indiscernables,
et ce sont deux conclusions différentes -- l'une dit d'enrôler plus de participants,
l'autre dit d'enregistrer plus longtemps les mêmes.

L'IDÉE : UNE SEULE MACHINERIE QUI CONTIENT LES DEUX EXTRÊMES
------------------------------------------------------------
Toute méthode d'utilisation d'un vivier revient à **répartir un budget d'essais** sur
ses M donneurs. On fixe le budget B, et on répartit selon un poids par donneur

    w_i ∝ exp(beta * z_i),   z_i = z-score d'une mesure de la sonde sur le vivier du pli

puis n_i = B * w_i, plafonné au nombre d'essais dont le donneur dispose (le surplus est
redistribué -- un remplissage par paliers, sinon un beta élevé « alloue » à un sujet
plus d'essais qu'il n'en possède et le budget n'est plus tenu).

Les deux protocoles déjà publiés sont des cas particuliers, et c'est ce qui rend le
balayage lisible :

    beta = 0        répartition uniforme sur TOUS les donneurs -- le bras manquant
    beta -> +inf    tout le budget aux meilleurs, plafonné : c'est la sélection top-K
    B = B_total     aucun essai perdu : la pondération de `all_pool`, pas sa sélection

`beta` interpole donc continûment entre l'étage 4 (beta=0, B total) et l'étage 3
(beta grand, B = K sujets). La question « faut-il sélectionner ? » devient « où est
l'optimum sur cet axe ? », qui est mesurable au lieu d'être tranchée par deux points.

LES DEUX CONTRASTES QUE ÇA REND ENFIN APPARIÉS
----------------------------------------------
1. **À BUDGET ÉGAL** (B = celui d'un pool de K sujets) : beta=0 étale ce même budget sur
   les M donneurs, la règle top-K le concentre sur K. Même nombre d'essais, même pli,
   mêmes seeds, même architecture. La seule chose qui diffère est **sur combien de
   sujets** le budget est réparti. C'est le contrôle que l'étage 4 n'avait pas.
2. **À BUDGET PLEIN** (B = tous les essais du vivier) : beta=0 EST `all_pool`, et
   beta>0 le repondère sans rien jeter. Si un beta>0 y bat beta=0, alors la mesure de la
   sonde sert à quelque chose d'opérationnel -- ce que la sélection dure n'a jamais
   réussi à montrer, et pour une raison qu'on comprend maintenant : elle payait la
   perte de sujets plus cher que ce que le tri rapportait.

CE QUI EST TENU CONSTANT, ET POURQUOI ÇA DÉCIDE DE LA VALIDITÉ
---------------------------------------------------------------
Les plis viennent de `donor_select.folds_of`, IMPORTÉE et non recopiée : les sujets de
test sont exactement ceux des campagnes déjà faites, donc les cellules d'ici
s'apparient à celles de `dsel2/` et `dsel_all/` sujet par sujet. Une réimplémentation
« équivalente » suffirait à casser cet appariement en silence.

Le budget est en ESSAIS, jamais en sujets, et il est vérifié après coup : la colonne
`n_train` du CSV doit valoir B à l'arrondi près. Le sous-échantillonnage par donneur est
**stratifié par classe**, sinon un tirage uniforme déséquilibre les classes d'un donneur
et la comparaison mesurerait aussi ce déséquilibre.

La mesure de tri (`--measure`) ne touche jamais le test : `params_probe` et `acc_probe`
viennent de la sonde within_session, sujet par sujet, comme à l'étage 3. Aucune fuite.

STATUT
------
POST-HOC et DESCRIPTIF, comme `all_pool` et pour la même raison : écrit après lecture
des étages 3, 4 et de l'analyse d'enveloppe. Il n'amende aucun endpoint pré-enregistré
et n'entre dans aucune famille de multiplicité. Il fournit l'axe le long duquel les
résultats déjà publiés se lisent -- il ne les renverse pas.

Usage::

    python benchmarks/donor_allocate.py --dataset cho2017 \\
        --ranking benchmarks/ranking_cho2017.csv --out /scratch/amounir/dalloc/cho2017 \\
        --cache /scratch/amounir/moabb_cache --budget-k 3 5 10 --budget-full \\
        --beta 0 1 2 4 --measure acc_probe params_probe --folds 4 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from donor_receiver import load_cfgs, load_subjects, transfer_scores  # noqa: E402
from donor_select import folds_of, load_ranking  # noqa: E402
from pipelines import build_pipeline  # noqa: E402
from utils import cap_cuda_fraction, logger, pick_device, provenance, set_seed  # noqa: E402

# `restored_probe`, et pas `restored_epoch` : c'est le nom que `donor_select_prep.py`
# écrit dans `ranking_*.csv` (la moyenne des 15 réplicats), et le fichier de classement
# est la seule source admissible. Le tuple portait l'autre nom, celui de la colonne brute
# de `gd_fits`, si bien que `--measure restored_epoch` se serait fait refuser à chaque
# lancement par le garde-fou plus bas -- une mesure déclarée mais inatteignable.
MEASURES = ("acc_probe", "params_probe", "restored_probe")


def allocate(weights: np.ndarray, avail: np.ndarray, budget: int) -> np.ndarray:
    """Répartir ``budget`` essais selon ``weights``, sans dépasser ``avail`` par donneur.

    Remplissage par paliers : on distribue au prorata, on écrête ceux qui débordent, et
    on redistribue le surplus sur les donneurs encore ouverts. Sans cette redistribution
    un beta élevé « alloue » 3 000 essais à un sujet qui en a 200 et le budget total
    n'est plus tenu -- la comparaison à budget égal, qui est tout l'intérêt du fichier,
    serait alors fausse sans que rien ne le signale.
    """
    budget = int(min(budget, avail.sum()))
    n = np.zeros(len(weights), dtype=int)
    open_ = np.ones(len(weights), dtype=bool)
    rem = budget
    while rem > 0 and open_.any():
        w = np.where(open_, weights, 0.0)
        if w.sum() <= 0:
            w = open_.astype(float)
        share = np.floor(rem * w / w.sum()).astype(int)
        if share.sum() == 0:
            # Reste plus petit que le nombre de donneurs ouverts : on l'attribue un par
            # un, du poids le plus fort au plus faible, pour que le budget tombe juste.
            for j in np.argsort(-w):
                if rem <= 0:
                    break
                if open_[j]:
                    n[j] += 1
                    rem -= 1
            open_ = n < avail
            continue
        new = np.minimum(n + share, avail)
        rem -= int((new - n).sum())
        n = new
        open_ = n < avail
    return n


def take_trials(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` indices tirés sans remise, en préservant les proportions de classes.

    Un tirage uniforme déséquilibrerait les classes d'un donneur d'autant plus qu'on lui
    prend peu d'essais -- et c'est précisément le régime des petits budgets. Le contraste
    mesurerait alors un déséquilibre de classes autant qu'une répartition.
    """
    if n >= len(y):
        return np.arange(len(y))
    idx = []
    classes, counts = np.unique(y, return_counts=True)
    quota = np.floor(n * counts / counts.sum()).astype(int)
    while quota.sum() < n:                       # le reste va aux classes majoritaires
        quota[np.argmax(counts - quota)] += 1
    for c, q in zip(classes, quota):
        pos = np.flatnonzero(y == c)
        idx.append(rng.choice(pos, size=int(min(q, len(pos))), replace=False))
    return np.sort(np.concatenate(idx))


def pool_xy_alloc(data: dict, subs: list[int], n_per: np.ndarray,
                  seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """L'union des essais RÉPARTIS : n_per[i] essais du donneur subs[i]."""
    rng = np.random.default_rng(seed)
    Xs, ys, used = [], [], {}
    for s, n in zip(subs, n_per):
        if n <= 0:
            continue
        X, y = data[s]
        take = take_trials(y, int(n), rng)
        Xs.append(X[take])
        ys.append(y[take])
        used[int(s)] = int(len(take))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), used


def weights_of(pool: pd.DataFrame, measure: str, beta: float) -> np.ndarray:
    """exp(beta * z), z le z-score de ``measure`` SUR LE VIVIER DU PLI.

    Le z-score est recalculé par pli et non une fois sur le dataset : le vivier change
    d'un pli à l'autre, et normaliser sur l'ensemble ferait entrer les sujets de test
    dans la statistique qui pondère l'entraînement. C'est une fuite ténue mais réelle.
    """
    if beta == 0:
        return np.ones(len(pool), dtype=float)
    x = pool[measure].to_numpy(dtype=float)
    sd = x.std(ddof=0)
    z = (x - x.mean()) / (sd if sd > 0 else 1.0)
    z = z - z.max()                              # stabilité numérique, exp <= 1
    return np.exp(float(beta) * z)


def run_cell(fold: int, budget_tag: str, beta: float, measure: str, seed: int,
             test_subs: list[int], pool: pd.DataFrame, budget: int,
             data: dict, meta: dict, mcfg: dict, tcfg: dict,
             out_dir: Path, device: str) -> Path:
    """Un entraînement sur le budget réparti, puis un score PAR SUJET DE TEST."""
    stem = f"f{fold}__b{budget_tag}__beta{beta:g}__{measure}__seed{seed}"
    csv = out_dir / f"{stem}.csv"
    if csv.exists():
        logger.info("skip %s (already done)", stem)
        return csv

    subs = [int(s) for s in pool["subject"]]
    avail = np.array([len(data[s][1]) for s in subs], dtype=int)
    n_per = allocate(weights_of(pool, measure, beta), avail, budget)

    set_seed(int(seed))
    n_classes = len(meta["classes"])
    record_path = out_dir / f"{stem}__fits.jsonl"
    record_path.unlink(missing_ok=True)
    pipeline = build_pipeline(
        mcfg, tcfg, n_chans=meta["n_chans"], n_times=meta["n_times"],
        n_outputs=n_classes, sfreq=meta["sfreq"], device=device,
        seed=int(seed), record_path=record_path)

    # La graine du sous-échantillonnage inclut le pli et le budget mais PAS beta : deux
    # beta différents qui allouent le même n_i à un donneur lui prennent alors les mêmes
    # essais, et la différence mesurée entre deux beta ne contient pas le bruit d'un
    # re-tirage inutile.
    Xtr, ytr, used = pool_xy_alloc(data, subs, n_per, seed=1000 * fold + seed)
    t0 = time.time()
    pipeline.fit(Xtr, ytr)
    fit_seconds = time.time() - t0

    rec = {}
    if record_path.exists():
        lines = [ln for ln in record_path.read_text().splitlines() if ln.strip()]
        if lines:
            rec = json.loads(lines[-1])

    rows = []
    for s in test_subs:
        Xr, yr = data[s]
        rows.append({"fold": fold, "rule": f"alloc_b{budget_tag}", "draw": 0,
                     "seed": seed, "rep": -1, "test_subject": s, "n_test": int(len(yr)),
                     **transfer_scores(pipeline, Xr, yr, n_classes)})
    df = pd.DataFrame(rows)
    df["dataset"] = meta["dataset"]
    df["model"] = mcfg.get("label")
    df["beta"] = beta
    df["measure"] = measure
    df["budget_tag"] = budget_tag
    df["budget_target"] = int(budget)
    df["n_donors_pool"] = len(subs)
    # Le nombre de donneurs qui reçoivent AU MOINS un essai : c'est la vraie variable
    # « combien de sujets » du continuum, et elle décroît quand beta monte sans qu'aucun
    # paramètre ne la fixe. Sans cette colonne l'axe du balayage serait invisible.
    df["n_donors_used"] = int((n_per > 0).sum())
    df["k"] = int((n_per > 0).sum())
    df["chosen"] = ";".join(str(s) for s, n in zip(subs, n_per) if n > 0)
    df["alloc"] = ";".join(f"{s}:{n}" for s, n in zip(subs, n_per) if n > 0)
    df["n_train"] = int(len(ytr))
    df["chance"] = 1.0 / n_classes
    for k in ("params_start", "params_end", "width_start", "width_end", "epochs",
              "restored_epoch", "stop_reason"):
        df[k] = rec.get(k)
    df["fit_seconds"] = round(fit_seconds, 2)
    df["device"] = device
    df["node"] = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    df["gpu_name"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "cpu")
    df["patience"] = tcfg["patience"]
    df["selection_monitor"] = tcfg["selection_monitor"]
    for k, v in provenance().items():
        df[k] = v
    df.to_csv(csv, index=False)
    logger.info("%s: %d essais visés %d, %d/%d donneurs servis, fit %.1fs, "
                "roc_auc moyen %.4f sur %d sujets tenus à l'écart",
                stem, len(ytr), budget, int((n_per > 0).sum()), len(subs),
                fit_seconds, float(df["roc_auc"].mean()), len(test_subs))
    return csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="cho2017")
    p.add_argument("--model", default="bd_shallow",
                   help="le modèle AVAL, fixe, comme à l'étage 3 : la croissance mesure "
                        "les données, elle ne les classifie pas.")
    p.add_argument("--ranking", required=True, type=Path)
    p.add_argument("--budget-k", type=int, nargs="*", default=[3, 5, 10],
                   help="budgets exprimés en ÉQUIVALENT-SUJETS : B = K x (essais moyens "
                        "par donneur du vivier). Prendre les mêmes K que l'étage 3 est "
                        "ce qui rend les cellules appariables à celles de dsel2/.")
    p.add_argument("--budget-full", action="store_true",
                   help="ajoute le budget PLEIN (tous les essais du vivier). beta=0 y "
                        "reproduit `all_pool` : c'est le contrôle de cohérence du "
                        "fichier, et il doit tomber sur le score déjà mesuré.")
    p.add_argument("--beta", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--measure", nargs="+", default=["acc_probe", "params_probe"],
                   choices=list(MEASURES))
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--selection-monitor", default="valid_acc")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="imprime le plan ET la répartition résultante par beta, sans "
                        "entraîner. C'est là qu'on voit si un beta est si grand que "
                        "l'allocation dégénère en top-1, ou si deux beta allouent la "
                        "même chose et qu'une des deux cellules est du gâchis.")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg, tcfg, pcfg = load_cfgs(a.model, a.patience, a.selection_monitor, a.max_epochs)
    rank = load_ranking(a.ranking, sorted(pd.read_csv(a.ranking)["subject"].astype(int)))
    # Une mesure absente ou trouée doit REFUSER ici, avant le moindre fit : un NaN dans
    # la colonne de tri traverse `weights_of` (exp(nan) = nan), et `allocate` répartirait
    # alors le budget sur des poids nan sans que rien ne le signale -- la cellule
    # tournerait, écrirait un CSV, et testerait une allocation qui n'est pas celle
    # déclarée. C'est le mode de défaillance de [[gates-must-refuse-not-warn]].
    for m in a.measure:
        if m not in rank.columns:
            raise SystemExit(
                f"--measure {m} : colonne absente de {a.ranking}. Régénérer le "
                f"classement avec analysis/donor_select_prep.py (il écrit "
                f"`params_probe`, `acc_probe`, `restored_probe`).")
        if rank[m].isna().any():
            bad = rank.loc[rank[m].isna(), "subject"].tolist()
            raise SystemExit(f"--measure {m} : valeur manquante pour les sujets {bad}.")
    subjects = sorted(rank["subject"].astype(int))
    fold_tests = folds_of(subjects, a.folds)

    # Le plan est déterministe et construit avant tout chargement de données : le
    # --dry-run coûte zéro et inspecte exactement ce que la campagne fera.
    plan = []
    for fi, test in enumerate(fold_tests):
        pool = rank[~rank["subject"].isin(test)]
        for beta in a.beta:
            for measure in a.measure:
                # beta=0 ignore la mesure : une seule cellule, étiquetée `uniform`.
                # Sans ce court-circuit on paierait |mesures| fois le même fit.
                if beta == 0 and measure != a.measure[0]:
                    continue
                tag_m = "uniform" if beta == 0 else measure
                for seed in a.seeds:
                    for k in a.budget_k:
                        plan.append((fi, str(k), float(beta), tag_m, seed, test, pool))
                    # LE BUDGET PLEIN N'ADMET QU'UN SEUL beta, ET LE --dry-run L'A
                    # MONTRÉ AVANT LA SOUMISSION. Quand B égale la somme des essais
                    # disponibles, le remplissage par paliers sature TOUS les donneurs à
                    # leur plafond : l'allocation devient l'identité et les sept beta
                    # produisent la même cellule, bit pour bit. Ce n'était pas 84 fits
                    # informatifs, c'étaient 84 copies d'`all_pool`.
                    #
                    # Ce n'est pas un défaut de l'implémentation mais une propriété du
                    # problème : on ne peut pas repondérer un vivier qu'on est obligé
                    # d'épuiser, sauf à dupliquer des essais -- ce qui ferait entrer des
                    # doublons dans le split de validation interne de skorch et
                    # biaiserait le choix d'époque. On s'y refuse. Le budget plein reste
                    # donc comme ANCRE (beta=0 doit retomber sur `all_pool`), et l'axe
                    # du continuum est porté par les budgets contraints, où beta mord.
                    if a.budget_full and beta == 0:
                        plan.append((fi, "full", float(beta), tag_m, seed, test, pool))
    plan.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
    mine = [c for i, c in enumerate(plan) if i % a.n_shards == a.shard]
    logger.info("plan : %d cellules au total, %d pour le shard %d/%d",
                len(plan), len(mine), a.shard, a.n_shards)

    if a.dry_run:
        # On charge seulement les COMPTES d'essais, pas les essais : l'allocation ne
        # dépend que d'eux, donc le plan complet est inspectable sans toucher au cache.
        print(f"{'pli':>4} {'budget':>7} {'beta':>5} {'mesure':>14} {'B visé':>8} "
              f"{'donneurs servis':>16} {'max n_i':>8} {'part du top-1':>14}")
        seen = set()
        # Sans les données on suppose des donneurs équilibrés : le dry-run sert à voir
        # la FORME de l'allocation, pas à la valider à l'essai près.
        per = int(rank["n_trials"].median()) if "n_trials" in rank.columns else 200
        for fi, tag, beta, measure, seed, test, pool in plan:
            key = (fi, tag, beta, measure)
            if key in seen:
                continue
            seen.add(key)
            subs = [int(s) for s in pool["subject"]]
            avail = np.full(len(subs), per, dtype=int)
            budget = avail.sum() if tag == "full" else int(tag) * per
            n = allocate(weights_of(pool, measure if beta else a.measure[0], beta),
                         avail, budget)
            print(f"{fi:>4} {tag:>7} {beta:>5g} {measure:>14} {budget:>8} "
                  f"{int((n > 0).sum()):>10}/{len(subs):<5} {n.max():>8} "
                  f"{n.max() / max(n.sum(), 1):>14.3f}")
        return 0

    device = pick_device(mcfg)
    cap_cuda_fraction()
    if a.threads:
        torch.set_num_threads(int(a.threads))
    data, meta = load_subjects(a.dataset, pcfg, a.cache, subjects)

    for fi, tag, beta, measure, seed, test, pool in mine:
        subs = [int(s) for s in pool["subject"]]
        avail = np.array([len(data[s][1]) for s in subs], dtype=int)
        # Le budget est ancré sur les essais RÉELLEMENT disponibles, pas sur une
        # constante : un dataset dont les sujets n'ont pas tous le même nombre d'essais
        # verrait sinon son « équivalent K sujets » dériver d'un pli à l'autre.
        budget = int(avail.sum()) if tag == "full" else \
            int(round(int(tag) * avail.mean()))
        run_cell(fi, tag, beta, measure, seed, test, pool, budget,
                 data, meta, mcfg, tcfg, out_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
