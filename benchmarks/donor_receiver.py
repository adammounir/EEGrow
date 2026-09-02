"""La matrice donneur-receveur : un sujet entraîne, tous les autres testent.

Claim 2 du chantier ICLR (réunion Sylvain du 01/09) : la taille à laquelle un réseau
growing s'arrête sur un sujet prédit la valeur de ce sujet comme **donnée
d'entraînement** -- et le prédit mieux que l'accuracy de ce sujet.

Le protocole tient en une phrase : pour chaque sujet ``d``, entraîner un modèle sur SES
données seules, puis mesurer ce que ce modèle vaut sur chacun des ``N-1`` autres. La
ligne ``d`` de la matrice est la qualité de ``d`` comme donneur ; la colonne ``r`` est
la difficulté de ``r`` comme receveur.

CE QUE ÇA COÛTE, ET POURQUOI CE N'EST PAS N².
---------------------------------------------
Une case ``(d, r)`` est une **inférence**, pas un entraînement : le modèle de ``d`` est
déjà là. Le coût est donc de ``N`` fits (x seeds), pas de ``N²`` -- 52 x 3 = 156 fits
sur cho2017, pas 2704. C'est ce qui permet de faire tourner le dataset COMPLET, et le
faire complet n'est pas un luxe : à n=52 le MDE sur une corrélation est rho=0.38, à
n=10 il monte à 0.76. Un sous-échantillon « pour économiser » achèterait un nul
ininterprétable (cf. [[underpowered-not-null]] et l'étage 0, ``donor_predictor.py``).

POURQUOI cho2017 ET PAS UN AUTRE
--------------------------------
Trois mesures indépendantes de l'étage 0 désignent le même dataset :

* fiabilité   -- ICC_k = 0.86 sur la moyenne des 15 réplicats de la sonde ;
* non-redondance -- 98 % de la variance de ``params_end`` est indépendante de
  l'accuracy et du nombre d'essais (R² = 0.042, rho = +0.042) ;
* puissance   -- n = 52, le plus grand des datasets où les deux premières tiennent.

Sur lee2019_mi (rho = +0.65) et physionetmi (rho = +0.52) la taille EST en grande
partie l'accuracy, donc « meilleur prédicteur que l'accuracy » y serait une phrase vide.
schirrmeister2017 est hors-jeu (CV inter-sujets 1,9 %, ICC_k nul : tout le monde sature
à la même taille).

CE QUE CE SCRIPT NE DÉCIDE PAS
------------------------------
Il produit la matrice, pas la conclusion. Le prédicteur testé ensuite n'est PAS le
``params_end`` du fit donneur qu'on lit ici -- un fit isolé a un ICC(1) de 0.29 et ne
prédit rien -- mais la moyenne des 15 réplicats de la sonde within_session déjà
mesurée par la campagne. Les colonnes ``params_end``/``width_end`` écrites ici servent
de contrôle (le donneur s'arrête-t-il à la même taille quand il voit 100 % de ses
essais au lieu de 80 % ?), pas de prédicteur principal. Voir
``analysis/donor_matrix.py``.

DEUX CHOIX DE PROTOCOLE, EXPLICITES
-----------------------------------
1. **Le donneur donne tout.** Le fit voit 100 % des essais de ``d`` (moins le split
   interne de 20 % que skorch prélève pour la sélection d'époque -- il est nécessaire,
   c'est lui qui décide quel modèle est rendu). La sonde within_session, elle, en voit
   80 % par fold. C'est délibéré : la question posée est « que vaut ce sujet comme jeu
   d'entraînement », et un donneur qui garde 20 % de ses données ne répond pas à
   celle-là. La conséquence -- la taille mesurée ici n'est pas exactement celle de la
   sonde -- est mesurable et se lit dans les colonnes de contrôle.

2. **Le protocole d'entraînement corrigé, passé explicitement.** ``config.yaml`` livre
   volontairement les valeurs perdantes (les basculer re-daterait toutes les figures
   archivées de v5) ; ``patience=200`` et ``selection_monitor=valid_acc`` interagissent
   (+0.0854, p=1.1e-07) et ne peuvent pas être passés séparément. Ils sont ci-dessous
   les défauts de CE script, parce qu'ici rien d'historique n'en dépend -- mais ils
   sont écrits, pas hérités.

REPRISE
-------
Un CSV par (donneur, seed), écrit en dernier : relancer la même commande saute ce qui
est fait. C'est ce qui permet de sonder deux donneurs, de mesurer, puis de lancer le
reste sans rien recalculer.

Usage::

    python benchmarks/donor_receiver.py --out /scratch/amounir/dxr/cho2017 \
        --cache /scratch/amounir/moabb_cache --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipelines import build_pipeline  # noqa: E402
from utils import (  # noqa: E402
    cap_cuda_fraction,
    logger,
    pick_device,
    provenance,
    set_seed,
)

CONFIG = HERE / "config"


def load_cfgs(model: str, patience: int, selection_monitor: str,
              max_epochs: int | None) -> tuple[dict, dict, dict]:
    """Les configs Hydra de la campagne, lues telles quelles puis surchargées.

    Relire les YAML plutôt que recopier les valeurs : le modèle sonde est défini par
    ``config/model/grow_shallow.yaml`` (largeur de départ 8, cible 40, ``grow_every``
    5), et une deuxième copie de ces nombres ici dériverait du jour où quelqu'un
    touche à l'un des deux. Idem pour le paradigme et le rééchantillonnage, qui font
    la comparabilité avec la sonde within_session.
    """
    from omegaconf import OmegaConf

    root = OmegaConf.load(CONFIG / "config.yaml")
    mcfg = OmegaConf.to_container(OmegaConf.load(CONFIG / "model" / f"{model}.yaml"),
                                  resolve=True)
    tcfg = OmegaConf.to_container(root.train, resolve=True)
    pcfg = OmegaConf.to_container(root.paradigm, resolve=True)
    tcfg["patience"] = int(patience)
    tcfg["selection_monitor"] = str(selection_monitor)
    if max_epochs is not None:
        tcfg["max_epochs"] = int(max_epochs)
    return mcfg, tcfg, pcfg


def load_subjects(dataset_name: str, pcfg: dict, cache: str | None,
                  subjects: list[int] | None) -> tuple[dict, dict]:
    """Toutes les époques du dataset, une fois, en float32, sujet par sujet.

    MOABB re-dérive les époques des fichiers bruts à chaque appel ; ici il y en aurait
    52 x 52. On paie le filtrage/rééchantillonnage/époquage UNE fois (le cache disque
    de la campagne rend même ce passage-là bon marché) et tout le reste est de
    l'indexation mémoire.

    float32 dès le chargement, pas au moment du fit : c'est le type que consomment les
    convolutions, MOABB sert du float64, et 52 sujets en double dépassent ce qu'on veut
    tenir sur un nœud partagé pour rien.
    """
    import moabb.datasets as mds
    import moabb.paradigms as mpar
    from omegaconf import OmegaConf

    dcfg = OmegaConf.to_container(
        OmegaConf.load(CONFIG / "dataset" / f"{dataset_name}.yaml"), resolve=True)
    dataset = getattr(mds, dcfg["moabb_class"])(**(dcfg.get("kwargs") or {}))
    # Même correctif que le runner de la campagne : MOABB 1.5 filtre les sessions sur
    # `_selected_sessions` avec un décalage de 1 sur la famille Lee2019 et en perd la
    # moitié en silence. No-op partout ailleurs, mais il n'y a pas de raison que ce
    # script serve des époques différentes de celles que la sonde a vues.
    if getattr(dataset, "_selected_sessions", None) is not None:
        dataset._selected_sessions = None
    pkw = {}
    for k in ("resample", "tmin", "tmax"):
        if dcfg.get(k) is not None:
            pkw[k] = float(dcfg[k])
    paradigm = getattr(mpar, dcfg["paradigm"])(
        fmin=float(pcfg["fmin"]), fmax=float(pcfg["fmax"]), **pkw)

    cc = None
    if cache:
        cc = {"save_raw": False, "save_epochs": True, "save_array": True, "use": True,
              "overwrite_raw": False, "overwrite_epochs": False,
              "overwrite_array": False, "path": str(Path(cache).expanduser())}

    subs = list(subjects) if subjects else list(dataset.subject_list)
    data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    labels: set[str] = set()
    t0 = time.time()
    for s in subs:
        X, y, _meta = paradigm.get_data(dataset=dataset, subjects=[int(s)],
                                        **({"cache_config": cc} if cc else {}))
        data[int(s)] = (np.asarray(X, dtype="float32"), np.asarray(y))
        labels |= set(np.unique(y).tolist())
        logger.info("loaded subject %s: %s trials, %s chans, %s times (%.0fs elapsed)",
                    s, X.shape[0], X.shape[1], X.shape[2], time.time() - t0)

    # Un encodage de classes GLOBAL, pas par sujet. Un modèle entraîné sur le donneur
    # est appliqué au receveur : si `left_hand` valait 0 chez l'un et 1 chez l'autre,
    # le transfert serait mesuré à 1 - accuracy, et la matrice serait pleine de scores
    # sous la chance sans qu'aucune ligne ne soit fausse.
    classes = sorted(labels)
    enc = {c: i for i, c in enumerate(classes)}
    for s, (X, y) in data.items():
        yi = np.array([enc[v] for v in y], dtype="int64")
        if len(np.unique(yi)) != len(classes):
            raise RuntimeError(
                f"subject {s} carries {len(np.unique(yi))} of the {len(classes)} "
                "classes: a donor or receiver with a missing class makes its row or "
                "column of the matrix incomparable to the others")
        data[s] = (X, yi)
    n_times = int(next(iter(data.values()))[0].shape[2])
    # Même ordre que le runner de la campagne : rééchantillonnage explicite > sfreq
    # déclaré > dérivé de la fenêtre d'époque. Pas de constante de repli : `sfreq` est
    # un argument de module pour les arms `needs_sfreq` (sccnet, eegnex), et un 250
    # supposé sur un dataset servi à 160 Hz construirait un réseau différent sans que
    # rien ne le signale.
    interval = getattr(dataset, "interval", None)
    if dcfg.get("resample") is not None:
        sfreq = float(dcfg["resample"])
    elif dcfg.get("sfreq"):
        sfreq = float(dcfg["sfreq"])
    elif interval and (interval[1] - interval[0]) > 0:
        sfreq = float(round(n_times / (interval[1] - interval[0])))
    else:
        raise RuntimeError(f"cannot determine sfreq for {dcfg['name']}")
    meta = {"dataset": dcfg["name"], "classes": classes,
            "n_chans": int(next(iter(data.values()))[0].shape[1]),
            "n_times": n_times, "sfreq": sfreq,
            "load_seconds": round(time.time() - t0, 1)}
    return data, meta


def transfer_scores(pipeline, X: np.ndarray, y: np.ndarray,
                    n_classes: int) -> dict[str, float]:
    """Ce que le modèle du donneur vaut sur un receveur : accuracy et ROC-AUC.

    Les deux, et pas seulement la première, parce que la campagne score cho2017 en
    ROC-AUC (le défaut MOABB en binaire) et que la matrice doit rester comparable aux
    colonnes ``score`` déjà mesurées. L'accuracy est gardée parce qu'elle est ce que la
    narrative « sélectionner un jeu d'entraînement » optimisera au bout du compte.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    out = {"acc": float(accuracy_score(y, pipeline.predict(X)))}
    try:
        proba = pipeline.predict_proba(X)
        out["roc_auc"] = float(
            roc_auc_score(y, proba[:, 1]) if n_classes == 2
            else roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except Exception as exc:  # pragma: no cover - diagnostic, never fatal
        logger.warning("roc_auc unavailable: %s", exc)
        out["roc_auc"] = float("nan")
    return out


def run_donor(donor: int, seed: int, data: dict, meta: dict, mcfg: dict, tcfg: dict,
              out_dir: Path, device: str) -> Path:
    """Un fit sur le donneur, puis une inférence sur chacun des autres sujets.

    Écrit ``donor<D>__seed<S>.csv`` EN DERNIER, après le JSONL du fit : la présence du
    CSV est le seul témoin de complétude que la reprise consulte, donc il ne doit
    jamais exister avant que tout le reste soit sur le disque.
    """
    stem = f"donor{donor}__seed{seed}"
    csv = out_dir / f"{stem}.csv"
    if csv.exists():
        logger.info("skip %s (already done)", stem)
        return csv

    set_seed(int(seed))
    Xd, yd = data[donor]
    n_classes = len(meta["classes"])
    record_path = out_dir / f"{stem}__fits.jsonl"
    record_path.unlink(missing_ok=True)  # une reprise ne doit pas empiler deux fits
    pipeline = build_pipeline(
        mcfg, tcfg, n_chans=meta["n_chans"], n_times=meta["n_times"],
        n_outputs=n_classes, sfreq=meta["sfreq"], device=device,
        seed=int(seed), record_path=record_path)

    t0 = time.time()
    pipeline.fit(Xd, yd)
    fit_seconds = time.time() - t0

    # Le même instrument que la campagne : `params_end` est relevé par FitRecorder
    # APRÈS `RestoreBestModel`, donc il décrit le modèle réellement rendu -- qui peut
    # être plus étroit que la dernière époque de la courbe. Le recompter ici à la main
    # serait une deuxième définition de la quantité qui porte tout le papier.
    rec = {}
    if record_path.exists():
        lines = [ln for ln in record_path.read_text().splitlines() if ln.strip()]
        if lines:
            rec = json.loads(lines[-1])

    t1 = time.time()
    rows = []
    for r, (Xr, yr) in data.items():
        sc = transfer_scores(pipeline, Xr, yr, n_classes)
        rows.append({
            "donor": donor, "receiver": r, "seed": seed,
            "self": int(r == donor),  # la diagonale N'EST PAS un score de transfert
            "n_test": int(len(yr)),
            **sc,
        })
    infer_seconds = time.time() - t1

    df = pd.DataFrame(rows)
    df["dataset"] = meta["dataset"]
    df["model"] = mcfg.get("label")
    df["n_train"] = int(len(yd))
    df["chance"] = 1.0 / n_classes
    for k in ("params_start", "params_end", "width_start", "width_end", "epochs",
              "restored_epoch", "stop_reason"):
        df[k] = rec.get(k)
    df["fit_seconds"] = round(fit_seconds, 2)
    df["infer_seconds"] = round(infer_seconds, 2)
    df["device"] = device
    df["patience"] = tcfg["patience"]
    df["selection_monitor"] = tcfg["selection_monitor"]
    for k, v in provenance().items():
        df[k] = v
    df.to_csv(csv, index=False)
    logger.info("%s: fit %.1fs (%s epochs, params %s -> %s), transfer %.1fs, "
                "mean roc_auc off-diagonal %.4f",
                stem, fit_seconds, rec.get("epochs"), rec.get("params_start"),
                rec.get("params_end"), infer_seconds,
                float(df.loc[df["self"] == 0, "roc_auc"].mean()))
    return csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="cho2017")
    p.add_argument("--model", default="grow_shallow")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--donors", type=int, nargs="+", default=None,
                   help="restreindre les LIGNES de la matrice (sonde de coût). Les "
                        "colonnes restent tous les sujets du dataset.")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None, help="MOABB cache_config path")
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--selection-monitor", default="valid_acc")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--threads", type=int, default=None,
                   help="threads torch par processus. Sur GPU laisser à 1 (le défaut "
                        "hérité de la campagne) ; sur CPU c'est le facteur qui décide "
                        "du temps de fit, et la valeur 1 des scripts de la campagne "
                        "est un réglage de CO-TENANCE GPU, pas une recommandation.")
    p.add_argument("--shard", type=int, default=0,
                   help="index de ce processus parmi --n-shards (partage les fits)")
    p.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg, tcfg, pcfg = load_cfgs(a.model, a.patience, a.selection_monitor,
                                 a.max_epochs)
    if mcfg["kind"] == "ml":
        raise SystemExit("the probe has to be a deep arm: an ML pipeline has no "
                         "params_end, which is the predictor this matrix tests")
    device = pick_device(mcfg)
    cap_cuda_fraction()
    if a.threads:
        import torch
        torch.set_num_threads(int(a.threads))
    import torch as _t
    logger.info("device=%s model=%s torch_threads=%d protocol: patience=%s "
                "selection=%s", device, a.model, _t.get_num_threads(),
                tcfg["patience"], tcfg["selection_monitor"])

    # Les colonnes sont TOUJOURS tous les sujets, même quand --donors restreint les
    # lignes : une sonde de coût qui ne chargerait que ses deux donneurs mesurerait un
    # temps de transfert faux, et c'est justement le nombre qu'elle sert à mesurer.
    data, meta = load_subjects(a.dataset, pcfg, a.cache, None)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("data ready: %d subjects, %d chans, %d times, %.0fs",
                len(data), meta["n_chans"], meta["n_times"], meta["load_seconds"])

    donors = a.donors if a.donors else sorted(data)
    # (donneur, seed) est l'unité de travail, et le sharding se fait sur le couple :
    # sharder par sujet laisserait le dernier shard porter les trois seeds du sujet le
    # plus lent, alors que le makespan est ce qu'on essaie de réduire.
    units = [(d, s) for d in donors for s in a.seeds]
    units = [u for i, u in enumerate(units) if i % max(a.n_shards, 1) == a.shard]
    logger.info("shard %d/%d: %d (donor, seed) units", a.shard, a.n_shards, len(units))

    t0 = time.time()
    for i, (d, s) in enumerate(units, 1):
        run_donor(d, s, data, meta, mcfg, tcfg, out_dir, device)
        done = time.time() - t0
        logger.info("progress %d/%d units, %.1f min elapsed, ETA %.1f min",
                    i, len(units), done / 60, done / i * (len(units) - i) / 60)
    logger.info("shard done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
