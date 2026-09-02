"""Claim 3 : sélectionner le jeu d'entraînement par `#params` bat-il l'aléatoire ?

Claim 2 (`donor_receiver.py` + `analysis/donor_matrix.py`) a montré une CORRÉLATION :
sur physionetmi, la taille à laquelle un réseau growing s'arrête sur un sujet classe ce
sujet comme donneur au-delà de ce que son accuracy prédit déjà (rho partielle +0.385
[+0.199, +0.548], n=109). Une corrélation classe ; elle ne dit pas que **sélectionner**
par cette variable produit un meilleur modèle. Ce script pose la question
interventionnelle, et c'est la seule des trois qui fait une méthode plutôt qu'une
observation.

LE PROTOCOLE EN UNE PHRASE
--------------------------
On coupe les 109 sujets en F plis ; pour chaque pli, les sujets du pli sont le TEST et
les autres forment le VIVIER ; dans le vivier on choisit K sujets selon une règle, on
entraîne UN modèle sur l'union de leurs essais, et on le score sur chaque sujet de test
séparément. Chaque sujet est ainsi testé exactement une fois, par un modèle qui ne l'a
jamais vu : l'unité d'analyse est le sujet de test, n=109, et les règles se comparent
APPARIÉES sur ces mêmes sujets (cf. [[unit-of-analysis-subject]]).

POURQUOI LE MODÈLE AVAL EST `bd_shallow` ET NON `grow_shallow`
--------------------------------------------------------------
C'est le choix de conception le plus important du fichier. La narrative n'est pas « un
réseau qui grandit classifie mieux » -- la campagne dit le contraire et ce papier-là
n'existe pas. Elle est « **un réseau qui grandit MESURE les données** ». Le growing sert
donc à produire le classement (il l'a déjà fait : la sonde within_session de la
campagne), et le modèle entraîné en aval est une architecture standard, fixe, que
n'importe qui utiliserait. Si la sélection marche dans ces conditions, le résultat se
transporte : il dit quelque chose sur les DONNÉES, pas sur un artefact de l'optimiseur
qui a servi à les mesurer. Accessoirement c'est aussi le régime le moins cher (pas
d'événement de croissance) et le plus proche de l'usage réel.

LES CINQ RÈGLES, ET CE QUE CHACUNE SERT À EXCLURE
-------------------------------------------------
`params_top`      -- le prédicteur déclaré. C'est lui qu'on veut voir gagner.
`resid_top`       -- les K plus grands RÉSIDUS de `#params` après régression sur
                     l'accuracy (en rangs). C'est le test le plus serré, parce que c'est
                     exactement la quantité que la rho partielle a mesurée : ce que la
                     taille sait du sujet et que son accuracy ne sait pas. Si une seule
                     règle doit gagner, c'est celle-là.
`acc_top`         -- le concurrent ennuyeux. Sélectionner par l'accuracy du sujet est ce
                     que tout le monde ferait sans ce papier ; battre l'aléatoire sans
                     battre `acc_top` ne fait pas une contribution.
`random`          -- la vraie ligne de base. R tirages par pli, pour avoir la
                     DISTRIBUTION de l'aléatoire et pas un seul tirage qui pourrait être
                     bon ou mauvais par chance.
`params_bottom`   -- le contrôle négatif, et il n'est pas décoratif. Si `params_top` bat
                     l'aléatoire, `params_bottom` doit le PERDRE : un effet qui ne
                     s'inverse pas quand on inverse le tri n'est pas un effet de tri,
                     c'est un effet de quelque chose d'autre qui a bougé en même temps.

CE QUI EST TENU CONSTANT, ET POURQUOI C'EST LA MOITIÉ DU PROTOCOLE
------------------------------------------------------------------
Toutes les règles choisissent **le même nombre K de sujets**. Sans ça on comparerait
« plus de données » à « meilleures données » et la conclusion serait vide. Les règles
partagent aussi le pli, la seed, l'architecture et le protocole d'entraînement : la
seule chose qui diffère entre deux fits appariés est QUELS sujets sont dedans.

La variable de sélection ne touche jamais le test. `params_probe` et `acc_probe` sont
mesurés par la campagne within_session, sujet par sujet, indépendamment les uns des
autres : trier le vivier avec elles n'utilise aucune information venue des sujets de
test. Il n'y a donc pas de fuite, et c'est vérifiable en lisant `load_ranking`.

--------------------------------------------------------------------------------
V2 (02/09, après-midi) : POURQUOI LE PREMIER PROTOCOLE NE POUVAIT PAS TRANCHER (c)
--------------------------------------------------------------------------------
La campagne v1 (job 514161, 72 cellules, `--candidates 0`) a répondu à (a) et (b) sans
ambiguïté -- 4 plis sur 4 concordants -- mais elle NE POUVAIT PAS répondre à (c), et le
défaut est de conception, pas de taille d'échantillon.

Une règle déterministe appliquée au vivier entier ne produit **qu'un seul pool par
pli**. Les 27 sujets d'un pli partagent donc exactement les mêmes 20 donneurs, et la
variance « sur quel pool cette règle est-elle tombée » est entièrement confondue avec le
pli. Le contraste `resid_top` - `acc_top` n'avait pas 109 réplicats mais **4**, et un
seul pli (le 3) suffisait à en renverser le signe : +0.008, +0.013, +0.008, **-0.019**.
Le bootstrap sur les sujets ne voyait rien de tout ça -- c'est exactement la faute de
niveau d'analyse de [[unit-of-analysis-subject]], transposée du sujet au POOL. Les 3
seeds ne sauvaient rien : elles font varier l'initialisation, jamais le pool.

LE CORRECTIF : `--candidates M`. À chaque réplicat on tire au hasard M sujets du vivier,
et **toutes les règles choisissent leurs K dans ce même jeu de candidats**. Deux
conséquences, et les deux comptent :

  * les règles déterministes deviennent stochastiques *par la même source de hasard que
    `random`* -- elles ont enfin une distribution de pools, donc l'unité de rééchan-
    tillonnage (le réplicat) est réplicable F x R fois au lieu de F ;
  * le jeu de candidats étant PARTAGÉ à l'intérieur d'un réplicat, deux règles y sont
    comparées sur le même ensemble de sujets disponibles. La différence mesurée est
    alors purement « quel critère a choisi quoi dedans », ce qui est la question, et non
    « quels sujets étaient disponibles », ce qui est du bruit.

L'ordre est calculé une fois par (pli, réplicat, règle) et les K sont pris en préfixe :
les pools sont donc **emboîtés** en K (K=5 ⊂ K=10 ⊂ K=20), y compris pour `random`, qui
utilise une permutation tronquée et non un tirage indépendant par K. Balayer K est ainsi
un axe « on ajoute des donneurs » propre, et non trois expériences sans rapport.

CE QUE LE BALAYAGE EN K TESTE, ET C'EST UNE PRÉDICTION FALSIFIABLE
------------------------------------------------------------------
v1 tournait à K=20 sur un vivier de 82, soit **24 % du vivier** : le régime où tous les
critères raisonnables ramassent à peu près les mêmes bons sujets. Si cette explication
est la bonne, alors le gain de n'importe quelle règle sur `random` doit **décroître**
quand K/M augmente, et les recouvrements entre règles doivent croître. Si le gain ne
décroît pas, mon explication est fausse et il faut le dire. Le balayage n'est donc pas
une pêche : c'est le test de la raison que j'ai avancée pour expliquer v1.

UNITÉ D'ANALYSE EN V2, ET CE QUE ÇA PERMET DE CONCLURE
------------------------------------------------------
Les 109 sujets sont le dataset ENTIER, pas un échantillon : pour la question « sur
physionetmi, la règle A bat-elle la règle B en espérance sur le tirage du pool ? », ils
sont fixes et l'unité de rééchantillonnage est le **réplicat**. C'est ce que fait
``analysis/donor_select2_analysis.py``. La généralisation à d'autres sujets ou d'autres
datasets n'est PAS couverte par ce protocole, quel que soit son n : elle demande la
réplication sur lee2019_mi / cho2017, et il faut l'écrire ainsi.

--------------------------------------------------------------------------------
LE BRAS `all_pool` (02/09, soir) : « ET SI ON NE SÉLECTIONNAIT PAS DU TOUT ? »
--------------------------------------------------------------------------------
Sylvain avait nommé TROIS comparateurs, pas deux : « tu pourrais comparer le
cross-subject où tu mets tout le monde dedans... que les cinq meilleurs... et des sujets
au hasard ». Les deux derniers sont dans les cinq règles depuis le début ; le premier
manquait, parce que `draw_candidates` plafonne les candidats strictement sous la taille
du vivier et que `--k` n'est jamais allé au-delà de 20 sur des viviers de 39 et 82.

C'est un manque de fond, pas un oubli cosmétique. Battre 5 sujets tirés au hasard est
facile ; **battre les 39 sujets réunis** est le test qui intéresse un relecteur, parce
que « tout mettre » est ce que fait tout le monde par défaut. Sans ce bras on peut
seulement dire « parmi les pools de taille K, ce critère-ci vaut mieux que celui-là »,
jamais « sélectionner vaut mieux que ne pas sélectionner ».

CE BRAS EST POST-HOC ET IL EST DÉCLARÉ TEL QUEL. Il est écrit APRÈS lecture des
résultats de claim 3 (physionetmi P(K=5) = -0.0377, cho2017 garde-fou G non tenu). Il
n'amende donc PAS l'endpoint primaire, il n'entre dans aucune correction de multiplicité
de la famille pré-enregistrée, et le mot « confirme » lui est interdit. Il est
DESCRIPTIF : il fournit la référence manquante contre laquelle lire les quatre règles,
et il peut au mieux recadrer leur interprétation, jamais la renverser.

DEUX PROPRIÉTÉS QUI LE SÉPARENT DES RÈGLES, ET QU'IL NE FAUT PAS EFFACER
------------------------------------------------------------------------
1. **K n'est pas apparié.** `all_pool` entraîne sur |vivier| sujets (39 ou 82) contre
   K ∈ {3,5,10,20}. Toute la section « CE QUI EST TENU CONSTANT » ci-dessus est donc
   VIOLÉE ici, délibérément : la comparaison mélange « plus de données » et « meilleures
   données ». C'est acceptable pour une référence -- la question posée est justement
   « est-ce que trier bat le fait d'avoir tout » -- et inacceptable pour un contraste
   entre critères. Ne jamais lire un écart `all_pool` - `params_top` comme un effet de
   critère.
2. **Il n'a pas de réplicats.** Le pool est déterministe étant donné le pli : il n'y a
   rien à tirer. Sa seule variance est celle de l'initialisation, donc S seeds par pli.
   L'unité de rééchantillonnage y est le pli (F=4), pas le réplicat -- ce qui suffit
   pour une référence descriptive et ne suffirait pas pour un test.

Sortie dans un répertoire SÉPARÉ (`/scratch/amounir/dsel_all/<dataset>`) et nom de
fichier au format v1 (`f0__all_pool__d0__seed3.csv`), pour la même raison que la
séparation v1/v2 : le glob de `donor_select2_analysis.py` ne peut structurellement pas
ramasser une cellule dont l'unité d'analyse diffère.

CE QUE CE SCRIPT NE DÉCIDE PAS
------------------------------
Il produit les scores, pas le verdict. Le test apparié règle par règle, le contraste
avec le contrôle négatif et l'intervalle bootstrap sont dans
``analysis/donor_select_analysis.py``. Un CSV par (pli, règle, tirage, seed), écrit en
dernier : relancer la même commande saute ce qui est fait.

Usage::

    python benchmarks/donor_select.py --dataset physionetmi \
        --out /scratch/amounir/dsel/physionetmi --cache /scratch/amounir/moabb_cache \
        --ranking benchmarks/analysis/ranking_physionetmi.csv --k 20 --folds 4
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
from pipelines import build_pipeline  # noqa: E402
from utils import cap_cuda_fraction, logger, pick_device, provenance, set_seed  # noqa: E402

RULES = ("params_top", "resid_top", "acc_top", "params_bottom", "random")


def load_ranking(path: Path, subjects: list[int]) -> pd.DataFrame:
    """Le classement des sujets, mesuré AILLEURS et lu ici tel quel.

    Le fichier est produit par ``analysis/donor_select_prep.py`` à partir de la sonde
    within_session de la campagne : une ligne par sujet, colonnes `params_probe` et
    `acc_probe`, chacune moyennée sur ses 15 réplicats (5 folds x 3 seeds). Le lire
    plutôt que le recalculer garantit que la variable de sélection est exactement celle
    dont claim 2 a mesuré la rho -- une deuxième dérivation ici pourrait diverger sans
    que rien ne le signale.
    """
    r = pd.read_csv(path)
    r["subject"] = pd.to_numeric(r["subject"], errors="coerce").astype("Int64")
    r = r[r["subject"].isin(subjects)].dropna(subset=["params_probe", "acc_probe"])
    missing = sorted(set(subjects) - set(r["subject"].astype(int)))
    if missing:
        raise SystemExit(
            f"{len(missing)} subjects have no probe ({missing[:8]}...): a subject "
            "without a selection variable cannot be in the pool, and dropping it "
            "silently would change K without saying so")
    return r.astype({"subject": int}).reset_index(drop=True)


def residual_rank(r: pd.DataFrame) -> np.ndarray:
    """`#params` débarrassé de ce que l'accuracy explique déjà, en rangs.

    Même transformation que ``partial_spearman`` : rangs des deux variables, régression
    linéaire de l'une sur l'autre, résidu. Trier sur ce résidu, c'est trier sur la seule
    part de la taille dont claim 2 a montré qu'elle porte une information propre -- un
    sujet est haut ici s'il grandit PLUS que son accuracy ne le laisserait attendre.
    """
    rp = pd.Series(r["params_probe"].to_numpy(float)).rank().to_numpy()
    ra = pd.Series(r["acc_probe"].to_numpy(float)).rank().to_numpy()
    return rp - np.polyval(np.polyfit(ra, rp, 1), ra)


def rank_candidates(cand: pd.DataFrame, rule: str,
                    rng: np.random.Generator) -> list[int]:
    """L'ORDRE COMPLET des candidats selon la règle, meilleur d'abord.

    Rendre l'ordre entier plutôt que les K premiers est ce qui rend les pools emboîtés
    en K : `chosen(k) = rank_candidates(...)[:k]`, donc le pool de K=5 est inclus dans
    celui de K=10. `random` n'échappe pas à la règle -- il rend une permutation, pas un
    tirage indépendant par K -- sans quoi l'axe K mélangerait « plus de donneurs » et
    « d'autres donneurs » et ne mesurerait plus rien.

    Les rangs sont recalculés SUR LE JEU DE CANDIDATS, pas hérités du dataset entier :
    le résidu d'un sujet dépend de la population à laquelle on le compare, et un
    réplicat qui aurait perdu les sujets extrêmes doit trier sur ce qui lui reste.
    """
    p = cand.reset_index(drop=True)
    if rule == "random":
        return [int(s) for s in rng.permutation(p["subject"].to_numpy())]
    if rule == "params_top":
        order = p["params_probe"].to_numpy(float)
    elif rule == "params_bottom":
        order = -p["params_probe"].to_numpy(float)
    elif rule == "acc_top":
        order = p["acc_probe"].to_numpy(float)
    elif rule == "resid_top":
        order = residual_rank(p)
    else:
        raise ValueError(f"unknown rule {rule}")
    idx = np.argsort(-order, kind="stable")
    return [int(s) for s in p.loc[idx, "subject"]]


def select(pool: pd.DataFrame, rule: str, k: int, rng: np.random.Generator) -> list[int]:
    """K sujets selon la règle -- CHEMIN V1, conservé au bit près.

    `random` garde `rng.choice` et n'emprunte PAS `rank_candidates`, qui rend une
    permutation. Les deux consomment le générateur différemment et ne donnent donc pas
    le même sous-ensemble pour la même graine : passer par la permutation ici aurait
    silencieusement changé les 6 tirages aléatoires par pli du job 514161, dont les
    résultats sont publiés. La v2 n'a pas ce problème -- elle n'existait pas encore.
    """
    p = pool.reset_index(drop=True)
    if rule == "random":
        return sorted(rng.choice(p["subject"].to_numpy(), size=k, replace=False).tolist())
    return sorted(rank_candidates(p, rule, rng)[:k])


def draw_candidates(pool: pd.DataFrame, m: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Les M candidats d'un réplicat, tirés UNE fois et partagés par toutes les règles.

    C'est la pièce qui répare (c) : sans elle une règle déterministe rend le même pool à
    chaque fois dans un pli donné et n'a donc aucune variance à rééchantillonner. Le
    partage entre règles est tout aussi essentiel que le tirage -- deux règles comparées
    sur des candidats différents diffèrent d'abord par ce qui leur était disponible.

    M=0 (ou M >= |vivier|) rend le vivier entier : c'est le mode v1, conservé pour que
    les 72 cellules du job 514161 restent reproductibles par ce même script.
    """
    if m <= 0 or m >= len(pool):
        return pool.reset_index(drop=True)
    idx = rng.choice(len(pool), size=m, replace=False)
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


def folds_of(subjects: list[int], n_folds: int, seed: int = 20260902) -> list[list[int]]:
    """Les plis de test. Partition, pas tirage : chaque sujet est testé exactement une fois.

    Une partition et non des tirages indépendants, parce que c'est elle qui donne
    n=109 différences appariées couvrant tout le dataset. Des tirages avec remise
    donneraient des sujets testés deux fois et une corrélation entre les « n » que le
    bootstrap sur les sujets ne saurait pas défaire.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(subjects))
    return [sorted(a.tolist()) for a in np.array_split(perm, n_folds)]


def pool_xy(data: dict, subs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """L'union des essais des K donneurs, dans l'ordre des sujets.

    Le mélange est laissé au DataLoader (shuffle) et au split interne de skorch ; ce
    dernier est stratifié sur y mais PAS sur le sujet, donc les 20 % de validation
    piochent dans les K donneurs sans garantie d'équilibre. C'est accepté : la
    validation sert à choisir une époque, pas à estimer une performance, et l'estimation
    qui compte se fait sur des sujets entièrement tenus à l'écart.
    """
    X = np.concatenate([data[s][0] for s in subs], axis=0)
    y = np.concatenate([data[s][1] for s in subs], axis=0)
    return X, y


def run_cell(fold: int, rule: str, draw: int, seed: int, rep: int,
             test_subs: list[int], chosen: list[int], cand: list[int],
             data: dict, meta: dict, mcfg: dict, tcfg: dict,
             out_dir: Path, device: str) -> Path:
    """Un entraînement sur le pool choisi, puis un score PAR SUJET DE TEST.

    Une ligne par sujet de test, jamais une moyenne : la moyenne se refait à l'analyse
    et se refait appariée, alors qu'une moyenne écrite ici serait irrécupérable. Le CSV
    est écrit en dernier, après le JSONL du fit -- il est le seul témoin de complétude
    que la reprise consulte.

    Deux formats de nom, et c'est délibéré. `rep < 0` est le mode v1 : on garde le nom
    d'origine pour que les 72 cellules déjà publiées restent reprises et non recalculées,
    et pour que le glob de `donor_select_analysis.py` continue de les voir. Le mode v2
    encode le réplicat et K, qui sont les deux axes ajoutés -- et son glob à lui ne peut
    pas ramasser de cellules v1 par accident, ce qui mélangerait deux protocoles dans une
    même analyse.
    """
    stem = (f"f{fold}__{rule}__d{draw}__seed{seed}" if rep < 0 else
            f"f{fold}__r{rep}__k{len(chosen)}__{rule}__seed{seed}")
    csv = out_dir / f"{stem}.csv"
    if csv.exists():
        logger.info("skip %s (already done)", stem)
        return csv

    set_seed(int(seed))
    n_classes = len(meta["classes"])
    record_path = out_dir / f"{stem}__fits.jsonl"
    record_path.unlink(missing_ok=True)
    pipeline = build_pipeline(
        mcfg, tcfg, n_chans=meta["n_chans"], n_times=meta["n_times"],
        n_outputs=n_classes, sfreq=meta["sfreq"], device=device,
        seed=int(seed), record_path=record_path)

    Xtr, ytr = pool_xy(data, chosen)
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
        rows.append({"fold": fold, "rule": rule, "draw": draw, "seed": seed,
                     "rep": rep, "test_subject": s, "n_test": int(len(yr)),
                     **transfer_scores(pipeline, Xr, yr, n_classes)})
    df = pd.DataFrame(rows)
    df["dataset"] = meta["dataset"]
    df["model"] = mcfg.get("label")
    df["k"] = len(chosen)
    df["chosen"] = ";".join(str(c) for c in chosen)
    # Le jeu de candidats est écrit lui aussi : c'est lui qui définit le REPLICAT, donc
    # l'unité de rééchantillonnage de l'analyse v2. Sans cette colonne on ne pourrait pas
    # vérifier après coup que deux règles d'un même réplicat ont bien vu les mêmes
    # candidats -- or c'est toute la validité de l'appariement.
    df["n_candidates"] = len(cand)
    df["candidates"] = ";".join(str(c) for c in cand)
    df["n_train"] = int(len(ytr))
    df["chance"] = 1.0 / n_classes
    for k in ("params_start", "params_end", "width_start", "width_end", "epochs",
              "restored_epoch", "stop_reason"):
        df[k] = rec.get(k)
    df["fit_seconds"] = round(fit_seconds, 2)
    df["device"] = device
    # La MACHINE, pas seulement "cuda". Les règles se comparent appariées, donc une
    # différence de carte entre deux cellules appariées entrerait dans la différence
    # mesurée. Plutôt que de l'interdire par le scheduler (ce qui coûte des jours de file
    # sur une partition saturée), on l'ENREGISTRE : l'affectation des cellules aux
    # cartes est indépendante de la règle, donc un effet carte est du bruit et non un
    # biais -- et l'analyse peut le VÉRIFIER au lieu de le supposer.
    df["node"] = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    df["gpu_name"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "cpu")
    df["patience"] = tcfg["patience"]
    df["selection_monitor"] = tcfg["selection_monitor"]
    for k, v in provenance().items():
        df[k] = v
    df.to_csv(csv, index=False)
    logger.info("%s: %d train trials from %d donors, fit %.1fs, mean roc_auc on %d "
                "held-out subjects %.4f", stem, len(ytr), len(chosen), fit_seconds,
                len(test_subs), float(df["roc_auc"].mean()))
    return csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="physionetmi")
    p.add_argument("--model", default="bd_shallow",
                   help="le modèle AVAL, pas la sonde. Fixe par défaut : la croissance "
                        "sert à mesurer les données, pas à les classifier.")
    p.add_argument("--ranking", required=True, type=Path)
    p.add_argument("--k", type=int, nargs="+", default=[20],
                   help="taille du pool de donneurs. Accepte une LISTE : les pools sont "
                        "emboîtés en K, donc balayer K est l'axe « on ajoute des "
                        "donneurs » et il teste la saturation prédite pour v1.")
    p.add_argument("--candidates", type=int, default=0,
                   help="M candidats tirés au hasard dans le vivier par réplicat, "
                        "PARTAGÉS par toutes les règles. C'est le correctif v2 : sans "
                        "lui une règle déterministe n'a qu'un pool par pli et le "
                        "contraste entre règles n'a que F réplicats. 0 = vivier entier "
                        "(mode v1).")
    p.add_argument("--reps", type=int, default=0,
                   help="réplicats par pli en mode v2. L'unité de rééchantillonnage de "
                        "l'analyse est (pli, réplicat), donc F x R est le n qui compte "
                        "pour le contraste entre deux règles. 0 = mode v1.")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="mode v1 uniquement. En v2 la seed est le réplicat : elle est "
                        "donc COMMUNE aux règles d'un même réplicat, et le bruit "
                        "d'initialisation s'annule en partie dans la différence "
                        "appariée au lieu de s'y ajouter.")
    p.add_argument("--random-draws", type=int, default=6,
                   help="mode v1 uniquement : tirages aléatoires par pli. Ils "
                        "remplacent les seeds pour cette règle. En v2 `random` est une "
                        "règle comme les autres et ses tirages sont les réplicats.")
    p.add_argument("--rules", nargs="+", default=list(RULES))
    p.add_argument("--all-pool", action="store_true",
                   help="le bras SANS SÉLECTION : entraîner sur le vivier ENTIER de "
                        "chaque pli. Mode exclusif -- il ignore --rules, --k, --reps et "
                        "--candidates, parce que son K n'est pas apparié à ceux des "
                        "règles et qu'aucune analyse ne doit pouvoir les additionner. "
                        "Post-hoc et descriptif : voir la section `all_pool` du "
                        "docstring.")
    p.add_argument("--all-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7],
                   help="seeds du bras --all-pool. Le pool étant déterministe étant "
                        "donné le pli, la seed est sa SEULE source de variance ; 8 "
                        "cadre avec les 8 réplicats des règles pour que les deux "
                        "moyennes reposent sur un nombre de fits comparable.")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--selection-monitor", default="valid_acc")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="imprime le plan (cellules, sujets choisis par règle) sans "
                        "entraîner. À passer AVANT toute soumission : c'est là qu'on "
                        "voit si deux règles choisissent le même pool.")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg, tcfg, pcfg = load_cfgs(a.model, a.patience, a.selection_monitor, a.max_epochs)
    rank = load_ranking(a.ranking, sorted(pd.read_csv(a.ranking)["subject"].astype(int)))
    subjects = sorted(rank["subject"].astype(int))
    fold_tests = folds_of(subjects, a.folds)

    v2 = a.reps > 0
    k_list = sorted(set(int(k) for k in a.k))

    # Le plan est construit AVANT de charger la moindre donnée : il est déterministe,
    # il ne dépend que du ranking, et c'est ce qui permet au --dry-run de coûter zéro.
    # `cand_of[(fi, rep)]` est mémorisé pour que le dry-run inspecte EXACTEMENT les jeux
    # de candidats que la campagne utilisera, et pas une redérivation qui pourrait
    # diverger en silence.
    plan, cand_of = [], {}
    for fi, test in enumerate(fold_tests):
        pool = rank[~rank["subject"].isin(test)]
        if a.all_pool:
            # Un seul « choix » possible : tout. `rep=-1` place le nom au format v1, donc
            # hors de portée du glob v2 ; le nom de règle `all_pool` n'est dans aucun
            # RULES, donc hors de portée du glob v1 lui aussi. Double garde volontaire.
            allsubs = sorted(pool["subject"].astype(int))
            for seed in a.all_seeds:
                plan.append((fi, "all_pool", 0, seed, -1, test, allsubs, allsubs))
            continue
        if len(pool) < max(k_list):
            raise SystemExit(f"fold {fi}: pool of {len(pool)} < k={max(k_list)}")
        if not v2:
            for rule in a.rules:
                if rule == "random":
                    for draw in range(a.random_draws):
                        rng = np.random.default_rng(20260902 * 1000 + fi * 100 + draw)
                        plan.append((fi, rule, draw, 0, -1, test,
                                     select(pool, rule, k_list[0], rng),
                                     sorted(pool["subject"].astype(int))))
                else:
                    chosen = select(pool, rule, k_list[0], np.random.default_rng(0))
                    for seed in a.seeds:
                        plan.append((fi, rule, 0, seed, -1, test, chosen,
                                     sorted(pool["subject"].astype(int))))
            continue
        for rep in range(a.reps):
            # Un tirage de candidats par (pli, réplicat) -- pas par règle, pas par K.
            cand = draw_candidates(pool, a.candidates,
                                   np.random.default_rng(20260902_000 + fi * 1000 + rep))
            cand_of[(fi, rep)] = cand
            if a.candidates and len(cand) <= max(k_list):
                raise SystemExit(
                    f"fold {fi} rep {rep}: {len(cand)} candidats pour K={max(k_list)}. "
                    "Une règle qui doit prendre presque tout ce qu'on lui donne ne "
                    "sélectionne plus rien : augmenter --candidates ou baisser --k.")
            for rule in a.rules:
                # L'ordre complet une fois, les K en préfixe : pools emboîtés en K.
                order = rank_candidates(
                    cand, rule, np.random.default_rng(20260902_777 + fi * 1000 + rep))
                for k in k_list:
                    plan.append((fi, rule, 0, rep, rep, test, sorted(order[:k]),
                                 sorted(cand["subject"].astype(int))))

    if a.dry_run:
        if a.all_pool:
            # Le seul chiffre à vérifier avant de soumettre est le NOMBRE D'ESSAIS
            # d'entraînement : c'est lui qui fixe le coût, et il est 4 à 8 fois celui
            # d'une cellule de règle. Le sous-estimer est l'erreur qui avait fait
            # demander 12 h en v1.
            for fi, test in enumerate(fold_tests):
                pool = sorted(rank[~rank["subject"].isin(test)]["subject"].astype(int))
                print(f"  pli {fi} : test {len(test)} sujets, all_pool = {len(pool)} "
                      f"donneurs x {len(a.all_seeds)} seeds")
            print(f"\n{len(plan)} cellules (pas de réplicat : le pool est déterministe "
                  f"étant donné le pli). BRAS POST-HOC ET DESCRIPTIF, K non apparié "
                  f"aux règles.")
            return 0
        if not v2:
            for fi, test in enumerate(fold_tests):
                pool = rank[~rank["subject"].isin(test)]
                print(f"\n--- pli {fi} : {len(test)} sujets de test, vivier {len(pool)}")
                sets = {}
                for rule in [r for r in a.rules if r != "random"]:
                    sets[rule] = set(select(pool, rule, k_list[0],
                                            np.random.default_rng(0)))
                    print(f"  {rule:14s} {sorted(sets[rule])}")
                # Le recouvrement est le nombre à regarder : deux règles qui choisissent
                # le même pool ne peuvent pas produire une différence, quelle que soit la
                # puissance. Si `params_top` et `acc_top` se recouvrent à 90 %, le
                # protocole n'a pas de prise et il faut baisser K avant de brûler du GPU.
                keys = sorted(sets)
                for i, r1 in enumerate(keys):
                    for r2 in keys[i + 1:]:
                        inter = len(sets[r1] & sets[r2])
                        print(f"    recouvrement {r1} ∩ {r2} = {inter}/{k_list[0]} "
                              f"({100 * inter / k_list[0]:.0f} %)")
        else:
            # En v2 le recouvrement se lit EN FONCTION DE K, parce que c'est la
            # prédiction testée : si l'explication de v1 est la bonne, les règles doivent
            # converger quand K/M grandit. On imprime aussi le recouvrement entre
            # réplicats, qui mesure la variance qu'on vient d'ajouter -- s'il est proche
            # de 1, les réplicats sont des copies et le correctif n'a rien corrigé.
            det = [r for r in a.rules if r != "random"]
            pairs = [(x, y) for i, x in enumerate(det) for y in det[i + 1:]]
            print(f"\ncandidats M={a.candidates or 'vivier entier'}, "
                  f"{a.reps} réplicats x {a.folds} plis = "
                  f"{a.reps * a.folds} unités de rééchantillonnage")
            for k in k_list:
                print(f"\n--- K={k}")
                for r1, r2 in pairs:
                    ov = []
                    for (fi, rep), cand in cand_of.items():
                        s1 = set(rank_candidates(cand, r1, np.random.default_rng(
                            20260902_777 + fi * 1000 + rep))[:k])
                        s2 = set(rank_candidates(cand, r2, np.random.default_rng(
                            20260902_777 + fi * 1000 + rep))[:k])
                        ov.append(len(s1 & s2) / k)
                    print(f"  recouvrement {r1:14s} ∩ {r2:14s} = "
                          f"{100 * np.mean(ov):3.0f} % (K/M = {100 * k / len(cand):.0f} %)")
            same = []
            for fi in range(a.folds):
                for r1 in range(a.reps):
                    for r2 in range(r1 + 1, a.reps):
                        s1 = set(cand_of[(fi, r1)]["subject"])
                        s2 = set(cand_of[(fi, r2)]["subject"])
                        same.append(len(s1 & s2) / max(len(s1), 1))
            if same:
                print(f"\n  recouvrement moyen ENTRE RÉPLICATS des candidats : "
                      f"{100 * np.mean(same):.0f} % — c'est la variance ajoutée ; "
                      "proche de 100 % voudrait dire que le correctif ne corrige rien.")
        by_k = {k: sum(1 for u in plan if len(u[6]) == k) for k in k_list}
        print(f"\n{len(plan)} cellules au total, {by_k} par K "
              f"({len(a.rules)} règles)")
        return 0

    device = pick_device(mcfg)
    cap_cuda_fraction()
    if a.threads:
        import torch
        torch.set_num_threads(int(a.threads))
    logger.info("device=%s model=%s (AVAL) k=%s folds=%d %s protocol: patience=%s "
                "selection=%s", device, a.model, k_list, a.folds,
                f"v2 M={a.candidates} reps={a.reps}" if v2 else "v1",
                tcfg["patience"], tcfg["selection_monitor"])

    data, meta = load_subjects(a.dataset, pcfg, a.cache, subjects)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("data ready: %d subjects, %d chans, %d times, %.0fs",
                len(data), meta["n_chans"], meta["n_times"], meta["load_seconds"])

    # Sharding en ROND et non par blocs : le plan est ordonné pli > réplicat > règle > K,
    # or le coût d'une cellule est à peu près proportionnel à K. Un découpage par blocs
    # donnerait des shards « tous les petits K » et d'autres « tous les grands », donc un
    # array dont le dernier shard tient l'horloge. Le round-robin les mélange.
    units = [u for i, u in enumerate(plan) if i % max(a.n_shards, 1) == a.shard]
    logger.info("shard %d/%d: %d cells", a.shard, a.n_shards, len(units))
    t0 = time.time()
    for i, (fi, rule, draw, seed, rep, test, chosen, cand) in enumerate(units, 1):
        run_cell(fi, rule, draw, seed, rep, test, chosen, cand, data, meta, mcfg, tcfg,
                 out_dir, device)
        done = time.time() - t0
        logger.info("progress %d/%d cells, %.1f min elapsed, ETA %.1f min",
                    i, len(units), done / 60, done / i * (len(units) - i) / 60)
    logger.info("shard done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
