"""Chauffer un dataset trop gros pour le quota, en effaçant le brut au fil de l'eau.

POURQUOI CE SCRIPT EXISTE
-------------------------
Le 05/09 la campagne « prévalence du motif » a perdu 7 datasets sur 11 sur
``OSError: [Errno 122] Disk quota exceeded``. Le quota /scratch est de 1 To par
utilisateur ; ``df`` en affichait 4,2 To libres, mais ``df`` mesure le filesystem
partagé (50 To) et ne borne rien. Le calcul d'après-coup est sans appel : hors
stieger2021 l'arbre pesait déjà 878 G, et stieger2021 complet demande ~345 G. La
campagne ne rentrait pas, et aucune quantité de parallélisme n'y changeait rien.

Le levier tient en une mesure : **le brut pèse ~10× le cache traité, et le pipeline ne
lit que le cache**. zhou2020, 68 G de brut pour 6,6 G de cache. guttmannflury_mi, 25 G
pour 1,9 G. Un dataset dont le cache est complet n'a plus besoin de son brut.

Ce script applique ça *pendant* la chauffe au lieu d'attendre la fin : sujet chauffé →
sujet vérifié → brut du sujet effacé. L'empreinte de pointe cesse d'être la taille du
dataset (345 G pour stieger2021) pour devenir celle d'un sujet (~6 G).

CE QUI GARDE CE SCRIPT DE DÉTRUIRE DES DONNÉES
----------------------------------------------
Effacer du brut est irréversible à l'échelle d'une soirée : re-télécharger stieger2021
coûte des heures. Deux gardes, toutes deux conçues pour REFUSER plutôt qu'avertir --
cf. l'incident des 276 sujets, où un chemin d'erreur a effacé une entrée partagée.

1. ON N'EFFACE QUE CE QUE LA PORTE CONFIRME. Après chaque sujet on relit l'état du
   cache avec ``warm_cache._state``, c'est-à-dire *la fonction que le gate utilise*, et
   on exige lockfile ET données présents sous CHAQUE digest de la campagne. Pas un
   marqueur maison, pas un « get_data n'a pas levé » : la même vérification que celle
   qui décidera plus tard si les fits ont le droit de partir. Si elle ne confirme pas,
   le brut reste.

2. ON N'EFFACE QU'UN FICHIER QU'AUCUN AUTRE SUJET NE RÉCLAME, AVEC UN SUJET DE RETARD.
   ``data_path(s)`` renvoie les fichiers d'un sujet, mais pour un dataset livré en
   archive unique il renvoie LE MÊME fichier pour les 51 sujets -- l'effacer après le
   sujet 1 casse les 50 autres.

   La première version de cette garde lisait le nom de fichier : « effacer si le
   basename contient l'identifiant du sujet ». Testée sur les 287 ``.mat`` réels de
   stieger2021 avant tout effacement, elle produisait **570 revendications pour 287
   fichiers** : le sujet 1 réclamait ``S2_Session_1.mat``, parce que le numéro de
   *session* matche aussi. Lancée telle quelle, elle aurait effacé tous les
   ``*_Session_1.mat`` des 62 sujets. Une heuristique de nom n'a aucun moyen de savoir
   quel champ elle lit.

   Donc plus d'heuristique : on tient ``claimed``, la carte fichier -> sujets qui l'ont
   renvoyé, et on n'efface qu'un fichier réclamé par exactement un sujet. Comme un sujet
   *futur* pourrait encore le réclamer, la suppression prend **un sujet de retard** : le
   brut du sujet i n'est effacé qu'une fois le sujet i+1 chauffé et sa liste de fichiers
   connue. L'empreinte de pointe est donc de deux sujets au lieu d'un, et chaque
   effacement est vérifié contre le sujet suivant en plus de tous les précédents.

   Un dataset en archive unique traverse ce script sans qu'un octet soit effacé -- c'est
   le comportement correct, et ``--purge-raw-at-end`` reste là pour ce cas, gardé par un
   ``check_cache`` complet sur tout le dataset.

USAGE
-----
    python benchmarks/warm_stream.py --dataset stieger2021 --cache /scratch/.../moabb_cache
    python benchmarks/warm_stream.py --dataset yang2025 --cache ... --purge-raw-at-end
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_cache import build_campaign  # noqa: E402
from warm_cache import _state  # noqa: E402


def quota_free_gb() -> int | None:
    """Go restants sous le quota /scratch, ou ``None`` si illisible.

    ``quota`` sans ``-s`` donne des blocs de 1 Ko, donc aucune unité à reparser. Le nom
    du filesystem occupe sa propre ligne, d'où le ``getline``.
    """
    try:
        out = subprocess.run(["quota"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if "data/scratch" in ln:
            fields = (lines[i + 1] if i + 1 < len(lines) else "").split()
            if len(fields) >= 2:
                try:
                    return (int(fields[1]) - int(fields[0])) // 1048576
                except ValueError:
                    return None
    return None


def subject_complete(base: Path, keys: dict, subject) -> bool:
    """Même critère que le gate : chaque session découverte, sous chaque digest."""
    locks, data = _state(base, keys)
    sessions = {e for s, e, _ in locks | data if s == str(subject)}
    if not sessions:
        return False
    return all((str(subject), e, h) in locks and (str(subject), e, h) in data
               for e in sessions for h in keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", default="/scratch/amounir/moabb_cache")
    ap.add_argument("--floor-gb", type=int, default=25,
                    help="refuser de démarrer sous ce reste de quota")
    ap.add_argument("--purge-raw-at-end", action="store_true",
                    help="effacer tout le répertoire brut si check_cache passe à la fin")
    ap.add_argument("--dry-run", action="store_true",
                    help="tout chauffer, ne rien effacer, dire ce qui aurait été effacé")
    args = ap.parse_args()

    free = quota_free_gb()
    if free is None:
        print("REFUS: quota illisible. Sans plancher fiable on n'écrit pas.", flush=True)
        return 3
    if free < args.floor_gb:
        print(f"REFUS: {free}G sous le quota < plancher {args.floor_gb}G.", flush=True)
        return 3

    from moabb.datasets.bids_interface import get_bids_root
    from omegaconf import OmegaConf
    from utils import cache_config

    ds, par, keys = build_campaign(args.dataset)
    base = Path(args.cache) / get_bids_root(ds.code, None).name
    ccfg = cache_config(OmegaConf.create({"enabled": True, "path": args.cache}))

    print(f"=== {args.dataset}  {len(ds.subject_list)} sujets  quota_libre={free}G"
          f"  clés={' '.join(f'{h[:8]}:{k}' for h, k in keys.items())}", flush=True)

    claimed: dict[Path, set] = {}   # fichier -> sujets qui l'ont renvoyé
    pending: list[tuple] = []       # (sujet, fichiers) en attente du sujet suivant
    stats = {"freed_gb": 0.0, "shared": 0, "deleted": 0}
    n_ok = n_fail = 0
    t_start = time.time()

    def flush(keep_last: bool) -> None:
        """Effacer le brut des sujets confirmés, en gardant éventuellement le dernier.

        Un fichier n'est effacé que si ``claimed`` ne lui connaît qu'un seul sujet. Le
        dernier sujet reste en attente tant qu'un sujet suivant peut encore le réclamer.
        """
        cut = len(pending) - 1 if keep_last else len(pending)
        for subj, files in pending[:max(cut, 0)]:
            gb = 0.0
            for p in files:
                if len(claimed.get(p, ())) > 1:
                    stats["shared"] += 1
                    continue
                if not p.is_file():
                    continue
                size = p.stat().st_size / 2**30
                if args.dry_run:
                    gb += size
                    stats["deleted"] += 1
                    continue
                try:
                    p.unlink()
                    gb += size
                    stats["deleted"] += 1
                except OSError as exc:
                    print(f"    non effacé {p.name}: {exc}", flush=True)
            stats["freed_gb"] += gb
            if gb:
                print(f"    {'aurait libéré' if args.dry_run else 'libéré'} {gb:.1f}G "
                      f"du sub-{subj} (cumul {stats['freed_gb']:.0f}G, "
                      f"quota_libre {quota_free_gb()}G)", flush=True)
        del pending[:max(cut, 0)]

    for i, s in enumerate(ds.subject_list, 1):
        t0 = time.time()
        if subject_complete(base, keys, s):
            msg = "déjà en cache"
        else:
            try:
                par.get_data(dataset=ds, subjects=[s], cache_config=ccfg)
                msg = "chauffé"
            except Exception as exc:
                n_fail += 1
                print(f"[{i}/{len(ds.subject_list)}] sub-{s} {time.time()-t0:7.1f}s "
                      f"ÉCHEC {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                if isinstance(exc, OSError) and exc.errno in (28, 122):
                    print("ABANDON : plus de place. Le brut en attente reste intact.",
                          flush=True)
                    return 3
                continue

        # GARDE 1 : la porte, pas l'absence d'exception.
        if not subject_complete(base, keys, s):
            n_fail += 1
            print(f"[{i}/{len(ds.subject_list)}] sub-{s} {time.time()-t0:7.1f}s "
                  f"{msg} mais CACHE INCOMPLET -- brut conservé", flush=True)
            continue
        n_ok += 1

        try:
            raw = [p for p in (Path(x) for x in _flatten(ds.data_path(s))) if p.is_file()]
        except Exception as exc:
            print(f"[{i}/{len(ds.subject_list)}] sub-{s} {time.time()-t0:7.1f}s "
                  f"{msg}, chemins bruts illisibles ({type(exc).__name__}) -- rien effacé",
                  flush=True)
            continue

        # GARDE 2 : enregistrer les revendications AVANT d'effacer quoi que ce soit,
        # puis vider la file en gardant le dernier sujet -- c'est le sujet de retard.
        for p in raw:
            claimed.setdefault(p, set()).add(s)
        pending.append((s, raw))
        print(f"[{i}/{len(ds.subject_list)}] sub-{s} {time.time()-t0:7.1f}s {msg}, "
              f"{len(raw)} fichiers bruts", flush=True)
        flush(keep_last=True)

    flush(keep_last=False)   # plus de sujet à venir : la file peut être vidée entière

    print(f"=== {args.dataset} terminé en {(time.time()-t_start)/60:.1f} min : "
          f"{n_ok} ok, {n_fail} en échec, {stats['deleted']} fichiers effacés pour "
          f"{stats['freed_gb']:.0f}G, {stats['shared']} conservés car réclamés par "
          f"plusieurs sujets", flush=True)

    if args.purge_raw_at_end and not args.dry_run:
        if n_fail:
            print("PURGE FINALE REFUSÉE : des sujets ont échoué, le brut reste.", flush=True)
        else:
            rc = subprocess.run(
                [sys.executable, str(HERE / "check_cache.py"),
                 "--cache", args.cache, "--datasets", args.dataset]).returncode
            if rc != 0:
                print("PURGE FINALE REFUSÉE : check_cache a refusé, le brut reste.",
                      flush=True)
            else:
                root = Path(ds.data_path(ds.subject_list[0])[0]).parents
                # Remonter jusqu'au répertoire MNE-<code>-data, jamais au-dessus.
                target = next((d for d in root if d.name.startswith("MNE-")), None)
                if target is None:
                    print("PURGE FINALE REFUSÉE : répertoire MNE-* introuvable.", flush=True)
                else:
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"PURGE FINALE : {target} effacé (check_cache vert).", flush=True)

    return 1 if n_fail else 0


def _flatten(x):
    """``data_path`` renvoie une liste, parfois de listes selon le dataset."""
    if isinstance(x, (str, Path)):
        return [x]
    out = []
    for e in x:
        out.extend(_flatten(e))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
