"""What electrodes does each dataset actually give us?

Training one network across datasets needs a single channel space, and there are only
two ways to get one: keep the channels every dataset shares (intersection), or project
each dataset onto a canonical montage by spherical-spline interpolation. Which of the
two is viable is not a design preference, it is a fact about the data -- an
intersection of 6 electrodes would make cross-dataset pooling pointless, an
intersection of 20 would make interpolation unnecessary. So measure first.

Interpolation additionally requires *positions*, not just names: a dataset whose
channel names do not resolve against a standard montage cannot be interpolated at all
(spherical splines need coordinates on the scalp). That is reported per dataset too.

One subject per dataset is enough: the channel set is a property of the recording
setup, not of the subject. Cheap and cached.

    python benchmarks/montage_inventory.py                    # every dataset
    python benchmarks/montage_inventory.py bnci2014_001 cho2017
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mne  # noqa: E402
import yaml  # noqa: E402

from utils import logger, set_data_dir  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "config" / "dataset"
OUT = Path(__file__).resolve().parent / "results" / "montage_inventory.json"

#: Reference montage used to decide whether a channel has a known scalp position.
#: 10-05 is a superset of 10-10 and 10-20, so it resolves the widest set of names.
STANDARD = "standard_1005"


def _canon(name: str) -> str:
    """MOABB/MNE spell the same electrode several ways across datasets.

    ``standard_1005`` uses ``Fp1``/``Cz``/``FC3``, but datasets ship ``FP1``, ``fc3``,
    ``EEG-C3``, ``T3``/``T4`` (the pre-1991 names for ``T7``/``T8``), and trailing
    dots from the PhysioNet EDF headers (``Fc3.``). Canonicalise before comparing, or
    the intersection is empty for purely typographic reasons.
    """
    n = name.strip().replace(".", "")
    for prefix in ("EEG-", "EEG "):
        if n.upper().startswith(prefix.upper()):
            n = n[len(prefix):]
    upper = n.upper()
    old = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    if upper in old:
        return old[upper]
    return upper


def _standard_positions() -> set[str]:
    m = mne.channels.make_standard_montage(STANDARD)
    return {_canon(c) for c in m.ch_names}


def inspect(cfg_path: Path, known: set[str]) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    import moabb.datasets as mds

    ds = getattr(mds, cfg["moabb_class"])(**(cfg.get("kwargs") or {}))
    subj = (list(cfg["subjects"]) if cfg.get("subjects") else ds.subject_list)[0]
    data = ds.get_data(subjects=[subj])
    # subject -> session -> run -> Raw
    raw = next(iter(next(iter(data[subj].values())).values()))
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    names = [raw.ch_names[i] for i in eeg]
    canon = [_canon(n) for n in names]
    dup = [n for n, c in Counter(canon).items() if c > 1]
    return {
        "name": cfg["name"],
        "n_eeg": len(names),
        "sfreq_native": float(raw.info["sfreq"]),
        "ch_names": names,
        "canon": canon,
        "unresolved": sorted(set(canon) - known),
        "duplicate_after_canon": sorted(dup),
    }


def main(argv: list[str]) -> int:
    warnings.filterwarnings("ignore")
    set_data_dir(None)
    known = _standard_positions()
    wanted = argv or sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

    out = {}
    for name in wanted:
        try:
            out[name] = inspect(CONFIG_DIR / f"{name}.yaml", known)
            r = out[name]
            logger.info("%-20s %3d ch  %6.1f Hz  non resolus=%s",
                        name, r["n_eeg"], r["sfreq_native"], r["unresolved"] or "-")
        except Exception as e:  # a missing download must not hide the others
            out[name] = {"name": name, "error": f"{type(e).__name__}: {e}"}
            logger.error("%-20s ECHEC %s", name, e)

    ok = {k: v for k, v in out.items() if "canon" in v}
    if ok:
        sets = {k: set(v["canon"]) for k, v in ok.items()}
        inter = set.intersection(*sets.values())
        union = set.union(*sets.values())
        print(f"\n=== {len(ok)} datasets lus ===")
        print(f"intersection stricte : {len(inter)} electrodes -> {sorted(inter)}")
        print(f"union                : {len(union)} electrodes")
        # How big would the common space be if we dropped the k smallest caps?
        for k in (1, 2, 3):
            small = sorted(sets, key=lambda d: len(sets[d]))[:k]
            rest = [s for d, s in sets.items() if d not in small]
            if rest:
                print(f"  sans {small}: intersection = "
                      f"{len(set.intersection(*rest))}")
        out["_summary"] = {"intersection": sorted(inter), "union": sorted(union)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\necrit -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
