"""How much does montage interpolation cost, measured against ground truth?

Without this the interpolation ablation is uninterpretable. If pooling ``core+interp``
does not beat ``core``, there are two very different explanations -- the 33 extra
subjects carry no transferable information, or the interpolation destroyed the signal
before the network saw it -- and the pooled result alone cannot tell them apart.

The trick is to interpolate a dataset that does not need it. Cho2017 records all 64
electrodes of a 10-10 cap, so the 22 target channels are measured. Throw away everything
except the 14 electrodes Zhou2016 actually records, reconstruct the 22 from those, and
compare against the recorded 22. That is a controlled experiment with a ground truth on
real EEG, and it validates precisely the marginal arm of the pool: Zhou2016 is the worst
supported dataset that the geometric guard admits (worst gap 4.44 cm against a 4.5 cm
threshold). Shin2017A is *better* supported (3.87 cm worst, 2.64 median), so a pass here
covers it a fortiori.

Two quantities, because they answer different questions:

* **signal fidelity** -- relative error per reconstructed channel. Says what the spline
  did. Cheap, and reported per electrode so a systematically bad position shows up.
* **decoding fidelity** -- the accuracy of the same classifier on the recorded 22 versus
  the reconstructed 22, same subjects, same folds. Says what the *network* loses, which
  is the only thing the benchmark cares about. A large signal error on an electrode that
  carries no motor-imagery information costs nothing.

    python benchmarks/interp_fidelity.py --source cho2017 --emulate zhou2016 \
        --subjects 8 --model ml_csp_lda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool as poolmod  # noqa: E402
from pipelines import build_pipeline  # noqa: E402
from utils import default_results_root, logger, pick_device, set_seed  # noqa: E402

from eegrow.montage import (  # noqa: E402
    SENSORIMOTOR_22,
    canonical,
    interpolate_to_montage,
    nearest_source_gaps,
)

CONFIG = Path(__file__).resolve().parent / "config"
INVENTORY = default_results_root() / "montage_inventory.json"


def emulated_cap(emulate: str) -> list[str]:
    """The electrode names ``emulate`` actually records, from the measured inventory.

    Read off the real dataset rather than typed in: a hand-copied list that drifts from
    what MOABB serves would make the whole validation measure the wrong cap.
    """
    inv = json.loads(INVENTORY.read_text())
    if emulate not in inv or "canon" not in inv[emulate]:
        raise SystemExit(f"{emulate} absent from {INVENTORY}; run montage_inventory.py")
    return list(inv[emulate]["canon"])


def load_native(source: str, subjects: list, cache: Path | None = None):
    """Epochs of ``source`` on its own channels, same preprocessing as the pool.

    Deliberately duplicates :func:`pool.build_subject`'s paradigm construction instead of
    reusing the cache, because the cache is already projected onto the 22 -- the native
    channels are exactly what this script needs and what the cache has thrown away.
    """
    import moabb.paradigms as mpar

    if cache and cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            logger.info("epochs natives relues depuis %s", cache.name)
            return [(str(s), z[f"X_{s}"], z[f"y_{s}"], list(z[f"c_{s}"]))
                    for s in z["subs"]]
    ds = poolmod._dataset(source)
    t0 = float(ds.interval[0])
    paradigm = mpar.LeftRightImagery(
        fmin=poolmod.FMIN, fmax=poolmod.FMAX, resample=poolmod.SFREQ,
        tmin=t0, tmax=t0 + poolmod.WINDOW)
    out = []
    for s in subjects:
        epochs, y, _meta = paradigm.get_data(
            dataset=ds, subjects=[s], return_epochs=True)
        X = epochs.get_data(copy=False).astype(np.float32)[:, :, :poolmod.N_TIMES]
        yi = (np.asarray(list(map(str, y))) == "right_hand").astype(np.int64)
        out.append((str(s), X, yi, list(epochs.ch_names)))
        logger.info("%s sub-%s: %d trials, %d native channels",
                    source, s, len(yi), X.shape[1])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, subs=np.asarray([s for s, _, _, _ in out]),
                 **{f"X_{s}": X for s, X, _, _ in out},
                 **{f"y_{s}": y for s, _, y, _ in out},
                 **{f"c_{s}": np.asarray(c) for s, _, _, c in out})
    return out


def signal_fidelity(X, ch_names, cap) -> pd.DataFrame:
    """Reconstruct the 22 from ``cap`` only, and compare to the recorded 22."""
    truth, _ = interpolate_to_montage(X, ch_names, SENSORIMOTOR_22,
                                      sfreq=poolmod.SFREQ)
    keep = [i for i, c in enumerate(ch_names)
            if canonical(c) in {canonical(x) for x in cap}]
    if not keep:
        raise SystemExit("the source records none of the emulated cap's electrodes")
    sub_names = [ch_names[i] for i in keep]
    recon, diag = interpolate_to_montage(
        X[:, keep, :], sub_names, SENSORIMOTOR_22, sfreq=poolmod.SFREQ,
        max_gap_cm=None)  # the guard's verdict is what we are measuring, not enforcing

    gaps = nearest_source_gaps(
        [t for t in SENSORIMOTOR_22 if canonical(t) not in
         {canonical(c) for c in sub_names}], sub_names)
    rows = []
    for j, name in enumerate(SENSORIMOTOR_22):
        t, r = truth[:, j, :], recon[:, j, :]
        denom = float(np.linalg.norm(t))
        rows.append({
            "electrode": name,
            "reconstructed": name in diag["interpolated"],
            "gap_cm": float(gaps.get(name, 0.0)),
            "rel_error": float(np.linalg.norm(r - t) / denom) if denom else np.nan,
            "corr": float(np.corrcoef(t.ravel(), r.ravel())[0, 1]),
        })
    return pd.DataFrame(rows), truth, recon, len(keep)


def decoding_fidelity(per_subject, cap, model: str, seed: int) -> pd.DataFrame:
    """Same classifier, same leave-one-subject-out folds, recorded vs reconstructed.

    Cross-subject rather than within-subject on purpose: interpolation is only ever used
    to make subjects from *different* recordings comparable, so the regime that matters
    is the one where the classifier has to generalise across them.
    """
    from sklearn.metrics import accuracy_score

    model_cfg = yaml.safe_load((CONFIG / "model" / f"{model}.yaml").read_text())
    train_cfg = dict(yaml.safe_load((CONFIG / "config.yaml").read_text())["train"])
    device = pick_device(model_cfg)

    banks = {}
    for variant in ("recorded", "reconstructed"):
        banks[variant] = []
    for s, X, y, names in per_subject:
        fid, truth, recon, _ = signal_fidelity(X, names, cap)
        banks["recorded"].append((s, truth, y))
        banks["reconstructed"].append((s, recon, y))

    rows = []
    subs = [s for s, _, _ in banks["recorded"]]
    for variant, bank in banks.items():
        for held in subs:
            Xtr = np.concatenate([X for s, X, _ in bank if s != held])
            ytr = np.concatenate([y for s, _, y in bank if s != held])
            Xte = next(X for s, X, _ in bank if s == held)
            yte = next(y for s, _, y in bank if s == held)
            set_seed(seed)
            pipe = build_pipeline(
                model_cfg, train_cfg, n_chans=Xtr.shape[1], n_times=Xtr.shape[2],
                n_outputs=2, sfreq=poolmod.SFREQ, device=device, seed=seed)
            pipe.fit(Xtr, ytr)
            acc = float(accuracy_score(yte, pipe.predict(Xte)))
            rows.append({"variant": variant, "subject": held, "accuracy": acc,
                         "n_test": int(len(yte))})
            logger.info("%-14s held=%s acc=%.4f", variant, held, acc)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cho2017",
                    help="a dataset that records all 22 target electrodes")
    ap.add_argument("--emulate", default="zhou2016",
                    help="whose sparse cap to emulate")
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--model", default="ml_csp_lda",
                    help="config/model name; a cheap ML arm is enough to compare two "
                         "inputs, and removes training noise from the comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-decoding", action="store_true")
    a = ap.parse_args(argv)

    cap = emulated_cap(a.emulate)
    ds = poolmod._dataset(a.source)
    subjects = list(ds.subject_list)[:a.subjects]
    logger.info("fidelity: source=%s (%d subjects) emulating %s (%d electrodes)",
                a.source, len(subjects), a.emulate, len(cap))

    out = default_results_root().parent / "results_cross_dataset" / "_fidelity"
    per_subject = load_native(
        a.source, subjects, cache=out / f"native__{a.source}__{a.subjects}.npz")

    fid = pd.concat([signal_fidelity(X, names, cap)[0].assign(subject=s)
                     for s, X, _y, names in per_subject])
    agg = (fid[fid.reconstructed]
           .groupby("electrode")[["gap_cm", "rel_error", "corr"]]
           .mean().sort_values("rel_error", ascending=False))
    print(f"\n=== fidelite du signal : {len(agg)} electrodes reconstruites "
          f"depuis le cap de {a.emulate} ===")
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nerreur relative : mediane {agg.rel_error.median():.4f}  "
          f"pire {agg.rel_error.max():.4f}")
    print(f"correlation     : mediane {agg['corr'].median():.4f}  "
          f"pire {agg['corr'].min():.4f}")

    out.mkdir(parents=True, exist_ok=True)
    fid.to_csv(out / f"signal__{a.source}__as__{a.emulate}.csv", index=False)

    if not a.no_decoding:
        dec = decoding_fidelity(per_subject, cap, a.model, a.seed)
        dec.to_csv(out / f"decoding__{a.source}__as__{a.emulate}__{a.model}.csv",
                   index=False)
        piv = dec.pivot(index="subject", columns="variant", values="accuracy")
        delta = piv["reconstructed"] - piv["recorded"]
        print(f"\n=== fidelite de decodage ({a.model}, LOSO sur {len(piv)} sujets) ===")
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
        # paired over subjects: the same subject decoded from two versions of the same
        # trials, so the pairing removes the (large) between-subject variance
        from scipy import stats

        t, p = stats.ttest_rel(piv["reconstructed"], piv["recorded"])
        print(f"\ndelta moyen = {delta.mean():+.4f} "
              f"(ecart-type {delta.std(ddof=1):.4f}) apparie p = {p:.4f}")
        print("Un delta proche de 0 rend l'ablation core vs core+interp interpretable : "
              "un resultat nul voudra dire 'pas d'information transferable', pas "
              "'signal detruit'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
