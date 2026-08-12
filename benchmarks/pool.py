"""Assemble several MOABB datasets into one trainable pool.

The three axes that have to be unified before a single network can see all of them, and
the reason each one is not a free choice:

**Electrodes.** The intersection over the datasets with known positions is one channel
(Cz), so the common space is built by spherical-spline interpolation onto
``SENSORIMOTOR_22`` -- see :mod:`eegrow.montage` for what that can and cannot do.

**Labels.** ``left_hand`` vs ``right_hand`` is the only label pair shared widely. Three
otherwise usable datasets (AlexMI, BNCI2014_002, BNCI2015_001) offer ``feet``/
``right_hand`` and are therefore out of the pool entirely, not merely down-weighted.
Note this makes the task 2-class, whereas the single-dataset benchmark ran BNCI2014_001
as 4-class MotorImagery: a cross-dataset number is *not* comparable to that one, and any
comparison needs a baseline re-run under this paradigm (see ``arm="within"``).

**Time.** This is the subtle one. The event marker does not sit at the same place in
every protocol: MOABB's ``dataset.interval`` starts at 0 s for Cho2017 but at 2 s for
BNCI2014_001 and 3 s for Weibo2014, because those protocols mark the visual cue and the
imagery begins seconds later. Passing one common ``tmin`` would epoch the cue-evoked
response on some datasets and the mu rhythm on others -- two different physiological
signals in the same tensor. Every window is therefore anchored on the dataset's *own*
``interval[0]``, with a common width of :data:`WINDOW` seconds. 3.0 s is not a
preference either: it is the floor, set by Cho2017 and PhysionetMI whose intervals are
exactly 3 s long.

Build once, train many times::

    python benchmarks/pool.py build --dataset cho2017
    python benchmarks/pool.py build --tier core        # every core dataset
    python benchmarks/pool.py inventory
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import logger, set_data_dir  # noqa: E402

from eegrow.montage import (  # noqa: E402
    MAX_GAP_CM,
    SENSORIMOTOR_22,
    interpolate_to_montage,
)

#: Common epoch width, in seconds. The floor over the pool (Cho2017, PhysionetMI).
WINDOW = 3.0
SFREQ = 250.0
N_TIMES = int(round(WINDOW * SFREQ))
FMIN, FMAX = 8.0, 32.0

#: Global RMS every arm is rescaled to, just before it is handed to a model.
#:
#: ``build_subject`` calls ``paradigm.get_data(..., return_epochs=True)``, and MOABB applies
#: its ``unit_factor = 1e6`` only on the *array* branch of ``paradigms/base.py`` (the
#: ``FunctionTransformer(methodcaller("__mul__", dataset.unit_factor))`` around line 684).
#: Asking for Epochs therefore skips it, and the cache holds **volts** -- std ~5e-6, and
#: ~5e-7 on the low-gain datasets -- whereas everything that goes through MOABB's own array
#: path gets microvolts. This is a property of *this* loader, not of MOABB.
#:
#: Measured on one BNCI2014_001 fold with ShallowFBCSPNet, 40 epochs, everything else fixed:
#:
#:     volts        acc 0.504  auc 0.539  -- predicts one class for 99.7% of trials
#:     x1e6         acc 0.590  auc 0.692
#:     unit RMS     acc 0.649  auc 0.733
#:
#: The initial training loss is 9.71 in volts against 1.04 at unit RMS: the net starts in
#: a regime where the gradient is numerically dead, and early stopping on valid accuracy
#: then locks in the constant prediction. BatchNorm does not save it -- it makes the
#: *forward* pass scale-free, but AdamW's weight decay still pulls against the ~1e6 first
#: layer weights that volt-scale inputs require.
#:
#: The exact constant is irrelevant (1e6 and unit RMS both work); what matters is that it
#: is O(1) and that it is the *same* for all three alignment arms, since a different
#: amplitude per arm would hand the optimiser a different effective learning rate and stop
#: the arms from being controls for each other.
TARGET_RMS = 1.0

#: (moabb class, kwargs, tier). Tiers are the interpolation ablation, not bookkeeping:
#:
#: ``core``   -- all 22 target electrodes recorded natively (Lee2019_MI misses one).
#:               Pooling these needs no interpolation, so they are the control.
#: ``interp`` -- reachable only through interpolation, yet still FULL RANK. Shin2017A's
#:               30 electrodes cover the scalp in the 10-05 intermediate nomenclature
#:               (AFF5h, FCC3h) and simply do not *name* the target ones; 30 sources for
#:               22 outputs keeps the projection full rank. Its 29 subjects are what the
#:               interpolation buys, and core vs core+interp measures whether it buys
#:               information or noise -- with rank held constant, so the contrast is
#:               about interpolation and nothing else.
#: ``lowrank`` -- Zhou2016. Well supported geometrically (worst gap 4.44 cm) but only 14
#:               electrodes, so the 22 columns span 14 dimensions and the covariance is
#:               singular. That breaks CSP outright and makes Euclidean alignment whiten
#:               inside a subspace -- i.e. it would confound the interpolation axis with
#:               the alignment axis, which are the two things this experiment separates.
#:               Kept out of the default pool for 4 subjects' worth of data; available as
#:               its own arm for the deep models, which invert nothing.
#: ``extrap`` -- BNCI2014_004 records C3/Cz/C4. 19 of 22 channels would come from 3
#:               spatial degrees of freedom, the worst of them 10.1 cm from any recorded
#:               electrode. Refused by the geometric guard; usable only with an explicit
#:               ``max_gap_cm=None``, as a negative control.
POOL = {
    "bnci2014_001":      ("BNCI2014_001", {}, "core"),
    "cho2017":           ("Cho2017", {}, "core"),
    "lee2019_mi":        ("Lee2019_MI", {}, "core"),
    "physionetmi":       ("PhysionetMI", {}, "core"),
    "schirrmeister2017": ("Schirrmeister2017", {}, "core"),
    "weibo2014":         ("Weibo2014", {}, "core"),
    "shin2017a":         ("Shin2017A", {"accept": True}, "interp"),
    "zhou2016":          ("Zhou2016", {}, "lowrank"),
    "bnci2014_004":      ("BNCI2014_004", {}, "extrap"),
}


def pool_root() -> Path:
    return Path(__file__).resolve().parent / "pool"


def tier(names_or_tier) -> list[str]:
    """Accept a tier name, ``"core+interp"``, or an explicit comma list."""
    s = str(names_or_tier)
    if all(t in {"core", "interp", "lowrank", "extrap"} for t in s.split("+")):
        want = set(s.split("+"))
        return [k for k, v in POOL.items() if v[2] in want]
    return [n.strip() for n in s.split(",") if n.strip()]


def _dataset(name: str):
    import moabb.datasets as mds

    cls, kwargs, _ = POOL[name]
    ds = getattr(mds, cls)(**kwargs)
    # MOABB 1.5.0 drops sessions on the Lee2019 family (its _selected_sessions are
    # 1-based while the session names are 0-based, so the intersection loses one).
    # Same neutralisation as run_moabb_hydra.
    if getattr(ds, "_selected_sessions", None) is not None:
        logger.warning("%s: dropping _selected_sessions=%s (MOABB off-by-one)",
                       name, ds._selected_sessions)
        ds._selected_sessions = None
    return ds


def build_subject(name: str, subject, *, max_gap_cm: float | None = MAX_GAP_CM,
                  overwrite: bool = False) -> dict:
    """Epoch, band-pass, resample and montage-project one subject; cache to ``.npz``."""
    import moabb.paradigms as mpar

    out = pool_root() / name / f"sub-{subject}.npz"
    if out.exists() and not overwrite:
        with np.load(out, allow_pickle=True) as z:
            return {"path": str(out), "n_trials": int(z["X"].shape[0]),
                    "cached": True, "diag": json.loads(str(z["diag"]))}

    ds = _dataset(name)
    t0 = float(ds.interval[0])
    paradigm = mpar.LeftRightImagery(
        fmin=FMIN, fmax=FMAX, resample=SFREQ, tmin=t0, tmax=t0 + WINDOW)
    epochs, y, meta = paradigm.get_data(
        dataset=ds, subjects=[subject], return_epochs=True)

    X = epochs.get_data(copy=False).astype(np.float32)
    # MNE's epoch window is inclusive of both endpoints, so 3.0 s at 250 Hz yields 751
    # samples, not 750. Crop rather than resample: a one-sample difference between
    # datasets would still make the tensors unstackable, and cropping the tail of a
    # 3 s imagery window loses 4 ms.
    if X.shape[2] < N_TIMES:
        raise ValueError(f"{name} sub-{subject}: {X.shape[2]} samples < {N_TIMES}")
    X = X[:, :, :N_TIMES]

    Xt, diag = interpolate_to_montage(
        X, epochs.ch_names, SENSORIMOTOR_22, sfreq=SFREQ,
        max_gap_cm=max_gap_cm)
    Xt = np.ascontiguousarray(Xt, dtype=np.float32)

    classes = np.asarray(sorted(set(map(str, y))))
    if list(classes) != ["left_hand", "right_hand"]:
        raise ValueError(f"{name}: unexpected classes {classes}")
    yi = (np.asarray(list(map(str, y))) == "right_hand").astype(np.int64)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=Xt, y=yi,
        session=meta["session"].astype(str).to_numpy(),
        subject=np.asarray([str(subject)] * len(yi)),
        dataset=np.asarray([name] * len(yi)),
        diag=json.dumps(diag))
    logger.info("%s sub-%s: %d trials, %d interpolated ch (pire ecart %.2f cm), "
                "support=%d -> %s", name, subject, len(yi),
                len(diag["interpolated"]), diag["max_gap_cm"], diag["n_support"],
                out.name)
    return {"path": str(out), "n_trials": int(len(yi)), "cached": False, "diag": diag}


def _global_rms(X, chunk: int = 512) -> float:
    """Root-mean-square of the whole array, accumulated in float64 chunk by chunk.

    The pooled arm runs to a few GB in float32; ``X.astype(np.float64)`` would ask for a
    second copy at twice the size. Chunking keeps the accumulator exact without ever
    materialising it.
    """
    sq = 0.0
    for i in range(0, len(X), chunk):
        sq += float((X[i:i + chunk].astype(np.float64) ** 2).sum())
    return float(np.sqrt(sq / X.size)) if X.size else 0.0


def load(names, *, align: str = "none", exclude=(), subjects=None):
    """Concatenate cached subjects into one training set.

    Parameters
    ----------
    names : sequence of str
        Datasets to include.
    align : {"none", "scale", "euclidean"}
        Applied here rather than baked into the cache, because it is cheap and because
        the arms must come from bit-identical epochs -- one cache, three arms, no chance
        of them drifting apart on a preprocessing detail.

        ``euclidean`` whitens each subject by its own mean trial covariance; the
        reference is per (dataset, subject), because a subject is a subject regardless
        of which dataset recorded them and a per-dataset reference would leave exactly
        the between-subject mixing EA exists to remove.

        ``scale`` divides each subject by its own global standard deviation and does
        nothing else. It exists because pooling datasets is confounded by amplitude --
        different amplifiers, references and impedances put them on different scales, so
        one dataset can dominate the loss for non-neural reasons. Whitening fixes that
        as a side effect, so without this control a cross-dataset EA gain cannot be
        attributed: it could be whitening, or it could be rescaling. Only
        ``euclidean`` minus ``scale`` is evidence about whitening.
    exclude : iterable of (dataset, subject)
        Held-out test subjects. Passed explicitly so a caller cannot forget them.
    subjects : dict, optional
        ``{dataset: [subjects]}`` to restrict to; default is every cached subject.

    Returns
    -------
    X, y, groups : ndarray
        ``groups`` holds ``"<dataset>|<subject>"`` per trial, for grouped CV.
    """
    from eegrow.alignment import euclidean_align

    excl = {(str(d), str(s)) for d, s in exclude}
    Xs, ys, gs = [], [], []
    for name in names:
        d = pool_root() / name
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not built -- run `pool.py build`")
        for f in sorted(d.glob("sub-*.npz")):
            subj = f.stem.split("sub-")[1]
            if (name, subj) in excl:
                continue
            if subjects and name in subjects and subj not in {
                    str(s) for s in subjects[name]}:
                continue
            with np.load(f, allow_pickle=True) as z:
                x = z["X"]
                Xs.append(x)
                ys.append(z["y"])
                gs.append(np.asarray([f"{name}|{subj}"] * len(z["y"])))
    if not Xs:
        raise ValueError(f"nothing loaded for {list(names)} (exclude={sorted(excl)})")
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    Xs.clear()  # the concatenation copied; holding these doubles peak memory

    if align == "euclidean":
        X = euclidean_align(X, g, preserve_scale=True).astype(np.float32)
    elif align == "scale":
        for grp in np.unique(g):
            m = g == grp
            s = float(X[m].std())
            if s > 0:
                X[m] = X[m] / np.float32(s)
    elif align != "none":
        raise ValueError(f"unknown align={align!r}")

    # Every arm ends on the same O(1) global amplitude -- see TARGET_RMS. Done last and
    # with a single global factor, so it cannot disturb what distinguishes the arms:
    # under ``none`` the amplitude *ratios* between datasets survive untouched, which is
    # precisely the confound the ``scale`` arm exists to expose.
    rms = _global_rms(X)
    if rms > 0:
        X *= np.float32(TARGET_RMS / rms)
    return X, y, g


def inventory() -> dict:
    out = {}
    for name, (_cls, _kw, t) in POOL.items():
        d = pool_root() / name
        files = sorted(d.glob("sub-*.npz")) if d.is_dir() else []
        n_tr = 0
        diag = None
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                n_tr += int(len(z["y"]))
                diag = json.loads(str(z["diag"]))
        out[name] = {"tier": t, "n_subjects": len(files), "n_trials": n_tr,
                     "n_interpolated": len(diag["interpolated"]) if diag else None,
                     "n_support": diag["n_support"] if diag else None,
                     "max_gap_cm": diag.get("max_gap_cm") if diag else None,
                     # older cache files predate the rank diagnostic; it is a property
                     # of the projection, so it can be recovered from the support count
                     "rank": (diag.get("rank")
                              or min(diag["n_support"], len(SENSORIMOTOR_22)))
                     if diag else None}
    return out


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    # without this the module's logger.info calls are filtered at WARNING and a build
    # that works looks exactly like a build that did nothing
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset", help="one dataset name")
    b.add_argument("--tier", default="core+interp",
                   help="core / interp / extrap / core+interp / comma list")
    b.add_argument("--index", type=int,
                   help="SLURM array index into the selected dataset list")
    b.add_argument("--max-gap-cm", type=float, default=MAX_GAP_CM,
                   help="0 or negative => no guard (negative control)")
    b.add_argument("--overwrite", action="store_true")
    sub.add_parser("inventory")
    a = ap.parse_args(argv)

    set_data_dir(None)
    if a.cmd == "inventory":
        inv = inventory()
        print(f"{'dataset':20s} {'tier':7s} {'suj':>4s} {'essais':>8s} "
              f"{'interp':>7s} {'support':>8s} {'ecart':>7s} {'rang':>5s}")
        tot_s = tot_t = 0
        for k, v in inv.items():
            gap = "-" if v["max_gap_cm"] is None else f"{v['max_gap_cm']:.2f}"
            print(f"{k:20s} {v['tier']:7s} {v['n_subjects']:4d} {v['n_trials']:8d} "
                  f"{str(v['n_interpolated']):>7s} {str(v['n_support']):>8s} "
                  f"{gap:>7s} {str(v['rank']):>5s}")
            tot_s += v["n_subjects"]
            tot_t += v["n_trials"]
        print(f"{'TOTAL':20s} {'':7s} {tot_s:4d} {tot_t:8d}")
        return 0

    names = [a.dataset] if a.dataset else tier(a.tier)
    if a.index is not None:
        names = [sorted(names)[a.index]]
    for name in names:
        ds = _dataset(name)
        for s in ds.subject_list:
            try:
                build_subject(name, s,
                              max_gap_cm=(a.max_gap_cm if a.max_gap_cm > 0
                                          else None),
                              overwrite=a.overwrite)
            except Exception as e:
                # one unreadable subject must not cost the other 108
                logger.error("%s sub-%s ECHEC %s: %s", name, s, type(e).__name__, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
