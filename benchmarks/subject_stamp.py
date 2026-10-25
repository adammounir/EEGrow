"""Name the subject each MOABB fit was fitted for, inside the fit record itself.

WHY THIS EXISTS
---------------
``FitRecorder`` writes one JSONL record per fit -- learning curve, growth trajectory,
stop reason, restored epoch -- but it cannot name the subject, and several of the
diagnostics discussed with Stella are questions *about* subjects: on which subject did
the model stop early, where did growth stall, is the tail of the accuracy distribution
one hard subject or many. Today those are unanswerable from the records.

The reason is structural, not an oversight. The benchmark builds **one** pipeline and
hands it to ``evaluation.process()``; MOABB owns the subject loop, loads the data
itself, and clones the pipeline per fold. From inside a skorch callback the held-out
subject is simply not in scope, so no amount of care in ``recording.py`` recovers it.

It matters that this is settled *before* the final grid rather than after: a column
missing from 396 cells cannot be added by re-reading them. It costs the 817 GPU-hours
again.

HOW
---
In MOABB 1.5.0 ``process()`` takes the *flattened parallel* path
(``_process_parallel`` -> ``_evaluate_parallel_dataset`` -> ``_build_task_list`` ->
``_evaluate_fold``). ``_build_task_list`` produces one task dict per fold, and that
dict already carries everything wanted -- ``subject`` (read off
``test_meta["subject"].iloc[0]``, i.e. the held-out subject), ``session``, ``cv_ind``
-- next to the ``pipeline`` that fold will fit.

So the mixin overrides ``_build_task_list`` and, for each task, swaps in a pipeline
whose ``FitRecorder`` carries those three fields in its ``meta``. ``meta`` is a
constructor parameter stored verbatim (``recording.py`` keeps it that way precisely so
``clone`` round-trips it), so it survives the ``clone(pipeline)`` that
``_evaluate_fold`` performs, and ``on_train_end`` writes the subject into the record
as an ordinary field.

That is why this is not a sidecar file joined by position. A positional join would be
sound today -- one writing process per cell, ``n_jobs=1`` -- but it is a convention
that cannot be checked from the data, and a subject column quietly off by one would
misattribute every early stop in the paper with nothing downstream able to notice.
Here the subject is written by the same callback, in the same record, on the same fit.
There is no join to get wrong.

The rejected alternative was ``BaseEvaluation._maybe_save_model_cv``, which does
receive the subject and looks like the natural hook. It is only ever called from
``_evaluate``, on the deprecated ``_process_legacy`` path, and is never reached by the
code that actually runs -- verified by instrumenting it and counting zero calls across
a full ``WithinSessionEvaluation``. An override there would have produced 396 empty
stamp files and no error.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from sklearn.base import clone

logger = logging.getLogger(__name__)

#: Fields lifted from the MOABB task and written into the fit record.
STAMP_KEYS = ("subject", "session", "cv_ind")


def _is_recorder(cb) -> bool:
    """Duck-type a :class:`FitRecorder`.

    By its two constructor parameters rather than by class, so this module does not
    import ``eegrow`` merely to run an ``isinstance`` check -- and so it keeps working
    if the recorder is ever subclassed.
    """
    return hasattr(cb, "meta") and hasattr(cb, "path")


def with_stamp(pipeline, stamp: dict):
    """``pipeline``, cloned, with ``stamp`` merged into its recorder's ``meta``.

    A fresh recorder is *constructed* rather than mutated in place. Whether
    ``sklearn.clone`` deep-clones a list of ``(name, callback)`` tuples is an
    implementation detail; if it ever aliased them, mutating would leak the last
    fold's subject into every task, and the resulting records would be wrong in a way
    that looks entirely plausible. Constructing removes the question.

    Returns the pipeline unchanged when no recorder is present (the ML arms, and the
    fixed arms of a run launched without ``record_path``).
    """
    p = clone(pipeline)
    steps = [s for _, s in p.steps] if hasattr(p, "steps") else [p]
    for step in steps:
        cbs = getattr(step, "callbacks", None)
        if not cbs:
            continue
        rebuilt, hit = [], False
        for item in cbs:
            named = isinstance(item, tuple) and len(item) == 2
            name, cb = item if named else (None, item)
            if _is_recorder(cb):
                # Every constructor parameter carried over, not just path/meta: a
                # recorder flag dropped here (save_selected) would be silently off
                # in every stamped fold while the config says on.
                params = dict(cb.get_params(deep=False))
                params["meta"] = {**(cb.meta or {}), **stamp}
                cb = type(cb)(**params)
                hit = True
            rebuilt.append((name, cb) if named else cb)
        if hit:
            step.set_params(callbacks=rebuilt)
            return p
    return p


def _find_recorder(pipeline):
    """The (unfitted) FitRecorder in ``pipeline``'s callbacks, or None."""
    steps = [s for _, s in pipeline.steps] if hasattr(pipeline, "steps") else [pipeline]
    for step in steps:
        for item in getattr(step, "callbacks", None) or []:
            cb = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if _is_recorder(cb):
                return cb
    return None


class LogitScorer:
    """MOABB's scorer, unchanged, plus the raw test logits of the fold on disk.

    WHY HERE: the test set never enters the skorch fit, so ``FitRecorder`` cannot see
    it; the only code that holds both the fitted net and the test trials is MOABB's
    ``_evaluate_fold``, through ``scorer(cvclf, X[group_idx], group_y)``. Wrapping the
    scorer is therefore the one hook that needs no copy of the data and no re-run of
    the forward pass outside the fold that produced the score.

    The returned score is ``check_scoring(est, scoring)(est, X, y)`` -- the exact call
    ``_create_scorer`` makes for a string -- so the CSV is unchanged by construction.
    The logits are then ``net.forward`` (no softmax) on what the net receives, i.e. after
    the pipeline's own transforms. Calls are matched to MOABB's score groups by order
    (one group, or ``np.unique`` of the test sessions under ``score_per_session``) and
    refused on a length mismatch rather than mislabelled.

    Writes ``<stem>__logits/<fit_uid>[__g<i>].npz``: ``logits``, ``y`` (as scored),
    ``test_idx`` (rows of MOABB's metadata), ``subject``/``session``/``run`` per trial,
    ``test_fingerprint`` (SHA-1 of the net's input and ``y``, the train_fingerprint
    convention) and the ``score`` itself, so a reader checks argmax == CSV.
    """

    def __init__(self, scoring, groups, folder):
        if not isinstance(scoring, str):
            raise ValueError(f"LogitScorer wraps a single named scorer, got {scoring!r}")
        self.scoring = scoring
        # [(test_idx, {"subject": .., "session": .., "run": ..}), ...] in MOABB's group
        # order. Only the test rows are kept: sklearn's get_scorer deep-copies a
        # callable scorer once per fold, and the full metadata would go with it.
        self.groups = groups
        self.folder = Path(folder)
        self.calls = 0

    def __call__(self, est, X, y):
        from sklearn.metrics import check_scoring
        score = check_scoring(est, scoring=self.scoring)(est, X, y)
        i = self.calls
        self.calls += 1
        idx, cols = self.groups[i]
        if len(idx) != len(X):
            raise RuntimeError(f"logit scorer call {i}: {len(X)} trials, group has {len(idx)}")
        net = est.steps[-1][1] if hasattr(est, "steps") else est
        Xt = est[:-1].transform(X) if hasattr(est, "steps") and len(est.steps) > 1 else X
        logits = net.forward(Xt, training=False, device="cpu").numpy().astype("float32")
        rec = next((cb for _, cb in getattr(net, "callbacks_", []) if _is_recorder(cb)), None)
        uid = getattr(rec, "last_fit_uid_", None)
        if uid is None:
            raise RuntimeError("logit scorer: the fitted net carries no FitRecorder fit_uid")
        h = hashlib.sha1(np.ascontiguousarray(np.asarray(Xt)).tobytes())
        h.update(np.ascontiguousarray(np.asarray(y)).tobytes())
        self.folder.mkdir(parents=True, exist_ok=True)
        name = uid if len(self.groups) == 1 else f"{uid}__g{i}"
        np.savez_compressed(
            self.folder / f"{name}.npz", logits=logits, y=np.asarray(y), test_idx=np.asarray(idx),
            **cols,
            test_fingerprint=np.array(h.hexdigest()), score=np.array(float(score)))
        return score


class _SubjectStampMixin:
    """Carry the held-out subject into every fit record of a MOABB evaluation.

    Placed **first** in the MRO of the classes built by :func:`stamped`, so the
    override wins; it delegates to ``super()`` for the task list itself and only
    rewrites the ``pipeline`` entry.
    """

    def process(self, *args, **kwargs):
        # Fail here rather than 800 GPU-hours later with an unusable column. Both
        # conditions hold today for all three evaluations the benchmark uses; the
        # point is that if a MOABB upgrade breaks one, the cell stops instead of
        # quietly writing records that cannot name their subject.
        if self._create_splitter() is None:
            raise RuntimeError(
                f"{type(self).__mro__[2].__name__} has no splitter, so MOABB falls "
                "back to the legacy evaluation loop, which never calls "
                "_build_task_list. Subject stamps would be silently absent."
            )
        if int(getattr(self, "n_jobs", 1)) != 1:
            # The fit records are one JSONL per cell appended by a single process;
            # joblib's loky backend would give each fold its own process and let the
            # appends interleave. See run_moabb_hydra.py on the one-writer property.
            raise RuntimeError(
                f"n_jobs={self.n_jobs}: fit records assume one writing process per "
                "cell. Parallelise at the sweep level, not inside the evaluation."
            )
        return super().process(*args, **kwargs)

    def _build_task_list(self, dataset, X, y, metadata, splitter, work_plan,
                         pipelines, param_grid):
        tasks = super()._build_task_list(dataset, X, y, metadata, splitter,
                                         work_plan, pipelines, param_grid)
        only = getattr(self, "only_subject_", None)
        if only is not None:
            # One fold per job (cross_subject) / one subject per job (within, cross
            # session). Filtered HERE, after MOABB built the folds on the full subject
            # list, so the training pool of a cross-subject fold is untouched. A filter
            # that keeps nothing refuses: an empty job would write an empty CSV that
            # looks like a finished cell.
            tasks = [t for t in tasks if str(t["subject"]) == str(only)]
            if not tasks:
                raise RuntimeError(f"only_subject={only!r}: no MOABB task holds this subject out")
        for task in tasks:
            try:
                stamp = {k: task[k] for k in STAMP_KEYS}
                stamp["subject"] = str(stamp["subject"])
                stamp["session"] = str(stamp["session"])
                stamp["cv_ind"] = int(stamp["cv_ind"])
                # What the fit was TRAINED on, read off the fold's own indices: the
                # weight analyses (subject/session fingerprint, adaptation from a
                # cross-subject model) need it, and it cannot be rebuilt from the
                # protocol name alone once MOABB's splitter is not at hand.
                tr = metadata.iloc[task["train_idx"]]
                stamp["train_subjects"] = sorted({str(s) for s in tr["subject"]})
                stamp["train_sessions"] = sorted({f"{s}:{se}" for s, se in zip(tr["subject"], tr["session"])})
                te = metadata.iloc[task["test_idx"]]
                stamp["n_test"] = int(len(te))
                stamp["test_sessions"] = sorted({f"{s}:{se}" for s, se in zip(te["subject"], te["session"])})
                task["pipeline"] = with_stamp(task["pipeline"], stamp)
            except Exception:  # pragma: no cover - never break a fit to label it
                logger.warning("subject stamp: task left unstamped", exc_info=True)
            # NOT inside the guard above: a campaign that asked for logits and cannot
            # write them must stop, not run 200 epochs for a missing file.
            rec = _find_recorder(task["pipeline"])
            if rec is not None and getattr(rec, "save_test_logits", False):
                test_idx = np.asarray(task["test_idx"])
                if task["config"]["score_per_session"]:
                    sess = metadata.iloc[test_idx]["session"].values
                    groups = [test_idx[sess == s] for s in np.unique(sess)]
                else:
                    groups = [test_idx]
                cols = [c for c in ("subject", "session", "run") if c in metadata]
                # Fixed-width unicode, not object: np.load refuses object arrays
                # without allow_pickle.
                groups = [(g, {c: np.asarray(metadata.iloc[g][c].astype(str).tolist(), dtype=str)
                               for c in cols})
                          for g in groups]
                folder = Path(rec.path).with_name(Path(rec.path).stem + "__logits")
                task["config"] = dict(task["config"])
                task["config"]["scoring"] = LogitScorer(task["config"]["scoring"], groups,
                                                        folder)
        return tasks


def stamped(eval_cls, only_subject=None):
    """A subclass of ``eval_cls`` that stamps subjects into its fit records.

    Built on demand rather than declared at import time, so this module stays free of
    a MOABB import for the callers that do not need one. ``only_subject`` keeps only
    the task(s) whose TEST subject it is (see ``_build_task_list``).
    """
    return type(f"Stamped{eval_cls.__name__}", (_SubjectStampMixin, eval_cls),
                {"only_subject_": only_subject})
