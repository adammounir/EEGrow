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

import logging

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
                cb = type(cb)(path=cb.path, meta={**(cb.meta or {}), **stamp})
                hit = True
            rebuilt.append((name, cb) if named else cb)
        if hit:
            step.set_params(callbacks=rebuilt)
            return p
    return p


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
        for task in tasks:
            try:
                stamp = {k: task[k] for k in STAMP_KEYS}
                stamp["subject"] = str(stamp["subject"])
                stamp["session"] = str(stamp["session"])
                stamp["cv_ind"] = int(stamp["cv_ind"])
                task["pipeline"] = with_stamp(task["pipeline"], stamp)
            except Exception:  # pragma: no cover - never break a fit to label it
                logger.warning("subject stamp: task left unstamped", exc_info=True)
        return tasks


def stamped(eval_cls):
    """A subclass of ``eval_cls`` that stamps subjects into its fit records.

    Built on demand rather than declared at import time, so this module stays free of
    a MOABB import for the callers that do not need one.
    """
    return type(f"Stamped{eval_cls.__name__}", (_SubjectStampMixin, eval_cls), {})
