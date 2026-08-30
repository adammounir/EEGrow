"""Plug Euclidean Alignment into a MOABB paradigm.

Where to apply EA is the whole difficulty. EA is defined *per subject*, but a
scikit-learn step inside the pipeline never sees the subject ids: MOABB calls
``pipeline.fit(X[train], y[train])`` with no groups. The subject ids only exist one
level up, in the ``metadata`` frame returned by ``paradigm.get_data``.

MOABB does expose a ``postprocess_pipeline`` hook, but it fires in the innermost
``subject -> session -> run`` loop, i.e. it would align per *run* -- a different (and
undocumented) estimator from the one the papers evaluate. So we wrap ``get_data``
instead: it is public API, it hands us ``metadata`` alongside ``X``, and it lets the
alignment unit be an explicit knob.

The wrapper is a real subclass, built on the fly, so ``isinstance`` checks and every
attribute MOABB's evaluations read off the paradigm keep working.
"""

from __future__ import annotations

import numpy as np

from eegrow.alignment import euclidean_align
from utils import logger

logger_msg = "euclidean alignment: level=%s groups=%d trials=%d"

#: Alignment units, in increasing order of aggressiveness. ``subject`` reproduces
#: Junqueira et al.; ``session`` additionally removes session drift (cap re-placed,
#: impedances re-settled), which is the right unit when a subject spans days.
LEVELS = {
    "subject": ("subject",),
    "session": ("subject", "session"),
}


def group_keys(metadata, level: str) -> np.ndarray:
    """One label per trial identifying its alignment unit."""
    try:
        cols = LEVELS[level]
    except KeyError:
        raise ValueError(
            f"unknown alignment level {level!r}, expected one of {sorted(LEVELS)}"
        ) from None
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing column(s) {missing} for level={level!r}")
    return (
        metadata[list(cols)].astype(str).agg("|".join, axis=1).to_numpy()
        if len(cols) > 1
        else metadata[cols[0]].to_numpy()
    )


def make_aligned_paradigm(
    paradigm_cls,
    *,
    level: str = "subject",
    preserve_scale: bool = True,
    rcond: float = 1e-10,
):
    """Return a subclass of ``paradigm_cls`` whose ``get_data`` is EA-aligned."""
    if level not in LEVELS:
        raise ValueError(
            f"unknown alignment level {level!r}, expected one of {sorted(LEVELS)}"
        )

    class EuclideanAlignedParadigm(paradigm_cls):  # type: ignore[valid-type,misc]
        __doc__ = f"{paradigm_cls.__name__} with per-{level} Euclidean Alignment."

        align_level = level
        align_preserve_scale = preserve_scale
        align_rcond = rcond

        def get_data(self, dataset, subjects=None, return_epochs=False,
                     return_raws=False, **kwargs):
            X, labels, metadata = super().get_data(
                dataset=dataset, subjects=subjects, return_epochs=return_epochs,
                return_raws=return_raws, **kwargs)
            if return_epochs or return_raws:
                raise NotImplementedError(
                    "Euclidean alignment is implemented on the array output of "
                    "get_data; return_epochs/return_raws are not supported")
            if np.ndim(X) != 3:
                # a filter bank yields (n, chans, times, n_filters): each band has its
                # own covariance and would need its own reference. Out of scope here,
                # and silently aligning the wrong axis would be worse than failing.
                raise NotImplementedError(
                    f"expected single-band epochs (n, chans, times), got shape "
                    f"{np.shape(X)}; filter-bank alignment is not implemented")

            groups = group_keys(metadata, self.align_level)
            logger.info(logger_msg, self.align_level, len(np.unique(groups)), len(X))
            return (
                euclidean_align(X, groups, preserve_scale=self.align_preserve_scale,
                                rcond=self.align_rcond),
                labels,
                metadata,
            )

    EuclideanAlignedParadigm.__name__ = f"EA{paradigm_cls.__name__}"
    EuclideanAlignedParadigm.__qualname__ = EuclideanAlignedParadigm.__name__
    return EuclideanAlignedParadigm
