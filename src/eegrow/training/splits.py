"""A validation split that also sets aside a growth held-out, disjoint from both sides.

WHY THIS EXISTS
---------------
A growth step decides two things: *which* neurons (gromo's statistics) and *how loud*
(the line search over the scaling factor ``s``). The second decision was taken on data
the net had already fit. ``GromoGrowth`` held out the last 20 % of the epoch's
training batches -- batches the net had just trained on -- and only when the epoch had
at least 4 batches; on the previous campaign that left **92.7 %** of the 542 187 growth
events with no held-out at all, the line search running on the very batches the
statistics came from. Applied steps then chose the top of the grid on 97.8-100 % of
events (``analysis/step0_catalogue/line_search_censoring.py``): on data a candidate
was fitted to, the largest amplitude always wins.

The fix needs trials that are neither trained on nor used for selection. Taking them
from the validation split would let growth adapt to the split that carries the
``restored_epoch`` measure; halving that split would push lee2019_mi (n_valid 16)
under the n_valid >= 16 floor. So they come out of the TRAINING side, and -- decided
13/10 -- for every arm, fixed and braindecode references included: the arms are
compared on the same training trials, and a fixed arm simply never reads its held-out.

HOW
---
The validation split is drawn first, by skorch's own ``ValidSplit`` with the same
arguments as before, so ``n_valid`` and the valid trials are bit-identical to the
historical protocol. ``holdout`` (a fraction of the WHOLE fold, not of what is left)
is then drawn, stratified, from the training side. The held-out subset is left on the
split object as ``holdout_`` for ``GromoGrowth`` to read; nothing else sees it.
"""

from __future__ import annotations

import numpy as np
import torch
from skorch.dataset import ValidSplit, get_len
from skorch.utils import to_numpy


class HoldoutValidSplit(ValidSplit):
    """skorch ``ValidSplit`` plus a growth held-out carved from the training side.

    Parameters
    ----------
    cv, stratified, random_state
        As ``skorch.dataset.ValidSplit``; the validation side is exactly what
        ``ValidSplit(cv, stratified, random_state)`` returns.
    holdout : float
        Fraction of the fold (the data passed to ``fit``) set aside for the growth line
        search. ``0.0`` (default) is exactly ``ValidSplit``.

    Attributes set at each call
    ---------------------------
    ``holdout_`` : the held-out ``Subset``, or ``None``.
    ``n_train_``, ``n_valid_``, ``n_holdout_`` : the three sizes, for the fit record
    (``n_valid_`` is what the n_valid >= 16 floor is checked on).
    """

    def __init__(self, cv=5, stratified=False, random_state=None, holdout: float = 0.0):
        super().__init__(cv=cv, stratified=stratified, random_state=random_state)
        if not 0.0 <= float(holdout) < 1.0:
            raise ValueError(f"holdout must be in [0, 1), got {holdout!r}")
        self.holdout = holdout

    def __call__(self, dataset, y=None, groups=None):
        ds_train, ds_valid = super().__call__(dataset, y=y, groups=groups)
        self.holdout_ = None
        idx_train = np.asarray(ds_train.indices)
        n_hold = int(round(float(self.holdout) * get_len(dataset)))
        if n_hold > 0:
            from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

            y_train = None if y is None else to_numpy(y)[idx_train]
            if self.stratified:
                n_classes = len(np.unique(y_train))
                if n_hold < n_classes:
                    # A gate refuses, it does not degrade: a held-out missing a class
                    # would score the line search on a different task.
                    raise ValueError(
                        f"growth held-out of {n_hold} trials cannot hold all "
                        f"{n_classes} classes (fold of {get_len(dataset)} trials, "
                        f"holdout={self.holdout})")
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_hold,
                                                  random_state=self.random_state)
            else:
                splitter = ShuffleSplit(n_splits=1, test_size=n_hold,
                                        random_state=self.random_state)
            keep, hold = next(splitter.split(idx_train, y_train))
            self.holdout_ = torch.utils.data.Subset(dataset, idx_train[hold])
            ds_train = torch.utils.data.Subset(dataset, idx_train[keep])
        self.n_train_ = get_len(ds_train)
        self.n_valid_ = get_len(ds_valid)
        self.n_holdout_ = 0 if self.holdout_ is None else get_len(self.holdout_)
        return ds_train, ds_valid
