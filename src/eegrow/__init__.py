"""eegrow -- a bridge between `gromo` (architecture growth) and `braindecode`
(end-to-end EEG decoding).

The package provides braindecode-compatible models whose convolution-junction width
can *grow* during training via gromo, plus a training loop that handles periodic
growth.
"""

from __future__ import annotations

from eegrow.convert import GrowableSequential, make_growable
from eegrow.models.growing_deepeeg import GrowingDeepEEGNet
from eegrow.models.growing_shallow import GrowingShallowFBCSPNet

__all__ = [
    "GrowingShallowFBCSPNet",
    "GrowingDeepEEGNet",
    "make_growable",
    "GrowableSequential",
]
__version__ = "0.1.0"
