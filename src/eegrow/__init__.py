"""eegrow -- a bridge between `gromo` (architecture growth) and `braindecode`
(end-to-end EEG decoding).

The package provides braindecode-compatible models whose convolution-junction width
can *grow* during training via gromo, plus a training loop that handles periodic
growth.
"""

from __future__ import annotations

from eegrow.alignment import euclidean_align
from eegrow.convert import GrowableSequential, make_growable
from eegrow.models.growing_deepeeg import GrowingDeepEEGNet
from eegrow.models.growing_eegnex import GrowingEEGNeX
from eegrow.models.growing_sccnet import GrowingSCCNet
from eegrow.models.growing_shallow import GrowingShallowFBCSPNet
from eegrow.training.recording import FitRecorder
from eegrow.training.skorch_integration import GromoGrowth

__all__ = [
    "GrowingShallowFBCSPNet",
    "GrowingDeepEEGNet",
    "GrowingSCCNet",
    "GrowingEEGNeX",
    "make_growable",
    "GrowableSequential",
    "GromoGrowth",
    "euclidean_align",
    "FitRecorder",
]
__version__ = "0.1.0"
