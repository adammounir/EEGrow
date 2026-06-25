from __future__ import annotations

from eegrow.models.growing_deepeeg import GrowingDeepEEGNet
from eegrow.models.growing_sccnet import GrowingSCCNet
from eegrow.models.growing_shallow import GrowingShallowFBCSPNet

__all__ = [
    "GrowingShallowFBCSPNet",
    "GrowingDeepEEGNet",
    "GrowingSCCNet",
]
