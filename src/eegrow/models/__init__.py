from __future__ import annotations

from eegrow.models.growing_bd_junctions import (GrowingContraWR, GrowingCTNet,
                                                GrowingEEGConformer, GrowingEEGNeX,
                                                GrowingMSVTNet, GrowingTIDNet)
from eegrow.models.growing_deep4 import GrowingDeep4Net
from eegrow.models.growing_deepeeg import GrowingDeepEEGNet
from eegrow.models.growing_eegnex import GrowingEEGNeX
from eegrow.models.growing_sccnet import GrowingSCCNet
from eegrow.models.growing_shallow import GrowingShallowFBCSPNet

__all__ = [
    "GrowingShallowFBCSPNet",
    "GrowingDeepEEGNet",
    "GrowingDeep4Net",
    "GrowingSCCNet",
    "GrowingEEGNeX",
    "GrowingEEGConformer",
    "GrowingContraWR",
    "GrowingTIDNet",
    "GrowingCTNet",
    "GrowingMSVTNet",
    "GrowingEEGNeX",
]
