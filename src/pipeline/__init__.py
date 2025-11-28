# Pipeline components for online 3D Gaussian Splatting
# ====================================================
# 
# This package provides the orchestration layer for online 3DGS mapping:
# - Frame handling (data classes, sources, iterators)
# - Rolling-shutter correction abstractions
# - The main online pipeline that ties everything together

from .frames import Frame, FrameSource
from .rs_corrector import RSCorrector, IdentityRSCorrector
from .online_gs import OnlineGSPipeline

__all__ = [
    "Frame",
    "FrameSource", 
    "RSCorrector",
    "IdentityRSCorrector",
    "OnlineGSPipeline",
]
