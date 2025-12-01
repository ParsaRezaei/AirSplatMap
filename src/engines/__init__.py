# Engine implementations for 3D Gaussian Splatting backends
# =========================================================
# 
# This package provides pluggable engine backends that implement the
# BaseGSEngine interface. Each engine wraps a different 3DGS implementation.
#
# Available engines:
#   - GraphdecoEngine: Original 3DGS from GRAPHDECO (gaussian-splatting repo)
#   - GSplatEngine: Nerfstudio's optimized gsplat library (4x less memory)
#   - SplaTAMEngine: Dense RGB-D SLAM with Gaussian Splatting
#   - GSLAMEngine: Gaussian-SLAM - Submap-based RGB-D SLAM
#   - MonoGSEngine: Gaussian Splatting SLAM (CVPR'24) - real-time mono/stereo/RGB-D
#   - PhotoSLAMEngine: Photo-SLAM (CVPR'24) - real-time photorealistic SLAM
#
# Usage:
#   from src.engines import GraphdecoEngine, GSplatEngine, get_engine
#   engine = get_engine("gsplat")  # or "graphdeco", "splatam", "gslam", "monogs", "photoslam"

from .base import BaseGSEngine
from .graphdeco_engine import GraphdecoEngine

# Optional engines (may require additional installation)
try:
    from .gsplat_engine import GSplatEngine
    _GSPLAT_AVAILABLE = True
except ImportError:
    GSplatEngine = None
    _GSPLAT_AVAILABLE = False

try:
    from .splatam_engine import SplaTAMEngine
    _SPLATAM_AVAILABLE = True
except ImportError:
    SplaTAMEngine = None
    _SPLATAM_AVAILABLE = False

try:
    from .gslam_engine import GSLAMEngine
    _GSLAM_AVAILABLE = True
except ImportError:
    GSLAMEngine = None
    _GSLAM_AVAILABLE = False

try:
    from .monogs_engine import MonoGSEngine
    _MONOGS_AVAILABLE = True
except ImportError:
    MonoGSEngine = None
    _MONOGS_AVAILABLE = False

try:
    from .photoslam_engine import PhotoSLAMEngine
    _PHOTOSLAM_AVAILABLE = True
except ImportError:
    PhotoSLAMEngine = None
    _PHOTOSLAM_AVAILABLE = False


def get_engine(name: str, **kwargs):
    """
    Factory function to get an engine by name.
    
    Args:
        name: Engine name - one of "graphdeco", "gsplat", "splatam", "gslam", "monogs", "photoslam"
        **kwargs: Additional arguments passed to engine constructor
        
    Returns:
        BaseGSEngine instance
        
    Raises:
        ValueError: If engine name is unknown
        ImportError: If required dependencies are not installed
    """
    name = name.lower()
    
    if name in ("graphdeco", "original", "3dgs"):
        return GraphdecoEngine(**kwargs)
    
    elif name in ("gsplat", "nerfstudio"):
        if not _GSPLAT_AVAILABLE:
            raise ImportError(
                "gsplat not available. Install with: pip install gsplat"
            )
        return GSplatEngine(**kwargs)
    
    elif name in ("splatam",):
        if not _SPLATAM_AVAILABLE:
            raise ImportError(
                "SplaTAM not available. Install from: "
                "https://github.com/spla-tam/SplaTAM"
            )
        return SplaTAMEngine(**kwargs)
    
    elif name in ("gslam", "gaussianslam", "gaussian-slam"):
        if not _GSLAM_AVAILABLE:
            raise ImportError(
                "Gaussian-SLAM not available. Install from: "
                "git clone https://github.com/VladimirYugay/Gaussian-SLAM.git"
            )
        return GSLAMEngine(**kwargs)
    
    elif name in ("monogs", "gaussiansplattingslam"):
        if not _MONOGS_AVAILABLE:
            raise ImportError(
                "MonoGS not available. Install from: "
                "git clone https://github.com/muskie82/MonoGS.git --recursive"
            )
        return MonoGSEngine(**kwargs)
    
    elif name in ("photoslam", "photo"):
        if not _PHOTOSLAM_AVAILABLE:
            raise ImportError(
                "Photo-SLAM not available. Install from: "
                "git clone https://github.com/HuajianUP/Photo-SLAM.git --recursive"
            )
        return PhotoSLAMEngine(**kwargs)
    
    else:
        available = ["graphdeco"]
        if _GSPLAT_AVAILABLE:
            available.append("gsplat")
        if _SPLATAM_AVAILABLE:
            available.append("splatam")
        if _GSLAM_AVAILABLE:
            available.append("gslam")
        if _MONOGS_AVAILABLE:
            available.append("monogs")
        if _PHOTOSLAM_AVAILABLE:
            available.append("photoslam")
        raise ValueError(
            f"Unknown engine: {name}. Available: {available}"
        )


def list_engines():
    """List available engines and their status."""
    engines = {
        "graphdeco": {
            "available": True,
            "description": "Original 3DGS from GRAPHDECO research",
            "install": "Included (gaussian-splatting submodule)",
            "speed": "~2-5 FPS",
            "realtime": False,
        },
        "gsplat": {
            "available": _GSPLAT_AVAILABLE,
            "description": "Nerfstudio's optimized implementation (4x less memory)",
            "install": "pip install gsplat",
            "speed": "~17 FPS",
            "realtime": True,
        },
        "splatam": {
            "available": _SPLATAM_AVAILABLE,
            "description": "Dense RGB-D SLAM with Gaussian Splatting",
            "install": "git clone https://github.com/spla-tam/SplaTAM",
            "speed": "~0.4 FPS",
            "realtime": False,
        },
        "gslam": {
            "available": _GSLAM_AVAILABLE,
            "description": "Gaussian-SLAM - Submap-based RGB-D SLAM",
            "install": "git clone https://github.com/VladimirYugay/Gaussian-SLAM.git",
            "speed": "~1-2 FPS",
            "realtime": False,
        },
        "monogs": {
            "available": _MONOGS_AVAILABLE,
            "description": "Gaussian Splatting SLAM (CVPR'24 Highlight) - Mono/Stereo/RGB-D",
            "install": "git clone https://github.com/muskie82/MonoGS.git --recursive",
            "speed": "~10 FPS",
            "realtime": True,
        },
        "photoslam": {
            "available": _PHOTOSLAM_AVAILABLE,
            "description": "Photo-SLAM (CVPR'24) - Photorealistic real-time SLAM",
            "install": "git clone https://github.com/HuajianUP/Photo-SLAM.git --recursive && ./build.sh",
            "speed": "Real-time",
            "realtime": True,
        },
    }
    return engines


__all__ = [
    "BaseGSEngine",
    "GraphdecoEngine", 
    "GSplatEngine",
    "SplaTAMEngine",
    "GSLAMEngine",
    "MonoGSEngine",
    "PhotoSLAMEngine",
    "get_engine",
    "list_engines",
]
