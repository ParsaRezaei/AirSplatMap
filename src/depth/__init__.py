"""
Depth Estimation Module
=======================

Provides modular depth estimation backends for AirSplatMap.

Estimator Categories:
---------------------

**Relative Depth (requires scaling for metric):**
- depth_anything_v2: Depth Anything V2 - fast, accurate
- depth_anything_v3: Depth Anything V3 - latest version
- midas: MiDaS - robust, well-tested

**Metric Depth (outputs meters directly):**
- depth_pro: Apple Depth Pro - high quality metric depth
- depth_pro_lite: Apple Depth Pro Lite - faster version

**Stereo Depth (from camera pairs):**
- stereo: SGBM stereo matching with WLS filter
- stereo_fast: Simple block matching (faster)

**Passthrough:**
- none/passthrough: Use when depth comes from sensor

Usage Examples:
---------------

    from src.depth import get_depth_estimator, DepthScaler, DepthFilter
    
    # Relative depth estimation
    estimator = get_depth_estimator('depth_anything_v2', model_size='vits')
    result = estimator.estimate(rgb_image)
    relative_depth = result.depth
    
    # Metric depth estimation (no scaling needed!)
    estimator = get_depth_estimator('depth_pro')
    result = estimator.estimate(rgb_image)
    metric_depth = result.depth  # Already in meters
    
    # Stereo depth from camera pair
    stereo = get_depth_estimator('stereo', baseline=0.05, focal_length=382.6)
    result = stereo.estimate_stereo(left_ir, right_ir)
    
    # Scale relative depth using sparse 3D points
    scaler = DepthScaler(method='ransac')
    metric_depth = scaler.scale_to_metric(relative_depth, points_3d, points_2d)
    
    # Filter depth for temporal consistency
    filter = DepthFilter(temporal_alpha=0.3)
    filtered = filter.temporal_filter(depth)
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Base classes (always available)
from .base import (
    BaseDepthEstimator,
    DepthResult,
    PassthroughDepthEstimator,
    DepthScaler,
    DepthFilter,
)

# Import estimators with graceful fallbacks
_ESTIMATORS: Dict[str, type] = {
    'none': PassthroughDepthEstimator,
    'passthrough': PassthroughDepthEstimator,
    'ground_truth': PassthroughDepthEstimator,
}

# Depth Anything V2 and V3
try:
    from .depth_anything import DepthAnythingV2Estimator, DepthAnythingV3Estimator
    _ESTIMATORS['depth_anything_v2'] = DepthAnythingV2Estimator
    _ESTIMATORS['depth_anything'] = DepthAnythingV2Estimator
    _ESTIMATORS['dav2'] = DepthAnythingV2Estimator
    _ESTIMATORS['depth_anything_v3'] = DepthAnythingV3Estimator
    _ESTIMATORS['dav3'] = DepthAnythingV3Estimator
    _ESTIMATORS['da3'] = DepthAnythingV3Estimator
except ImportError as e:
    logger.debug(f"Depth Anything not available: {e}")
    DepthAnythingV2Estimator = None
    DepthAnythingV3Estimator = None

# MiDaS
try:
    from .midas import MiDaSEstimator
    _ESTIMATORS['midas'] = MiDaSEstimator
    _ESTIMATORS['midas_small'] = lambda **kw: MiDaSEstimator(model_type='small', **kw)
    _ESTIMATORS['midas_large'] = lambda **kw: MiDaSEstimator(model_type='large', **kw)
except ImportError as e:
    logger.debug(f"MiDaS not available: {e}")
    MiDaSEstimator = None

# Stereo depth
try:
    from .stereo import StereoDepthEstimator, FastStereoEstimator
    _ESTIMATORS['stereo'] = StereoDepthEstimator
    _ESTIMATORS['stereo_sgbm'] = StereoDepthEstimator
    _ESTIMATORS['stereo_fast'] = FastStereoEstimator
    _ESTIMATORS['stereo_bm'] = FastStereoEstimator
except ImportError as e:
    logger.debug(f"Stereo depth not available: {e}")
    StereoDepthEstimator = None
    FastStereoEstimator = None

# Apple Depth Pro (metric, high quality)
try:
    from .depth_pro import DepthProEstimator, DepthProLiteEstimator
    _ESTIMATORS['depth_pro'] = DepthProEstimator
    _ESTIMATORS['apple_depth_pro'] = DepthProEstimator
    _ESTIMATORS['depthpro'] = DepthProEstimator
    _ESTIMATORS['depth_pro_lite'] = DepthProLiteEstimator
except ImportError as e:
    logger.debug(f"Apple Depth Pro not available: {e}")
    DepthProEstimator = None
    DepthProLiteEstimator = None


def get_depth_estimator(name: str, **kwargs) -> BaseDepthEstimator:
    """
    Get a depth estimator by name.
    
    Args:
        name: Estimator name (see list_depth_estimators())
        **kwargs: Additional arguments for estimator constructor
        
    Returns:
        BaseDepthEstimator instance
        
    Raises:
        ValueError: If estimator name is unknown
        
    Examples:
        # Relative depth (fast)
        estimator = get_depth_estimator('depth_anything_v2', model_size='vits')
        
        # Metric depth (no scaling needed)
        estimator = get_depth_estimator('depth_pro')
        
        # Stereo depth
        estimator = get_depth_estimator('stereo', baseline=0.05, focal_length=382.6)
    """
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    
    if name_lower not in _ESTIMATORS:
        available = list(_ESTIMATORS.keys())
        raise ValueError(f"Unknown depth estimator: {name}. Available: {available}")
    
    estimator_cls = _ESTIMATORS[name_lower]
    
    if estimator_cls is None:
        raise ValueError(f"Depth estimator '{name}' dependencies not installed")
    
    # Handle factory functions
    if callable(estimator_cls) and not isinstance(estimator_cls, type):
        return estimator_cls(**kwargs)
    
    return estimator_cls(**kwargs)


def list_depth_estimators() -> Dict[str, Dict[str, Any]]:
    """
    List available depth estimators with their properties.
    
    Returns:
        Dictionary mapping estimator names to their info
    """
    estimators = {
        'depth_pro': {
            'description': 'Apple Depth Pro - sharp metric depth with focal length estimation',
            'metric': True,
            'available': DepthProEstimator is not None,
            'features': ['metric depth', 'focal length estimation', 'high boundary accuracy'],
            'speed': 'fast (~0.3s)',
            'gpu': True,
        },
        'depth_pro_lite': {
            'description': 'Apple Depth Pro Lite - faster version with lower resolution',
            'metric': True,
            'available': DepthProLiteEstimator is not None,
            'speed': 'faster',
            'gpu': True,
        },
        'depth_anything_v3': {
            'description': 'Depth Anything V3 - latest and most accurate',
            'metric': False,
            'available': DepthAnythingV3Estimator is not None,
            'models': ['small (fast)', 'base (balanced)', 'large (accurate)', 'giant (best)'],
        },
        'depth_anything_v2': {
            'description': 'Depth Anything V2 - fast and accurate relative depth',
            'metric': False,
            'available': DepthAnythingV2Estimator is not None,
            'models': ['vits (fast)', 'vitb (balanced)', 'vitl (accurate)'],
        },
        'midas': {
            'description': 'MiDaS - robust relative depth estimation',
            'metric': False,
            'available': MiDaSEstimator is not None,
            'models': ['small (fast)', 'hybrid (balanced)', 'large (accurate)'],
        },
        'stereo': {
            'description': 'Stereo SGBM - metric depth from stereo pairs',
            'metric': True,
            'available': StereoDepthEstimator is not None,
            'requires': 'Stereo camera pair (e.g., RealSense IR cameras)',
        },
        'stereo_fast': {
            'description': 'Stereo BM - fast stereo matching',
            'metric': True,
            'available': FastStereoEstimator is not None,
        },
        'none': {
            'description': 'Passthrough - use when depth from sensor',
            'metric': True,
            'available': True,
        },
    }
    return estimators


def list_available() -> list:
    """List names of available (installed) estimators."""
    return [name for name, cls in _ESTIMATORS.items() if cls is not None]


__all__ = [
    # Base classes
    'BaseDepthEstimator',
    'DepthResult',
    'DepthScaler',
    'DepthFilter',
    # Estimator classes
    'PassthroughDepthEstimator',
    'DepthAnythingV2Estimator',
    'DepthAnythingV3Estimator',
    'MiDaSEstimator',
    'StereoDepthEstimator',
    'FastStereoEstimator',
    'DepthProEstimator',
    'DepthProLiteEstimator',
    # Factory functions
    'get_depth_estimator',
    'list_depth_estimators',
    'list_available',
]
