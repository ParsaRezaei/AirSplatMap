"""
External VIO/VO System Wrappers
===============================

Unified interfaces for external Visual-Inertial Odometry systems.

These wrappers provide a common API to integrate external pose estimation:
- ORB-SLAM3: Feature-based VO/VIO/SLAM (C++ with Python bindings)
- OpenVINS: Research-grade tightly-coupled VIO
- DPVO: Deep Patch Visual Odometry (learning-based)
- DROID-SLAM: Deep visual SLAM (accurate but slower)

Note: Gaussian Splatting systems (MonoGS, SplaTAM, Gaussian-SLAM) are NOT
included here as they are mapping systems, not pure pose estimators.
Those belong in src/engines/ for the mapping pipeline.

All wrappers provide:
- Real-time pose output (when supported)
- Common ExternalVIOResult format
- IMU data integration (accelerometer + gyroscope)

Usage:
    from src.pose.external import ORBSlam3Wrapper, DPVOWrapper, list_available_backends
    
    # Check what's available
    backends = list_available_backends()
    print(backends)  # {'orbslam3': False, 'openvins': False, ...}
    
    # ORB-SLAM3 (requires separate installation)
    if backends['orbslam3']:
        slam = ORBSlam3Wrapper(vocab_path="...", config_path="...")
        slam.initialize()
        result = slam.process(image, timestamp, accel=accel, gyro=gyro)
"""

# Base classes (always available)
from .base_wrapper import BaseVIOWrapper, ExternalVIOResult, TrackingState

# Import wrappers - they handle missing dependencies gracefully
try:
    from .orbslam3_wrapper import ORBSlam3Wrapper, ORBSlam3Result
    _HAS_ORBSLAM3 = ORBSlam3Wrapper.is_available()
except ImportError:
    _HAS_ORBSLAM3 = False
    ORBSlam3Wrapper = None
    ORBSlam3Result = None

try:
    from .openvins_wrapper import OpenVINSWrapper, OpenVINSResult
    _HAS_OPENVINS = OpenVINSWrapper.is_available()
except ImportError:
    _HAS_OPENVINS = False
    OpenVINSWrapper = None
    OpenVINSResult = None

try:
    from .dpvo_wrapper import DPVOWrapper, DPVOResult
    _HAS_DPVO = DPVOWrapper.is_available()
except ImportError:
    _HAS_DPVO = False
    DPVOWrapper = None
    DPVOResult = None

try:
    from .droid_slam_wrapper import DROIDSLAMWrapper, DROIDSLAMResult
    _HAS_DROID = DROIDSLAMWrapper.is_available()
except ImportError:
    _HAS_DROID = False
    DROIDSLAMWrapper = None
    DROIDSLAMResult = None


def list_available_backends() -> dict:
    """
    List which external VIO backends are available.
    
    Returns:
        Dictionary with backend names and availability status
    """
    return {
        'orbslam3': _HAS_ORBSLAM3,
        'openvins': _HAS_OPENVINS,
        'dpvo': _HAS_DPVO,
        'droid_slam': _HAS_DROID,
    }


def get_available_wrapper(preference: list = None):
    """
    Get the first available VIO wrapper based on preference order.
    
    Args:
        preference: List of backend names in order of preference.
                   Default: ['orbslam3', 'openvins', 'dpvo', 'droid_slam']
    
    Returns:
        Wrapper class or None if none available
    """
    if preference is None:
        preference = ['orbslam3', 'openvins', 'dpvo', 'droid_slam']
    
    wrappers = {
        'orbslam3': ORBSlam3Wrapper if _HAS_ORBSLAM3 else None,
        'openvins': OpenVINSWrapper if _HAS_OPENVINS else None,
        'dpvo': DPVOWrapper if _HAS_DPVO else None,
        'droid_slam': DROIDSLAMWrapper if _HAS_DROID else None,
    }
    
    for name in preference:
        if wrappers.get(name) is not None:
            return wrappers[name]
    
    return None


__all__ = [
    # Base classes
    'BaseVIOWrapper',
    'ExternalVIOResult',
    'TrackingState',
    # Utility functions
    'list_available_backends',
    'get_available_wrapper',
    # Wrappers (may be None if not installed)
    'ORBSlam3Wrapper',
    'ORBSlam3Result',
    'OpenVINSWrapper',
    'OpenVINSResult',
    'DPVOWrapper',
    'DPVOResult',
    'DROIDSLAMWrapper',
    'DROIDSLAMResult',
]
