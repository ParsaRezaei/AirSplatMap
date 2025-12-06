"""
Base class for external VIO/VO system wrappers.

Provides a common interface for integrating external pose estimation systems.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from enum import Enum


class TrackingState(Enum):
    """Tracking state for external systems."""
    NOT_INITIALIZED = 0
    INITIALIZING = 1
    TRACKING = 2
    LOST = 3
    RELOCALIZING = 4


@dataclass
class ExternalVIOResult:
    """
    Common result format for external VIO systems.
    
    All external wrappers should return this format for consistency.
    """
    pose: np.ndarray  # 4x4 camera-to-world transformation
    velocity: Optional[np.ndarray]  # 3D velocity if available
    timestamp: float
    tracking_state: TrackingState
    confidence: float  # 0.0 to 1.0
    num_features: int  # Number of tracked features
    processing_time_ms: float
    
    # Optional additional data
    covariance: Optional[np.ndarray] = None  # 6x6 pose covariance
    keyframe: bool = False  # Whether this frame is a keyframe
    map_points: int = 0  # Number of map points
    extra: Optional[Dict[str, Any]] = None  # System-specific data


class BaseVIOWrapper(ABC):
    """
    Abstract base class for external VIO system wrappers.
    
    Subclasses should implement:
    - initialize(): Set up the system
    - process(): Process a frame and return pose
    - shutdown(): Clean up resources
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize wrapper with optional configuration.
        
        Args:
            config: System-specific configuration dictionary
        """
        self._config = config or {}
        self._initialized = False
        self._pose = np.eye(4)
        self._frame_count = 0
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the external system.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        image_right: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> ExternalVIOResult:
        """
        Process a frame and return pose estimate.
        
        Args:
            image: RGB or grayscale image (left image for stereo)
            timestamp: Frame timestamp in seconds
            depth: Optional depth image
            image_right: Optional right stereo image
            accel: Optional accelerometer reading [ax, ay, az]
            gyro: Optional gyroscope reading [gx, gy, gz]
            
        Returns:
            ExternalVIOResult with pose and tracking info
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown the system and release resources."""
        pass
    
    def reset(self):
        """Reset the system to initial state."""
        self.shutdown()
        self._pose = np.eye(4)
        self._frame_count = 0
        self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._initialized
    
    def get_pose(self) -> np.ndarray:
        """Get current pose estimate."""
        return self._pose.copy()
    
    def set_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: Optional[float] = None,
    ):
        """
        Set camera intrinsics.
        
        Args:
            fx, fy: Focal lengths
            cx, cy: Principal point
            baseline: Stereo baseline (for stereo systems)
        """
        self._config['fx'] = fx
        self._config['fy'] = fy
        self._config['cx'] = cx
        self._config['cy'] = cy
        if baseline is not None:
            self._config['baseline'] = baseline
    
    def set_intrinsics_from_dict(self, intrinsics: Dict):
        """Set intrinsics from dictionary."""
        self.set_intrinsics(
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy'],
            baseline=intrinsics.get('baseline'),
        )
    
    @property
    def name(self) -> str:
        """Get system name."""
        return self.__class__.__name__.replace('Wrapper', '')
    
    @staticmethod
    def is_available() -> bool:
        """Check if this system is available (dependencies installed)."""
        return False
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get full trajectory if available."""
        return []
    
    def save_trajectory(self, path: str, format: str = 'tum'):
        """
        Save trajectory to file.
        
        Args:
            path: Output file path
            format: 'tum' or 'kitti'
        """
        trajectory = self.get_trajectory()
        if not trajectory:
            return
        
        with open(path, 'w') as f:
            for i, pose in enumerate(trajectory):
                if format == 'tum':
                    # TUM format: timestamp tx ty tz qx qy qz qw
                    t = pose[:3, 3]
                    R = pose[:3, :3]
                    q = self._rotation_to_quaternion(R)
                    f.write(f"{i} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
                elif format == 'kitti':
                    # KITTI format: 12 values (3x4 matrix row-major)
                    row = pose[:3, :].flatten()
                    f.write(' '.join(map(str, row)) + '\n')
    
    @staticmethod
    def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [qx, qy, qz, qw]."""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
