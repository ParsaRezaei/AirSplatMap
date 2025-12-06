"""
OpenVINS Wrapper for AirSplatMap
================================

Wrapper for OpenVINS, a research-grade tightly-coupled Visual-Inertial
state estimator developed at the University of Delaware.

OpenVINS Features:
- Tightly-coupled Visual-Inertial Odometry
- Multi-State Constraint Kalman Filter (MSCKF)
- Online camera-IMU calibration
- FEJ (First Estimates Jacobian) for consistency

Requirements:
- OpenVINS compiled with Python bindings (ROS optional)
- Camera and IMU configuration

Installation:
    # With ROS (recommended):
    cd ~/catkin_ws/src
    git clone https://github.com/rpng/open_vins.git
    cd .. && catkin_make
    
    # Without ROS: see OpenVINS documentation

Note: This wrapper provides a template. OpenVINS typically runs as a ROS
node, but can be used standalone with custom Python bindings.
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
import logging

from .base_wrapper import BaseVIOWrapper, ExternalVIOResult, TrackingState

logger = logging.getLogger(__name__)


# Check if OpenVINS is available
_OPENVINS_AVAILABLE = False
_openvins = None

try:
    import openvins as _openvins
    _OPENVINS_AVAILABLE = True
except ImportError:
    try:
        # ROS-based import
        import rospy
        from ov_msckf.msg import State
        _OPENVINS_AVAILABLE = True
        _openvins = 'ros'
    except ImportError:
        pass


@dataclass
class OpenVINSResult(ExternalVIOResult):
    """OpenVINS specific result with MSCKF state."""
    imu_bias_accel: Optional[np.ndarray] = None  # Accelerometer bias
    imu_bias_gyro: Optional[np.ndarray] = None   # Gyroscope bias
    num_slam_features: int = 0
    num_msckf_features: int = 0


class OpenVINSWrapper(BaseVIOWrapper):
    """
    Wrapper for OpenVINS Visual-Inertial State Estimator.
    
    OpenVINS uses a tightly-coupled MSCKF approach with:
    - Sliding window of camera poses
    - 3D SLAM features for long-term tracking
    - MSCKF features for short-term tracking
    - Online IMU bias estimation
    
    Usage:
        wrapper = OpenVINSWrapper(
            config_path="/path/to/estimator_config.yaml",
            use_stereo=True
        )
        wrapper.initialize()
        
        # Feed IMU at high rate (200-400 Hz)
        wrapper.add_imu(timestamp, accel, gyro)
        
        # Feed images at camera rate (20-30 Hz)
        result = wrapper.process(image, timestamp, image_right=right_img)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_stereo: bool = True,
        max_cameras: int = 2,
        config: Optional[Dict] = None,
    ):
        """
        Initialize OpenVINS wrapper.
        
        Args:
            config_path: Path to OpenVINS config YAML
            use_stereo: Use stereo camera setup
            max_cameras: Maximum number of cameras
            config: Additional configuration overrides
        """
        super().__init__(config)
        
        self._config_path = config_path
        self._use_stereo = use_stereo
        self._max_cameras = max_cameras
        
        self._vio = None
        self._trajectory: List[np.ndarray] = []
        self._imu_buffer: List[tuple] = []
        
        # State estimates
        self._bias_accel = np.zeros(3)
        self._bias_gyro = np.zeros(3)
    
    @staticmethod
    def is_available() -> bool:
        """Check if OpenVINS is available."""
        return _OPENVINS_AVAILABLE
    
    def initialize(self) -> bool:
        """Initialize OpenVINS system."""
        if not _OPENVINS_AVAILABLE:
            logger.error("OpenVINS not available")
            logger.info("To use OpenVINS:")
            logger.info("1. Install ROS (recommended) or build standalone")
            logger.info("2. Clone: git clone https://github.com/rpng/open_vins.git")
            logger.info("3. Build and source the workspace")
            return False
        
        try:
            if _openvins == 'ros':
                # ROS-based initialization
                logger.info("OpenVINS will run as ROS node")
                # Would typically start ROS node here
                self._initialized = True
            else:
                # Standalone Python bindings
                self._vio = _openvins.VioManager(self._config_path)
                self._initialized = True
            
            logger.info("OpenVINS initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINS: {e}")
            return False
    
    def add_imu(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        """
        Add IMU measurement to buffer.
        
        Should be called at IMU rate (typically 200-400 Hz).
        
        Args:
            timestamp: IMU timestamp in seconds
            accel: Accelerometer [ax, ay, az] in m/s²
            gyro: Gyroscope [gx, gy, gz] in rad/s
        """
        self._imu_buffer.append((timestamp, accel.copy(), gyro.copy()))
    
    def process(
        self,
        image: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        image_right: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> OpenVINSResult:
        """
        Process camera frame with OpenVINS.
        
        Note: IMU data should be added via add_imu() at high rate.
        The accel/gyro parameters here are for convenience if IMU
        is synced with camera.
        
        Args:
            image: Left camera image
            timestamp: Image timestamp
            depth: Not used (OpenVINS is feature-based)
            image_right: Right stereo image if stereo mode
            accel: Optional synced accelerometer
            gyro: Optional synced gyroscope
            
        Returns:
            OpenVINSResult with pose and VIO state
        """
        t_start = time.perf_counter()
        
        if not self._initialized:
            return self._create_failed_result(timestamp, t_start)
        
        self._frame_count += 1
        
        # Add synced IMU if provided
        if accel is not None and gyro is not None:
            self.add_imu(timestamp, accel, gyro)
        
        try:
            # Process IMU data first
            for ts, acc, gyr in self._imu_buffer:
                if self._vio is not None:
                    self._vio.feed_imu(ts, acc, gyr)
            self._imu_buffer.clear()
            
            # Process image(s)
            if self._vio is not None:
                if self._use_stereo and image_right is not None:
                    self._vio.feed_stereo(timestamp, image, image_right)
                else:
                    self._vio.feed_mono(timestamp, image)
                
                # Get state estimate
                state = self._vio.get_state()
                
                if state is not None:
                    # Extract pose (position + orientation)
                    position = np.array(state.position)
                    orientation = np.array(state.orientation)  # quaternion [x,y,z,w]
                    
                    # Build pose matrix
                    R = self._quaternion_to_rotation(orientation)
                    self._pose[:3, :3] = R
                    self._pose[:3, 3] = position
                    
                    # Extract biases
                    self._bias_accel = np.array(state.bias_accel)
                    self._bias_gyro = np.array(state.bias_gyro)
                    
                    # Extract velocity
                    velocity = np.array(state.velocity) if hasattr(state, 'velocity') else None
                    
                    self._trajectory.append(self._pose.copy())
                    tracking_state = TrackingState.TRACKING
                    confidence = 1.0
                else:
                    tracking_state = TrackingState.INITIALIZING
                    velocity = None
                    confidence = 0.5
            else:
                # ROS mode - would subscribe to state topic
                tracking_state = TrackingState.TRACKING
                velocity = None
                confidence = 0.8
            
            proc_time = (time.perf_counter() - t_start) * 1000
            
            return OpenVINSResult(
                pose=self._pose.copy(),
                velocity=velocity,
                timestamp=timestamp,
                tracking_state=tracking_state,
                confidence=confidence,
                num_features=0,  # Would get from state
                processing_time_ms=proc_time,
                imu_bias_accel=self._bias_accel.copy(),
                imu_bias_gyro=self._bias_gyro.copy(),
            )
            
        except Exception as e:
            logger.error(f"OpenVINS processing error: {e}")
            return self._create_failed_result(timestamp, t_start)
    
    def _create_failed_result(self, timestamp: float, t_start: float) -> OpenVINSResult:
        """Create a failed result."""
        proc_time = (time.perf_counter() - t_start) * 1000
        return OpenVINSResult(
            pose=self._pose.copy(),
            velocity=None,
            timestamp=timestamp,
            tracking_state=TrackingState.LOST,
            confidence=0.0,
            num_features=0,
            processing_time_ms=proc_time,
        )
    
    @staticmethod
    def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [x,y,z,w] to rotation matrix."""
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def get_imu_biases(self) -> Dict[str, np.ndarray]:
        """Get current IMU bias estimates."""
        return {
            'accel': self._bias_accel.copy(),
            'gyro': self._bias_gyro.copy(),
        }
    
    def shutdown(self):
        """Shutdown OpenVINS."""
        if self._vio is not None:
            self._vio = None
        self._initialized = False
        self._imu_buffer.clear()
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get trajectory."""
        return self._trajectory.copy()
