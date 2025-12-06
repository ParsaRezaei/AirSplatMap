"""
ORB-SLAM3 Wrapper for AirSplatMap
=================================

Wrapper for ORB-SLAM3, a versatile feature-based SLAM system supporting:
- Monocular, Stereo, RGB-D cameras
- Visual-Inertial modes (Mono-Inertial, Stereo-Inertial)
- IMU integration with accelerometer and gyroscope

Requirements:
- ORB-SLAM3 compiled with Python bindings
- Vocabulary file (ORBvoc.txt)
- Camera configuration YAML file

Installation:
    # Clone and build ORB-SLAM3
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
    cd ORB_SLAM3
    chmod +x build.sh && ./build.sh
    
    # For Python bindings, additional setup may be required
    # See: https://github.com/UZ-SLAMLab/ORB_SLAM3

Note: This wrapper provides a template interface. Actual Python bindings
for ORB-SLAM3 require custom compilation or third-party wrappers.
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
import logging

from .base_wrapper import BaseVIOWrapper, ExternalVIOResult, TrackingState

logger = logging.getLogger(__name__)


# Check if ORB-SLAM3 Python bindings are available
_ORBSLAM3_AVAILABLE = False
_orbslam3 = None

try:
    # Try common binding names
    import orbslam3 as _orbslam3
    _ORBSLAM3_AVAILABLE = True
except ImportError:
    try:
        import ORB_SLAM3 as _orbslam3
        _ORBSLAM3_AVAILABLE = True
    except ImportError:
        pass


@dataclass
class ORBSlam3Result(ExternalVIOResult):
    """ORB-SLAM3 specific result with additional fields."""
    loop_closure_detected: bool = False
    relocalization: bool = False


class ORBSlam3Wrapper(BaseVIOWrapper):
    """
    Wrapper for ORB-SLAM3 Visual(-Inertial) SLAM.
    
    Supports multiple sensor configurations:
    - MONOCULAR: Single camera
    - STEREO: Stereo camera pair
    - RGBD: RGB-D camera with depth
    - IMU_MONOCULAR: Monocular + IMU
    - IMU_STEREO: Stereo + IMU
    
    Usage:
        wrapper = ORBSlam3Wrapper(
            vocab_path="/path/to/ORBvoc.txt",
            config_path="/path/to/camera.yaml",
            sensor_type="IMU_STEREO"
        )
        wrapper.initialize()
        
        result = wrapper.process(
            image=left_image,
            timestamp=time.time(),
            image_right=right_image,
            accel=np.array([ax, ay, az]),
            gyro=np.array([gx, gy, gz])
        )
    """
    
    # Sensor type mappings
    SENSOR_TYPES = {
        'MONOCULAR': 0,
        'STEREO': 1,
        'RGBD': 2,
        'IMU_MONOCULAR': 3,
        'IMU_STEREO': 4,
    }
    
    def __init__(
        self,
        vocab_path: str,
        config_path: str,
        sensor_type: str = 'STEREO',
        use_viewer: bool = False,
        config: Optional[Dict] = None,
    ):
        """
        Initialize ORB-SLAM3 wrapper.
        
        Args:
            vocab_path: Path to ORB vocabulary file (ORBvoc.txt)
            config_path: Path to camera configuration YAML
            sensor_type: One of MONOCULAR, STEREO, RGBD, IMU_MONOCULAR, IMU_STEREO
            use_viewer: Enable visualization (requires Pangolin)
            config: Additional configuration
        """
        super().__init__(config)
        
        self._vocab_path = vocab_path
        self._config_path = config_path
        self._sensor_type = sensor_type.upper()
        self._use_viewer = use_viewer
        
        if self._sensor_type not in self.SENSOR_TYPES:
            raise ValueError(f"Invalid sensor type: {sensor_type}. "
                           f"Valid options: {list(self.SENSOR_TYPES.keys())}")
        
        self._slam = None
        self._trajectory: List[np.ndarray] = []
        self._imu_buffer: List[tuple] = []  # Buffer for IMU measurements
    
    @staticmethod
    def is_available() -> bool:
        """Check if ORB-SLAM3 is available."""
        return _ORBSLAM3_AVAILABLE
    
    def initialize(self) -> bool:
        """Initialize ORB-SLAM3 system."""
        if not _ORBSLAM3_AVAILABLE:
            logger.error("ORB-SLAM3 Python bindings not available")
            logger.info("To use ORB-SLAM3, you need to:")
            logger.info("1. Clone: git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git")
            logger.info("2. Build with Python bindings")
            logger.info("3. Ensure the module is in your PYTHONPATH")
            return False
        
        try:
            sensor_code = self.SENSOR_TYPES[self._sensor_type]
            
            # Create SLAM system
            self._slam = _orbslam3.System(
                self._vocab_path,
                self._config_path,
                sensor_code,
                self._use_viewer
            )
            
            self._initialized = True
            logger.info(f"ORB-SLAM3 initialized with {self._sensor_type} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ORB-SLAM3: {e}")
            return False
    
    def process(
        self,
        image: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        image_right: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> ORBSlam3Result:
        """
        Process a frame with ORB-SLAM3.
        
        Args:
            image: Left/mono image (BGR or grayscale)
            timestamp: Frame timestamp in seconds
            depth: Depth image for RGB-D mode
            image_right: Right image for stereo mode
            accel: Accelerometer [ax, ay, az] in m/s²
            gyro: Gyroscope [gx, gy, gz] in rad/s
            
        Returns:
            ORBSlam3Result with pose and tracking info
        """
        t_start = time.perf_counter()
        
        if not self._initialized:
            return self._create_failed_result(timestamp, t_start)
        
        self._frame_count += 1
        
        try:
            # Buffer IMU measurements if available
            if accel is not None and gyro is not None:
                self._imu_buffer.append((timestamp, accel, gyro))
            
            # Process based on sensor type
            if self._sensor_type == 'MONOCULAR':
                pose = self._slam.TrackMonocular(image, timestamp)
                
            elif self._sensor_type == 'STEREO':
                if image_right is None:
                    raise ValueError("Stereo mode requires right image")
                pose = self._slam.TrackStereo(image, image_right, timestamp)
                
            elif self._sensor_type == 'RGBD':
                if depth is None:
                    raise ValueError("RGB-D mode requires depth image")
                pose = self._slam.TrackRGBD(image, depth, timestamp)
                
            elif self._sensor_type == 'IMU_MONOCULAR':
                imu_data = self._prepare_imu_data()
                pose = self._slam.TrackMonocularIMU(image, timestamp, imu_data)
                
            elif self._sensor_type == 'IMU_STEREO':
                if image_right is None:
                    raise ValueError("Stereo-Inertial mode requires right image")
                imu_data = self._prepare_imu_data()
                pose = self._slam.TrackStereoIMU(image, image_right, timestamp, imu_data)
            
            # Convert pose to numpy if needed
            if pose is not None:
                pose = np.array(pose)
                if pose.shape == (4, 4):
                    self._pose = pose
                    self._trajectory.append(pose.copy())
                    tracking_state = TrackingState.TRACKING
                    confidence = 1.0
                else:
                    tracking_state = TrackingState.LOST
                    confidence = 0.0
            else:
                tracking_state = TrackingState.LOST
                confidence = 0.0
            
            # Get tracking statistics
            num_features = self._slam.GetTrackingState() if hasattr(self._slam, 'GetTrackingState') else 0
            
            proc_time = (time.perf_counter() - t_start) * 1000
            
            return ORBSlam3Result(
                pose=self._pose.copy(),
                velocity=None,
                timestamp=timestamp,
                tracking_state=tracking_state,
                confidence=confidence,
                num_features=num_features,
                processing_time_ms=proc_time,
                loop_closure_detected=False,  # Would need callback
                relocalization=False,
            )
            
        except Exception as e:
            logger.error(f"ORB-SLAM3 processing error: {e}")
            return self._create_failed_result(timestamp, t_start)
    
    def _prepare_imu_data(self) -> list:
        """Prepare IMU data buffer for processing."""
        imu_data = []
        for ts, accel, gyro in self._imu_buffer:
            # Format: [ax, ay, az, gx, gy, gz, timestamp]
            imu_data.append([
                accel[0], accel[1], accel[2],
                gyro[0], gyro[1], gyro[2],
                ts
            ])
        self._imu_buffer.clear()
        return imu_data
    
    def _create_failed_result(self, timestamp: float, t_start: float) -> ORBSlam3Result:
        """Create a failed/lost result."""
        proc_time = (time.perf_counter() - t_start) * 1000
        return ORBSlam3Result(
            pose=self._pose.copy(),
            velocity=None,
            timestamp=timestamp,
            tracking_state=TrackingState.LOST,
            confidence=0.0,
            num_features=0,
            processing_time_ms=proc_time,
        )
    
    def shutdown(self):
        """Shutdown ORB-SLAM3."""
        if self._slam is not None:
            try:
                self._slam.Shutdown()
            except Exception as e:
                logger.warning(f"Error during ORB-SLAM3 shutdown: {e}")
            self._slam = None
        self._initialized = False
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get the full trajectory."""
        return self._trajectory.copy()
    
    def get_map_points(self) -> Optional[np.ndarray]:
        """Get 3D map points if available."""
        if self._slam is None:
            return None
        try:
            if hasattr(self._slam, 'GetAllMapPoints'):
                points = self._slam.GetAllMapPoints()
                return np.array(points)
        except Exception:
            pass
        return None
    
    def save_map(self, path: str):
        """Save the map to file."""
        if self._slam is not None:
            try:
                self._slam.SaveMap(path)
                logger.info(f"Map saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save map: {e}")
    
    def load_map(self, path: str) -> bool:
        """Load a map from file."""
        if self._slam is not None:
            try:
                self._slam.LoadMap(path)
                logger.info(f"Map loaded from {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load map: {e}")
        return False
