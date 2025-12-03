#!/usr/bin/env python3
"""
ArduPilot Frame Source
======================

Combines camera streaming with ArduPilot MAVLink pose data for
drone/vehicle-mounted 3D Gaussian Splatting.

Supports:
- Any OpenCV-compatible camera (USB, CSI, IP cameras)
- RTSP streams from companion computers
- GStreamer pipelines
- RealSense cameras (if available)

Pose sources:
- ArduPilot MAVLink (primary)
- Fallback to visual odometry if MAVLink unavailable

Usage:
    from src.pipeline.ardupilot_source import ArduPilotSource
    
    # Drone with RTSP camera and Pixhawk
    source = ArduPilotSource(
        camera_source="rtsp://192.168.1.100:8554/main",
        mavlink_connection="udpin:0.0.0.0:14550",
    )
    
    for frame in source:
        pipeline.step(frame)
    
    source.stop()
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ArduPilotSourceConfig:
    """Configuration for ArduPilot frame source."""
    # Camera settings
    camera_source: Any = 0  # Camera index, RTSP URL, or GStreamer pipeline
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_fov_deg: float = 60.0  # Horizontal FOV for intrinsics estimation
    
    # MAVLink settings
    mavlink_connection: str = "udpin:0.0.0.0:14550"
    mavlink_timeout: float = 5.0
    
    # Camera mounting (offset from vehicle center, body frame)
    camera_offset_x: float = 0.0  # Forward (meters)
    camera_offset_y: float = 0.0  # Right (meters)
    camera_offset_z: float = -0.1  # Down (meters, negative = up)
    
    # Camera rotation relative to body (roll, pitch, yaw in degrees)
    # Default: looking forward and down 45 degrees
    camera_roll_deg: float = 0.0
    camera_pitch_deg: float = -45.0
    camera_yaw_deg: float = 0.0
    
    # Processing settings
    target_fps: float = 15.0
    max_frames: Optional[int] = None
    use_depth_estimation: bool = True
    depth_model: str = 'depth_anything_v2'
    
    # Fallback settings
    fallback_to_visual_odom: bool = True
    visual_odom_model: str = 'robust_flow'


class ArduPilotSource:
    """
    Frame source for drone/vehicle-mounted cameras with ArduPilot pose.
    
    Implements the FrameSource interface for seamless integration with
    the AirSplatMap pipeline.
    """
    
    def __init__(
        self,
        camera_source: Any = 0,
        mavlink_connection: str = "udpin:0.0.0.0:14550",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fov_deg: float = 60.0,
        camera_offset: Optional[np.ndarray] = None,
        camera_pitch_deg: float = -45.0,
        target_fps: float = 15.0,
        max_frames: Optional[int] = None,
        use_depth_estimation: bool = True,
        depth_model: str = 'depth_anything_v2',
        fallback_to_visual_odom: bool = True,
        intrinsics: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize ArduPilot frame source.
        
        Args:
            camera_source: OpenCV camera source (index, URL, or GStreamer)
            mavlink_connection: MAVLink connection string
            width: Frame width
            height: Frame height
            fps: Camera FPS
            fov_deg: Horizontal field of view (for intrinsics estimation)
            camera_offset: [x, y, z] camera offset from vehicle center (body frame)
            camera_pitch_deg: Camera pitch angle (negative = looking down)
            target_fps: Target processing FPS
            max_frames: Maximum frames to capture
            use_depth_estimation: Enable monocular depth estimation
            depth_model: Depth model to use
            fallback_to_visual_odom: Fall back to visual odometry if MAVLink fails
            intrinsics: Optional camera intrinsics dict
        """
        self._camera_source = camera_source
        self._mavlink_connection = mavlink_connection
        self._width = width
        self._height = height
        self._fps = fps
        self._fov_deg = fov_deg
        self._target_fps = target_fps
        self._max_frames = max_frames
        self._use_depth = use_depth_estimation
        self._depth_model_name = depth_model
        self._fallback_to_vo = fallback_to_visual_odom
        
        # Camera mounting
        if camera_offset is not None:
            self._camera_offset = np.array(camera_offset, dtype=np.float64)
        else:
            self._camera_offset = np.array([0.0, 0.0, -0.1])  # 10cm above vehicle center
        
        # Camera rotation (body -> camera)
        pitch_rad = np.radians(camera_pitch_deg)
        self._camera_rotation = self._compute_camera_rotation(0, pitch_rad, 0)
        
        # Compute intrinsics
        if intrinsics:
            self._intrinsics = intrinsics
        else:
            self._intrinsics = self._compute_intrinsics()
        
        # Components (initialized later)
        self._cap: Optional[cv2.VideoCapture] = None
        self._ardupilot = None
        self._depth_estimator = None
        self._pose_estimator = None
        
        # State
        self._frame_idx = 0
        self._last_mavlink_pose: Optional[np.ndarray] = None
        self._using_fallback = False
        self._running = False
        
    def _compute_intrinsics(self) -> Dict[str, float]:
        """Compute camera intrinsics from FOV."""
        fx = self._width / (2.0 * np.tan(np.radians(self._fov_deg) / 2.0))
        fy = fx  # Assume square pixels
        cx = self._width / 2.0
        cy = self._height / 2.0
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'width': self._width,
            'height': self._height,
        }
    
    @staticmethod
    def _compute_camera_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute camera rotation matrix from Euler angles."""
        import math
        
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        # ZYX rotation
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        
        return Rz @ Ry @ Rx
    
    def _init_camera(self) -> bool:
        """Initialize camera capture."""
        logger.info(f"Opening camera: {self._camera_source}")
        
        # Handle GStreamer pipelines
        if isinstance(self._camera_source, str) and 'gst-launch' in self._camera_source.lower():
            self._cap = cv2.VideoCapture(self._camera_source, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self._camera_source)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera: {self._camera_source}")
            return False
        
        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        
        # Get actual resolution
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w != self._width or actual_h != self._height:
            logger.warning(f"Camera resolution {actual_w}x{actual_h} differs from requested {self._width}x{self._height}")
            self._width = actual_w
            self._height = actual_h
            self._intrinsics = self._compute_intrinsics()
        
        logger.info(f"Camera opened: {self._width}x{self._height}")
        return True
    
    def _init_ardupilot(self) -> bool:
        """Initialize ArduPilot connection."""
        try:
            from src.pose.ardupilot_mavlink import ArduPilotPoseProvider
            
            self._ardupilot = ArduPilotPoseProvider(
                connection_string=self._mavlink_connection,
                camera_offset=self._camera_offset,
                camera_rotation=self._camera_rotation,
            )
            
            if self._ardupilot.start():
                logger.info("ArduPilot connection established")
                return True
            else:
                logger.warning("ArduPilot connection failed")
                return False
                
        except ImportError:
            logger.warning("pymavlink not installed, ArduPilot connection unavailable")
            return False
        except Exception as e:
            logger.warning(f"ArduPilot init failed: {e}")
            return False
    
    def _init_depth_estimator(self):
        """Initialize depth estimator."""
        if not self._use_depth:
            return
        
        try:
            from src.depth import get_depth_estimator
            self._depth_estimator = get_depth_estimator(self._depth_model_name)
            logger.info(f"Depth estimator initialized: {self._depth_model_name}")
        except Exception as e:
            logger.warning(f"Depth estimator init failed: {e}")
            self._depth_estimator = None
    
    def _init_pose_estimator(self):
        """Initialize fallback pose estimator."""
        if not self._fallback_to_vo:
            return
        
        try:
            from src.pose import get_pose_estimator
            self._pose_estimator = get_pose_estimator('robust_flow')
            self._pose_estimator.set_intrinsics_from_dict(self._intrinsics)
            logger.info("Fallback pose estimator initialized")
        except Exception as e:
            logger.warning(f"Pose estimator init failed: {e}")
            self._pose_estimator = None
    
    def start(self) -> bool:
        """Start all components."""
        # Initialize camera
        if not self._init_camera():
            return False
        
        # Initialize ArduPilot
        ardupilot_ok = self._init_ardupilot()
        
        # Initialize fallback if needed
        if not ardupilot_ok and self._fallback_to_vo:
            self._init_pose_estimator()
            self._using_fallback = True
        
        # Initialize depth
        self._init_depth_estimator()
        
        self._running = True
        return True
    
    def stop(self):
        """Stop all components."""
        self._running = False
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._ardupilot:
            self._ardupilot.stop()
            self._ardupilot = None
    
    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics."""
        return self._intrinsics.copy()
    
    def get_ardupilot_state(self):
        """Get current ArduPilot state (if connected)."""
        if self._ardupilot:
            return self._ardupilot.get_state()
        return None
    
    def is_mavlink_connected(self) -> bool:
        """Check if MAVLink is connected."""
        return self._ardupilot is not None and self._ardupilot.is_connected()
    
    def __iter__(self) -> Iterator:
        """Iterate over frames."""
        # Lazy initialization
        if not self._running:
            if not self.start():
                raise RuntimeError("Failed to start ArduPilotSource")
        
        # Import Frame here to avoid circular imports
        from src.pipeline.frames import Frame
        
        frame_interval = 1.0 / self._target_fps
        last_frame_time = 0.0
        
        while self._running:
            # Rate limiting
            now = time.time()
            if now - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue
            
            # Check max frames
            if self._max_frames and self._frame_idx >= self._max_frames:
                break
            
            # Capture frame
            ret, bgr = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue
            
            # Convert to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if rgb.shape[1] != self._width or rgb.shape[0] != self._height:
                rgb = cv2.resize(rgb, (self._width, self._height))
            
            # Get pose from ArduPilot
            pose = None
            if self._ardupilot and not self._using_fallback:
                pose = self._ardupilot.get_pose()
                self._last_mavlink_pose = pose
            
            # Fallback to visual odometry
            if pose is None and self._pose_estimator:
                result = self._pose_estimator.estimate(rgb)
                if result and result.tracking_status == 'ok':
                    pose = result.pose
            
            # Use identity if no pose available
            if pose is None:
                pose = np.eye(4, dtype=np.float64)
            
            # Estimate depth
            depth = None
            if self._depth_estimator:
                try:
                    depth = self._depth_estimator.estimate(rgb)
                except Exception as e:
                    logger.debug(f"Depth estimation failed: {e}")
            
            # Create frame
            frame = Frame(
                idx=self._frame_idx,
                timestamp=now,
                rgb=rgb,
                depth=depth,
                pose=pose,
                intrinsics=self._intrinsics,
                metadata={
                    'source': 'ardupilot',
                    'using_fallback': self._using_fallback,
                    'mavlink_connected': self._ardupilot is not None and not self._using_fallback,
                }
            )
            
            self._frame_idx += 1
            last_frame_time = now
            
            yield frame
    
    def __len__(self) -> int:
        """Return frame count (-1 for streaming)."""
        if self._max_frames:
            return self._max_frames
        return -1


def create_ardupilot_source(config: ArduPilotSourceConfig) -> ArduPilotSource:
    """Factory function to create ArduPilotSource from config."""
    camera_offset = np.array([
        config.camera_offset_x,
        config.camera_offset_y,
        config.camera_offset_z
    ])
    
    return ArduPilotSource(
        camera_source=config.camera_source,
        mavlink_connection=config.mavlink_connection,
        width=config.camera_width,
        height=config.camera_height,
        fps=config.camera_fps,
        fov_deg=config.camera_fov_deg,
        camera_offset=camera_offset,
        camera_pitch_deg=config.camera_pitch_deg,
        target_fps=config.target_fps,
        max_frames=config.max_frames,
        use_depth_estimation=config.use_depth_estimation,
        depth_model=config.depth_model,
        fallback_to_visual_odom=config.fallback_to_visual_odom,
    )
