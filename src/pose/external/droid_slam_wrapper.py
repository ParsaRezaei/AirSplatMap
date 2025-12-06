"""
DROID-SLAM Wrapper for AirSplatMap
==================================

Wrapper for DROID-SLAM, a deep visual SLAM system that achieves
state-of-the-art accuracy through dense bundle adjustment.

DROID-SLAM Features:
- Dense optical flow via correlation volumes  
- Differentiable bundle adjustment layer
- Global optimization over all frames
- Very high accuracy but computationally intensive

Requirements:
- PyTorch with CUDA
- DROID-SLAM package

Installation:
    git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
    cd DROID-SLAM
    pip install -e .
    pip install lietorch
    
    # Download pretrained weights
    ./download_weights.sh

Note: DROID-SLAM is designed for offline/batch processing due to
its computational requirements. Real-time use requires high-end GPU.

Paper: "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
import logging

from .base_wrapper import BaseVIOWrapper, ExternalVIOResult, TrackingState

logger = logging.getLogger(__name__)


# Check if DROID-SLAM is available
_DROID_AVAILABLE = False
_droid = None
_torch = None

try:
    import torch as _torch
    if _torch.cuda.is_available():
        from droid_slam import Droid as _Droid
        _droid = _Droid
        _DROID_AVAILABLE = True
    else:
        logger.warning("DROID-SLAM requires CUDA but no GPU available")
except ImportError:
    pass


@dataclass
class DROIDSLAMResult(ExternalVIOResult):
    """DROID-SLAM specific result."""
    num_edges: int = 0  # BA graph edges
    mean_depth: float = 0.0
    gpu_memory_mb: float = 0.0


class DROIDSLAMWrapper(BaseVIOWrapper):
    """
    Wrapper for DROID-SLAM deep visual SLAM.
    
    DROID-SLAM achieves excellent accuracy through:
    - Dense feature correlation volumes
    - Differentiable bundle adjustment
    - Global optimization over full trajectory
    
    Suitable for:
    - High-accuracy offline processing
    - Benchmark evaluation
    - When accuracy > speed
    
    Not ideal for:
    - Real-time applications (slow on consumer GPUs)
    - Low-memory systems
    
    Usage:
        wrapper = DROIDSLAMWrapper(
            weights_path="droid.pth",
            image_size=(480, 640),
            stereo=True
        )
        wrapper.initialize()
        
        # Process sequence
        for img, ts in images:
            result = wrapper.process(img, ts)
        
        # Get optimized trajectory
        trajectory = wrapper.get_trajectory()
    """
    
    def __init__(
        self,
        weights_path: str,
        image_size: tuple = (480, 640),
        stereo: bool = False,
        buffer_size: int = 512,
        config: Optional[Dict] = None,
    ):
        """
        Initialize DROID-SLAM wrapper.
        
        Args:
            weights_path: Path to pretrained weights
            image_size: Input image size (height, width)
            stereo: Use stereo mode
            buffer_size: Frame buffer size for BA
            config: Additional configuration
        """
        super().__init__(config)
        
        self._weights_path = weights_path
        self._image_size = image_size
        self._stereo = stereo
        self._buffer_size = buffer_size
        
        self._droid = None
        self._device = 'cuda' if _torch and _torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def is_available() -> bool:
        """Check if DROID-SLAM is available."""
        return _DROID_AVAILABLE
    
    def initialize(self) -> bool:
        """Initialize DROID-SLAM."""
        if not _DROID_AVAILABLE:
            logger.error("DROID-SLAM not available")
            logger.info("To use DROID-SLAM:")
            logger.info("1. Ensure CUDA is available")
            logger.info("2. git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git")
            logger.info("3. cd DROID-SLAM && pip install -e .")
            logger.info("4. pip install lietorch")
            logger.info("5. Download weights: ./download_weights.sh")
            return False
        
        try:
            # Build args
            class Args:
                pass
            
            args = Args()
            args.weights = self._weights_path
            args.image_size = list(self._image_size)
            args.stereo = self._stereo
            args.buffer = self._buffer_size
            args.disable_vis = True
            
            # Get intrinsics
            fx = self._config.get('fx', 320.0)
            fy = self._config.get('fy', 320.0)
            cx = self._config.get('cx', self._image_size[1] / 2)
            cy = self._config.get('cy', self._image_size[0] / 2)
            
            args.calib = [fx, fy, cx, cy]
            
            # Initialize
            self._droid = _droid(args)
            
            self._initialized = True
            logger.info(f"DROID-SLAM initialized on {self._device}")
            logger.info(f"Mode: {'stereo' if self._stereo else 'monocular'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DROID-SLAM: {e}")
            return False
    
    def process(
        self,
        image: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        image_right: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> DROIDSLAMResult:
        """
        Process frame with DROID-SLAM.
        
        Note: DROID-SLAM does not use IMU, accel/gyro are ignored.
        
        Args:
            image: Left/mono RGB image
            timestamp: Frame timestamp
            depth: Optional depth (for RGB-D mode)
            image_right: Right image (for stereo mode)
            accel: Ignored
            gyro: Ignored
            
        Returns:
            DROIDSLAMResult with pose estimate
        """
        t_start = time.perf_counter()
        
        if not self._initialized:
            return self._create_failed_result(timestamp, t_start)
        
        self._frame_count += 1
        
        try:
            # Prepare image
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            
            # Resize if needed
            h, w = image.shape[:2]
            if (h, w) != self._image_size:
                import cv2
                image = cv2.resize(image, (self._image_size[1], self._image_size[0]))
                if image_right is not None:
                    image_right = cv2.resize(image_right, (self._image_size[1], self._image_size[0]))
            
            # Convert to tensor
            image_tensor = _torch.from_numpy(image).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0).to(self._device)
            
            if self._stereo and image_right is not None:
                if len(image_right.shape) == 2:
                    image_right = np.stack([image_right]*3, axis=-1)
                right_tensor = _torch.from_numpy(image_right).permute(2, 0, 1).float()
                right_tensor = right_tensor.unsqueeze(0).to(self._device)
                image_tensor = _torch.cat([image_tensor, right_tensor], dim=0)
            
            # Track frame
            self._droid.track(timestamp, image_tensor)
            
            # Get current pose
            poses, tstamps = self._droid.get_trajectory()
            
            if len(poses) > 0:
                # Get latest pose
                pose_tensor = poses[-1]
                
                # Convert to numpy 4x4 matrix
                if pose_tensor.shape == (7,):
                    # [tx, ty, tz, qx, qy, qz, qw]
                    t = pose_tensor[:3].cpu().numpy()
                    q = pose_tensor[3:].cpu().numpy()
                    R = self._quaternion_to_rotation(q)
                    self._pose[:3, :3] = R
                    self._pose[:3, 3] = t
                else:
                    self._pose = pose_tensor.cpu().numpy().reshape(4, 4)
                
                tracking_state = TrackingState.TRACKING
                confidence = 0.95
            else:
                tracking_state = TrackingState.INITIALIZING
                confidence = 0.3
            
            # Get memory usage
            gpu_mem = 0.0
            if _torch.cuda.is_available():
                gpu_mem = _torch.cuda.memory_allocated() / 1024 / 1024
            
            proc_time = (time.perf_counter() - t_start) * 1000
            
            return DROIDSLAMResult(
                pose=self._pose.copy(),
                velocity=None,
                timestamp=timestamp,
                tracking_state=tracking_state,
                confidence=confidence,
                num_features=0,
                processing_time_ms=proc_time,
                gpu_memory_mb=gpu_mem,
            )
            
        except Exception as e:
            logger.error(f"DROID-SLAM processing error: {e}")
            return self._create_failed_result(timestamp, t_start)
    
    def _create_failed_result(self, timestamp: float, t_start: float) -> DROIDSLAMResult:
        """Create a failed result."""
        proc_time = (time.perf_counter() - t_start) * 1000
        return DROIDSLAMResult(
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
        """Convert quaternion [qx, qy, qz, qw] to rotation matrix."""
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def run_global_ba(self):
        """Run global bundle adjustment over all frames."""
        if self._droid is not None:
            try:
                self._droid.terminate()
                logger.info("Global BA completed")
            except Exception as e:
                logger.error(f"Global BA failed: {e}")
    
    def get_trajectory(self) -> List[np.ndarray]:
        """
        Get optimized trajectory.
        
        Note: Call run_global_ba() first for best results.
        """
        if self._droid is None:
            return []
        
        try:
            poses, _ = self._droid.get_trajectory()
            
            trajectory = []
            for pose in poses:
                if pose.shape == (7,):
                    t = pose[:3].cpu().numpy()
                    q = pose[3:].cpu().numpy()
                    R = self._quaternion_to_rotation(q)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t
                    trajectory.append(T)
                else:
                    trajectory.append(pose.cpu().numpy().reshape(4, 4))
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Failed to get trajectory: {e}")
            return []
    
    def get_depth_maps(self) -> List[np.ndarray]:
        """Get estimated depth maps."""
        if self._droid is None:
            return []
        
        try:
            depths = self._droid.get_depths()
            return [d.cpu().numpy() for d in depths]
        except Exception:
            return []
    
    def shutdown(self):
        """Shutdown DROID-SLAM."""
        if self._droid is not None:
            try:
                self._droid.terminate()
            except Exception:
                pass
            del self._droid
            self._droid = None
        
        if _torch and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        
        self._initialized = False
