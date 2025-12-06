"""
DPVO Wrapper for AirSplatMap
============================

Wrapper for Deep Patch Visual Odometry (DPVO), a learning-based
visual odometry system that achieves near real-time performance.

DPVO Features:
- Deep learning-based feature extraction and matching
- Patch-based correlation volumes
- Recurrent pose updates
- GPU accelerated (CUDA required)

Requirements:
- PyTorch with CUDA
- DPVO package: pip install dpvo (or from source)

Installation:
    git clone https://github.com/princeton-vl/DPVO.git
    cd DPVO
    pip install -e .
    
    # Download pretrained weights
    wget https://www.dropbox.com/s/... -O dpvo.pth

Paper: "Deep Patch Visual Odometry" (NeurIPS 2022)
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import time
import logging

from .base_wrapper import BaseVIOWrapper, ExternalVIOResult, TrackingState

logger = logging.getLogger(__name__)


# Check if DPVO is available
_DPVO_AVAILABLE = False
_dpvo = None
_torch = None

try:
    import torch as _torch
    if _torch.cuda.is_available():
        from dpvo.dpvo import DPVO as _DPVO
        _dpvo = _DPVO
        _DPVO_AVAILABLE = True
    else:
        logger.warning("DPVO requires CUDA but no GPU available")
except ImportError:
    pass


@dataclass
class DPVOResult(ExternalVIOResult):
    """DPVO specific result."""
    num_patches: int = 0
    gpu_memory_mb: float = 0.0


class DPVOWrapper(BaseVIOWrapper):
    """
    Wrapper for Deep Patch Visual Odometry (DPVO).
    
    DPVO is a learning-based VO system that uses:
    - CNN feature extraction
    - Patch-based correlation volumes
    - GRU-based iterative pose refinement
    
    Advantages:
    - High accuracy comparable to classical methods
    - Near real-time on modern GPUs (~15-20 FPS)
    - Robust to challenging conditions
    
    Limitations:
    - Requires CUDA GPU
    - Monocular only (no IMU fusion)
    - May drift without loop closure
    
    Usage:
        wrapper = DPVOWrapper(
            weights_path="dpvo.pth",
            image_size=(480, 640)
        )
        wrapper.initialize()
        
        result = wrapper.process(image, timestamp)
    """
    
    def __init__(
        self,
        weights_path: str,
        image_size: tuple = (480, 640),
        config: Optional[Dict] = None,
    ):
        """
        Initialize DPVO wrapper.
        
        Args:
            weights_path: Path to pretrained weights (.pth file)
            image_size: Input image size (height, width)
            config: Additional configuration
        """
        super().__init__(config)
        
        self._weights_path = weights_path
        self._image_size = image_size
        
        self._dpvo = None
        self._trajectory: List[np.ndarray] = []
        self._device = 'cuda' if _torch and _torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def is_available() -> bool:
        """Check if DPVO is available."""
        return _DPVO_AVAILABLE
    
    def initialize(self) -> bool:
        """Initialize DPVO system."""
        if not _DPVO_AVAILABLE:
            logger.error("DPVO not available")
            logger.info("To use DPVO:")
            logger.info("1. Ensure CUDA is available")
            logger.info("2. Clone: git clone https://github.com/princeton-vl/DPVO.git")
            logger.info("3. Install: cd DPVO && pip install -e .")
            logger.info("4. Download pretrained weights")
            return False
        
        try:
            # Get intrinsics
            fx = self._config.get('fx', 320.0)
            fy = self._config.get('fy', 320.0)
            cx = self._config.get('cx', self._image_size[1] / 2)
            cy = self._config.get('cy', self._image_size[0] / 2)
            
            intrinsics = _torch.tensor([fx, fy, cx, cy], device=self._device)
            
            # Initialize DPVO
            self._dpvo = _dpvo(
                self._weights_path,
                intrinsics,
                self._image_size,
            )
            
            self._initialized = True
            logger.info(f"DPVO initialized on {self._device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DPVO: {e}")
            return False
    
    def process(
        self,
        image: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        image_right: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> DPVOResult:
        """
        Process frame with DPVO.
        
        Note: DPVO is monocular-only, depth/stereo/IMU are ignored.
        
        Args:
            image: RGB image
            timestamp: Frame timestamp
            depth: Ignored
            image_right: Ignored
            accel: Ignored
            gyro: Ignored
            
        Returns:
            DPVOResult with pose estimate
        """
        t_start = time.perf_counter()
        
        if not self._initialized:
            return self._create_failed_result(timestamp, t_start)
        
        self._frame_count += 1
        
        try:
            # Convert image to tensor
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image, image, image], axis=-1)
            
            # Resize if needed
            h, w = image.shape[:2]
            if (h, w) != self._image_size:
                import cv2
                image = cv2.resize(image, (self._image_size[1], self._image_size[0]))
            
            # Convert to tensor [1, 3, H, W]
            image_tensor = _torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.float().to(self._device) / 255.0
            
            # Process frame
            self._dpvo(timestamp, image_tensor)
            
            # Get pose
            poses = self._dpvo.get_poses()
            
            if len(poses) > 0:
                # Latest pose
                pose = poses[-1].cpu().numpy()
                
                # DPVO returns [tx, ty, tz, qx, qy, qz, qw]
                if pose.shape == (7,):
                    t = pose[:3]
                    q = pose[3:]
                    R = self._quaternion_to_rotation(q)
                    self._pose[:3, :3] = R
                    self._pose[:3, 3] = t
                elif pose.shape == (4, 4):
                    self._pose = pose
                
                self._trajectory.append(self._pose.copy())
                tracking_state = TrackingState.TRACKING
                confidence = 0.9
            else:
                tracking_state = TrackingState.INITIALIZING
                confidence = 0.3
            
            # Get GPU memory usage
            gpu_mem = 0.0
            if _torch.cuda.is_available():
                gpu_mem = _torch.cuda.memory_allocated() / 1024 / 1024
            
            proc_time = (time.perf_counter() - t_start) * 1000
            
            return DPVOResult(
                pose=self._pose.copy(),
                velocity=None,
                timestamp=timestamp,
                tracking_state=tracking_state,
                confidence=confidence,
                num_features=0,
                processing_time_ms=proc_time,
                num_patches=0,
                gpu_memory_mb=gpu_mem,
            )
            
        except Exception as e:
            logger.error(f"DPVO processing error: {e}")
            return self._create_failed_result(timestamp, t_start)
    
    def _create_failed_result(self, timestamp: float, t_start: float) -> DPVOResult:
        """Create a failed result."""
        proc_time = (time.perf_counter() - t_start) * 1000
        return DPVOResult(
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
    
    def shutdown(self):
        """Shutdown DPVO."""
        if self._dpvo is not None:
            del self._dpvo
            self._dpvo = None
        if _torch and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        self._initialized = False
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get trajectory."""
        return self._trajectory.copy()
