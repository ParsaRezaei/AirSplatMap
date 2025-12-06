"""
Stereo Depth Estimation
=======================

Depth estimation from stereo camera pairs using classical and 
semi-global matching algorithms.

Designed for:
- RealSense D435/D435i stereo IR cameras
- Any rectified stereo pair

Algorithms:
- SGBM: Semi-Global Block Matching (OpenCV) - fast, good quality
- BM: Simple Block Matching - fastest, lower quality
- RAFT-Stereo: Deep learning stereo (optional)

Usage:
    from src.depth.stereo import StereoDepthEstimator
    
    estimator = StereoDepthEstimator(
        baseline=0.05,  # 50mm for RealSense
        focal_length=382.6
    )
    result = estimator.estimate_stereo(left_ir, right_ir)
"""

import numpy as np
import logging
from typing import Optional, Tuple

from .base import BaseDepthEstimator, DepthResult

logger = logging.getLogger(__name__)

# Check OpenCV availability
_CV2_AVAILABLE = False
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass


class StereoDepthEstimator(BaseDepthEstimator):
    """
    Stereo depth estimation using Semi-Global Block Matching.
    
    Computes depth from stereo disparity: depth = (focal * baseline) / disparity
    
    Optimized for RealSense D435 IR camera pairs but works with
    any rectified stereo images.
    
    Usage:
        estimator = StereoDepthEstimator(baseline=0.05, focal_length=382.6)
        result = estimator.estimate_stereo(left_ir, right_ir)
        depth = result.depth  # Metric depth in meters
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        focal_length: float = 382.6,
        num_disparities: int = 96,
        block_size: int = 5,
        min_depth: float = 0.2,
        max_depth: float = 10.0,
        device: str = "cpu",  # Stereo matching is CPU-based
    ):
        """
        Initialize stereo depth estimator.
        
        Args:
            baseline: Stereo baseline in meters (default 0.05 for RealSense)
            focal_length: Focal length in pixels
            num_disparities: Max disparity (must be divisible by 16)
            block_size: Matching block size (odd number, 3-11)
            min_depth: Minimum valid depth in meters
            max_depth: Maximum valid depth in meters
            device: Not used (CPU only for OpenCV stereo)
        """
        super().__init__(device)
        
        self.baseline = baseline
        self.focal_length = focal_length
        self.num_disparities = (num_disparities // 16) * 16  # Must be divisible by 16
        self.block_size = block_size | 1  # Must be odd
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self._stereo_matcher = None
        self._wls_filter = None
    
    @staticmethod
    def is_available() -> bool:
        return _CV2_AVAILABLE
    
    def _lazy_init(self):
        if self._initialized:
            return
        
        if not _CV2_AVAILABLE:
            logger.error("OpenCV not available")
            self._initialized = True
            return
        
        try:
            # Create SGBM matcher
            self._stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * self.block_size ** 2,
                P2=32 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
            
            # Create right matcher for WLS filter
            self._right_matcher = cv2.ximgproc.createRightMatcher(self._stereo_matcher)
            
            # WLS filter for better edges
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._stereo_matcher)
            self._wls_filter.setLambda(8000)
            self._wls_filter.setSigmaColor(1.5)
            
            self._initialized = True
            logger.info("Stereo depth estimator initialized (SGBM + WLS)")
            
        except AttributeError:
            # ximgproc not available, use basic SGBM
            self._stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * self.block_size ** 2,
                P2=32 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
            )
            self._wls_filter = None
            self._initialized = True
            logger.info("Stereo depth estimator initialized (SGBM only)")
        except Exception as e:
            logger.error(f"Failed to initialize stereo matcher: {e}")
            self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """
        Single-image estimate - not supported for stereo.
        
        Use estimate_stereo() instead with left/right image pair.
        """
        logger.warning("Stereo estimator requires two images. Use estimate_stereo().")
        h, w = rgb.shape[:2]
        return DepthResult(
            depth=np.zeros((h, w), dtype=np.float32),
            is_metric=True,
        )
    
    def estimate_stereo(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> DepthResult:
        """
        Estimate depth from stereo image pair.
        
        Args:
            left: Left image (grayscale or BGR)
            right: Right image (grayscale or BGR)
            
        Returns:
            DepthResult with metric depth in meters
        """
        self._lazy_init()
        
        if self._stereo_matcher is None:
            h, w = left.shape[:2]
            return DepthResult(
                depth=np.zeros((h, w), dtype=np.float32),
                is_metric=True,
            )
        
        # Convert to grayscale if needed
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left
            right_gray = right
        
        h, w = left_gray.shape
        
        try:
            # Compute disparity
            if self._wls_filter is not None:
                # Use WLS-filtered disparity
                disparity_left = self._stereo_matcher.compute(left_gray, right_gray)
                disparity_right = self._right_matcher.compute(right_gray, left_gray)
                
                disparity = self._wls_filter.filter(
                    disparity_left, left_gray, None, disparity_right
                )
            else:
                disparity = self._stereo_matcher.compute(left_gray, right_gray)
            
            # Convert to float (SGBM outputs fixed-point with 4 fractional bits)
            disparity = disparity.astype(np.float32) / 16.0
            
            # Convert disparity to depth: Z = f * B / d
            depth = np.zeros_like(disparity)
            valid = disparity > 0
            depth[valid] = (self.focal_length * self.baseline) / disparity[valid]
            
            # Apply depth range limits
            depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0
            
            # Compute confidence from disparity validity
            confidence = (disparity > 0).astype(np.float32)
            
            return DepthResult(
                depth=depth.astype(np.float32),
                confidence=confidence,
                is_metric=True,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
            
        except Exception as e:
            logger.error(f"Stereo matching failed: {e}")
            return DepthResult(
                depth=np.zeros((h, w), dtype=np.float32),
                is_metric=True,
            )
    
    def set_camera_params(
        self,
        baseline: Optional[float] = None,
        focal_length: Optional[float] = None,
    ):
        """Update camera parameters."""
        if baseline is not None:
            self.baseline = baseline
        if focal_length is not None:
            self.focal_length = focal_length
    
    def get_name(self) -> str:
        return "stereo_sgbm"
    
    def is_metric(self) -> bool:
        return True


class FastStereoEstimator(BaseDepthEstimator):
    """
    Fast stereo depth using simple Block Matching.
    
    Faster than SGBM but lower quality. Good for real-time
    applications where speed is critical.
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        focal_length: float = 382.6,
        num_disparities: int = 64,
        block_size: int = 15,
        device: str = "cpu",
    ):
        super().__init__(device)
        self.baseline = baseline
        self.focal_length = focal_length
        self.num_disparities = (num_disparities // 16) * 16
        self.block_size = block_size | 1
        self._matcher = None
    
    @staticmethod
    def is_available() -> bool:
        return _CV2_AVAILABLE
    
    def _lazy_init(self):
        if self._initialized:
            return
        
        if not _CV2_AVAILABLE:
            self._initialized = True
            return
        
        self._matcher = cv2.StereoBM_create(
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
        )
        self._initialized = True
        logger.info("Fast stereo (BM) initialized")
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        logger.warning("Use estimate_stereo() for stereo depth")
        h, w = rgb.shape[:2]
        return DepthResult(depth=np.zeros((h, w), dtype=np.float32), is_metric=True)
    
    def estimate_stereo(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        self._lazy_init()
        
        if self._matcher is None:
            h, w = left.shape[:2]
            return DepthResult(depth=np.zeros((h, w), dtype=np.float32), is_metric=True)
        
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        disparity = self._matcher.compute(left, right).astype(np.float32) / 16.0
        
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = (self.focal_length * self.baseline) / disparity[valid]
        depth[(depth < 0.1) | (depth > 10.0)] = 0
        
        return DepthResult(depth=depth.astype(np.float32), is_metric=True)
    
    def get_name(self) -> str:
        return "stereo_bm"
    
    def is_metric(self) -> bool:
        return True
