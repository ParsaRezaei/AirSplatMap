"""
Base classes for depth estimation.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_best_device(preferred: str = "cuda") -> str:
    """
    Get the best available device, preferring GPU.
    
    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu')
        
    Returns:
        Device string that's actually available
    """
    try:
        import torch
        if preferred == "cuda" or preferred.startswith("cuda:"):
            if torch.cuda.is_available():
                return preferred if preferred.startswith("cuda:") else "cuda:0"
            logger.warning("CUDA requested but not available, falling back to CPU")
        elif preferred == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available, trying CUDA")
            if torch.cuda.is_available():
                return "cuda:0"
        # Try CUDA anyway before falling back to CPU
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


@dataclass
class DepthResult:
    """Result from depth estimation."""
    depth: np.ndarray  # HxW float32 depth map
    confidence: Optional[np.ndarray] = None  # HxW confidence map [0, 1]
    min_depth: float = 0.0
    max_depth: float = 10.0
    is_metric: bool = False  # True if depth is in meters, False if relative


class BaseDepthEstimator(ABC):
    """Abstract base class for depth estimators."""
    
    def __init__(self, device: str = "cuda"):
        # Always try to use GPU if available
        self.device = get_best_device(device)
        self._initialized = False
        if self.device != device:
            logger.info(f"Using device: {self.device} (requested: {device})")
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb: HxWx3 RGB image (uint8 or float32)
            
        Returns:
            DepthResult with depth map
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get estimator name."""
        pass
    
    def is_metric(self) -> bool:
        """Returns True if this estimator produces metric depth."""
        return False
    
    def reset(self):
        """Reset estimator state (if any)."""
        pass
    
    def cleanup(self):
        """
        Release GPU memory and cleanup resources.
        
        Call this when done using the estimator to free VRAM.
        """
        self._initialized = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
    
    def warmup(self, image_size: Tuple[int, int] = (480, 640)):
        """
        Warmup the model with a dummy image.
        
        Useful for ensuring consistent timing on first real inference.
        """
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        self.estimate(dummy)
    
    @staticmethod
    def is_available() -> bool:
        """Check if this estimator's dependencies are available."""
        return True


class PassthroughDepthEstimator(BaseDepthEstimator):
    """
    No-op depth estimator that returns None.
    
    Use when depth comes from another source (e.g., RGB-D sensor ground truth).
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        h, w = rgb.shape[:2]
        return DepthResult(
            depth=np.zeros((h, w), dtype=np.float32),
            is_metric=True,
        )
    
    def get_name(self) -> str:
        return "passthrough"
    
    def is_metric(self) -> bool:
        return True


class DepthScaler:
    """
    Scales relative depth to metric depth using sparse reference points.
    
    Useful for converting monocular depth estimates to metric scale
    using sparse depth from stereo, LiDAR, or SfM points.
    
    Methods:
    - scale_shift: Solves for scale and shift (depth_metric = scale * depth_rel + shift)
    - scale_only: Solves for scale only (depth_metric = scale * depth_rel)
    - ransac: Robust fitting with outlier rejection
    """
    
    def __init__(self, method: str = "scale_shift"):
        """
        Initialize depth scaler.
        
        Args:
            method: 'scale_shift', 'scale_only', or 'ransac'
        """
        self.method = method
        self._scale = 1.0
        self._shift = 0.0
    
    def fit(
        self,
        relative_depth: np.ndarray,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Fit scale and shift parameters from sparse correspondences.
        
        Args:
            relative_depth: HxW relative depth map
            points_3d: Nx3 3D points with known depth (Z is depth)
            points_2d: Nx2 corresponding 2D pixel coordinates
            mask: Optional HxW mask of valid depth pixels
            
        Returns:
            (scale, shift) parameters
        """
        if len(points_3d) < 3:
            return 1.0, 0.0
        
        h, w = relative_depth.shape
        
        # Sample relative depth at 2D points
        u = np.clip(points_2d[:, 0].astype(int), 0, w - 1)
        v = np.clip(points_2d[:, 1].astype(int), 0, h - 1)
        
        rel_samples = relative_depth[v, u]
        metric_samples = points_3d[:, 2]  # Z coordinate is depth
        
        # Filter invalid samples
        valid = (rel_samples > 0) & (metric_samples > 0) & np.isfinite(rel_samples) & np.isfinite(metric_samples)
        if mask is not None:
            valid &= mask[v, u] > 0
        
        rel_samples = rel_samples[valid]
        metric_samples = metric_samples[valid]
        
        if len(rel_samples) < 3:
            return 1.0, 0.0
        
        if self.method == "scale_only":
            # Least squares: metric = scale * relative
            scale = np.sum(rel_samples * metric_samples) / (np.sum(rel_samples ** 2) + 1e-8)
            self._scale = scale
            self._shift = 0.0
            
        elif self.method == "ransac":
            # RANSAC for robust fitting
            best_scale, best_shift = 1.0, 0.0
            best_inliers = 0
            
            for _ in range(100):
                # Random sample
                idx = np.random.choice(len(rel_samples), min(3, len(rel_samples)), replace=False)
                
                # Fit scale+shift on sample
                A = np.column_stack([rel_samples[idx], np.ones(len(idx))])
                b = metric_samples[idx]
                try:
                    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    scale, shift = params
                except:
                    continue
                
                # Count inliers
                predicted = scale * rel_samples + shift
                errors = np.abs(predicted - metric_samples)
                inliers = np.sum(errors < 0.1 * metric_samples)  # 10% error threshold
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_scale, best_shift = scale, shift
            
            self._scale = best_scale
            self._shift = best_shift
            
        else:  # scale_shift (default)
            # Least squares: metric = scale * relative + shift
            A = np.column_stack([rel_samples, np.ones(len(rel_samples))])
            b = metric_samples
            try:
                params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                self._scale, self._shift = params
            except:
                self._scale, self._shift = 1.0, 0.0
        
        return self._scale, self._shift
    
    def scale_to_metric(
        self,
        relative_depth: np.ndarray,
        points_3d: Optional[np.ndarray] = None,
        points_2d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert relative depth to metric depth.
        
        If points are provided, will fit new scale/shift parameters.
        Otherwise uses previously fitted parameters.
        
        Args:
            relative_depth: HxW relative depth map
            points_3d: Optional Nx3 3D reference points
            points_2d: Optional Nx2 2D pixel coordinates
            
        Returns:
            HxW metric depth map
        """
        if points_3d is not None and points_2d is not None:
            self.fit(relative_depth, points_3d, points_2d)
        
        metric_depth = self._scale * relative_depth + self._shift
        return np.maximum(metric_depth, 0).astype(np.float32)
    
    def get_parameters(self) -> Tuple[float, float]:
        """Get current scale and shift parameters."""
        return self._scale, self._shift


class DepthFilter:
    """
    Filters and post-processes depth maps.
    
    Provides:
    - Temporal filtering (smoothing over frames)
    - Bilateral filtering (edge-preserving smoothing)
    - Hole filling
    - Outlier removal
    """
    
    def __init__(
        self,
        temporal_alpha: float = 0.3,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
    ):
        self._temporal_alpha = temporal_alpha
        self._bilateral_d = bilateral_d
        self._bilateral_sigma_color = bilateral_sigma_color
        self._bilateral_sigma_space = bilateral_sigma_space
        self._prev_depth: Optional[np.ndarray] = None
    
    def temporal_filter(self, depth: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing using exponential moving average."""
        if self._prev_depth is None or self._prev_depth.shape != depth.shape:
            self._prev_depth = depth.copy()
            return depth
        
        # EMA: new = alpha * current + (1 - alpha) * previous
        filtered = self._temporal_alpha * depth + (1 - self._temporal_alpha) * self._prev_depth
        self._prev_depth = filtered.copy()
        return filtered
    
    def bilateral_filter(self, depth: np.ndarray) -> np.ndarray:
        """Apply edge-preserving bilateral filter."""
        try:
            import cv2
            # Normalize depth to 0-255 for cv2
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
            depth_uint8 = depth_norm.astype(np.uint8)
            
            filtered = cv2.bilateralFilter(
                depth_uint8,
                self._bilateral_d,
                self._bilateral_sigma_color,
                self._bilateral_sigma_space,
            )
            
            # Convert back to original range
            filtered = filtered.astype(np.float32) / 255 * (depth.max() - depth.min()) + depth.min()
            return filtered
        except ImportError:
            return depth
    
    def fill_holes(self, depth: np.ndarray, max_hole_size: int = 10) -> np.ndarray:
        """Fill small holes in depth map using inpainting."""
        try:
            import cv2
            
            # Create mask of holes (zero or invalid depth)
            mask = ((depth <= 0) | ~np.isfinite(depth)).astype(np.uint8)
            
            if mask.sum() == 0:
                return depth
            
            # Normalize for inpainting
            valid_mask = ~mask.astype(bool)
            if valid_mask.sum() == 0:
                return depth
            
            depth_norm = depth.copy()
            depth_min, depth_max = depth[valid_mask].min(), depth[valid_mask].max()
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8) * 255
            depth_uint8 = np.clip(depth_norm, 0, 255).astype(np.uint8)
            
            # Inpaint
            filled = cv2.inpaint(depth_uint8, mask, max_hole_size, cv2.INPAINT_TELEA)
            
            # Convert back
            filled = filled.astype(np.float32) / 255 * (depth_max - depth_min) + depth_min
            return filled
        except ImportError:
            return depth
    
    def remove_outliers(
        self,
        depth: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
    ) -> np.ndarray:
        """Remove depth outliers outside valid range."""
        filtered = depth.copy()
        filtered[(depth < min_depth) | (depth > max_depth) | ~np.isfinite(depth)] = 0
        return filtered
    
    def reset(self):
        """Reset temporal state."""
        self._prev_depth = None
