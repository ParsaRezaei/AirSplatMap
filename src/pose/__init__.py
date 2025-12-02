"""
Pose Estimation Module
======================

Provides modular pose estimation backends for AirSplatMap.

Available estimators:
- orb: Fast ORB-based visual odometry
- sift: SIFT-based with fundamental matrix
- robust_flow: Optical flow with robust estimation
- loftr: LoFTR learned feature matching (requires kornia)

Usage:
    from src.pose import get_pose_estimator
    
    estimator = get_pose_estimator('robust_flow')
    estimator.set_intrinsics_from_dict({'fx': 525, 'fy': 525, 'cx': 320, 'cy': 240})
    result = estimator.estimate(rgb_image)
    pose = result.pose  # 4x4 camera-to-world matrix
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass 
class PoseResult:
    """Result from pose estimation."""
    pose: np.ndarray  # 4x4 camera-to-world transformation
    confidence: float = 1.0  # Tracking confidence [0, 1]
    num_inliers: int = 0  # Number of inlier matches
    is_keyframe: bool = False
    tracking_status: str = "ok"  # "ok", "lost", "initializing"


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators."""
    
    def __init__(self):
        self._K = None  # 3x3 intrinsic matrix
        self._current_pose = np.eye(4)  # Current camera-to-world pose
        self._initialized = False
    
    def set_intrinsics(self, K: np.ndarray):
        """Set camera intrinsic matrix (3x3)."""
        self._K = K.astype(np.float64)
    
    def set_intrinsics_from_dict(self, intrinsics: Dict[str, float]):
        """Set intrinsics from dictionary with fx, fy, cx, cy."""
        self._K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        """
        Estimate pose from RGB image.
        
        Args:
            rgb: HxWx3 RGB image (uint8)
            
        Returns:
            PoseResult with estimated pose
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get estimator name."""
        pass
    
    def reset(self):
        """Reset estimator state."""
        self._current_pose = np.eye(4)
        self._initialized = False


class ORBPoseEstimator(BasePoseEstimator):
    """Fast pose estimation using ORB features."""
    
    def __init__(self, n_features: int = 2000, scale_factor: float = 1.2, n_levels: int = 8):
        super().__init__()
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect ORB features
        kp, desc = self._orb.detectAndCompute(gray, None)
        
        # First frame - just store features
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(kp) if kp else 0,
                tracking_status="initializing",
            )
        
        # Match features
        matches = self._bf.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:min(100, len(matches))]
        
        if len(good_matches) < 8:
            # Not enough matches - tracking lost
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_matches),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Extract matched points
        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        
        if E is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Recover pose
        num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
        
        # Build relative transformation (previous to current)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Update cumulative pose: T_world_curr = T_world_prev @ T_prev_curr
        # But we have T_curr_prev from recoverPose, so need inverse
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        # Compute confidence based on inliers
        confidence = min(1.0, num_inliers / 50.0)
        
        # Store for next frame
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "orb"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None


class RobustFlowEstimator(BasePoseEstimator):
    """
    Robust pose estimation using optical flow and fundamental matrix.
    
    Combines dense optical flow with feature points for robust tracking.
    """
    
    def __init__(self, grid_size: int = 20, use_farneback: bool = True):
        super().__init__()
        self._grid_size = grid_size
        self._use_farneback = use_farneback
        
        self._prev_gray = None
        self._prev_points = None
        
        # Lucas-Kanade parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    
    def _create_grid_points(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create grid of points for tracking."""
        h, w = shape
        step = self._grid_size
        
        x = np.arange(step, w - step, step)
        y = np.arange(step, h - step, step)
        xx, yy = np.meshgrid(x, y)
        
        points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
        return points.reshape(-1, 1, 2)
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # First frame
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(self._prev_points),
                tracking_status="initializing",
            )
        
        # Track points with Lucas-Kanade optical flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_points, None,
            **self._lk_params
        )
        
        if next_points is None:
            # Flow failed - reinitialize
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Filter by status
        status = status.flatten()
        good_prev = self._prev_points[status == 1].reshape(-1, 2)
        good_next = next_points[status == 1].reshape(-1, 2)
        
        if len(good_prev) < 8:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_prev),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Filter by bounds
        valid = (
            (good_next[:, 0] >= 0) & (good_next[:, 0] < w) &
            (good_next[:, 1] >= 0) & (good_next[:, 1] < h)
        )
        good_prev = good_prev[valid]
        good_next = good_next[valid]
        
        if len(good_prev) < 8:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_prev),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            good_prev, good_next, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.5,
        )
        
        if E is None or mask is None:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Recover pose
        num_inliers, R, t, _ = cv2.recoverPose(E, good_prev, good_next, self._K, mask=mask)
        
        # Build relative transformation
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Update cumulative pose
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        # Compute confidence
        confidence = min(1.0, num_inliers / 100.0)
        
        # Prepare for next frame
        self._prev_gray = gray
        
        # Reinitialize points periodically or when too few tracked
        if len(good_next) < 100 or num_inliers < 30:
            self._prev_points = self._create_grid_points(gray.shape)
        else:
            # Keep tracked points
            self._prev_points = good_next.reshape(-1, 1, 2)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "robust_flow"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_points = None


class SIFTPoseEstimator(BasePoseEstimator):
    """Pose estimation using SIFT features (more robust but slower)."""
    
    def __init__(self, n_features: int = 1000):
        super().__init__()
        self._sift = cv2.SIFT_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, None)
        
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(kp) if kp else 0,
                tracking_status="initializing",
            )
        
        # KNN matching with ratio test
        matches = self._bf.knnMatch(self._prev_desc, desc, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 8:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_matches),
                tracking_status="lost",
                confidence=0.0,
            )
        
        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        
        if E is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
        
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        confidence = min(1.0, num_inliers / 50.0)
        
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "sift"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None


# Registry of available estimators
_ESTIMATORS: Dict[str, type] = {
    'orb': ORBPoseEstimator,
    'robust_flow': RobustFlowEstimator,
    'flow': RobustFlowEstimator,
    'sift': SIFTPoseEstimator,
}


def get_pose_estimator(name: str, **kwargs) -> BasePoseEstimator:
    """
    Get a pose estimator by name.
    
    Args:
        name: Estimator name ('orb', 'robust_flow', 'sift')
        **kwargs: Additional arguments passed to constructor
        
    Returns:
        BasePoseEstimator instance
        
    Raises:
        ValueError: If estimator name is unknown
    """
    name_lower = name.lower().replace('-', '_')
    
    if name_lower not in _ESTIMATORS:
        available = list(_ESTIMATORS.keys())
        raise ValueError(f"Unknown pose estimator: {name}. Available: {available}")
    
    return _ESTIMATORS[name_lower](**kwargs)


def list_pose_estimators() -> Dict[str, Dict[str, Any]]:
    """List available pose estimators."""
    return {
        'orb': {
            'description': 'Fast ORB-based visual odometry',
            'speed': 'fast',
        },
        'robust_flow': {
            'description': 'Optical flow with robust estimation',
            'speed': 'medium',
        },
        'sift': {
            'description': 'SIFT-based feature matching',
            'speed': 'slow',
        },
    }


__all__ = [
    'BasePoseEstimator',
    'PoseResult',
    'ORBPoseEstimator',
    'RobustFlowEstimator',
    'SIFTPoseEstimator',
    'get_pose_estimator',
    'list_pose_estimators',
]
