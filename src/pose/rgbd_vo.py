"""
RGB-D Visual Odometry
=====================

Real-time monocular VO using RGB image + depth sensor.

Uses depth directly for 3D point triangulation, providing metric scale
without stereo matching. Suitable for RGB-D cameras like:
- RealSense D435/D435i (using RGB + depth)
- Azure Kinect
- Structure sensors

Features:
- Direct depth lookup (no stereo matching overhead)
- Multiple feature detectors (FAST, GFTT, ORB)
- Pose filtering with outlier rejection
- Stationary detection

Coordinate Conventions:
- Internal: OpenCV convention (X-right, Y-down, Z-forward)
- Depth: Assumed in meters

Usage:
    from src.pose.rgbd_vo import RGBDVO
    
    vo = RGBDVO(max_features=300)
    vo.set_intrinsics(fx=615, fy=615, cx=320, cy=240)
    
    result = vo.process(rgb, depth, timestamp)
    pose = result.pose  # 4x4 camera-to-world matrix
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time


class FeatureType(Enum):
    """Available feature detectors."""
    FAST = "fast"
    GFTT = "gftt"
    ORB = "orb"


@dataclass
class VOResult:
    """Result from VO processing."""
    pose: np.ndarray  # 4x4 camera-to-world
    velocity: np.ndarray  # 3D velocity in m/s
    tracking_status: str  # "ok", "lost", "initializing"
    num_inliers: int
    confidence: float  # 0.0 to 1.0
    processing_time_ms: float


class PoseFilter:
    """Pose smoothing and outlier rejection."""
    
    def __init__(
        self,
        smoothing: float = 0.5,
        max_translation: float = 0.3,
        max_rotation: float = 0.3,
    ):
        self._smoothing = smoothing
        self._max_trans = max_translation
        self._max_rot = max_rotation
        self._prev_pose: Optional[np.ndarray] = None
        self._smoothed_pose: Optional[np.ndarray] = None
    
    def filter(self, pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self._smoothed_pose is None:
            self._prev_pose = pose.copy()
            self._smoothed_pose = pose.copy()
            return pose.copy(), True
        
        # Check translation
        trans = np.linalg.norm(pose[:3, 3] - self._prev_pose[:3, 3])
        if trans > self._max_trans:
            return self._smoothed_pose.copy(), False
        
        # Check rotation
        R_delta = pose[:3, :3] @ self._prev_pose[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))
        if angle > self._max_rot:
            return self._smoothed_pose.copy(), False
        
        # Smooth
        alpha = self._smoothing
        t_smooth = alpha * self._smoothed_pose[:3, 3] + (1 - alpha) * pose[:3, 3]
        R_blend = alpha * self._smoothed_pose[:3, :3] + (1 - alpha) * pose[:3, :3]
        U, _, Vt = np.linalg.svd(R_blend)
        R_smooth = U @ Vt
        if np.linalg.det(R_smooth) < 0:
            R_smooth = -R_smooth
        
        smoothed = np.eye(4)
        smoothed[:3, :3] = R_smooth
        smoothed[:3, 3] = t_smooth
        
        self._prev_pose = pose.copy()
        self._smoothed_pose = smoothed.copy()
        
        return smoothed, True
    
    def reset(self):
        self._prev_pose = None
        self._smoothed_pose = None


class RGBDVO:
    """
    Real-time RGB-D Visual Odometry.
    
    Uses RGB image for feature tracking and depth for 3D reconstruction.
    Provides metric-scale pose estimation.
    """
    
    def __init__(
        self,
        max_features: int = 300,
        min_features: int = 50,
        feature_type: FeatureType = FeatureType.GFTT,
        min_depth: float = 0.1,
        max_depth: float = 8.0,
        pose_smoothing: float = 0.4,
        motion_threshold: float = 0.8,
    ):
        """
        Initialize RGB-D VO.
        
        Args:
            max_features: Maximum features to track
            min_features: Minimum features before re-detection
            feature_type: Feature detector type
            min_depth: Minimum valid depth in meters
            max_depth: Maximum valid depth in meters
            pose_smoothing: EMA smoothing factor
            motion_threshold: Pixel motion threshold for stationary detection
        """
        self.max_features = max_features
        self.min_features = min_features
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.motion_threshold = motion_threshold
        
        # Intrinsics
        self._K: Optional[np.ndarray] = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # State
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Previous frame data
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._prev_points_3d: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        
        # Feature detector
        self._detector = self._create_detector(feature_type, max_features)
        
        # Optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Pose filtering
        self._pose_filter = PoseFilter(smoothing=pose_smoothing)
        
        # Motion history
        self._motion_history: deque = deque(maxlen=10)
        
        self._frame_count = 0
    
    @staticmethod
    def _create_detector(feature_type: FeatureType, max_features: int):
        if feature_type == FeatureType.FAST:
            return cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        elif feature_type == FeatureType.GFTT:
            return cv2.GFTTDetector_create(
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7,
            )
        else:
            return cv2.ORB_create(nfeatures=max_features)
    
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsics."""
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    
    def set_intrinsics_from_dict(self, intrinsics: Dict):
        """Set intrinsics from dictionary."""
        self.set_intrinsics(
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy'],
        )
    
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect features in image."""
        kps = self._detector.detect(gray, None)
        if len(kps) == 0:
            return np.array([]).reshape(0, 2)
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:self.max_features]
        return np.array([kp.pt for kp in kps], dtype=np.float32)
    
    def _unproject_points(
        self,
        points_2d: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unproject 2D points to 3D using depth image.
        
        Returns (points_3d, valid_mask).
        """
        if len(points_2d) == 0:
            return np.array([]).reshape(0, 3), np.array([], dtype=bool)
        
        h, w = depth.shape
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        
        # Clamp to image bounds
        u_int = np.clip(u.astype(int), 0, w - 1)
        v_int = np.clip(v.astype(int), 0, h - 1)
        
        # Sample depth
        d = depth[v_int, u_int]
        
        # Valid depth mask
        valid = (d > self.min_depth) & (d < self.max_depth)
        
        # Unproject
        X = (u - self._cx) * d / self._fx
        Y = (v - self._cy) * d / self._fy
        Z = d
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return points_3d, valid
    
    def _estimate_motion_pnp(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Estimate motion using PnP RANSAC."""
        if len(points_3d) < 6:
            return None, 0
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            self._K,
            None,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success or inliers is None or len(inliers) < 4:
            return None, 0
        
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T, len(inliers)
    
    def _is_stationary(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> bool:
        """Check if camera is stationary."""
        if len(prev_pts) == 0 or len(curr_pts) == 0:
            return False
        
        motion = np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1))
        self._motion_history.append(motion)
        
        if len(self._motion_history) < 5:
            return False
        
        return np.mean(self._motion_history) < self.motion_threshold
    
    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> VOResult:
        """
        Process RGB-D frame.
        
        Args:
            rgb: RGB image (HxWx3 uint8 or HxW grayscale)
            depth: Depth image in meters (HxW float32)
            timestamp: Frame timestamp in seconds
            accel: Optional accelerometer (not used, for API compatibility)
            gyro: Optional gyroscope (not used, for API compatibility)
            
        Returns:
            VOResult with pose and tracking status
        """
        t_start = time.perf_counter()
        
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        self._frame_count += 1
        
        # Convert to grayscale
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        # Compute dt
        dt = 0.033
        if self._prev_time is not None:
            dt = max(0.001, min(0.5, timestamp - self._prev_time))
        
        # Initialize
        if not self._initialized:
            points_2d = self._detect_features(gray)
            if len(points_2d) >= self.min_features:
                points_3d, valid = self._unproject_points(points_2d, depth)
                points_2d = points_2d[valid]
                points_3d = points_3d[valid]
                
                if len(points_2d) >= self.min_features:
                    self._prev_gray = gray
                    self._prev_depth = depth
                    self._prev_points = points_2d
                    self._prev_points_3d = points_3d
                    self._prev_time = timestamp
                    self._initialized = True
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                tracking_status="initializing",
                num_inliers=len(points_2d) if self._initialized else 0,
                confidence=0.5 if self._initialized else 0.0,
                processing_time_ms=proc_time,
            )
        
        # Re-detect if too few points
        if self._prev_points is None or len(self._prev_points) < self.min_features:
            points_2d = self._detect_features(self._prev_gray if self._prev_gray is not None else gray)
            points_3d, valid = self._unproject_points(
                points_2d, 
                self._prev_depth if self._prev_depth is not None else depth
            )
            self._prev_points = points_2d[valid]
            self._prev_points_3d = points_3d[valid]
        
        if self._prev_points is None or len(self._prev_points) < 6:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                tracking_status="lost",
                num_inliers=0,
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # Track with optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_points.reshape(-1, 1, 2),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                tracking_status="lost",
                num_inliers=0,
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # Filter tracked points
        status = status.flatten().astype(bool)
        prev_2d = self._prev_points[status]
        prev_3d = self._prev_points_3d[status]
        curr_2d = curr_points[status].reshape(-1, 2)
        
        # Bounds check
        h, w = gray.shape
        valid = (
            (curr_2d[:, 0] >= 0) & (curr_2d[:, 0] < w) &
            (curr_2d[:, 1] >= 0) & (curr_2d[:, 1] < h)
        )
        prev_2d = prev_2d[valid]
        prev_3d = prev_3d[valid]
        curr_2d = curr_2d[valid]
        
        # Check stationary
        if self._is_stationary(prev_2d, curr_2d):
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            
            new_pts = self._detect_features(gray)
            new_3d, valid = self._unproject_points(new_pts, depth)
            new_pts = new_pts[valid]
            new_3d = new_3d[valid]
            if len(new_pts) >= self.min_features:
                self._prev_points = new_pts
                self._prev_points_3d = new_3d
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                velocity=np.zeros(3),
                tracking_status="ok",
                num_inliers=len(self._prev_points) if self._prev_points is not None else 0,
                confidence=0.9,
                processing_time_ms=proc_time,
            )
        
        # Estimate motion
        T_curr_prev, num_inliers = self._estimate_motion_pnp(prev_3d, curr_2d)
        
        if T_curr_prev is None or num_inliers < 4:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                tracking_status="lost",
                num_inliers=num_inliers,
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # Update pose
        T_rel = np.linalg.inv(T_curr_prev)
        new_pose = self._pose @ T_rel
        
        # Filter
        filtered_pose, accepted = self._pose_filter.filter(new_pose)
        
        if accepted:
            self._pose = filtered_pose
            self._velocity = T_rel[:3, 3] / dt
        
        confidence = min(1.0, num_inliers / 50.0) * (0.9 if accepted else 0.5)
        
        # Update state
        self._prev_gray = gray
        self._prev_depth = depth
        self._prev_time = timestamp
        
        # Update tracked points with new depth
        curr_3d, valid = self._unproject_points(curr_2d, depth)
        self._prev_points = curr_2d[valid]
        self._prev_points_3d = curr_3d[valid]
        
        # Add new features if needed
        if len(self._prev_points) < self.max_features // 2:
            new_pts = self._detect_features(gray)
            new_3d, valid = self._unproject_points(new_pts, depth)
            new_pts = new_pts[valid]
            new_3d = new_3d[valid]
            if len(new_pts) > 0:
                self._prev_points = np.vstack([self._prev_points, new_pts])[:self.max_features]
                self._prev_points_3d = np.vstack([self._prev_points_3d, new_3d])[:self.max_features]
        
        proc_time = (time.perf_counter() - t_start) * 1000
        
        return VOResult(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            tracking_status="ok",
            num_inliers=num_inliers,
            confidence=confidence,
            processing_time_ms=proc_time,
        )
    
    def get_pose(self) -> np.ndarray:
        """Get current camera-to-world pose."""
        return self._pose.copy()
    
    def get_pose_opengl(self) -> np.ndarray:
        """Get pose in OpenGL convention."""
        CV_TO_GL = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        return CV_TO_GL @ self._pose @ CV_TO_GL.T
    
    def reset(self):
        """Reset VO state."""
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._prev_gray = None
        self._prev_depth = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_time = None
        self._motion_history.clear()
        self._frame_count = 0
        self._pose_filter.reset()


def create_rgbd_vo(
    intrinsics: Optional[Dict] = None,
    fast: bool = False,
) -> RGBDVO:
    """
    Create an RGBDVO instance with common settings.
    
    Args:
        intrinsics: Dictionary with fx, fy, cx, cy
        fast: If True, use FAST features; else GFTT (more stable)
        
    Returns:
        Configured RGBDVO instance
    """
    vo = RGBDVO(
        max_features=250 if fast else 400,
        min_features=40 if fast else 80,
        feature_type=FeatureType.FAST if fast else FeatureType.GFTT,
        pose_smoothing=0.3 if fast else 0.5,
    )
    
    if intrinsics:
        vo.set_intrinsics_from_dict(intrinsics)
    
    return vo
