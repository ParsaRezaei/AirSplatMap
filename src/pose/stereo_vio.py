"""
Stereo Visual-Inertial Odometry for RealSense D435i
====================================================

Real-time stereo VO/VIO optimized for 3D Gaussian Splatting pipelines.

Features:
- Stereo triangulation from IR cameras with proper scale
- Optional IMU-based gravity alignment
- Pose filtering with outlier rejection
- Stationary detection to prevent drift
- Multiple feature detection backends (FAST, GFTT, ORB)

Coordinate Conventions:
- Internal: OpenCV convention (X-right, Y-down, Z-forward)
- Output: Configurable (OpenCV, OpenGL, or gravity-aligned)

Usage:
    from src.pose.stereo_vio import StereoVIO
    
    vio = StereoVIO(baseline=0.05, max_features=300)
    vio.set_intrinsics(fx=382.6, fy=382.6, cx=320, cy=240)
    
    result = vio.process(left_ir, right_ir, timestamp, accel=accel, gyro=gyro)
    pose = result.pose  # 4x4 camera-to-world matrix
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time


class FeatureType(Enum):
    """Available feature detectors."""
    FAST = "fast"
    GFTT = "gftt"  # Good Features To Track
    ORB = "orb"


@dataclass
class VIOResult:
    """Result from VIO processing."""
    pose: np.ndarray  # 4x4 camera-to-world
    velocity: np.ndarray  # 3D velocity in m/s
    tracking_status: str  # "ok", "lost", "initializing", "calibrating"
    num_inliers: int
    confidence: float  # 0.0 to 1.0
    processing_time_ms: float


class GravityEstimator:
    """
    Estimates gravity direction from accelerometer for world frame alignment.
    
    When calibrated, provides rotation to align camera Y-axis with gravity.
    Also detects motion vs stationary states from accelerometer variance.
    """
    
    def __init__(self, num_samples: int = 30, alpha: float = 0.98):
        self._num_samples = num_samples
        self._alpha = alpha  # For online filtering after calibration
        self._samples: List[np.ndarray] = []
        self._gravity = np.array([0.0, 9.81, 0.0])  # Default: Y-down (OpenCV)
        self._calibrated = False
        
        # For motion detection from accelerometer
        self._accel_history: deque = deque(maxlen=10)
        self._accel_bias = np.zeros(3)  # Accelerometer bias estimate
    
    def add_sample(self, accel: np.ndarray) -> bool:
        """Add accelerometer sample. Returns True when calibrated."""
        self._accel_history.append(accel.copy())
        
        if self._calibrated:
            # Online update with low-pass filter
            accel_norm = np.linalg.norm(accel)
            if 8.0 < accel_norm < 12.0:  # Only update when ~1g (stationary)
                self._gravity = self._alpha * self._gravity + (1 - self._alpha) * accel
            return True
        
        self._samples.append(accel.copy())
        
        if len(self._samples) >= self._num_samples:
            self._gravity = np.mean(self._samples, axis=0)
            self._accel_bias = self._gravity - np.array([0, 9.81, 0])  # Estimate bias
            self._calibrated = True
            return True
        return False
    
    def get_gravity(self) -> np.ndarray:
        """Get current gravity vector estimate."""
        return self._gravity.copy()
    
    def get_gravity_magnitude(self) -> float:
        """Get gravity magnitude (should be ~9.81 m/s²)."""
        return np.linalg.norm(self._gravity)
    
    def get_gravity_rotation(self) -> np.ndarray:
        """
        Get rotation matrix that aligns gravity with +Y axis (down in OpenCV).
        
        Returns 3x3 rotation R such that R @ gravity_unit = [0, 1, 0]
        """
        g = self._gravity
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-6:
            return np.eye(3)
        
        g_unit = g / g_norm
        target = np.array([0.0, 1.0, 0.0])  # +Y is down in OpenCV
        
        # Rotation from g to target using Rodrigues formula
        v = np.cross(g_unit, target)
        c = np.dot(g_unit, target)
        
        if np.linalg.norm(v) < 1e-6:
            return np.eye(3) if c > 0 else np.diag([1, -1, -1])
        
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        return np.eye(3) + vx + vx @ vx * (1 - c) / (np.linalg.norm(v) ** 2)
    
    def is_stationary_from_accel(self, threshold: float = 0.5) -> bool:
        """
        Detect if device is stationary based on accelerometer variance.
        
        When stationary, accel should be constant (just gravity).
        When moving, accel will have higher variance.
        """
        if len(self._accel_history) < 5:
            return False
        
        accel_array = np.array(self._accel_history)
        variance = np.var(accel_array, axis=0).sum()
        return variance < threshold
    
    def get_linear_acceleration(self, accel: np.ndarray) -> np.ndarray:
        """
        Get linear acceleration by removing gravity component.
        
        Useful for motion detection and velocity estimation.
        """
        if not self._calibrated:
            return np.zeros(3)
        return accel - self._gravity
    
    def is_calibrated(self) -> bool:
        return self._calibrated
    
    def reset(self):
        self._samples.clear()
        self._accel_history.clear()
        self._calibrated = False
        self._gravity = np.array([0.0, 9.81, 0.0])
        self._accel_bias = np.zeros(3)


class PoseFilter:
    """
    Filters pose estimates to reduce noise and reject outliers.
    
    Uses exponential moving average with outlier rejection based on
    maximum allowed translation and rotation per frame.
    """
    
    def __init__(
        self,
        smoothing: float = 0.5,
        max_translation: float = 0.3,  # meters per frame
        max_rotation: float = 0.3,  # radians per frame (~17 degrees)
    ):
        self._smoothing = smoothing
        self._max_trans = max_translation
        self._max_rot = max_rotation
        self._prev_pose: Optional[np.ndarray] = None
        self._smoothed_pose: Optional[np.ndarray] = None
    
    def filter(self, pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filter pose. Returns (filtered_pose, was_accepted).
        """
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
        
        # Apply smoothing
        alpha = self._smoothing
        
        # Smooth translation
        t_smooth = alpha * self._smoothed_pose[:3, 3] + (1 - alpha) * pose[:3, 3]
        
        # Smooth rotation (matrix averaging + orthogonalization)
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


class StereoVIO:
    """
    Real-time Stereo Visual-Inertial Odometry.
    
    Designed for RealSense D435i with dual IR cameras.
    Provides metric-scale pose estimation suitable for 3DGS mapping.
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        max_features: int = 300,
        min_features: int = 50,
        feature_type: FeatureType = FeatureType.FAST,
        use_imu: bool = True,
        gravity_samples: int = 30,
        pose_smoothing: float = 0.3,
        motion_threshold: float = 0.8,  # pixel motion for stationary detection
    ):
        """
        Initialize Stereo VIO.
        
        Args:
            baseline: Stereo baseline in meters (default 0.05m for RealSense)
            max_features: Maximum features to track
            min_features: Minimum features before re-detection
            feature_type: Feature detector type (FAST, GFTT, ORB)
            use_imu: Enable IMU-based gravity alignment
            gravity_samples: Samples needed for gravity calibration
            pose_smoothing: EMA smoothing factor (0=none, 1=max)
            motion_threshold: Pixel motion threshold for stationary detection
        """
        self.baseline = baseline
        self.max_features = max_features
        self.min_features = min_features
        self.use_imu = use_imu
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
        self._prev_points: Optional[np.ndarray] = None
        self._prev_points_3d: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        
        # Feature detector
        self._feature_type = feature_type
        self._detector = self._create_detector(feature_type, max_features)
        
        # Optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Gravity estimation (optional)
        self._gravity_estimator = GravityEstimator(num_samples=gravity_samples) if use_imu else None
        
        # Pose filtering
        self._pose_filter = PoseFilter(smoothing=pose_smoothing)
        
        # Motion history for stationary detection
        self._motion_history: deque = deque(maxlen=10)
        
        # Statistics
        self._frame_count = 0
    
    @staticmethod
    def _create_detector(feature_type: FeatureType, max_features: int):
        """Create feature detector based on type."""
        if feature_type == FeatureType.FAST:
            return cv2.FastFeatureDetector_create(
                threshold=20,
                nonmaxSuppression=True,
                type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
            )
        elif feature_type == FeatureType.GFTT:
            return cv2.GFTTDetector_create(
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7,
            )
        else:  # ORB
            return cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=1.2,
                nlevels=8,
            )
    
    def set_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: Optional[float] = None,
    ):
        """Set camera intrinsics."""
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        if baseline is not None:
            self.baseline = abs(baseline)
    
    def set_intrinsics_from_dict(self, intrinsics: Dict):
        """Set intrinsics from dictionary with fx, fy, cx, cy, baseline."""
        self.set_intrinsics(
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy'],
            baseline=intrinsics.get('baseline', self.baseline),
        )
    
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect features in image."""
        kps = self._detector.detect(gray, None)
        if len(kps) == 0:
            return np.array([]).reshape(0, 2)
        
        # Sort by response and take top N
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:self.max_features]
        return np.array([kp.pt for kp in kps], dtype=np.float32)
    
    def _triangulate_stereo(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        points_left: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match and triangulate 3D points from stereo pair.
        
        Uses optical flow to find correspondences (faster than descriptors).
        Returns (points_2d, points_3d) of valid triangulated points.
        """
        if len(points_left) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        # Track from left to right using optical flow
        points_right, status, _ = cv2.calcOpticalFlowPyrLK(
            left_gray, right_gray,
            points_left.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        
        if points_right is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        status = status.flatten().astype(bool)
        pts_l = points_left[status]
        pts_r = points_right[status].reshape(-1, 2)
        
        if len(pts_l) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        # Epipolar constraint: same row (rectified stereo)
        row_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
        valid = row_diff < 2.0
        
        # Disparity must be positive and reasonable
        disparity = pts_l[:, 0] - pts_r[:, 0]
        valid &= (disparity > 1.0) & (disparity < 200)
        
        if not np.any(valid):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        pts_l = pts_l[valid]
        disp = disparity[valid]
        
        # Triangulate: Z = f * B / d
        Z = (self._fx * self.baseline) / disp
        
        # Filter by depth range
        valid_depth = (Z > 0.2) & (Z < 8.0)
        if not np.any(valid_depth):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        pts_l = pts_l[valid_depth]
        Z = Z[valid_depth]
        
        # Back-project to 3D
        X = (pts_l[:, 0] - self._cx) * Z / self._fx
        Y = (pts_l[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return pts_l, points_3d
    
    def _estimate_motion_pnp(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate relative motion using PnP RANSAC.
        
        Returns (T_curr_prev, num_inliers) or (None, 0) on failure.
        """
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
    
    def _is_stationary(self, prev_pts: np.ndarray, curr_pts: np.ndarray, use_accel: bool = True) -> bool:
        """
        Check if camera is stationary based on feature motion and accelerometer.
        
        Combines visual motion detection with IMU-based detection for robustness.
        """
        if len(prev_pts) == 0 or len(curr_pts) == 0:
            return False
        
        # Visual motion check
        motion = np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1))
        self._motion_history.append(motion)
        
        if len(self._motion_history) < 5:
            return False
        
        visual_stationary = np.mean(self._motion_history) < self.motion_threshold
        
        # IMU-based stationary detection (if available and calibrated)
        if use_accel and self.use_imu and self._gravity_estimator and self._gravity_estimator.is_calibrated():
            imu_stationary = self._gravity_estimator.is_stationary_from_accel(threshold=0.3)
            # Both visual and IMU must agree for stationary
            return visual_stationary and imu_stationary
        
        return visual_stationary
    
    def process(
        self,
        left_ir: np.ndarray,
        right_ir: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> VIOResult:
        """
        Process stereo frame pair with optional IMU data.
        
        Args:
            left_ir: Left IR camera image (grayscale or BGR)
            right_ir: Right IR camera image (grayscale or BGR)
            timestamp: Frame timestamp in seconds
            depth: Optional depth image (not used, for API compatibility)
            accel: Optional 3D accelerometer reading (m/s²)
            gyro: Optional 3D gyroscope reading (rad/s)
            
        Returns:
            VIOResult with pose, velocity, and tracking status
        """
        t_start = time.perf_counter()
        
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        self._frame_count += 1
        
        # Ensure grayscale
        left_gray = left_ir if len(left_ir.shape) == 2 else cv2.cvtColor(left_ir, cv2.COLOR_BGR2GRAY)
        right_gray = right_ir if len(right_ir.shape) == 2 else cv2.cvtColor(right_ir, cv2.COLOR_BGR2GRAY)
        
        # Compute dt
        dt = 0.033
        if self._prev_time is not None:
            dt = max(0.001, min(0.5, timestamp - self._prev_time))
        
        # Process IMU data (accelerometer)
        if self.use_imu and self._gravity_estimator is not None and accel is not None:
            calibration_done = self._gravity_estimator.add_sample(accel)
            
            # During calibration phase, return early
            if not calibration_done:
                proc_time = (time.perf_counter() - t_start) * 1000
                return VIOResult(
                    pose=self._pose.copy(),
                    velocity=self._velocity.copy(),
                    tracking_status="calibrating",
                    num_inliers=0,
                    confidence=len(self._gravity_estimator._samples) / self._gravity_estimator._num_samples,
                    processing_time_ms=proc_time,
                )
        
        # Initialize on first frame
        if not self._initialized:
            points_2d = self._detect_features(left_gray)
            if len(points_2d) >= self.min_features:
                points_2d, points_3d = self._triangulate_stereo(left_gray, right_gray, points_2d)
                if len(points_2d) >= self.min_features:
                    self._prev_gray = left_gray
                    self._prev_points = points_2d
                    self._prev_points_3d = points_3d
                    self._prev_time = timestamp
                    self._initialized = True
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                tracking_status="initializing",
                num_inliers=len(points_2d) if self._initialized else 0,
                confidence=0.5 if self._initialized else 0.0,
                processing_time_ms=proc_time,
            )
        
        # Re-detect if too few points
        if self._prev_points is None or len(self._prev_points) < self.min_features:
            points_2d = self._detect_features(self._prev_gray)
            points_2d, points_3d = self._triangulate_stereo(self._prev_gray, right_gray, points_2d)
            self._prev_points = points_2d
            self._prev_points_3d = points_3d
        
        if self._prev_points is None or len(self._prev_points) < 6:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                tracking_status="lost",
                num_inliers=0,
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # Track with optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            left_gray,
            self._prev_points.reshape(-1, 1, 2),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
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
        h, w = left_gray.shape
        valid = (
            (curr_2d[:, 0] >= 0) & (curr_2d[:, 0] < w) &
            (curr_2d[:, 1] >= 0) & (curr_2d[:, 1] < h)
        )
        prev_2d = prev_2d[valid]
        prev_3d = prev_3d[valid]
        curr_2d = curr_2d[valid]
        
        # Check if stationary (skip pose update to prevent drift)
        if self._is_stationary(prev_2d, curr_2d):
            self._prev_gray = left_gray
            self._prev_time = timestamp
            
            # Re-detect for next frame
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._triangulate_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) >= self.min_features:
                self._prev_points = new_pts
                self._prev_points_3d = new_3d
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._pose.copy(),
                velocity=np.zeros(3),
                tracking_status="ok",
                num_inliers=len(self._prev_points) if self._prev_points is not None else 0,
                confidence=0.9,
                processing_time_ms=proc_time,
            )
        
        # Estimate motion with PnP
        T_curr_prev, num_inliers = self._estimate_motion_pnp(prev_3d, curr_2d)
        
        if T_curr_prev is None or num_inliers < 4:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
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
        
        # Apply filtering
        filtered_pose, accepted = self._pose_filter.filter(new_pose)
        
        if accepted:
            self._pose = filtered_pose
            self._velocity = T_rel[:3, 3] / dt
        
        # Compute confidence
        confidence = min(1.0, num_inliers / 50.0) * (0.9 if accepted else 0.5)
        
        # Update tracking state
        self._prev_gray = left_gray
        self._prev_time = timestamp
        self._prev_points = curr_2d
        self._prev_points_3d = prev_3d
        
        # Add new features if running low
        if len(self._prev_points) < self.max_features // 2:
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._triangulate_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) > 0:
                self._prev_points = np.vstack([self._prev_points, new_pts])[:self.max_features]
                self._prev_points_3d = np.vstack([self._prev_points_3d, new_3d])[:self.max_features]
        
        proc_time = (time.perf_counter() - t_start) * 1000
        
        return VIOResult(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            tracking_status="ok",
            num_inliers=num_inliers,
            confidence=confidence,
            processing_time_ms=proc_time,
        )
    
    def get_pose(self) -> np.ndarray:
        """Get current camera-to-world pose in OpenCV convention."""
        return self._pose.copy()
    
    def get_pose_opengl(self) -> np.ndarray:
        """
        Get pose in OpenGL convention (Y-up, Z-backward).
        
        Suitable for OpenGL renderers and some 3DGS viewers.
        """
        CV_TO_GL = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)
        return CV_TO_GL @ self._pose @ CV_TO_GL.T
    
    def get_pose_gravity_aligned(self) -> np.ndarray:
        """
        Get pose in gravity-aligned world frame.
        
        Only available if use_imu=True and gravity is calibrated.
        World frame has Y pointing up (opposite to gravity).
        """
        if not self.use_imu or not self._gravity_estimator.is_calibrated():
            return self.get_pose()
        
        R_gravity = self._gravity_estimator.get_gravity_rotation()
        
        pose_aligned = np.eye(4)
        pose_aligned[:3, :3] = R_gravity @ self._pose[:3, :3]
        pose_aligned[:3, 3] = R_gravity @ self._pose[:3, 3]
        
        return pose_aligned
    
    def is_gravity_calibrated(self) -> bool:
        """Check if gravity calibration is complete."""
        return self.use_imu and self._gravity_estimator is not None and self._gravity_estimator.is_calibrated()
    
    def get_gravity_vector(self) -> Optional[np.ndarray]:
        """Get the estimated gravity vector in camera frame."""
        if not self.is_gravity_calibrated():
            return None
        return self._gravity_estimator.get_gravity()
    
    def get_imu_status(self) -> Dict:
        """
        Get current IMU status for debugging/monitoring.
        
        Returns dict with:
            - calibrated: bool
            - gravity: [x, y, z] or None
            - gravity_magnitude: float (should be ~9.81)
            - samples_collected: int (during calibration)
        """
        if not self.use_imu or self._gravity_estimator is None:
            return {
                'enabled': False,
                'calibrated': False,
                'gravity': None,
                'gravity_magnitude': 0.0,
                'samples_collected': 0,
            }
        
        return {
            'enabled': True,
            'calibrated': self._gravity_estimator.is_calibrated(),
            'gravity': self._gravity_estimator.get_gravity().tolist() if self._gravity_estimator.is_calibrated() else None,
            'gravity_magnitude': self._gravity_estimator.get_gravity_magnitude(),
            'samples_collected': len(self._gravity_estimator._samples),
        }
    
    def reset(self):
        """Reset VIO state to initial."""
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_time = None
        self._motion_history.clear()
        self._frame_count = 0
        
        if self._gravity_estimator:
            self._gravity_estimator.reset()
        self._pose_filter.reset()


# Convenience factory function
def create_stereo_vio(
    baseline: float = 0.05,
    intrinsics: Optional[Dict] = None,
    fast: bool = True,
    use_imu: bool = True,
) -> StereoVIO:
    """
    Create a StereoVIO instance with common settings.
    
    Args:
        baseline: Stereo baseline in meters
        intrinsics: Dictionary with fx, fy, cx, cy
        fast: If True, use FAST features (faster); else GFTT (more stable)
        use_imu: Enable IMU-based gravity alignment
        
    Returns:
        Configured StereoVIO instance
    """
    vio = StereoVIO(
        baseline=baseline,
        max_features=250 if fast else 400,
        min_features=40 if fast else 80,
        feature_type=FeatureType.FAST if fast else FeatureType.GFTT,
        use_imu=use_imu,
        pose_smoothing=0.3 if fast else 0.5,
    )
    
    if intrinsics:
        vio.set_intrinsics_from_dict(intrinsics)
    
    return vio
