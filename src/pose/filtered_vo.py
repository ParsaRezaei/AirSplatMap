"""
Filtered Visual Odometry for RealSense D435i

Key improvements over SimpleVO:
1. Pose filtering with exponential moving average
2. Outlier rejection based on motion magnitude
3. Better feature tracking with forward-backward check
4. Minimum inlier threshold for pose acceptance
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import time
from scipy.spatial.transform import Rotation


@dataclass 
class VOResult:
    """Result from VO processing."""
    pose: np.ndarray  # 4x4 camera-to-world
    tracking_status: str  # "ok", "lost", "initializing"
    num_features: int
    processing_time_ms: float


class FilteredVO:
    """
    Filtered stereo visual odometry with pose smoothing.
    
    Features:
    - EMA pose filtering to reduce jitter
    - Forward-backward optical flow check
    - Motion magnitude rejection
    - Minimum inlier threshold
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        max_features: int = 400,
        motion_threshold: float = 0.8,  # Pixel motion for stationary detection
        pose_filter_alpha: float = 0.6,  # Higher = more filtering (0-1)
        max_translation_per_frame: float = 0.15,  # meters - reject larger translations
        max_rotation_per_frame: float = 0.15,  # radians (~8.5 degrees) - reject larger rotations
        min_inliers: int = 15,  # Minimum inliers for valid pose
    ):
        self.baseline = baseline
        self.max_features = max_features
        self.motion_threshold = motion_threshold
        self.pose_filter_alpha = pose_filter_alpha
        self.max_translation_per_frame = max_translation_per_frame
        self.max_rotation_per_frame = max_rotation_per_frame
        self.min_inliers = min_inliers
        
        # Intrinsics
        self._K = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # State
        self._raw_pose = np.eye(4, dtype=np.float64)
        self._filtered_pose = np.eye(4, dtype=np.float64)
        self._initialized = False
        
        # Previous frame data
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_time = None
        
        # Feature detector - use GFTT (Good Features to Track) for stability
        self._gftt_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )
        
        # Optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Motion history for stationary detection
        self._motion_history = deque(maxlen=10)
        
        # Translation/rotation history for outlier detection
        self._translation_history = deque(maxlen=20)
        self._rotation_history = deque(maxlen=20)
        
        self._frame_count = 0
        self._consecutive_failures = 0
    
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float, baseline: float = None):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        if baseline:
            self.baseline = abs(baseline)
    
    def set_intrinsics_from_dict(self, d: Dict):
        self.set_intrinsics(d['fx'], d['fy'], d['cx'], d['cy'], d.get('baseline'))
    
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect good features to track."""
        corners = cv2.goodFeaturesToTrack(gray, **self._gftt_params)
        if corners is None or len(corners) == 0:
            return np.array([]).reshape(0, 2)
        return corners.reshape(-1, 2).astype(np.float32)
    
    def _match_stereo(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        points_left: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match features between stereo pair and triangulate 3D points."""
        if len(points_left) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        # Track from left to right
        points_right, status, _ = cv2.calcOpticalFlowPyrLK(
            left_gray, right_gray,
            points_left.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
        )
        
        if points_right is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        status = status.flatten().astype(bool)
        pts_l = points_left[status]
        pts_r = points_right[status].reshape(-1, 2)
        
        if len(pts_l) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        # Epipolar constraint - points should be on same row (rectified stereo)
        row_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
        valid = row_diff < 1.5  # Tighter threshold
        
        # Disparity must be positive and reasonable
        disparity = pts_l[:, 0] - pts_r[:, 0]
        valid &= (disparity > 1.0) & (disparity < 150)
        
        if not np.any(valid):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        pts_l = pts_l[valid]
        disp = disparity[valid]
        
        # Triangulate: Z = f * baseline / disparity
        Z = (self._fx * self.baseline) / disp
        
        # Filter valid depths - tighter range
        valid_depth = (Z > 0.2) & (Z < 6.0)
        if not np.any(valid_depth):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        pts_l = pts_l[valid_depth]
        Z = Z[valid_depth]
        
        # Back-project to 3D
        X = (pts_l[:, 0] - self._cx) * Z / self._fx
        Y = (pts_l[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return pts_l, points_3d
    
    def _track_features_bidirectional(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track features with forward-backward consistency check."""
        if len(prev_points) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Forward tracking
        curr_points, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            prev_points.reshape(-1, 1, 2),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            return np.array([]), np.array([]), np.array([])
        
        # Backward tracking for consistency check
        back_points, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray,
            curr_points,
            None,
            **self._lk_params
        )
        
        if back_points is None:
            return np.array([]), np.array([]), np.array([])
        
        # Check forward-backward consistency
        status_fwd = status_fwd.flatten().astype(bool)
        status_bwd = status_bwd.flatten().astype(bool)
        
        back_error = np.linalg.norm(
            prev_points.reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1
        )
        
        # Only keep points with good forward-backward consistency
        valid = status_fwd & status_bwd & (back_error < 1.0)
        
        prev_valid = prev_points[valid]
        curr_valid = curr_points[valid].reshape(-1, 2)
        
        return prev_valid, curr_valid, valid
    
    def _estimate_motion(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Estimate relative motion using PnP RANSAC."""
        if len(pts_3d) < 6:
            return None, 0
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            self._K, None,
            iterationsCount=200,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success or inliers is None or len(inliers) < self.min_inliers:
            return None, 0
        
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T, len(inliers)
    
    def _is_motion_valid(self, T_rel: np.ndarray) -> bool:
        """Check if relative motion is within reasonable bounds."""
        # Extract translation and rotation
        translation = np.linalg.norm(T_rel[:3, 3])
        
        # Rotation angle from rotation matrix
        R = T_rel[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        # Check bounds
        if translation > self.max_translation_per_frame:
            return False
        if angle > self.max_rotation_per_frame:
            return False
        
        # Check against history (reject outliers)
        if len(self._translation_history) >= 5:
            median_trans = np.median(self._translation_history)
            if translation > median_trans * 5 + 0.02:  # Allow small + 5x median
                return False
        
        return True
    
    def _filter_pose(self, new_pose: np.ndarray) -> np.ndarray:
        """Apply exponential moving average filter to pose."""
        alpha = self.pose_filter_alpha
        
        # Filter translation
        new_t = new_pose[:3, 3]
        old_t = self._filtered_pose[:3, 3]
        filtered_t = alpha * old_t + (1 - alpha) * new_t
        
        # Filter rotation using SLERP approximation
        # Convert to quaternions for proper interpolation
        try:
            R_old = Rotation.from_matrix(self._filtered_pose[:3, :3])
            R_new = Rotation.from_matrix(new_pose[:3, :3])
            
            # SLERP
            q_old = R_old.as_quat()
            q_new = R_new.as_quat()
            
            # Ensure shortest path
            if np.dot(q_old, q_new) < 0:
                q_new = -q_new
            
            q_filtered = alpha * q_old + (1 - alpha) * q_new
            q_filtered = q_filtered / np.linalg.norm(q_filtered)
            
            R_filtered = Rotation.from_quat(q_filtered).as_matrix()
        except:
            # Fallback: simple matrix blend + orthogonalization
            R_blend = alpha * self._filtered_pose[:3, :3] + (1 - alpha) * new_pose[:3, :3]
            U, _, Vt = np.linalg.svd(R_blend)
            R_filtered = U @ Vt
            if np.linalg.det(R_filtered) < 0:
                R_filtered = -R_filtered
        
        # Build filtered pose
        filtered_pose = np.eye(4)
        filtered_pose[:3, :3] = R_filtered
        filtered_pose[:3, 3] = filtered_t
        
        return filtered_pose
    
    def _is_stationary(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> bool:
        """Check if camera is stationary."""
        if len(prev_pts) == 0 or len(curr_pts) == 0:
            return False
        
        motion = np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1))
        self._motion_history.append(motion)
        
        if len(self._motion_history) < 3:
            return False
        
        avg_motion = np.mean(self._motion_history)
        return avg_motion < self.motion_threshold
    
    def process(
        self,
        left_ir: np.ndarray,
        right_ir: np.ndarray,
        timestamp: float = None,
        depth: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> VOResult:
        """Process stereo frame pair."""
        t_start = time.perf_counter()
        
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        self._frame_count += 1
        
        # Ensure grayscale
        left_gray = left_ir if len(left_ir.shape) == 2 else cv2.cvtColor(left_ir, cv2.COLOR_BGR2GRAY)
        right_gray = right_ir if len(right_ir.shape) == 2 else cv2.cvtColor(right_ir, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if not self._initialized:
            points_2d = self._detect_features(left_gray)
            if len(points_2d) >= 30:
                points_2d, points_3d = self._match_stereo(left_gray, right_gray, points_2d)
                if len(points_2d) >= 30:
                    self._prev_gray = left_gray
                    self._prev_points = points_2d
                    self._prev_points_3d = points_3d
                    self._prev_time = timestamp
                    self._initialized = True
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._filtered_pose.copy(),
                tracking_status="initializing",
                num_features=len(points_2d) if self._initialized else 0,
                processing_time_ms=proc_time,
            )
        
        # Re-detect if we don't have enough features
        if self._prev_points is None or len(self._prev_points) < 30:
            points_2d = self._detect_features(self._prev_gray if self._prev_gray is not None else left_gray)
            points_2d, points_3d = self._match_stereo(
                self._prev_gray if self._prev_gray is not None else left_gray,
                right_gray,
                points_2d
            )
            self._prev_points = points_2d
            self._prev_points_3d = points_3d
        
        if self._prev_points is None or len(self._prev_points) < 6:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            self._consecutive_failures += 1
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._filtered_pose.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        # Track features with bidirectional check
        prev_valid, curr_valid, valid_mask = self._track_features_bidirectional(
            self._prev_gray, left_gray, self._prev_points
        )
        
        if len(prev_valid) < 6:
            self._prev_gray = left_gray
            self._prev_points = None
            self._prev_time = timestamp
            self._consecutive_failures += 1
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._filtered_pose.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        # Get corresponding 3D points
        prev_3d = self._prev_points_3d[valid_mask]
        
        # Bounds check
        h, w = left_gray.shape
        bounds_valid = (curr_valid[:, 0] >= 0) & (curr_valid[:, 0] < w) & \
                       (curr_valid[:, 1] >= 0) & (curr_valid[:, 1] < h)
        prev_3d = prev_3d[bounds_valid]
        curr_valid = curr_valid[bounds_valid]
        prev_valid = prev_valid[bounds_valid]
        
        # Check if stationary
        if self._is_stationary(prev_valid, curr_valid):
            self._prev_gray = left_gray
            self._prev_time = timestamp
            # Re-detect features for next frame
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._match_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) >= 20:
                self._prev_points = new_pts
                self._prev_points_3d = new_3d
            
            self._consecutive_failures = 0
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._filtered_pose.copy(),
                tracking_status="ok",
                num_features=len(self._prev_points) if self._prev_points is not None else 0,
                processing_time_ms=proc_time,
            )
        
        # Estimate motion with PnP
        T_curr_prev, num_inliers = self._estimate_motion(prev_3d, curr_valid)
        
        if T_curr_prev is None or num_inliers < self.min_inliers:
            self._prev_gray = left_gray
            self._prev_points = None
            self._prev_time = timestamp
            self._consecutive_failures += 1
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._filtered_pose.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        # Compute relative motion
        T_rel = np.linalg.inv(T_curr_prev)
        
        # Validate motion
        if not self._is_motion_valid(T_rel):
            self._prev_gray = left_gray
            self._prev_points = None
            self._prev_time = timestamp
            self._consecutive_failures += 1
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._filtered_pose.copy(),
                tracking_status="rejected",
                num_features=num_inliers,
                processing_time_ms=proc_time,
            )
        
        # Update history
        translation_mag = np.linalg.norm(T_rel[:3, 3])
        self._translation_history.append(translation_mag)
        
        # Update raw pose
        self._raw_pose = self._raw_pose @ T_rel
        
        # Apply filtering
        self._filtered_pose = self._filter_pose(self._raw_pose)
        
        # Update tracking state
        self._prev_gray = left_gray
        self._prev_points = curr_valid
        self._prev_points_3d = prev_3d
        self._prev_time = timestamp
        self._consecutive_failures = 0
        
        # Add new features if running low
        if len(self._prev_points) < 80:
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._match_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) > 0 and len(self._prev_points) > 0:
                self._prev_points = np.vstack([self._prev_points, new_pts])[:self.max_features]
                self._prev_points_3d = np.vstack([self._prev_points_3d, new_3d])[:self.max_features]
            elif len(new_pts) > 0:
                self._prev_points = new_pts
                self._prev_points_3d = new_3d
        
        proc_time = (time.perf_counter() - t_start) * 1000
        
        return VOResult(
            pose=self._filtered_pose.copy(),
            tracking_status="ok",
            num_features=len(self._prev_points),
            processing_time_ms=proc_time,
        )
    
    def get_pose(self) -> np.ndarray:
        """Get current filtered camera-to-world pose."""
        return self._filtered_pose.copy()
    
    def reset(self):
        """Reset VO state."""
        self._raw_pose = np.eye(4, dtype=np.float64)
        self._filtered_pose = np.eye(4, dtype=np.float64)
        self._initialized = False
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_time = None
        self._motion_history.clear()
        self._translation_history.clear()
        self._rotation_history.clear()
        self._frame_count = 0
        self._consecutive_failures = 0
