"""
Simple Visual Odometry for RealSense D435i

Outputs raw camera-to-world poses in OpenCV convention.
No gravity alignment - keeps coordinate frame as-is for 3DGS compatibility.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
import time


@dataclass 
class VOResult:
    """Result from VO processing."""
    pose: np.ndarray  # 4x4 camera-to-world
    tracking_status: str  # "ok", "lost", "initializing"
    num_features: int
    processing_time_ms: float


class SimpleVO:
    """
    Simple stereo visual odometry.
    
    Outputs camera-to-world poses in standard OpenCV convention:
    - X right
    - Y down  
    - Z forward
    
    No coordinate transforms - raw odometry output for maximum compatibility.
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        max_features: int = 300,
        motion_threshold: float = 1.0,  # Pixel motion threshold for stationary detection
    ):
        self.baseline = baseline
        self.max_features = max_features
        self.motion_threshold = motion_threshold
        
        # Intrinsics
        self._K = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # State - start at identity (camera at origin, looking along +Z)
        self._pose = np.eye(4, dtype=np.float64)
        self._initialized = False
        
        # Previous frame data
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        
        # ORB detector (more robust than FAST)
        self._orb = cv2.ORB_create(nfeatures=max_features)
        
        # Optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Motion history for smoothing
        self._motion_history = deque(maxlen=5)
        
        self._frame_count = 0
    
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
        """Detect ORB keypoints."""
        kps = self._orb.detect(gray, None)
        if len(kps) == 0:
            return np.array([]).reshape(0, 2)
        
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:self.max_features]
        return np.array([kp.pt for kp in kps], dtype=np.float32)
    
    def _match_stereo(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        points_left: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match features between stereo pair and triangulate 3D points."""
        if len(points_left) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        # Track from left to right using optical flow
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
        valid = row_diff < 2.0
        
        # Disparity must be positive (left camera sees objects further left)
        disparity = pts_l[:, 0] - pts_r[:, 0]
        valid &= (disparity > 0.5) & (disparity < 200)
        
        if not np.any(valid):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        pts_l = pts_l[valid]
        disp = disparity[valid]
        
        # Triangulate: Z = f * baseline / disparity
        Z = (self._fx * self.baseline) / disp
        
        # Filter valid depths
        valid_depth = (Z > 0.1) & (Z < 10.0)
        if not np.any(valid_depth):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
        
        pts_l = pts_l[valid_depth]
        Z = Z[valid_depth]
        
        # Back-project to 3D
        X = (pts_l[:, 0] - self._cx) * Z / self._fx
        Y = (pts_l[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return pts_l, points_3d
    
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
            iterationsCount=100,
            reprojectionError=3.0,
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
        """Check if camera is stationary based on feature motion."""
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
            if len(points_2d) >= 20:
                points_2d, points_3d = self._match_stereo(left_gray, right_gray, points_2d)
                if len(points_2d) >= 20:
                    self._prev_gray = left_gray
                    self._prev_points = points_2d
                    self._prev_points_3d = points_3d
                    self._initialized = True
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                tracking_status="initializing",
                num_features=len(points_2d) if self._initialized else 0,
                processing_time_ms=proc_time,
            )
        
        # Re-detect if we don't have enough features
        if self._prev_points is None or len(self._prev_points) < 20:
            points_2d = self._detect_features(self._prev_gray)
            points_2d, points_3d = self._match_stereo(self._prev_gray, right_gray, points_2d)
            self._prev_points = points_2d
            self._prev_points_3d = points_3d
        
        if len(self._prev_points) < 6:
            self._prev_gray = left_gray
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        # Track features with optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, left_gray,
            self._prev_points.reshape(-1, 1, 2),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._prev_gray = left_gray
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        status = status.flatten().astype(bool)
        prev_3d = self._prev_points_3d[status]
        curr_2d = curr_points[status].reshape(-1, 2)
        prev_2d = self._prev_points[status]
        
        # Bounds check
        h, w = left_gray.shape
        valid = (curr_2d[:, 0] >= 0) & (curr_2d[:, 0] < w) & (curr_2d[:, 1] >= 0) & (curr_2d[:, 1] < h)
        prev_3d = prev_3d[valid]
        curr_2d = curr_2d[valid]
        prev_2d = prev_2d[valid]
        
        # Check if stationary - don't update pose if not moving
        if self._is_stationary(prev_2d, curr_2d):
            self._prev_gray = left_gray
            # Re-detect features for next frame
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._match_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) >= 10:
                self._prev_points = new_pts
                self._prev_points_3d = new_3d
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                tracking_status="ok",
                num_features=len(self._prev_points) if self._prev_points is not None else 0,
                processing_time_ms=proc_time,
            )
        
        # Estimate motion with PnP
        T_curr_prev, num_inliers = self._estimate_motion(prev_3d, curr_2d)
        
        if T_curr_prev is None or num_inliers < 6:
            self._prev_gray = left_gray
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VOResult(
                pose=self._pose.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        # Update pose: new_pose = old_pose * inv(T_curr_prev)
        # T_curr_prev transforms points from prev frame to curr frame
        # So camera moved by inv(T_curr_prev) in world
        T_rel = np.linalg.inv(T_curr_prev)
        self._pose = self._pose @ T_rel
        
        # Update tracking state
        self._prev_gray = left_gray
        self._prev_points = curr_2d
        self._prev_points_3d = prev_3d
        
        # Add new features if running low
        if len(self._prev_points) < 50:
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
            pose=self._pose.copy(),
            tracking_status="ok",
            num_features=len(self._prev_points),
            processing_time_ms=proc_time,
        )
    
    def get_pose(self) -> np.ndarray:
        """Get current camera-to-world pose."""
        return self._pose.copy()
    
    def reset(self):
        """Reset VO state."""
        self._pose = np.eye(4, dtype=np.float64)
        self._initialized = False
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._motion_history.clear()
        self._frame_count = 0
