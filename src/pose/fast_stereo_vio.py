"""
Fast Stereo Visual-Inertial Odometry for RealSense D435i

Optimized for speed:
1. FAST features instead of ORB (10x faster detection)
2. Skip stereo matching most frames - use optical flow
3. Reduced feature count
4. Frame skipping for VIO (every other frame)
5. No descriptor computation for tracking
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
import time


@dataclass 
class FastVIOResult:
    """Result from fast VIO processing."""
    pose: np.ndarray
    velocity: np.ndarray
    num_inliers: int
    tracking_status: str
    confidence: float
    processing_time_ms: float


class FastStereoVIO:
    """
    Fast Stereo VIO optimized for real-time performance.
    
    Key optimizations:
    - FAST corner detection (no descriptors needed for tracking)
    - Sparse optical flow for frame-to-frame tracking
    - Stereo triangulation only for new features
    - IMU gravity alignment for world frame
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        max_features: int = 200,  # Reduced from 500
        min_features: int = 50,
        pose_smoothing: float = 0.3,  # Less smoothing = more responsive
        max_translation_per_frame: float = 0.2,
        process_every_n_frames: int = 1,  # Process every frame
    ):
        self.baseline = baseline
        self.max_features = max_features
        self.min_features = min_features
        self.process_every_n = process_every_n_frames
        
        # Intrinsics
        self._K = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # State
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Gravity calibration
        self._gravity_samples: List[np.ndarray] = []
        self._gravity_calibrated = False
        self._R_to_world = np.eye(3)
        
        # Pose filtering
        self._smoothing = pose_smoothing
        self._max_trans = max_translation_per_frame
        self._prev_pose = None
        
        # Previous frame
        self._prev_gray = None
        self._prev_points = None  # 2D points in image
        self._prev_points_3d = None  # 3D points in camera frame
        self._prev_time = None
        
        # FAST detector - much faster than ORB
        self._fast = cv2.FastFeatureDetector_create(
            threshold=20,
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        )
        
        # Optical flow params - tuned for speed
        self._lk_params = dict(
            winSize=(15, 15),  # Smaller window = faster
            maxLevel=2,  # Fewer pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        
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
        """Detect FAST features - very fast."""
        kps = self._fast.detect(gray, None)
        if len(kps) == 0:
            return np.array([]).reshape(0, 2)
        
        # Sort by response, take top N
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:self.max_features]
        return np.array([kp.pt for kp in kps], dtype=np.float32)
    
    def _triangulate_stereo_fast(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        points_left: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast stereo triangulation using optical flow to find correspondences.
        Much faster than descriptor matching.
        """
        if len(points_left) == 0:
            return np.array([]), np.array([])
        
        # Use optical flow to find right image correspondences
        # This is faster than descriptor matching
        points_right, status, _ = cv2.calcOpticalFlowPyrLK(
            left_gray, right_gray,
            points_left.reshape(-1, 1, 2),
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        
        if points_right is None:
            return np.array([]), np.array([])
        
        status = status.flatten().astype(bool)
        points_left_valid = points_left[status]
        points_right_valid = points_right[status].reshape(-1, 2)
        
        # Epipolar constraint: same row (stereo rectified)
        row_diff = np.abs(points_left_valid[:, 1] - points_right_valid[:, 1])
        epipolar_valid = row_diff < 2.0
        
        # Disparity constraint: right should be left of left point
        disparity = points_left_valid[:, 0] - points_right_valid[:, 0]
        disparity_valid = disparity > 1.0
        
        valid = epipolar_valid & disparity_valid
        
        if not np.any(valid):
            return np.array([]), np.array([])
        
        pts_l = points_left_valid[valid]
        disp = disparity[valid]
        
        # Triangulate
        Z = (self._fx * self.baseline) / disp
        depth_valid = (Z > 0.2) & (Z < 8.0)
        
        if not np.any(depth_valid):
            return np.array([]), np.array([])
        
        pts_l = pts_l[depth_valid]
        Z = Z[depth_valid]
        
        X = (pts_l[:, 0] - self._cx) * Z / self._fx
        Y = (pts_l[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return pts_l, points_3d
    
    def _estimate_pose_pnp(self, pts_3d: np.ndarray, pts_2d: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """Fast PnP with minimal iterations."""
        if len(pts_3d) < 6:
            return None, 0
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            self._K, None,
            iterationsCount=50,  # Fewer iterations
            reprojectionError=3.0,  # More lenient
            confidence=0.95,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success or inliers is None or len(inliers) < 4:
            return None, 0
        
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T, len(inliers)
    
    def _smooth_pose(self, new_pose: np.ndarray) -> np.ndarray:
        """Simple pose smoothing."""
        if self._prev_pose is None:
            self._prev_pose = new_pose.copy()
            return new_pose
        
        # Check for outlier
        trans_delta = np.linalg.norm(new_pose[:3, 3] - self._prev_pose[:3, 3])
        if trans_delta > self._max_trans:
            # Reject outlier
            return self._prev_pose.copy()
        
        # Smooth
        smoothed = np.eye(4)
        smoothed[:3, 3] = self._smoothing * self._prev_pose[:3, 3] + (1 - self._smoothing) * new_pose[:3, 3]
        
        # Smooth rotation
        R_blend = self._smoothing * self._prev_pose[:3, :3] + (1 - self._smoothing) * new_pose[:3, :3]
        U, _, Vt = np.linalg.svd(R_blend)
        smoothed[:3, :3] = U @ Vt
        
        self._prev_pose = smoothed.copy()
        return smoothed
    
    def process(
        self,
        left_ir: np.ndarray,
        right_ir: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> FastVIOResult:
        """Process stereo frame pair."""
        t_start = time.perf_counter()
        
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        self._frame_count += 1
        
        # Ensure grayscale
        left_gray = left_ir if len(left_ir.shape) == 2 else cv2.cvtColor(left_ir, cv2.COLOR_BGR2GRAY)
        right_gray = right_ir if len(right_ir.shape) == 2 else cv2.cvtColor(right_ir, cv2.COLOR_BGR2GRAY)
        
        dt = 0.033
        if self._prev_time is not None:
            dt = max(0.001, min(0.5, timestamp - self._prev_time))
        
        # Gravity calibration (first ~1 second)
        if accel is not None and not self._gravity_calibrated:
            self._gravity_samples.append(accel.copy())
            if len(self._gravity_samples) >= 30:
                g = np.mean(self._gravity_samples, axis=0)
                g_norm = np.linalg.norm(g)
                if g_norm > 5:
                    g_unit = g / g_norm
                    # Align gravity to -Z in world frame
                    target = np.array([0, 0, -1])
                    v = np.cross(g_unit, target)
                    c = np.dot(g_unit, target)
                    if np.linalg.norm(v) > 1e-6:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        self._R_to_world = np.eye(3) + vx + vx @ vx * (1 - c) / (np.linalg.norm(v) ** 2)
                self._gravity_calibrated = True
                print(f"Gravity calibrated")
            else:
                proc_time = (time.perf_counter() - t_start) * 1000
                return FastVIOResult(
                    pose=self._pose.copy(),
                    velocity=self._velocity.copy(),
                    num_inliers=0,
                    tracking_status="calibrating",
                    confidence=len(self._gravity_samples) / 30.0,
                    processing_time_ms=proc_time,
                )
        
        # Initialize
        if not self._initialized:
            points_2d = self._detect_features(left_gray)
            if len(points_2d) >= self.min_features:
                points_2d, points_3d = self._triangulate_stereo_fast(left_gray, right_gray, points_2d)
                if len(points_2d) >= self.min_features:
                    self._prev_gray = left_gray
                    self._prev_points = points_2d
                    self._prev_points_3d = points_3d
                    self._prev_time = timestamp
                    self._initialized = True
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return FastVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=len(points_2d) if self._initialized else 0,
                tracking_status="initializing",
                confidence=0.5 if self._initialized else 0.0,
                processing_time_ms=proc_time,
            )
        
        # Track features with optical flow
        if self._prev_points is None or len(self._prev_points) < self.min_features:
            # Need new features
            points_2d = self._detect_features(self._prev_gray)
            points_2d, points_3d = self._triangulate_stereo_fast(self._prev_gray, right_gray, points_2d)
            self._prev_points = points_2d
            self._prev_points_3d = points_3d
        
        if len(self._prev_points) < 6:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            proc_time = (time.perf_counter() - t_start) * 1000
            return FastVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # Optical flow tracking
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, left_gray,
            self._prev_points.reshape(-1, 1, 2),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            proc_time = (time.perf_counter() - t_start) * 1000
            return FastVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        status = status.flatten().astype(bool)
        prev_3d = self._prev_points_3d[status]
        curr_2d = curr_points[status].reshape(-1, 2)
        
        # Bounds check
        h, w = left_gray.shape
        valid = (curr_2d[:, 0] >= 0) & (curr_2d[:, 0] < w) & (curr_2d[:, 1] >= 0) & (curr_2d[:, 1] < h)
        prev_3d = prev_3d[valid]
        curr_2d = curr_2d[valid]
        
        if len(prev_3d) < 6:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            self._prev_points = None  # Force re-detection
            proc_time = (time.perf_counter() - t_start) * 1000
            return FastVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=len(prev_3d),
                tracking_status="lost",
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # PnP
        T_curr_prev, num_inliers = self._estimate_pose_pnp(prev_3d, curr_2d)
        
        if T_curr_prev is None:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return FastVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
                processing_time_ms=proc_time,
            )
        
        # Update pose
        T_rel = np.linalg.inv(T_curr_prev)
        new_pose_cam = self._pose @ T_rel
        
        # For 3DGS compatibility, do NOT apply gravity rotation
        # Just use standard camera-to-world accumulation
        # The gravity alignment would break Graphdeco's expected coordinate frame
        self._pose = self._smooth_pose(new_pose_cam)
        self._velocity = T_rel[:3, 3] / dt
        
        # Update tracking state
        self._prev_gray = left_gray
        self._prev_time = timestamp
        self._prev_points = curr_2d
        self._prev_points_3d = prev_3d
        
        # Add new features if running low
        if len(self._prev_points) < self.min_features:
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._triangulate_stereo_fast(left_gray, right_gray, new_pts)
            if len(new_pts) > 0:
                self._prev_points = np.vstack([self._prev_points, new_pts])[:self.max_features]
                self._prev_points_3d = np.vstack([self._prev_points_3d, new_3d])[:self.max_features]
        
        proc_time = (time.perf_counter() - t_start) * 1000
        confidence = min(1.0, num_inliers / 30.0)
        
        return FastVIOResult(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            num_inliers=num_inliers,
            tracking_status="ok",
            confidence=confidence,
            processing_time_ms=proc_time,
        )
    
    def get_pose(self) -> np.ndarray:
        """Get camera-to-world pose in OpenCV convention (no coordinate transform)."""
        return self._pose.copy()
    
    def get_pose_opengl(self) -> np.ndarray:
        """
        Get camera-to-world pose for OpenGL rendering.
        
        The internal pose is camera-to-world in OpenCV convention:
        - X right, Y down, Z forward
        
        For OpenGL viewers (Y-up, Z-backward), we flip Y and Z axes.
        """
        # OpenCV to OpenGL: flip Y and Z
        flip = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)
        return self._pose @ flip
    
    def get_pose_for_3dgs(self) -> np.ndarray:
        """
        Get camera-to-world pose for 3DGS (Graphdeco format).
        
        Graphdeco expects standard camera-to-world 4x4 matrix.
        Returns the raw pose without any coordinate transform.
        """
        return self._pose.copy()
    
    def reset(self):
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._gravity_samples = []
        self._gravity_calibrated = False
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_pose = None
        self._frame_count = 0
