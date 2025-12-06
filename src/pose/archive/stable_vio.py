"""
Stable Visual-Inertial Odometry for RealSense D435i

Designed for 3D Gaussian Splatting with focus on:
1. Stability - minimal drift when stationary
2. World alignment - gravity-aligned coordinate frame  
3. Smooth motion - filtered pose output
4. Accurate scale - from stereo baseline
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
import time


@dataclass 
class VIOResult:
    """Result from VIO processing."""
    pose: np.ndarray  # 4x4 camera-to-world
    velocity: np.ndarray  # 3D velocity
    tracking_status: str  # "ok", "lost", "initializing"
    num_features: int
    processing_time_ms: float


class StableVIO:
    """
    Stable stereo visual odometry optimized for 3DGS.
    
    Key features:
    - Motion detection to avoid drift when stationary
    - Exponential moving average pose smoothing
    - Outlier rejection based on motion consistency
    - Proper gravity alignment from IMU
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        max_features: int = 200,
        motion_threshold: float = 0.5,  # Pixel motion threshold
        translation_threshold: float = 0.001,  # 1mm minimum motion
        smoothing_alpha: float = 0.7,  # Higher = more smoothing
        max_velocity: float = 1.0,  # m/s - reject faster as outlier
    ):
        self.baseline = baseline
        self.max_features = max_features
        self.motion_threshold = motion_threshold
        self.translation_threshold = translation_threshold
        self.smoothing_alpha = smoothing_alpha
        self.max_velocity = max_velocity
        
        # Intrinsics
        self._K = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # State
        self._pose = np.eye(4, dtype=np.float64)
        self._smoothed_pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Gravity alignment
        self._gravity_samples: List[np.ndarray] = []
        self._gravity_calibrated = False
        self._R_gravity = np.eye(3)  # Rotation to align gravity with -Y (OpenCV convention)
        
        # Previous frame data
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_time = None
        
        # FAST detector
        self._fast = cv2.FastFeatureDetector_create(
            threshold=20,
            nonmaxSuppression=True,
        )
        
        # Optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Motion history for stationary detection
        self._motion_history = deque(maxlen=10)
        
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
    
    def _calibrate_gravity(self, accel: np.ndarray) -> bool:
        """
        Calibrate gravity direction from accelerometer.
        
        RealSense D435i has different IMU and camera coordinate frames:
        - IMU frame: X right, Y UP, Z forward
        - Camera frame: X right, Y DOWN, Z forward
        
        When camera is level:
        - IMU reads gravity as ~[0, -9.8, 0] (Y is up, so gravity is -Y)
        - In camera frame, "down" is +Y
        
        We want a Y-up world frame where camera Y-down maps to world -Y.
        """
        self._gravity_samples.append(accel.copy())
        
        if len(self._gravity_samples) < 30:
            return False
        
        # Average gravity vector in IMU frame
        g_imu = np.mean(self._gravity_samples, axis=0)
        g_norm = np.linalg.norm(g_imu)
        
        if g_norm < 5.0:
            print("Warning: Low gravity magnitude, skipping alignment")
            self._gravity_calibrated = True
            return True
        
        # Normalize gravity vector (points in direction of gravity, i.e., DOWN)
        g_unit = g_imu / g_norm
        
        # IMU Y is UP, camera Y is DOWN
        # So gravity in camera frame is -g_imu (flip Y)
        # When level: IMU says [0, -1, 0], camera frame would be [0, 1, 0]
        g_camera = g_unit * np.array([1, -1, 1])  # Flip Y for camera frame
        
        # Now g_camera represents "down" direction in camera frame
        # We want "down" to be [0, -1, 0] in world frame (Y-up world)
        target_down = np.array([0, -1, 0])
        
        # Find rotation that takes g_camera to target_down
        self._R_gravity = self._rotation_between_vectors(g_camera, target_down)
        
        # Verify
        rotated_g = self._R_gravity @ g_camera
        
        self._gravity_calibrated = True
        print(f"Gravity calibrated: |g|={g_norm:.2f} m/s²")
        print(f"  Gravity in camera frame: [{g_camera[0]:.3f}, {g_camera[1]:.3f}, {g_camera[2]:.3f}]")
        print(f"  After alignment: [{rotated_g[0]:.3f}, {rotated_g[1]:.3f}, {rotated_g[2]:.3f}] (should be [0, -1, 0])")
        
        return True
    
    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute rotation matrix that rotates v1 to v2."""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        if np.linalg.norm(cross) < 1e-6:
            if dot > 0:
                return np.eye(3)
            else:
                # 180 degree rotation
                perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(v1, perp)
                axis = axis / np.linalg.norm(axis)
                K = np.array([[0, -axis[2], axis[1]], 
                              [axis[2], 0, -axis[0]], 
                              [-axis[1], axis[0], 0]])
                return np.eye(3) + 2 * K @ K
        
        # Rodrigues formula
        K = np.array([[0, -cross[2], cross[1]], 
                      [cross[2], 0, -cross[0]], 
                      [-cross[1], cross[0], 0]])
        
        R = np.eye(3) + K + K @ K * (1 - dot) / (np.linalg.norm(cross) ** 2)
        return R
    
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect FAST features."""
        kps = self._fast.detect(gray, None)
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
        """Match features between stereo pair using optical flow."""
        if len(points_left) == 0:
            return np.array([]), np.array([])
        
        # Track from left to right
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
        pts_l = points_left[status]
        pts_r = points_right[status].reshape(-1, 2)
        
        # Epipolar constraint (same row in rectified stereo)
        row_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
        valid = row_diff < 2.0
        
        # Disparity must be positive and reasonable
        disparity = pts_l[:, 0] - pts_r[:, 0]
        valid &= (disparity > 1.0) & (disparity < 200)
        
        if not np.any(valid):
            return np.array([]), np.array([])
        
        pts_l = pts_l[valid]
        disp = disparity[valid]
        
        # Triangulate
        Z = (self._fx * self.baseline) / disp
        valid_depth = (Z > 0.1) & (Z < 10.0)
        
        if not np.any(valid_depth):
            return np.array([]), np.array([])
        
        pts_l = pts_l[valid_depth]
        Z = Z[valid_depth]
        
        X = (pts_l[:, 0] - self._cx) * Z / self._fx
        Y = (pts_l[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return pts_l, points_3d
    
    def _estimate_motion(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Estimate relative motion using PnP."""
        if len(pts_3d) < 6:
            return None, 0
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            self._K, None,
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
    
    def _is_stationary(self, motion: float) -> bool:
        """Check if camera is stationary based on motion history."""
        self._motion_history.append(motion)
        
        if len(self._motion_history) < 5:
            return False
        
        avg_motion = np.mean(self._motion_history)
        return avg_motion < self.motion_threshold
    
    def _smooth_pose(self, new_pose: np.ndarray, dt: float) -> np.ndarray:
        """Apply exponential moving average smoothing to pose."""
        # Check for outlier (too much motion)
        translation = new_pose[:3, 3] - self._smoothed_pose[:3, 3]
        velocity = np.linalg.norm(translation) / max(dt, 0.001)
        
        if velocity > self.max_velocity:
            # Reject outlier, keep previous pose
            return self._smoothed_pose.copy()
        
        # Check minimum motion threshold
        if np.linalg.norm(translation) < self.translation_threshold:
            # Too small motion, might be noise - dampen heavily
            alpha = 0.95  # Very heavy smoothing for tiny motions
        else:
            alpha = self.smoothing_alpha
        
        # Smooth translation
        smoothed = np.eye(4)
        smoothed[:3, 3] = alpha * self._smoothed_pose[:3, 3] + (1 - alpha) * new_pose[:3, 3]
        
        # Smooth rotation using SLERP-like blending
        R_prev = self._smoothed_pose[:3, :3]
        R_new = new_pose[:3, :3]
        R_blend = alpha * R_prev + (1 - alpha) * R_new
        
        # Re-orthogonalize
        U, _, Vt = np.linalg.svd(R_blend)
        smoothed[:3, :3] = U @ Vt
        if np.linalg.det(smoothed[:3, :3]) < 0:
            smoothed[:3, :3] *= -1
        
        return smoothed
    
    def process(
        self,
        left_ir: np.ndarray,
        right_ir: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> VIOResult:
        """Process stereo frame pair with optional IMU."""
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
        
        # Gravity calibration phase
        if accel is not None and not self._gravity_calibrated:
            if not self._calibrate_gravity(accel):
                proc_time = (time.perf_counter() - t_start) * 1000
                return VIOResult(
                    pose=self._get_world_pose(),
                    velocity=self._velocity.copy(),
                    tracking_status="calibrating",
                    num_features=0,
                    processing_time_ms=proc_time,
                )
        
        # Initialize
        if not self._initialized:
            points_2d = self._detect_features(left_gray)
            if len(points_2d) >= 20:
                points_2d, points_3d = self._match_stereo(left_gray, right_gray, points_2d)
                if len(points_2d) >= 20:
                    self._prev_gray = left_gray
                    self._prev_points = points_2d
                    self._prev_points_3d = points_3d
                    self._prev_time = timestamp
                    self._initialized = True
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._get_world_pose(),
                velocity=self._velocity.copy(),
                tracking_status="initializing",
                num_features=len(points_2d) if self._initialized else 0,
                processing_time_ms=proc_time,
            )
        
        # Track features
        if self._prev_points is None or len(self._prev_points) < 20:
            points_2d = self._detect_features(self._prev_gray)
            points_2d, points_3d = self._match_stereo(self._prev_gray, right_gray, points_2d)
            self._prev_points = points_2d
            self._prev_points_3d = points_3d
        
        if len(self._prev_points) < 6:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._get_world_pose(),
                velocity=self._velocity.copy(),
                tracking_status="lost",
                num_features=0,
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
            return VIOResult(
                pose=self._get_world_pose(),
                velocity=self._velocity.copy(),
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
        
        # Calculate pixel motion
        if len(prev_2d) > 0:
            pixel_motion = np.mean(np.linalg.norm(curr_2d - prev_2d, axis=1))
        else:
            pixel_motion = 0
        
        # Check if stationary
        is_stationary = self._is_stationary(pixel_motion)
        
        if is_stationary or len(prev_3d) < 6:
            # Don't update pose when stationary - prevents drift
            self._prev_gray = left_gray
            self._prev_time = timestamp
            
            # Re-detect features for next frame
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._match_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) >= 10:
                self._prev_points = new_pts
                self._prev_points_3d = new_3d
            
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._get_world_pose(),
                velocity=np.zeros(3),
                tracking_status="ok",
                num_features=len(self._prev_points) if self._prev_points is not None else 0,
                processing_time_ms=proc_time,
            )
        
        # Estimate motion with PnP
        T_curr_prev, num_inliers = self._estimate_motion(prev_3d, curr_2d)
        
        if T_curr_prev is None:
            self._prev_gray = left_gray
            self._prev_time = timestamp
            self._prev_points = None
            proc_time = (time.perf_counter() - t_start) * 1000
            return VIOResult(
                pose=self._get_world_pose(),
                velocity=self._velocity.copy(),
                tracking_status="lost",
                num_features=0,
                processing_time_ms=proc_time,
            )
        
        # Update pose (camera-to-world)
        T_rel = np.linalg.inv(T_curr_prev)
        new_pose = self._pose @ T_rel
        
        # Apply smoothing
        self._smoothed_pose = self._smooth_pose(new_pose, dt)
        self._pose = new_pose
        self._velocity = T_rel[:3, 3] / dt
        
        # Update tracking state
        self._prev_gray = left_gray
        self._prev_time = timestamp
        self._prev_points = curr_2d
        self._prev_points_3d = prev_3d
        
        # Add new features if needed
        if len(self._prev_points) < 50:
            new_pts = self._detect_features(left_gray)
            new_pts, new_3d = self._match_stereo(left_gray, right_gray, new_pts)
            if len(new_pts) > 0:
                self._prev_points = np.vstack([self._prev_points, new_pts])[:self.max_features]
                self._prev_points_3d = np.vstack([self._prev_points_3d, new_3d])[:self.max_features]
        
        proc_time = (time.perf_counter() - t_start) * 1000
        
        return VIOResult(
            pose=self._get_world_pose(),
            velocity=self._velocity.copy(),
            tracking_status="ok",
            num_features=len(self._prev_points),
            processing_time_ms=proc_time,
        )
    
    def _get_world_pose(self) -> np.ndarray:
        """Get pose in gravity-aligned world frame (Y-up)."""
        pose_world = np.eye(4)
        
        # Apply gravity alignment
        # R_gravity aligns the IMU/camera frame to world frame where Y is up
        # We need to apply it carefully to the camera-to-world transform
        
        # The smoothed pose is in the original camera frame
        # We want to rotate the entire coordinate system so gravity aligns with -Y
        
        # For camera-to-world pose T_wc:
        # - R_wc (rotation) tells us where camera axes point in world
        # - t_wc (translation) tells us camera position in world
        
        # Both need to be rotated by R_gravity
        pose_world[:3, :3] = self._R_gravity @ self._smoothed_pose[:3, :3]
        pose_world[:3, 3] = self._R_gravity @ self._smoothed_pose[:3, 3]
        
        return pose_world
    
    def get_pose(self) -> np.ndarray:
        """Get current camera-to-world pose (gravity-aligned, Y-up)."""
        return self._get_world_pose()
    
    def reset(self):
        """Reset VIO state."""
        self._pose = np.eye(4, dtype=np.float64)
        self._smoothed_pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._gravity_samples = []
        self._gravity_calibrated = False
        self._R_gravity = np.eye(3)
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_time = None
        self._motion_history.clear()
        self._frame_count = 0
