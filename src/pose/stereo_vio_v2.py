"""
Stereo Visual-Inertial Odometry v2 for RealSense D435i

Improvements over v1:
1. IMU-based gravity alignment for proper world coordinates
2. Pose filtering to reject outliers and smooth trajectory
3. Better initialization with gravity calibration
4. Bounded pose output to prevent runaway values
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque


@dataclass 
class StereoVIOResult:
    """Result from stereo VIO processing."""
    pose: np.ndarray  # 4x4 camera-to-world in world frame (Z-up, Y-forward)
    velocity: np.ndarray
    num_inliers: int
    tracking_status: str  # "ok", "lost", "initializing", "calibrating"
    confidence: float


class GravityAligner:
    """
    Estimates gravity direction from accelerometer to establish world frame.
    
    World frame convention:
    - Z-up (opposite to gravity)
    - X-right  
    - Y-forward (camera looks in +Y direction at start)
    """
    
    def __init__(self, num_samples: int = 30):
        self._samples: List[np.ndarray] = []
        self._num_samples = num_samples
        self._gravity_world = np.array([0, 0, -9.81])  # Z-down in world = gravity
        self._R_imu_to_world: Optional[np.ndarray] = None
        self._calibrated = False
    
    def add_sample(self, accel: np.ndarray) -> bool:
        """Add accelerometer sample. Returns True when calibrated."""
        if self._calibrated:
            return True
        
        self._samples.append(accel.copy())
        
        if len(self._samples) >= self._num_samples:
            self._calibrate()
            return True
        return False
    
    def _calibrate(self):
        """Compute rotation from IMU frame to world frame."""
        # Average acceleration when stationary = gravity in IMU frame
        g_imu = np.mean(self._samples, axis=0)
        g_norm = np.linalg.norm(g_imu)
        
        if g_norm < 5.0:  # Sanity check - should be ~9.81
            print("Warning: Gravity magnitude too low, using default")
            self._R_imu_to_world = np.eye(3)
            self._calibrated = True
            return
        
        # Normalize gravity vector
        g_imu_unit = g_imu / g_norm
        
        # World gravity is [0, 0, -1] (Z-up means gravity is -Z)
        g_world_unit = np.array([0, 0, -1])
        
        # Find rotation from IMU gravity to world gravity
        # R * g_imu = g_world
        self._R_imu_to_world = self._rotation_between_vectors(g_imu_unit, g_world_unit)
        self._calibrated = True
        
        print(f"Gravity calibrated: |g|={g_norm:.2f}, g_imu={g_imu}")
    
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
                # 180 degree rotation - find perpendicular axis
                perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(v1, perp)
                axis = axis / np.linalg.norm(axis)
                return self._axis_angle_to_matrix(axis, np.pi)
        
        # Rodrigues formula
        K = np.array([
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0]
        ])
        
        R = np.eye(3) + K + K @ K * (1 - dot) / (np.linalg.norm(cross) ** 2)
        return R
    
    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to rotation matrix."""
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    
    def is_calibrated(self) -> bool:
        return self._calibrated
    
    def get_world_rotation(self) -> np.ndarray:
        """Get rotation matrix from IMU/camera frame to world frame."""
        if self._R_imu_to_world is None:
            return np.eye(3)
        return self._R_imu_to_world.copy()


class PoseFilter:
    """
    Filters pose estimates to:
    1. Reject outliers (sudden jumps)
    2. Smooth trajectory
    3. Bound values to reasonable range
    """
    
    def __init__(
        self,
        max_translation_per_frame: float = 0.5,  # Max 50cm per frame
        max_rotation_per_frame: float = 0.5,  # Max ~30 degrees per frame
        smoothing_alpha: float = 0.7,  # EMA smoothing factor
        max_distance_from_origin: float = 50.0,  # Max 50m from start
    ):
        self.max_trans = max_translation_per_frame
        self.max_rot = max_rotation_per_frame
        self.alpha = smoothing_alpha
        self.max_dist = max_distance_from_origin
        
        self._prev_pose: Optional[np.ndarray] = None
        self._smoothed_pose: Optional[np.ndarray] = None
    
    def filter(self, pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Filter pose estimate.
        
        Returns:
            filtered_pose: The filtered pose
            accepted: Whether the pose was accepted (False if rejected as outlier)
        """
        if self._prev_pose is None:
            self._prev_pose = pose.copy()
            self._smoothed_pose = pose.copy()
            return pose.copy(), True
        
        # Check translation jump
        trans_delta = pose[:3, 3] - self._prev_pose[:3, 3]
        trans_dist = np.linalg.norm(trans_delta)
        
        if trans_dist > self.max_trans:
            # Reject outlier - return smoothed pose
            return self._smoothed_pose.copy(), False
        
        # Check rotation jump (using Frobenius norm of rotation difference)
        R_delta = pose[:3, :3] @ self._prev_pose[:3, :3].T
        rot_angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))
        
        if rot_angle > self.max_rot:
            # Reject outlier
            return self._smoothed_pose.copy(), False
        
        # Check distance from origin
        dist_from_origin = np.linalg.norm(pose[:3, 3])
        if dist_from_origin > self.max_dist:
            # Clamp to max distance
            direction = pose[:3, 3] / dist_from_origin
            pose[:3, 3] = direction * self.max_dist
        
        # Apply smoothing
        smoothed = np.eye(4)
        
        # Smooth translation
        smoothed[:3, 3] = self.alpha * self._smoothed_pose[:3, 3] + (1 - self.alpha) * pose[:3, 3]
        
        # Smooth rotation (simple matrix blend + orthogonalization)
        R_blend = self.alpha * self._smoothed_pose[:3, :3] + (1 - self.alpha) * pose[:3, :3]
        U, _, Vt = np.linalg.svd(R_blend)
        smoothed[:3, :3] = U @ Vt
        if np.linalg.det(smoothed[:3, :3]) < 0:
            smoothed[:3, :3] = -smoothed[:3, :3]
        
        self._prev_pose = pose.copy()
        self._smoothed_pose = smoothed.copy()
        
        return smoothed, True
    
    def reset(self):
        self._prev_pose = None
        self._smoothed_pose = None


class StereoVIOv2:
    """
    Stereo Visual-Inertial Odometry v2.
    
    Features:
    - IMU-based gravity alignment for proper Z-up world frame
    - Pose filtering to reject outliers
    - Bounded output to prevent runaway values
    - Uses RealSense IR stereo cameras
    """
    
    def __init__(
        self,
        baseline: float = 0.05,
        max_features: int = 500,
        gravity_samples: int = 30,
        pose_smoothing: float = 0.5,
        max_velocity: float = 2.0,  # m/s - reject faster motion as outlier
    ):
        self.baseline = baseline
        self.max_features = max_features
        self.max_velocity = max_velocity
        
        # Camera intrinsics
        self._K = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # Gravity alignment
        self._gravity_aligner = GravityAligner(num_samples=gravity_samples)
        self._R_cam_to_world = np.eye(3)  # Will be set after gravity calibration
        
        # Pose filtering
        self._pose_filter = PoseFilter(
            max_translation_per_frame=0.3,
            max_rotation_per_frame=0.3,
            smoothing_alpha=pose_smoothing,
            max_distance_from_origin=30.0,
        )
        
        # State - pose is in WORLD frame (Z-up)
        self._pose_world = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Previous frame data
        self._prev_left = None
        self._prev_right = None
        self._prev_points_left = None
        self._prev_points_3d = None
        self._prev_time = None
        
        # Feature detector
        self._orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
        )
        
        # Optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Matcher
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self._frame_count = 0
    
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float, baseline: float = None):
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
        """Set intrinsics from dictionary."""
        self.set_intrinsics(
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy'],
            baseline=intrinsics.get('baseline', self.baseline),
        )
    
    def _triangulate_stereo(
        self,
        points_left: np.ndarray,
        points_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Triangulate 3D points from stereo correspondences."""
        disparity = points_left[:, 0] - points_right[:, 0]
        valid = disparity > 1.0
        
        Z = np.zeros(len(disparity))
        Z[valid] = (self._fx * self.baseline) / disparity[valid]
        
        # Filter by depth range (0.2m to 8m)
        valid &= (Z > 0.2) & (Z < 8.0)
        
        X = (points_left[:, 0] - self._cx) * Z / self._fx
        Y = (points_left[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        return points_3d, valid
    
    def _match_stereo_features(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and match features between stereo pair."""
        kp_left, desc_left = self._orb.detectAndCompute(left_gray, None)
        kp_right, desc_right = self._orb.detectAndCompute(right_gray, None)
        
        if desc_left is None or desc_right is None or len(kp_left) < 10:
            return np.array([]), np.array([]), np.array([])
        
        matches = self._bf_matcher.match(desc_left, desc_right)
        
        if len(matches) < 10:
            return np.array([]), np.array([]), np.array([])
        
        # Filter by epipolar constraint
        good_matches = []
        for m in matches:
            pt_l = kp_left[m.queryIdx].pt
            pt_r = kp_right[m.trainIdx].pt
            
            if abs(pt_l[1] - pt_r[1]) < 2.0 and pt_l[0] > pt_r[0]:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            return np.array([]), np.array([]), np.array([])
        
        points_left = np.array([kp_left[m.queryIdx].pt for m in good_matches])
        points_right = np.array([kp_right[m.trainIdx].pt for m in good_matches])
        
        points_3d, valid = self._triangulate_stereo(points_left, points_right)
        
        return points_left[valid], points_right[valid], points_3d[valid]
    
    def _estimate_pose_pnp(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Estimate pose using PnP with RANSAC."""
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
    
    def _camera_to_world_pose(self, pose_camera: np.ndarray) -> np.ndarray:
        """
        Convert pose from camera frame to world frame.
        
        Camera frame: X-right, Y-down, Z-forward (OpenCV convention)
        World frame: X-right, Y-forward, Z-up (after gravity alignment)
        """
        # Build 4x4 rotation matrix for frame conversion
        R_cam_to_world_4x4 = np.eye(4)
        R_cam_to_world_4x4[:3, :3] = self._R_cam_to_world
        
        # Apply frame transformation
        pose_world = R_cam_to_world_4x4 @ pose_camera
        
        return pose_world
    
    def process(
        self,
        left_ir: np.ndarray,
        right_ir: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> StereoVIOResult:
        """Process stereo frame pair with optional IMU data."""
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        self._frame_count += 1
        
        # Ensure grayscale
        if len(left_ir.shape) == 3:
            left_gray = cv2.cvtColor(left_ir, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_ir
            
        if len(right_ir.shape) == 3:
            right_gray = cv2.cvtColor(right_ir, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_ir
        
        # Compute dt
        dt = 0.033
        if self._prev_time is not None:
            dt = max(0.001, min(0.5, timestamp - self._prev_time))
        
        # Gravity calibration phase
        if accel is not None and not self._gravity_aligner.is_calibrated():
            calibrated = self._gravity_aligner.add_sample(accel)
            
            if calibrated:
                # Set up camera-to-world rotation
                # RealSense IMU is aligned with camera: X-right, Y-down, Z-forward
                # We want: X-right, Y-forward, Z-up
                # First apply gravity alignment, then coordinate swap
                R_gravity = self._gravity_aligner.get_world_rotation()
                
                # OpenCV camera to OpenGL-style world (Y-up becomes Z-up)
                R_cv_to_gl = np.array([
                    [1,  0,  0],
                    [0,  0, -1],  # Y -> -Z
                    [0,  1,  0],  # Z -> Y
                ])
                
                self._R_cam_to_world = R_cv_to_gl @ R_gravity
                print("Gravity calibration complete - world frame established")
            else:
                return StereoVIOResult(
                    pose=self._pose_world.copy(),
                    velocity=self._velocity.copy(),
                    num_inliers=0,
                    tracking_status="calibrating",
                    confidence=len(self._gravity_aligner._samples) / self._gravity_aligner._num_samples,
                )
        
        # First frame after calibration - initialize
        if not self._initialized:
            pts_left, pts_right, pts_3d = self._match_stereo_features(left_gray, right_gray)
            
            if len(pts_left) >= 10:
                self._prev_left = left_gray
                self._prev_right = right_gray
                self._prev_points_left = pts_left
                self._prev_points_3d = pts_3d
                self._prev_time = timestamp
                self._initialized = True
                
                return StereoVIOResult(
                    pose=self._pose_world.copy(),
                    velocity=self._velocity.copy(),
                    num_inliers=len(pts_left),
                    tracking_status="initializing",
                    confidence=0.5,
                )
            else:
                return StereoVIOResult(
                    pose=self._pose_world.copy(),
                    velocity=self._velocity.copy(),
                    num_inliers=0,
                    tracking_status="initializing",
                    confidence=0.0,
                )
        
        # Re-detect if too few points
        if len(self._prev_points_left) < 20:
            pts_left, pts_right, pts_3d = self._match_stereo_features(
                self._prev_left, self._prev_right
            )
            if len(pts_left) >= 10:
                self._prev_points_left = pts_left
                self._prev_points_3d = pts_3d
        
        if len(self._prev_points_left) < 6:
            # Lost tracking
            pts_left, pts_right, pts_3d = self._match_stereo_features(left_gray, right_gray)
            
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_points_left = pts_left if len(pts_left) > 0 else np.array([])
            self._prev_points_3d = pts_3d if len(pts_3d) > 0 else np.array([])
            self._prev_time = timestamp
            
            return StereoVIOResult(
                pose=self._pose_world.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Track with optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_left,
            left_gray,
            self._prev_points_left.reshape(-1, 1, 2).astype(np.float32),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_time = timestamp
            return StereoVIOResult(
                pose=self._pose_world.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Filter tracked points
        status = status.flatten().astype(bool)
        prev_pts_3d = self._prev_points_3d[status]
        curr_pts_2d = curr_points[status].reshape(-1, 2)
        
        h, w = left_gray.shape
        valid = (
            (curr_pts_2d[:, 0] >= 0) & (curr_pts_2d[:, 0] < w) &
            (curr_pts_2d[:, 1] >= 0) & (curr_pts_2d[:, 1] < h)
        )
        prev_pts_3d = prev_pts_3d[valid]
        curr_pts_2d = curr_pts_2d[valid]
        
        if len(prev_pts_3d) < 6:
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_time = timestamp
            return StereoVIOResult(
                pose=self._pose_world.copy(),
                velocity=self._velocity.copy(),
                num_inliers=len(prev_pts_3d),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Estimate pose with PnP (in camera frame)
        T_curr_prev_cam, num_inliers = self._estimate_pose_pnp(prev_pts_3d, curr_pts_2d)
        
        if T_curr_prev_cam is None or num_inliers < 4:
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_time = timestamp
            return StereoVIOResult(
                pose=self._pose_world.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Update pose in camera frame
        T_prev_curr_cam = np.linalg.inv(T_curr_prev_cam)
        
        # Check velocity (reject if too fast)
        translation = T_prev_curr_cam[:3, 3]
        velocity = np.linalg.norm(translation) / dt
        
        if velocity > self.max_velocity:
            # Reject as outlier
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_time = timestamp
            return StereoVIOResult(
                pose=self._pose_world.copy(),
                velocity=self._velocity.copy(),
                num_inliers=num_inliers,
                tracking_status="ok",
                confidence=0.3,  # Low confidence due to velocity rejection
            )
        
        # Accumulate pose in camera frame, then convert to world
        # First, get current camera pose
        pose_cam = self._world_to_camera_pose(self._pose_world)
        
        # Apply relative motion
        new_pose_cam = pose_cam @ T_prev_curr_cam
        
        # Convert back to world frame
        new_pose_world = self._camera_to_world_pose(new_pose_cam)
        
        # Apply pose filter
        filtered_pose, accepted = self._pose_filter.filter(new_pose_world)
        
        if accepted:
            self._pose_world = filtered_pose
            self._velocity = translation / dt
        
        confidence = min(1.0, num_inliers / 50.0) * (0.9 if accepted else 0.5)
        
        # Update for next frame
        pts_left_new, pts_right_new, pts_3d_new = self._match_stereo_features(
            left_gray, right_gray
        )
        
        self._prev_left = left_gray
        self._prev_right = right_gray
        self._prev_time = timestamp
        
        if len(pts_left_new) >= 10:
            self._prev_points_left = pts_left_new
            self._prev_points_3d = pts_3d_new
        else:
            self._prev_points_left = curr_pts_2d
            self._prev_points_3d = prev_pts_3d
        
        return StereoVIOResult(
            pose=self._pose_world.copy(),
            velocity=self._velocity.copy(),
            num_inliers=num_inliers,
            tracking_status="ok",
            confidence=confidence,
        )
    
    def _world_to_camera_pose(self, pose_world: np.ndarray) -> np.ndarray:
        """Convert pose from world frame back to camera frame."""
        R_world_to_cam = self._R_cam_to_world.T
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R_world_to_cam
        return R_4x4 @ pose_world
    
    def get_pose(self) -> np.ndarray:
        """Get current pose in world frame (Z-up)."""
        return self._pose_world.copy()
    
    def get_pose_opengl(self) -> np.ndarray:
        """
        Get pose in OpenGL convention (Y-up).
        
        Our world frame is Z-up, OpenGL wants Y-up.
        """
        # Transform Z-up to Y-up
        R_zup_to_yup = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],  # Y <- Z
            [0, -1, 0, 0],  # Z <- -Y
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        return R_zup_to_yup @ self._pose_world
    
    def reset(self):
        """Reset VIO state."""
        self._pose_world = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._prev_left = None
        self._prev_right = None
        self._prev_points_left = None
        self._prev_points_3d = None
        self._prev_time = None
        self._frame_count = 0
        self._gravity_aligner = GravityAligner()
        self._pose_filter.reset()
    
    def is_calibrated(self) -> bool:
        """Check if gravity calibration is complete."""
        return self._gravity_aligner.is_calibrated()
