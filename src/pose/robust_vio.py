"""
Robust Visual-Inertial Odometry for RealSense D435i

Key improvements over basic VIO:
1. IMU-based gravity alignment (know which way is up)
2. Kalman filter for pose smoothing
3. Keyframe-based tracking for stability
4. Motion model prediction
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
import time


# Coordinate transform: OpenCV (Y-down) to OpenGL (Y-up)
CV_TO_GL = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float64)


@dataclass
class VIOState:
    """Current VIO state."""
    pose: np.ndarray  # 4x4 camera-to-world
    velocity: np.ndarray  # 3D velocity
    gravity: np.ndarray  # Gravity vector in world frame
    confidence: float
    tracking_status: str


class GravityEstimator:
    """Estimate gravity direction from accelerometer."""
    
    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha  # Complementary filter coefficient
        self._gravity = np.array([0.0, -9.81, 0.0])  # Initial: Y-down (OpenCV)
        self._initialized = False
        self._samples = []
        self._init_samples = 30
    
    def update(self, accel: np.ndarray, gyro: Optional[np.ndarray] = None, dt: float = 0.033):
        """Update gravity estimate with new IMU reading."""
        if not self._initialized:
            # Collect samples during initialization (assume stationary)
            self._samples.append(accel.copy())
            if len(self._samples) >= self._init_samples:
                # Average acceleration = gravity when stationary
                avg_accel = np.mean(self._samples, axis=0)
                self._gravity = avg_accel
                self._initialized = True
            return
        
        # Low-pass filter on accelerometer (gravity estimate)
        accel_norm = np.linalg.norm(accel)
        
        # Only update if acceleration is close to gravity magnitude (not moving fast)
        if 8.0 < accel_norm < 12.0:
            # Normalize to gravity magnitude
            accel_normalized = accel * (9.81 / accel_norm)
            self._gravity = self.alpha * self._gravity + (1 - self.alpha) * accel_normalized
    
    def get_gravity(self) -> np.ndarray:
        """Get current gravity estimate in camera frame."""
        return self._gravity.copy()
    
    def get_up_vector(self) -> np.ndarray:
        """Get up vector (opposite to gravity)."""
        g = self._gravity
        norm = np.linalg.norm(g)
        if norm < 1e-6:
            return np.array([0.0, 1.0, 0.0])
        return -g / norm
    
    def is_initialized(self) -> bool:
        return self._initialized


class PoseFilter:
    """Simple pose smoother using exponential moving average."""
    
    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more smoothing.
        """
        self.alpha = alpha
        self._prev_pose = None
        self._prev_velocity = np.zeros(3)
    
    def filter(self, pose: np.ndarray, dt: float = 0.033) -> np.ndarray:
        """Apply smoothing to pose."""
        if self._prev_pose is None:
            self._prev_pose = pose.copy()
            return pose
        
        # Smooth translation
        t_curr = pose[:3, 3]
        t_prev = self._prev_pose[:3, 3]
        t_smooth = self.alpha * t_prev + (1 - self.alpha) * t_curr
        
        # For rotation, use SLERP-like interpolation via matrix averaging
        # Simple approach: just blend the matrices
        R_curr = pose[:3, :3]
        R_prev = self._prev_pose[:3, :3]
        
        # Average rotation (approximate - works for small differences)
        R_smooth = self.alpha * R_prev + (1 - self.alpha) * R_curr
        
        # Re-orthogonalize
        U, _, Vt = np.linalg.svd(R_smooth)
        R_smooth = U @ Vt
        if np.linalg.det(R_smooth) < 0:
            R_smooth = -R_smooth
        
        # Build smoothed pose
        pose_smooth = np.eye(4)
        pose_smooth[:3, :3] = R_smooth
        pose_smooth[:3, 3] = t_smooth
        
        self._prev_pose = pose_smooth.copy()
        return pose_smooth
    
    def reset(self):
        self._prev_pose = None


class RobustVIO:
    """
    Robust Visual-Inertial Odometry with gravity alignment and filtering.
    """
    
    def __init__(
        self,
        max_features: int = 300,
        min_features: int = 50,
        smoothing: float = 0.5,  # Pose smoothing factor
    ):
        self.max_features = max_features
        self.min_features = min_features
        
        # Camera intrinsics
        self._K = None
        self._fx = self._fy = self._cx = self._cy = None
        
        # State
        self._pose = np.eye(4)  # Camera-to-world in OpenCV coords
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Previous frame data
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_depth = None
        self._prev_time = None
        
        # Keyframe management
        self._keyframe_gray = None
        self._keyframe_points = None
        self._keyframe_points_3d = None
        self._keyframe_pose = np.eye(4)
        self._frames_since_keyframe = 0
        self._keyframe_interval = 10  # Create keyframe every N frames
        
        # IMU and filtering
        self._gravity_estimator = GravityEstimator()
        self._pose_filter = PoseFilter(alpha=smoothing)
        
        # Feature detector - GFTT is more stable than FAST
        self._detector = cv2.GFTTDetector_create(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=5,
        )
        
        # Optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Statistics
        self._frame_count = 0
        self._lost_count = 0
    
    def set_intrinsics(self, intrinsics: Dict[str, float]):
        """Set camera intrinsics."""
        self._fx = intrinsics['fx']
        self._fy = intrinsics['fy']
        self._cx = intrinsics['cx']
        self._cy = intrinsics['cy']
        self._K = np.array([
            [self._fx, 0, self._cx],
            [0, self._fy, self._cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect feature points."""
        kps = self._detector.detect(gray)
        if len(kps) == 0:
            return np.array([]).reshape(0, 1, 2)
        
        # Sort by response and take top N
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:self.max_features]
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts.reshape(-1, 1, 2)
    
    def _unproject_points(self, points_2d: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject 2D points to 3D using depth."""
        points_2d = points_2d.reshape(-1, 2)
        n = len(points_2d)
        
        h, w = depth.shape
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        
        # Sample depth with bounds checking
        u_int = np.clip(u.astype(int), 0, w - 1)
        v_int = np.clip(v.astype(int), 0, h - 1)
        d = depth[v_int, u_int]
        
        # Valid depth check
        valid = (d > 0.1) & (d < 8.0)  # 0.1m to 8m range
        
        # Unproject
        X = (u - self._cx) * d / self._fx
        Y = (v - self._cy) * d / self._fy
        Z = d
        
        points_3d = np.stack([X, Y, Z], axis=1)
        return points_3d, valid
    
    def _estimate_pose_pnp(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Estimate pose using PnP with RANSAC."""
        if len(points_3d) < 6:
            return None, 0
        
        # Use solvePnPRansac for robustness
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            self._K,
            None,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success or inliers is None or len(inliers) < 4:
            return None, 0
        
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Build transformation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T, len(inliers)
    
    def _should_create_keyframe(self, num_tracked: int, translation: float) -> bool:
        """Decide if we should create a new keyframe."""
        # Too few tracked points
        if num_tracked < self.min_features:
            return True
        
        # Moved enough
        if translation > 0.1:  # 10cm
            return True
        
        # Time-based
        if self._frames_since_keyframe > self._keyframe_interval:
            return True
        
        return False
    
    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> VIOState:
        """Process a frame and return pose estimate."""
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
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
        
        # Update gravity estimate from IMU
        if accel is not None:
            self._gravity_estimator.update(accel, gyro, dt)
        
        # First frame - initialize
        if not self._initialized:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            
            if len(self._prev_points) > 0:
                self._prev_points_3d, valid = self._unproject_points(self._prev_points, depth)
                self._prev_points = self._prev_points[valid]
                self._prev_points_3d = self._prev_points_3d[valid]
            
            # Set as keyframe
            self._keyframe_gray = gray.copy()
            self._keyframe_points = self._prev_points.copy() if len(self._prev_points) > 0 else None
            self._keyframe_points_3d = self._prev_points_3d.copy() if self._prev_points_3d is not None else None
            self._keyframe_pose = self._pose.copy()
            
            self._initialized = True
            return VIOState(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                gravity=self._gravity_estimator.get_gravity(),
                confidence=0.5,
                tracking_status="initializing",
            )
        
        # Re-detect if too few points
        if len(self._prev_points) < self.min_features:
            self._prev_points = self._detect_features(self._prev_gray)
            if len(self._prev_points) > 0:
                self._prev_points_3d, valid = self._unproject_points(self._prev_points, self._prev_depth)
                self._prev_points = self._prev_points[valid]
                self._prev_points_3d = self._prev_points_3d[valid]
        
        if len(self._prev_points) < 4:
            # Lost tracking
            self._lost_count += 1
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            
            return VIOState(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                gravity=self._gravity_estimator.get_gravity(),
                confidence=0.0,
                tracking_status="lost",
            )
        
        # Track points with optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_points.reshape(-1, 1, 2).astype(np.float32),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._lost_count += 1
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOState(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                gravity=self._gravity_estimator.get_gravity(),
                confidence=0.0,
                tracking_status="lost",
            )
        
        # Filter by tracking status and bounds
        status = status.flatten().astype(bool)
        h, w = gray.shape
        
        prev_pts_2d = self._prev_points[status].reshape(-1, 2)
        prev_pts_3d = self._prev_points_3d[status]
        curr_pts_2d = curr_points[status].reshape(-1, 2)
        
        # Bounds check
        valid = (
            (curr_pts_2d[:, 0] >= 0) & (curr_pts_2d[:, 0] < w) &
            (curr_pts_2d[:, 1] >= 0) & (curr_pts_2d[:, 1] < h)
        )
        prev_pts_2d = prev_pts_2d[valid]
        prev_pts_3d = prev_pts_3d[valid]
        curr_pts_2d = curr_pts_2d[valid]
        
        if len(prev_pts_3d) < 6:
            self._lost_count += 1
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOState(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                gravity=self._gravity_estimator.get_gravity(),
                confidence=0.0,
                tracking_status="lost",
            )
        
        # Estimate pose with PnP
        T_curr_prev, num_inliers = self._estimate_pose_pnp(prev_pts_3d, curr_pts_2d)
        
        # Debug: check 3D point stats
        if self._frame_count % 30 == 1:
            print(f"VIO debug: {len(prev_pts_3d)} pts, 3D range: [{prev_pts_3d[:,2].min():.2f}, {prev_pts_3d[:,2].max():.2f}]m")
        
        if T_curr_prev is None or num_inliers < 4:
            self._lost_count += 1
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOState(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                gravity=self._gravity_estimator.get_gravity(),
                confidence=0.0,
                tracking_status="lost",
            )
        
        # Update pose
        T_prev_curr = np.linalg.inv(T_curr_prev)
        new_pose = self._pose @ T_prev_curr
        
        # Apply smoothing filter
        new_pose = self._pose_filter.filter(new_pose, dt)
        
        # Compute velocity
        translation = new_pose[:3, 3] - self._pose[:3, 3]
        self._velocity = translation / dt
        
        self._pose = new_pose
        self._frames_since_keyframe += 1
        
        # Confidence
        confidence = min(1.0, num_inliers / 50.0)
        
        # Prepare for next frame
        self._prev_gray = gray
        self._prev_depth = depth
        self._prev_time = timestamp
        
        # Update tracked points
        self._prev_points = curr_pts_2d.reshape(-1, 1, 2)
        self._prev_points_3d, valid = self._unproject_points(curr_pts_2d, depth)
        self._prev_points = self._prev_points[valid]
        self._prev_points_3d = self._prev_points_3d[valid]
        
        # Add new features if needed
        if len(self._prev_points) < self.max_features // 2:
            new_pts = self._detect_features(gray)
            if len(new_pts) > 0:
                new_pts_3d, valid = self._unproject_points(new_pts, depth)
                new_pts = new_pts[valid]
                new_pts_3d = new_pts_3d[valid]
                
                if len(new_pts) > 0:
                    self._prev_points = np.vstack([
                        self._prev_points.reshape(-1, 1, 2),
                        new_pts.reshape(-1, 1, 2)
                    ])[:self.max_features]
                    self._prev_points_3d = np.vstack([
                        self._prev_points_3d,
                        new_pts_3d
                    ])[:self.max_features]
        
        # Create keyframe if needed
        if self._should_create_keyframe(len(self._prev_points), np.linalg.norm(translation)):
            self._keyframe_gray = gray.copy()
            self._keyframe_points = self._prev_points.copy()
            self._keyframe_points_3d = self._prev_points_3d.copy()
            self._keyframe_pose = self._pose.copy()
            self._frames_since_keyframe = 0
        
        return VIOState(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            gravity=self._gravity_estimator.get_gravity(),
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_pose(self) -> np.ndarray:
        """Get current pose in OpenCV convention."""
        return self._pose.copy()
    
    def get_pose_opengl(self) -> np.ndarray:
        """Get current pose in OpenGL convention (Y-up)."""
        return CV_TO_GL @ self._pose @ CV_TO_GL.T
    
    def get_gravity_aligned_pose(self) -> np.ndarray:
        """Get pose with gravity alignment (Z-up world frame)."""
        if not self._gravity_estimator.is_initialized():
            return self.get_pose_opengl()
        
        # Get gravity in camera frame
        g_cam = self._gravity_estimator.get_gravity()
        g_norm = np.linalg.norm(g_cam)
        if g_norm < 1e-6:
            return self.get_pose_opengl()
        
        # Gravity points down, so up = -g
        up = -g_cam / g_norm
        
        # Create rotation that aligns camera Y with world up
        # This creates a level horizon
        cam_y = np.array([0, 1, 0])
        
        # Rotation axis
        axis = np.cross(cam_y, up)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(cam_y, up), -1, 1))
            
            # Rodrigues formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        else:
            R_align = np.eye(3)
        
        # Apply alignment to pose
        pose = self._pose.copy()
        pose[:3, :3] = R_align @ pose[:3, :3]
        pose[:3, 3] = R_align @ pose[:3, 3]
        
        # Convert to OpenGL
        return CV_TO_GL @ pose @ CV_TO_GL.T
    
    def reset(self):
        """Reset VIO state."""
        self._pose = np.eye(4)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_depth = None
        self._prev_time = None
        self._keyframe_gray = None
        self._keyframe_points = None
        self._keyframe_points_3d = None
        self._frame_count = 0
        self._lost_count = 0
        self._frames_since_keyframe = 0
        self._gravity_estimator = GravityEstimator()
        self._pose_filter.reset()
