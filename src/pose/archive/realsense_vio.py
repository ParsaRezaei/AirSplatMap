"""
Visual-Inertial-Depth Odometry for RealSense D435i

Uses:
- Visual features for rotation estimation
- Metric depth for absolute scale
- IMU for orientation and gravity alignment

Coordinate systems:
- RealSense/OpenCV: X-right, Y-down, Z-forward
- OpenGL/3DGS: X-right, Y-up, Z-backward
- We output OpenGL convention for 3DGS compatibility
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# Transform from OpenCV to OpenGL coordinates
# OpenCV: X-right, Y-down, Z-forward
# OpenGL: X-right, Y-up, Z-backward  
CV_TO_GL = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float64)


@dataclass
class VIOResult:
    """Result from visual-inertial odometry."""
    pose: np.ndarray  # 4x4 camera-to-world
    velocity: np.ndarray  # 3D velocity in world frame
    confidence: float
    num_inliers: int
    tracking_status: str  # "ok", "lost", "initializing"


class RealSenseVIO:
    """
    Visual-Inertial-Depth Odometry for RealSense cameras.
    
    Key insight: We have METRIC DEPTH, so we can get absolute scale!
    The essential matrix only gives relative pose up to scale, but
    by using depth we can recover the true scale.
    """
    
    def __init__(
        self,
        use_imu: bool = True,
        feature_type: str = 'gftt',  # 'gftt', 'orb', 'fast'
        max_features: int = 500,
        min_features: int = 50,
    ):
        self.use_imu = use_imu
        self.feature_type = feature_type
        self.max_features = max_features
        self.min_features = min_features
        
        # Camera intrinsics
        self._K = None
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        
        # State
        self._pose = np.eye(4)  # Current camera-to-world pose
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Previous frame data
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_depth = None
        self._prev_time = None
        
        # IMU state
        self._gravity = np.array([0, 9.81, 0])  # Initial guess (Y-down)
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        
        # Feature detector - FAST is fastest
        if feature_type == 'fast':
            self._detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        elif feature_type == 'gftt':
            self._detector = cv2.GFTTDetector_create(
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=15,  # Larger min distance = fewer points = faster
                blockSize=3,
            )
        else:
            self._detector = cv2.ORB_create(nfeatures=max_features)
        
        # Optical flow parameters - smaller window = faster
        self._lk_params = dict(
            winSize=(15, 15),  # Reduced from 21
            maxLevel=2,  # Reduced from 3
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
    
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
        if self.feature_type in ('gftt', 'fast'):
            kps = self._detector.detect(gray)
            if len(kps) == 0:
                return np.array([]).reshape(0, 1, 2)
            pts = np.array([kp.pt for kp in kps], dtype=np.float32)
            return pts.reshape(-1, 1, 2)
        else:
            kps = self._detector.detect(gray)
            if len(kps) == 0:
                return np.array([]).reshape(0, 1, 2)
            pts = np.array([kp.pt for kp in kps], dtype=np.float32)
            return pts.reshape(-1, 1, 2)
    
    def _unproject_points(self, points_2d: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unproject 2D points to 3D using depth.
        
        Returns:
            points_3d: Nx3 array of 3D points in camera frame
            valid_mask: N boolean array of valid points
        """
        points_2d = points_2d.reshape(-1, 2)
        n = len(points_2d)
        
        # Get depth at each point (bilinear interpolation)
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        
        # Clamp to image bounds
        h, w = depth.shape
        u_int = np.clip(u.astype(int), 0, w - 1)
        v_int = np.clip(v.astype(int), 0, h - 1)
        
        # Sample depth
        d = depth[v_int, u_int]
        
        # Valid if depth > 0 and < max range
        valid = (d > 0.1) & (d < 10.0)
        
        # Unproject: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        X = (u - self._cx) * d / self._fx
        Y = (v - self._cy) * d / self._fy
        Z = d
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return points_3d, valid
    
    def _estimate_pose_3d_2d(
        self, 
        points_3d: np.ndarray, 
        points_2d: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate pose using 3D-2D correspondences (PnP).
        
        Args:
            points_3d: Nx3 3D points in previous camera frame
            points_2d: Nx2 2D points in current image
            
        Returns:
            T_curr_prev: 4x4 transformation from previous to current frame
            num_inliers: Number of inliers
        """
        if len(points_3d) < 4:
            return None, 0
        
        # Use PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            self._K,
            None,  # No distortion
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success or inliers is None:
            return None, 0
        
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T, len(inliers)
    
    def _integrate_imu(
        self,
        accel: Optional[np.ndarray],
        gyro: Optional[np.ndarray],
        dt: float
    ) -> np.ndarray:
        """
        Integrate IMU to get relative rotation.
        
        Returns:
            R_imu: 3x3 rotation from IMU integration
        """
        if gyro is None or dt <= 0:
            return np.eye(3)
        
        # Remove bias
        omega = gyro - self._gyro_bias
        
        # Small angle approximation for short dt
        angle = np.linalg.norm(omega) * dt
        if angle < 1e-6:
            return np.eye(3)
        
        # Rodrigues formula
        axis = omega / np.linalg.norm(omega)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        return R
    
    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> VIOResult:
        """
        Process a frame and update pose estimate.
        
        Args:
            rgb: HxWx3 RGB image (uint8)
            depth: HxW depth in meters (float32)
            timestamp: Frame timestamp in seconds
            accel: 3D accelerometer reading (m/s^2)
            gyro: 3D gyroscope reading (rad/s)
            
        Returns:
            VIOResult with updated pose
        """
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        # Convert to grayscale
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        # First frame - initialize
        if not self._initialized:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            
            if len(self._prev_points) > 0:
                self._prev_points_3d, valid = self._unproject_points(
                    self._prev_points, depth
                )
                # Filter to valid only
                self._prev_points = self._prev_points[valid]
                self._prev_points_3d = self._prev_points_3d[valid]
            
            self._initialized = True
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                confidence=0.5,
                num_inliers=len(self._prev_points),
                tracking_status="initializing",
            )
        
        # Compute dt
        dt = timestamp - self._prev_time if self._prev_time else 0.033
        dt = max(0.001, min(dt, 0.5))  # Clamp to reasonable range
        
        # Track features
        if len(self._prev_points) < self.min_features:
            # Re-detect features
            self._prev_points = self._detect_features(self._prev_gray)
            self._prev_points_3d, valid = self._unproject_points(
                self._prev_points, self._prev_depth
            )
            self._prev_points = self._prev_points[valid]
            self._prev_points_3d = self._prev_points_3d[valid]
        
        if len(self._prev_points) < 4:
            # Cannot track
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                confidence=0.0,
                num_inliers=0,
                tracking_status="lost",
            )
        
        # Track with optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_points.reshape(-1, 1, 2).astype(np.float32),
            None,
            **self._lk_params
        )
        
        if curr_points is None:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                confidence=0.0,
                num_inliers=0,
                tracking_status="lost",
            )
        
        # Filter by tracking status
        status = status.flatten().astype(bool)
        prev_pts_2d = self._prev_points[status].reshape(-1, 2)
        prev_pts_3d = self._prev_points_3d[status]
        curr_pts_2d = curr_points[status].reshape(-1, 2)
        
        # Filter by image bounds
        h, w = gray.shape
        valid = (
            (curr_pts_2d[:, 0] >= 0) & (curr_pts_2d[:, 0] < w) &
            (curr_pts_2d[:, 1] >= 0) & (curr_pts_2d[:, 1] < h)
        )
        prev_pts_2d = prev_pts_2d[valid]
        prev_pts_3d = prev_pts_3d[valid]
        curr_pts_2d = curr_pts_2d[valid]
        
        if len(prev_pts_3d) < 4:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                confidence=0.0,
                num_inliers=len(prev_pts_3d),
                tracking_status="lost",
            )
        
        # Estimate pose with PnP (3D points in prev frame, 2D in current)
        T_curr_prev, num_inliers = self._estimate_pose_3d_2d(prev_pts_3d, curr_pts_2d)
        
        if T_curr_prev is None:
            self._prev_gray = gray
            self._prev_depth = depth
            self._prev_time = timestamp
            self._prev_points = self._detect_features(gray)
            return VIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                confidence=0.0,
                num_inliers=0,
                tracking_status="lost",
            )
        
        # Update pose: T_world_curr = T_world_prev @ inv(T_curr_prev)
        # T_curr_prev transforms points from prev camera to current camera
        # We want world-to-camera, so: T_wc_curr = T_wc_prev @ T_prev_curr
        T_prev_curr = np.linalg.inv(T_curr_prev)
        self._pose = self._pose @ T_prev_curr
        
        # Compute velocity from translation
        translation = T_prev_curr[:3, 3]
        self._velocity = translation / dt
        
        # Confidence based on inliers
        confidence = min(1.0, num_inliers / 100.0)
        
        # Prepare for next frame
        self._prev_gray = gray
        self._prev_depth = depth
        self._prev_time = timestamp
        
        # Update tracked points (keep inliers, add new if needed)
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
                
                # Merge
                self._prev_points = np.vstack([
                    self._prev_points.reshape(-1, 1, 2),
                    new_pts.reshape(-1, 1, 2)
                ])[:self.max_features]
                self._prev_points_3d = np.vstack([
                    self._prev_points_3d,
                    new_pts_3d
                ])[:self.max_features]
        
        return VIOResult(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            confidence=confidence,
            num_inliers=num_inliers,
            tracking_status="ok",
        )
    
    def get_pose_opengl(self) -> np.ndarray:
        """Get pose in OpenGL convention (Y-up, Z-backward)."""
        return CV_TO_GL @ self._pose @ CV_TO_GL.T
    
    def reset(self):
        """Reset to initial state."""
        self._pose = np.eye(4)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._prev_gray = None
        self._prev_points = None
        self._prev_points_3d = None
        self._prev_depth = None
        self._prev_time = None
    
    def get_pose(self) -> np.ndarray:
        """Get current pose."""
        return self._pose.copy()
