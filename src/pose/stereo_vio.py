"""
Stereo Visual-Inertial Odometry for RealSense D435i

Uses both infrared cameras for stereo matching, which provides:
1. Better depth estimation than monocular
2. Scale-aware odometry from stereo baseline
3. More robust tracking with two viewpoints

The D435i has:
- Left IR camera
- Right IR camera  
- ~50mm baseline between them
- RGB camera (separate)
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass 
class StereoVIOResult:
    """Result from stereo VIO processing."""
    pose: np.ndarray  # 4x4 camera-to-world
    velocity: np.ndarray
    num_inliers: int
    tracking_status: str  # "ok", "lost", "initializing"


class StereoVIO:
    """
    Stereo Visual Odometry using RealSense D435i IR cameras.
    
    Uses OpenCV's stereo matching and feature tracking for
    robust pose estimation with proper scale.
    """
    
    def __init__(
        self,
        baseline: float = 0.05,  # 50mm default baseline
        max_features: int = 500,
        use_gpu: bool = False,
    ):
        self.baseline = baseline
        self.max_features = max_features
        self.use_gpu = use_gpu
        
        # Camera intrinsics (will be set from RealSense)
        self._K_left = None
        self._K_right = None
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        
        # State
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        
        # Previous frame data
        self._prev_left = None
        self._prev_right = None
        self._prev_points_left = None
        self._prev_points_3d = None
        self._prev_time = None
        
        # Feature detector - ORB is fast and works well with IR
        self._orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
        )
        
        # For stereo matching
        self._stereo_matcher = cv2.StereoBM_create(
            numDisparities=64,
            blockSize=15,
        )
        
        # Optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        # Matcher for stereo features
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self._frame_count = 0
    
    def set_intrinsics(
        self,
        fx: float,
        fy: float, 
        cx: float,
        cy: float,
        baseline: float = None,
    ):
        """Set camera intrinsics."""
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        
        self._K_left = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume same intrinsics for right camera (typical for RealSense)
        self._K_right = self._K_left.copy()
        
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
        """
        Triangulate 3D points from stereo correspondences.
        
        Returns:
            points_3d: Nx3 array of 3D points
            valid: N boolean array
        """
        # Disparity = x_left - x_right
        disparity = points_left[:, 0] - points_right[:, 0]
        
        # Filter invalid disparities
        valid = disparity > 1.0  # At least 1 pixel disparity
        
        # Depth from disparity: Z = f * B / d
        Z = np.zeros(len(disparity))
        Z[valid] = (self._fx * self.baseline) / disparity[valid]
        
        # Filter by depth range
        valid &= (Z > 0.1) & (Z < 10.0)
        
        # Back-project to 3D
        X = (points_left[:, 0] - self._cx) * Z / self._fx
        Y = (points_left[:, 1] - self._cy) * Z / self._fy
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        return points_3d, valid
    
    def _match_stereo_features(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect and match features between stereo pair.
        
        Returns:
            points_left: Nx2 array of left image points
            points_right: Nx2 array of corresponding right image points  
            points_3d: Nx3 array of triangulated 3D points
        """
        # Detect features in left image
        kp_left, desc_left = self._orb.detectAndCompute(left_gray, None)
        kp_right, desc_right = self._orb.detectAndCompute(right_gray, None)
        
        if desc_left is None or desc_right is None or len(kp_left) < 10:
            return np.array([]), np.array([]), np.array([])
        
        # Match features
        matches = self._bf_matcher.match(desc_left, desc_right)
        
        if len(matches) < 10:
            return np.array([]), np.array([]), np.array([])
        
        # Filter matches by epipolar constraint (y coordinates should be similar)
        good_matches = []
        for m in matches:
            pt_l = kp_left[m.queryIdx].pt
            pt_r = kp_right[m.trainIdx].pt
            
            # Epipolar constraint: same row (within tolerance)
            if abs(pt_l[1] - pt_r[1]) < 2.0:
                # Right point should be to the left of left point (positive disparity)
                if pt_l[0] > pt_r[0]:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return np.array([]), np.array([]), np.array([])
        
        # Extract matched points
        points_left = np.array([kp_left[m.queryIdx].pt for m in good_matches])
        points_right = np.array([kp_right[m.trainIdx].pt for m in good_matches])
        
        # Triangulate
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
            self._K_left,
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
    
    def process(
        self,
        left_ir: np.ndarray,
        right_ir: np.ndarray,
        timestamp: float,
        depth: Optional[np.ndarray] = None,
    ) -> StereoVIOResult:
        """
        Process stereo IR frame pair.
        
        Args:
            left_ir: Left IR camera image (grayscale)
            right_ir: Right IR camera image (grayscale)
            timestamp: Frame timestamp
            depth: Optional depth image (can use for validation)
        """
        if self._K_left is None:
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
        
        # First frame - initialize
        if not self._initialized:
            # Get stereo features
            pts_left, pts_right, pts_3d = self._match_stereo_features(left_gray, right_gray)
            
            if len(pts_left) >= 10:
                self._prev_left = left_gray
                self._prev_right = right_gray
                self._prev_points_left = pts_left
                self._prev_points_3d = pts_3d
                self._prev_time = timestamp
                self._initialized = True
                
                return StereoVIOResult(
                    pose=self._pose.copy(),
                    velocity=self._velocity.copy(),
                    num_inliers=len(pts_left),
                    tracking_status="initializing",
                )
            else:
                return StereoVIOResult(
                    pose=self._pose.copy(),
                    velocity=self._velocity.copy(),
                    num_inliers=0,
                    tracking_status="initializing",
                )
        
        # Track features from previous frame using optical flow
        if len(self._prev_points_left) < 20:
            # Re-detect stereo features
            pts_left, pts_right, pts_3d = self._match_stereo_features(
                self._prev_left, self._prev_right
            )
            if len(pts_left) >= 10:
                self._prev_points_left = pts_left
                self._prev_points_3d = pts_3d
        
        if len(self._prev_points_left) < 6:
            # Lost tracking, try to reinitialize
            pts_left, pts_right, pts_3d = self._match_stereo_features(left_gray, right_gray)
            
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_points_left = pts_left if len(pts_left) > 0 else np.array([])
            self._prev_points_3d = pts_3d if len(pts_3d) > 0 else np.array([])
            self._prev_time = timestamp
            
            return StereoVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
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
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
            )
        
        # Filter by status
        status = status.flatten().astype(bool)
        prev_pts_3d = self._prev_points_3d[status]
        curr_pts_2d = curr_points[status].reshape(-1, 2)
        
        # Bounds check
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
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=len(prev_pts_3d),
                tracking_status="lost",
            )
        
        # Estimate pose with PnP
        T_curr_prev, num_inliers = self._estimate_pose_pnp(prev_pts_3d, curr_pts_2d)
        
        if T_curr_prev is None or num_inliers < 4:
            self._prev_left = left_gray
            self._prev_right = right_gray
            self._prev_time = timestamp
            return StereoVIOResult(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                num_inliers=0,
                tracking_status="lost",
            )
        
        # Update pose
        T_prev_curr = np.linalg.inv(T_curr_prev)
        self._pose = self._pose @ T_prev_curr
        
        # Compute velocity
        translation = T_prev_curr[:3, 3]
        self._velocity = translation / dt
        
        # Update for next frame - get new stereo features
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
            # Keep tracking existing points
            self._prev_points_left = curr_pts_2d
            # Re-triangulate with current stereo
            # For simplicity, just invalidate and re-detect next frame
            self._prev_points_3d = prev_pts_3d
        
        return StereoVIOResult(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            num_inliers=num_inliers,
            tracking_status="ok",
        )
    
    def get_pose(self) -> np.ndarray:
        """Get current pose in camera convention."""
        return self._pose.copy()
    
    def get_pose_opengl(self) -> np.ndarray:
        """Get pose in OpenGL convention (Y-up, Z-backward)."""
        # Transform from OpenCV to OpenGL
        CV_TO_GL = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)
        return CV_TO_GL @ self._pose @ CV_TO_GL.T
    
    def reset(self):
        """Reset VIO state."""
        self._pose = np.eye(4, dtype=np.float64)
        self._velocity = np.zeros(3)
        self._initialized = False
        self._prev_left = None
        self._prev_right = None
        self._prev_points_left = None
        self._prev_points_3d = None
        self._prev_time = None
        self._frame_count = 0
