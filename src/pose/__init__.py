"""
Pose Estimation Module
======================

Provides modular pose estimation backends for AirSplatMap.

Available estimators:
- orb: Fast ORB-based visual odometry
- sift: SIFT-based with fundamental matrix
- robust_flow: Optical flow with robust estimation
- loftr: LoFTR learned feature matching (requires kornia)

Usage:
    from src.pose import get_pose_estimator
    
    estimator = get_pose_estimator('robust_flow')
    estimator.set_intrinsics_from_dict({'fx': 525, 'fy': 525, 'cx': 320, 'cy': 240})
    result = estimator.estimate(rgb_image)
    pose = result.pose  # 4x4 camera-to-world matrix
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass 
class PoseResult:
    """Result from pose estimation."""
    pose: np.ndarray  # 4x4 camera-to-world transformation
    confidence: float = 1.0  # Tracking confidence [0, 1]
    num_inliers: int = 0  # Number of inlier matches
    is_keyframe: bool = False
    tracking_status: str = "ok"  # "ok", "lost", "initializing"


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators."""
    
    def __init__(self):
        self._K = None  # 3x3 intrinsic matrix
        self._current_pose = np.eye(4)  # Current camera-to-world pose
        self._initialized = False
    
    def set_intrinsics(self, K: np.ndarray):
        """Set camera intrinsic matrix (3x3)."""
        self._K = K.astype(np.float64)
    
    def set_intrinsics_from_dict(self, intrinsics: Dict[str, float]):
        """Set intrinsics from dictionary with fx, fy, cx, cy."""
        self._K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        """
        Estimate pose from RGB image.
        
        Args:
            rgb: HxWx3 RGB image (uint8)
            
        Returns:
            PoseResult with estimated pose
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get estimator name."""
        pass
    
    def reset(self):
        """Reset estimator state."""
        self._current_pose = np.eye(4)
        self._initialized = False


class ORBPoseEstimator(BasePoseEstimator):
    """Fast pose estimation using ORB features."""
    
    def __init__(self, n_features: int = 2000, scale_factor: float = 1.2, n_levels: int = 8):
        super().__init__()
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect ORB features
        kp, desc = self._orb.detectAndCompute(gray, None)
        
        # First frame - just store features
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(kp) if kp else 0,
                tracking_status="initializing",
            )
        
        # Match features
        matches = self._bf.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:min(100, len(matches))]
        
        if len(good_matches) < 8:
            # Not enough matches - tracking lost
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_matches),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Extract matched points
        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        
        if E is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Recover pose
        num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
        
        # Build relative transformation (previous to current)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Update cumulative pose: T_world_curr = T_world_prev @ T_prev_curr
        # But we have T_curr_prev from recoverPose, so need inverse
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        # Compute confidence based on inliers
        confidence = min(1.0, num_inliers / 50.0)
        
        # Store for next frame
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "orb"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None


class RobustFlowEstimator(BasePoseEstimator):
    """
    Robust pose estimation using optical flow and fundamental matrix.
    
    Combines dense optical flow with feature points for robust tracking.
    """
    
    def __init__(self, grid_size: int = 20, use_farneback: bool = True):
        super().__init__()
        self._grid_size = grid_size
        self._use_farneback = use_farneback
        
        self._prev_gray = None
        self._prev_points = None
        
        # Lucas-Kanade parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    
    def _create_grid_points(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create grid of points for tracking."""
        h, w = shape
        step = self._grid_size
        
        x = np.arange(step, w - step, step)
        y = np.arange(step, h - step, step)
        xx, yy = np.meshgrid(x, y)
        
        points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
        return points.reshape(-1, 1, 2)
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # First frame
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(self._prev_points),
                tracking_status="initializing",
            )
        
        # Track points with Lucas-Kanade optical flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_points, None,
            **self._lk_params
        )
        
        if next_points is None:
            # Flow failed - reinitialize
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Filter by status
        status = status.flatten()
        good_prev = self._prev_points[status == 1].reshape(-1, 2)
        good_next = next_points[status == 1].reshape(-1, 2)
        
        if len(good_prev) < 8:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_prev),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Filter by bounds
        valid = (
            (good_next[:, 0] >= 0) & (good_next[:, 0] < w) &
            (good_next[:, 1] >= 0) & (good_next[:, 1] < h)
        )
        good_prev = good_prev[valid]
        good_next = good_next[valid]
        
        if len(good_prev) < 8:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_prev),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            good_prev, good_next, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.5,
        )
        
        if E is None or mask is None:
            self._prev_gray = gray
            self._prev_points = self._create_grid_points(gray.shape)
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Recover pose
        num_inliers, R, t, _ = cv2.recoverPose(E, good_prev, good_next, self._K, mask=mask)
        
        # Build relative transformation
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Update cumulative pose
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        # Compute confidence
        confidence = min(1.0, num_inliers / 100.0)
        
        # Prepare for next frame
        self._prev_gray = gray
        
        # Reinitialize points periodically or when too few tracked
        if len(good_next) < 100 or num_inliers < 30:
            self._prev_points = self._create_grid_points(gray.shape)
        else:
            # Keep tracked points
            self._prev_points = good_next.reshape(-1, 1, 2)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "robust_flow"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_points = None


class SIFTPoseEstimator(BasePoseEstimator):
    """Pose estimation using SIFT features (more robust but slower)."""
    
    def __init__(self, n_features: int = 1000):
        super().__init__()
        self._sift = cv2.SIFT_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set. Call set_intrinsics() first.")
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, None)
        
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(kp) if kp else 0,
                tracking_status="initializing",
            )
        
        # KNN matching with ratio test
        matches = self._bf.knnMatch(self._prev_desc, desc, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 8:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(good_matches),
                tracking_status="lost",
                confidence=0.0,
            )
        
        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        
        if E is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
        
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        confidence = min(1.0, num_inliers / 50.0)
        
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "sift"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None


class KeyframePoseEstimator(BasePoseEstimator):
    """
    Keyframe-based pose estimation.
    
    Maintains a set of keyframes and matches against them for more stable tracking.
    Good for loop closure and reducing drift.
    """
    
    def __init__(self, n_features: int = 1000, keyframe_threshold: float = 0.3, max_keyframes: int = 20):
        super().__init__()
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self._keyframe_threshold = keyframe_threshold  # Min motion to create keyframe
        self._max_keyframes = max_keyframes
        
        # Keyframe storage: list of (gray, keypoints, descriptors, pose)
        self._keyframes: List[Tuple[np.ndarray, Any, np.ndarray, np.ndarray]] = []
        
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
        self._last_keyframe_pose = None
    
    def _add_keyframe(self, gray: np.ndarray, kp, desc: np.ndarray, pose: np.ndarray):
        """Add a new keyframe."""
        self._keyframes.append((gray.copy(), kp, desc.copy(), pose.copy()))
        
        # Remove oldest if too many
        if len(self._keyframes) > self._max_keyframes:
            self._keyframes.pop(0)
        
        self._last_keyframe_pose = pose.copy()
    
    def _match_to_keyframe(self, desc: np.ndarray, kf_desc: np.ndarray) -> List:
        """Match descriptors to a keyframe."""
        if desc is None or kf_desc is None:
            return []
        
        matches = self._bf.knnMatch(kf_desc, desc, k=2)
        
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        return good
    
    def _should_create_keyframe(self, current_pose: np.ndarray) -> bool:
        """Check if we should create a new keyframe based on motion."""
        if self._last_keyframe_pose is None:
            return True
        
        # Compute relative motion
        rel = np.linalg.inv(self._last_keyframe_pose) @ current_pose
        translation = np.linalg.norm(rel[:3, 3])
        
        # Rotation angle
        R = rel[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        return translation > self._keyframe_threshold or angle > 0.3
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        
        # First frame
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._initialized = True
            
            if desc is not None and len(kp) >= 10:
                self._add_keyframe(gray, kp, desc, self._current_pose)
            
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(kp) if kp else 0,
                tracking_status="initializing",
                is_keyframe=True,
            )
        
        best_inliers = 0
        best_pose = None
        matched_keyframe = False
        
        # Try to match against keyframes (most recent first)
        for kf_gray, kf_kp, kf_desc, kf_pose in reversed(self._keyframes[-5:]):
            matches = self._match_to_keyframe(desc, kf_desc)
            
            if len(matches) < 15:
                continue
            
            pts1 = np.float32([kf_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
            
            E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is None:
                continue
            
            num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = t.flatten()
                best_pose = kf_pose @ np.linalg.inv(T_rel)
                matched_keyframe = True
        
        # Fall back to frame-to-frame tracking
        if best_pose is None:
            matches = self._bf.match(self._prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)[:100]
            
            if len(matches) >= 8:
                pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                
                E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)
                    best_inliers = num_inliers
                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = t.flatten()
                    best_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        if best_pose is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        self._current_pose = best_pose
        
        # Check if we should add a keyframe
        is_keyframe = self._should_create_keyframe(self._current_pose)
        if is_keyframe:
            self._add_keyframe(gray, kp, desc, self._current_pose)
        
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        confidence = min(1.0, best_inliers / 50.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=best_inliers,
            confidence=confidence,
            tracking_status="ok",
            is_keyframe=is_keyframe,
        )
    
    def get_name(self) -> str:
        return "keyframe"
    
    def reset(self):
        super().reset()
        self._keyframes.clear()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
        self._last_keyframe_pose = None


class LoFTRPoseEstimator(BasePoseEstimator):
    """
    Pose estimation using LoFTR (Local Feature TRansformer).
    
    Deep learning based dense feature matching. Requires kornia library.
    Very robust but slower, best with GPU.
    """
    
    def __init__(self, pretrained: str = 'outdoor', device: str = 'cuda'):
        super().__init__()
        self._device = device
        self._pretrained = pretrained
        self._matcher = None
        self._prev_gray = None
        self._prev_tensor = None
        
        # Try to initialize LoFTR
        self._available = False
        try:
            import torch
            import kornia
            from kornia.feature import LoFTR
            
            self._torch = torch
            self._kornia = kornia
            
            # Check if GPU available
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA not available for LoFTR, falling back to CPU")
                self._device = 'cpu'
            
            self._matcher = LoFTR(pretrained=pretrained).to(self._device).eval()
            self._available = True
            logger.info(f"LoFTR initialized on {self._device}")
        except ImportError as e:
            logger.warning(f"LoFTR not available: {e}. Install with: pip install kornia")
        except Exception as e:
            logger.warning(f"LoFTR initialization failed: {e}")
    
    def _to_tensor(self, gray: np.ndarray):
        """Convert grayscale image to tensor."""
        import torch
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        return tensor.to(self._device)
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        if not self._available:
            # Fall back to ORB if LoFTR not available
            logger.warning("LoFTR not available, returning identity pose")
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Resize for efficiency (LoFTR works best at lower res)
        h, w = gray.shape
        scale = min(640 / w, 480 / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            gray_resized = cv2.resize(gray, (new_w, new_h))
        else:
            gray_resized = gray
            scale = 1.0
        
        curr_tensor = self._to_tensor(gray_resized)
        
        # First frame
        if self._prev_tensor is None:
            self._prev_gray = gray
            self._prev_tensor = curr_tensor
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="initializing",
            )
        
        # Match with LoFTR
        with self._torch.no_grad():
            input_dict = {
                'image0': self._prev_tensor,
                'image1': curr_tensor,
            }
            correspondences = self._matcher(input_dict)
        
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        conf = correspondences['confidence'].cpu().numpy()
        
        # Filter by confidence
        valid = conf > 0.5
        mkpts0 = mkpts0[valid]
        mkpts1 = mkpts1[valid]
        
        if len(mkpts0) < 8:
            self._prev_gray = gray
            self._prev_tensor = curr_tensor
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(mkpts0),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Scale points back to original resolution
        mkpts0 = mkpts0 / scale
        mkpts1 = mkpts1 / scale
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            self._prev_gray = gray
            self._prev_tensor = curr_tensor
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        num_inliers, R, t, _ = cv2.recoverPose(E, mkpts0, mkpts1, self._K, mask=mask)
        
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        self._prev_gray = gray
        self._prev_tensor = curr_tensor
        
        confidence = min(1.0, num_inliers / 100.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "loftr"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_tensor = None


class SuperPointPoseEstimator(BasePoseEstimator):
    """
    Pose estimation using SuperPoint + SuperGlue.
    
    Deep learning based sparse feature detection and matching.
    Very robust to lighting changes. Requires kornia or torch.
    """
    
    def __init__(self, device: str = 'cuda', max_keypoints: int = 1024):
        super().__init__()
        self._device = device
        self._max_keypoints = max_keypoints
        self._extractor = None
        self._matcher = None
        self._available = False
        
        self._prev_gray = None
        self._prev_feats = None
        
        try:
            import torch
            import kornia
            from kornia.feature import LAFDescriptor, KeyNetAffNetHardNet
            
            self._torch = torch
            
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA not available for SuperPoint, falling back to CPU")
                self._device = 'cpu'
            
            # Use KeyNet + AffNet + HardNet as SuperPoint alternative
            # (More widely available than actual SuperPoint)
            self._extractor = KeyNetAffNetHardNet(num_features=max_keypoints).to(self._device).eval()
            self._available = True
            logger.info(f"SuperPoint (KeyNetAffNet) initialized on {self._device}")
            
        except ImportError as e:
            logger.warning(f"SuperPoint not available: {e}. Install with: pip install kornia")
        except Exception as e:
            logger.warning(f"SuperPoint initialization failed: {e}")
    
    def _extract_features(self, gray: np.ndarray):
        """Extract features using the deep network."""
        import torch
        
        # Convert to tensor
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            lafs, resps, descs = self._extractor(tensor)
        
        return {
            'lafs': lafs,
            'responses': resps,
            'descriptors': descs,
        }
    
    def _match_features(self, feats0, feats1):
        """Match features between two frames."""
        import torch
        
        desc0 = feats0['descriptors'][0]  # NxD
        desc1 = feats1['descriptors'][0]
        
        # Compute distance matrix
        dists = torch.cdist(desc0, desc1)
        
        # Mutual nearest neighbors
        nn01 = dists.argmin(dim=1)
        nn10 = dists.argmin(dim=0)
        
        ids0 = torch.arange(len(nn01), device=self._device)
        mutual = (nn10[nn01] == ids0)
        
        matches0 = ids0[mutual].cpu().numpy()
        matches1 = nn01[mutual].cpu().numpy()
        
        return matches0, matches1
    
    def _lafs_to_points(self, lafs) -> np.ndarray:
        """Convert LAFs (Local Affine Frames) to keypoint coordinates."""
        # LAFs are Bx N x 2 x 3 tensors, center is at [:, :, :, 2]
        centers = lafs[0, :, :, 2].cpu().numpy()  # Nx2
        return centers
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        if not self._available:
            logger.warning("SuperPoint not available, returning identity pose")
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Resize for efficiency
        h, w = gray.shape
        scale = min(640 / w, 480 / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            gray_resized = cv2.resize(gray, (new_w, new_h))
        else:
            gray_resized = gray
            scale = 1.0
        
        curr_feats = self._extract_features(gray_resized)
        
        # First frame
        if self._prev_feats is None:
            self._prev_gray = gray
            self._prev_feats = curr_feats
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="initializing",
            )
        
        # Match features
        try:
            matches0, matches1 = self._match_features(self._prev_feats, curr_feats)
        except Exception as e:
            logger.warning(f"Feature matching failed: {e}")
            self._prev_gray = gray
            self._prev_feats = curr_feats
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        if len(matches0) < 8:
            self._prev_gray = gray
            self._prev_feats = curr_feats
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(matches0),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Get matched points
        pts0 = self._lafs_to_points(self._prev_feats['lafs'])[matches0] / scale
        pts1 = self._lafs_to_points(curr_feats['lafs'])[matches1] / scale
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts0, pts1, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            self._prev_gray = gray
            self._prev_feats = curr_feats
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        num_inliers, R, t, _ = cv2.recoverPose(E, pts0, pts1, self._K, mask=mask)
        
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        self._prev_gray = gray
        self._prev_feats = curr_feats
        
        confidence = min(1.0, num_inliers / 50.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "superpoint"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_feats = None


# Registry of available estimators
_ESTIMATORS: Dict[str, type] = {
    'orb': ORBPoseEstimator,
    'robust_flow': RobustFlowEstimator,
    'flow': RobustFlowEstimator,
    'sift': SIFTPoseEstimator,
    'keyframe': KeyframePoseEstimator,
    'loftr': LoFTRPoseEstimator,
    'superpoint': SuperPointPoseEstimator,
}


def get_pose_estimator(name: str, **kwargs) -> BasePoseEstimator:
    """
    Get a pose estimator by name.
    
    Args:
        name: Estimator name ('orb', 'robust_flow', 'sift')
        **kwargs: Additional arguments passed to constructor
        
    Returns:
        BasePoseEstimator instance
        
    Raises:
        ValueError: If estimator name is unknown
    """
    name_lower = name.lower().replace('-', '_')
    
    if name_lower not in _ESTIMATORS:
        available = list(_ESTIMATORS.keys())
        raise ValueError(f"Unknown pose estimator: {name}. Available: {available}")
    
    return _ESTIMATORS[name_lower](**kwargs)


def list_pose_estimators() -> Dict[str, Dict[str, Any]]:
    """List available pose estimators."""
    return {
        'orb': {
            'description': 'Fast ORB-based visual odometry',
            'speed': 'fast',
            'gpu': False,
        },
        'robust_flow': {
            'description': 'Optical flow with robust estimation',
            'speed': 'medium',
            'gpu': False,
        },
        'flow': {
            'description': 'Basic optical flow tracking',
            'speed': 'fast',
            'gpu': False,
        },
        'sift': {
            'description': 'SIFT-based feature matching (robust)',
            'speed': 'slow',
            'gpu': False,
        },
        'keyframe': {
            'description': 'Keyframe-based tracking with loop closure',
            'speed': 'medium',
            'gpu': False,
        },
        'loftr': {
            'description': 'LoFTR deep learning matcher (very robust)',
            'speed': 'slow',
            'gpu': True,
            'requires': 'kornia',
        },
        'superpoint': {
            'description': 'SuperPoint deep features (robust to lighting)',
            'speed': 'medium',
            'gpu': True,
            'requires': 'kornia',
        },
    }


__all__ = [
    'BasePoseEstimator',
    'PoseResult',
    'ORBPoseEstimator',
    'RobustFlowEstimator',
    'SIFTPoseEstimator',
    'KeyframePoseEstimator',
    'LoFTRPoseEstimator',
    'SuperPointPoseEstimator',
    'get_pose_estimator',
    'list_pose_estimators',
]
