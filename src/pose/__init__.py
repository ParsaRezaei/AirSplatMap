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


def _get_best_torch_device(preferred: str = "cuda") -> str:
    """
    Get the best available PyTorch device.
    
    Priority: CUDA > MPS (Apple Metal) > CPU
    
    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu', 'auto')
        
    Returns:
        Device string that's actually available
    """
    try:
        import torch
        
        if preferred == "auto" or preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            # Try MPS for macOS with AMD/Apple GPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            if preferred == "cuda":
                logger.info("CUDA not available, using CPU")
            return "cpu"
        
        if preferred == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        
        return preferred
    except ImportError:
        return "cpu"


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
    
    def cleanup(self):
        """Cleanup resources and free GPU memory. Override in subclasses."""
        self.reset()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


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
        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        # Ensure contiguous arrays for OpenCV
        pts1 = np.ascontiguousarray(pts1)
        pts2 = np.ascontiguousarray(pts2)
        
        # Ensure we have valid points
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
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
        
        # Ensure contiguous arrays for OpenCV
        good_prev = np.ascontiguousarray(good_prev)
        good_next = np.ascontiguousarray(good_next)
        
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
        
        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        # Ensure contiguous arrays for OpenCV
        pts1 = np.ascontiguousarray(pts1)
        pts2 = np.ascontiguousarray(pts2)
        
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
            
            pts1 = np.ascontiguousarray(np.float32([kf_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2))
            pts2 = np.ascontiguousarray(np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2))
            
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
                pts1 = np.ascontiguousarray(np.float32([self._prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2))
                pts2 = np.ascontiguousarray(np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2))
                
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
            
            # Get best available device (CUDA > MPS > CPU)
            self._device = _get_best_torch_device(device)
            
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
        
        # Ensure contiguous arrays for OpenCV
        mkpts0 = np.ascontiguousarray(mkpts0.reshape(-1, 2))
        mkpts1 = np.ascontiguousarray(mkpts1.reshape(-1, 2))
        
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
    
    def cleanup(self):
        """Release GPU memory."""
        super().cleanup()
        self._prev_tensor = None
        if self._matcher is not None:
            del self._matcher
            self._matcher = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


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
            
            # Get best available device (CUDA > MPS > CPU)
            self._device = _get_best_torch_device(device)
            
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
        
        # Ensure contiguous arrays for OpenCV
        pts0 = np.ascontiguousarray(pts0.reshape(-1, 2))
        pts1 = np.ascontiguousarray(pts1.reshape(-1, 2))
        
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
    
    def cleanup(self):
        """Release GPU memory."""
        super().cleanup()
        self._prev_gray = None
        self._prev_feats = None
        if self._extractor is not None:
            del self._extractor
            self._extractor = None
        if hasattr(self, '_matcher') and self._matcher is not None:
            del self._matcher
            self._matcher = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


class LightGluePoseEstimator(BasePoseEstimator):
    """
    Pose estimation using LightGlue matcher.
    
    LightGlue is a fast and accurate deep learning-based feature matcher.
    It's a lightweight alternative to SuperGlue with comparable accuracy.
    Can use SuperPoint, DISK, or ALIKED as feature extractor.
    
    Requires: pip install lightglue
    """
    
    def __init__(
        self, 
        device: str = 'cuda', 
        max_keypoints: int = 2048,
        extractor: str = 'superpoint',  # 'superpoint', 'disk', 'aliked'
    ):
        super().__init__()
        self._device = device
        self._max_keypoints = max_keypoints
        self._extractor_name = extractor
        self._extractor = None
        self._matcher = None
        self._available = False
        
        self._prev_gray = None
        self._prev_feats = None
        
        try:
            import torch
            
            self._torch = torch
            
            # Get best available device (CUDA > MPS > CPU)
            self._device = _get_best_torch_device(device)
            
            # Try lightglue package first
            try:
                from lightglue import LightGlue, SuperPoint, DISK, ALIKED
                from lightglue.utils import rbd  # Remove batch dimension
                
                self._rbd = rbd
                
                # Select extractor
                if extractor == 'superpoint':
                    self._extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self._device)
                elif extractor == 'disk':
                    self._extractor = DISK(max_num_keypoints=max_keypoints).eval().to(self._device)
                elif extractor == 'aliked':
                    self._extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(self._device)
                else:
                    self._extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self._device)
                    extractor = 'superpoint'
                
                # Initialize LightGlue matcher
                self._matcher = LightGlue(features=extractor).eval().to(self._device)
                self._available = True
                logger.info(f"LightGlue ({extractor}) initialized on {self._device}")
                
            except ImportError:
                # Fall back to kornia if lightglue not available
                logger.warning("lightglue package not found, trying kornia LightGlue")
                import kornia
                from kornia.feature import LightGlue as KorniaLightGlue, KeyNetAffNetHardNet
                
                # KeyNetAffNetHardNet + LightGlue combo works well
                self._extractor = KeyNetAffNetHardNet(num_features=max_keypoints).to(self._device).eval()
                self._matcher = KorniaLightGlue('keynet_affnet_hardnet').to(self._device).eval()
                self._use_kornia = True
                self._available = True
                logger.info(f"LightGlue (kornia) initialized on {self._device}")
                
        except ImportError as e:
            logger.warning(f"LightGlue not available: {e}. Install with: pip install lightglue")
        except Exception as e:
            logger.warning(f"LightGlue initialization failed: {e}")
    
    def _extract_and_match(self, gray0: np.ndarray, gray1: np.ndarray):
        """Extract features and match between two images."""
        import torch
        
        if hasattr(self, '_use_kornia') and self._use_kornia:
            return self._extract_and_match_kornia(gray0, gray1)
        
        # Using lightglue package
        img0 = torch.from_numpy(gray0).float() / 255.0
        img1 = torch.from_numpy(gray1).float() / 255.0
        
        img0 = img0.unsqueeze(0).unsqueeze(0).to(self._device)
        img1 = img1.unsqueeze(0).unsqueeze(0).to(self._device)
        
        # Extract features
        with torch.no_grad():
            feats0 = self._extractor.extract(img0)
            feats1 = self._extractor.extract(img1)
            
            # Match
            matches01 = self._matcher({'image0': feats0, 'image1': feats1})
        
        # Get matched keypoints
        feats0, feats1, matches01 = [self._rbd(x) for x in [feats0, feats1, matches01]]
        
        kpts0 = feats0['keypoints'].cpu().numpy()
        kpts1 = feats1['keypoints'].cpu().numpy()
        matches = matches01['matches'].cpu().numpy()
        
        # matches is [M, 2] where each row is [idx0, idx1]
        if len(matches) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        
        idx0 = matches[:, 0]  # Indices into kpts0
        idx1 = matches[:, 1]  # Indices into kpts1
        
        # Filter valid indices (within bounds)
        valid = (idx0 >= 0) & (idx0 < len(kpts0)) & (idx1 >= 0) & (idx1 < len(kpts1))
        
        if not np.any(valid):
            return np.zeros((0, 2)), np.zeros((0, 2))
        
        idx0 = idx0[valid]
        idx1 = idx1[valid]
        
        mkpts0 = kpts0[idx0]
        mkpts1 = kpts1[idx1]
        
        return mkpts0, mkpts1
    
    def _extract_and_match_kornia(self, gray0: np.ndarray, gray1: np.ndarray):
        """Extract and match using kornia backend (KeyNetAffNetHardNet + LightGlue)."""
        import torch
        
        h, w = gray0.shape[:2]
        
        # Convert to tensors - kornia expects [B, C, H, W]
        img0 = torch.from_numpy(gray0).float() / 255.0
        img1 = torch.from_numpy(gray1).float() / 255.0
        
        img0 = img0.unsqueeze(0).unsqueeze(0).to(self._device)
        img1 = img1.unsqueeze(0).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            # KeyNetAffNetHardNet returns (lafs, responses, descriptors)
            lafs0, resp0, desc0 = self._extractor(img0)
            lafs1, resp1, desc1 = self._extractor(img1)
            
            # Extract keypoint centers from LAFs
            kpts0 = lafs0[:, :, :, 2]  # [B, N, 2] - (x, y) in pixels
            kpts1 = lafs1[:, :, :, 2]
            
            # Build feature dicts for LightGlue
            # Note: image_size must be (w, h) to match (x, y) keypoint format
            feats0 = {
                'keypoints': kpts0,
                'descriptors': desc0,
                'lafs': lafs0,
                'image_size': torch.tensor([[w, h]], device=self._device),
            }
            feats1 = {
                'keypoints': kpts1,
                'descriptors': desc1,
                'lafs': lafs1,
                'image_size': torch.tensor([[w, h]], device=self._device),
            }
            
            # Match
            out = self._matcher({'image0': feats0, 'image1': feats1})
            
            # matches is a list of [M, 2] tensors where each row is [idx0, idx1]
            matches = out['matches'][0]  # [M, 2] for batch 0
            
            if len(matches) == 0:
                return np.zeros((0, 2)), np.zeros((0, 2))
            
            idx0 = matches[:, 0]  # Indices into kpts0
            idx1 = matches[:, 1]  # Indices into kpts1
            
            mkpts0 = kpts0[0, idx0].cpu().numpy()  # [M, 2]
            mkpts1 = kpts1[0, idx1].cpu().numpy()  # [M, 2]
        
        return mkpts0, mkpts1
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        if not self._available:
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
        
        # First frame
        if self._prev_gray is None:
            self._prev_gray = gray_resized
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="initializing",
            )
        
        # Extract and match
        try:
            mkpts0, mkpts1 = self._extract_and_match(self._prev_gray, gray_resized)
        except Exception as e:
            logger.warning(f"LightGlue matching failed: {e}")
            self._prev_gray = gray_resized
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        if len(mkpts0) < 8:
            self._prev_gray = gray_resized
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(mkpts0),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Scale points back to original resolution and ensure correct format for OpenCV
        mkpts0 = (mkpts0 / scale).astype(np.float64)
        mkpts1 = (mkpts1 / scale).astype(np.float64)
        
        # Ensure points are contiguous and 2D
        mkpts0 = np.ascontiguousarray(mkpts0.reshape(-1, 2))
        mkpts1 = np.ascontiguousarray(mkpts1.reshape(-1, 2))
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            self._prev_gray = gray_resized
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
        self._prev_gray = gray_resized
        
        confidence = min(1.0, num_inliers / 100.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "lightglue"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_feats = None
    
    def cleanup(self):
        """Release GPU memory."""
        super().cleanup()
        self._prev_gray = None
        self._prev_feats = None
        if self._extractor is not None:
            del self._extractor
            self._extractor = None
        if self._matcher is not None:
            del self._matcher
            self._matcher = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


class RAFTPoseEstimator(BasePoseEstimator):
    """
    Pose estimation using RAFT optical flow.
    
    RAFT (Recurrent All-Pairs Field Transforms) is a state-of-the-art
    optical flow method. This estimator uses dense optical flow for
    robust pose estimation.
    
    Requires: torchvision >= 0.14 (includes RAFT)
    """
    
    def __init__(
        self, 
        device: str = 'cuda',
        model: str = 'large',  # 'small' or 'large'
        grid_size: int = 20,
    ):
        super().__init__()
        self._device = device
        self._model_name = model
        self._grid_size = grid_size
        self._raft = None
        self._available = False
        
        self._prev_gray = None
        self._prev_tensor = None
        
        try:
            import torch
            import torchvision
            
            self._torch = torch
            
            # Get best available device (CUDA > MPS > CPU)
            self._device = _get_best_torch_device(device)
            
            # Load RAFT model from torchvision
            if model == 'small':
                weights = torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT
                self._raft = torchvision.models.optical_flow.raft_small(weights=weights)
            else:
                weights = torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
                self._raft = torchvision.models.optical_flow.raft_large(weights=weights)
            
            self._raft = self._raft.to(self._device).eval()
            self._transforms = weights.transforms()
            self._available = True
            logger.info(f"RAFT ({model}) initialized on {self._device}")
            
        except ImportError as e:
            logger.warning(f"RAFT not available: {e}. Requires torchvision >= 0.14")
        except Exception as e:
            logger.warning(f"RAFT initialization failed: {e}")
    
    def _preprocess(self, img: np.ndarray):
        """Preprocess image for RAFT."""
        import torch
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        
        # Convert to tensor [H, W, 3] -> [1, 3, H, W]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self._device)
        
        return tensor
    
    def _compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two images."""
        import torch
        
        t1 = self._preprocess(img1)
        t2 = self._preprocess(img2)
        
        # Apply RAFT transforms
        t1, t2 = self._transforms(t1, t2)
        
        with torch.no_grad():
            # RAFT returns list of flow predictions at different iterations
            flows = self._raft(t1, t2)
            flow = flows[-1]  # Take final prediction
        
        # Convert to numpy [1, 2, H, W] -> [H, W, 2]
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow
    
    def _sample_points_from_flow(self, flow: np.ndarray, grid_size: int):
        """Sample sparse correspondences from dense flow."""
        h, w = flow.shape[:2]
        
        # Create grid of points
        step = grid_size
        y_coords = np.arange(step, h - step, step)
        x_coords = np.arange(step, w - step, step)
        
        pts1 = []
        pts2 = []
        
        for y in y_coords:
            for x in x_coords:
                fx, fy = flow[y, x]
                
                # Skip invalid flow
                if np.isnan(fx) or np.isnan(fy):
                    continue
                if abs(fx) > 200 or abs(fy) > 200:
                    continue
                
                x2 = x + fx
                y2 = y + fy
                
                # Check bounds
                if 0 <= x2 < w and 0 <= y2 < h:
                    pts1.append([x, y])
                    pts2.append([x2, y2])
        
        return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        if not self._available:
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Resize for efficiency
        h, w = rgb.shape[:2]
        scale = min(512 / w, 384 / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            # RAFT needs dimensions divisible by 8
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            rgb_resized = cv2.resize(rgb, (new_w, new_h))
        else:
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            rgb_resized = cv2.resize(rgb, (new_w, new_h))
            scale = new_w / w
        
        # First frame
        if self._prev_gray is None:
            self._prev_gray = rgb_resized
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="initializing",
            )
        
        # Compute flow
        try:
            flow = self._compute_flow(self._prev_gray, rgb_resized)
            pts1, pts2 = self._sample_points_from_flow(flow, self._grid_size)
        except Exception as e:
            logger.warning(f"RAFT flow computation failed: {e}")
            self._prev_gray = rgb_resized
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        if len(pts1) < 8:
            self._prev_gray = rgb_resized
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(pts1),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Scale points back
        pts1 = pts1 / scale
        pts2 = pts2 / scale
        
        # Ensure contiguous arrays for OpenCV
        pts1 = np.ascontiguousarray(pts1.reshape(-1, 2))
        pts2 = np.ascontiguousarray(pts2.reshape(-1, 2))
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        
        if E is None or mask is None:
            self._prev_gray = rgb_resized
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
        self._prev_gray = rgb_resized
        
        confidence = min(1.0, num_inliers / 100.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "raft"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
    
    def cleanup(self):
        """Release GPU memory."""
        super().cleanup()
        self._prev_gray = None
        if self._raft is not None:
            del self._raft
            self._raft = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


class R2D2PoseEstimator(BasePoseEstimator):
    """
    Pose estimation using R2D2 (Reliable and Repeatable Detector and Descriptor).
    
    R2D2 is a deep learning-based local feature detector and descriptor that 
    jointly learns keypoint detection and description.
    
    Paper: "R2D2: Repeatable and Reliable Detector and Descriptor" (NeurIPS 2019)
    
    Requires: 
        pip install kornia  # Has R2D2 implementation
        # Or clone official repo: https://github.com/naver/r2d2
    """
    
    def __init__(
        self, 
        device: str = 'cuda',
        max_keypoints: int = 2000,
        reliability_threshold: float = 0.7,
        repeatability_threshold: float = 0.7,
    ):
        super().__init__()
        self._device = device
        self._max_keypoints = max_keypoints
        self._rel_th = reliability_threshold
        self._rep_th = repeatability_threshold
        self._r2d2 = None
        self._available = False
        
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
        
        try:
            import torch
            
            self._torch = torch
            
            # Get best available device (CUDA > MPS > CPU)
            self._device = _get_best_torch_device(device)
            
            # Try kornia's R2D2 implementation first
            try:
                from kornia.feature import KeyNetAffNetHardNet
                
                # Use KeyNet+AffNet+HardNet as proxy (similar learned features)
                # Full R2D2 requires downloading official weights
                self._r2d2 = KeyNetAffNetHardNet(
                    num_features=max_keypoints,
                ).to(self._device).eval()
                self._use_kornia_proxy = True
                self._available = True
                logger.info(f"R2D2 (KeyNet proxy) initialized on {self._device}")
                
            except ImportError:
                logger.warning("R2D2/KeyNet not available in kornia")
                
        except ImportError as e:
            logger.warning(f"R2D2 not available: {e}")
        except Exception as e:
            logger.warning(f"R2D2 initialization failed: {e}")
    
    def _extract_features(self, gray: np.ndarray):
        """Extract R2D2 features from grayscale image."""
        import torch
        
        # Convert to tensor
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            if self._use_kornia_proxy:
                lafs, responses, descriptors = self._r2d2(tensor)
                # Convert LAFs to keypoints
                keypoints = lafs[0, :, :, 2].cpu().numpy()  # Nx2
                descriptors = descriptors[0].cpu().numpy()  # NxD
            else:
                # Native R2D2 output
                keypoints, descriptors = self._r2d2.detect_and_compute(tensor)
        
        return keypoints, descriptors
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        if not self._available:
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
        
        # Extract features
        try:
            kp, desc = self._extract_features(gray_resized)
        except Exception as e:
            logger.warning(f"R2D2 feature extraction failed: {e}")
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # First frame
        if self._prev_desc is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(kp) if len(kp) > 0 else 0,
                tracking_status="initializing",
            )
        
        if len(kp) < 10 or len(self._prev_kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Match descriptors using mutual nearest neighbor
        import torch
        desc1_t = torch.from_numpy(self._prev_desc).to(self._device)
        desc2_t = torch.from_numpy(desc).to(self._device)
        
        dists = torch.cdist(desc1_t, desc2_t)
        nn01 = dists.argmin(dim=1)
        nn10 = dists.argmin(dim=0)
        
        ids0 = torch.arange(len(nn01), device=self._device)
        mutual = (nn10[nn01] == ids0)
        
        matches0 = ids0[mutual].cpu().numpy()
        matches1 = nn01[mutual].cpu().numpy()
        
        if len(matches0) < 8:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(matches0),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Get matched points and scale back
        pts1 = self._prev_kp[matches0] / scale
        pts2 = kp[matches1] / scale
        
        # Ensure contiguous arrays for OpenCV
        pts1 = np.ascontiguousarray(pts1.reshape(-1, 2))
        pts2 = np.ascontiguousarray(pts2.reshape(-1, 2))
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
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
        
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        confidence = min(1.0, num_inliers / 50.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "r2d2"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
    
    def cleanup(self):
        """Release GPU memory."""
        super().cleanup()
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
        if self._r2d2 is not None:
            del self._r2d2
            self._r2d2 = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


class RoMaPoseEstimator(BasePoseEstimator):
    """
    Pose estimation using RoMa (Robust Dense Feature Matching).
    
    RoMa is a robust dense feature matcher that handles wide baselines
    and challenging conditions better than traditional methods.
    Uses DINOv2 as backbone for robust features.
    
    Paper: "RoMa: Revisiting Robust Losses for Dense Feature Matching" (CVPR 2024)
    
    Requires: pip install roma-matcher
    """
    
    def __init__(
        self, 
        device: str = 'cuda',
        model: str = 'outdoor',  # 'outdoor' or 'indoor'
    ):
        super().__init__()
        self._device = device
        self._model_name = model
        self._roma = None
        self._available = False
        
        self._prev_gray = None
        self._prev_tensor = None
        
        try:
            import torch
            
            self._torch = torch
            
            # Get best available device (CUDA > MPS > CPU)
            self._device = _get_best_torch_device(device)
            
            # Try to import RoMa
            try:
                from romatch import roma_outdoor, roma_indoor
                
                # Use fallback correlation for ARM/Jetson (no fused-local-corr available)
                # Also reduce resolution for memory efficiency on edge devices
                import platform
                is_arm = platform.machine().startswith('aarch')
                
                roma_kwargs = {'device': self._device}
                if is_arm:
                    # Disable custom corr and use smaller resolution for ARM
                    roma_kwargs['use_custom_corr'] = False
                    roma_kwargs['coarse_res'] = 280  # Default is 560
                    roma_kwargs['upsample_res'] = 420  # Default is 864
                    logger.info("RoMa: Using ARM-optimized settings (no fused-local-corr)")
                
                if model == 'indoor':
                    self._roma = roma_indoor(**roma_kwargs)
                else:
                    self._roma = roma_outdoor(**roma_kwargs)
                
                self._available = True
                logger.info(f"RoMa ({model}) initialized on {self._device}")
                
            except ImportError:
                # Try alternative import
                try:
                    import roma
                    self._roma = roma.RoMa(pretrained=True, device=self._device)
                    self._available = True
                    logger.info(f"RoMa initialized on {self._device}")
                except ImportError:
                    logger.warning("RoMa not available. Install with: pip install romatch")
                    
        except ImportError as e:
            logger.warning(f"RoMa not available: {e}")
        except Exception as e:
            logger.warning(f"RoMa initialization failed: {e}")
    
    def _match_images(self, img0: np.ndarray, img1: np.ndarray):
        """Match two images using RoMa."""
        import torch
        from PIL import Image
        
        # RoMa expects PIL images
        if len(img0.shape) == 2:
            # Grayscale - convert to RGB
            img0 = np.stack([img0, img0, img0], axis=-1)
            img1 = np.stack([img1, img1, img1], axis=-1)
        
        pil0 = Image.fromarray(img0.astype(np.uint8))
        pil1 = Image.fromarray(img1.astype(np.uint8))
        
        with torch.no_grad():
            # RoMa returns warp [B, H, W, 4] and certainty [B, H, W]
            # warp contains (x0, y0, x1, y1) normalized coordinates
            warp, certainty = self._roma.match(pil0, pil1)
            
            # Sample sparse matches from dense correspondence
            h, w = certainty.shape[1:3]
            
            # Create grid of source points (sample every 8 pixels)
            y_coords = torch.arange(0, h, 8, device=certainty.device)
            x_coords = torch.arange(0, w, 8, device=certainty.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Sample indices
            idx_y = yy.flatten().long()
            idx_x = xx.flatten().long()
            
            # Get certainty scores at sample points
            cert = certainty[0, idx_y, idx_x]  # [N]
            
            # Get warp coordinates (normalized [-1, 1] -> pixel coords)
            warp_sampled = warp[0, idx_y, idx_x]  # [N, 4] - (x0, y0, x1, y1)
            
            # Convert normalized coords to pixel coords
            orig_h, orig_w = img0.shape[:2]
            pts1_x = (warp_sampled[:, 0] + 1) / 2 * orig_w
            pts1_y = (warp_sampled[:, 1] + 1) / 2 * orig_h
            pts2_x = (warp_sampled[:, 2] + 1) / 2 * orig_w
            pts2_y = (warp_sampled[:, 3] + 1) / 2 * orig_h
            
            pts1 = torch.stack([pts1_x, pts1_y], dim=1)
            pts2 = torch.stack([pts2_x, pts2_y], dim=1)
            
            # Filter by certainty
            valid = cert > 0.5
            pts1 = pts1[valid].cpu().numpy()
            pts2 = pts2[valid].cpu().numpy()
        
        return pts1, pts2
    
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if self._K is None:
            raise RuntimeError("Intrinsics not set")
        
        if not self._available:
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Resize for efficiency (RoMa is memory hungry)
        h, w = rgb.shape[:2]
        scale = min(560 / w, 420 / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_resized = cv2.resize(rgb, (new_w, new_h))
        else:
            rgb_resized = rgb
            scale = 1.0
        
        # First frame
        if self._prev_gray is None:
            self._prev_gray = rgb_resized
            self._initialized = True
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="initializing",
            )
        
        # Match images
        try:
            pts1, pts2 = self._match_images(self._prev_gray, rgb_resized)
        except Exception as e:
            logger.warning(f"RoMa matching failed: {e}")
            self._prev_gray = rgb_resized
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=0,
                tracking_status="lost",
                confidence=0.0,
            )
        
        if len(pts1) < 8:
            self._prev_gray = rgb_resized
            return PoseResult(
                pose=self._current_pose.copy(),
                num_inliers=len(pts1),
                tracking_status="lost",
                confidence=0.0,
            )
        
        # Scale points back
        pts1 = pts1 / scale
        pts2 = pts2 / scale
        
        # Ensure contiguous arrays for OpenCV
        pts1 = np.ascontiguousarray(pts1.reshape(-1, 2))
        pts2 = np.ascontiguousarray(pts2.reshape(-1, 2))
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            self._prev_gray = rgb_resized
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
        self._prev_gray = rgb_resized
        
        confidence = min(1.0, num_inliers / 100.0)
        
        return PoseResult(
            pose=self._current_pose.copy(),
            num_inliers=num_inliers,
            confidence=confidence,
            tracking_status="ok",
        )
    
    def get_name(self) -> str:
        return "roma"
    
    def reset(self):
        super().reset()
        self._prev_gray = None
    
    def cleanup(self):
        """Release GPU memory."""
        super().cleanup()
        self._prev_gray = None
        self._prev_tensor = None if hasattr(self, '_prev_tensor') else None
        if self._roma is not None:
            del self._roma
            self._roma = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass


# Registry of available estimators
_ESTIMATORS: Dict[str, type] = {
    'orb': ORBPoseEstimator,
    'robust_flow': RobustFlowEstimator,
    'flow': RobustFlowEstimator,
    'sift': SIFTPoseEstimator,
    'keyframe': KeyframePoseEstimator,
    'loftr': LoFTRPoseEstimator,
    'superpoint': SuperPointPoseEstimator,
    # New deep learning matchers
    'lightglue': LightGluePoseEstimator,
    'raft': RAFTPoseEstimator,
    'r2d2': R2D2PoseEstimator,
    'roma': RoMaPoseEstimator,
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
    """List available pose estimators with availability status."""
    
    # Check which estimators are actually available
    def check_available(name: str) -> bool:
        try:
            est = get_pose_estimator(name)
            return est is not None
        except Exception:
            return False
    
    estimators = {
        'orb': {
            'description': 'Fast ORB-based visual odometry',
            'speed': 'fast',
            'requires_gpu': False,
        },
        'robust_flow': {
            'description': 'Optical flow with robust estimation',
            'speed': 'medium',
            'requires_gpu': False,
        },
        'flow': {
            'description': 'Basic optical flow tracking',
            'speed': 'fast',
            'requires_gpu': False,
        },
        'sift': {
            'description': 'SIFT-based feature matching (robust)',
            'speed': 'slow',
            'requires_gpu': False,
        },
        'keyframe': {
            'description': 'Keyframe-based tracking with loop closure',
            'speed': 'medium',
            'requires_gpu': False,
        },
        'loftr': {
            'description': 'LoFTR deep learning matcher (very robust)',
            'speed': 'slow',
            'requires_gpu': True,
            'requires': 'kornia',
        },
        'superpoint': {
            'description': 'SuperPoint deep features (robust to lighting)',
            'speed': 'medium',
            'requires_gpu': True,
            'requires': 'kornia',
        },
        'lightglue': {
            'description': 'LightGlue fast deep matcher (CVPR 2023)',
            'speed': 'medium',
            'requires_gpu': True,
            'requires': 'lightglue or kornia',
        },
        'raft': {
            'description': 'RAFT optical flow (state-of-the-art)',
            'speed': 'medium',
            'requires_gpu': True,
            'requires': 'torchvision>=0.14',
        },
        'r2d2': {
            'description': 'R2D2 learned features (NeurIPS 2019)',
            'speed': 'medium',
            'requires_gpu': True,
            'requires': 'kornia',
        },
        'roma': {
            'description': 'RoMa robust dense matcher (CVPR 2024)',
            'speed': 'slow',
            'requires_gpu': True,
            'requires': 'romatch',
        },
    }
    
    # Add availability status
    for name in estimators:
        estimators[name]['available'] = check_available(name)
    
    return estimators


__all__ = [
    # Base classes and types
    'BasePoseEstimator',
    'PoseResult',
    # Monocular VO estimators (from this module)
    'ORBPoseEstimator',
    'RobustFlowEstimator',
    'SIFTPoseEstimator',
    'KeyframePoseEstimator',
    'LoFTRPoseEstimator',
    'SuperPointPoseEstimator',
    'LightGluePoseEstimator',
    'RAFTPoseEstimator',
    'R2D2PoseEstimator',
    'RoMaPoseEstimator',
    'get_pose_estimator',
    'list_pose_estimators',
    # Stereo VIO (from stereo_vio module)
    'StereoVIO',
    'VIOResult',
    'create_stereo_vio',
    # RGB-D VO (from rgbd_vo module)
    'RGBDVO',
    'VOResult',
    'create_rgbd_vo',
    # External pose sources
    'ArduPilotPoseProvider',
    # External VIO systems (submodule)
    'external',
]

# Import from submodules for convenience
try:
    from .stereo_vio import StereoVIO, VIOResult, create_stereo_vio
except ImportError:
    pass

try:
    from .rgbd_vo import RGBDVO, VOResult, create_rgbd_vo
except ImportError:
    pass

try:
    from .ardupilot_mavlink import ArduPilotPoseProvider
except ImportError:
    pass

# External VIO backends (ORB-SLAM3, OpenVINS, DPVO, DROID-SLAM)
from . import external
