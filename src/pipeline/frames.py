"""
Frame Data Structures and Sources
================================

This module provides:
- Frame: A dataclass representing a single observation (RGB/RGBD + pose)
- FrameSource: Abstract base class for frame iteration
- TumRGBDSource: Implementation for TUM RGB-D dataset format

The goal is to provide a clean abstraction over different data sources
so the pipeline doesn't need to know about specific dataset formats.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """
    Represents a single observation frame for 3DGS mapping.
    
    This is the core data structure passed through the pipeline. It contains
    all information needed to add a new viewpoint to the Gaussian scene.
    
    Attributes:
        idx: Frame index (sequential integer, 0-based)
        timestamp: Original timestamp from the dataset (seconds)
        rgb: RGB image as HxWx3 numpy array (uint8 or float [0,1])
        depth: Optional depth image as HxW numpy array (metric depth in meters)
        pose: 4x4 numpy array for camera-to-world transformation
        intrinsics: Camera intrinsic parameters dictionary with keys:
            - 'fx', 'fy': Focal lengths in pixels
            - 'cx', 'cy': Principal point in pixels
            - 'width', 'height': Image dimensions
        metadata: Optional dictionary for additional frame-specific data
    
    Coordinate Conventions:
        - pose is camera-to-world (transforms points from camera frame to world)
        - depth is positive into the scene (OpenCV convention)
        - rgb is HxWx3 with channel order RGB (not BGR)
    """
    idx: int
    timestamp: float
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    pose: np.ndarray  # 4x4 world_from_camera
    intrinsics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate frame data after initialization."""
        # Validate RGB
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"RGB must be HxWx3, got shape {self.rgb.shape}")
        
        # Validate depth if present
        if self.depth is not None:
            if self.depth.ndim != 2:
                raise ValueError(f"Depth must be HxW, got shape {self.depth.shape}")
            if self.depth.shape != self.rgb.shape[:2]:
                raise ValueError(
                    f"Depth shape {self.depth.shape} doesn't match RGB {self.rgb.shape[:2]}"
                )
        
        # Validate pose
        if self.pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4, got shape {self.pose.shape}")
        
        # Validate intrinsics
        required_keys = ['fx', 'fy', 'cx', 'cy', 'width', 'height']
        for key in required_keys:
            if key not in self.intrinsics:
                raise ValueError(f"Missing intrinsic parameter: {key}")
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Return (width, height) of the frame."""
        return (int(self.intrinsics['width']), int(self.intrinsics['height']))
    
    def get_rgb_float(self) -> np.ndarray:
        """Return RGB image normalized to [0, 1] float32."""
        if self.rgb.dtype == np.uint8:
            return self.rgb.astype(np.float32) / 255.0
        return self.rgb.astype(np.float32)
    
    def get_rgb_uint8(self) -> np.ndarray:
        """Return RGB image as uint8 in [0, 255]."""
        if self.rgb.dtype == np.uint8:
            return self.rgb
        return (np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)


class FrameSource(ABC):
    """
    Abstract base class for frame sources.
    
    A FrameSource provides an iterator over Frame objects. Different
    implementations can read from:
    - Disk datasets (TUM, KITTI, etc.)
    - Live camera streams
    - ROS topics
    - Video files
    
    All sources must implement __iter__ and __len__ (if known).
    
    Example usage:
        source = TumRGBDSource(dataset_root="/path/to/tum")
        for frame in source:
            pipeline.step(frame)
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        """
        Iterate over frames in the source.
        
        Yields:
            Frame objects in temporal order
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of frames in the source.
        
        Returns:
            Total number of frames, or -1 if unknown (streaming source)
        """
        pass
    
    def get_intrinsics(self) -> Dict[str, float]:
        """
        Get the camera intrinsics for this source.
        
        Returns:
            Dictionary with fx, fy, cx, cy, width, height
        """
        raise NotImplementedError("Subclass should implement get_intrinsics()")


class TumRGBDSource(FrameSource):
    """
    Frame source for TUM RGB-D benchmark datasets.
    
    TUM RGB-D format:
        <dataset>/
            rgb/           # RGB images (png)
            depth/         # Depth images (png, 16-bit, scale 5000)
            rgb.txt        # Timestamps and filenames for RGB
            depth.txt      # Timestamps and filenames for depth
            groundtruth.txt  # Timestamps and poses (tx ty tz qx qy qz qw)
    
    The loader associates RGB, depth, and poses by timestamp matching.
    
    Args:
        dataset_root: Path to search for TUM datasets. Can be:
            - Direct path to a TUM dataset folder
            - Parent directory containing TUM datasets
            
        sequence: Optional specific sequence name (e.g., 'rgbd_dataset_freiburg1_desk')
            If not provided, will use the first found sequence.
            
        max_time_diff: Maximum allowed time difference (seconds) for matching
            RGB, depth, and groundtruth timestamps. Default: 0.05 (50ms)
            
        depth_scale: Scale factor to convert depth image values to meters.
            Default: 5000.0 (TUM standard: depth_meters = pixel_value / 5000)
    
    Example:
        # Load from direct path
        source = TumRGBDSource(dataset_root="/data/tum/rgbd_dataset_freiburg1_desk")
        
        # Search in datasets directory
        source = TumRGBDSource(dataset_root="../datasets", sequence="rgbd_dataset_freiburg1_desk")
    """
    
    # TUM Freiburg camera intrinsics (default for fr1 sequences)
    DEFAULT_INTRINSICS = {
        'fx': 517.3,
        'fy': 516.5,
        'cx': 318.6,
        'cy': 255.3,
        'width': 640,
        'height': 480,
    }
    
    def __init__(
        self,
        dataset_root: str,
        sequence: Optional[str] = None,
        max_time_diff: float = 0.1,  # Increased from 0.05 to handle TUM dataset stream offsets
        depth_scale: float = 5000.0,
        intrinsics: Optional[Dict[str, float]] = None,
    ):
        self._dataset_root = Path(dataset_root)
        self._sequence = sequence
        self._max_time_diff = max_time_diff
        self._depth_scale = depth_scale
        self._intrinsics = intrinsics or self.DEFAULT_INTRINSICS.copy()
        
        # Find the dataset path
        self._dataset_path = self._find_dataset()
        
        if self._dataset_path is None:
            logger.warning(
                f"No TUM dataset found at {dataset_root}. "
                "Source will be empty. Check the path or create a TUM dataset."
            )
            self._frames_info: List[Dict[str, Any]] = []
        else:
            logger.info(f"Found TUM dataset at: {self._dataset_path}")
            self._frames_info = self._load_frame_info()
            logger.info(f"Loaded {len(self._frames_info)} frames")
    
    def _find_dataset(self) -> Optional[Path]:
        """
        Search for a TUM dataset directory.
        
        Returns:
            Path to dataset or None if not found
        """
        # Check if dataset_root is directly a TUM dataset
        if self._is_tum_dataset(self._dataset_root):
            return self._dataset_root
        
        # Check if sequence is specified
        if self._sequence:
            # Try direct path
            seq_path = self._dataset_root / self._sequence
            if self._is_tum_dataset(seq_path):
                return seq_path
            
            # Try tum/ subdirectory
            seq_path = self._dataset_root / "tum" / self._sequence
            if self._is_tum_dataset(seq_path):
                return seq_path
        
        # Search for any TUM dataset
        candidates = [
            self._dataset_root,
            self._dataset_root / "tum",
        ]
        
        for candidate in candidates:
            if not candidate.exists():
                continue
            
            for item in candidate.iterdir():
                if item.is_dir() and self._is_tum_dataset(item):
                    return item
        
        return None
    
    def _is_tum_dataset(self, path: Path) -> bool:
        """Check if a path is a valid TUM RGB-D dataset."""
        required_files = ['rgb.txt', 'groundtruth.txt']
        required_dirs = ['rgb']
        
        if not path.is_dir():
            return False
        
        for f in required_files:
            if not (path / f).exists():
                return False
        
        for d in required_dirs:
            if not (path / d).is_dir():
                return False
        
        return True
    
    def _load_frame_info(self) -> List[Dict[str, Any]]:
        """
        Load and associate RGB, depth, and pose timestamps.
        
        Returns:
            List of dicts with keys: timestamp, rgb_path, depth_path, pose
        """
        # Read RGB timestamps
        rgb_data = self._read_file_list(self._dataset_path / "rgb.txt")
        
        # Read depth timestamps (if available)
        depth_file = self._dataset_path / "depth.txt"
        depth_data = self._read_file_list(depth_file) if depth_file.exists() else {}
        
        # Read groundtruth poses
        gt_data = self._read_groundtruth(self._dataset_path / "groundtruth.txt")
        
        # Associate by timestamp
        frames = []
        
        for ts, rgb_path in sorted(rgb_data.items()):
            # Find closest depth
            depth_path = None
            if depth_data:
                closest_depth_ts = self._find_closest_timestamp(ts, depth_data.keys())
                if closest_depth_ts is not None and abs(closest_depth_ts - ts) < self._max_time_diff:
                    depth_path = depth_data[closest_depth_ts]
            
            # Find closest pose
            closest_pose_ts = self._find_closest_timestamp(ts, gt_data.keys())
            if closest_pose_ts is None or abs(closest_pose_ts - ts) > self._max_time_diff:
                # Skip frame without valid pose
                continue
            
            pose = gt_data[closest_pose_ts]
            
            frames.append({
                'timestamp': ts,
                'rgb_path': self._dataset_path / rgb_path,
                'depth_path': self._dataset_path / depth_path if depth_path else None,
                'pose': pose,
            })
        
        return frames
    
    def _read_file_list(self, filepath: Path) -> Dict[float, str]:
        """
        Read a TUM file list (rgb.txt or depth.txt).
        
        Returns:
            Dict mapping timestamp -> filename
        """
        result = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    ts = float(parts[0])
                    filename = parts[1]
                    result[ts] = filename
        
        return result
    
    def _read_groundtruth(self, filepath: Path) -> Dict[float, np.ndarray]:
        """
        Read TUM groundtruth.txt file.
        
        Format: timestamp tx ty tz qx qy qz qw
        
        Returns:
            Dict mapping timestamp -> 4x4 pose matrix (camera-to-world)
        """
        result = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 8:
                    ts = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    
                    # Convert quaternion to rotation matrix
                    pose = self._quat_to_matrix(tx, ty, tz, qx, qy, qz, qw)
                    result[ts] = pose
        
        return result
    
    def _quat_to_matrix(
        self, 
        tx: float, ty: float, tz: float,
        qx: float, qy: float, qz: float, qw: float
    ) -> np.ndarray:
        """
        Convert translation + quaternion to 4x4 transformation matrix.
        
        Args:
            tx, ty, tz: Translation
            qx, qy, qz, qw: Quaternion (x, y, z, w order)
        
        Returns:
            4x4 camera-to-world transformation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        
        # Build 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        
        return T
    
    def _find_closest_timestamp(
        self, 
        target: float, 
        timestamps: Any
    ) -> Optional[float]:
        """Find the timestamp closest to target."""
        timestamps = list(timestamps)
        if not timestamps:
            return None
        
        # Binary search would be faster, but linear is fine for typical dataset sizes
        closest = min(timestamps, key=lambda t: abs(t - target))
        return closest
    
    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames in timestamp order."""
        import cv2
        
        for idx, info in enumerate(self._frames_info):
            # Load RGB
            rgb = cv2.imread(str(info['rgb_path']))
            if rgb is None:
                logger.warning(f"Failed to load RGB: {info['rgb_path']}")
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Load depth (if available)
            depth = None
            if info['depth_path'] is not None and info['depth_path'].exists():
                depth_raw = cv2.imread(str(info['depth_path']), cv2.IMREAD_UNCHANGED)
                if depth_raw is not None:
                    # Convert to meters
                    depth = depth_raw.astype(np.float32) / self._depth_scale
                    # Mark invalid depth as 0
                    depth[depth_raw == 0] = 0
            
            # Update intrinsics with actual image size
            intrinsics = self._intrinsics.copy()
            intrinsics['width'] = rgb.shape[1]
            intrinsics['height'] = rgb.shape[0]
            
            yield Frame(
                idx=idx,
                timestamp=info['timestamp'],
                rgb=rgb,
                depth=depth,
                pose=info['pose'],
                intrinsics=intrinsics,
            )
    
    def __len__(self) -> int:
        """Return number of frames."""
        return len(self._frames_info)
    
    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics."""
        return self._intrinsics.copy()


class ColmapSource(FrameSource):
    """
    Frame source for COLMAP-style datasets.
    
    TODO: Implement reading from COLMAP sparse reconstruction.
    This would read cameras.bin, images.bin, and points3D.bin.
    """
    
    def __init__(self, dataset_root: str):
        raise NotImplementedError("ColmapSource is not yet implemented")
    
    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0


class VideoSource(FrameSource):
    """
    Frame source for video files with pose tracking.
    
    TODO: Implement video reading with external pose file or SLAM integration.
    """
    
    def __init__(self, video_path: str, poses_path: Optional[str] = None):
        raise NotImplementedError("VideoSource is not yet implemented")
    
    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return -1  # Unknown for streaming
