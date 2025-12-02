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


class LiveVideoSource(FrameSource):
    """
    Live video source from RTSP streams, webcams, or video files.
    
    Integrates:
    - Video capture (OpenCV)
    - Modular depth estimation (src.depth)
    - Modular pose estimation (src.pose)
    
    Args:
        source: Video source - RTSP URL, webcam index (0,1,..), or video file path
        fov_deg: Camera field of view in degrees (used to estimate intrinsics)
        target_fps: Target processing frame rate
        resize: Optional (width, height) to resize frames
        depth_model: Depth estimation model ('depth_anything', 'midas', 'zoedepth', 'none')
        pose_model: Pose estimation model ('orb', 'sift', 'loftr', 'flow', 'hybrid')
        max_frames: Maximum frames to process (None = unlimited)
        
    Example:
        # RTSP stream with LoFTR pose estimation
        source = LiveVideoSource(
            "rtsp://192.168.1.100:554/stream",
            pose_model='loftr',
            depth_model='midas'
        )
        
        # Webcam with fast ORB
        source = LiveVideoSource(0, pose_model='orb')
        
        # HTTP MJPEG stream
        source = LiveVideoSource("http://localhost:8554/tum")
    """
    
    def __init__(
        self,
        source,
        fov_deg: float = 60.0,
        target_fps: float = 15.0,
        resize: Optional[Tuple[int, int]] = (640, 480),
        depth_model: str = 'depth_anything_v3',
        pose_model: str = 'robust_flow',
        max_frames: Optional[int] = None,
        scale_pose_to_depth: bool = True,
        smooth_pose: bool = True,
        use_server_pose: bool = False,
        pose_server_url: Optional[str] = None,
    ):
        import cv2
        
        self._source = source
        self._fov_deg = fov_deg
        self._target_fps = target_fps  # 0 or None means use source FPS
        self._resize = resize
        self._depth_model_name = depth_model
        self._pose_model_name = pose_model
        self._max_frames = max_frames
        self._scale_pose_to_depth = scale_pose_to_depth
        self._enable_smooth_pose = smooth_pose
        
        # Server-based pose and depth (e.g., from TUM simulator with ground truth)
        self._use_server_pose = use_server_pose
        self._pose_server_url = pose_server_url
        self._use_server_depth = (depth_model == 'ground_truth')  # Explicit GT depth request
        # Use synchronized frame fetching when both GT pose and depth are needed
        self._use_synchronized_fetch = self._use_server_pose and self._use_server_depth

        
        # Auto-detect server URLs from HTTP stream source
        if (self._use_server_pose or self._use_server_depth) and self._pose_server_url is None:
            if isinstance(source, str) and source.startswith('http'):
                # Extract base URL: http://localhost:8554/stream -> http://localhost:8554
                base_url = source.rsplit('/', 1)[0]
                self._pose_server_url = f"{base_url}/pose"
                self._depth_server_url = f"{base_url}/depth"
                self._intrinsics_url = f"{base_url}/intrinsics"
                self._frame_server_url = f"{base_url}/frame"  # Synchronized bundle
                if self._use_server_pose:
                    logger.info(f"Auto-detected pose server: {self._pose_server_url}")
                if self._use_server_depth:
                    logger.info(f"Auto-detected depth server: {self._depth_server_url}")
                if self._use_synchronized_fetch:
                    logger.info(f"Mode: Synchronized GT from {self._frame_server_url}")
        
        # Pose history for smoothing and scale estimation
        self._pose_history = []
        self._depth_scale_factor = 1.0
        self._scale_initialized = False
        
        # Detect source type
        self._is_rtsp = isinstance(source, str) and source.startswith('rtsp://')
        self._is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        self._is_file = isinstance(source, str) and Path(source).exists()
        self._is_http = isinstance(source, str) and source.startswith('http')
        
        # Initialize capture
        src = int(source) if self._is_webcam else source
        self._cap = cv2.VideoCapture(src)
        
        if self._is_rtsp or self._is_http:
            # Reduce RTSP/HTTP latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get video properties
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._is_file else -1
        
        if resize:
            self._width, self._height = resize
        
        # Calculate intrinsics from FOV
        self._intrinsics = self._compute_intrinsics()
        
        # Initialize estimators (lazy loaded)
        self._depth_estimator = None
        self._pose_estimator = None
        
        logger.info(f"LiveVideoSource: {self._width}x{self._height} @ {self._fps:.1f} FPS")
        logger.info(f"  Source: {source}")
        logger.info(f"  Pose: {pose_model}, Depth: {depth_model}")
    
    def _compute_intrinsics(self) -> Dict[str, float]:
        """Compute intrinsics from FOV."""
        fx = self._width / (2 * np.tan(np.radians(self._fov_deg) / 2))
        fy = fx  # Assume square pixels
        return {
            'fx': fx, 'fy': fy,
            'cx': self._width / 2,
            'cy': self._height / 2,
            'width': self._width,
            'height': self._height,
        }
    
    def _init_estimators(self):
        """Initialize depth and pose estimators."""
        # Import here to avoid circular imports
        try:
            from src.pose import get_pose_estimator
            from src.depth import get_depth_estimator
            
            # Initialize pose estimator
            self._pose_estimator = get_pose_estimator(self._pose_model_name)
            self._pose_estimator.set_intrinsics_from_dict(self._intrinsics)
            logger.info(f"Initialized pose estimator: {self._pose_model_name}")
            
            # Initialize depth estimator
            self._depth_estimator = get_depth_estimator(self._depth_model_name)
            logger.info(f"Initialized depth estimator: {self._depth_model_name}")
            
        except ImportError as e:
            logger.warning(f"Could not import estimators: {e}, using fallback")
            self._use_fallback = True
            self._init_fallback()
    
    def _smooth_pose(self, pose: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """Apply exponential moving average smoothing to pose."""
        if len(self._pose_history) == 0:
            self._pose_history.append(pose.copy())
            return pose
        
        # Smooth translation with EMA
        prev_pose = self._pose_history[-1]
        smoothed = pose.copy()
        smoothed[:3, 3] = alpha * pose[:3, 3] + (1 - alpha) * prev_pose[:3, 3]
        
        # Keep rotation from current pose (rotation smoothing is tricky)
        self._pose_history.append(smoothed.copy())
        
        # Keep only last 10 poses for memory
        if len(self._pose_history) > 10:
            self._pose_history = self._pose_history[-10:]
        
        return smoothed
    
    def _estimate_depth_scale(self, depth: np.ndarray, pose: np.ndarray) -> float:
        """
        Estimate scale factor to align monocular pose with depth.
        
        Uses median depth at image center as reference.
        """
        if depth is None:
            return 1.0
        
        # Get median depth in center region
        h, w = depth.shape
        center_region = depth[h//3:2*h//3, w//3:2*w//3]
        valid_depths = center_region[(center_region > 0.1) & (center_region < 10.0)]
        
        if len(valid_depths) < 100:
            return self._depth_scale_factor
        
        median_depth = np.median(valid_depths)
        
        # For first few frames, establish baseline
        if not self._scale_initialized and len(self._pose_history) > 3:
            # Estimate scale from pose translation magnitude vs depth
            translations = [p[:3, 3] for p in self._pose_history[-5:]]
            avg_translation = np.mean([np.linalg.norm(t) for t in translations])
            
            if avg_translation > 0.001:  # Avoid divide by zero
                # Assume median depth corresponds to typical scene depth
                self._depth_scale_factor = median_depth / max(avg_translation * 10, 0.1)
                self._depth_scale_factor = np.clip(self._depth_scale_factor, 0.1, 10.0)
                self._scale_initialized = True
                logger.info(f"Initialized depth scale factor: {self._depth_scale_factor:.3f}")
        
        return self._depth_scale_factor
    
    def _scale_pose(self, pose: np.ndarray, scale: float) -> np.ndarray:
        """Scale pose translation by given factor."""
        scaled = pose.copy()
        scaled[:3, 3] *= scale
        return scaled
    
    def _fetch_synchronized_frame(self) -> Optional[Dict]:
        """Fetch synchronized frame bundle (image + pose + depth) from server.
        
        This ensures all data is from the same moment, avoiding desync issues.
        Returns dict with 'rgb', 'pose', 'depth', 'intrinsics' or None on failure.
        """
        if not hasattr(self, '_frame_server_url') or not self._frame_server_url:
            return None
        
        try:
            import urllib.request
            import json
            import base64
            import cv2
            
            with urllib.request.urlopen(self._frame_server_url, timeout=2.0) as response:
                data = json.loads(response.read().decode())
            
            result = {}
            
            # Decode RGB frame
            if data.get('has_frame') and data.get('frame_jpg_b64'):
                jpg_bytes = base64.b64decode(data['frame_jpg_b64'])
                jpg_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    result['rgb'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Decode pose
            if data.get('has_pose') and data.get('pose'):
                result['pose'] = np.array(data['pose'], dtype=np.float64)
            
            # Decode depth
            if data.get('has_depth') and data.get('depth_png_b64'):
                depth_bytes = base64.b64decode(data['depth_png_b64'])
                depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
                depth_uint16 = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
                if depth_uint16 is not None:
                    scale = data.get('depth_scale', 5000)
                    result['depth'] = depth_uint16.astype(np.float32) / scale
            
            # Get intrinsics
            if data.get('intrinsics'):
                result['intrinsics'] = data['intrinsics']
            
            result['timestamp'] = data.get('timestamp')
            result['frame_idx'] = data.get('frame_idx')
            
            if result.get('rgb') is None:
                return None
            
            return result
            
        except Exception as e:
            logger.debug(f"Sync fetch failed: {e}")
        
        return None
    
    def _fetch_server_pose(self) -> Optional[np.ndarray]:
        """Fetch ground truth pose from server (e.g., TUM simulator)."""
        if not self._pose_server_url:
            return None
        
        try:
            import urllib.request
            import json
            
            with urllib.request.urlopen(self._pose_server_url, timeout=0.5) as response:
                data = json.loads(response.read().decode())
                
            if data.get('has_pose') and data.get('pose'):
                pose = np.array(data['pose'], dtype=np.float64)
                return pose
        except Exception as e:
            logger.debug(f"Failed to fetch pose from server: {e}")
        
        return None
    
    def _fetch_server_depth(self) -> Optional[np.ndarray]:
        """Fetch ground truth depth from server (e.g., TUM simulator)."""
        if not hasattr(self, '_depth_server_url'):
            return None
        
        try:
            import urllib.request
            import json
            import base64
            import cv2
            
            with urllib.request.urlopen(self._depth_server_url, timeout=0.5) as response:
                data = json.loads(response.read().decode())
                
            if data.get('has_depth') and data.get('depth_png_b64'):
                # Decode base64 PNG
                depth_bytes = base64.b64decode(data['depth_png_b64'])
                depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
                depth_uint16 = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
                
                if depth_uint16 is not None:
                    # Convert back to meters
                    scale = data.get('scale_factor', 5000)
                    depth = depth_uint16.astype(np.float32) / scale
                    return depth
        except Exception as e:
            logger.debug(f"Failed to fetch depth from server: {e}")
        
        return None
    
    def _fetch_server_intrinsics(self) -> Optional[Dict]:
        """Fetch camera intrinsics from server."""
        if not hasattr(self, '_intrinsics_url'):
            return None
        
        try:
            import urllib.request
            import json
            
            with urllib.request.urlopen(self._intrinsics_url, timeout=1.0) as response:
                data = json.loads(response.read().decode())
                return data
        except Exception as e:
            logger.debug(f"Failed to fetch intrinsics from server: {e}")
        
        return None
    
    def _init_fallback(self):
        """Initialize fallback ORB-based estimation."""
        import cv2
        self._orb = cv2.ORB_create(nfeatures=2000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._K = np.array([
            [self._intrinsics['fx'], 0, self._intrinsics['cx']],
            [0, self._intrinsics['fy'], self._intrinsics['cy']],
            [0, 0, 1]
        ])
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
        self._current_pose = np.eye(4)
    
    def _estimate_fallback(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Fallback pose estimation using ORB."""
        import cv2
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return None, self._current_pose.copy()
        
        matches = self._bf.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        
        if len(matches) >= 8:
            pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
            
            E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K)
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = t.flatten()
                self._current_pose = self._current_pose @ np.linalg.inv(T_rel)
        
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        return None, self._current_pose.copy()  # No depth in fallback
    
    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames from video source."""
        import cv2
        
        # Initialize estimators on first iteration
        if self._pose_estimator is None and self._depth_estimator is None:
            self._use_fallback = False
            self._init_estimators()
        
        frame_idx = 0
        # Use source FPS if target_fps is 0 or None
        effective_fps = self._target_fps if self._target_fps and self._target_fps > 0 else self._fps
        frame_interval = 1.0 / effective_fps
        last_frame_time = 0
        
        logger.info(f"Streaming at {effective_fps:.1f} FPS (source: {self._fps:.1f}, target: {self._target_fps})")
        
        # Try to get intrinsics from server if using server pose
        if self._use_server_pose:
            server_intrinsics = self._fetch_server_intrinsics()
            if server_intrinsics:
                self._intrinsics.update(server_intrinsics)
                logger.info(f"Using server intrinsics: fx={self._intrinsics['fx']:.1f}")
        
        while True:
            if self._max_frames and frame_idx >= self._max_frames:
                break
            
            # Rate limiting
            now = time.time()
            if now - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue
            
            rgb = None
            pose = None
            depth = None
            
            # Use synchronized fetching when both GT pose and depth are needed
            if self._use_synchronized_fetch:
                sync_data = self._fetch_synchronized_frame()
                if sync_data:
                    rgb = sync_data.get('rgb')
                    pose = sync_data.get('pose')
                    depth = sync_data.get('depth')
                    if sync_data.get('intrinsics'):
                        self._intrinsics.update(sync_data['intrinsics'])
                    if frame_idx < 3:
                        logger.info(f"Frame {frame_idx}: sync OK - rgb={rgb.shape if rgb is not None else None}, pose={pose is not None}, depth={depth is not None}")
                else:
                    logger.warning(f"Frame {frame_idx}: sync fetch failed")
                    continue
            else:
                # Traditional: read MJPEG then fetch pose/depth separately
                ret, frame = self._cap.read()
                
                if not ret:
                    if self._is_file:
                        break  # End of file
                    elif self._is_rtsp or self._is_http:
                        logger.warning("Frame grab failed, reconnecting...")
                        time.sleep(1.0)
                        src = int(self._source) if self._is_webcam else self._source
                        self._cap = cv2.VideoCapture(src)
                        continue
                    else:
                        continue
                
                # Resize if needed
                if self._resize:
                    frame = cv2.resize(frame, self._resize)
                
                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get pose - prefer server ground truth if available
                if self._use_server_pose:
                    pose = self._fetch_server_pose()
                    if pose is not None:
                        logger.debug(f"Using server pose for frame {frame_idx}")
                
                # Get depth - prefer server ground truth if available  
                if self._use_server_depth:
                    depth = self._fetch_server_depth()
            
            if rgb is None:
                continue
                
            last_frame_time = now
            
            # Fall back to estimation if no server pose (and not using sync fetch)
            if pose is None and not self._use_synchronized_fetch:
                if hasattr(self, '_use_fallback') and self._use_fallback:
                    _, pose = self._estimate_fallback(rgb)
                elif self._pose_estimator is not None:
                    pose_result = self._pose_estimator.estimate(rgb)
                    pose = pose_result.pose
                
                # Apply pose improvements only for estimated poses
                if pose is not None:
                    if self._scale_pose_to_depth:
                        # Get depth for scaling
                        depth_result = self._depth_estimator.estimate(rgb)
                        depth_for_scale = depth_result.depth if depth_result else None
                        if depth_for_scale is not None:
                            scale = self._estimate_depth_scale(depth_for_scale, pose)
                            pose = self._scale_pose(pose, scale)
                    
                    if self._enable_smooth_pose:
                        pose = self._smooth_pose(pose, alpha=0.7)
            
            # Fall back to depth estimation if no server depth (and no sync depth)
            if depth is None and not self._use_synchronized_fetch:
                if hasattr(self, '_use_fallback') and self._use_fallback:
                    depth = None
                elif self._depth_estimator is not None:
                    depth_result = self._depth_estimator.estimate(rgb)
                    depth = depth_result.depth if depth_result else None
            
            yield Frame(
                idx=frame_idx,
                timestamp=now,
                rgb=rgb,
                depth=depth,
                pose=pose,
                intrinsics=self._intrinsics.copy(),
            )
            
            frame_idx += 1
    
    def __len__(self) -> int:
        """Return frame count (-1 for live streams)."""
        if self._max_frames:
            return self._max_frames
        return self._frame_count if self._frame_count > 0 else -1
    
    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics."""
        return self._intrinsics.copy()
    
    def release(self):
        """Release video capture."""
        if self._cap:
            self._cap.release()


# Need time module for LiveVideoSource
import time
