# API Reference

Complete API reference for AirSplatMap.

> 📚 **Related Documentation**:
> - [Getting Started](getting_started.md) - Installation and first run
> - [Architecture](architecture.md) - System design overview
> - [Engines](engines.md) - 3DGS engine comparison

---

## Quick Import Reference

```python
# Core pipeline
from src.pipeline import OnlineGSPipeline
from src.pipeline.frames import Frame, TumRGBDSource, RealSenseSource

# Engines
from src.engines import get_engine, list_engines
from src.engines.base import BaseGSEngine

# Pose estimation
from src.pose import get_pose_estimator, list_pose_estimators

# Depth estimation
from src.depth import get_depth_estimator, list_depth_estimators

# Dashboard
from dashboard.web_dashboard import Server
```

## Engines (`src.engines`)

### `get_engine(name, **config)`

Get a 3DGS engine by name.

```python
from src.engines import get_engine

engine = get_engine("gsplat")
engine = get_engine("graphdeco", sh_degree=3)
```

**Parameters:**
- `name`: Engine name (`gsplat`, `graphdeco`, `monogs`, `splatam`, `gslam`, `photoslam`)
- `**config`: Engine-specific configuration

**Returns:** `BaseGSEngine` instance

### `list_engines()`

List available engines.

```python
from src.engines import list_engines

for name, info in list_engines().items():
    print(f"{name}: {info['description']} (available: {info['available']})")
```

**Returns:** `Dict[str, dict]` with engine info

### `BaseGSEngine`

Abstract base class for 3DGS engines.

```python
from src.engines.base import BaseGSEngine

class BaseGSEngine(ABC):
    @abstractmethod
    def initialize_scene(self, intrinsics: dict, config: dict = None) -> None:
        """
        Initialize the scene.
        
        Args:
            intrinsics: Camera parameters {fx, fy, cx, cy, width, height}
            config: Engine configuration
        """
        pass
    
    @abstractmethod
    def add_frame(self, frame_id: int, rgb: np.ndarray, 
                  depth: Optional[np.ndarray], pose_world_cam: np.ndarray) -> None:
        """
        Add observation to scene.
        
        Args:
            frame_id: Unique frame identifier
            rgb: HxWx3 uint8 RGB image
            depth: HxW float32 depth map (meters), optional
            pose_world_cam: 4x4 camera-to-world transformation matrix
        """
        pass
    
    @abstractmethod
    def optimize_step(self, n_steps: int = 1) -> dict:
        """
        Run optimization.
        
        Args:
            n_steps: Number of optimization iterations
            
        Returns:
            dict with 'loss', 'num_gaussians', etc.
        """
        pass
    
    @abstractmethod
    def render(self, pose_world_cam: np.ndarray, 
               width: int = None, height: int = None) -> np.ndarray:
        """
        Render scene from viewpoint.
        
        Args:
            pose_world_cam: 4x4 camera-to-world matrix
            width: Output width (default: initialization width)
            height: Output height (default: initialization height)
            
        Returns:
            HxWx3 uint8 RGB image
        """
        pass
    
    @abstractmethod
    def get_gaussians(self) -> dict:
        """
        Get Gaussian parameters.
        
        Returns:
            dict with 'positions', 'colors', 'opacities', 'scales', 'rotations'
        """
        pass
    
    @abstractmethod
    def get_num_gaussians(self) -> int:
        """Get number of Gaussians."""
        pass
    
    @abstractmethod
    def save_ply(self, path: str) -> None:
        """Save Gaussians to PLY file."""
        pass
    
    def save_state(self, path: str) -> None:
        """Save full checkpoint."""
        pass
    
    def load_state(self, path: str) -> None:
        """Load from checkpoint."""
        pass
```

---

## Pipeline (`src.pipeline`)

### `OnlineGSPipeline`

Main orchestrator for online 3DGS reconstruction.

```python
from src.pipeline import OnlineGSPipeline

pipeline = OnlineGSPipeline(
    engine=engine,
    frame_source=source,
    steps_per_frame=5,
    warmup_frames=1,
    render_every=10,
    save_every=0,
    output_dir="./output"
)
```

**Parameters:**
- `engine`: `BaseGSEngine` instance
- `frame_source`: `FrameSource` instance
- `steps_per_frame`: Optimization steps per frame (default: 5)
- `warmup_frames`: Frames before optimization starts (default: 1)
- `render_every`: Render preview every N frames (default: 0 = never)
- `save_every`: Save checkpoint every N frames (default: 0 = never)
- `output_dir`: Output directory path

**Methods:**

```python
# Run all frames
summary = pipeline.run(max_frames=None, callback=None)

# Step through one frame
metrics = pipeline.step(frame)

# Save final results
pipeline.save_final()
```

### `Frame`

Dataclass for a single observation.

```python
from src.pipeline.frames import Frame

@dataclass
class Frame:
    idx: int                          # Frame index
    timestamp: float                  # Unix timestamp
    rgb: np.ndarray                   # HxWx3 uint8
    depth: Optional[np.ndarray]       # HxW float32 (meters)
    pose: np.ndarray                  # 4x4 world-to-camera
    intrinsics: dict                  # {fx, fy, cx, cy}
    image_size: Tuple[int, int]       # (width, height)
```

### `TumRGBDSource`

Frame source for TUM RGB-D datasets.

```python
from src.pipeline.frames import TumRGBDSource

source = TumRGBDSource(
    dataset_root="/path/to/tum",
    sequence="fr1_desk",              # or full path to sequence
    max_frames=None,
    skip_frames=1,
    start_frame=0
)

for frame in source:
    print(f"Frame {frame.idx}: shape={frame.rgb.shape}")
```

### `RealSenseSource`

Frame source for Intel RealSense cameras.

```python
from src.pipeline import RealSenseSource

source = RealSenseSource(
    serial_number=None,               # Specific camera, or None for first
    width=640,
    height=480,
    fps=30,
    enable_depth=True,
    align_depth=True,
    pose_estimator=None               # Or 'orb', 'sift', etc.
)

for frame in source:
    print(f"Frame {frame.idx}: depth range [{frame.depth.min():.2f}, {frame.depth.max():.2f}]")
```

---

## Pose Estimation (`src.pose`)

### `get_pose_estimator(name, **config)`

Get a pose estimator by name.

```python
from src.pose import get_pose_estimator

estimator = get_pose_estimator("orb")
estimator = get_pose_estimator("loftr", device="cuda")
```

**Available methods:** `orb`, `sift`, `robust_flow`, `flow`, `keyframe`, `loftr`, `superpoint`, `lightglue`, `raft`, `r2d2`, `roma`

### `list_pose_estimators()`

List available pose estimators.

```python
from src.pose import list_pose_estimators

for name, info in list_pose_estimators().items():
    print(f"{name}: {info['description']}")
```

### `BasePoseEstimator`

Base class for pose estimators.

```python
from src.pose import BasePoseEstimator

class BasePoseEstimator(ABC):
    def set_intrinsics(self, fx: float, fy: float, 
                       cx: float, cy: float) -> None:
        """Set camera intrinsics."""
        pass
    
    def set_intrinsics_from_dict(self, intrinsics: dict) -> None:
        """Set intrinsics from dict."""
        pass
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray, 
                 depth: np.ndarray = None) -> PoseResult:
        """
        Estimate camera pose.
        
        Args:
            rgb: HxWx3 uint8 RGB image
            depth: HxW float32 depth (optional)
            
        Returns:
            PoseResult with pose, confidence, num_inliers
        """
        pass
    
    def reset(self) -> None:
        """Reset tracking state."""
        pass
```

### `PoseResult`

Result of pose estimation.

```python
@dataclass
class PoseResult:
    pose: np.ndarray        # 4x4 camera-to-world matrix
    confidence: float       # 0-1 confidence score
    num_inliers: int        # Number of inlier matches
    timestamp: float        # Estimation timestamp
    
    @property
    def position(self) -> np.ndarray:
        """Get 3D position (translation)."""
        return self.pose[:3, 3]
    
    @property
    def rotation(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        return self.pose[:3, :3]
```

---

## Depth Estimation

### `get_depth_estimator(name)`

```python
from src.depth import get_depth_estimator

estimator = get_depth_estimator("midas")
result = estimator.estimate(rgb_image)
depth = result.depth  # HxW float32
```

**Available methods:** `midas`, `midas_small`, `depth_anything_v2`, `depth_anything_v3`, `depth_pro`, `depth_pro_lite`, `stereo`, `ground_truth`

### `list_depth_estimators()`

```python
from src.depth import list_depth_estimators

for name, info in list_depth_estimators().items():
    print(f"{name}: {info['description']}")
```

---

## Dashboard (`dashboard`)

### `Server`

Web dashboard server.

```python
from dashboard.web_dashboard import Server

server = Server(http_port=9002, ws_port=9003)

# Add live source
server.add_live_source(
    name="camera",
    source="rtsp://localhost:8554/stream",
    pose_method="orb",
    depth_method="midas"
)

# Start server
server.start()  # Blocking

# Or in background
server.start_background()
```

**Methods:**

```python
# Add live source
server.add_live_source(name, source, source_type='live', 
                       pose_method='orb', depth_method='midas')

# Get available datasets
datasets = server.get_datasets()

# Get run history
history = server.get_history()

# Send manual update (for custom pipelines)
server.send_frame(rgb, depth, frame_id)
server.send_metrics(fps=17.2, loss=0.02, gaussians=180000)
server.send_points(positions, colors)
```

---

## Viewer (`src.viewer`)

### `ViewerServer`

WebSocket server for real-time visualization.

```python
from src.viewer import ViewerServer

server = ViewerServer(port=9003)
server.start()

# Send updates
server.send_frame(rgb, depth)
server.send_gaussians(positions, colors, opacities)
server.send_metrics(fps=17.2, loss=0.02, psnr=24.5)
```

### `ViewerClient`

Client for connecting to viewer server.

```python
from src.viewer import ViewerClient

client = ViewerClient("ws://localhost:9003")
client.connect()

# Receive updates
for message in client.receive():
    print(message['type'], message.get('frame_id'))
```

---

## Evaluation (`src.evaluation`)

### `compute_ate(estimated_trajectory, ground_truth_trajectory)`

Compute Absolute Trajectory Error.

```python
from src.evaluation.metrics import compute_ate

ate = compute_ate(estimated, ground_truth)
print(f"ATE RMSE: {ate['rmse']:.4f}m")
```

### `compute_rpe(estimated_trajectory, ground_truth_trajectory)`

Compute Relative Pose Error.

```python
from src.evaluation.metrics import compute_rpe

rpe = compute_rpe(estimated, ground_truth)
print(f"RPE Trans: {rpe['trans_rmse']:.4f}m")
print(f"RPE Rot: {rpe['rot_rmse']:.4f}deg")
```

### `compute_depth_metrics(pred, gt)`

Compute depth evaluation metrics.

```python
from benchmarks.depth.benchmark_depth import compute_depth_metrics

metrics = compute_depth_metrics(predicted_depth, ground_truth_depth, align=True)
print(f"AbsRel: {metrics['abs_rel']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}m")
print(f"δ<1.25: {metrics['delta1']:.4f}")
```

---

## Utilities

### Transform Utilities

```python
from src.utils import (
    pose_to_matrix,       # Convert pose tuple to 4x4 matrix
    matrix_to_pose,       # Convert 4x4 matrix to pose tuple
    invert_pose,          # Invert 4x4 transformation
    compose_poses,        # Multiply transformations
    rotation_to_quaternion,
    quaternion_to_rotation,
)
```

### Point Cloud Utilities

```python
from src.utils import (
    depth_to_points,      # Depth image to 3D points
    project_points,       # 3D points to 2D image
    transform_points,     # Apply transformation to points
    filter_points_fov,    # Filter points in field of view
)
```

### Image Utilities

```python
from src.utils import (
    resize_image,         # Resize with aspect ratio
    normalize_image,      # Normalize to [0, 1]
    denormalize_image,    # Back to [0, 255]
    rgb_to_gray,          # Color to grayscale
)
```

---

## Type Hints

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Common types
ImageArray = np.ndarray  # HxWx3 uint8
DepthArray = np.ndarray  # HxW float32
PoseMatrix = np.ndarray  # 4x4 float64
Intrinsics = Dict[str, float]  # {fx, fy, cx, cy}
```
