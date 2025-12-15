# Architecture

AirSplatMap is designed as a modular, extensible framework for real-time 3D Gaussian Splatting.

> 📚 **Related Documentation**:
> - [Getting Started](getting_started.md) - Installation and first run
> - [Engines](engines.md) - 3DGS engine comparison
> - [API Reference](api_reference.md) - Python API docs

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AirSplatMap                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Frame Source │───▶│   Pipeline   │───▶│   Engine     │               │
│  │              │    │              │    │              │               │
│  │ - TUM RGB-D  │    │ - Orchestrate│    │ - GraphDeco  │               │
│  │ - RealSense  │    │ - Metrics    │    │ - GSplat     │               │
│  │ - Webcam     │    │ - Callbacks  │    │ - MonoGS     │               │
│  │ - Video      │    │              │    │ - SplaTAM    │               │
│  │ - ArduPilot  │    │              │    │              │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│          │                   │                   │                       │
│          │                   │                   │                       │
│          ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Pose Est.    │    │   Viewer     │    │   Output     │               │
│  │              │    │              │    │              │               │
│  │ - ORB        │    │ - Dashboard  │    │ - .ply       │               │
│  │ - SIFT       │    │ - WebSocket  │    │ - Video      │               │
│  │ - LoFTR      │    │ - 3D Canvas  │    │ - Metrics    │               │
│  │ - SuperPoint │    │              │    │              │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Frame Sources (`src/pipeline/frames.py`)

Frame sources provide RGB-D frames with poses to the pipeline.

```python
from src.pipeline.frames import Frame, FrameSource

class Frame:
    idx: int                    # Frame index
    timestamp: float            # Unix timestamp
    rgb: np.ndarray            # HxWx3 RGB image
    depth: np.ndarray          # HxW depth map (optional)
    pose: np.ndarray           # 4x4 camera-to-world matrix
    intrinsics: dict           # fx, fy, cx, cy
```

**Available Sources:**

| Source | Description | Pose | Depth |
|--------|-------------|------|-------|
| `TumRGBDSource` | TUM RGB-D benchmark | Ground truth | Sensor |
| `RealSenseSource` | Intel RealSense camera | VIO/Estimated | Sensor |
| `WebcamSource` | USB webcam | Estimated | Estimated |
| `VideoSource` | Video file | Estimated | Estimated |
| `ArduPilotSource` | MAVLink telemetry | MAVLink | Optional |

### 2. Pipeline (`src/pipeline/online_gs.py`)

The pipeline orchestrates frame processing and engine updates.

```python
from src.pipeline import OnlineGSPipeline

pipeline = OnlineGSPipeline(
    engine=engine,
    frame_source=source,
    steps_per_frame=5,      # Optimization steps per frame
    warmup_frames=1,        # Frames before optimization
    render_every=10,        # Render preview interval
)

# Run all frames
summary = pipeline.run(max_frames=200)

# Or step manually
for frame in source:
    metrics = pipeline.step(frame)
```

### 3. Engines (`src/engines/`)

Engines implement the 3D Gaussian Splatting algorithm.

```python
from src.engines.base import BaseGSEngine

class BaseGSEngine(ABC):
    @abstractmethod
    def initialize_scene(self, intrinsics: dict, config: dict = None):
        """Initialize with camera parameters."""
        pass
    
    @abstractmethod
    def add_frame(self, frame_id: int, rgb: np.ndarray, 
                  depth: np.ndarray, pose_world_cam: np.ndarray):
        """Add a new observation."""
        pass
    
    @abstractmethod
    def optimize_step(self, n_steps: int = 1) -> dict:
        """Run optimization, return metrics."""
        pass
    
    @abstractmethod
    def render(self, pose: np.ndarray) -> np.ndarray:
        """Render from a viewpoint."""
        pass
    
    @abstractmethod
    def get_gaussians(self) -> dict:
        """Get Gaussian parameters (positions, colors, etc.)."""
        pass
```

### 4. Pose Estimation (`src/pose/`)

Visual odometry for camera tracking when ground truth isn't available.

```python
from src.pose import get_pose_estimator

estimator = get_pose_estimator("orb")  # or sift, loftr, superpoint
estimator.set_intrinsics(fx, fy, cx, cy)

result = estimator.estimate(rgb_image)
# result.pose: 4x4 matrix
# result.confidence: 0-1 score
# result.num_inliers: feature count
```

### 5. Depth Estimation (`src/depth/`)

Monocular depth estimation for RGB-only inputs.

```python
from benchmarks.depth.benchmark_depth import get_depth_estimator

estimator = get_depth_estimator("midas")
depth = estimator.estimate(rgb_image)  # HxW depth map
```

### 6. Viewer (`src/viewer/`, `dashboard/`)

Real-time visualization via WebSocket.

```python
from src.viewer import ViewerServer

server = ViewerServer(port=9003)
server.start()

# Send updates
server.send_frame(rgb, depth)
server.send_gaussians(positions, colors)
server.send_metrics(loss=0.01, psnr=25.3)
```

## Data Flow

```
                     ┌─────────────────┐
                     │   Frame Source   │
                     │                  │
                     │  RGB + Depth +   │
                     │     Pose         │
                     └────────┬─────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │         Pipeline              │
              │                               │
              │  1. Preprocess frame          │
              │  2. Add to engine             │
              │  3. Run optimization          │
              │  4. Collect metrics           │
              │  5. Send to viewer            │
              └───────────────┬───────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │  Engine  │      │  Viewer  │      │  Output  │
     │          │      │          │      │          │
     │ Gaussians│      │ Dashboard│      │  Files   │
     │ Render   │      │ WebGL    │      │  Video   │
     └──────────┘      └──────────┘      └──────────┘
```

## Extension Points

### Adding a New Engine

1. Create `src/engines/my_engine.py`
2. Inherit from `BaseGSEngine`
3. Implement required methods
4. Register in `src/engines/__init__.py`

```python
# src/engines/my_engine.py
from src.engines.base import BaseGSEngine

class MyEngine(BaseGSEngine):
    def initialize_scene(self, intrinsics, config=None):
        # Your initialization
        pass
    
    # ... implement other methods
```

### Adding a New Frame Source

1. Create source class inheriting from `FrameSource`
2. Implement `__iter__` and `__len__`

```python
from src.pipeline.frames import FrameSource, Frame

class MySource(FrameSource):
    def __iter__(self):
        for i, data in enumerate(self.data):
            yield Frame(
                idx=i,
                timestamp=time.time(),
                rgb=data['rgb'],
                depth=data['depth'],
                pose=data['pose'],
                intrinsics=self.intrinsics
            )
```

### Adding a New Pose Estimator

1. Create class inheriting from `BasePoseEstimator`
2. Implement `estimate()` method
3. Register in `src/pose/__init__.py`

```python
from src.pose import BasePoseEstimator, PoseResult

class MyEstimator(BasePoseEstimator):
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        # Your pose estimation
        return PoseResult(
            pose=np.eye(4),
            confidence=1.0,
            num_inliers=100
        )
```

## Configuration

### Engine Configuration

```python
engine_config = {
    'sh_degree': 3,              # Spherical harmonics degree
    'white_background': False,   # Background color
    'position_lr_init': 0.00016, # Learning rate
    'densify_grad_threshold': 0.0002,
}

engine = get_engine("gsplat", **engine_config)
```

### Pipeline Configuration

```python
pipeline_config = {
    'steps_per_frame': 5,    # Optimization steps
    'warmup_frames': 1,      # Frames before optimization
    'render_every': 10,      # Render interval
    'save_every': 100,       # Checkpoint interval
}

pipeline = OnlineGSPipeline(engine, source, **pipeline_config)
```

## Thread Safety

- Engines are **not thread-safe** - use from single thread
- ViewerServer handles WebSocket connections asynchronously
- Pipeline callbacks are called from main thread

## Memory Management

- Gaussians stored on GPU
- Images cached in RAM with configurable limit
- Use `gsplat` engine for 4x less memory than `graphdeco`
