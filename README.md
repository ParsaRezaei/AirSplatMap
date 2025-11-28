# AirSplatMap

An extensible, online 3D Gaussian Splatting mapping pipeline.

## Overview

AirSplatMap provides a modular framework for incremental 3D Gaussian Splatting (3DGS) from live or replayed streams of RGB/RGB-D frames with poses. The main contribution is the pipeline/orchestration layer; the actual 3DGS implementation sits behind a clean interface allowing different backends to be swapped.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FrameSource   │────▶ OnlineGSPipeline  ────▶   BaseGSEngine   │
│  (TUM, COLMAP,  │     │                  │     │  (Graphdeco,    │
│   video, etc.)  │     │  - Frame loop    │     │   SplaTAM, etc) │
└─────────────────┘     │  - RS correction │     └─────────────────┘
                        │  - Metrics       │
                        └──────────────────┘
```

### Key Components

- **`src/engines/base.py`**: Abstract `BaseGSEngine` interface that all 3DGS backends must implement
- **`src/engines/graphdeco_engine.py`**: Implementation using the original Graphdeco 3DGS codebase
- **`src/pipeline/frames.py`**: `Frame` dataclass and `FrameSource` abstractions (includes `TumRGBDSource`)
- **`src/pipeline/online_gs.py`**: Main `OnlineGSPipeline` orchestrator
- **`src/pipeline/rs_corrector.py`**: Rolling shutter correction abstractions (placeholder for now)

## Installation

1. **Clone and setup the workspace** (if not already done):
   ```bash
   cd /path/to/workspace
   ```

2. **Install dependencies**:
   ```bash
   cd AirSplatMap
   pip install -r requirements.txt
   ```

3. **Ensure Graphdeco 3DGS is available**:
   The `GraphdecoEngine` expects the Graphdeco gaussian-splatting repository to be available. It searches in:
   - `GRAPHDECO_PATH` environment variable
   - `../gaussian-splatting` (relative to AirSplatMap)
   
   Make sure the CUDA extensions are compiled:
   ```bash
   cd gaussian-splatting/submodules/diff-gaussian-rasterization
   pip install .
   cd ../simple-knn
   pip install .
   ```

## Usage

### Quick Start with TUM RGB-D Dataset

```bash
# Run from AirSplatMap directory
python scripts/online_tum_graphdeco.py \
    --dataset-root ../datasets \
    --max-frames 200 \
    --steps-per-frame 5 \
    --render-every 50
```

### Python API

```python
from src.engines import GraphdecoEngine
from src.pipeline import OnlineGSPipeline
from src.pipeline.frames import TumRGBDSource

# Create components
engine = GraphdecoEngine()
source = TumRGBDSource(dataset_root="../datasets")

# Create pipeline
pipeline = OnlineGSPipeline(
    engine=engine,
    frame_source=source,
    steps_per_frame=5,
    render_every=50,
)

# Run
summary = pipeline.run(max_frames=200)

# Or step manually
for frame in source:
    metrics = pipeline.step(frame)
    if frame.idx % 50 == 0:
        preview = engine.render_view(frame.pose, frame.image_size)

# Save results
pipeline.save_final()
```

### Adding a New Engine

Implement the `BaseGSEngine` interface:

```python
from src.engines.base import BaseGSEngine

class MyEngine(BaseGSEngine):
    def initialize_scene(self, intrinsics, config):
        # Setup your 3DGS implementation
        pass
    
    def add_frame(self, frame_id, rgb, depth, pose_world_cam):
        # Add a new observation
        pass
    
    def optimize_step(self, n_steps=1):
        # Run optimization
        return {'loss': 0.0, 'num_gaussians': 0}
    
    def render_view(self, pose_world_cam, image_size):
        # Render from a pose
        return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    def save_state(self, path):
        # Save to disk
        pass
    
    def load_state(self, path):
        # Load from disk
        pass
```

### Adding a New Frame Source

Implement the `FrameSource` interface:

```python
from src.pipeline.frames import FrameSource, Frame

class MySource(FrameSource):
    def __iter__(self):
        for i, data in enumerate(self.data):
            yield Frame(
                idx=i,
                timestamp=data['ts'],
                rgb=data['rgb'],
                depth=data['depth'],
                pose=data['pose'],
                intrinsics=self.intrinsics,
            )
    
    def __len__(self):
        return len(self.data)
```

## Project Structure

```
AirSplatMap/
├── src/
│   ├── __init__.py
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseGSEngine interface
│   │   └── graphdeco_engine.py  # Graphdeco implementation
│   └── pipeline/
│       ├── __init__.py
│       ├── frames.py            # Frame, FrameSource, TumRGBDSource
│       ├── online_gs.py         # OnlineGSPipeline
│       └── rs_corrector.py      # RS correction (placeholder)
├── scripts/
│   └── online_tum_graphdeco.py  # Demo script
├── requirements.txt
└── README.md
```

## Configuration

### Engine Config (passed to `initialize_scene`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sh_degree` | 3 | Spherical harmonics degree |
| `white_background` | False | Use white background |
| `position_lr_init` | 0.00016 | Initial position learning rate |
| `densify_grad_threshold` | 0.0002 | Gradient threshold for densification |
| `densify_until_iter` | 15000 | Stop densification after N iterations |
| `lambda_dssim` | 0.2 | Weight for SSIM loss |
| `recency_weight` | 0.7 | Weight for recent frames in sampling |

### Pipeline Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps_per_frame` | 5 | Optimization steps per new frame |
| `warmup_frames` | 1 | Frames before starting optimization |
| `render_every` | 0 | Render preview every N frames |
| `save_every` | 0 | Save checkpoint every N frames |

## Supported Datasets

- **TUM RGB-D**: Use `TumRGBDSource` - expects TUM benchmark format with `rgb/`, `depth/`, `rgb.txt`, `depth.txt`, `groundtruth.txt`
- **COLMAP**: (Planned) `ColmapSource` for COLMAP sparse reconstructions
- **Video**: (Planned) `VideoSource` for video files with external pose tracking

## License

See LICENSE file for details.
