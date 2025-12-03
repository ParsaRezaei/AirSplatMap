# 3DGS Engines

AirSplatMap supports multiple 3D Gaussian Splatting engines, each with different tradeoffs.

## Engine Comparison

| Engine | Speed | Memory | Quality | Real-time | RGB-D | Mono |
|--------|-------|--------|---------|-----------|-------|------|
| **gsplat** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ |
| **graphdeco** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ✅ | ✅ |
| **monogs** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ | ✅ |
| **splatam** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ❌ | ✅ | ❌ |
| **gslam** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ✅ | ❌ |
| **photoslam** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ |

## Choosing an Engine

```
                    Need real-time?
                         │
              ┌──────────┴──────────┐
              │                     │
             YES                   NO
              │                     │
        Have depth?           Best quality?
              │                     │
       ┌──────┴──────┐        ┌─────┴─────┐
       │             │        │           │
      YES           NO       YES          NO
       │             │        │           │
    monogs       gsplat  graphdeco    splatam
```

### Quick Recommendation

- **Real-time demo**: `gsplat` or `monogs`
- **Best quality**: `graphdeco`
- **RGB-D SLAM**: `splatam` or `monogs`
- **Drone mapping**: `gsplat` (low memory)

## Engine Details

### GSplat (Recommended for Real-time)

The fastest engine with excellent memory efficiency.

```python
from src.engines import get_engine

engine = get_engine("gsplat")
```

**Pros:**
- 4x less memory than GraphDeco
- 17+ FPS real-time
- Active development by Nerfstudio team

**Cons:**
- Slightly lower quality than GraphDeco
- Requires `pip install gsplat`

**Config:**
```python
engine = get_engine("gsplat",
    sh_degree=3,
    densify_grad_threshold=0.0002,
    position_lr_init=0.00016,
)
```

### GraphDeco (Best Quality)

The original 3DGS implementation from INRIA.

```python
engine = get_engine("graphdeco")
```

**Pros:**
- Highest reconstruction quality
- Well-tested, reference implementation

**Cons:**
- High memory usage
- 2-5 FPS (not real-time)
- Requires CUDA extension build

**Config:**
```python
engine = get_engine("graphdeco",
    sh_degree=3,
    white_background=False,
    lambda_dssim=0.2,
)
```

### MonoGS (Real-time SLAM)

Gaussian Splatting SLAM from CVPR'24.

```python
engine = get_engine("monogs")
```

**Pros:**
- Integrated SLAM (tracking + mapping)
- Good balance of speed and quality
- Works with monocular RGB

**Cons:**
- Moderate memory usage
- Requires specific config

**Config:**
```python
engine = get_engine("monogs",
    tracking=True,  # Enable tracking
    mapping=True,   # Enable mapping
)
```

### SplaTAM (RGB-D SLAM)

Dense RGB-D SLAM with Gaussian Splatting.

```python
engine = get_engine("splatam")
```

**Pros:**
- Excellent for RGB-D sensors
- Dense reconstruction

**Cons:**
- Requires depth input
- Slower than others
- High memory usage

### Gaussian-SLAM (GSLAM)

Submap-based Gaussian SLAM.

```python
engine = get_engine("gslam")
```

**Pros:**
- Handles large scenes with submaps
- Good for long sequences

**Cons:**
- Complex configuration
- Requires RGB-D

### Photo-SLAM (Photorealistic)

Photorealistic SLAM from CVPR'24.

```python
engine = get_engine("photoslam")
```

**Pros:**
- Very high visual quality
- Real-time capable

**Cons:**
- Requires C++ build
- More complex setup

## Usage Examples

### Basic Usage

```python
from src.engines import get_engine

# Initialize
engine = get_engine("gsplat")
engine.initialize_scene({
    'fx': 525.0, 'fy': 525.0,
    'cx': 319.5, 'cy': 239.5,
    'width': 640, 'height': 480
})

# Add frames
engine.add_frame(
    frame_id=0,
    rgb=rgb_image,      # HxWx3 uint8
    depth=depth_image,  # HxW float32 (meters)
    pose_world_cam=pose # 4x4 matrix
)

# Optimize
metrics = engine.optimize_step(n_steps=10)
print(f"Loss: {metrics['loss']:.4f}, Gaussians: {metrics['num_gaussians']}")

# Render
rendered = engine.render(pose)  # HxWx3 uint8

# Get Gaussians
gaussians = engine.get_gaussians()
# gaussians['positions']: Nx3
# gaussians['colors']: Nx3
# gaussians['opacities']: N
```

### Switching Engines

```python
# Compare engines on same data
for engine_name in ['gsplat', 'graphdeco', 'monogs']:
    engine = get_engine(engine_name)
    engine.initialize_scene(intrinsics)
    
    for frame in frames:
        engine.add_frame(frame.idx, frame.rgb, frame.depth, frame.pose)
        engine.optimize_step(5)
    
    psnr = evaluate_psnr(engine, test_frames)
    print(f"{engine_name}: PSNR={psnr:.2f}dB")
```

### Custom Configuration

```python
# High-quality settings
engine = get_engine("graphdeco",
    sh_degree=4,                    # More spherical harmonics
    densify_grad_threshold=0.0001,  # More densification
    position_lr_init=0.0001,        # Slower learning
    lambda_dssim=0.3,               # More perceptual loss
)

# Fast settings
engine = get_engine("gsplat",
    sh_degree=2,                    # Fewer harmonics
    densify_grad_threshold=0.001,   # Less densification
    position_lr_init=0.001,         # Faster learning
)
```

## Engine API Reference

### `BaseGSEngine`

All engines implement this interface:

```python
class BaseGSEngine(ABC):
    @abstractmethod
    def initialize_scene(self, intrinsics: dict, config: dict = None) -> None:
        """Initialize the scene with camera intrinsics."""
        pass
    
    @abstractmethod
    def add_frame(self, frame_id: int, rgb: np.ndarray, 
                  depth: Optional[np.ndarray], pose_world_cam: np.ndarray) -> None:
        """Add a new observation to the scene."""
        pass
    
    @abstractmethod
    def optimize_step(self, n_steps: int = 1) -> dict:
        """Run optimization steps. Returns metrics dict."""
        pass
    
    @abstractmethod
    def render(self, pose_world_cam: np.ndarray, 
               width: int = None, height: int = None) -> np.ndarray:
        """Render the scene from a viewpoint."""
        pass
    
    @abstractmethod
    def get_gaussians(self) -> dict:
        """Get Gaussian parameters."""
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
        """Save full state for checkpoint."""
        pass
    
    def load_state(self, path: str) -> None:
        """Load state from checkpoint."""
        pass
```

### Metrics Dictionary

`optimize_step()` returns:

```python
{
    'loss': float,           # Total loss
    'loss_rgb': float,       # RGB reconstruction loss
    'loss_depth': float,     # Depth loss (if applicable)
    'num_gaussians': int,    # Current Gaussian count
    'densified': int,        # Gaussians added this step
    'pruned': int,           # Gaussians removed this step
}
```

### Gaussians Dictionary

`get_gaussians()` returns:

```python
{
    'positions': np.ndarray,   # Nx3 XYZ positions
    'colors': np.ndarray,      # Nx3 RGB colors [0-1]
    'opacities': np.ndarray,   # N opacities [0-1]
    'scales': np.ndarray,      # Nx3 scales
    'rotations': np.ndarray,   # Nx4 quaternions
    'sh_coeffs': np.ndarray,   # NxCx3 spherical harmonics
}
```

## Benchmarks

See [benchmarks.md](benchmarks.md) for detailed performance comparisons.

Quick summary on TUM `fr1_desk` (200 frames):

| Engine | PSNR (dB) | FPS | Memory (GB) | Gaussians |
|--------|-----------|-----|-------------|-----------|
| gsplat | 24.8 | 17.2 | 2.1 | 185,000 |
| graphdeco | 25.3 | 3.5 | 8.4 | 210,000 |
| monogs | 24.5 | 10.1 | 4.2 | 175,000 |
| splatam | 24.2 | 0.8 | 6.8 | 195,000 |
