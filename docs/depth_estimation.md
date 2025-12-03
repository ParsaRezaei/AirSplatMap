# Depth Estimation

AirSplatMap supports monocular depth estimation for RGB-only inputs where depth sensors aren't available.

## Available Methods

| Method | Speed | Quality | GPU | Metric | Description |
|--------|-------|---------|-----|--------|-------------|
| `midas` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ | MiDaS DPT-Large |
| `midas_small` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ❌ | MiDaS Small (fast) |
| `midas_hybrid` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ | MiDaS DPT-Hybrid |
| `depth_anything` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ | Depth Anything V2 |
| `depth_anything_vits` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ | Depth Anything Small |
| `zoedepth` | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ZoeDepth (metric) |

**Metric**: Outputs metric depth (meters) vs relative depth

## Quick Start

```python
from benchmarks.depth.benchmark_depth import get_depth_estimator, list_depth_estimators

# List available methods
print(list_depth_estimators())

# Create estimator
estimator = get_depth_estimator("midas")

# Estimate depth
rgb = cv2.imread("image.jpg")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

depth = estimator.estimate(rgb)  # HxW float32

# Visualize
import matplotlib.pyplot as plt
plt.imshow(depth, cmap='plasma')
plt.colorbar(label='Depth')
plt.show()
```

## Method Details

### MiDaS (Recommended)

Robust monocular depth estimation from Intel.

```python
estimator = get_depth_estimator("midas")
```

**Variants:**
- `midas` - DPT-Large (best quality)
- `midas_small` - Small model (fastest)
- `midas_hybrid` - DPT-Hybrid (balanced)

**Output:**
- Relative (inverse) depth
- Higher values = closer
- Scale-invariant

### Depth Anything V2 (Best Quality)

State-of-the-art foundation model for depth.

```python
estimator = get_depth_estimator("depth_anything")
```

**Variants:**
- `depth_anything` - ViT-Large (best)
- `depth_anything_vits` - ViT-Small (faster)
- `depth_anything_vitb` - ViT-Base (balanced)

**Requirements:**
```bash
pip install transformers
```

### ZoeDepth (Metric Depth)

Outputs actual metric depth in meters.

```python
estimator = get_depth_estimator("zoedepth")
```

**Advantage:**
- Directly usable for 3DGS (no scale ambiguity)
- Trained on indoor/outdoor datasets

**Output:**
- Metric depth in meters
- Range typically 0.1-10m indoors

## Usage Examples

### Basic Depth Estimation

```python
from benchmarks.depth.benchmark_depth import get_depth_estimator
import cv2
import numpy as np

# Initialize
estimator = get_depth_estimator("midas")

# Load image
rgb = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)

# Estimate
depth = estimator.estimate(rgb)

# Convert to point cloud (with intrinsics)
fx, fy = 525, 525
cx, cy = 319.5, 239.5
H, W = depth.shape

# Create point cloud
u, v = np.meshgrid(np.arange(W), np.arange(H))
z = depth
x = (u - cx) * z / fx
y = (v - cy) * z / fy

points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
```

### Depth + Pose for 3DGS

```python
from src.pipeline.frames import VideoSource
from src.pose import get_pose_estimator
from benchmarks.depth.benchmark_depth import get_depth_estimator

# Setup
pose_est = get_pose_estimator("orb")
depth_est = get_depth_estimator("midas")

pose_est.set_intrinsics(fx=525, fy=525, cx=319.5, cy=239.5)

# Process video
for frame in VideoSource("video.mp4"):
    # Estimate pose
    pose_result = pose_est.estimate(frame.rgb)
    
    # Estimate depth
    depth = depth_est.estimate(frame.rgb)
    
    # Scale depth to metric (if using relative depth)
    # This is approximate - better to use zoedepth for metric
    depth_metric = depth * 5.0  # Scale factor
    
    # Now can use with 3DGS engine
    engine.add_frame(
        frame_id=frame.idx,
        rgb=frame.rgb,
        depth=depth_metric,
        pose=pose_result.pose
    )
```

### Scale Alignment (for relative depth)

When using MiDaS or Depth Anything with ground truth:

```python
from benchmarks.depth.benchmark_depth import compute_depth_metrics

# Get metrics with automatic alignment
metrics = compute_depth_metrics(
    pred=estimated_depth,
    gt=ground_truth_depth,
    align=True  # Least-squares scale+shift alignment
)

print(f"Scale: {metrics['scale']:.3f}")
print(f"Shift: {metrics['shift']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}m")

# Apply alignment to future frames
aligned_depth = metrics['scale'] * estimated_depth + metrics['shift']
```

### Compare Methods

```python
methods = ['midas', 'midas_small', 'depth_anything']
results = {}

for method in methods:
    est = get_depth_estimator(method)
    
    times = []
    for rgb in images:
        t0 = time.time()
        depth = est.estimate(rgb)
        times.append(time.time() - t0)
    
    results[method] = {
        'fps': 1.0 / np.mean(times),
        'mean_depth': np.mean(depth)
    }

print(results)
```

## Integration with Pipeline

### Automatic Depth Estimation

```python
from src.pipeline.frames import WebcamSource

# Source with automatic depth estimation
source = WebcamSource(
    camera_id=0,
    depth_estimator="midas",  # Auto-estimate depth
    pose_estimator="orb",
    intrinsics={'fx': 525, 'fy': 525, 'cx': 319.5, 'cy': 239.5}
)

for frame in source:
    # frame.depth is estimated automatically
    print(f"Frame {frame.idx}: depth range [{frame.depth.min():.2f}, {frame.depth.max():.2f}]")
```

### With Dashboard

The web dashboard can show estimated depth in real-time:

```python
from dashboard.web_dashboard import Server

server = Server()
server.add_live_source(
    name="webcam",
    source="0",  # Camera ID
    pose_method="orb",
    depth_method="midas"  # Estimate depth
)
server.start()
```

## Depth Metrics

Standard evaluation metrics:

| Metric | Description | Better |
|--------|-------------|--------|
| AbsRel | Mean absolute relative error | Lower |
| SqRel | Mean squared relative error | Lower |
| RMSE | Root mean squared error (m) | Lower |
| RMSElog | RMSE in log space | Lower |
| δ < 1.25 | % within 25% of GT | Higher |
| δ < 1.25² | % within 56% of GT | Higher |
| δ < 1.25³ | % within 95% of GT | Higher |

## Benchmarks

On TUM RGB-D `fr1_desk`:

| Method | AbsRel | RMSE (m) | δ < 1.25 | FPS |
|--------|--------|----------|----------|-----|
| midas | 0.12 | 0.38 | 0.85 | 15 |
| midas_small | 0.18 | 0.52 | 0.78 | 35 |
| depth_anything | 0.09 | 0.31 | 0.91 | 12 |
| zoedepth | 0.11 | 0.35 | 0.88 | 8 |

Run benchmarks:
```bash
python benchmarks/depth/benchmark_depth.py --methods midas depth_anything zoedepth
```

## Tips

### For Best Quality
- Use `depth_anything` or `zoedepth`
- Process at full resolution
- Use consistent lighting

### For Speed
- Use `midas_small`
- Reduce input resolution
- Batch process if possible

### For Metric Depth
- Use `zoedepth` directly
- Or calibrate MiDaS scale with known measurements
- Or align to sparse LiDAR/SfM points

### Handling Failures
```python
depth = estimator.estimate(rgb)

# Check for invalid depth
if np.isnan(depth).any():
    depth = np.nan_to_num(depth, nan=0.0)

# Clamp extreme values
depth = np.clip(depth, 0.1, 10.0)
```

## Troubleshooting

### Out of Memory
```python
# Use smaller model
estimator = get_depth_estimator("midas_small")

# Or reduce input size
rgb_small = cv2.resize(rgb, (320, 240))
depth_small = estimator.estimate(rgb_small)
depth = cv2.resize(depth_small, (640, 480))
```

### Slow Performance
```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors
```bash
# Install dependencies
pip install timm transformers
pip install torch torchvision --upgrade
```
