# Depth Estimation

AirSplatMap supports monocular depth estimation for RGB-only inputs where depth sensors aren't available.

> 📊 **View depth benchmarks**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)

---

## Available Methods

| Method | Speed | Quality | GPU | Metric | Description |
|--------|-------|---------|-----|--------|-------------|
| `depth_pro` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | Apple Depth Pro (metric, sharp) |
| `depth_pro_lite` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ | Depth Pro Lite (faster) |
| `ground_truth` | N/A | N/A | ❌ | ✅ | Passthrough for sensor depth |
| `depth_anything_v3` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ | Depth Anything V3 (latest) |
| `depth_anything_v2` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ | Depth Anything V2 |
| `midas` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ | MiDaS DPT-Large |
| `midas_small` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ❌ | MiDaS Small (fast) |
| `stereo` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ✅ | SGBM stereo matching |

**Metric**: Outputs metric depth (meters) vs relative depth

## Quick Start

\`\`\`python
from src.depth import get_depth_estimator, list_depth_estimators

# List available methods
print(list_depth_estimators())

# Create estimator - metric depth recommended!
estimator = get_depth_estimator("depth_pro")

# Estimate depth
import cv2
rgb = cv2.imread("image.jpg")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

result = estimator.estimate(rgb)  # DepthResult object
depth = result.depth  # HxW float32 in meters

# Visualize
import matplotlib.pyplot as plt
plt.imshow(depth, cmap='plasma')
plt.colorbar(label='Depth (m)')
plt.show()
\`\`\`

## Method Details

### Apple Depth Pro (Recommended) ⭐

Sharp monocular metric depth from Apple. Best quality with direct metric output.

\`\`\`python
from src.depth import get_depth_estimator

estimator = get_depth_estimator("depth_pro")
result = estimator.estimate(rgb)
depth = result.depth  # Metric depth in meters!
\`\`\`

**Features:**
- Zero-shot metric depth (no calibration needed)
- Excellent boundary sharpness
- Automatic focal length estimation
- ~0.3s per 2.25MP image

**Variants:**
- \`depth_pro\` - Full model (best quality)
- \`depth_pro_lite\` - Faster with lower resolution processing

**Installation:**
\`\`\`bash
# Already included as submodule
cd submodules/ml-depth-pro
pip install -e .

# Download pretrained models
source get_pretrained_models.sh
\`\`\`

**Get Focal Length:**
\`\`\`python
from src.depth.depth_pro import DepthProEstimator

estimator = DepthProEstimator()
depth, focal_px = estimator.estimate_with_focal(rgb)
print(f"Estimated focal length: {focal_px:.1f} pixels")
\`\`\`

**Paper:** [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)

### Depth Anything V3 (Best Relative Depth)

Latest foundation model for relative depth estimation.

\`\`\`python
estimator = get_depth_estimator("depth_anything_v3")
# Or: get_depth_estimator("dav3")
\`\`\`

**Variants:**
- \`depth_anything_v3\` / \`dav3\` - Latest version
- \`depth_anything_v2\` / \`dav2\` - V2 (also excellent)

**Model sizes:**
- \`small\` (fast)
- \`base\` (balanced)
- \`large\` (accurate)
- \`giant\` (best, V3 only)

### MiDaS (Robust Relative Depth)

Robust monocular depth estimation from Intel.

\`\`\`python
estimator = get_depth_estimator("midas")
\`\`\`

**Variants:**
- \`midas\` - DPT-Large (best quality)
- \`midas_small\` - Small model (fastest)
- \`midas_large\` - DPT-Large explicit

### Stereo Depth

For stereo camera pairs (e.g., RealSense IR cameras):

\`\`\`python
from src.depth import get_depth_estimator

stereo = get_depth_estimator("stereo", baseline=0.05, focal_length=382.6)
result = stereo.estimate_stereo(left_ir, right_ir)
depth = result.depth  # Metric depth in meters
\`\`\`

## Scaling Relative Depth

When using relative depth methods (MiDaS, Depth Anything), you need to scale to metric:

\`\`\`python
from src.depth import DepthScaler

# Using sparse 3D points (e.g., from SfM or LiDAR)
scaler = DepthScaler(method='ransac')
metric_depth = scaler.scale_to_metric(relative_depth, points_3d, points_2d)

# Or with known reference depth
metric_depth = relative_depth * scale_factor + shift
\`\`\`

## Temporal Filtering

For video sequences, use temporal filtering to reduce flickering:

\`\`\`python
from src.depth import DepthFilter

filter = DepthFilter(temporal_alpha=0.3)

for frame in video:
    depth = estimator.estimate(frame).depth
    filtered_depth = filter.temporal_filter(depth)
\`\`\`

## Integration with Pipeline

### Automatic Depth Estimation

\`\`\`python
from src.pipeline.frames import WebcamSource

# Source with automatic depth estimation
source = WebcamSource(
    camera_id=0,
    depth_estimator="depth_pro",  # Auto-estimate depth
    pose_estimator="orb",
    intrinsics={'fx': 525, 'fy': 525, 'cx': 319.5, 'cy': 239.5}
)

for frame in source:
    # frame.depth is estimated automatically
    print(f"Frame {frame.idx}: depth range [{frame.depth.min():.2f}, {frame.depth.max():.2f}]")
\`\`\`

## Benchmarks

On TUM RGB-D \`fr1_desk\`:

| Method | AbsRel | RMSE (m) | δ < 1.25 | FPS |
|--------|--------|----------|----------|-----|
| depth_pro | 0.07 | 0.25 | 0.94 | 12 |
| depth_anything_v3 | 0.08 | 0.28 | 0.92 | 10 |
| depth_anything_v2 | 0.09 | 0.31 | 0.91 | 12 |
| midas | 0.12 | 0.38 | 0.85 | 15 |
| midas_small | 0.18 | 0.52 | 0.78 | 35 |

Run benchmarks:
\`\`\`bash
python -m benchmarks depth --methods depth_pro depth_anything_v3 midas
\`\`\`

## Tips

### For Best Quality
- Use \`depth_pro\` for metric depth
- Use \`depth_anything_v3\` for relative depth
- Process at full resolution

### For Speed
- Use \`midas_small\` or \`depth_pro_lite\`
- Reduce input resolution
- Batch process if possible

### For Metric Depth
- Prefer \`depth_pro\` (direct metric output)
- Or use \`stereo\` with calibrated cameras
- Scale relative depth with sparse 3D points

## Troubleshooting

### Out of Memory
\`\`\`python
# Use smaller model
estimator = get_depth_estimator("midas_small")

# Or reduce input size
import cv2
rgb_small = cv2.resize(rgb, (320, 240))
depth_small = estimator.estimate(rgb_small).depth
depth = cv2.resize(depth_small, (640, 480))
\`\`\`

### Slow Performance
\`\`\`bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
\`\`\`

### Import Errors
\`\`\`bash
# Install dependencies
pip install timm transformers torch torchvision --upgrade

# For Apple Depth Pro
cd submodules/ml-depth-pro
pip install -e .
source get_pretrained_models.sh
\`\`\`

---

## See Also

- [Getting Started](getting_started.md) - Installation and first run
- [Pose Estimation](pose_estimation.md) - Visual odometry methods
- [Engines](engines.md) - 3DGS engine comparison
- [Benchmarks Guide](benchmarks.md) - Running comprehensive evaluations
- [API Reference](api_reference.md) - Python API documentation
