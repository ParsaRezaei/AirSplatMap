# Pose Estimation

AirSplatMap includes multiple visual odometry methods for camera tracking when ground truth poses aren't available.

## Available Methods

| Method | Speed | Accuracy | GPU | Description |
|--------|-------|----------|-----|-------------|
| `orb` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ORB features + PnP |
| `sift` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | SIFT features + PnP |
| `robust_flow` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Optical flow with outlier rejection |
| `flow` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | Basic optical flow |
| `keyframe` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Keyframe-based ORB |
| `loftr` | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | LoFTR deep matching |
| `superpoint` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | SuperPoint + SuperGlue |

## Quick Start

```python
from src.pose import get_pose_estimator, list_pose_estimators

# List available methods
print(list_pose_estimators())

# Create estimator
estimator = get_pose_estimator("orb")

# Set camera intrinsics
estimator.set_intrinsics(fx=525, fy=525, cx=319.5, cy=239.5)

# Estimate pose
result = estimator.estimate(rgb_image)

print(f"Pose:\n{result.pose}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Inliers: {result.num_inliers}")
```

## Method Details

### ORB (Recommended for Speed)

Fast feature-based tracking using ORB descriptors.

```python
estimator = get_pose_estimator("orb")
```

**How it works:**
1. Detect ORB keypoints in current frame
2. Match with previous frame using brute-force
3. Estimate Essential matrix with RANSAC
4. Recover pose from Essential matrix

**Config:**
```python
estimator = get_pose_estimator("orb",
    n_features=2000,        # Number of features
    scale_factor=1.2,       # Pyramid scale
    n_levels=8,             # Pyramid levels
    ransac_threshold=1.0,   # RANSAC threshold (pixels)
)
```

### SIFT (Better Accuracy)

More robust features at the cost of speed.

```python
estimator = get_pose_estimator("sift")
```

**How it works:**
1. Detect SIFT keypoints (scale-invariant)
2. Match with FLANN-based matcher
3. Filter matches with ratio test
4. Estimate pose with PnP

### Robust Flow (Balanced)

Optical flow with geometric verification.

```python
estimator = get_pose_estimator("robust_flow")
```

**How it works:**
1. Compute dense optical flow
2. Sample reliable correspondences
3. Apply geometric consistency check
4. Estimate pose with RANSAC

**Config:**
```python
estimator = get_pose_estimator("robust_flow",
    flow_method='farneback',  # or 'dis'
    consistency_threshold=1.0,
    min_correspondences=100,
)
```

### Keyframe-Based (For Drift Reduction)

Maintains keyframes for loop closure.

```python
estimator = get_pose_estimator("keyframe")
```

**How it works:**
1. Track features between consecutive frames
2. Create keyframe when motion threshold exceeded
3. Match against keyframes for drift correction
4. Bundle adjustment (optional)

### LoFTR (Best Accuracy, Slow)

Deep learning-based dense matching.

```python
estimator = get_pose_estimator("loftr")
```

**Requirements:**
- CUDA GPU
- `pip install kornia`

**How it works:**
1. Extract deep features with transformer
2. Dense matching in feature space
3. Coarse-to-fine refinement
4. Pose from correspondences

### SuperPoint + SuperGlue

State-of-the-art learned features.

```python
estimator = get_pose_estimator("superpoint")
```

**Requirements:**
- CUDA GPU
- SuperGlue weights

## Usage Examples

### Basic Tracking

```python
from src.pose import get_pose_estimator
import cv2

# Initialize
estimator = get_pose_estimator("orb")
estimator.set_intrinsics(fx=525, fy=525, cx=319.5, cy=239.5)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = estimator.estimate(rgb)
    
    # Get position
    position = result.pose[:3, 3]
    print(f"Position: {position}")
    
    # Visualize
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == 27:
        break
```

### With Depth

```python
estimator = get_pose_estimator("orb", use_depth=True)

# Estimate with depth
result = estimator.estimate(rgb, depth=depth_image)
```

### Compare Methods

```python
methods = ['orb', 'sift', 'robust_flow']
results = {}

for method in methods:
    est = get_pose_estimator(method)
    est.set_intrinsics(fx=525, fy=525, cx=319.5, cy=239.5)
    
    times = []
    for frame in frames:
        t0 = time.time()
        result = est.estimate(frame.rgb)
        times.append(time.time() - t0)
    
    results[method] = {
        'fps': 1.0 / np.mean(times),
        'inliers': np.mean([r.num_inliers for r in results])
    }

print(results)
```

### Reset Tracking

```python
# Reset when tracking is lost
if result.confidence < 0.3:
    estimator.reset()
    print("Tracking lost, resetting...")
```

## API Reference

### `PoseResult`

```python
@dataclass
class PoseResult:
    pose: np.ndarray        # 4x4 camera-to-world matrix
    confidence: float       # 0-1 confidence score
    num_inliers: int        # Number of inlier matches
    timestamp: float        # Estimation timestamp
    
    @property
    def position(self) -> np.ndarray:
        """3D position (translation)."""
        return self.pose[:3, 3]
    
    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation matrix."""
        return self.pose[:3, :3]
```

### `BasePoseEstimator`

```python
class BasePoseEstimator(ABC):
    def set_intrinsics(self, fx: float, fy: float, 
                       cx: float, cy: float) -> None:
        """Set camera intrinsics."""
        pass
    
    def set_intrinsics_from_dict(self, intrinsics: dict) -> None:
        """Set intrinsics from dict with fx, fy, cx, cy."""
        pass
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray, 
                 depth: np.ndarray = None) -> PoseResult:
        """Estimate pose from image."""
        pass
    
    def reset(self) -> None:
        """Reset tracking state."""
        pass
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get accumulated trajectory."""
        pass
```

## Integration with Pipeline

### Automatic Pose Estimation

```python
from src.pipeline.frames import WebcamSource
from src.pose import get_pose_estimator

# Source with pose estimation
source = WebcamSource(
    camera_id=0,
    pose_estimator=get_pose_estimator("orb"),
    intrinsics={'fx': 525, 'fy': 525, 'cx': 319.5, 'cy': 239.5}
)

for frame in source:
    # frame.pose is estimated automatically
    print(f"Frame {frame.idx}: position = {frame.pose[:3, 3]}")
```

### RealSense VIO

```python
from src.pipeline import RealSenseSource

# Use RealSense's built-in tracking
source = RealSenseSource(use_t265=True)  # T265 tracking camera

# Or with visual odometry
source = RealSenseSource(
    pose_estimator="robust_flow",  # Use our VO
)
```

## Benchmarks

On TUM `fr1_desk`:

| Method | ATE (m) | RPE (m/f) | FPS |
|--------|---------|-----------|-----|
| orb | 0.089 | 0.023 | 45 |
| sift | 0.072 | 0.019 | 12 |
| robust_flow | 0.065 | 0.017 | 28 |
| loftr | 0.048 | 0.012 | 3 |
| superpoint | 0.052 | 0.014 | 8 |

Run benchmarks:
```bash
python benchmarks/pose/benchmark_pose.py --methods orb sift robust_flow
```

## Troubleshooting

### Tracking Lost Frequently

1. Ensure good lighting
2. Reduce motion blur (faster shutter)
3. Try `robust_flow` instead of `orb`
4. Increase feature count

### Drift Accumulation

1. Use `keyframe` method
2. Reduce `steps_per_frame` in pipeline
3. Add loop closure (coming soon)

### GPU Methods Not Working

```bash
# Install CUDA dependencies
pip install kornia torch torchvision --upgrade
```
