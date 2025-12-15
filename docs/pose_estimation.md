# Pose Estimation

AirSplatMap includes multiple visual odometry methods for camera tracking when ground truth poses aren't available.

> 📊 **View pose benchmarks**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)

---

## Available Methods

### Classical Methods (CPU)

| Method | Speed | Accuracy | GPU | Description |
|--------|-------|----------|-----|-------------|
| `orb` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ORB features + PnP |
| `sift` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | SIFT features + PnP |
| `robust_flow` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Optical flow with outlier rejection |
| `flow` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | Basic optical flow |
| `keyframe` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Keyframe-based ORB |

### Deep Learning Methods (GPU)

| Method | Speed | Accuracy | GPU | Description |
|--------|-------|----------|-----|-------------|
| `loftr` | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | LoFTR dense matching |
| `superpoint` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | SuperPoint + SuperGlue |
| `lightglue` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | LightGlue fast matcher (CVPR 2023) |
| `raft` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | RAFT optical flow (NeurIPS 2020) |
| `r2d2` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | R2D2 reliable descriptors |
| `roma` | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | RoMa dense matcher (CVPR 2024) |

### Specialized Methods

| Method | Type | Description |
|--------|------|-------------|
| RGB-D VO | RGB-D | Visual odometry using depth sensor (\`src.pose.rgbd_vo\`) |
| Stereo VIO | Stereo | Stereo visual-inertial odometry (\`src.pose.stereo_vio\`) |
| External | Various | ORB-SLAM3, OpenVINS, DPVO, DROID-SLAM wrappers |

## Quick Start

\`\`\`python
from src.pose import get_pose_estimator, list_pose_estimators

# List available methods
print(list_pose_estimators())

# Create estimator
estimator = get_pose_estimator("orb")

# Set camera intrinsics
estimator.set_intrinsics_from_dict({'fx': 525, 'fy': 525, 'cx': 319.5, 'cy': 239.5})

# Estimate pose
result = estimator.estimate(rgb_image)

print(f"Pose:\n{result.pose}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Inliers: {result.num_inliers}")
\`\`\`

## Method Details

### ORB (Recommended for Speed)

Fast feature-based tracking using ORB descriptors.

\`\`\`python
estimator = get_pose_estimator("orb")
\`\`\`

**How it works:**
1. Detect ORB keypoints in current frame
2. Match with previous frame using brute-force
3. Estimate Essential matrix with RANSAC
4. Recover pose from Essential matrix

**Config:**
\`\`\`python
estimator = get_pose_estimator("orb",
    n_features=2000,        # Number of features
    scale_factor=1.2,       # Pyramid scale
    n_levels=8,             # Pyramid levels
)
\`\`\`

### LightGlue (Fast Deep Learning) ⭐ NEW

LightGlue is a fast and accurate deep learning-based feature matcher from CVPR 2023.

\`\`\`python
estimator = get_pose_estimator("lightglue")
\`\`\`

**How it works:**
1. Extract SuperPoint/DISK/ALIKED features
2. Match using lightweight attention mechanism
3. ~5-10x faster than SuperGlue with similar accuracy

**Requirements:**
\`\`\`bash
pip install lightglue  # or use kornia backend
\`\`\`

### RAFT (Dense Flow) ⭐ NEW

State-of-the-art optical flow for robust dense correspondence.

\`\`\`python
estimator = get_pose_estimator("raft", model='large')
\`\`\`

**How it works:**
1. Compute dense optical flow using RAFT
2. Sample reliable correspondences from flow
3. Estimate pose with RANSAC

**Requirements:**
- torchvision >= 0.14 (includes RAFT)

### RoMa (Wide Baseline) ⭐ NEW

RoMa handles challenging wide baseline matching using DINOv2 features.

\`\`\`python
estimator = get_pose_estimator("roma", model='outdoor')
\`\`\`

**Requirements:**
\`\`\`bash
pip install romatch
\`\`\`

### LoFTR (Best Accuracy, Slow)

Deep learning-based dense matching.

\`\`\`python
estimator = get_pose_estimator("loftr")
\`\`\`

**Requirements:**
- CUDA GPU
- \`pip install kornia\`

### Keyframe-Based (For Drift Reduction)

Maintains keyframes for loop closure.

\`\`\`python
estimator = get_pose_estimator("keyframe")
\`\`\`

## RGB-D Visual Odometry

For RGB-D cameras (like RealSense), use the dedicated RGB-D VO:

\`\`\`python
from src.pose.rgbd_vo import RGBDVO, create_rgbd_vo

# Simple creation
vo = create_rgbd_vo(intrinsics={'fx': 615, 'fy': 615, 'cx': 320, 'cy': 240})

# Or with full config
vo = RGBDVO(
    max_features=300,
    min_depth=0.1,
    max_depth=8.0,
    pose_smoothing=0.4,
)
vo.set_intrinsics(fx=615, fy=615, cx=320, cy=240)

# Process frames
result = vo.process(rgb, depth, timestamp)
pose = result.pose  # 4x4 camera-to-world
\`\`\`

## External VIO Systems

AirSplatMap provides wrappers for external VIO/SLAM systems:

\`\`\`python
from src.pose.external import list_available_backends, get_available_wrapper

# Check available backends
backends = list_available_backends()
# {'orbslam3': False, 'openvins': False, 'dpvo': False, 'droid_slam': False}

# Available wrappers (require separate installation):
# - ORB-SLAM3: Feature-based VO/VIO/SLAM
# - OpenVINS: Tightly-coupled VIO (MSCKF)
# - DPVO: Deep Patch Visual Odometry
# - DROID-SLAM: Deep visual SLAM
\`\`\`

### ORB-SLAM3 Example

\`\`\`python
from src.pose.external import ORBSlam3Wrapper

slam = ORBSlam3Wrapper(
    vocab_path="ORBvoc.txt",
    config_path="camera.yaml",
    sensor_type="IMU_STEREO"
)
slam.initialize()

result = slam.process(
    image=left_image,
    timestamp=time.time(),
    image_right=right_image,
    accel=np.array([ax, ay, az]),
    gyro=np.array([gx, gy, gz])
)
\`\`\`

## API Reference

### \`PoseResult\`

\`\`\`python
@dataclass
class PoseResult:
    pose: np.ndarray        # 4x4 camera-to-world matrix
    confidence: float       # 0-1 confidence score
    num_inliers: int        # Number of inlier matches
    tracking_status: str    # "ok", "lost", "initializing"
    is_keyframe: bool       # Whether this is a keyframe
\`\`\`

### \`BasePoseEstimator\`

\`\`\`python
class BasePoseEstimator(ABC):
    def set_intrinsics(self, K: np.ndarray) -> None:
        """Set 3x3 camera intrinsic matrix."""
        pass
    
    def set_intrinsics_from_dict(self, intrinsics: dict) -> None:
        """Set intrinsics from dict with fx, fy, cx, cy."""
        pass
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray) -> PoseResult:
        """Estimate pose from image."""
        pass
    
    def reset(self) -> None:
        """Reset tracking state."""
        pass
\`\`\`

## Integration with Pipeline

### Automatic Pose Estimation

\`\`\`python
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
\`\`\`

## Benchmarks

On TUM \`fr1_desk\`:

| Method | ATE (m) | RPE (m/f) | FPS |
|--------|---------|-----------|-----|
| orb | 0.089 | 0.023 | 45 |
| sift | 0.072 | 0.019 | 12 |
| robust_flow | 0.065 | 0.017 | 28 |
| lightglue | 0.055 | 0.015 | 15 |
| loftr | 0.048 | 0.012 | 3 |
| roma | 0.045 | 0.011 | 2 |

Run benchmarks:
\`\`\`bash
python -m benchmarks pose --methods orb sift lightglue loftr
\`\`\`

## Troubleshooting

### Tracking Lost Frequently

1. Ensure good lighting
2. Reduce motion blur (faster shutter)
3. Try \`robust_flow\` or \`lightglue\`
4. Increase feature count

### Drift Accumulation

1. Use \`keyframe\` method
2. Use RGB-D VO if depth available
3. Consider external VIO (ORB-SLAM3)

### GPU Methods Not Working

\`\`\`bash
# Install CUDA dependencies
pip install kornia torch torchvision --upgrade

# For LightGlue
pip install lightglue

# For RoMa
pip install romatch
\`\`\`

---

## See Also

- [Getting Started](getting_started.md) - Installation and first run
- [Depth Estimation](depth_estimation.md) - Monocular depth methods
- [Engines](engines.md) - 3DGS engine comparison
- [Benchmarks Guide](benchmarks.md) - Running comprehensive evaluations
- [ArduPilot Integration](ardupilot_integration.md) - Drone/rover support with MAVLink poses
- [API Reference](api_reference.md) - Python API documentation
