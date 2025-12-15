# Pose Estimation Benchmark

Benchmark visual odometry methods against TUM RGB-D ground truth trajectories.

> 📊 **View Results Online**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)
>
> 📖 **Full Documentation**: [docs/pose_estimation.md](../../docs/pose_estimation.md)

---

## Metrics

- **ATE RMSE**: Absolute Trajectory Error (root mean square) in meters
- **ATE Mean**: Average absolute error in meters
- **RPE Trans**: Relative Pose Error for translation (m/frame)
- **RPE Rot**: Relative Pose Error for rotation (deg/frame)
- **FPS**: Processing speed (frames per second)
- **Inliers**: Average matched features per frame

## Available Methods

### Classical Methods (CPU)

| Method | Speed | Description |
|--------|-------|-------------|
| \`orb\` | ⭐⭐⭐⭐⭐ | ORB features + PnP |
| \`sift\` | ⭐⭐⭐ | SIFT features + PnP |
| \`robust_flow\` | ⭐⭐⭐⭐ | Optical flow with outlier rejection |
| \`keyframe\` | ⭐⭐⭐⭐ | Keyframe-based ORB |

### Deep Learning Methods (GPU)

| Method | Speed | Description |
|--------|-------|-------------|
| \`loftr\` | ⭐⭐ | LoFTR dense matching |
| \`superpoint\` | ⭐⭐⭐ | SuperPoint + SuperGlue |
| \`lightglue\` | ⭐⭐⭐⭐ | LightGlue fast matcher |
| \`r2d2\` | ⭐⭐⭐ | R2D2 reliable descriptors |
| \`roma\` | ⭐⭐ | RoMa dense matcher |
| \`raft\` | ⭐⭐⭐ | RAFT optical flow |

## Usage

\`\`\`bash
# Run all methods
python benchmarks/pose/benchmark_pose.py

# Specific methods
python benchmarks/pose/benchmark_pose.py --methods orb sift loftr

# Custom dataset
python benchmarks/pose/benchmark_pose.py --dataset-root /path/to/datasets

# Specific sequence
python benchmarks/pose/benchmark_pose.py --dataset fr1_desk
\`\`\`

---

## See Also

- [Pose Estimation Guide](../../docs/pose_estimation.md) - Full documentation
- [Benchmarks Guide](../../docs/benchmarks.md) - Comprehensive benchmarking
- [Interactive Results](https://ParsaRezaei.github.io/AirSplatMap/) - View all benchmark results
