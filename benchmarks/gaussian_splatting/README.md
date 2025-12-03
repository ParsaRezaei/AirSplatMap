# Gaussian Splatting Benchmark

Benchmark 3DGS engines on RGB-D datasets.

## Metrics

### Quality
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **LPIPS**: Learned Perceptual Similarity (lower is better)

### Efficiency
- **FPS**: Processing speed
- **Train Time**: Total optimization time
- **# Gaussians**: Final Gaussian count
- **Memory**: Peak GPU memory usage

## Available Engines

- `graphdeco`: Original 3DGS implementation
- `gsplat`: GSplat optimized implementation
- `monogs`: MonoGS SLAM engine
- `splatam`: SplaTAM engine

## Usage

```bash
# Run all engines
python benchmarks/gaussian_splatting/benchmark_gs.py

# Specific engines
python benchmarks/gaussian_splatting/benchmark_gs.py --engines graphdeco gsplat

# Custom settings
python benchmarks/gaussian_splatting/benchmark_gs.py --max-frames 100 --iterations 100
```
