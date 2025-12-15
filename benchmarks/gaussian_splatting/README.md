# Gaussian Splatting Benchmark

Benchmark 3DGS engines on RGB-D datasets.

> 📊 **View Results Online**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)
>
> 📖 **Full Documentation**: [docs/engines.md](../../docs/engines.md)

---

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

| Engine | Speed | Real-time | Description |
|--------|-------|-----------|-------------|
| \`gsplat\` | ⭐⭐⭐⭐⭐ | ✅ | GSplat optimized implementation |
| \`graphdeco\` | ⭐⭐⭐ | ❌ | Original 3DGS implementation |
| \`monogs\` | ⭐⭐⭐⭐ | ✅ | MonoGS SLAM engine |
| \`splatam\` | ⭐⭐ | ❌ | SplaTAM RGB-D SLAM |
| \`gslam\` | ⭐⭐⭐ | ❌ | Gaussian-SLAM with submaps |

## Usage

\`\`\`bash
# Run all engines
python benchmarks/gaussian_splatting/benchmark_gs.py

# Specific engines
python benchmarks/gaussian_splatting/benchmark_gs.py --engines graphdeco gsplat

# Custom settings
python benchmarks/gaussian_splatting/benchmark_gs.py --max-frames 100 --iterations 100
\`\`\`

---

## See Also

- [Engines Guide](../../docs/engines.md) - Full engine documentation
- [Benchmarks Guide](../../docs/benchmarks.md) - Comprehensive benchmarking
- [Interactive Results](https://ParsaRezaei.github.io/AirSplatMap/) - View all benchmark results
