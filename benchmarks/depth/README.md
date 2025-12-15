# Depth Estimation Benchmark

Benchmark depth estimation methods against TUM RGB-D ground truth.

> 📊 **View Results Online**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)
>
> 📖 **Full Documentation**: [docs/depth_estimation.md](../../docs/depth_estimation.md)

---

## Metrics

- **AbsRel**: Absolute relative error (lower is better)
- **SqRel**: Squared relative error (lower is better)  
- **RMSE**: Root mean squared error in meters (lower is better)
- **RMSElog**: RMSE in log space (lower is better)
- **δ < 1.25**: Accuracy within 25% (higher is better)
- **δ < 1.25²**: Accuracy within 56% (higher is better)
- **δ < 1.25³**: Accuracy within 95% (higher is better)

## Available Methods

| Method | Description | Metric Depth |
|--------|-------------|--------------|
| `depth_pro` | Apple Depth Pro (metric) | ✅ |
| `depth_anything_v3` | Depth Anything V3 | ❌ |
| `depth_anything_v2` | Depth Anything V2 | ❌ |
| `midas` | MiDaS DPT-Large (relative depth) | ❌ |
| `midas_small` | MiDaS Small (faster) | ❌ |

## Usage

\`\`\`bash
# Run all methods
python benchmarks/depth/benchmark_depth.py

# Specific methods
python benchmarks/depth/benchmark_depth.py --methods midas depth_pro

# Custom dataset
python benchmarks/depth/benchmark_depth.py --dataset-root /path/to/datasets
\`\`\`

---

## See Also

- [Depth Estimation Guide](../../docs/depth_estimation.md) - Full documentation
- [Benchmarks Guide](../../docs/benchmarks.md) - Comprehensive benchmarking
- [Interactive Results](https://ParsaRezaei.github.io/AirSplatMap/) - View all benchmark results
