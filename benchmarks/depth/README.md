# Depth Estimation Benchmark

Benchmark depth estimation methods against TUM RGB-D ground truth.

## Metrics

- **AbsRel**: Absolute relative error (lower is better)
- **SqRel**: Squared relative error (lower is better)  
- **RMSE**: Root mean squared error in meters (lower is better)
- **RMSElog**: RMSE in log space (lower is better)
- **δ < 1.25**: Accuracy within 25% (higher is better)
- **δ < 1.25²**: Accuracy within 56% (higher is better)
- **δ < 1.25³**: Accuracy within 95% (higher is better)

## Available Methods

- `midas`: MiDaS DPT-Large (relative depth)
- `midas_small`: MiDaS Small (faster)
- `depth_anything`: Depth Anything V2 Large
- `zoedepth`: ZoeDepth (metric depth)

## Usage

```bash
# Run all methods
python benchmarks/depth/benchmark_depth.py

# Specific methods
python benchmarks/depth/benchmark_depth.py --methods midas zoedepth

# Custom dataset
python benchmarks/depth/benchmark_depth.py --dataset-root /path/to/datasets
```
