#!/usr/bin/env python3
"""
AirSplatMap Benchmark CLI
=========================

Main entry point for benchmarks. Supports both the new subcommand style
and the original run.py style for backward compatibility.

Usage (new style):
    python -m benchmarks pose --methods orb sift --max-frames 200
    python -m benchmarks depth --methods midas --max-frames 50
    python -m benchmarks gs --engines graphdeco --max-frames 30
    python -m benchmarks all --quick

Usage (original style - redirects to benchmarks.run):
    python -m benchmarks.run --quick
    python -m benchmarks.run --pose --depth
    python -m benchmarks.run --multi-dataset --comprehensive
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import RESULTS_DIR, PLOTS_DIR, DATASETS_DIR, get_host_results_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_pose(args):
    """Run pose estimation benchmark."""
    from benchmarks.runners.pose import PoseBenchmark
    
    benchmark = PoseBenchmark(
        methods=args.methods,
        dataset_root=args.dataset_root or DATASETS_DIR,
    )
    
    results = benchmark.run(
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )
    
    benchmark.print_table()
    
    if args.output:
        benchmark.save_results(args.output)
    elif args.save:
        host_dir = get_host_results_dir()
        host_dir.mkdir(parents=True, exist_ok=True)
        benchmark.save_results(host_dir / 'pose_benchmark.json')
    
    return results


def cmd_depth(args):
    """Run depth estimation benchmark."""
    from benchmarks.runners.depth import DepthBenchmark
    
    benchmark = DepthBenchmark(
        methods=args.methods,
        dataset_root=args.dataset_root or DATASETS_DIR,
    )
    
    results = benchmark.run(
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )
    
    benchmark.print_table()
    
    if args.output:
        benchmark.save_results(args.output)
    elif args.save:
        host_dir = get_host_results_dir()
        host_dir.mkdir(parents=True, exist_ok=True)
        benchmark.save_results(host_dir / 'depth_benchmark.json')
    
    return results


def cmd_gs(args):
    """Run Gaussian splatting benchmark."""
    from benchmarks.runners.gs import GSBenchmark
    
    benchmark = GSBenchmark(
        engines=args.engines,
        dataset_root=args.dataset_root or DATASETS_DIR,
    )
    
    results = benchmark.run(
        max_frames=args.max_frames,
        iterations=args.iterations,
    )
    
    benchmark.print_table()
    
    if args.output:
        benchmark.save_results(args.output)
    elif args.save:
        host_dir = get_host_results_dir()
        host_dir.mkdir(parents=True, exist_ok=True)
        benchmark.save_results(host_dir / 'gs_benchmark.json')
    
    return results


def cmd_all(args):
    """Run all benchmarks."""
    from benchmarks.runners.pose import PoseBenchmark
    from benchmarks.runners.depth import DepthBenchmark
    from benchmarks.runners.gs import GSBenchmark
    
    print("\n" + "=" * 70)
    print("  AIRSPLATMAP COMPREHENSIVE BENCHMARK")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70 + "\n")
    
    results = {'pose': [], 'depth': [], 'gs': []}
    
    # Quick mode adjustments
    if args.quick:
        max_frames_pose = 100
        max_frames_depth = 30
        max_frames_gs = 30
        iterations = 20
    else:
        max_frames_pose = args.max_frames or 200
        max_frames_depth = args.max_frames or 50
        max_frames_gs = args.max_frames or 50
        iterations = args.iterations or 30
    
    # Create host-specific results directory
    host_dir = get_host_results_dir()
    host_dir.mkdir(parents=True, exist_ok=True)
    
    # Pose benchmark
    if not args.skip_pose:
        logger.info("Running POSE benchmark...")
        pose_bench = PoseBenchmark(dataset_root=DATASETS_DIR)
        results['pose'] = pose_bench.run(max_frames=max_frames_pose, skip_frames=2)
        pose_bench.print_table()
        if args.save:
            pose_bench.save_results(host_dir / 'pose_benchmark.json')
    
    # Depth benchmark
    if not args.skip_depth:
        logger.info("Running DEPTH benchmark...")
        depth_bench = DepthBenchmark(dataset_root=DATASETS_DIR)
        results['depth'] = depth_bench.run(max_frames=max_frames_depth, skip_frames=5)
        depth_bench.print_table()
        if args.save:
            depth_bench.save_results(host_dir / 'depth_benchmark.json')
    
    # GS benchmark
    if not args.skip_gs:
        logger.info("Running GS benchmark...")
        gs_bench = GSBenchmark(dataset_root=DATASETS_DIR)
        results['gs'] = gs_bench.run(max_frames=max_frames_gs, iterations=iterations)
        gs_bench.print_table()
        if args.save:
            gs_bench.save_results(host_dir / 'gs_benchmark.json')
    
    # Generate plots
    if args.plots:
        generate_plots(results)
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print(f"  Results: {host_dir}")
    print("=" * 70)
    
    return results


def cmd_report(args):
    """Generate report from existing results."""
    import json
    
    print("Generating report from existing results...")
    
    results = {'pose': [], 'depth': [], 'gs': []}
    
    # Load existing results from host-specific directory
    host_dir = get_host_results_dir()
    pose_file = host_dir / 'pose_benchmark.json'
    depth_file = host_dir / 'depth_benchmark.json'
    gs_file = host_dir / 'gs_benchmark.json'
    
    if pose_file.exists():
        with open(pose_file) as f:
            results['pose'] = json.load(f)
        print(f"  Loaded {len(results['pose'])} pose results")
    
    if depth_file.exists():
        with open(depth_file) as f:
            results['depth'] = json.load(f)
        print(f"  Loaded {len(results['depth'])} depth results")
    
    if gs_file.exists():
        with open(gs_file) as f:
            results['gs'] = json.load(f)
        print(f"  Loaded {len(results['gs'])} GS results")
    
    if args.plots:
        generate_plots(results)
    
    # Generate markdown report
    report = generate_markdown_report(results)
    report_path = host_dir / 'BENCHMARK_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved report to {report_path}")


def generate_plots(results):
    """Generate visualization plots."""
    from benchmarks.visualization.plot_utils import (
        plot_pose_metrics_bar,
        plot_depth_metrics_bar,
        plot_latency_comparison,
    )
    
    host_plots_dir = get_host_results_dir() / "plots"
    host_plots_dir.mkdir(parents=True, exist_ok=True)
    
    if results.get('pose'):
        pose_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results['pose']]
        try:
            plot_pose_metrics_bar(pose_dicts, output_path=host_plots_dir / 'pose_comparison')
            plot_latency_comparison(pose_dicts, benchmark_type='pose', output_path=host_plots_dir / 'pose_latency')
            logger.info("  Generated pose plots")
        except Exception as e:
            logger.warning(f"  Pose plots failed: {e}")
    
    if results.get('depth'):
        depth_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results['depth']]
        try:
            plot_depth_metrics_bar(depth_dicts, output_path=host_plots_dir / 'depth_comparison')
            plot_latency_comparison(depth_dicts, benchmark_type='depth', output_path=host_plots_dir / 'depth_latency')
            logger.info("  Generated depth plots")
        except Exception as e:
            logger.warning(f"  Depth plots failed: {e}")


def generate_markdown_report(results) -> str:
    """Generate markdown benchmark report."""
    lines = [
        "# AirSplatMap Benchmark Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    
    if results.get('pose'):
        lines.append("## Pose Estimation\n")
        lines.append("| Method | Dataset | ATE RMSE | RPE Trans | FPS | Latency |")
        lines.append("|--------|---------|----------|-----------|-----|---------|")
        for r in results['pose']:
            lines.append(f"| {r['method']} | {r['dataset']} | {r['ate_rmse']:.4f}m | "
                        f"{r['rpe_trans_rmse']:.4f}m | {r['fps']:.1f} | {r['avg_latency_ms']:.1f}ms |")
        lines.append("")
    
    if results.get('depth'):
        lines.append("## Depth Estimation\n")
        lines.append("| Method | Dataset | AbsRel | RMSE | δ<1.25 | FPS |")
        lines.append("|--------|---------|--------|------|--------|-----|")
        for r in results['depth']:
            lines.append(f"| {r['method']} | {r['dataset']} | {r['abs_rel']:.4f} | "
                        f"{r['rmse']:.4f}m | {r['delta1']:.3f} | {r['fps']:.1f} |")
        lines.append("")
    
    if results.get('gs'):
        lines.append("## Gaussian Splatting\n")
        lines.append("| Engine | Dataset | PSNR | SSIM | Gaussians | FPS |")
        lines.append("|--------|---------|------|------|-----------|-----|")
        for r in results['gs']:
            lines.append(f"| {r['engine']} | {r['dataset']} | {r['psnr']:.2f}dB | "
                        f"{r['ssim']:.4f} | {r['final_gaussians']:,} | {r['fps']:.2f} |")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="AirSplatMap Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarks pose --methods orb sift robust_flow
    python -m benchmarks depth --methods midas_small
    python -m benchmarks gs --engines graphdeco
    python -m benchmarks all --quick --save
    python -m benchmarks report --plots
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Benchmark type')
    
    # Pose subcommand
    pose_parser = subparsers.add_parser('pose', help='Pose estimation benchmark')
    pose_parser.add_argument('--methods', nargs='+', default=['orb', 'sift', 'robust_flow', 'keyframe'])
    pose_parser.add_argument('--max-frames', type=int, default=200)
    pose_parser.add_argument('--skip-frames', type=int, default=2)
    pose_parser.add_argument('--dataset-root', type=Path)
    pose_parser.add_argument('--output', type=Path, help='Output JSON file')
    pose_parser.add_argument('--save', action='store_true', help='Save to default location')
    
    # Depth subcommand
    depth_parser = subparsers.add_parser('depth', help='Depth estimation benchmark')
    depth_parser.add_argument('--methods', nargs='+', default=['midas_small'])
    depth_parser.add_argument('--max-frames', type=int, default=50)
    depth_parser.add_argument('--skip-frames', type=int, default=5)
    depth_parser.add_argument('--dataset-root', type=Path)
    depth_parser.add_argument('--output', type=Path)
    depth_parser.add_argument('--save', action='store_true')
    
    # GS subcommand
    gs_parser = subparsers.add_parser('gs', help='Gaussian splatting benchmark')
    gs_parser.add_argument('--engines', nargs='+', default=['graphdeco'])
    gs_parser.add_argument('--max-frames', type=int, default=50)
    gs_parser.add_argument('--iterations', type=int, default=30)
    gs_parser.add_argument('--dataset-root', type=Path)
    gs_parser.add_argument('--output', type=Path)
    gs_parser.add_argument('--save', action='store_true')
    
    # All subcommand
    all_parser = subparsers.add_parser('all', help='Run all benchmarks')
    all_parser.add_argument('--quick', action='store_true', help='Quick mode with fewer frames')
    all_parser.add_argument('--max-frames', type=int)
    all_parser.add_argument('--iterations', type=int, default=30)
    all_parser.add_argument('--skip-pose', action='store_true')
    all_parser.add_argument('--skip-depth', action='store_true')
    all_parser.add_argument('--skip-gs', action='store_true')
    all_parser.add_argument('--save', action='store_true', default=True)
    all_parser.add_argument('--plots', action='store_true', default=True)
    
    # Report subcommand
    report_parser = subparsers.add_parser('report', help='Generate report from results')
    report_parser.add_argument('--input', type=Path, default=RESULTS_DIR)
    report_parser.add_argument('--plots', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.command == 'pose':
        cmd_pose(args)
    elif args.command == 'depth':
        cmd_depth(args)
    elif args.command == 'gs':
        cmd_gs(args)
    elif args.command == 'all':
        cmd_all(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
