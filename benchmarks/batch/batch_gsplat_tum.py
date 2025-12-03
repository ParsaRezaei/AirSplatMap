#!/usr/bin/env python3
"""
Batch process all TUM sequences with GSplat engine and generate voxel grids.

This script:
1. Runs gsplat on all TUM RGB-D sequences
2. Generates voxel occupancy grids from the Gaussian splats
3. Creates visualization videos

Usage:
    python scripts/batch_gsplat_tum.py --dataset-root /path/to/tum --output-root ./output/gsplat
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_gsplat_demo(
    dataset_root: str,
    sequence: str,
    output_dir: Path,
    max_frames: int = -1,
    steps_per_frame: int = 8,
    quality: str = "fast",
    refinement_iters: int = 1000,
) -> bool:
    """Run live_tum_demo.py with gsplat engine."""
    
    script_path = Path(__file__).parent / "live_tum_demo.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--dataset-root", dataset_root,
        "--sequence", sequence,
        "--engine", "gsplat",
        "--steps-per-frame", str(steps_per_frame),
        "--quality", quality,
        "--refinement-iters", str(refinement_iters),
        "--output", str(output_dir),
    ]
    
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running gsplat demo: {e}")
        return False


def run_voxel_grid_generation(output_root: str, scenes: list) -> bool:
    """Generate voxel grids for all processed scenes."""
    
    script_path = Path(__file__).parent / "generate_voxel_grid.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--output-root", output_root,
        "--voxel-size", "0.05",
        "--opacity-threshold", "0.3",
        "--safety-distances", "0.5", "1.0", "2.0",
    ]
    
    if scenes:
        cmd.extend(["--scenes"] + scenes)
    
    logger.info(f"Generating voxel grids...")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error generating voxel grids: {e}")
        return False


def find_tum_sequences(dataset_root: str) -> list:
    """Find all TUM RGB-D sequences."""
    
    root = Path(dataset_root)
    sequences = []
    
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("rgbd_dataset_"):
            # Check if it has the required files
            if (d / "rgb.txt").exists() and (d / "groundtruth.txt").exists():
                sequences.append(d.name)
    
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Batch process TUM sequences with GSplat"
    )
    parser.add_argument(
        "--dataset-root", type=str, 
        default="/home/past/parsa/datasets/tum",
        help="TUM dataset root directory"
    )
    parser.add_argument(
        "--output-root", type=str,
        default="./output/gsplat",
        help="Output directory root"
    )
    parser.add_argument(
        "--sequences", nargs="*",
        help="Specific sequences to process (default: all)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=-1,
        help="Max frames per sequence (-1 for all)"
    )
    parser.add_argument(
        "--steps-per-frame", type=int, default=8,
        help="Optimization steps per frame"
    )
    parser.add_argument(
        "--quality", type=str, default="fast",
        choices=["fast", "balanced", "quality"],
        help="Quality preset"
    )
    parser.add_argument(
        "--refinement-iters", type=int, default=1000,
        help="Global refinement iterations"
    )
    parser.add_argument(
        "--skip-voxel", action="store_true",
        help="Skip voxel grid generation"
    )
    parser.add_argument(
        "--only-voxel", action="store_true",
        help="Only generate voxel grids (skip 3DGS training)"
    )
    
    args = parser.parse_args()
    
    # Find sequences
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = find_tum_sequences(args.dataset_root)
    
    if not sequences:
        logger.error(f"No TUM sequences found in {args.dataset_root}")
        return 1
    
    logger.info("=" * 60)
    logger.info("BATCH GSPLAT PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Sequences: {len(sequences)}")
    for seq in sequences:
        logger.info(f"  - {seq}")
    logger.info(f"Quality: {args.quality}")
    logger.info(f"Max frames: {args.max_frames if args.max_frames > 0 else 'all'}")
    logger.info("=" * 60)
    
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = {}
    processed_scenes = []
    
    # Run gsplat on each sequence
    if not args.only_voxel:
        for i, seq in enumerate(sequences, 1):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"[{i}/{len(sequences)}] Processing: {seq}")
            logger.info("=" * 60)
            
            output_dir = output_root / seq
            
            success = run_gsplat_demo(
                dataset_root=args.dataset_root,
                sequence=seq,
                output_dir=output_dir,
                max_frames=args.max_frames,
                steps_per_frame=args.steps_per_frame,
                quality=args.quality,
                refinement_iters=args.refinement_iters,
            )
            
            results[seq] = success
            if success:
                processed_scenes.append(seq)
    else:
        # Find already processed scenes
        for seq in sequences:
            ply_path = output_root / seq / "final" / "point_cloud.ply"
            if ply_path.exists():
                processed_scenes.append(seq)
                results[seq] = True
    
    # Generate voxel grids
    if not args.skip_voxel and processed_scenes:
        logger.info("")
        logger.info("=" * 60)
        logger.info("GENERATING VOXEL GRIDS")
        logger.info("=" * 60)
        
        voxel_success = run_voxel_grid_generation(
            str(output_root),
            processed_scenes
        )
        
        if not voxel_success:
            logger.warning("Voxel grid generation had issues")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Processed: {success_count}/{len(results)} sequences")
    
    for seq, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {seq}")
    
    logger.info("")
    logger.info(f"Output directory: {output_root}")
    logger.info("Files per sequence:")
    logger.info("  final/point_cloud.ply - Gaussian splat model")
    logger.info("  final/checkpoint.pth - Model checkpoint")
    logger.info("  1_live_rendering.mp4 - Training visualization")
    logger.info("  2_splat_visualization.mp4 - Gaussian splat view")
    logger.info("  3_model_flythrough.mp4 - Orbit around model")
    logger.info("  4_mesh_flythrough.mp4 - Extracted mesh")
    logger.info("  voxel_grid/ - Occupancy grids and SDF")
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
