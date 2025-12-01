#!/usr/bin/env python3
"""
Batch Process Multiple Scenes for 3D Gaussian Splatting

This script processes multiple TUM RGB-D scenes and saves results to 
individual folders named after each scene.

Usage:
    python scripts/batch_process_scenes.py --dataset-root /path/to/tum
    
    # Or specify specific scenes:
    python scripts/batch_process_scenes.py --dataset-root /path/to/tum \
        --scenes rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_room
"""

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_tum_scenes(dataset_root):
    """Find all TUM RGB-D dataset sequences."""
    dataset_root = Path(dataset_root)
    scenes = []
    
    for d in dataset_root.iterdir():
        if d.is_dir():
            # Check for TUM dataset structure (rgb/ and depth/ folders, or association file)
            has_rgb = (d / "rgb").exists() or (d / "rgb.txt").exists()
            has_depth = (d / "depth").exists() or (d / "depth.txt").exists()
            
            if has_rgb and has_depth:
                scenes.append(d.name)
                
    return sorted(scenes)


def generate_combined_video(output_dir):
    """Generate combined [Input | Splat | Render] video."""
    live_path = output_dir / "1_live_rendering.mp4"
    splat_path = output_dir / "2_splat_visualization.mp4"
    combined_path = output_dir / "combined_view.mp4"
    
    if not live_path.exists() or not splat_path.exists():
        logger.warning(f"Cannot create combined video: missing source videos")
        return None
        
    if combined_path.exists():
        logger.info(f"Combined video already exists: {combined_path}")
        return combined_path
        
    logger.info("Generating combined video...")
    
    cap_live = cv2.VideoCapture(str(live_path))
    cap_splat = cv2.VideoCapture(str(splat_path))
    
    width_live = int(cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_live = int(cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_live.get(cv2.CAP_PROP_FPS)
    
    half_width = width_live // 2
    
    width_splat = int(cap_splat.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_splat = int(cap_splat.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Strip metrics panel from splat video if present
    # The metrics panel is 120 pixels at the bottom
    splat_content_height = height_splat - 120 if height_splat > 500 else height_splat
    
    # Target dimensions matching live video height
    target_height = height_live - 120  # Live video also has metrics panel
    scale_splat = target_height / splat_content_height
    target_width_splat = int(width_splat * scale_splat)
    
    # Total width: Input + Splat + Render
    total_width = half_width + target_width_splat + half_width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(combined_path), fourcc, fps, (total_width, height_live))
    
    frame_count = 0
    while True:
        ret1, frame_live = cap_live.read()
        ret2, frame_splat = cap_splat.read()
        
        if not ret1 or not ret2:
            break
            
        # Split live frame (Input on left, Render on right)
        frame_input = frame_live[:target_height, :half_width]
        frame_render = frame_live[:target_height, half_width:]
        metrics_panel = frame_live[target_height:, :]
        
        # Get splat content (remove metrics panel)
        frame_splat_content = frame_splat[:splat_content_height, :]
        
        # Resize splat to match
        frame_splat_resized = cv2.resize(frame_splat_content, (target_width_splat, target_height))
        
        # Combine top row
        top_row = np.hstack([frame_input, frame_splat_resized, frame_render])
        
        # Add labels
        cv2.putText(top_row, "INPUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(top_row, "GAUSSIAN SPLATS", (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(top_row, "3DGS RENDER", (half_width + target_width_splat + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Extend metrics panel to match new width
        metrics_extended = np.zeros((120, total_width, 3), dtype=np.uint8)
        metrics_extended[:] = (30, 30, 30)
        # Copy original metrics (centered or stretched)
        metrics_extended[:, :width_live] = metrics_panel
        
        # Combined frame
        combined = np.vstack([top_row, metrics_extended])
        out.write(combined)
        frame_count += 1
        
    cap_live.release()
    cap_splat.release()
    out.release()
    
    logger.info(f"Combined video saved: {combined_path} ({frame_count} frames)")
    return combined_path


def process_scene(dataset_root, scene_name, output_root, args):
    """Process a single scene."""
    # Organize by engine: output/<engine>/<scene>
    output_dir = output_root / args.engine / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Processing scene: {scene_name}")
    logger.info(f"Engine: {args.engine}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"=" * 60)
    
    # Build command
    script_path = Path(__file__).parent / "live_tum_demo.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--dataset-root", str(dataset_root),
        "--sequence", scene_name,
        "--max-frames", str(args.max_frames),
        "--steps-per-frame", str(args.steps_per_frame),
        "--quality", args.quality,
        "--mode", "mapping",
        "--refinement-iters", str(args.refinement_iters),
        "--orbit-frames", str(args.orbit_frames),
        "--orbit-radius", str(args.orbit_radius),
        "--output", str(output_dir),
        "--engine", args.engine,
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Scene {scene_name} failed with return code {result.returncode}")
        return False
        
    logger.info(f"Scene {scene_name} completed in {elapsed:.1f}s")
    
    # Generate combined video
    generate_combined_video(output_dir)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch process TUM RGB-D scenes")
    parser.add_argument("--dataset-root", type=str, default="/home/past/parsa/datasets/tum",
                       help="Root directory containing TUM sequences")
    parser.add_argument("--output-root", type=str, default="./output",
                       help="Root output directory")
    parser.add_argument("--scenes", nargs="*", default=None,
                       help="Specific scenes to process (default: all found)")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames per scene")
    parser.add_argument("--steps-per-frame", type=int, default=12, help="Optimization steps")
    parser.add_argument("--quality", choices=['fast', 'balanced', 'quality'], default='fast')
    parser.add_argument("--refinement-iters", type=int, default=800, help="Global refinement iterations")
    parser.add_argument("--orbit-frames", type=int, default=90, help="Frames for flythrough")
    parser.add_argument("--orbit-radius", type=float, default=1.5, help="Orbit radius")
    parser.add_argument("--engine", choices=['graphdeco', 'gsplat', 'splatam'], default='graphdeco',
                       help="3DGS engine backend")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    
    # Find scenes
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = find_tum_scenes(dataset_root)
        
    if not scenes:
        logger.error(f"No TUM scenes found in {dataset_root}")
        logger.info("Expected structure: <dataset_root>/<scene>/rgb/ and depth/")
        return
        
    logger.info(f"Found {len(scenes)} scene(s) to process:")
    for s in scenes:
        logger.info(f"  - {s}")
    
    # Process each scene
    results = {}
    total_start = time.time()
    
    for scene in scenes:
        success = process_scene(dataset_root, scene, output_root, args)
        results[scene] = success
        
    total_elapsed = time.time() - total_start
    
    # Summary
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Scenes processed: {len(results)}")
    
    for scene, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {status}: {scene}")
        
    logger.info(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
