#!/usr/bin/env python3
"""
Run SplaTAM on TUM RGB-D sequences with full output generation.

Outputs (consistent with gsplat):
1. 1_live_rendering.mp4 - Side-by-side GT vs render during processing
2. 2_splat_visualization.mp4 - Splat/point cloud visualization
3. 3_model_flythrough.mp4 - Orbit around final model
4. 4_voxel_flythrough.mp4 - Voxel grid visualization (if generated)
5. final/ - Saved model (point_cloud.ply, checkpoint.pth)
6. final_map_renders/ - Renders from all viewpoints
7. metrics_summary.txt - Comprehensive metrics

Usage:
    python scripts/run_splatam_tum.py --sequence rgbd_dataset_freiburg1_xyz
    python scripts/run_splatam_tum.py --sequence rgbd_dataset_freiburg1_xyz --max-frames 200
"""

import os
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def draw_metrics_panel(img, metrics, panel_height=100):
    """Draw a metrics panel at the bottom of the image."""
    h, w = img.shape[:2]
    
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    cv2.line(panel, (0, 0), (w, 0), (100, 100, 100), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Timing
    col1_x = 20
    cv2.putText(panel, "TIMING", (col1_x, 20), font, 0.45, (150, 150, 150), 1)
    cv2.putText(panel, f"FPS: {metrics.get('fps', 0):.1f}", (col1_x, 42), font, 0.45, (0, 255, 0), 1)
    cv2.putText(panel, f"Elapsed: {metrics.get('elapsed', 0):.1f}s", (col1_x, 62), font, 0.45, (200, 200, 200), 1)
    cv2.putText(panel, f"Frame: {metrics.get('frame', 0)}/{metrics.get('total_frames', 0)}", (col1_x, 82), font, 0.45, (200, 200, 200), 1)
    
    # Model
    col2_x = w // 4 + 20
    cv2.putText(panel, "MODEL", (col2_x, 20), font, 0.45, (150, 150, 150), 1)
    cv2.putText(panel, f"Gaussians: {metrics.get('gaussians', 0):,}", (col2_x, 42), font, 0.45, (0, 200, 255), 1)
    cv2.putText(panel, f"Iterations: {metrics.get('iterations', 0):,}", (col2_x, 62), font, 0.45, (200, 200, 200), 1)
    
    # Quality
    col3_x = w // 2 + 20
    cv2.putText(panel, "QUALITY", (col3_x, 20), font, 0.45, (150, 150, 150), 1)
    loss = metrics.get('loss', 0)
    loss_color = (0, 255, 0) if loss < 0.1 else (0, 255, 255) if loss < 0.2 else (0, 100, 255)
    cv2.putText(panel, f"Loss: {loss:.4f}", (col3_x, 42), font, 0.45, loss_color, 1)
    cv2.putText(panel, f"PSNR: {metrics.get('psnr', 0):.1f} dB", (col3_x, 62), font, 0.45, (200, 200, 200), 1)
    
    # Memory & Progress
    col4_x = 3 * w // 4 + 20
    cv2.putText(panel, "MEMORY", (col4_x, 20), font, 0.45, (150, 150, 150), 1)
    cv2.putText(panel, f"GPU: {metrics.get('gpu_mb', 0):.0f} MB", (col4_x, 42), font, 0.45, (255, 150, 0), 1)
    
    # Progress bar
    progress = metrics.get('progress', 0)
    bar_y = 60
    bar_width = w // 4 - 40
    cv2.rectangle(panel, (col4_x, bar_y), (col4_x + bar_width, bar_y + 15), (60, 60, 60), -1)
    cv2.rectangle(panel, (col4_x, bar_y), (col4_x + int(bar_width * progress), bar_y + 15), (0, 200, 0), -1)
    cv2.putText(panel, f"{int(progress * 100)}%", (col4_x + bar_width + 5, bar_y + 12), font, 0.4, (255, 255, 255), 1)
    
    return np.vstack([img, panel])


def render_splat_visualization(engine, pose, img_size):
    """Render Gaussians as colored points from a given viewpoint."""
    width, height = img_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get Gaussian parameters
    if not hasattr(engine, '_params') or 'means3D' not in engine._params:
        return img
    
    xyz = engine._params['means3D'].detach().cpu().numpy()
    
    # Get colors from RGB params if available
    if 'rgb_colors' in engine._params:
        colors = engine._params['rgb_colors'].detach().cpu().numpy()
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    else:
        # Color by depth
        colors = np.full((len(xyz), 3), 128, dtype=np.uint8)
    
    if len(xyz) == 0:
        return img
    
    # Transform to camera frame
    pose_inv = np.linalg.inv(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    xyz_cam = (R @ xyz.T).T + t
    
    # Filter points in front of camera
    mask = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[mask]
    colors = colors[mask]
    
    if len(xyz_cam) == 0:
        return img
    
    # Project to image
    intrinsics = engine._intrinsics
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    u = (fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx).astype(int)
    v = (fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy).astype(int)
    depths = xyz_cam[:, 2]
    
    # Filter in bounds
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, depths, colors = u[mask], v[mask], depths[mask], colors[mask]
    
    # Sort by depth (far to near)
    order = np.argsort(-depths)
    u, v, colors = u[order], v[order], colors[order]
    
    # Draw points (larger for visibility)
    for i in range(len(u)):
        color = tuple(int(c) for c in colors[i])
        cv2.circle(img, (u[i], v[i]), 2, color, -1)
    
    return img


def generate_orbit_poses(center, radius, height, n_frames):
    """Generate camera poses for orbiting around a point."""
    poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height
        
        forward = np.array([center[0] - x, center[1] - y, center[2] - z])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        
        R = np.stack([right, -up, forward], axis=1)
        t = np.array([x, y, z])
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(pose)
    
    return poses


def main():
    parser = argparse.ArgumentParser(description="Run SplaTAM on TUM RGB-D")
    parser.add_argument("--dataset-root", type=str, default="../datasets/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--refinement-iters", type=int, default=500)
    parser.add_argument("--render-every", type=int, default=20)
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    args = parser.parse_args()
    
    # Find dataset
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root.parent / args.dataset_root.lstrip("../")
    
    dataset_path = dataset_root / args.sequence
    if not dataset_path.exists():
        for alt in [
            project_root.parent / "datasets" / "tum" / args.sequence,
            Path("/home/past/parsa/datasets/tum") / args.sequence,
        ]:
            if alt.exists():
                dataset_path = alt
                break
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "output" / "splatam" / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("SPLATAM - TUM RGB-D Processing")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load dataset
    from src.pipeline.frames import TumRGBDSource
    from src.engines import get_engine
    
    source = TumRGBDSource(str(dataset_path))
    frames = list(source)
    
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    
    if not frames:
        logger.error("No frames loaded!")
        sys.exit(1)
    
    if frames[0].depth is None:
        logger.error("First frame has no depth! SplaTAM requires depth.")
        sys.exit(1)
    
    intrinsics = frames[0].intrinsics.copy()
    width, height = int(intrinsics['width']), int(intrinsics['height'])
    
    logger.info(f"Frames: {len(frames)}")
    logger.info(f"Resolution: {width}x{height}")
    
    # Create subdirectories
    final_dir = output_dir / "final"
    renders_dir = output_dir / "final_map_renders"
    final_dir.mkdir(exist_ok=True)
    renders_dir.mkdir(exist_ok=True)
    
    # Initialize engine
    engine = get_engine('splatam')
    engine.initialize_scene(intrinsics, {'num_frames': len(frames)})
    
    # Video setup
    panel_height = 100
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 1. Live rendering video (side-by-side GT vs render)
    live_video_path = output_dir / "1_live_rendering.mp4"
    live_writer = cv2.VideoWriter(str(live_video_path), fourcc, args.fps, (width * 2, height + panel_height))
    
    # 2. Splat visualization video
    splat_video_path = output_dir / "2_splat_visualization.mp4"
    splat_writer = cv2.VideoWriter(str(splat_video_path), fourcc, args.fps, (width, height + panel_height))
    
    logger.info("")
    logger.info("Output videos:")
    logger.info(f"  1. {live_video_path.name}")
    logger.info(f"  2. {splat_video_path.name}")
    logger.info("")
    
    # Process frames
    logger.info("Processing frames...")
    t0 = time.time()
    total_iterations = 0
    all_poses = []
    
    for i, frame in enumerate(frames):
        frame_t0 = time.time()
        
        # Add frame
        engine.add_frame(frame.idx, frame.get_rgb_uint8(), frame.depth, frame.pose)
        total_iterations += 1
        all_poses.append(frame.pose)
        
        # Get data for visualization
        gt_rgb = frame.get_rgb_uint8()
        rendered = engine.render_view(frame.pose, (width, height))
        num_gaussians = engine.get_num_gaussians()
        
        # Calculate metrics
        elapsed = time.time() - t0
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        
        if rendered is not None:
            mse = np.mean((gt_rgb.astype(float) - rendered.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / (mse + 1e-10)) if mse > 0 else 40
        else:
            psnr = 0
            rendered = np.zeros_like(gt_rgb)
        
        metrics = {
            'fps': fps,
            'elapsed': elapsed,
            'frame': i + 1,
            'total_frames': len(frames),
            'gaussians': num_gaussians,
            'iterations': total_iterations,
            'loss': 0.0,
            'psnr': psnr,
            'gpu_mb': get_gpu_memory_mb(),
            'progress': (i + 1) / len(frames),
        }
        
        # === VIDEO 1: Live rendering (GT | Render) ===
        gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        
        cv2.putText(gt_bgr, f"INPUT (frame {frame.idx})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rendered_bgr, f"SPLATAM RENDER", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        live_frame = np.hstack([gt_bgr, rendered_bgr])
        live_frame = draw_metrics_panel(live_frame, metrics, panel_height)
        live_writer.write(live_frame)
        
        # === VIDEO 2: Splat visualization ===
        splat_view = render_splat_visualization(engine, frame.pose, (width, height))
        cv2.putText(splat_view, f"SPLATAM SPLATS ({num_gaussians:,})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        splat_frame = draw_metrics_panel(splat_view, metrics, panel_height)
        splat_writer.write(splat_frame)
        
        # Save periodic renders
        if (i + 1) % args.render_every == 0:
            cv2.imwrite(str(renders_dir / f"frame_{i+1:06d}.png"),
                       cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        
        # Progress
        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            logger.info(f"  Frame {i+1:4d}/{len(frames)}: {num_gaussians:,} Gaussians, "
                       f"PSNR={psnr:.1f}dB, {fps:.1f} fps")
    
    processing_time = time.time() - t0
    live_writer.release()
    splat_writer.release()
    
    # Refinement
    logger.info("")
    logger.info(f"Refining ({args.refinement_iters} iterations)...")
    refine_t0 = time.time()
    
    for step in range(args.refinement_iters):
        stats = engine.optimize_step(1)
        total_iterations += 1
        
        if (step + 1) % 100 == 0:
            logger.info(f"  Step {step+1}: loss={stats['loss']:.4f}, PSNR={stats['psnr']:.1f}dB")
    
    refinement_time = time.time() - refine_t0
    total_time = time.time() - t0
    
    final_loss = stats['loss']
    final_psnr = stats['psnr']
    final_gaussians = engine.get_num_gaussians()
    avg_fps = len(frames) / processing_time
    
    # === VIDEO 3: Model flythrough ===
    logger.info("")
    logger.info("Generating model flythrough...")
    
    if hasattr(engine, '_params') and 'means3D' in engine._params:
        means = engine._params['means3D'].detach().cpu().numpy()
        center = means.mean(axis=0)
        radius = np.linalg.norm(means - center, axis=1).max() * 1.5
    else:
        center = np.array([0, 0, 0])
        radius = 2.0
    
    orbit_video_path = output_dir / "3_model_flythrough.mp4"
    orbit_writer = cv2.VideoWriter(str(orbit_video_path), fourcc, 30, (width, height))
    
    orbit_poses = generate_orbit_poses(center, radius, radius * 0.3, 120)
    for pose in orbit_poses:
        rendered = engine.render_view(pose, (width, height))
        if rendered is not None:
            orbit_writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    orbit_writer.release()
    
    # Save final renders
    logger.info("Saving final renders...")
    for i in range(0, len(frames), args.render_every):
        frame = frames[i]
        gt = frame.get_rgb_uint8()
        rendered = engine.render_view(frame.pose, (width, height))
        
        if rendered is not None:
            cv2.imwrite(str(renders_dir / f"final_{i:06d}.png"),
                       cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            comp = np.hstack([gt, rendered])
            cv2.imwrite(str(renders_dir / f"comparison_{i:06d}.png"),
                       cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    
    # Save model
    logger.info("Saving model...")
    engine.save_state(str(final_dir))
    
    # Save metrics summary
    metrics_path = output_dir / "metrics_summary.txt"
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SPLATAM - RGB-D GAUSSIAN SPLATTING\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dataset: {args.sequence}\n")
        f.write(f"Resolution: {width}x{height}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"  Frames processed: {len(frames)}\n")
        f.write(f"  Processing time: {processing_time:.1f}s\n")
        f.write(f"  Refinement time: {refinement_time:.1f}s\n")
        f.write(f"  Total time: {total_time:.1f}s\n")
        f.write(f"  Average FPS: {avg_fps:.2f}\n")
        f.write(f"  Total iterations: {total_iterations:,}\n")
        f.write(f"  Refinement iterations: {args.refinement_iters}\n\n")
        
        f.write("MODEL:\n")
        f.write(f"  Final Gaussians: {final_gaussians:,}\n")
        f.write(f"  Final Loss: {final_loss:.4f}\n")
        f.write(f"  Final PSNR: {final_psnr:.2f} dB\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write(f"  1. {live_video_path.name} - Live rendering with metrics\n")
        f.write(f"  2. {splat_video_path.name} - Splat visualization\n")
        f.write(f"  3. {orbit_video_path.name} - Model flythrough\n")
        f.write(f"  final_map_renders/ - Renders from all viewpoints\n")
        f.write(f"  final/ - Saved Gaussian model (.ply, .pth)\n")
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Frames: {len(frames)}")
    logger.info(f"  Gaussians: {final_gaussians:,}")
    logger.info(f"  PSNR: {final_psnr:.2f} dB")
    logger.info(f"  Loss: {final_loss:.4f}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Avg FPS: {avg_fps:.2f}")
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  1. {live_video_path}")
    logger.info(f"  2. {splat_video_path}")
    logger.info(f"  3. {orbit_video_path}")
    logger.info(f"  4. {renders_dir}/")
    logger.info(f"  5. {final_dir}/")
    logger.info(f"  6. {metrics_path}")
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
