#!/usr/bin/env python3
"""
Enhanced Live 3D Gaussian Splatting Demo

Outputs multiple videos:
1. Live rendering - Side-by-side input vs render as frames come in
2. Splat visualization - Shows all Gaussians as points
3. Model flythrough - Orbit around the final 3DGS model

Displays comprehensive metrics:
- FPS (current and average)
- Gaussian count
- Loss / PSNR
- Memory usage
- Optimization iterations
- Time elapsed

Usage:
    python scripts/live_tum_demo.py --dataset-root /path/to/tum --sequence rgbd_dataset_freiburg1_desk
"""

import os
# Set CUDA arch for gsplat JIT compilation (RTX 2080 = sm_75)
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')
# Limit parallel jobs to avoid disk I/O issues during compilation
os.environ.setdefault('MAX_JOBS', '2')
# Set CUDA include path for gsplat compilation (if in conda env with cuda-nvcc)
if 'CONDA_PREFIX' in os.environ:
    cuda_include = os.path.join(os.environ['CONDA_PREFIX'], 'targets', 'x86_64-linux', 'include')
    if os.path.exists(cuda_include):
        current_cpath = os.environ.get('CPATH', '')
        if cuda_include not in current_cpath:
            os.environ['CPATH'] = f"{cuda_include}:{current_cpath}" if current_cpath else cuda_include

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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


def draw_metrics_panel(img, metrics, panel_height=120):
    """Draw a metrics panel at the bottom of the image."""
    h, w = img.shape[:2]
    
    # Create panel background
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background
    
    # Draw separator line
    cv2.line(panel, (0, 0), (w, 0), (100, 100, 100), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Left column - Timing
    col1_x = 20
    cv2.putText(panel, "TIMING", (col1_x, 25), font, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"Current FPS: {metrics.get('current_fps', 0):.1f}", (col1_x, 50), font, 0.5, (0, 255, 0), 1)
    cv2.putText(panel, f"Avg FPS: {metrics.get('avg_fps', 0):.2f}", (col1_x, 70), font, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Frame Time: {metrics.get('frame_time_ms', 0):.0f}ms", (col1_x, 90), font, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Elapsed: {metrics.get('elapsed_time', 0):.1f}s", (col1_x, 110), font, 0.5, (200, 200, 200), 1)
    
    # Middle column - Model stats
    col2_x = w // 4 + 20
    cv2.putText(panel, "MODEL", (col2_x, 25), font, 0.5, (150, 150, 150), 1)
    n_gauss = metrics.get('num_gaussians', 0)
    max_gauss = metrics.get('max_gaussians', 100000)
    gauss_pct = n_gauss / max_gauss * 100 if max_gauss > 0 else 0
    cv2.putText(panel, f"Gaussians: {n_gauss:,}", (col2_x, 50), font, 0.5, (0, 200, 255), 1)
    cv2.putText(panel, f"Capacity: {gauss_pct:.0f}%", (col2_x, 70), font, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Cameras: {metrics.get('num_cameras', 0)}", (col2_x, 90), font, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Iterations: {metrics.get('total_iterations', 0):,}", (col2_x, 110), font, 0.5, (200, 200, 200), 1)
    
    # Right-middle column - Quality
    col3_x = w // 2 + 20
    cv2.putText(panel, "QUALITY", (col3_x, 25), font, 0.5, (150, 150, 150), 1)
    loss = metrics.get('loss', 0)
    loss_color = (0, 255, 0) if loss < 0.1 else (0, 255, 255) if loss < 0.2 else (0, 100, 255)
    cv2.putText(panel, f"Loss: {loss:.4f}", (col3_x, 50), font, 0.5, loss_color, 1)
    cv2.putText(panel, f"PSNR: {metrics.get('psnr', 0):.1f} dB", (col3_x, 70), font, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"SH Degree: {metrics.get('sh_degree', 3)}", (col3_x, 90), font, 0.5, (200, 200, 200), 1)
    
    # Right column - Memory
    col4_x = 3 * w // 4 + 20
    cv2.putText(panel, "MEMORY", (col4_x, 25), font, 0.5, (150, 150, 150), 1)
    gpu_mem = metrics.get('gpu_memory_mb', 0)
    cv2.putText(panel, f"GPU: {gpu_mem:.0f} MB", (col4_x, 50), font, 0.5, (255, 150, 0), 1)
    cv2.putText(panel, f"Frame: {metrics.get('frame_idx', 0)}/{metrics.get('total_frames', 0)}", (col4_x, 70), font, 0.5, (200, 200, 200), 1)
    
    # Progress bar
    progress = metrics.get('progress', 0)
    bar_y = 95
    bar_width = w // 4 - 40
    cv2.rectangle(panel, (col4_x, bar_y), (col4_x + bar_width, bar_y + 15), (60, 60, 60), -1)
    cv2.rectangle(panel, (col4_x, bar_y), (col4_x + int(bar_width * progress), bar_y + 15), (0, 200, 0), -1)
    cv2.putText(panel, f"{int(progress * 100)}%", (col4_x + bar_width + 5, bar_y + 12), font, 0.4, (255, 255, 255), 1)
    
    # Combine with image
    result = np.vstack([img, panel])
    return result


def generate_orbit_poses(center, radius, height, n_frames, intrinsics):
    """Generate camera poses for orbiting around a point."""
    poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        # Camera position
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height
        
        # Look at center
        forward = np.array([center[0] - x, center[1] - y, center[2] - z])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Up vector (world Z)
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Recompute up
        up = np.cross(right, forward)
        
        # Rotation matrix (camera to world)
        R = np.stack([right, -up, forward], axis=1)
        
        # Pose matrix (camera to world)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = [x, y, z]
        
        poses.append(pose)
    
    return poses


def render_mesh_view(verts, faces, vert_colors, normals, pose, intrinsics, img_size):
    """Render a mesh using proper triangle rasterization with depth sorting."""
    width, height = img_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(verts) == 0 or len(faces) == 0:
        return img
    
    # Transform vertices to camera frame
    pose_inv = np.linalg.inv(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    
    verts_cam = (R @ verts.T).T + t
    normals_cam = (R @ normals.T).T
    
    # Project vertices to image
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    z = verts_cam[:, 2]
    valid = z > 0.1
    
    u = np.full(len(verts), -1.0)
    v = np.full(len(verts), -1.0)
    u[valid] = fx * verts_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * verts_cam[valid, 1] / z[valid] + cy
    
    # Collect face data for depth sorting
    face_data = []
    for fi, face in enumerate(faces):
        v0, v1, v2 = face
        
        # Skip if any vertex behind camera or off-screen
        if z[v0] <= 0.1 or z[v1] <= 0.1 or z[v2] <= 0.1:
            continue
        
        # Get 2D coordinates
        pts = np.array([
            [u[v0], v[v0]],
            [u[v1], v[v1]],
            [u[v2], v[v2]]
        ])
        
        # Skip if triangle is too small or degenerate
        area = 0.5 * abs((pts[1,0] - pts[0,0]) * (pts[2,1] - pts[0,1]) - 
                         (pts[2,0] - pts[0,0]) * (pts[1,1] - pts[0,1]))
        if area < 1:
            continue
        
        # Check if any part is in frame
        if pts[:, 0].max() < 0 or pts[:, 0].min() > width:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() > height:
            continue
        
        # Average depth for sorting
        avg_depth = (z[v0] + z[v1] + z[v2]) / 3
        
        # Average color
        color = (vert_colors[v0] + vert_colors[v1] + vert_colors[v2]) / 3
        
        # Face normal for shading
        normal_cam_avg = (normals_cam[v0] + normals_cam[v1] + normals_cam[v2]) / 3
        normal_cam_avg = normal_cam_avg / (np.linalg.norm(normal_cam_avg) + 1e-8)
        
        # Simple diffuse shading - light from camera
        light_dir = np.array([0, 0, -1])  # Light coming from camera
        shade = max(0.2, abs(np.dot(normal_cam_avg, light_dir)))
        
        color = np.clip(color * shade, 0, 1)
        
        face_data.append({
            'pts': pts.astype(np.int32),
            'depth': avg_depth,
            'color': (color * 255).astype(np.uint8)
        })
    
    # Sort faces by depth (far to near for painter's algorithm)
    face_data.sort(key=lambda x: -x['depth'])
    
    # Draw faces
    for fd in face_data:
        pts = fd['pts'].reshape((-1, 1, 2))
        color = tuple(int(c) for c in fd['color'])
        cv2.fillPoly(img, [pts], color)
        # Draw edges for better definition
        cv2.polylines(img, [pts], True, tuple(max(0, c - 30) for c in color), 1)
    
    return img


def render_point_cloud_view(xyz, colors, pose, intrinsics, img_size):
    """Render a simple point cloud visualization."""
    width, height = img_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(xyz) == 0:
        return img
    
    # Transform points to camera frame
    pose_inv = np.linalg.inv(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    
    xyz_cam = (R @ xyz.T).T + t
    
    # Filter points in front of camera
    mask = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[mask]
    colors_filtered = colors[mask] if colors is not None else None
    
    if len(xyz_cam) == 0:
        return img
    
    # Project to image
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    u = (fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx).astype(int)
    v = (fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy).astype(int)
    
    # Filter points in image bounds
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[mask], v[mask]
    depths = xyz_cam[mask, 2]
    
    if colors_filtered is not None:
        point_colors = colors_filtered[mask]
    else:
        # Color by depth
        depth_normalized = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
        point_colors = np.stack([
            (1 - depth_normalized) * 255,
            depth_normalized * 100,
            depth_normalized * 255
        ], axis=1).astype(np.uint8)
    
    # Sort by depth (far to near) for proper occlusion
    sort_idx = np.argsort(-depths)
    u, v = u[sort_idx], v[sort_idx]
    point_colors = point_colors[sort_idx]
    
    # Draw points
    for i in range(len(u)):
        color = tuple(int(c) for c in point_colors[i])
        cv2.circle(img, (u[i], v[i]), 2, color, -1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Enhanced Live 3DGS Demo")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to TUM dataset root")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence name")
    parser.add_argument("--max-frames", type=int, default=999999, help="Maximum frames to process (default: all)")
    parser.add_argument("--steps-per-frame", type=int, default=15, help="Optimization steps per frame")
    parser.add_argument("--output", type=str, default="./output/live_demo_enhanced", help="Output directory")
    parser.add_argument("--quality", choices=['fast', 'balanced', 'quality'], default='fast', help="Quality preset")
    parser.add_argument("--orbit-frames", type=int, default=120, help="Frames for orbit video")
    parser.add_argument("--orbit-radius", type=float, default=2.0, help="Orbit radius")
    parser.add_argument("--mode", choices=['online', 'mapping'], default='mapping', 
                       help="'online' = real-time with recency bias, 'mapping' = build complete persistent map")
    parser.add_argument("--refinement-iters", type=int, default=2000, 
                       help="Global refinement iterations after processing (mapping mode only)")
    parser.add_argument("--engine", choices=['graphdeco', 'gsplat', 'splatam'], default='graphdeco',
                       help="3DGS engine backend: graphdeco (original), gsplat (optimized), splatam (SLAM)")
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load TUM dataset
    logger.info(f"Loading TUM dataset: {args.sequence}")
    from src.pipeline.frames import TumRGBDSource
    
    source = TumRGBDSource(args.dataset_root, args.sequence)
    intrinsics = source.get_intrinsics()
    
    width = intrinsics['width']
    height = intrinsics['height']
    panel_height = 120
    
    logger.info(f"Dataset: {width}x{height}, {len(source)} total frames")
    
    # Initialize 3DGS engine
    logger.info(f"Initializing 3DGS engine (backend: {args.engine})...")
    from src.engines import get_engine, list_engines
    
    # Show available engines
    available = list_engines()
    for name, info in available.items():
        if info['available']:
            logger.info(f"  Available: {name} - {info['description']}")
    
    try:
        engine = get_engine(args.engine)
        logger.info(f"  Using: {args.engine}")
    except ImportError as e:
        logger.warning(f"Engine {args.engine} not available: {e}")
        logger.info("Falling back to graphdeco engine")
        engine = get_engine('graphdeco')
    
    # Quality presets
    presets = {
        'fast': {
            'sh_degree': 2,
            'steps_mult': 1,
            'depth_subsample': 4,
            'max_gaussians': 100000,
        },
        'balanced': {
            'sh_degree': 3,
            'steps_mult': 2,
            'depth_subsample': 3,
            'max_gaussians': 150000,
        },
        'quality': {
            'sh_degree': 3,
            'steps_mult': 3,
            'depth_subsample': 2,
            'max_gaussians': 200000,
        }
    }
    
    preset = presets[args.quality]
    
    # Mode-specific settings
    if args.mode == 'mapping':
        # Mapping mode: build persistent map, train on ALL cameras equally
        recency_weight = 0.0  # No recency bias - equal weight for all cameras
        max_gaussians = int(preset['max_gaussians'] * 1.5)  # Allow more Gaussians for complete map
        add_every = 5  # Add Gaussians more frequently
        logger.info("MODE: MAPPING - Building persistent 3D map of environment")
    else:
        # Online mode: real-time with recency bias
        recency_weight = 0.7  # Favor recent frames
        max_gaussians = preset['max_gaussians']
        add_every = 10
        logger.info("MODE: ONLINE - Real-time with recency bias")
    
    config = {
        'sh_degree': preset['sh_degree'],
        'white_background': False,
        'densify_grad_threshold': 0.0002,
        'densify_from_iter': 100,
        'densify_until_iter': 100000,
        'densification_interval': 100,
        'opacity_reset_interval': 1000,
        'percent_dense': 0.01,
        'lambda_dssim': 0.2,
        'position_lr_init': 0.0003,
        'position_lr_final': 0.000003,
        'feature_lr': 0.004,
        'opacity_lr': 0.05,
        'scaling_lr': 0.008,
        'rotation_lr': 0.002,
        'recency_weight': recency_weight,
        'depth_subsample': preset['depth_subsample'],
        'add_gaussians_per_frame': True,
        'add_gaussians_every': add_every,
        'add_gaussians_subsample': 12,
        'max_gaussians': max_gaussians,
    }
    
    engine.initialize_scene(intrinsics, config)
    
    steps_per_frame = args.steps_per_frame * preset['steps_mult']
    
    # Video writers
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 1. Live rendering video (side-by-side with metrics)
    live_video_path = output_dir / "1_live_rendering.mp4"
    live_writer = cv2.VideoWriter(str(live_video_path), fourcc, fps, (width * 2, height + panel_height))
    
    # 2. Splat visualization video (point cloud view)
    splat_video_path = output_dir / "2_splat_visualization.mp4"
    splat_writer = cv2.VideoWriter(str(splat_video_path), fourcc, fps, (width, height + panel_height))
    
    logger.info(f"Output videos:")
    logger.info(f"  1. Live rendering: {live_video_path}")
    logger.info(f"  2. Splat visualization: {splat_video_path}")
    
    # Tracking variables
    frame_count = 0
    total_time = 0
    total_iterations = 0
    fps_history = []
    all_poses = []
    
    logger.info("=" * 60)
    logger.info("Starting enhanced live 3DGS demo")
    logger.info(f"  Quality: {args.quality}")
    logger.info(f"  Steps per frame: {steps_per_frame}")
    logger.info(f"  Max Gaussians: {preset['max_gaussians']:,}")
    logger.info("=" * 60)
    
    try:
        for frame in source:
            if frame_count >= args.max_frames:
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Get frame data
            rgb = frame.get_rgb_uint8()
            depth = frame.depth
            pose = frame.pose
            all_poses.append(pose)
            
            # Add frame to engine
            engine.add_frame(frame.idx, rgb, depth, pose)
            
            # Optimize
            metrics = engine.optimize_step(steps_per_frame)
            total_iterations += steps_per_frame
            
            loss = metrics.get('loss', 0)
            num_gaussians = metrics.get('num_gaussians', 0)
            
            # Render current view
            rendered = engine.render_view(pose, (width, height))
            
            # Calculate timing
            elapsed = time.time() - start_time
            total_time += elapsed
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(current_fps)
            avg_fps = sum(fps_history[-30:]) / min(len(fps_history), 30)  # 30-frame moving average
            
            # Calculate PSNR
            mse = np.mean((rgb.astype(float) - rendered.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / (mse + 1e-10)) if mse > 0 else 40
            
            # Prepare metrics dict
            display_metrics = {
                'current_fps': current_fps,
                'avg_fps': avg_fps,
                'frame_time_ms': elapsed * 1000,
                'elapsed_time': total_time,
                'num_gaussians': num_gaussians,
                'max_gaussians': preset['max_gaussians'],
                'num_cameras': len(engine._cameras) if hasattr(engine, '_cameras') else 0,
                'total_iterations': total_iterations,
                'loss': loss,
                'psnr': psnr,
                'sh_degree': preset['sh_degree'],
                'gpu_memory_mb': get_gpu_memory_mb(),
                'frame_idx': frame_count,
                'total_frames': args.max_frames,
                'progress': frame_count / args.max_frames,
            }
            
            # === VIDEO 1: Live rendering ===
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            
            # Add labels with engine name
            engine_label = args.engine.upper()
            cv2.putText(rgb_bgr, f"INPUT (frame {frame.idx})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(rendered_bgr, f"{engine_label} RENDER", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            live_frame = np.concatenate([rgb_bgr, rendered_bgr], axis=1)
            live_frame = draw_metrics_panel(live_frame, display_metrics, panel_height)
            live_writer.write(live_frame)
            
            # === VIDEO 2: Splat visualization ===
            # Get Gaussian positions and colors (if any exist)
            if num_gaussians > 0:
                # Support both graphdeco and gsplat engines
                if hasattr(engine, '_gaussians'):
                    # graphdeco engine
                    xyz = engine._gaussians.get_xyz.detach().cpu().numpy()
                    sh_dc = engine._gaussians._features_dc.detach().cpu().numpy()
                    if sh_dc.ndim == 3:
                        colors = (np.clip(sh_dc[:, 0, :] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
                    else:
                        colors = np.full((len(xyz), 3), 128, dtype=np.uint8)
                elif hasattr(engine, '_means'):
                    # gsplat engine
                    xyz = engine._means.detach().cpu().numpy()
                    if hasattr(engine, '_sh_coeffs') and engine._sh_coeffs is not None:
                        sh_dc = engine._sh_coeffs[:, 0, :].detach().cpu().numpy()
                        C0 = 0.28209479177387814
                        colors = (np.clip(sh_dc * C0 + 0.5, 0, 1) * 255).astype(np.uint8)
                    else:
                        colors = np.full((len(xyz), 3), 128, dtype=np.uint8)
                else:
                    xyz = np.zeros((0, 3))
                    colors = np.zeros((0, 3), dtype=np.uint8)
                
                # Render point cloud from current viewpoint
                splat_view = render_point_cloud_view(xyz, colors, pose, intrinsics, (width, height))
            else:
                # No Gaussians yet - black frame
                splat_view = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add label with engine name
            cv2.putText(splat_view, f"{args.engine.upper()} SPLATS ({num_gaussians:,})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            splat_frame = draw_metrics_panel(splat_view, display_metrics, panel_height)
            splat_writer.write(splat_frame)
            
            # Log progress
            if frame_count % 25 == 0:
                logger.info(f"Frame {frame_count}/{args.max_frames}: {num_gaussians:,} Gaussians, "
                           f"loss={loss:.4f}, PSNR={psnr:.1f}dB, FPS={current_fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    
    finally:
        live_writer.release()
        splat_writer.release()
        
        # Final stats
        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info("=" * 60)
        logger.info("Frame processing complete!")
        logger.info(f"  Frames: {frame_count}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Average FPS: {avg_fps:.2f}")
        logger.info(f"  Gaussians before refinement: {engine.get_num_gaussians():,}")
        logger.info("=" * 60)
        
        # === GLOBAL REFINEMENT (Mapping mode only) ===
        if args.mode == 'mapping' and args.refinement_iters > 0 and engine.get_num_gaussians() > 0:
            logger.info(f"Starting global refinement ({args.refinement_iters} iterations)...")
            logger.info("  Training on ALL cameras equally to build complete map...")
            
            refinement_start = time.time()
            
            # Run refinement in batches for progress logging
            batch_size = 100
            n_batches = args.refinement_iters // batch_size
            
            for batch_idx in range(n_batches):
                metrics = engine.optimize_step(batch_size)
                total_iterations += batch_size
                
                if (batch_idx + 1) % 5 == 0:
                    loss = metrics.get('loss', 0)
                    n_gauss = metrics.get('num_gaussians', 0)
                    elapsed = time.time() - refinement_start
                    progress = (batch_idx + 1) / n_batches * 100
                    logger.info(f"  Refinement {progress:.0f}%: loss={loss:.4f}, "
                               f"gaussians={n_gauss:,}, time={elapsed:.1f}s")
            
            # Remaining iterations
            remaining = args.refinement_iters % batch_size
            if remaining > 0:
                engine.optimize_step(remaining)
                total_iterations += remaining
            
            refinement_time = time.time() - refinement_start
            logger.info(f"Global refinement complete in {refinement_time:.1f}s")
            logger.info(f"  Final Gaussians: {engine.get_num_gaussians():,}")
        
        logger.info("=" * 60)
        logger.info(f"Final model: {engine.get_num_gaussians():,} Gaussians")
        logger.info("=" * 60)
        
        # === VIDEO 3: Model flythrough (orbit) ===
        logger.info("Generating model flythrough video...")
        
        # Calculate scene center from Gaussian positions
        if engine.get_num_gaussians() > 0:
            # Support both graphdeco and gsplat engines
            if hasattr(engine, '_gaussians'):
                xyz = engine._gaussians.get_xyz.detach().cpu().numpy()
            elif hasattr(engine, '_means'):
                xyz = engine._means.detach().cpu().numpy()
            else:
                xyz = np.zeros((0, 3))
            scene_center = xyz.mean(axis=0) if len(xyz) > 0 else np.array([0, 0, 1.0])
        else:
            logger.warning("No Gaussians to render for flythrough")
            scene_center = np.array([0, 0, 1.0])
            xyz = np.zeros((0, 3))
        
        # Generate orbit poses
        orbit_poses = generate_orbit_poses(
            center=scene_center,
            radius=args.orbit_radius,
            height=0.5,
            n_frames=args.orbit_frames,
            intrinsics=intrinsics
        )
        
        orbit_video_path = output_dir / "3_model_flythrough.mp4"
        orbit_writer = cv2.VideoWriter(str(orbit_video_path), fourcc, 30, (width, height + 60))
        
        for i, orbit_pose in enumerate(orbit_poses):
            # Render from orbit pose
            rendered = engine.render_view(orbit_pose, (width, height))
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            
            # Add info panel
            info_panel = np.zeros((60, width, 3), dtype=np.uint8)
            info_panel[:] = (30, 30, 30)
            cv2.putText(info_panel, f"{args.engine.upper()} FLYTHROUGH - {engine.get_num_gaussians():,} Gaussians", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(info_panel, f"Angle: {i * 360 // args.orbit_frames}deg  |  Frame {i+1}/{args.orbit_frames}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            orbit_frame = np.vstack([rendered_bgr, info_panel])
            orbit_writer.write(orbit_frame)
            
            if (i + 1) % 30 == 0:
                logger.info(f"  Orbit frame {i+1}/{args.orbit_frames}")
        
        orbit_writer.release()
        logger.info(f"  3. Model flythrough: {orbit_video_path}")
        
        # === VIDEO 4: Final map quality - render all training views ===
        if args.mode == 'mapping':
            logger.info("Generating final map renders from all viewpoints...")
            
            final_renders_dir = output_dir / "final_map_renders"
            final_renders_dir.mkdir(exist_ok=True)
            
            # Render from all processed frames to show the complete map
            total_psnr = 0
            render_count = 0
            
            for i, frame in enumerate(source):
                if i >= args.max_frames:
                    break
                if frame.depth is None:
                    continue
                
                rendered = engine.render_view(frame.pose, (width, height))
                gt = frame.get_rgb_uint8()
                
                # Calculate PSNR
                mse = np.mean((gt.astype(float) - rendered.astype(float)) ** 2)
                psnr = 10 * np.log10(255**2 / (mse + 1e-10)) if mse > 0 else 40
                total_psnr += psnr
                render_count += 1
                
                # Save comparison every 10 frames
                if i % 10 == 0:
                    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
                    rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                    
                    # Add labels with PSNR and engine name
                    cv2.putText(gt_bgr, f"GT (frame {frame.idx})", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(rendered_bgr, f"{args.engine.upper()} (PSNR: {psnr:.1f}dB)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    comparison = np.concatenate([gt_bgr, rendered_bgr], axis=1)
                    save_path = final_renders_dir / f"map_frame{frame.idx:04d}.png"
                    cv2.imwrite(str(save_path), comparison)
            
            avg_psnr = total_psnr / render_count if render_count > 0 else 0
            logger.info(f"  Average PSNR across all views: {avg_psnr:.2f} dB")
            logger.info(f"  Final renders saved to: {final_renders_dir}")
        
        # === MESH EXTRACTION ===
        if args.mode == 'mapping' and engine.get_num_gaussians() > 0:
            logger.info("Extracting mesh from Gaussian splat...")
            
            try:
                from skimage import measure
                
                # Get Gaussian data - support both graphdeco and gsplat engines
                if hasattr(engine, '_gaussians'):
                    # graphdeco engine
                    xyz = engine._gaussians.get_xyz.detach().cpu().numpy()
                    opacities = engine._gaussians.get_opacity.detach().cpu().numpy().squeeze()
                    scales = engine._gaussians.get_scaling.detach().cpu().numpy()
                    sh_dc = engine._gaussians._features_dc.detach().cpu().numpy()
                    if sh_dc.ndim == 3:
                        colors = np.clip(sh_dc[:, 0, :] * 0.5 + 0.5, 0, 1)
                    else:
                        colors = np.full((len(xyz), 3), 0.5)
                elif hasattr(engine, '_means'):
                    # gsplat engine
                    xyz = engine._means.detach().cpu().numpy()
                    opacities = torch.sigmoid(engine._opacities).detach().cpu().numpy()
                    scales = torch.exp(engine._scales).detach().cpu().numpy()
                    if hasattr(engine, '_sh_coeffs') and engine._sh_coeffs is not None:
                        sh_dc = engine._sh_coeffs[:, 0, :].detach().cpu().numpy()
                        C0 = 0.28209479177387814
                        colors = np.clip(sh_dc * C0 + 0.5, 0, 1)
                    else:
                        colors = np.full((len(xyz), 3), 0.5)
                else:
                    logger.warning("Unknown engine type, skipping mesh extraction")
                    raise ValueError("Unknown engine type")
                
                # Filter by opacity - use higher threshold for cleaner mesh
                mask = opacities > 0.3
                xyz_filtered = xyz[mask]
                colors_filtered = colors[mask]
                scales_filtered = scales[mask]
                opacities_filtered = opacities[mask]
                
                logger.info(f"  Using {len(xyz_filtered):,} Gaussians with opacity > 0.3")
                
                # Create voxel grid for marching cubes - higher resolution
                margin = 0.2
                min_bound = xyz_filtered.min(axis=0) - margin
                max_bound = xyz_filtered.max(axis=0) + margin
                
                # Higher grid resolution for better mesh
                grid_res = 192
                voxel_size = (max_bound - min_bound) / grid_res
                
                logger.info(f"  Creating {grid_res}^3 voxel grid...")
                logger.info(f"  Scene bounds: {min_bound} to {max_bound}")
                
                # Create density field by splatting Gaussians
                density = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
                color_accum = np.zeros((grid_res, grid_res, grid_res, 3), dtype=np.float32)
                weight_accum = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
                
                # Splat each Gaussian into the voxel grid
                logger.info(f"  Splatting {len(xyz_filtered):,} Gaussians into voxel grid...")
                for i in range(len(xyz_filtered)):
                    # Position in grid coordinates
                    pos = (xyz_filtered[i] - min_bound) / voxel_size
                    
                    # Gaussian radius (use mean scale) - larger for better coverage
                    radius = np.mean(scales_filtered[i]) / np.mean(voxel_size) * 3
                    radius = max(1.5, min(radius, 8))  # Wider range
                    
                    # Splat into nearby voxels
                    x0, y0, z0 = int(pos[0]), int(pos[1]), int(pos[2])
                    r = int(np.ceil(radius))
                    
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            for dz in range(-r, r+1):
                                xi, yi, zi = x0 + dx, y0 + dy, z0 + dz
                                if 0 <= xi < grid_res and 0 <= yi < grid_res and 0 <= zi < grid_res:
                                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                    weight = opacities_filtered[i] * np.exp(-0.5 * (dist / radius)**2)
                                    density[xi, yi, zi] += weight
                                    color_accum[xi, yi, zi] += weight * colors_filtered[i]
                                    weight_accum[xi, yi, zi] += weight
                
                # Normalize colors by accumulated weight
                mask_nonzero = weight_accum > 1e-6
                for c in range(3):
                    color_accum[:, :, :, c][mask_nonzero] /= weight_accum[mask_nonzero]
                
                # Normalize density
                if density.max() > 0:
                    density = density / density.max()
                
                logger.info(f"  Density range: {density.min():.4f} to {density.max():.4f}")
                
                # Marching cubes with lower threshold for more complete mesh
                threshold = 0.05
                try:
                    verts, faces, normals, values = measure.marching_cubes(density, threshold)
                    
                    # Scale vertices to world coordinates
                    verts = verts * voxel_size + min_bound
                    
                    # Get vertex colors by trilinear interpolation
                    vert_colors = np.zeros((len(verts), 3))
                    for vi, v_world in enumerate(verts):
                        grid_pos = (v_world - min_bound) / voxel_size
                        xi = int(np.clip(grid_pos[0], 0, grid_res-1))
                        yi = int(np.clip(grid_pos[1], 0, grid_res-1))
                        zi = int(np.clip(grid_pos[2], 0, grid_res-1))
                        vert_colors[vi] = color_accum[xi, yi, zi]
                    
                    # Ensure colors are valid
                    vert_colors = np.clip(vert_colors, 0, 1)
                    # Default gray for vertices with no color
                    no_color = np.all(vert_colors < 0.01, axis=1)
                    vert_colors[no_color] = 0.5
                    
                    logger.info(f"  Extracted mesh: {len(verts):,} vertices, {len(faces):,} faces")
                    
                    # Save mesh as PLY - ensure directory exists first
                    mesh_dir = output_dir / "final"
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    mesh_path = mesh_dir / "mesh.ply"
                    with open(mesh_path, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(verts)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("property float nx\n")
                        f.write("property float ny\n")
                        f.write("property float nz\n")
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                        f.write(f"element face {len(faces)}\n")
                        f.write("property list uchar int vertex_indices\n")
                        f.write("end_header\n")
                        
                        for vi in range(len(verts)):
                            v = verts[vi]
                            n = normals[vi]
                            c = (vert_colors[vi] * 255).astype(int)
                            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} ")
                            f.write(f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f} ")
                            f.write(f"{c[0]} {c[1]} {c[2]}\n")
                        
                        for face in faces:
                            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                    
                    logger.info(f"  Mesh saved to: {mesh_path}")
                    
                    # === VIDEO 5: Mesh flythrough ===
                    logger.info("Generating mesh flythrough video...")
                    
                    mesh_video_path = output_dir / "4_mesh_flythrough.mp4"
                    mesh_writer = cv2.VideoWriter(str(mesh_video_path), fourcc, 30, (width, height + 60))
                    
                    # Render mesh from orbit poses
                    for i, orbit_pose in enumerate(orbit_poses):
                        # Render mesh using simple rasterization
                        mesh_img = render_mesh_view(verts, faces, vert_colors, normals, 
                                                   orbit_pose, intrinsics, (width, height))
                        mesh_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
                        
                        # Add info panel
                        info_panel = np.zeros((60, width, 3), dtype=np.uint8)
                        info_panel[:] = (30, 30, 30)
                        cv2.putText(info_panel, f"{args.engine.upper()} MESH - {len(verts):,} vertices, {len(faces):,} faces", 
                                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                        cv2.putText(info_panel, f"Angle: {i * 360 // args.orbit_frames}deg  |  Frame {i+1}/{args.orbit_frames}", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        mesh_frame = np.vstack([mesh_bgr, info_panel])
                        mesh_writer.write(mesh_frame)
                        
                        if (i + 1) % 30 == 0:
                            logger.info(f"    Mesh frame {i+1}/{args.orbit_frames}")
                    
                    mesh_writer.release()
                    logger.info(f"  4. Mesh flythrough: {mesh_video_path}")
                    
                except Exception as e:
                    logger.warning(f"  Marching cubes failed: {e}")
                    
            except ImportError:
                logger.warning("  scikit-image not available, skipping mesh extraction")
            except Exception as e:
                logger.warning(f"  Mesh extraction failed: {e}")
        
        # Save final state
        logger.info("Saving final state...")
        final_path = output_dir / "final"
        engine.save_state(str(final_path))
        
        # Save metrics summary
        metrics_path = output_dir / "metrics_summary.txt"
        with open(metrics_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("3D GAUSSIAN SPLATTING - ENVIRONMENT MAP\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Mode: {args.mode.upper()}\n")
            f.write(f"Dataset: {args.sequence}\n")
            f.write(f"Quality preset: {args.quality}\n")
            f.write(f"Resolution: {width}x{height}\n\n")
            f.write("PERFORMANCE:\n")
            f.write(f"  Frames processed: {frame_count}\n")
            f.write(f"  Total time: {total_time:.1f}s\n")
            f.write(f"  Average FPS: {avg_fps:.2f}\n")
            f.write(f"  Total iterations: {total_iterations:,}\n")
            if args.mode == 'mapping':
                f.write(f"  Refinement iterations: {args.refinement_iters}\n")
            f.write(f"\nMODEL:\n")
            f.write(f"  Final Gaussians: {engine.get_num_gaussians():,}\n")
            f.write(f"  Max Gaussians: {max_gaussians:,}\n")
            f.write(f"  SH Degree: {preset['sh_degree']}\n")
            f.write(f"  Recency weight: {recency_weight}\n\n")
            f.write("OUTPUT FILES:\n")
            f.write(f"  1. {live_video_path.name} - Live rendering with metrics\n")
            f.write(f"  2. {splat_video_path.name} - Splat visualization\n")
            f.write(f"  3. {orbit_video_path.name} - Model flythrough\n")
            if args.mode == 'mapping':
                f.write(f"  4. final_map_renders/ - Renders from all viewpoints\n")
            f.write(f"  final/ - Saved Gaussian model (.ply)\n")
        
        logger.info(f"Metrics saved to: {metrics_path}")
        logger.info("=" * 60)
        logger.info("OUTPUT FILES:")
        logger.info(f"  1. {live_video_path}")
        logger.info(f"  2. {splat_video_path}")
        logger.info(f"  3. {orbit_video_path}")
        logger.info(f"  4. {metrics_path}")
        logger.info(f"  5. {final_path}/")
        logger.info("=" * 60)
        logger.info("Done!")


if __name__ == "__main__":
    main()
