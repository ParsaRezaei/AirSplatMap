#!/usr/bin/env python3
"""
MonoGS Runner for TUM Dataset - Outputs to AirSplatMap format

Usage:
    python scripts/run_monogs_tum.py --sequence rgbd_dataset_freiburg1_xyz
    python scripts/run_monogs_tum.py --sequence rgbd_dataset_freiburg2_desk
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

MONOGS_PATH = Path("/home/past/parsa/MonoGS")
sys.path.insert(0, str(MONOGS_PATH))

import yaml
import numpy as np
import cv2
import torch
from plyfile import PlyData


def load_gaussian_ply(ply_path: str):
    """Load Gaussian splat model from 3DGS-format PLY file."""
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    
    # Extract Gaussian parameters
    means = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    
    # Scales are stored in log-space
    scales_log = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32)
    scales = np.exp(scales_log)
    
    # Quaternions (rotation)
    quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)
    # Normalize quaternions
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-10)
    
    # Opacity is stored as logit (inverse sigmoid)
    opacities_logit = v['opacity'].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities_logit))
    
    # Colors from spherical harmonics DC component
    # SH DC to RGB: color = 0.5 + SH_DC * C0 where C0 = 0.28209479
    C0 = 0.28209479177387814
    sh_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1).astype(np.float32)
    colors = np.clip(0.5 + sh_dc * C0, 0, 1)
    
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'colors': colors,
        'sh_dc': sh_dc  # Raw SH DC values
    }


def render_gaussians_3dgs(gaussians, pose, intrinsics, img_size):
    """Render Gaussians using GRAPHDECO's diff-gaussian-rasterization CUDA rasterizer."""
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except ImportError:
        print("Warning: diff-gaussian-rasterization not available")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    width, height = img_size
    
    # Convert to torch tensors
    means = torch.tensor(gaussians['means'], device=device, dtype=torch.float32)
    scales = torch.tensor(gaussians['scales'], device=device, dtype=torch.float32)
    quats = torch.tensor(gaussians['quats'], device=device, dtype=torch.float32)
    opacities = torch.tensor(gaussians['opacities'], device=device, dtype=torch.float32).unsqueeze(-1)
    
    # SH coefficients - use raw values
    sh_dc = torch.tensor(gaussians['sh_dc'], device=device, dtype=torch.float32)
    # Format for rasterizer: [N, (degree+1)^2, 3] -> for degree 0: [N, 1, 3]
    sh_coeffs = sh_dc.unsqueeze(1).contiguous()
    
    # Camera parameters
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Compute projection matrix
    znear, zfar = 0.01, 100.0
    
    # OpenGL-style projection matrix
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = (width - 2 * cx) / width
    proj[1, 2] = (height - 2 * cy) / height
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -2 * zfar * znear / (zfar - znear)
    proj[3, 2] = -1.0
    
    # View matrix (world to camera)
    pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(pose_tensor)
    
    # Full projection matrix
    full_proj = proj @ viewmat
    
    # Camera position
    cam_pos = pose_tensor[:3, 3]
    
    try:
        # Setup rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=width / (2 * fx),
            tanfovy=height / (2 * fy),
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=viewmat.T.contiguous(),  # Transpose for column-major
            projmatrix=full_proj.T.contiguous(),
            sh_degree=0,
            campos=cam_pos,
            prefiltered=False
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Rasterize - returns (rendered_image, radii, depth_image)
        rendered, radii, depth = rasterizer(
            means3D=means,
            means2D=torch.zeros_like(means[:, :2]),
            shs=sh_coeffs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None
        )
        
        # Convert to numpy [C, H, W] -> [H, W, C]
        rendered_np = rendered.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        rendered_np = (rendered_np * 255).astype(np.uint8)
        
        return rendered_np
        
    except Exception as e:
        print(f"3DGS rasterization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def render_gaussians_gsplat(gaussians, pose, intrinsics, img_size):
    """Render Gaussians using gsplat library."""
    # gsplat JIT compilation can fail on some systems, use software fallback
    return None


def render_gaussians_software(gaussians, pose, intrinsics, img_size):
    """Fallback software rendering of Gaussians as colored points with size based on scale."""
    width, height = img_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    
    means = gaussians['means']
    colors = gaussians['colors']
    scales = gaussians['scales']
    opacities = gaussians['opacities']
    
    if len(means) == 0:
        return img
    
    # Transform to camera frame
    pose_inv = np.linalg.inv(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    
    pts_cam = (R @ means.T).T + t
    
    # Project to image
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    z = pts_cam[:, 2]
    valid = z > 0.1
    
    u = np.zeros(len(means))
    v = np.zeros(len(means))
    u[valid] = fx * pts_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * pts_cam[valid, 1] / z[valid] + cy
    
    # Filter valid projections
    in_bounds = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    in_bounds &= opacities > 0.1  # Only render visible Gaussians
    
    # Sort by depth (back to front)
    indices = np.where(in_bounds)[0]
    depths = z[indices]
    sorted_idx = indices[np.argsort(-depths)]
    
    # Render with size based on scale
    for idx in sorted_idx:
        px, py = int(u[idx]), int(v[idx])
        d = z[idx]
        
        # Point size based on scale (project average scale to screen)
        avg_scale = scales[idx].mean()
        point_size = max(1, int(fx * avg_scale / d))
        point_size = min(point_size, 10)  # Cap size
        
        # Color with opacity blending
        color = (colors[idx] * 255 * opacities[idx]).astype(np.uint8)
        
        # Draw filled circle
        cv2.circle(img, (px, py), point_size, color.tolist(), -1)
    
    return img


def generate_orbit_poses(center, radius, height_offset, n_frames, intrinsics):
    """Generate camera poses orbiting around a center point."""
    poses = []
    
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        cam_pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + height_offset
        ])
        
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-10)
        up = np.cross(right, forward)
        
        R = np.stack([right, -up, forward], axis=1)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = cam_pos
        
        poses.append(pose)
    
    return poses


def generate_flythrough_video(ply_path: str, output_path: str, n_frames: int = 120):
    """Generate flythrough video from Gaussian splat model."""
    print(f"Generating Gaussian splat flythrough: {output_path}")
    
    try:
        # Load Gaussian model
        gaussians = load_gaussian_ply(ply_path)
        n_gaussians = len(gaussians['means'])
        print(f"  Loaded {n_gaussians:,} Gaussians")
        
        if n_gaussians < 100:
            print("  Warning: Too few Gaussians for flythrough")
            return False
        
        # Compute bounding box and orbit parameters
        means = gaussians['means']
        center = means.mean(axis=0)
        extent = means.max(axis=0) - means.min(axis=0)
        radius = np.linalg.norm(extent[:2]) * 0.8
        height_offset = extent[2] * 0.3
        
        intrinsics = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
        width, height = 640, 480
        
        # Generate orbit poses
        poses = generate_orbit_poses(center, radius, height_offset, n_frames, intrinsics)
        
        # Try 3DGS CUDA rasterizer first (highest quality)
        test_render = render_gaussians_3dgs(gaussians, poses[0], intrinsics, (width, height))
        if test_render is not None:
            print("  Using GRAPHDECO 3DGS CUDA rasterizer (high quality)")
            render_fn = lambda g, p: render_gaussians_3dgs(g, p, intrinsics, (width, height))
        else:
            # Fallback to gsplat
            test_render = render_gaussians_gsplat(gaussians, poses[0], intrinsics, (width, height))
            if test_render is not None:
                print("  Using gsplat GPU rendering")
                render_fn = lambda g, p: render_gaussians_gsplat(g, p, intrinsics, (width, height))
            else:
                print("  Using software rendering (fallback - lower quality)")
                render_fn = lambda g, p: render_gaussians_software(g, p, intrinsics, (width, height))
        
        # Render frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height + 60))
        
        for i, pose in enumerate(poses):
            img = render_fn(gaussians, pose)
            if img is None:
                img = render_gaussians_software(gaussians, pose, intrinsics, (width, height))
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add info panel
            panel = np.zeros((60, width, 3), dtype=np.uint8)
            panel[:] = (30, 30, 30)
            cv2.putText(panel, f"MonoGS FLYTHROUGH - {n_gaussians:,} Gaussians",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(panel, f"Angle: {i * 360 // n_frames}deg  |  Frame {i+1}/{n_frames}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            frame = np.vstack([img_bgr, panel])
            out.write(frame)
            
            if (i + 1) % 30 == 0:
                print(f"  Rendered frame {i+1}/{n_frames}")
        
        out.release()
        print(f"  Generated: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error generating flythrough: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Render points
    for idx in sorted_idx:
        px, py = int(u[idx]), int(v[idx])
        d = z[idx]
        
        # Draw point with size
        for dy in range(-point_size, point_size + 1):
            for dx in range(-point_size, point_size + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if d < depth_buffer[ny, nx]:
                        depth_buffer[ny, nx] = d
                        color = (colors[idx] * 255).astype(np.uint8)
                        img[ny, nx] = color
    
    return img


def generate_orbit_poses(center, radius, height, n_frames, intrinsics):
    """Generate camera poses orbiting around a center point."""
    poses = []
    
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        # Camera position
        cam_pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + height
        ])
        
        # Look at center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector (world Z)
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        
        # Rotation matrix (camera to world)
        R = np.stack([right, -up, forward], axis=1)
        
        # 4x4 pose matrix (camera to world)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = cam_pos
        
        poses.append(pose)
    
    return poses


def create_monogs_config(dataset_path: str, output_dir: str, sequence_name: str) -> str:
    """Create MonoGS config file for TUM sequence."""
    
    if "freiburg1" in sequence_name:
        calib = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3,
            "k1": 0.2624, "k2": -0.9531, "p1": -0.0054, "p2": 0.0026, "k3": 1.1633,
            "width": 640, "height": 480, "depth_scale": 5000.0, "distorted": True}
    elif "freiburg2" in sequence_name:
        calib = {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7,
            "k1": 0.2312, "k2": -0.7849, "p1": -0.0033, "p2": -0.0001, "k3": 0.9172,
            "width": 640, "height": 480, "depth_scale": 5000.0, "distorted": True}
    elif "freiburg3" in sequence_name:
        calib = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
            "width": 640, "height": 480, "depth_scale": 5000.0, "distorted": False}
    else:
        calib = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
            "width": 640, "height": 480, "depth_scale": 5000.0, "distorted": False}
    
    config = {
        "inherit_from": "configs/mono/tum/base_config.yaml",
        "Results": {"save_results": True, "save_dir": output_dir, 
                    "use_gui": False, "use_wandb": False, "eval_rendering": True},
        "Dataset": {"dataset_path": dataset_path, "Calibration": calib}
    }
    
    config_path = os.path.join(output_dir, "monogs_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_monogs(config_path: str, output_dir: str) -> dict:
    """Run MonoGS SLAM and return metrics."""
    
    start_time = time.time()
    cmd = ["python", str(MONOGS_PATH / "slam.py"), "--config", config_path, "--eval"]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    log_path = os.path.join(output_dir, "monogs_log.txt")
    
    print(f"Running MonoGS...")
    print(f"Log: {log_path}")
    
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   cwd=str(MONOGS_PATH), env=env, text=True)
        ate_values = []
        frame_count = 0
        
        for line in process.stdout:
            log_file.write(line)
            print(line, end="")
            if "RMSE ATE [m]" in line:
                try:
                    ate = float(line.split("RMSE ATE [m]")[-1].strip())
                    ate_values.append(ate)
                except: pass
            if "frame:" in line.lower():
                try:
                    frame = int(line.split("frame:")[-1].split()[0])
                    frame_count = max(frame_count, frame)
                except: pass
        
        process.wait()
    
    elapsed = time.time() - start_time
    return {
        "total_time": elapsed, "frames_processed": frame_count,
        "fps": frame_count / elapsed if elapsed > 0 else 0,
        "final_ate": ate_values[-1] if ate_values else None,
        "ate_history": ate_values, "success": process.returncode == 0
    }


def find_monogs_outputs(output_dir: str) -> dict:
    """Find MonoGS output files - they're in output_dir/datasets_tum/<timestamp>/"""
    
    # MonoGS creates datasets_tum/<timestamp> inside save_dir
    datasets_tum = Path(output_dir) / "datasets_tum"
    
    if datasets_tum.exists():
        subdirs = sorted([d for d in datasets_tum.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        if subdirs:
            latest = subdirs[0]
            pc_path = latest / "point_cloud" / "final" / "point_cloud.ply"
            if pc_path.exists():
                return {
                    "results_dir": str(latest),
                    "point_cloud": str(pc_path),
                    "plots": str(latest / "plot"),
                    "config": str(latest / "config.yml")
                }
    
    # Fallback: check MonoGS results folder
    results_base = MONOGS_PATH / "results" / "datasets_tum"
    if results_base.exists():
        subdirs = sorted([d for d in results_base.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        if subdirs:
            latest = subdirs[0]
            return {
                "results_dir": str(latest),
                "point_cloud": str(latest / "point_cloud" / "final" / "point_cloud.ply"),
                "plots": str(latest / "plot"),
                "config": str(latest / "config.yml")
            }
    
    return {}


def copy_trajectory_plots(plots_dir: str, output_dir: str):
    plots_path = Path(plots_dir)
    if not plots_path.exists():
        return
    
    final_plots = list(plots_path.glob("*final*.png")) or list(plots_path.glob("*.png"))
    for png in final_plots[:3]:
        dst = Path(output_dir) / f"2_trajectory_{png.name}"
        shutil.copy(png, dst)
        print(f"Copied: {dst}")


def copy_stats_json(plots_dir: str) -> dict:
    plots_path = Path(plots_dir)
    if not plots_path.exists():
        return None
    
    # First try final stats
    final_stats = plots_path / "stats_final.json"
    if final_stats.exists():
        with open(final_stats) as f:
            return json.load(f)
    
    # Otherwise get most recent
    stats_files = sorted(plots_path.glob("stats_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if stats_files:
        with open(stats_files[0]) as f:
            return json.load(f)
    return None


def write_metrics_summary(output_dir: str, metrics: dict, stats: dict, sequence_name: str):
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MonoGS - GAUSSIAN SPLATTING SLAM\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mode: SLAM (Monocular)\n")
        f.write(f"Dataset: {sequence_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"  Frames processed: {metrics.get('frames_processed', 'N/A')}\n")
        f.write(f"  Total time: {metrics.get('total_time', 0):.1f}s\n")
        f.write(f"  Average FPS: {metrics.get('fps', 0):.2f}\n\n")
        
        f.write("TRACKING ACCURACY (ATE):\n")
        if stats:
            f.write(f"  RMSE: {stats.get('rmse', 0)*1000:.2f}mm ({stats.get('rmse', 0):.6f}m)\n")
            f.write(f"  Mean: {stats.get('mean', 0)*1000:.2f}mm\n")
            f.write(f"  Median: {stats.get('median', 0)*1000:.2f}mm\n")
            f.write(f"  Std: {stats.get('std', 0)*1000:.2f}mm\n")
            f.write(f"  Min: {stats.get('min', 0)*1000:.2f}mm\n")
            f.write(f"  Max: {stats.get('max', 0)*1000:.2f}mm\n")
        elif metrics.get('final_ate'):
            f.write(f"  Final RMSE: {metrics['final_ate']*1000:.2f}mm\n")
        
        f.write("\nOUTPUT FILES:\n")
        f.write("  2_trajectory_*.png - Trajectory plots\n")
        f.write("  3_model_flythrough.mp4 - Gaussian model flythrough video\n")
        f.write("  final/point_cloud.ply - Gaussian splat model\n")
        f.write("  monogs_log.txt - Full SLAM log\n")
    
    print(f"Written: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MonoGS on TUM dataset")
    parser.add_argument("--dataset-root", type=str, default="/home/past/parsa/datasets/tum")
    parser.add_argument("--output-root", type=str, default="/home/past/parsa/AirSplatMap/output/monogs")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--skip-slam", action="store_true")
    args = parser.parse_args()
    
    dataset_path = os.path.join(args.dataset_root, args.sequence)
    output_dir = os.path.join(args.output_root, args.sequence)
    final_dir = os.path.join(output_dir, "final")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"MonoGS TUM Runner")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    metrics = {}
    
    if not args.skip_slam:
        config_path = create_monogs_config(dataset_path, output_dir, args.sequence)
        print(f"Created config: {config_path}")
        metrics = run_monogs(config_path, output_dir)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Find outputs
    outputs = find_monogs_outputs(output_dir)
    stats = None
    
    if outputs:
        print(f"\nFound outputs: {outputs['results_dir']}")
        
        if os.path.exists(outputs.get("point_cloud", "")):
            dst_ply = os.path.join(final_dir, "point_cloud.ply")
            shutil.copy(outputs["point_cloud"], dst_ply)
            print(f"Copied point cloud: {dst_ply}")
            
            # Generate flythrough video
            flythrough_path = os.path.join(output_dir, "3_model_flythrough.mp4")
            generate_flythrough_video(dst_ply, flythrough_path, n_frames=120)
        
        if os.path.exists(outputs.get("plots", "")):
            copy_trajectory_plots(outputs["plots"], output_dir)
            stats = copy_stats_json(outputs["plots"])
    
    if not metrics:
        metrics_file = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
    
    write_metrics_summary(output_dir, metrics, stats, args.sequence)
    
    print(f"\n{'='*60}")
    print(f"MonoGS Complete!")
    print(f"Results: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
