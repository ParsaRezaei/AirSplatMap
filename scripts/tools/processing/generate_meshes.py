#!/usr/bin/env python3
"""
Generate meshes and mesh flythrough videos for all processed scenes.

Uses TSDF fusion from depth maps for better mesh quality, and follows
the original camera trajectory for the flythrough video.
"""

import numpy as np
from pathlib import Path
from skimage import measure
import cv2
import json
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TSDFVolume:
    """Truncated Signed Distance Function volume for mesh extraction."""
    
    def __init__(self, vol_bounds, voxel_size=0.02, trunc_margin=0.06):
        self.vol_bounds = np.array(vol_bounds, dtype=np.float32)
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin
        
        vol_dim = np.ceil((self.vol_bounds[1] - self.vol_bounds[0]) / voxel_size).astype(int)
        self.vol_dim = vol_dim
        
        logger.info(f"  TSDF volume: {vol_dim[0]}x{vol_dim[1]}x{vol_dim[2]} voxels ({np.prod(vol_dim)/1e6:.1f}M)")
        logger.info(f"  Voxel size: {voxel_size*100:.1f}cm, truncation: {trunc_margin*100:.1f}cm")
        
        self.tsdf_vol = np.ones(vol_dim, dtype=np.float32)
        self.weight_vol = np.zeros(vol_dim, dtype=np.float32)
        self.color_vol = np.zeros((*vol_dim, 3), dtype=np.float32)
        
        # Precompute voxel coordinates
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        self.vox_coords = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        self.world_coords = self.vol_bounds[0] + self.voxel_size * (self.vox_coords + 0.5)
    
    def integrate(self, depth_img, color_img, intrinsics, pose):
        """Integrate a depth frame into the TSDF volume."""
        height, width = depth_img.shape[:2]
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        # Transform voxel centers to camera frame
        pose_inv = np.linalg.inv(pose)
        R = pose_inv[:3, :3]
        t = pose_inv[:3, 3]
        
        cam_coords = (R @ self.world_coords.T).T + t
        
        # Project to image
        pix_x = np.round(fx * cam_coords[:, 0] / cam_coords[:, 2] + cx).astype(int)
        pix_y = np.round(fy * cam_coords[:, 1] / cam_coords[:, 2] + cy).astype(int)
        
        valid_pix = (pix_x >= 0) & (pix_x < width) & (pix_y >= 0) & (pix_y < height) & (cam_coords[:, 2] > 0)
        
        depth_vals = np.zeros(len(cam_coords), dtype=np.float32)
        depth_vals[valid_pix] = depth_img[pix_y[valid_pix], pix_x[valid_pix]]
        
        depth_diff = depth_vals - cam_coords[:, 2]
        valid_pts = valid_pix & (depth_vals > 0) & (depth_diff >= -self.trunc_margin)
        
        tsdf_vals = np.clip(depth_diff / self.trunc_margin, -1.0, 1.0)
        
        colors = np.zeros((len(cam_coords), 3), dtype=np.float32)
        if color_img is not None:
            colors[valid_pix] = color_img[pix_y[valid_pix], pix_x[valid_pix]] / 255.0
        
        valid_vox = self.vox_coords[valid_pts]
        w_old = self.weight_vol[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]]
        tsdf_old = self.tsdf_vol[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]]
        color_old = self.color_vol[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]]
        
        w_new = 1.0
        w_sum = w_old + w_new
        
        self.tsdf_vol[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]] = \
            (w_old * tsdf_old + w_new * tsdf_vals[valid_pts]) / w_sum
        self.weight_vol[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]] = np.minimum(w_sum, 50.0)
        self.color_vol[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]] = \
            (w_old[:, None] * color_old + w_new * colors[valid_pts]) / w_sum[:, None]
    
    def extract_mesh(self):
        """Extract mesh using marching cubes."""
        logger.info("  Running marching cubes on TSDF...")
        
        tsdf = self.tsdf_vol.copy()
        tsdf[self.weight_vol < 2] = 1.0  # Mark unobserved as outside
        
        try:
            verts, faces, normals, _ = measure.marching_cubes(tsdf, level=0)
        except Exception as e:
            logger.error(f"  Marching cubes failed: {e}")
            return None, None, None, None
        
        verts = self.vol_bounds[0] + self.voxel_size * verts
        
        verts_vox = ((verts - self.vol_bounds[0]) / self.voxel_size).astype(int)
        verts_vox = np.clip(verts_vox, 0, np.array(self.vol_dim) - 1)
        vert_colors = self.color_vol[verts_vox[:, 0], verts_vox[:, 1], verts_vox[:, 2]]
        
        logger.info(f"  Extracted mesh: {len(verts):,} vertices, {len(faces):,} faces")
        return verts, faces, normals, vert_colors


def render_mesh_view(verts, faces, vert_colors, normals, pose, intrinsics, img_size):
    """Render mesh with triangle rasterization."""
    width, height = img_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(verts) == 0 or len(faces) == 0:
        return img
    
    pose_inv = np.linalg.inv(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    
    verts_cam = (R @ verts.T).T + t
    normals_cam = (R @ normals.T).T
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    z = verts_cam[:, 2]
    valid = z > 0.1
    
    u = np.full(len(verts), -1.0)
    v = np.full(len(verts), -1.0)
    u[valid] = fx * verts_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * verts_cam[valid, 1] / z[valid] + cy
    
    face_data = []
    for face in faces:
        v0, v1, v2 = face
        if z[v0] <= 0.1 or z[v1] <= 0.1 or z[v2] <= 0.1:
            continue
        
        pts = np.array([[u[v0], v[v0]], [u[v1], v[v1]], [u[v2], v[v2]]])
        if pts[:, 0].max() < 0 or pts[:, 0].min() > width or pts[:, 1].max() < 0 or pts[:, 1].min() > height:
            continue
        
        area = 0.5 * abs((pts[1,0]-pts[0,0])*(pts[2,1]-pts[0,1]) - (pts[2,0]-pts[0,0])*(pts[1,1]-pts[0,1]))
        if area < 0.5:
            continue
        
        avg_depth = (z[v0] + z[v1] + z[v2]) / 3
        color = (vert_colors[v0] + vert_colors[v1] + vert_colors[v2]) / 3
        
        n = (normals_cam[v0] + normals_cam[v1] + normals_cam[v2]) / 3
        n = n / (np.linalg.norm(n) + 1e-8)
        shade = max(0.3, abs(n[2]))
        color = np.clip(color * shade, 0, 1)
        
        face_data.append({'pts': pts.astype(np.int32), 'depth': avg_depth, 'color': (color * 255).astype(np.uint8)})
    
    face_data.sort(key=lambda x: -x['depth'])
    
    for fd in face_data:
        pts = fd['pts'].reshape((-1, 1, 2))
        color = tuple(int(c) for c in fd['color'])
        cv2.fillPoly(img, [pts], color)
    
    return img


def load_tum_data(scene_path, max_frames=200, skip=2):
    """Load RGB, depth and poses from TUM dataset."""
    
    # Load groundtruth poses
    gt_file = scene_path / "groundtruth.txt"
    poses_dict = {}
    
    if gt_file.exists():
        with open(gt_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                R = np.array([
                    [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                    [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                    [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
                ])
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = [tx, ty, tz]
                poses_dict[ts] = pose
    
    # Load RGB timestamps
    rgb_dict = {}
    rgb_file = scene_path / "rgb.txt"
    if rgb_file.exists():
        with open(rgb_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    rgb_dict[float(parts[0])] = parts[1]
    
    # Load depth timestamps
    depth_dict = {}
    depth_file = scene_path / "depth.txt"
    if depth_file.exists():
        with open(depth_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    depth_dict[float(parts[0])] = parts[1]
    
    # Match frames
    frames = []
    rgb_times = sorted(rgb_dict.keys())
    pose_times = sorted(poses_dict.keys())
    depth_times = sorted(depth_dict.keys())
    
    for i, rgb_ts in enumerate(rgb_times):
        if i % skip != 0:
            continue
        if max_frames and len(frames) >= max_frames:
            break
        
        # Find closest depth
        closest_depth_ts = min(depth_times, key=lambda x: abs(x - rgb_ts))
        if abs(closest_depth_ts - rgb_ts) > 0.1:
            continue
        
        # Find closest pose
        closest_pose_ts = min(pose_times, key=lambda x: abs(x - rgb_ts))
        if abs(closest_pose_ts - rgb_ts) > 0.1:
            continue
        
        rgb_path = scene_path / rgb_dict[rgb_ts]
        depth_path = scene_path / depth_dict[closest_depth_ts]
        
        if rgb_path.exists() and depth_path.exists():
            frames.append({
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'pose': poses_dict[closest_pose_ts]
            })
    
    return frames


def extract_mesh_tsdf(scene_path, voxel_size=0.015, max_frames=200):
    """Extract mesh using TSDF fusion."""
    
    logger.info(f"  Loading TUM frames...")
    frames = load_tum_data(scene_path, max_frames=max_frames, skip=2)
    
    if not frames:
        logger.error("  No frames loaded!")
        return None, None, None, None, []
    
    logger.info(f"  Loaded {len(frames)} frames")
    
    intrinsics = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
    
    # Compute volume bounds
    logger.info("  Computing volume bounds...")
    all_points = []
    
    for frame in frames[::10]:
        depth = cv2.imread(str(frame['depth_path']), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / 5000.0
        
        h, w = depth.shape
        ys, xs = np.mgrid[0:h:30, 0:w:30]
        zs = depth[ys, xs]
        valid = (zs > 0.1) & (zs < 5.0)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        
        pts_cam = np.stack([
            (xs - intrinsics['cx']) * zs / intrinsics['fx'],
            (ys - intrinsics['cy']) * zs / intrinsics['fy'],
            zs
        ], axis=-1)
        
        pose = frame['pose']
        pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]
        all_points.append(pts_world)
    
    if not all_points:
        return None, None, None, None, []
    
    all_points = np.vstack(all_points)
    margin = 0.3
    vol_bounds = [all_points.min(axis=0) - margin, all_points.max(axis=0) + margin]
    
    # Create TSDF
    tsdf = TSDFVolume(vol_bounds, voxel_size=voxel_size, trunc_margin=voxel_size * 4)
    
    # Integrate frames
    logger.info("  Integrating depth frames into TSDF...")
    for i, frame in enumerate(frames):
        rgb = cv2.imread(str(frame['rgb_path']))
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(str(frame['depth_path']), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / 5000.0
        
        tsdf.integrate(depth, rgb, intrinsics, frame['pose'])
        
        if (i + 1) % 25 == 0:
            logger.info(f"    Integrated {i+1}/{len(frames)}")
    
    verts, faces, normals, vert_colors = tsdf.extract_mesh()
    poses = [f['pose'] for f in frames]
    
    return verts, faces, normals, vert_colors, poses


def save_mesh_ply(verts, faces, normals, vert_colors, path):
    """Save mesh as PLY."""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        
        for vi in range(len(verts)):
            v, n = verts[vi], normals[vi]
            c = (np.clip(vert_colors[vi], 0, 1) * 255).astype(int)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    logger.info(f"  Saved: {path}")


def generate_flythrough(verts, faces, normals, vert_colors, poses, output_path, skip=3):
    """Generate flythrough video following camera trajectory."""
    
    intrinsics = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
    width, height = 640, 480
    
    poses = poses[::skip]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (width, height + 60))
    
    logger.info(f"  Rendering {len(poses)} frames...")
    
    for i, pose in enumerate(poses):
        mesh_img = render_mesh_view(verts, faces, vert_colors, normals, pose, intrinsics, (width, height))
        mesh_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
        
        panel = np.zeros((60, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        cv2.putText(panel, f"TSDF MESH - {len(verts):,} verts, {len(faces):,} faces", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(panel, f"Camera trajectory - Frame {i+1}/{len(poses)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        frame = np.vstack([mesh_bgr, panel])
        out.write(frame)
        
        if (i + 1) % 20 == 0:
            logger.info(f"    Frame {i+1}/{len(poses)}")
    
    out.release()
    logger.info(f"  Saved: {output_path}")


def process_scene(output_dir, dataset_root, voxel_size=0.015, force=False):
    """Process a single scene."""
    
    scene_name = output_dir.name
    mesh_path = output_dir / "final" / "mesh_tsdf.ply"
    video_path = output_dir / "4_mesh_flythrough.mp4"
    
    scene_path = dataset_root / scene_name
    if not scene_path.exists():
        logger.warning(f"  TUM scene not found: {scene_path}")
        return False
    
    if not force and mesh_path.exists() and video_path.exists():
        # Check if video is larger than 500KB (indicating it has actual content)
        if video_path.stat().st_size > 500000:
            logger.info(f"  Already complete, skipping")
            return True
    
    verts, faces, normals, vert_colors, poses = extract_mesh_tsdf(scene_path, voxel_size=voxel_size)
    
    if verts is None or len(verts) == 0:
        logger.error("  Mesh extraction failed!")
        return False
    
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    save_mesh_ply(verts, faces, normals, vert_colors, mesh_path)
    
    generate_flythrough(verts, faces, normals, vert_colors, poses, video_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate TSDF meshes and flythrough videos")
    parser.add_argument("--output-root", type=str, default="./output")
    parser.add_argument("--dataset-root", type=str, default="/home/past/parsa/datasets/tum")
    parser.add_argument("--scenes", nargs="*", help="Specific scenes")
    parser.add_argument("--voxel-size", type=float, default=0.015, help="Voxel size (meters)")
    parser.add_argument("--force", action="store_true", help="Regenerate all")
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    
    if args.scenes:
        scene_dirs = [output_root / s for s in args.scenes]
    else:
        scene_dirs = sorted([d for d in output_root.iterdir() if d.is_dir() and d.name.startswith('rgbd_')])
    
    logger.info(f"Processing {len(scene_dirs)} scene(s) with voxel size {args.voxel_size*100:.1f}cm")
    
    results = {}
    for scene_dir in scene_dirs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {scene_dir.name}")
        logger.info(f"{'='*60}")
        
        try:
            success = process_scene(scene_dir, dataset_root, args.voxel_size, args.force)
            results[scene_dir.name] = success
        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            results[scene_dir.name] = False
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for name, success in results.items():
        logger.info(f"  {'✓' if success else '✗'} {name}")


if __name__ == "__main__":
    main()
