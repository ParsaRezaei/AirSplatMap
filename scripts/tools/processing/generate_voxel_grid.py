#!/usr/bin/env python3
"""
Generate Voxel Occupancy Grids from Gaussian Splats for Autonomous Flight

This creates:
1. Binary occupancy grid (for collision detection)
2. Signed Distance Field (SDF) for path planning
3. Safety zones at configurable distances
4. Export formats: NumPy (.npy), PLY point cloud, binvox

The voxel grid can be used for:
- Collision avoidance
- Path planning (A*, RRT, etc.)
- Safety zone enforcement
- No-fly zone definition
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData
import argparse
import logging
import json
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gaussian_splat(ply_path):
    """Load Gaussian splat from PLY file."""
    logger.info(f"Loading Gaussian splat: {ply_path}")
    plydata = PlyData.read(str(ply_path))
    vertex = plydata['vertex']
    
    xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Get opacity (stored as logit)
    opacity_logit = vertex['opacity']
    opacities = 1 / (1 + np.exp(-opacity_logit))
    
    # Get scales (stored as log)
    scales = np.exp(np.vstack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']]).T)
    
    logger.info(f"  Loaded {len(xyz):,} Gaussians")
    
    return xyz, opacities, scales


def create_occupancy_grid(xyz, opacities, scales, voxel_size=0.05, opacity_threshold=0.5, 
                          padding=1.0, use_scale=True):
    """
    Create binary occupancy grid from Gaussian splat.
    
    Args:
        xyz: Gaussian positions (N, 3)
        opacities: Gaussian opacities (N,)
        scales: Gaussian scales (N, 3)
        voxel_size: Size of each voxel in meters
        opacity_threshold: Minimum opacity to consider as occupied
        padding: Extra space around scene bounds in meters
        use_scale: If True, splat Gaussians based on their scale; if False, just use centers
    
    Returns:
        occupancy: Binary occupancy grid (X, Y, Z)
        origin: World coordinates of grid origin
        grid_info: Dict with grid metadata
    """
    # Filter by opacity
    mask = opacities > opacity_threshold
    xyz_filtered = xyz[mask]
    scales_filtered = scales[mask] if use_scale else None
    opacities_filtered = opacities[mask]
    
    logger.info(f"  Using {len(xyz_filtered):,} Gaussians (opacity > {opacity_threshold})")
    
    if len(xyz_filtered) == 0:
        logger.error("No Gaussians pass opacity threshold!")
        return None, None, None
    
    # Compute bounds
    min_bound = xyz_filtered.min(axis=0) - padding
    max_bound = xyz_filtered.max(axis=0) + padding
    
    # Grid dimensions
    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    logger.info(f"  Grid shape: {grid_shape} ({np.prod(grid_shape):,} voxels)")
    logger.info(f"  Bounds: {min_bound} to {max_bound}")
    logger.info(f"  Voxel size: {voxel_size}m")
    
    # Initialize occupancy grid
    occupancy = np.zeros(grid_shape, dtype=np.float32)
    
    # Splat each Gaussian into the grid
    logger.info(f"  Splatting Gaussians into voxel grid...")
    
    for i in range(len(xyz_filtered)):
        # Position in grid coordinates
        pos = (xyz_filtered[i] - min_bound) / voxel_size
        
        if use_scale and scales_filtered is not None:
            # Use Gaussian scale to determine splat radius
            # Multiply by 2 to get ~95% of Gaussian mass
            radius_world = np.mean(scales_filtered[i]) * 2
            radius_voxels = max(1, int(np.ceil(radius_world / voxel_size)))
        else:
            radius_voxels = 1
        
        # Splat into nearby voxels
        x0, y0, z0 = int(pos[0]), int(pos[1]), int(pos[2])
        
        for dx in range(-radius_voxels, radius_voxels + 1):
            for dy in range(-radius_voxels, radius_voxels + 1):
                for dz in range(-radius_voxels, radius_voxels + 1):
                    xi = x0 + dx
                    yi = y0 + dy
                    zi = z0 + dz
                    
                    if 0 <= xi < grid_shape[0] and 0 <= yi < grid_shape[1] and 0 <= zi < grid_shape[2]:
                        # Gaussian falloff
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if radius_voxels > 0:
                            weight = opacities_filtered[i] * np.exp(-0.5 * (dist / radius_voxels)**2)
                        else:
                            weight = opacities_filtered[i]
                        occupancy[xi, yi, zi] = max(occupancy[xi, yi, zi], weight)
    
    # Binarize
    binary_occupancy = (occupancy > 0.5).astype(np.uint8)
    
    grid_info = {
        'voxel_size': voxel_size,
        'origin': min_bound.tolist(),
        'shape': grid_shape.tolist(),
        'bounds_min': min_bound.tolist(),
        'bounds_max': max_bound.tolist(),
        'num_occupied': int(binary_occupancy.sum()),
        'occupancy_ratio': float(binary_occupancy.sum() / binary_occupancy.size),
    }
    
    logger.info(f"  Occupied voxels: {grid_info['num_occupied']:,} ({grid_info['occupancy_ratio']*100:.1f}%)")
    
    return binary_occupancy, min_bound, grid_info


def create_safety_zones(occupancy, voxel_size, safety_distances=[0.5, 1.0, 2.0]):
    """
    Create dilated safety zones around occupied space.
    Uses distance transform for efficiency instead of morphological dilation.
    
    Args:
        occupancy: Binary occupancy grid
        voxel_size: Size of each voxel in meters
        safety_distances: List of safety distances in meters
    
    Returns:
        Dict mapping distance to dilated occupancy grid
    """
    # Compute distance transform once (distance to nearest occupied voxel)
    # For free space voxels
    dist_to_occupied = ndimage.distance_transform_edt(~occupancy.astype(bool)) * voxel_size
    
    safety_zones = {}
    
    for dist in safety_distances:
        # Voxels within 'dist' meters of occupied space
        zone = (dist_to_occupied <= dist).astype(np.uint8)
        safety_zones[dist] = zone
        
        logger.info(f"  Safety zone {dist}m: {zone.sum():,} voxels "
                   f"({zone.sum() / zone.size * 100:.1f}%)")
    
    return safety_zones


def compute_sdf(occupancy, voxel_size):
    """
    Compute Signed Distance Field from occupancy grid.
    Positive = free space (distance to nearest obstacle)
    Negative = inside obstacle
    
    Args:
        occupancy: Binary occupancy grid
        voxel_size: Size of each voxel in meters
    
    Returns:
        sdf: Signed distance field in meters
    """
    logger.info("  Computing SDF...")
    
    # Distance transform for free space (distance to nearest occupied)
    dist_free = ndimage.distance_transform_edt(~occupancy.astype(bool)) * voxel_size
    
    # Distance transform for occupied space (distance to nearest free)
    dist_occupied = ndimage.distance_transform_edt(occupancy.astype(bool)) * voxel_size
    
    # Combine: positive outside, negative inside
    sdf = dist_free - dist_occupied
    
    logger.info(f"  SDF range: {sdf.min():.2f}m to {sdf.max():.2f}m")
    
    return sdf


def save_occupancy_grid(occupancy, origin, grid_info, output_dir, name="occupancy"):
    """Save occupancy grid in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as NumPy array
    npy_path = output_dir / f"{name}.npy"
    np.save(npy_path, occupancy)
    logger.info(f"  Saved: {npy_path}")
    
    # Save metadata as JSON
    json_path = output_dir / f"{name}_info.json"
    with open(json_path, 'w') as f:
        json.dump(grid_info, f, indent=2)
    logger.info(f"  Saved: {json_path}")
    
    # Save as point cloud (occupied voxels only) for visualization
    occupied_indices = np.argwhere(occupancy > 0)
    if len(occupied_indices) > 0:
        voxel_size = grid_info['voxel_size']
        points = occupied_indices * voxel_size + np.array(origin)
        
        ply_path = output_dir / f"{name}_points.ply"
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        logger.info(f"  Saved: {ply_path}")
    
    return npy_path


def save_sdf(sdf, origin, grid_info, output_dir):
    """Save SDF."""
    output_dir = Path(output_dir)
    
    # Save as NumPy array
    npy_path = output_dir / "sdf.npy"
    np.save(npy_path, sdf.astype(np.float32))
    logger.info(f"  Saved: {npy_path}")
    
    # Save metadata
    sdf_info = grid_info.copy()
    sdf_info['sdf_min'] = float(sdf.min())
    sdf_info['sdf_max'] = float(sdf.max())
    
    json_path = output_dir / "sdf_info.json"
    with open(json_path, 'w') as f:
        json.dump(sdf_info, f, indent=2)
    logger.info(f"  Saved: {json_path}")


def load_tum_poses(tum_dataset_path, max_frames=None, sample_rate=1):
    """Load camera poses from TUM RGB-D dataset.
    
    Args:
        tum_dataset_path: Path to TUM dataset folder
        max_frames: Maximum number of frames to load (None = all)
        sample_rate: Sample every N frames (1 = all frames)
    """
    from scipy.spatial.transform import Rotation
    
    tum_path = Path(tum_dataset_path)
    groundtruth_path = tum_path / "groundtruth.txt"
    
    if not groundtruth_path.exists():
        logger.warning(f"No groundtruth.txt found at {groundtruth_path}")
        return None
    
    poses = []
    line_count = 0
    with open(groundtruth_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            line_count += 1
            
            # Sample frames
            if sample_rate > 1 and line_count % sample_rate != 0:
                continue
            
            # timestamp tx ty tz qx qy qz qw
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # Convert quaternion to rotation matrix
            rot = Rotation.from_quat([qx, qy, qz, qw])
            R = rot.as_matrix()
            
            # Build 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = [tx, ty, tz]
            
            poses.append(pose)
            
            if max_frames is not None and len(poses) >= max_frames:
                break
    
    logger.info(f"  Loaded {len(poses)} camera poses from TUM dataset")
    return poses


def process_scene(scene_dir, voxel_size=0.05, opacity_threshold=0.5, 
                  safety_distances=[0.5, 1.0, 2.0], compute_sdf_flag=True,
                  tum_dataset_root=None):
    """Process a single scene."""
    
    ply_path = scene_dir / "final" / "point_cloud.ply"
    output_dir = scene_dir / "voxel_grid"
    
    if not ply_path.exists():
        logger.warning(f"  No point_cloud.ply found")
        return False
    
    # Load Gaussian splat
    xyz, opacities, scales = load_gaussian_splat(ply_path)
    
    # Create occupancy grid
    occupancy, origin, grid_info = create_occupancy_grid(
        xyz, opacities, scales,
        voxel_size=voxel_size,
        opacity_threshold=opacity_threshold,
        use_scale=True
    )
    
    if occupancy is None:
        return False
    
    # Save base occupancy
    save_occupancy_grid(occupancy, origin, grid_info, output_dir, "occupancy")
    
    # Create and save safety zones
    if safety_distances:
        logger.info("Creating safety zones...")
        safety_zones = create_safety_zones(occupancy, voxel_size, safety_distances)
        
        for dist, zone in safety_zones.items():
            zone_info = grid_info.copy()
            zone_info['safety_distance'] = dist
            zone_info['num_occupied'] = int(zone.sum())
            zone_info['occupancy_ratio'] = float(zone.sum() / zone.size)
            save_occupancy_grid(zone, origin, zone_info, output_dir, f"safety_{dist}m")
    else:
        safety_zones = {}
    
    # Compute and save SDF
    if compute_sdf_flag:
        logger.info("Computing Signed Distance Field...")
        sdf = compute_sdf(occupancy, voxel_size)
        save_sdf(sdf, origin, grid_info, output_dir)
    
    # Save a combined visualization
    logger.info("Generating 2D visualization...")
    generate_visualization(occupancy, safety_zones if safety_distances else {}, 
                          origin, grid_info, output_dir)
    
    # Generate 3D flythrough video
    logger.info("Generating 3D voxel flythrough video...")
    
    # Load config for intrinsics
    config_path = scene_dir / "final" / "config.json"
    if config_path.exists():
        import json as json_module
        with open(config_path) as f:
            config = json_module.load(f)
        intrinsics = config.get('intrinsics', {
            'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3,
            'width': 640, 'height': 480
        })
    else:
        intrinsics = {'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3, 'width': 640, 'height': 480}
    
    # Use smallest safety zone for video
    safety_for_video = None
    if safety_zones:
        smallest_dist = min(safety_zones.keys())
        safety_for_video = safety_zones[smallest_dist]
    
    # Generate orbit flythrough (better overview of voxel structure)
    video_path = scene_dir / "4_voxel_flythrough.mp4"
    generate_voxel_flythrough(
        occupancy, safety_for_video, np.array(origin), grid_info,
        video_path, intrinsics, camera_poses=None, n_frames=120, orbit_radius=None
    )
    
    return True


def generate_visualization(occupancy, safety_zones, origin, grid_info, output_dir):
    """Generate a 2D slice visualization of the occupancy grid."""
    import cv2
    
    # Take middle Z slice
    z_mid = occupancy.shape[2] // 2
    
    # Create RGB image
    h, w = occupancy.shape[0], occupancy.shape[1]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw safety zones (largest to smallest)
    colors = [(50, 50, 100), (50, 100, 50), (100, 50, 50)]  # BGR
    for i, (dist, zone) in enumerate(sorted(safety_zones.items(), reverse=True)):
        color = colors[i % len(colors)]
        img[zone[:, :, z_mid] > 0] = color
    
    # Draw occupied space
    img[occupancy[:, :, z_mid] > 0] = (255, 255, 255)
    
    # Add legend and info
    img = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    
    cv2.putText(img, f"Voxel Grid (Z={z_mid})", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Voxel size: {grid_info['voxel_size']}m", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Shape: {grid_info['shape']}", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Legend
    y_offset = 120
    cv2.putText(img, "Legend:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(img, (10, y_offset + 10), (30, y_offset + 30), (255, 255, 255), -1)
    cv2.putText(img, "Occupied", (40, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    for i, dist in enumerate(sorted(safety_zones.keys())):
        color = colors[i % len(colors)]
        y = y_offset + 40 + i * 25
        cv2.rectangle(img, (10, y), (30, y + 20), color, -1)
        cv2.putText(img, f"Safety {dist}m", (40, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    viz_path = output_dir / "occupancy_slice.png"
    cv2.imwrite(str(viz_path), img)
    logger.info(f"  Saved: {viz_path}")


def render_voxel_view(occupancy, safety_zone, origin, voxel_size, pose, intrinsics, img_size):
    """Render ONLY occupied voxels as dots from a camera viewpoint."""
    import cv2
    width, height = img_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get ONLY occupied voxel indices (obstacles to avoid)
    occupied_idx = np.argwhere(occupancy > 0)
    
    if len(occupied_idx) == 0:
        return img
    
    # Camera transform
    pose_inv = np.linalg.inv(pose)
    R = pose_inv[:3, :3]
    t = pose_inv[:3, 3]
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Calculate world positions of voxel centers
    world_positions = origin + (occupied_idx + 0.5) * voxel_size
    
    # Transform to camera space
    cam_positions = (R @ world_positions.T).T + t
    
    # Filter points in front of camera
    valid = cam_positions[:, 2] > 0.1
    cam_positions = cam_positions[valid]
    
    if len(cam_positions) == 0:
        return img
    
    # Project to image coordinates
    depths = cam_positions[:, 2]
    u = (fx * cam_positions[:, 0] / depths + cx).astype(int)
    v = (fy * cam_positions[:, 1] / depths + cy).astype(int)
    
    # Filter to image bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    depths = depths[in_bounds]
    
    if len(u) == 0:
        return img
    
    # Sort by depth (far to near)
    sort_idx = np.argsort(-depths)
    u = u[sort_idx]
    v = v[sort_idx]
    depths = depths[sort_idx]
    
    # Color by depth (red=close, blue=far)
    depth_min, depth_max = depths.min(), depths.max()
    if depth_max > depth_min:
        depth_norm = (depths - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depths)
    
    # Point size based on depth - BIGGER points
    point_sizes = np.clip((8.0 / (depths + 0.5) * 3).astype(int), 2, 15)
    
    # Draw points
    for i in range(len(u)):
        t_val = depth_norm[i]
        # BGR: close=red, far=blue
        color = (
            int(255 * t_val),           # B
            int(100 * (1 - t_val)),     # G
            int(255 * (1 - t_val))      # R
        )
        size = point_sizes[i]
        cv2.circle(img, (u[i], v[i]), size, color, -1)
    
    return img


def generate_voxel_flythrough(occupancy, safety_zone, origin, grid_info, output_path, 
                              intrinsics, camera_poses=None, n_frames=120, orbit_radius=None):
    """Generate a flythrough video of the voxel grid using orbit around scene."""
    import cv2
    
    voxel_size = grid_info['voxel_size']
    shape = np.array(grid_info['shape'])
    
    # Scene center and size
    scene_center = origin + (shape * voxel_size) / 2
    scene_size = shape * voxel_size
    scene_diagonal = np.linalg.norm(scene_size)
    
    # Auto-calculate orbit radius - CLOSER to scene (0.4x diagonal, min 1.5m)
    if orbit_radius is None:
        orbit_radius = max(scene_diagonal * 0.4, 1.5)  # Much closer
    
    logger.info(f"  Scene size: {scene_size[0]:.1f} x {scene_size[1]:.1f} x {scene_size[2]:.1f} m")
    logger.info(f"  Orbit radius: {orbit_radius:.1f}m")
    
    # Always use orbit for voxel visualization (better overview)
    poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        # Position on circle around scene
        cam_x = scene_center[0] + orbit_radius * np.cos(angle)
        cam_y = scene_center[1] + orbit_radius * np.sin(angle)
        cam_z = scene_center[2] + scene_size[2] * 0.3  # Slightly above center
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # Look-at matrix construction (world-to-camera directly)
        # Forward points FROM camera TO scene (what camera sees)
        forward = scene_center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Right and up vectors
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # World-to-camera rotation: rows are camera axes in world coords
        R_w2c = np.array([right, up, forward])  # forward is +Z in camera
        t_w2c = -R_w2c @ cam_pos
        
        # Store as camera-to-world for consistency with other code
        # But we'll use pose_inv in rendering anyway
        pose = np.eye(4)
        pose[:3, :3] = R_w2c.T  # camera-to-world rotation
        pose[:3, 3] = cam_pos
        poses.append(pose)
    
    width, height = intrinsics['width'], intrinsics['height']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (width, height + 80))
    
    occupied_count = int(occupancy.sum())
    
    logger.info(f"  Rendering {n_frames} frames ({occupied_count:,} obstacle voxels)...")
    
    for i, pose in enumerate(poses):
        # Only render occupied voxels (obstacles)
        voxel_img = render_voxel_view(
            occupancy, None, origin, voxel_size,
            pose, intrinsics, (width, height)
        )
        
        # Info panel
        panel = np.zeros((80, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        cv2.putText(panel, f"OBSTACLE VOXELS - {occupied_count:,} voxels to avoid", 
                   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(panel, f"Voxel size: {voxel_size}m | Grid: {shape[0]}x{shape[1]}x{shape[2]}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(panel, f"Frame {i+1}/{n_frames} | 360 Orbit View | Radius: {orbit_radius:.1f}m", 
                   (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Legend (color = depth)
        cv2.putText(panel, "Depth:", (width - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.rectangle(panel, (width - 130, 12), (width - 115, 27), (0, 100, 255), -1)
        cv2.putText(panel, "Close", (width - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.rectangle(panel, (width - 130, 32), (width - 115, 47), (255, 50, 50), -1)
        cv2.putText(panel, "Far", (width - 110, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        frame = np.vstack([voxel_img, panel])
        out.write(frame)
        
        if (i + 1) % 30 == 0:
            logger.info(f"    Frame {i+1}/{n_frames}")
    
    out.release()
    logger.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate voxel occupancy grids for autonomous flight")
    parser.add_argument("--output-root", type=str, default="./output", help="Output directory root")
    parser.add_argument("--dataset-root", type=str, default="/home/past/parsa/datasets/tum",
                       help="TUM dataset root for camera poses")
    parser.add_argument("--scenes", nargs="*", help="Specific scenes to process")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size in meters")
    parser.add_argument("--opacity-threshold", type=float, default=0.3, help="Opacity threshold")
    parser.add_argument("--safety-distances", type=float, nargs="*", default=[0.5, 1.0, 2.0],
                       help="Safety zone distances in meters")
    parser.add_argument("--no-sdf", action="store_true", help="Skip SDF computation")
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    
    # Find scenes
    if args.scenes:
        scene_dirs = [output_root / s for s in args.scenes]
    else:
        scene_dirs = sorted([d for d in output_root.iterdir() 
                            if d.is_dir() and (d / "final" / "point_cloud.ply").exists()])
    
    logger.info(f"Processing {len(scene_dirs)} scene(s)")
    logger.info(f"Voxel size: {args.voxel_size}m")
    logger.info(f"Safety distances: {args.safety_distances}m")
    logger.info(f"TUM dataset root: {args.dataset_root}")
    
    results = {}
    for scene_dir in scene_dirs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {scene_dir.name}")
        logger.info(f"{'='*60}")
        
        try:
            success = process_scene(
                scene_dir,
                voxel_size=args.voxel_size,
                opacity_threshold=args.opacity_threshold,
                safety_distances=args.safety_distances,
                compute_sdf_flag=not args.no_sdf,
                tum_dataset_root=args.dataset_root
            )
            results[scene_dir.name] = success
        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            results[scene_dir.name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")
    
    logger.info(f"\nOutput files per scene:")
    logger.info(f"  voxel_grid/occupancy.npy - Base occupancy grid")
    logger.info(f"  voxel_grid/occupancy_info.json - Grid metadata")
    logger.info(f"  voxel_grid/occupancy_points.ply - Point cloud for visualization")
    logger.info(f"  voxel_grid/safety_Xm.npy - Dilated safety zones")
    logger.info(f"  voxel_grid/sdf.npy - Signed Distance Field")
    logger.info(f"  voxel_grid/occupancy_slice.png - 2D visualization")


if __name__ == "__main__":
    main()
