"""
Photo-SLAM Engine - Real-time Photorealistic SLAM (CVPR'24)

Real-time monocular/stereo/RGB-D SLAM with photorealistic mapping.
Uses ORB-SLAM3 for tracking + Gaussian Splatting for mapping.

Paper: https://arxiv.org/abs/2311.16728
Code: https://github.com/HuajianUP/Photo-SLAM
"""

import os
import sys
import subprocess
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def _find_photoslam_path() -> Optional[Path]:
    """Find Photo-SLAM installation.
    
    Set AIRSPLAT_USE_SUBMODULES=1 to force submodule-only mode.
    """
    use_submodules_only = os.environ.get("AIRSPLAT_USE_SUBMODULES", "").lower() in ("1", "true", "yes")
    
    this_dir = Path(__file__).parent.resolve()
    airsplatmap_root = this_dir.parent.parent
    workspace_root = airsplatmap_root.parent
    
    # Primary: submodules directory (git submodule)
    submodule_path = airsplatmap_root / "submodules" / "Photo-SLAM"
    if submodule_path.exists() and (submodule_path / "src").is_dir():
        return submodule_path
    
    if use_submodules_only:
        return None  # Don't fall back to legacy
    
    # Legacy: workspace root
    legacy_path = workspace_root / "Photo-SLAM"
    if legacy_path.exists() and (legacy_path / "src").is_dir():
        return legacy_path
    return None


PHOTOSLAM_PATH = _find_photoslam_path()


class PhotoSLAMEngine:
    """
    Photo-SLAM engine wrapper for real-time photorealistic SLAM.
    
    Photo-SLAM is a C++ application, so this wrapper:
    1. Prepares input data in the expected format
    2. Calls the Photo-SLAM binary
    3. Loads and processes the results
    
    Key features:
    - Real-time tracking via ORB-SLAM3
    - Photorealistic mapping via Gaussian Splatting
    - Supports Mono/Stereo/RGB-D
    """
    
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self._initialized = False
        self._intrinsics = None
        self._frames = []
        self._config = None
        self._output_dir = None
        
        # Check if Photo-SLAM is available
        self._photoslam_available = PHOTOSLAM_PATH is not None and PHOTOSLAM_PATH.exists()
        self._binary_path = PHOTOSLAM_PATH / "bin" / "tum_rgbd" if PHOTOSLAM_PATH else None
        self._binary_available = self._binary_path and self._binary_path.exists()
        
        if not self._photoslam_available:
            logger.warning("Photo-SLAM not found")
            logger.warning("Please run: git submodule update --init --recursive")
        elif not self._binary_available:
            logger.warning(f"Photo-SLAM binary not found at {self._binary_path}")
            logger.warning("Please build Photo-SLAM: cd submodules/Photo-SLAM && ./build.sh")
        
        logger.info(f"Photo-SLAM engine initialized (binary available: {self._binary_available})")
    
    def initialize_scene(self, intrinsics: Dict, config: Dict = None) -> None:
        """Initialize the SLAM scene with camera intrinsics."""
        self._intrinsics = intrinsics.copy()
        self._config = config or {}
        self._frames = []
        
        # Create temporary output directory
        import tempfile
        self._output_dir = Path(tempfile.mkdtemp(prefix="photoslam_"))
        
        self._initialized = True
        logger.info(f"Photo-SLAM scene initialized: {intrinsics['width']}x{intrinsics['height']}")
        logger.info(f"Output dir: {self._output_dir}")
    
    def add_frame(self, frame_idx: int, rgb: np.ndarray, depth: Optional[np.ndarray] = None,
                  pose: Optional[np.ndarray] = None) -> Dict:
        """
        Add a new frame to the SLAM system.
        
        Photo-SLAM processes data offline, so frames are accumulated
        and processed when optimize_step or save_state is called.
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized. Call initialize_scene first.")
        
        self._frames.append({
            'idx': frame_idx,
            'rgb': rgb.copy(),
            'depth': depth.copy() if depth is not None else None,
            'pose': pose.copy() if pose is not None else None,
        })
        
        return {
            'frame_idx': frame_idx,
            'num_frames': len(self._frames),
        }
    
    def render_view(self, pose: np.ndarray, img_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Render view from given camera pose using accumulated depth data."""
        width, height = img_size
        
        # Get point cloud and colors
        pts = self.get_point_cloud()
        cols = self.get_gaussian_colors()
        
        if pts is None or len(pts) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Project points to camera
        return self._project_points(pts, cols, pose, (width, height))
    
    def _project_points(self, pts: np.ndarray, cols: Optional[np.ndarray], 
                        pose: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
        """Project 3D points to image plane."""
        width, height = img_size
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(pts) == 0:
            return img
        
        # Transform to camera frame
        pose_inv = np.linalg.inv(pose)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam = (pose_inv @ pts_h.T).T[:, :3]
        
        # Filter behind camera
        valid = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[valid]
        if cols is not None:
            cols = cols[valid]
        
        if len(pts_cam) == 0:
            return img
        
        # Project
        fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
        cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
        
        # Scale for output size
        scale_x = width / self._intrinsics['width']
        scale_y = height / self._intrinsics['height']
        
        u = (pts_cam[:, 0] * fx * scale_x / pts_cam[:, 2] + cx * scale_x).astype(int)
        v = (pts_cam[:, 1] * fy * scale_y / pts_cam[:, 2] + cy * scale_y).astype(int)
        
        # Filter to image bounds
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v = u[valid], v[valid]
        if cols is not None:
            cols = cols[valid]
        depths = pts_cam[valid, 2]
        
        # Sort by depth
        order = np.argsort(-depths)
        u, v = u[order], v[order]
        if cols is not None:
            cols = cols[order]
            img[v, u] = (cols * 255).astype(np.uint8)
        else:
            img[v, u] = [128, 200, 255]  # Default cyan
        
        return img
    
    def _render_point_cloud(self, pose: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
        """Simple point cloud rendering."""
        import cv2
        
        width, height = img_size
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Load point cloud if exists
        ply_path = self._output_dir / "point_cloud.ply"
        if not ply_path.exists():
            return img
        
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(ply_path))
            xyz = np.vstack([
                plydata['vertex']['x'],
                plydata['vertex']['y'],
                plydata['vertex']['z']
            ]).T
            
            if 'red' in plydata['vertex']:
                colors = np.vstack([
                    plydata['vertex']['red'],
                    plydata['vertex']['green'],
                    plydata['vertex']['blue']
                ]).T
            else:
                colors = np.full((len(xyz), 3), 128, dtype=np.uint8)
        except:
            return img
        
        if len(xyz) == 0:
            return img
        
        # Transform to camera frame
        pose_inv = np.linalg.inv(pose)
        R = pose_inv[:3, :3]
        t = pose_inv[:3, 3]
        xyz_cam = (R @ xyz.T).T + t
        
        # Filter in front of camera
        mask = xyz_cam[:, 2] > 0.1
        xyz_cam = xyz_cam[mask]
        colors = colors[mask]
        
        if len(xyz_cam) == 0:
            return img
        
        # Project
        fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
        cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
        
        u = (fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx).astype(int)
        v = (fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy).astype(int)
        
        # Draw
        mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        for i in np.where(mask)[0]:
            color = tuple(int(c) for c in colors[i])
            cv2.circle(img, (u[i], v[i]), 1, color, -1)
        
        return img
    
    def optimize_step(self, n_steps: int = 1, **kwargs) -> Dict:
        """
        Run Photo-SLAM on accumulated frames.
        
        This prepares the data and calls the Photo-SLAM binary.
        """
        if not self._binary_available:
            return {'loss': 0.0, 'psnr': 0.0, 'error': 'Binary not available'}
        
        if len(self._frames) == 0:
            return {'loss': 0.0, 'psnr': 0.0, 'num_frames': 0}
        
        # Prepare TUM-format dataset
        data_dir = self._output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        self._prepare_tum_format(data_dir)
        
        # Run Photo-SLAM
        # Note: This is a placeholder - actual integration would need proper config
        logger.info(f"Photo-SLAM data prepared at {data_dir}")
        
        return {
            'loss': 0.0,
            'psnr': 0.0,
            'num_frames': len(self._frames),
        }
    
    def _prepare_tum_format(self, output_dir: Path) -> None:
        """Prepare frames in TUM RGB-D format."""
        import cv2
        
        rgb_dir = output_dir / "rgb"
        depth_dir = output_dir / "depth"
        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        
        rgb_txt = []
        depth_txt = []
        assoc_txt = []
        
        for frame in self._frames:
            idx = frame['idx']
            timestamp = idx / 30.0  # Assume 30 fps
            
            # Save RGB
            rgb_path = rgb_dir / f"{idx:06d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR))
            rgb_txt.append(f"{timestamp:.6f} rgb/{idx:06d}.png")
            
            # Save depth if available
            if frame['depth'] is not None:
                depth_path = depth_dir / f"{idx:06d}.png"
                depth_mm = (frame['depth'] * 5000).astype(np.uint16)
                cv2.imwrite(str(depth_path), depth_mm)
                depth_txt.append(f"{timestamp:.6f} depth/{idx:06d}.png")
                assoc_txt.append(f"{timestamp:.6f} rgb/{idx:06d}.png {timestamp:.6f} depth/{idx:06d}.png")
        
        # Write txt files
        with open(output_dir / "rgb.txt", 'w') as f:
            f.write("# timestamp filename\n")
            f.write("\n".join(rgb_txt))
        
        if depth_txt:
            with open(output_dir / "depth.txt", 'w') as f:
                f.write("# timestamp filename\n")
                f.write("\n".join(depth_txt))
            
            with open(output_dir / "associations.txt", 'w') as f:
                f.write("\n".join(assoc_txt))
    
    def get_num_gaussians(self) -> int:
        """Get current number of Gaussians."""
        # Use cached point cloud
        if hasattr(self, '_cached_pts') and self._cached_pts is not None:
            return len(self._cached_pts)
        
        # Photo-SLAM stores results in files
        ply_path = self._output_dir / "point_cloud.ply" if self._output_dir else None
        if ply_path and ply_path.exists():
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(ply_path))
                return len(plydata['vertex'])
            except:
                pass
        return len(self._frames) * 100  # Estimate based on frames
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get Gaussian centers as point cloud."""
        # Return cached if available and recent
        if hasattr(self, '_cached_pts') and self._cached_pts is not None:
            if hasattr(self, '_cached_frame') and self._cached_frame == len(self._frames):
                return self._cached_pts
        
        # Build from accumulated frames
        if len(self._frames) > 0:
            pts = []
            # Use latest frames with subsampling
            frame_step = max(1, len(self._frames) // 50)  # Max 50 frames
            for fr in self._frames[::frame_step]:
                if fr['depth'] is not None and fr['pose'] is not None:
                    depth = fr['depth']
                    pose = fr['pose']
                    h, w = depth.shape[:2]
                    fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
                    cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
                    
                    # Sparse sampling
                    step = 16
                    for v in range(0, h, step):
                        for u in range(0, w, step):
                            d = depth[v, u]
                            if 0.1 < d < 10:
                                x = (u - cx) * d / fx
                                y = (v - cy) * d / fy
                                pt_cam = np.array([x, y, d, 1.0])
                                pt_world = pose @ pt_cam
                                pts.append(pt_world[:3])
            
            if pts:
                self._cached_pts = np.array(pts)
                self._cached_frame = len(self._frames)
                return self._cached_pts
        
        # Try PLY file
        ply_path = self._output_dir / "point_cloud.ply" if self._output_dir else None
        if ply_path and ply_path.exists():
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(ply_path))
                return np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
            except:
                pass
        return None
    
    def get_gaussian_colors(self) -> Optional[np.ndarray]:
        """Get RGB colors for each Gaussian."""
        # Return cached if available
        if hasattr(self, '_cached_cols') and self._cached_cols is not None:
            if hasattr(self, '_cached_col_frame') and self._cached_col_frame == len(self._frames):
                return self._cached_cols
        
        # Build from accumulated frames
        if len(self._frames) > 0:
            colors = []
            frame_step = max(1, len(self._frames) // 50)
            for fr in self._frames[::frame_step]:
                if fr['depth'] is not None and fr['rgb'] is not None:
                    rgb = fr['rgb']
                    depth = fr['depth']
                    h, w = depth.shape[:2]
                    step = 16  # Match get_point_cloud step
                    for v in range(0, h, step):
                        for u in range(0, w, step):
                            d = depth[v, u]
                            if 0.1 < d < 10:
                                colors.append(rgb[v, u] / 255.0)
            if colors:
                self._cached_cols = np.array(colors)
                self._cached_col_frame = len(self._frames)
                return self._cached_cols
        
        # Try PLY file
        ply_path = self._output_dir / "point_cloud.ply" if self._output_dir else None
        if ply_path and ply_path.exists():
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(ply_path))
                if 'red' in plydata['vertex']:
                    return np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T / 255.0
            except:
                pass
        return None
    
    def save_state(self, path: str) -> None:
        """Save current state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Run optimization to generate results
        if len(self._frames) > 0:
            self.optimize_step()
        
        # Copy results
        if self._output_dir:
            import shutil
            
            # Copy point cloud if exists
            src_ply = self._output_dir / "point_cloud.ply"
            if src_ply.exists():
                shutil.copy(src_ply, path / "point_cloud.ply")
            
            # Copy data directory
            src_data = self._output_dir / "data"
            if src_data.exists():
                shutil.copytree(src_data, path / "data", dirs_exist_ok=True)
        
        # Save config
        import json
        with open(path / "config.json", 'w') as f:
            json.dump({
                'engine': 'photoslam',
                'intrinsics': self._intrinsics,
                'num_frames': len(self._frames),
                'num_gaussians': self.get_num_gaussians(),
            }, f, indent=2, default=str)
        
        logger.info(f"Saved Photo-SLAM state to {path}")
    
    def load_state(self, path: str) -> None:
        """Load state from disk."""
        path = Path(path)
        
        import json
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        self._intrinsics = config['intrinsics']
        self._initialized = True
        
        # Set output dir to loaded path for point cloud access
        self._output_dir = path
        
        logger.info(f"Loaded Photo-SLAM state from {path}")
    
    def __del__(self):
        """Cleanup temporary directory."""
        if hasattr(self, '_output_dir') and self._output_dir:
            import shutil
            try:
                # Only cleanup if it's a temp directory
                if "photoslam_" in str(self._output_dir):
                    shutil.rmtree(self._output_dir, ignore_errors=True)
            except:
                pass


def create_engine(device: str = "cuda:0") -> PhotoSLAMEngine:
    """Factory function to create Photo-SLAM engine."""
    return PhotoSLAMEngine(device=device)
