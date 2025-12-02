"""
Gaussian-SLAM Engine - Dense RGB-D SLAM with Gaussian Splatting
================================================================

This engine wraps Gaussian-SLAM (https://github.com/VladimirYugay/Gaussian-SLAM)
which provides online RGB-D SLAM using Gaussian Splatting with submaps.

Key features:
- Submap-based scene representation for large environments
- Joint tracking and mapping
- Works with TUM, Replica, ScanNet datasets

To use this engine, ensure the submodule is initialized:
    git submodule update --init --recursive
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json
import logging
import sys
import os
import tempfile
import shutil
from dataclasses import dataclass, asdict

from .base import BaseGSEngine

logger = logging.getLogger(__name__)


def _find_gslam_path() -> Optional[Path]:
    """Find Gaussian-SLAM installation.
    
    Set AIRSPLAT_USE_SUBMODULES=1 to force submodule-only mode.
    """
    use_submodules_only = os.environ.get("AIRSPLAT_USE_SUBMODULES", "").lower() in ("1", "true", "yes")
    
    this_dir = Path(__file__).parent.resolve()
    airsplatmap_root = this_dir.parent.parent
    workspace_root = airsplatmap_root.parent
    
    # Primary: submodules directory (git submodule)
    submodule_path = airsplatmap_root / "submodules" / "Gaussian-SLAM"
    if submodule_path.exists() and (submodule_path / "src" / "entities" / "gaussian_slam.py").exists():
        return submodule_path
    
    if use_submodules_only:
        return None  # Don't fall back to legacy
    
    # Legacy paths
    legacy_candidates = [
        workspace_root / "Gaussian-SLAM",
        Path.home() / "parsa" / "Gaussian-SLAM",
        Path.home() / "Gaussian-SLAM",
    ]
    
    for path in legacy_candidates:
        if path.exists() and (path / "src" / "entities" / "gaussian_slam.py").exists():
            return path
    return None


GSLAM_PATH = _find_gslam_path()


@dataclass
class GSLAMConfig:
    """Configuration for Gaussian-SLAM engine."""
    # Tracking parameters
    track_iterations: int = 40
    track_w_color_loss: float = 0.5
    track_alpha_thre: float = 0.99
    track_filter_alpha: bool = True
    track_filter_outlier: bool = True
    track_cam_trans_lr: float = 0.002
    
    # Mapping parameters
    map_every: int = 5
    map_iterations: int = 60
    new_submap_every: int = 50
    alpha_seeding_thre: float = 0.5
    
    # Scene parameters
    seed: int = 0
    use_wandb: bool = False
    
    # Dataset type
    dataset_name: str = "tum"


class GSLAMEngine(BaseGSEngine):
    """
    3D Gaussian Splatting SLAM engine using Gaussian-SLAM.
    
    This engine provides a wrapper around the Gaussian-SLAM implementation,
    which uses submaps for scalable scene representation.
    
    Note: Gaussian-SLAM is designed to run on complete sequences from a
    config file. This wrapper provides a frame-by-frame interface by
    accumulating frames and running the SLAM pipeline.
    """
    
    def __init__(
        self,
        device: str = "cuda:0",
        gslam_path: Optional[str] = None,
        config: Optional[GSLAMConfig] = None
    ):
        """
        Initialize the Gaussian-SLAM engine.
        
        Args:
            device: CUDA device to use
            gslam_path: Path to Gaussian-SLAM installation
            config: Gaussian-SLAM configuration
        """
        self.device = torch.device(device)
        self._initialized = False
        self._intrinsics = None
        
        if gslam_path:
            self._gslam_path = Path(gslam_path)
        else:
            self._gslam_path = GSLAM_PATH
        
        self.config = config or GSLAMConfig()
        
        # Frame buffers
        self._frames: List[Dict] = []
        self._rgb_images: List[np.ndarray] = []
        self._depth_images: List[np.ndarray] = []
        self._poses: List[np.ndarray] = []
        self._timestamps: List[float] = []
        
        # Output state
        self._output_path: Optional[Path] = None
        self._gaussians = None
        self._estimated_poses = None
        
        # Iteration counter
        self._iteration = 0
        self._frame_idx = 0
        
        # Image dimensions
        self._height = None
        self._width = None
        
        # Check availability
        self._available = self._check_available()
        
        if self._available:
            logger.info(f"Gaussian-SLAM engine initialized from {self._gslam_path}")
        else:
            logger.warning(
                "Gaussian-SLAM not available. Install from:\n"
                "  git clone https://github.com/VladimirYugay/Gaussian-SLAM.git\n"
                "  cd Gaussian-SLAM && pip install -e ."
            )
    
    def _check_available(self) -> bool:
        """Check if Gaussian-SLAM is available."""
        if self._gslam_path is None or not self._gslam_path.exists():
            return False
        
        try:
            sys.path.insert(0, str(self._gslam_path))
            from src.entities.gaussian_slam import GaussianSLAM
            from src.entities.gaussian_model import GaussianModel
            return True
        except ImportError as e:
            logger.debug(f"Gaussian-SLAM import failed: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def initialize_scene(
        self,
        intrinsics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """Initialize the Gaussian-SLAM scene."""
        if self._initialized:
            raise RuntimeError("Scene already initialized. Call reset() first.")
        
        self._intrinsics = intrinsics.copy()
        self._width = int(intrinsics.get('width', 640))
        self._height = int(intrinsics.get('height', 480))
        
        # Update config from dict
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Create temporary output directory
        self._output_path = Path(tempfile.mkdtemp(prefix="gslam_"))
        
        self._initialized = True
        logger.info(f"Gaussian-SLAM scene initialized: {self._width}x{self._height}")
    
    def add_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose_world_cam: np.ndarray
    ) -> None:
        """
        Add a new frame for tracking and mapping.
        
        Args:
            frame_id: Frame index
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            pose_world_cam: 4x4 camera-to-world transform
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        # Ensure RGB is uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # Store frame data
        self._frames.append({
            'id': frame_id,
            'timestamp': frame_id * 0.033,  # Assume 30 FPS
        })
        self._rgb_images.append(rgb.copy())
        self._depth_images.append(depth.copy() if depth is not None else None)
        self._poses.append(pose_world_cam.copy())
        self._timestamps.append(frame_id * 0.033)
        
        self._frame_idx += 1
        
        logger.debug(f"Added frame {frame_id}, total frames: {len(self._frames)}")
        
        # Build point cloud incrementally if GSLAM not available
        if not self._available and depth is not None:
            self._accumulate_pointcloud(rgb, depth, pose_world_cam)
    
    def _accumulate_pointcloud(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> None:
        """Accumulate point cloud from depth frames when GSLAM not available."""
        import torch
        
        # Subsample for speed
        step = 8
        h, w = depth.shape[:2]
        
        fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
        cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
        
        pts, cols = [], []
        for v in range(0, h, step):
            for u in range(0, w, step):
                d = depth[v, u]
                if 0.1 < d < 10:
                    x = (u - cx) * d / fx
                    y = (v - cy) * d / fy
                    pt_cam = np.array([x, y, d, 1.0])
                    pt_world = pose @ pt_cam
                    pts.append(pt_world[:3])
                    cols.append(rgb[v, u] / 255.0)
        
        if pts:
            pts = np.array(pts)
            cols = np.array(cols)
            
            if self._gaussians is None:
                self._gaussians = {
                    'xyz': torch.from_numpy(pts).float().to(self.device),
                    'colors': torch.from_numpy(cols).float().to(self.device),
                }
            else:
                # Append and subsample to keep size reasonable
                xyz = torch.cat([self._gaussians['xyz'], torch.from_numpy(pts).float().to(self.device)])
                colors = torch.cat([self._gaussians['colors'], torch.from_numpy(cols).float().to(self.device)])
                
                # Keep max 100k points, random subsample
                max_pts = 100000
                if len(xyz) > max_pts:
                    idx = torch.randperm(len(xyz))[:max_pts]
                    xyz = xyz[idx]
                    colors = colors[idx]
                
                self._gaussians['xyz'] = xyz
                self._gaussians['colors'] = colors
    
    def optimize_step(self, n_steps: int = 1) -> Dict[str, float]:
        """
        Run optimization steps.
        
        Note: Gaussian-SLAM runs as a complete pipeline, so this triggers
        the SLAM process on accumulated frames if enough have been collected.
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        # Run SLAM if we have enough frames and haven't run yet
        if len(self._frames) >= 10 and self._gaussians is None and self._available:
            self._run_slam()
        
        self._iteration += n_steps
        
        return {
            'loss': 0.0,
            'psnr': 0.0,
            'num_gaussians': self.get_num_gaussians(),
            'iteration': self._iteration,
            'num_frames': len(self._frames),
        }
    
    def _run_slam(self) -> None:
        """Run Gaussian-SLAM on accumulated frames."""
        if not self._available:
            return
        
        try:
            # Create dataset structure for Gaussian-SLAM
            self._create_dataset_structure()
            
            # Create config
            config = self._create_gslam_config()
            
            # Import and run
            sys.path.insert(0, str(self._gslam_path))
            from src.entities.gaussian_slam import GaussianSLAM
            from src.utils.utils import setup_seed
            
            setup_seed(config["seed"])
            
            # Run SLAM
            gslam = GaussianSLAM(config)
            gslam.run()
            
            # Load results
            self._load_results(gslam.output_path)
            
            logger.info(f"Gaussian-SLAM completed, {self.get_num_gaussians()} Gaussians")
            
        except Exception as e:
            logger.error(f"Gaussian-SLAM failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_dataset_structure(self) -> None:
        """Create TUM-style dataset structure for Gaussian-SLAM."""
        if self._output_path is None:
            return
        
        data_path = self._output_path / "data"
        rgb_path = data_path / "rgb"
        depth_path = data_path / "depth"
        
        rgb_path.mkdir(parents=True, exist_ok=True)
        depth_path.mkdir(parents=True, exist_ok=True)
        
        # Save images
        import cv2
        
        rgb_list = []
        depth_list = []
        gt_list = []
        
        for i, (rgb, depth, pose, ts) in enumerate(zip(
            self._rgb_images, self._depth_images, self._poses, self._timestamps
        )):
            # Save RGB
            rgb_file = f"{ts:.6f}.png"
            cv2.imwrite(str(rgb_path / rgb_file), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rgb_list.append(f"{ts:.6f} rgb/{rgb_file}")
            
            # Save depth (convert to mm for TUM format)
            if depth is not None:
                depth_file = f"{ts:.6f}.png"
                depth_mm = (depth * 5000).astype(np.uint16)  # TUM uses 5000 scale
                cv2.imwrite(str(depth_path / depth_file), depth_mm)
                depth_list.append(f"{ts:.6f} depth/{depth_file}")
            
            # Ground truth pose (TUM format: tx ty tz qx qy qz qw)
            from scipy.spatial.transform import Rotation
            t = pose[:3, 3]
            r = Rotation.from_matrix(pose[:3, :3])
            q = r.as_quat()  # xyzw format
            gt_list.append(f"{ts:.6f} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}")
        
        # Write association files
        with open(data_path / "rgb.txt", 'w') as f:
            f.write("# timestamp filename\n")
            f.write("\n".join(rgb_list))
        
        with open(data_path / "depth.txt", 'w') as f:
            f.write("# timestamp filename\n")
            f.write("\n".join(depth_list))
        
        with open(data_path / "groundtruth.txt", 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            f.write("\n".join(gt_list))
    
    def _create_gslam_config(self) -> Dict:
        """Create Gaussian-SLAM configuration dictionary."""
        return {
            "dataset_name": "tum",
            "seed": self.config.seed,
            "use_wandb": False,
            "project_name": "airsplatmap",
            "data": {
                "input_path": str(self._output_path / "data"),
                "output_path": str(self._output_path / "output"),
                "scene_name": "online_slam",
            },
            "cam": {
                "fx": self._intrinsics['fx'],
                "fy": self._intrinsics['fy'],
                "cx": self._intrinsics['cx'],
                "cy": self._intrinsics['cy'],
                "H": self._height,
                "W": self._width,
                "depth_scale": 5000.0,  # TUM convention
                "depth_trunc": 10.0,
                "png_depth_scale": 5000.0,
            },
            "tracking": {
                "iterations": self.config.track_iterations,
                "w_color_loss": self.config.track_w_color_loss,
                "alpha_thre": self.config.track_alpha_thre,
                "filter_alpha": self.config.track_filter_alpha,
                "filter_outlier_depth": self.config.track_filter_outlier,
                "cam_trans_lr": self.config.track_cam_trans_lr,
                "cam_rot_lr": 0.0004,
                "help_camera_initialization": False,
                "soft_alpha": False,
            },
            "mapping": {
                "iterations": self.config.map_iterations,
                "map_every": self.config.map_every,
                "new_submap_every": self.config.new_submap_every,
                "alpha_thre": self.config.alpha_seeding_thre,
                "submap_using_motion_heuristic": False,
                "new_submap_points_num": 10000,
                "prune_alpha_threshold": 0.5,
                "prune_large_scale": 0.1,
                "densify_grad_threshold": 0.0002,
                "densify_size_threshold": 0.01,
            },
        }
    
    def _load_results(self, output_path: Path) -> None:
        """Load SLAM results."""
        # Load PLY file
        ply_files = list(output_path.glob("*_global_map.ply"))
        if ply_files:
            self._load_ply(ply_files[0])
        
        # Load estimated poses
        pose_file = output_path / "estimated_c2w.ckpt"
        if pose_file.exists():
            self._estimated_poses = torch.load(pose_file)
    
    def _load_ply(self, ply_path: Path) -> None:
        """Load Gaussians from PLY file."""
        try:
            from plyfile import PlyData
            
            plydata = PlyData.read(str(ply_path))
            vertex = plydata['vertex']
            
            # Extract data
            xyz = np.stack([
                vertex['x'], vertex['y'], vertex['z']
            ], axis=1)
            
            # Try to get colors
            if 'red' in vertex:
                colors = np.stack([
                    vertex['red'], vertex['green'], vertex['blue']
                ], axis=1).astype(np.float32) / 255.0
            else:
                colors = np.ones_like(xyz) * 0.5
            
            # Try to get opacities
            if 'opacity' in vertex:
                opacities = vertex['opacity']
            else:
                opacities = np.ones(len(xyz))
            
            self._gaussians = {
                'xyz': torch.from_numpy(xyz).float().to(self.device),
                'colors': torch.from_numpy(colors).float().to(self.device),
                'opacities': torch.from_numpy(opacities).float().to(self.device),
            }
            
            logger.info(f"Loaded {len(xyz)} Gaussians from {ply_path}")
            
        except Exception as e:
            logger.error(f"Failed to load PLY: {e}")
    
    def render_view(
        self,
        pose_world_cam: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Render a view from the Gaussian map using GPU-accelerated rendering."""
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        width, height = image_size
        
        if self._gaussians is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        try:
            from src.engines.base import render_points_gsplat
            
            xyz = self._gaussians['xyz'].cpu().numpy()
            colors = self._gaussians['colors'].cpu().numpy()
            
            if len(xyz) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            
            return render_points_gsplat(
                points=xyz,
                colors=colors,
                pose=pose_world_cam,
                intrinsics=self._intrinsics,
                image_size=image_size,
                point_size=0.008,
                device=str(self.device)
            )
            
        except Exception as e:
            logger.debug(f"GPU render failed ({e}), using software fallback")
            return self._software_render(pose_world_cam, image_size)
    
    def _software_render(self, pose_world_cam: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """Software fallback renderer."""
        width, height = image_size
        
        try:
            xyz = self._gaussians['xyz'].cpu().numpy()
            colors = self._gaussians['colors'].cpu().numpy()
            
            # Project points
            c2w = pose_world_cam
            w2c = np.linalg.inv(c2w)
            
            # Transform to camera coords
            pts_cam = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
            
            # Filter behind camera
            valid = pts_cam[:, 2] > 0.1
            pts_cam = pts_cam[valid]
            cols = colors[valid]
            
            if len(pts_cam) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            
            # Project to image
            fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
            cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
            
            u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(int)
            v = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(int)
            
            # Filter to image bounds
            valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v = u[valid], v[valid]
            cols = cols[valid]
            depths = pts_cam[valid, 2]
            
            # Sort by depth (back to front)
            order = np.argsort(-depths)
            u, v, cols = u[order], v[order], cols[order]
            
            # Render points
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[v, u] = (cols * 255).astype(np.uint8)
            
            return img
            
        except Exception as e:
            logger.debug(f"Software render failed: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def get_num_gaussians(self) -> int:
        if self._gaussians is None:
            return 0
        return len(self._gaussians['xyz'])
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        if self._gaussians is None:
            return None
        return self._gaussians['xyz'].cpu().numpy()
    
    def get_gaussian_colors(self) -> Optional[np.ndarray]:
        """Get RGB colors for each Gaussian."""
        if self._gaussians is None or 'colors' not in self._gaussians:
            return None
        return self._gaussians['colors'].cpu().numpy()
    
    def save_state(self, path: str) -> None:
        """Save the Gaussian-SLAM state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_data = {
            'intrinsics': self._intrinsics,
            'width': self._width,
            'height': self._height,
            'num_frames': len(self._frames),
            'iteration': self._iteration,
            'engine': 'gslam',
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Copy SLAM output if available
        if self._output_path and (self._output_path / "output").exists():
            output_src = self._output_path / "output"
            for item in output_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, path / item.name)
        
        # Save Gaussians
        if self._gaussians is not None:
            torch.save({
                'xyz': self._gaussians['xyz'],
                'colors': self._gaussians['colors'],
                'opacities': self._gaussians['opacities'],
            }, path / "gaussians.pth")
        
        logger.info(f"Saved Gaussian-SLAM state to {path}")
    
    def load_state(self, path: str) -> None:
        """Load Gaussian-SLAM state from disk."""
        path = Path(path)
        
        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            self._intrinsics = config_data.get('intrinsics')
            self._width = config_data.get('width', 640)
            self._height = config_data.get('height', 480)
            self._iteration = config_data.get('iteration', 0)
        
        # Load Gaussians
        gaussians_path = path / "gaussians.pth"
        if gaussians_path.exists():
            data = torch.load(gaussians_path, map_location=self.device)
            self._gaussians = {
                'xyz': data['xyz'].to(self.device),
                'colors': data['colors'].to(self.device),
                'opacities': data['opacities'].to(self.device),
            }
        
        # Try to load PLY
        ply_files = list(path.glob("*.ply"))
        if ply_files and self._gaussians is None:
            self._load_ply(ply_files[0])
        
        self._initialized = True
        logger.info(f"Loaded Gaussian-SLAM state from {path}")
    
    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._initialized = False
        self._intrinsics = None
        self._frames = []
        self._rgb_images = []
        self._depth_images = []
        self._poses = []
        self._timestamps = []
        self._gaussians = None
        self._estimated_poses = None
        self._iteration = 0
        self._frame_idx = 0
        self._height = None
        self._width = None
        
        # Clean up temp directory
        if self._output_path and self._output_path.exists():
            try:
                shutil.rmtree(self._output_path)
            except Exception:
                pass
        self._output_path = None
