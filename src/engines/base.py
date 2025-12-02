"""
Base Engine Interface for 3D Gaussian Splatting Backends
========================================================

This module defines the abstract interface that all 3DGS engine implementations
must follow. The goal is to provide a clean separation between the online
pipeline/orchestration logic and the underlying 3DGS implementation.

New engines (e.g., LiveSplat, SplaTAM, Nerfstudio splatfacto) can be added
by implementing this interface without touching the pipeline code.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BaseGSEngine(ABC):
    """
    Abstract base class for 3D Gaussian Splatting engines.
    
    All 3DGS backends must implement this interface to be usable with the
    OnlineGSPipeline. The interface supports:
    
    - Scene initialization with camera intrinsics and configuration
    - Incremental frame addition (online mapping)
    - Step-wise optimization (for real-time / near-real-time operation)
    - View rendering for visualization and evaluation
    - State persistence (save/load)
    
    Coordinate conventions:
    - Poses are 4x4 transformation matrices in world-from-camera convention
      (i.e., pose_world_cam transforms points from camera frame to world frame)
    - RGB images are HxWx3 numpy arrays with values in [0, 255] (uint8) or [0, 1] (float)
    - Depth images are HxW numpy arrays with metric depth values
    
    Thread safety:
    - Implementations are NOT expected to be thread-safe by default
    - External synchronization is required for multi-threaded use
    """
    
    @abstractmethod
    def initialize_scene(
        self,
        intrinsics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize the 3DGS scene with camera intrinsics and engine configuration.
        
        This must be called before adding any frames. The engine should set up
        all internal state required for mapping.
        
        Args:
            intrinsics: Camera intrinsic parameters. Expected keys:
                - 'fx': Focal length in x (pixels)
                - 'fy': Focal length in y (pixels)  
                - 'cx': Principal point x (pixels)
                - 'cy': Principal point y (pixels)
                - 'width': Image width (pixels)
                - 'height': Image height (pixels)
                Optional keys may include distortion coefficients.
                
            config: Engine-specific configuration dictionary. Common options:
                - 'sh_degree': Spherical harmonics degree (default: 3)
                - 'initial_points': Initial point cloud or number of random points
                - 'learning_rates': Dict of learning rates for different params
                - 'densification': Dict of densification parameters
                Engine implementations should document their specific options.
                
        Raises:
            ValueError: If required intrinsic parameters are missing
            RuntimeError: If scene is already initialized (call reset first)
        """
        pass
    
    @abstractmethod
    def add_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose_world_cam: np.ndarray
    ) -> None:
        """
        Register a new observation (frame) with the 3DGS scene.
        
        This adds a new camera viewpoint to the scene. The frame will be used
        in subsequent optimization steps. Implementations may:
        - Add new Gaussians based on the depth/RGB
        - Simply store the frame for later training
        - Perform immediate local optimization
        
        Args:
            frame_id: Unique integer identifier for this frame. Should be
                monotonically increasing for proper temporal ordering.
                
            rgb: RGB image as HxWx3 numpy array. Can be:
                - uint8 with values in [0, 255]
                - float32/float64 with values in [0, 1]
                Implementations should handle both formats.
                
            depth: Optional depth image as HxW numpy array with metric depth
                values. None if depth is not available. Depth can significantly
                improve reconstruction quality if available.
                
            pose_world_cam: 4x4 numpy array representing the camera-to-world
                transformation. The matrix should be:
                [[R | t]
                 [0 | 1]]
                where R is 3x3 rotation and t is 3x1 translation.
                
        Raises:
            RuntimeError: If scene is not initialized
            ValueError: If frame dimensions don't match initialized intrinsics
        """
        pass
    
    @abstractmethod
    def optimize_step(self, n_steps: int = 1) -> Dict[str, float]:
        """
        Run n_steps of optimization/mapping on the current scene.
        
        This performs incremental optimization of the Gaussian parameters
        using the registered frames. For online operation, this is typically
        called with a small number of steps after each new frame.
        
        The optimization may include:
        - Gradient descent on Gaussian parameters (position, color, opacity, etc.)
        - Densification (splitting/cloning Gaussians)
        - Pruning (removing low-opacity or large Gaussians)
        
        Args:
            n_steps: Number of optimization iterations to run. Default is 1.
                Higher values give better quality but take more time.
                
        Returns:
            Dictionary with optimization metrics. Common keys:
                - 'loss': Total loss value
                - 'psnr': Peak signal-to-noise ratio (if computed)
                - 'num_gaussians': Current number of Gaussians
            The specific metrics depend on the engine implementation.
            
        Raises:
            RuntimeError: If scene is not initialized or no frames added
        """
        pass
    
    @abstractmethod
    def render_view(
        self,
        pose_world_cam: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Render an RGB image from the current Gaussian scene at a given pose.
        
        This renders the scene without updating any parameters. Useful for:
        - Visualization during training
        - Evaluation against ground truth
        - Novel view synthesis
        
        Args:
            pose_world_cam: 4x4 numpy array for camera-to-world transformation.
                Same convention as in add_frame().
                
            image_size: Tuple of (width, height) for the output image.
                This may differ from the training image size.
                
        Returns:
            RGB image as HxWx3 numpy array with values in [0, 255] (uint8).
            
        Raises:
            RuntimeError: If scene is not initialized
        """
        pass
    
    @abstractmethod
    def save_state(self, path: str) -> None:
        """
        Save the current engine state to disk.
        
        This should save all information needed to restore the scene:
        - Gaussian parameters (positions, colors, opacities, covariances)
        - Optimizer state (for continued training)
        - Configuration and intrinsics
        
        The exact format is engine-specific. Common formats:
        - PLY file for Gaussians (standard for 3DGS)
        - PyTorch checkpoint for optimizer state
        - JSON for configuration
        
        Args:
            path: Directory or file path where state should be saved.
                If a directory, the engine creates appropriate files inside.
                
        Raises:
            RuntimeError: If scene is not initialized
            IOError: If unable to write to the specified path
        """
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> None:
        """
        Load engine state from disk.
        
        This restores a previously saved scene. After loading:
        - render_view() should work immediately
        - optimize_step() should continue training from saved state
        - add_frame() should work to add new observations
        
        Args:
            path: Directory or file path from which to load state.
                Should match the format used by save_state().
                
        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If the saved state is incompatible or corrupted
        """
        pass
    
    # ----- Optional methods with default implementations -----
    
    def reset(self) -> None:
        """
        Reset the engine to uninitialized state.
        
        This clears all internal state, allowing initialize_scene() to be
        called again. Useful for processing multiple scenes sequentially.
        
        Default implementation does nothing; override if cleanup is needed.
        """
        pass
    
    def get_num_gaussians(self) -> int:
        """
        Get the current number of Gaussians in the scene.
        
        Returns:
            Number of active Gaussians, or 0 if not initialized.
        """
        return 0
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        """
        Get the current Gaussian centers as a point cloud.
        
        Returns:
            Nx3 numpy array of Gaussian positions, or None if not available.
        """
        return None
    
    @property
    def is_initialized(self) -> bool:
        """
        Check if the engine has been initialized.
        
        Returns:
            True if initialize_scene() has been called successfully.
        """
        return False


def render_points_gsplat(
    points: np.ndarray,
    colors: np.ndarray,
    pose: np.ndarray,
    intrinsics: Dict[str, float],
    image_size: Tuple[int, int],
    point_size: float = 0.01,
    device: str = "cuda"
) -> np.ndarray:
    """
    Render point cloud using gsplat's rasterizer for GPU-accelerated rendering.
    
    This creates small spherical Gaussians from points for efficient rendering.
    
    Args:
        points: Nx3 array of 3D positions
        colors: Nx3 array of RGB colors (0-1)
        pose: 4x4 world-to-camera transformation
        intrinsics: Camera intrinsics dict
        image_size: (width, height) tuple
        point_size: Size of each point as a Gaussian
        device: CUDA device
        
    Returns:
        HxWx3 uint8 image
    """
    import torch
    import math
    
    width, height = image_size
    
    if len(points) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        import gsplat
        
        # Convert to tensors
        means = torch.from_numpy(points.astype(np.float32)).to(device)
        rgbs = torch.from_numpy(colors.astype(np.float32)).to(device)
        
        # Create uniform small scales (spherical Gaussians)
        n = len(means)
        scales = torch.full((n, 3), point_size, device=device)
        
        # Identity rotations (quaternions: w, x, y, z)
        quats = torch.zeros((n, 4), device=device)
        quats[:, 0] = 1.0  # w = 1 for identity
        
        # Full opacity
        opacities = torch.ones((n,), device=device)
        
        # Camera parameters
        fx = intrinsics['fx'] * width / intrinsics['width']
        fy = intrinsics['fy'] * height / intrinsics['height']
        cx = intrinsics['cx'] * width / intrinsics['width']
        cy = intrinsics['cy'] * height / intrinsics['height']
        
        # View matrix (world to camera)
        viewmat = torch.from_numpy(np.linalg.inv(pose).astype(np.float32)).to(device)
        
        # Camera intrinsics matrix
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # Render using gsplat
        with torch.no_grad():
            # Use gsplat's rasterization
            renders, alphas, meta = gsplat.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=rgbs,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=100.0,
                sh_degree=None,  # Using direct colors, not SH
            )
            
            # renders is (1, H, W, 3)
            img = renders[0].clamp(0, 1)
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            return img
            
    except Exception as e:
        # Fallback to software rendering
        return _software_render_points(points, colors, pose, intrinsics, image_size)


def _software_render_points(
    points: np.ndarray,
    colors: np.ndarray,
    pose: np.ndarray,
    intrinsics: Dict[str, float],
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Software fallback for point rendering."""
    width, height = image_size
    
    if len(points) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Transform to camera frame
    pose_inv = np.linalg.inv(pose)
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    pts_cam = (pose_inv @ pts_h.T).T[:, :3]
    
    # Filter behind camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    cols = colors[valid] if colors is not None else np.full((len(pts_cam), 3), 0.6)
    
    if len(pts_cam) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Project
    fx = intrinsics['fx'] * width / intrinsics['width']
    fy = intrinsics['fy'] * height / intrinsics['height']
    cx = intrinsics['cx'] * width / intrinsics['width']
    cy = intrinsics['cy'] * height / intrinsics['height']
    
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
    
    # Render
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(cols) > 0:
        cols_uint8 = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        img[v, u] = cols_uint8
    
    return img
