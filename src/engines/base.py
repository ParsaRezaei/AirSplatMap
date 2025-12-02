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

# Cache the available renderer
_AVAILABLE_RENDERER: Optional[str] = None


def get_available_renderer() -> str:
    """
    Check which GPU renderer is available.
    
    Returns:
        One of: "diff-gaussian-rasterization", "gsplat", "software"
    """
    global _AVAILABLE_RENDERER
    
    if _AVAILABLE_RENDERER is not None:
        return _AVAILABLE_RENDERER
    
    # Try diff-gaussian-rasterization first
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings
        _AVAILABLE_RENDERER = "diff-gaussian-rasterization"
        return _AVAILABLE_RENDERER
    except ImportError:
        pass
    
    # Try gsplat
    try:
        import torch
        from gsplat import rasterization
        # Quick test to see if CUDA kernels work
        means = torch.rand(2, 3, device='cuda')
        quats = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32, device='cuda')
        scales = torch.ones(2, 3, device='cuda') * 0.1
        opacities = torch.ones(2, device='cuda')
        colors = torch.ones(2, 3, device='cuda')
        viewmat = torch.eye(4, device='cuda')
        viewmat[2, 3] = -3.0
        K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32, device='cuda')
        _ = rasterization(means=means, quats=quats, scales=scales, opacities=opacities,
                         colors=colors, viewmats=viewmat[None], Ks=K[None], width=640, height=480)
        _AVAILABLE_RENDERER = "gsplat"
        return _AVAILABLE_RENDERER
    except Exception:
        pass
    
    _AVAILABLE_RENDERER = "software"
    return _AVAILABLE_RENDERER


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
    Render point cloud using GPU-accelerated Gaussian rasterization.
    
    Tries renderers in order:
    1. diff-gaussian-rasterization (graphdeco) - most compatible
    2. gsplat (nerfstudio) - faster but requires JIT compilation
    3. Software fallback
    
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
    
    width, height = image_size
    
    if len(points) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Try diff-gaussian-rasterization first (most compatible on Windows)
    try:
        return _render_with_diff_gaussian_rasterization(
            points, colors, pose, intrinsics, image_size, point_size, device
        )
    except Exception as e1:
        pass
    
    # Try gsplat next
    try:
        return _render_with_gsplat(
            points, colors, pose, intrinsics, image_size, point_size, device
        )
    except Exception as e2:
        pass
    
    # Fallback to software rendering
    return _software_render_points(points, colors, pose, intrinsics, image_size)


def _render_with_diff_gaussian_rasterization(
    points: np.ndarray,
    colors: np.ndarray,
    pose: np.ndarray,
    intrinsics: Dict[str, float],
    image_size: Tuple[int, int],
    point_size: float = 0.01,
    device: str = "cuda"
) -> np.ndarray:
    """Render using diff-gaussian-rasterization (graphdeco)."""
    import torch
    import sys
    from pathlib import Path
    
    # Add graphdeco paths if needed
    graphdeco_path = Path(__file__).parent.parent.parent / "submodules" / "gaussian-splatting"
    if str(graphdeco_path) not in sys.path:
        sys.path.insert(0, str(graphdeco_path))
    
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    
    width, height = image_size
    n = len(points)
    
    # Camera parameters
    fx = intrinsics['fx'] * width / intrinsics['width']
    fy = intrinsics['fy'] * height / intrinsics['height']
    cx = intrinsics['cx'] * width / intrinsics['width']
    cy = intrinsics['cy'] * height / intrinsics['height']
    
    # Compute FoV from focal length
    import math
    fov_x = 2 * math.atan(width / (2 * fx))
    fov_y = 2 * math.atan(height / (2 * fy))
    
    # World to camera transform
    w2c = np.linalg.inv(pose).astype(np.float32)
    
    # View matrix (transposed for column-major)
    view_matrix = torch.from_numpy(w2c.T).to(device)
    
    # Projection matrix
    znear, zfar = 0.01, 100.0
    tan_fov_x = math.tan(fov_x / 2)
    tan_fov_y = math.tan(fov_y / 2)
    
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 1 / tan_fov_x
    proj[1, 1] = 1 / tan_fov_y
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -2 * zfar * znear / (zfar - znear)
    proj[3, 2] = -1.0
    
    full_proj = view_matrix @ proj
    
    # Camera center
    cam_center = torch.from_numpy(pose[:3, 3].astype(np.float32)).to(device)
    
    # Rasterizer settings
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Prepare Gaussian parameters
    means3D = torch.from_numpy(points.astype(np.float32)).to(device)
    
    # Project means to 2D for screen space
    ones = torch.ones(n, 1, device=device)
    means_h = torch.cat([means3D, ones], dim=1)
    means_proj = means_h @ full_proj
    means2D = means_proj[:, :2] / (means_proj[:, 3:4] + 1e-8)
    means2D.requires_grad_(True)  # Needed for rasterizer
    
    # Colors as SH DC component
    C0 = 0.28209479177387814
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors_tensor = torch.from_numpy(colors.astype(np.float32)).to(device)
    shs = (colors_tensor - 0.5) / C0
    shs = shs.unsqueeze(1)  # (N, 1, 3)
    
    # Scales and rotations
    scales = torch.full((n, 3), point_size, device=device)
    rotations = torch.zeros((n, 4), device=device)
    rotations[:, 0] = 1.0  # w = 1 for identity quaternion
    
    # Opacities (full)
    opacities = torch.ones((n, 1), device=device)
    
    # Covariance from scales and rotations
    def build_covariance(scales, rotations):
        """Build 3D covariance matrices from scales and quaternions."""
        # Rotation matrix from quaternion
        r, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - r*z), 2*(x*z + r*y),
            2*(x*y + r*z), 1 - 2*(x*x + z*z), 2*(y*z - r*x),
            2*(x*z - r*y), 2*(y*z + r*x), 1 - 2*(x*x + y*y)
        ], dim=-1).reshape(-1, 3, 3)
        
        # Scale matrix
        S = torch.diag_embed(scales)
        
        # Covariance = R @ S @ S^T @ R^T
        L = R @ S
        cov = L @ L.transpose(-1, -2)
        
        # Return upper triangle (6 values)
        return torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]
        ], dim=-1)
    
    cov3D = build_covariance(scales, rotations)
    
    with torch.no_grad():
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
    
    # rendered_image is (3, H, W)
    img = rendered_image.permute(1, 2, 0).clamp(0, 1)
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    return img


def _render_with_gsplat(
    points: np.ndarray,
    colors: np.ndarray,
    pose: np.ndarray,
    intrinsics: Dict[str, float],
    image_size: Tuple[int, int],
    point_size: float = 0.01,
    device: str = "cuda"
) -> np.ndarray:
    """Render using gsplat (nerfstudio)."""
    import torch
    import gsplat
    
    width, height = image_size
    n = len(points)
    
    # Convert to tensors
    means = torch.from_numpy(points.astype(np.float32)).to(device)
    if colors.max() > 1.0:
        colors = colors / 255.0
    rgbs = torch.from_numpy(colors.astype(np.float32)).to(device)
    
    # Create uniform small scales (spherical Gaussians)
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
            sh_degree=None,
        )
        
        # renders is (1, H, W, 3)
        img = renders[0].clamp(0, 1)
        img = (img.cpu().numpy() * 255).astype(np.uint8)
        return img


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
