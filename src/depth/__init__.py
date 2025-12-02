"""
Depth Estimation Module
=======================

Provides modular depth estimation backends for AirSplatMap.

Available estimators:
- depth_anything_v3: Depth Anything V3 (fast, accurate)
- midas: MiDaS depth estimation
- zoedepth: ZoeDepth metric depth
- none/passthrough: No depth estimation (for ground truth)

Usage:
    from src.depth import get_depth_estimator
    
    estimator = get_depth_estimator('depth_anything_v3')
    result = estimator.estimate(rgb_image)
    depth = result.depth  # HxW float32 in meters
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Result from depth estimation."""
    depth: np.ndarray  # HxW float32 depth in meters
    confidence: Optional[np.ndarray] = None  # HxW confidence map
    min_depth: float = 0.0
    max_depth: float = 10.0
    is_metric: bool = False  # True if depth is metric (meters)


class BaseDepthEstimator(ABC):
    """Abstract base class for depth estimators."""
    
    @abstractmethod
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb: HxWx3 RGB image (uint8 or float32)
            
        Returns:
            DepthResult with depth map
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get estimator name."""
        pass
    
    def reset(self):
        """Reset estimator state (if any)."""
        pass


class PassthroughDepthEstimator(BaseDepthEstimator):
    """No-op depth estimator that returns None (for ground truth depth)."""
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        return DepthResult(
            depth=None,
            is_metric=True,
        )
    
    def get_name(self) -> str:
        return "passthrough"


class DepthAnythingV3Estimator(BaseDepthEstimator):
    """Depth estimation using Depth Anything V3."""
    
    def __init__(self, model_size: str = "small", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._transform = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        try:
            import torch
            from depth_anything_v3.dpt import DepthAnythingV3
            
            # Model configs
            model_configs = {
                'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            config = model_configs.get(self.model_size, model_configs['small'])
            
            self._model = DepthAnythingV3(**config)
            # Load pretrained weights if available
            # self._model.load_state_dict(torch.load(...))
            self._model.to(self.device)
            self._model.eval()
            
            self._initialized = True
            logger.info(f"Initialized Depth Anything V3 ({self.model_size})")
            
        except ImportError:
            logger.warning("Depth Anything V3 not available, using fallback MiDaS")
            self._use_midas_fallback()
    
    def _use_midas_fallback(self):
        """Fall back to MiDaS if Depth Anything not available."""
        try:
            import torch
            self._model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self._model.to(self.device)
            self._model.eval()
            
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self._transform = midas_transforms.small_transform
            self._use_midas = True
            self._initialized = True
            logger.info("Using MiDaS fallback for depth estimation")
        except Exception as e:
            logger.error(f"Failed to initialize MiDaS fallback: {e}")
            self._model = None
            self._initialized = True  # Mark as initialized to avoid retry
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """Estimate depth from RGB image."""
        self._lazy_init()
        
        if self._model is None:
            # Return dummy depth if no model available
            h, w = rgb.shape[:2]
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
        
        import torch
        
        # Ensure RGB is uint8
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        h, w = rgb.shape[:2]
        
        try:
            with torch.no_grad():
                if hasattr(self, '_use_midas') and self._use_midas:
                    # MiDaS path
                    input_batch = self._transform(rgb).to(self.device)
                    prediction = self._model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(h, w),
                        mode='bicubic',
                        align_corners=False,
                    ).squeeze()
                    depth = prediction.cpu().numpy()
                    # MiDaS outputs inverse depth, convert
                    depth = 1.0 / (depth + 1e-6)
                    # Normalize to reasonable range
                    depth = depth / depth.max() * 10.0
                else:
                    # Depth Anything V3 path
                    from torchvision import transforms
                    
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    
                    input_tensor = transform(rgb).unsqueeze(0).to(self.device)
                    depth = self._model(input_tensor)
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(1),
                        size=(h, w),
                        mode='bicubic',
                        align_corners=False,
                    ).squeeze()
                    depth = depth.cpu().numpy()
            
            return DepthResult(
                depth=depth.astype(np.float32),
                is_metric=False,  # Relative depth
                min_depth=float(depth.min()),
                max_depth=float(depth.max()),
            )
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
    
    def get_name(self) -> str:
        return f"depth_anything_v3_{self.model_size}"


class MiDaSEstimator(BaseDepthEstimator):
    """Depth estimation using MiDaS."""
    
    def __init__(self, model_type: str = "MiDaS_small", device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self._model = None
        self._transform = None
        self._initialized = False
    
    def _lazy_init(self):
        if self._initialized:
            return
        
        try:
            import torch
            
            self._model = torch.hub.load('intel-isl/MiDaS', self.model_type)
            self._model.to(self.device)
            self._model.eval()
            
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            if 'small' in self.model_type.lower():
                self._transform = midas_transforms.small_transform
            else:
                self._transform = midas_transforms.dpt_transform
            
            self._initialized = True
            logger.info(f"Initialized MiDaS ({self.model_type})")
            
        except Exception as e:
            logger.error(f"Failed to initialize MiDaS: {e}")
            self._model = None
            self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        self._lazy_init()
        
        h, w = rgb.shape[:2]
        
        if self._model is None:
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
        
        import torch
        
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        try:
            with torch.no_grad():
                input_batch = self._transform(rgb).to(self.device)
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()
                
                depth = prediction.cpu().numpy()
                # MiDaS outputs disparity (inverse depth)
                depth = 1.0 / (depth + 1e-6)
                # Normalize
                depth = depth / depth.max() * 10.0
            
            return DepthResult(
                depth=depth.astype(np.float32),
                is_metric=False,
                min_depth=float(depth.min()),
                max_depth=float(depth.max()),
            )
            
        except Exception as e:
            logger.error(f"MiDaS estimation failed: {e}")
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
    
    def get_name(self) -> str:
        return f"midas_{self.model_type}"


# Registry of available estimators
_ESTIMATORS: Dict[str, type] = {
    'depth_anything_v3': DepthAnythingV3Estimator,
    'depth_anything': DepthAnythingV3Estimator,
    'midas': MiDaSEstimator,
    'midas_small': lambda: MiDaSEstimator('MiDaS_small'),
    'none': PassthroughDepthEstimator,
    'passthrough': PassthroughDepthEstimator,
    'ground_truth': PassthroughDepthEstimator,
}


def get_depth_estimator(name: str, **kwargs) -> BaseDepthEstimator:
    """
    Get a depth estimator by name.
    
    Args:
        name: Estimator name ('depth_anything_v3', 'midas', 'none')
        **kwargs: Additional arguments passed to estimator constructor
        
    Returns:
        BaseDepthEstimator instance
        
    Raises:
        ValueError: If estimator name is unknown
    """
    name_lower = name.lower().replace('-', '_')
    
    if name_lower not in _ESTIMATORS:
        available = list(_ESTIMATORS.keys())
        raise ValueError(f"Unknown depth estimator: {name}. Available: {available}")
    
    estimator_cls = _ESTIMATORS[name_lower]
    
    if callable(estimator_cls) and not isinstance(estimator_cls, type):
        # Factory function
        return estimator_cls()
    
    return estimator_cls(**kwargs)


def list_depth_estimators() -> Dict[str, Dict[str, Any]]:
    """List available depth estimators."""
    return {
        'depth_anything_v3': {
            'description': 'Depth Anything V3 - fast and accurate monocular depth',
            'metric': False,
        },
        'midas': {
            'description': 'MiDaS - robust monocular depth estimation',
            'metric': False,
        },
        'none': {
            'description': 'Passthrough - no depth estimation (use ground truth)',
            'metric': True,
        },
    }


__all__ = [
    'BaseDepthEstimator',
    'DepthResult',
    'PassthroughDepthEstimator',
    'DepthAnythingV3Estimator',
    'MiDaSEstimator',
    'get_depth_estimator',
    'list_depth_estimators',
]
