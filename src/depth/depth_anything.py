"""
Depth Anything V2 Estimator
===========================

Wrapper for Depth Anything V2 - state-of-the-art monocular depth estimation.

Features:
- Fast inference (~30ms for small model)
- Three model sizes: small (24M), base (97M), large (335M)
- Relative depth output (requires scaling for metric)

Installation:
    pip install depth-anything-v2
    
    # Or from source:
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    cd Depth-Anything-V2 && pip install -e .
"""

import numpy as np
import logging
from typing import Optional

from .base import BaseDepthEstimator, DepthResult

logger = logging.getLogger(__name__)

# Check availability
_AVAILABLE = False
_torch = None
_DepthAnythingV2 = None

try:
    import torch as _torch
    _AVAILABLE = True
except ImportError:
    pass


class DepthAnythingV2Estimator(BaseDepthEstimator):
    """
    Depth estimation using Depth Anything V2.
    
    Produces high-quality relative depth maps. For metric depth,
    use DepthScaler with sparse reference points.
    
    Model sizes:
    - 'vits': Small (24M params) - fastest, ~30ms
    - 'vitb': Base (97M params) - balanced
    - 'vitl': Large (335M params) - most accurate
    
    Usage:
        estimator = DepthAnythingV2Estimator(model_size='vits')
        result = estimator.estimate(rgb_image)
        depth = result.depth  # Relative depth
    """
    
    # HuggingFace model IDs
    MODEL_IDS = {
        'vits': 'depth-anything/Depth-Anything-V2-Small-hf',
        'vitb': 'depth-anything/Depth-Anything-V2-Base-hf',
        'vitl': 'depth-anything/Depth-Anything-V2-Large-hf',
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    }
    
    def __init__(
        self,
        model_size: str = "vits",
        device: str = "cuda",
        max_resolution: int = 518,
    ):
        """
        Initialize Depth Anything V2.
        
        Args:
            model_size: 'vits', 'vitb', 'vitl' (or 'small', 'base', 'large')
            device: 'cuda' or 'cpu'
            max_resolution: Maximum image dimension for processing
        """
        super().__init__(device)
        self.model_size = model_size.lower()
        self.max_resolution = max_resolution
        self._model = None
        self._processor = None
    
    @staticmethod
    def is_available() -> bool:
        return _AVAILABLE
    
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        if not _AVAILABLE:
            logger.error("PyTorch not available")
            self._initialized = True
            return
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_id = self.MODEL_IDS.get(self.model_size, self.MODEL_IDS['vits'])
            
            logger.info(f"Loading Depth Anything V2 ({self.model_size})...")
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self._model.to(self.device)
            self._model.eval()
            
            self._initialized = True
            logger.info(f"Depth Anything V2 ({self.model_size}) initialized on {self.device}")
            
        except ImportError:
            logger.warning("transformers not available, trying torch hub...")
            self._try_torch_hub()
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V2: {e}")
            self._try_torch_hub()
    
    def _try_torch_hub(self):
        """Try loading from torch hub as fallback."""
        try:
            self._model = _torch.hub.load(
                'LiheYoung/Depth-Anything',
                f'depth_anything_{self.model_size}14',
                pretrained=True,
            )
            self._model.to(self.device)
            self._model.eval()
            self._use_hub = True
            self._initialized = True
            logger.info(f"Loaded Depth Anything V2 from torch hub")
        except Exception as e:
            logger.error(f"Failed to load from torch hub: {e}")
            self._model = None
            self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """Estimate depth from RGB image."""
        self._lazy_init()
        
        h, w = rgb.shape[:2]
        
        if self._model is None:
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
        
        # Ensure uint8 RGB
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        try:
            with _torch.no_grad():
                if hasattr(self, '_use_hub') and self._use_hub:
                    # Torch hub path
                    depth = self._estimate_hub(rgb)
                else:
                    # HuggingFace path
                    depth = self._estimate_hf(rgb)
            
            # Resize to original size
            if depth.shape != (h, w):
                import cv2
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return DepthResult(
                depth=depth.astype(np.float32),
                is_metric=False,
                min_depth=float(depth.min()),
                max_depth=float(depth.max()),
            )
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
    
    def _estimate_hf(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate using HuggingFace model."""
        from PIL import Image
        
        image = Image.fromarray(rgb)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self._model(**inputs)
        depth = outputs.predicted_depth
        
        # Interpolate to original size
        depth = _torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return depth.cpu().numpy()
    
    def _estimate_hub(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate using torch hub model."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        h, w = rgb.shape[:2]
        
        # Resize if too large
        if max(h, w) > self.max_resolution:
            scale = self.max_resolution / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            rgb = cv2.resize(rgb, (new_w, new_h))
        
        input_tensor = transform(rgb).unsqueeze(0).to(self.device)
        depth = self._model(input_tensor)
        
        return depth.squeeze().cpu().numpy()
    
    def get_name(self) -> str:
        return f"depth_anything_v2_{self.model_size}"
    
    def is_metric(self) -> bool:
        return False
    
    def cleanup(self):
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, '_processor') and self._processor is not None:
            del self._processor
            self._processor = None
        self._initialized = False
        
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            import gc
            gc.collect()


class DepthAnythingV3Estimator(BaseDepthEstimator):
    """
    Depth estimation using Depth Anything V3 (DA3).
    
    DA3 provides improved accuracy over V2 with similar speed.
    Uses the official depth_anything_3 package.
    
    Model sizes:
    - 'small': DA3-SMALL (34M params) - fastest
    - 'base': DA3-BASE (120M params) - balanced
    - 'large': DA3-LARGE (350M params) - most accurate
    - 'giant': DA3-GIANT (1.15B params) - highest quality
    
    Installation:
        pip install depth-anything-3
        # Or from source:
        git clone https://github.com/ByteDance-Seed/Depth-Anything-3
        cd Depth-Anything-3 && pip install -e .
    
    Usage:
        estimator = DepthAnythingV3Estimator(model_size='base')
        result = estimator.estimate(rgb_image)
        depth = result.depth  # Relative depth
    """
    
    # HuggingFace model IDs for DA3
    MODEL_IDS = {
        'small': 'depth-anything/DA3-SMALL',
        'base': 'depth-anything/DA3-BASE',
        'large': 'depth-anything/DA3-LARGE',
        'giant': 'depth-anything/DA3-GIANT',
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        max_resolution: int = 518,
    ):
        """
        Initialize Depth Anything V3.
        
        Args:
            model_size: 'small', 'base', 'large', or 'giant'
            device: 'cuda' or 'cpu'
            max_resolution: Maximum image dimension for processing
        """
        super().__init__(device)
        self.model_size = model_size.lower()
        self.max_resolution = max_resolution
        self._model = None
    
    @staticmethod
    def is_available() -> bool:
        if not _AVAILABLE:
            return False
        try:
            from depth_anything_3.api import DepthAnything3
            return True
        except ImportError:
            return False
    
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        if not _AVAILABLE:
            logger.error("PyTorch not available")
            self._initialized = True
            return
        
        try:
            from depth_anything_3.api import DepthAnything3
            
            model_id = self.MODEL_IDS.get(self.model_size, self.MODEL_IDS['base'])
            
            logger.info(f"Loading Depth Anything V3 ({self.model_size})...")
            self._model = DepthAnything3.from_pretrained(model_id)
            self._model = self._model.to(device=self.device)
            
            self._initialized = True
            logger.info(f"Depth Anything V3 ({self.model_size}) initialized on {self.device}")
            
        except ImportError as e:
            logger.error(f"depth_anything_3 package not installed: {e}")
            logger.error("Install with: pip install depth-anything-3")
            self._model = None
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V3: {e}")
            self._model = None
            self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """Estimate depth from RGB image."""
        self._lazy_init()
        
        h, w = rgb.shape[:2]
        
        if self._model is None:
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
        
        # Ensure uint8 RGB
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        try:
            import cv2
            
            # Resize if too large
            if max(h, w) > self.max_resolution:
                scale = self.max_resolution / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                rgb_resized = cv2.resize(rgb, (new_w, new_h))
            else:
                rgb_resized = rgb
            
            # Run inference using DA3 API
            prediction = self._model.inference([rgb_resized])
            depth = prediction.depth[0]  # (H, W) numpy array
            
            # Resize back to original size if needed
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return DepthResult(
                depth=depth.astype(np.float32),
                is_metric=False,
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
    
    def is_metric(self) -> bool:
        return False
    
    def cleanup(self):
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._initialized = False
        
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            import gc
            gc.collect()
