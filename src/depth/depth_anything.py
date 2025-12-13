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
    Depth estimation using Depth Anything V3 (ByteDance).
    
    DA3 is a major upgrade over V2 with:
    - Metric depth estimation (DA3METRIC-LARGE)
    - Multi-view depth estimation
    - Pose estimation capabilities
    - Much better accuracy
    
    Models available:
    - 'small': DA3-SMALL (80M params) - fastest
    - 'base': DA3-BASE (120M params) - balanced  
    - 'large': DA3-LARGE (350M params) - accurate
    - 'giant': DA3-GIANT (1.15B params) - best quality
    - 'metric': DA3METRIC-LARGE (350M) - metric depth in meters
    - 'mono': DA3MONO-LARGE (350M) - high-quality relative depth
    
    Installation:
        pip install xformers torch>=2 torchvision
        pip install depth-anything-3
        # Or from source:
        git clone https://github.com/ByteDance-Seed/Depth-Anything-3
        cd Depth-Anything-3 && pip install -e .
    
    Usage:
        estimator = DepthAnythingV3Estimator(model_size='base')
        result = estimator.estimate(rgb_image)
        depth = result.depth  # Depth map
    """
    
    # HuggingFace model IDs for DA3
    MODEL_IDS = {
        'small': 'depth-anything/DA3-SMALL',
        'base': 'depth-anything/DA3-BASE',
        'large': 'depth-anything/DA3-LARGE-1.1',
        'giant': 'depth-anything/DA3-GIANT-1.1',
        'metric': 'depth-anything/DA3METRIC-LARGE',
        'mono': 'depth-anything/DA3MONO-LARGE',
    }
    
    def __init__(
        self,
        model_size: str = None,  # Auto-detect based on platform
        device: str = "cuda",
        max_resolution: int = 518,
    ):
        """
        Initialize Depth Anything V3.
        
        Args:
            model_size: 'small', 'base', 'large', 'giant', 'metric', or 'mono'
            device: 'cuda' or 'cpu'
            max_resolution: Maximum image dimension for processing
        """
        super().__init__(device)
        
        # Auto-detect model size based on platform
        if model_size is None:
            import platform
            if platform.machine().startswith('aarch'):
                model_size = 'small'  # Use small model for ARM/Jetson
                logger.info("DA3: Using 'small' model for ARM platform")
            else:
                model_size = 'base'  # Default for x86
        
        self.model_size = model_size.lower()
        self.max_resolution = max_resolution
        self._model = None
        self._is_metric = (self.model_size == 'metric')
        
        # Check if DA3 is available, fall back to V2 if not
        self._use_v3 = self._check_da3_available()
        if not self._use_v3:
            logger.warning("DA3 not installed, falling back to V2")
            v2_size_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl', 
                          'giant': 'vitl', 'metric': 'vitl', 'mono': 'vitl'}
            v2_size = v2_size_map.get(self.model_size, 'vitl')
            self._v2_estimator = DepthAnythingV2Estimator(
                model_size=v2_size, device=device, max_resolution=max_resolution
            )
    
    @staticmethod
    def _check_da3_available() -> bool:
        """Check if DA3 package is installed or available via submodule."""
        # Suppress verbose DA3 logging
        import os
        os.environ['DA3_LOG_LEVEL'] = 'ERROR'
        
        # First, try to add submodule path
        import sys
        submodule_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'submodules', 'Depth-Anything-3', 'src'
        )
        if os.path.exists(submodule_path) and submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
            logger.info(f"Added DA3 submodule path: {submodule_path}")
        
        try:
            from depth_anything_3.api import DepthAnything3
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_available() -> bool:
        # Available if either DA3 or V2 (fallback) is available
        if DepthAnythingV3Estimator._check_da3_available():
            return True
        return DepthAnythingV2Estimator.is_available()
    
    def _lazy_init(self):
        """Lazy initialization - load model on first use."""
        if self._initialized:
            return
            
        if not self._use_v3:
            self._v2_estimator._lazy_init()
            self._initialized = True
            return
        
        # Suppress verbose DA3 logging
        import os
        os.environ['DA3_LOG_LEVEL'] = 'ERROR'
        logging.getLogger('depth_anything_3').setLevel(logging.WARNING)
        
        from depth_anything_3.api import DepthAnything3
        
        model_id = self.MODEL_IDS.get(self.model_size, self.MODEL_IDS['base'])
        logger.info(f"Loading DA3 model: {model_id}")
        
        self._model = DepthAnything3.from_pretrained(model_id)
        self._model = self._model.to(device=_torch.device(self.device))
        
        logger.info(f"DA3 ({self.model_size}) initialized on {self.device}")
        self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """Estimate depth from RGB image."""
        self._lazy_init()
        
        if not self._use_v3:
            return self._v2_estimator.estimate(rgb)
        
        # DA3 expects list of image paths or numpy arrays
        # Resize if needed
        h, w = rgb.shape[:2]
        if max(h, w) > self.max_resolution:
            scale = self.max_resolution / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_resized = rgb
            new_h, new_w = h, w
        
        # Run inference
        with _torch.no_grad():
            prediction = self._model.inference([rgb_resized])
        
        # Get depth (shape: [1, H, W])
        depth = prediction.depth[0]  # float32 numpy array
        
        # Resize back to original
        if depth.shape[0] != h or depth.shape[1] != w:
            import cv2
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Get confidence if available
        confidence = None
        if hasattr(prediction, 'conf') and prediction.conf is not None:
            confidence = prediction.conf[0]
            if confidence.shape[0] != h or confidence.shape[1] != w:
                confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return DepthResult(
            depth=depth,
            confidence=confidence,
            is_metric=self._is_metric,
            min_depth=float(depth.min()),
            max_depth=float(depth.max()),
        )
    
    def get_name(self) -> str:
        return f"depth_anything_v3_{self.model_size}"
    
    def is_metric(self) -> bool:
        """Return True if this model outputs metric depth."""
        return self._is_metric
    
    def cleanup(self):
        """Release GPU memory."""
        if not self._use_v3:
            self._v2_estimator.cleanup()
        elif self._model is not None:
            del self._model
            self._model = None
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        self._initialized = False
