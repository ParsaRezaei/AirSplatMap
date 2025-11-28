"""
Online 3D Gaussian Splatting Pipeline
=====================================

This module provides the main orchestration class for online/incremental
3D Gaussian Splatting. It ties together:

- A pluggable 3DGS engine (BaseGSEngine implementations)
- A frame source (FrameSource implementations)
- Optional rolling shutter correction (RSCorrector implementations)

The pipeline is engine-agnostic - it only talks to the BaseGSEngine interface,
allowing different 3DGS backends to be swapped without changing pipeline code.

Example usage:
    from src.engines import GraphdecoEngine
    from src.pipeline import OnlineGSPipeline, TumRGBDSource, IdentityRSCorrector
    
    engine = GraphdecoEngine()
    source = TumRGBDSource(dataset_root="../datasets")
    corrector = IdentityRSCorrector()
    
    pipeline = OnlineGSPipeline(
        engine=engine,
        frame_source=source,
        rs_corrector=corrector,
        steps_per_frame=5,
    )
    
    # Run the full pipeline
    pipeline.run(max_frames=200)
    
    # Or step through manually
    for frame in source:
        metrics = pipeline.step(frame)
        if frame.idx % 50 == 0:
            preview = engine.render_view(frame.pose, frame.image_size)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import numpy as np

from ..engines.base import BaseGSEngine
from .frames import Frame, FrameSource
from .rs_corrector import RSCorrector, IdentityRSCorrector

logger = logging.getLogger(__name__)


class OnlineGSPipeline:
    """
    Online/incremental 3D Gaussian Splatting pipeline.
    
    This class orchestrates the online mapping process:
    1. Reads frames from a FrameSource
    2. Optionally applies rolling shutter correction
    3. Feeds frames to the 3DGS engine
    4. Runs optimization steps after each frame
    5. Optionally renders preview images and logs metrics
    
    The pipeline is designed to be:
    - Engine-agnostic: Works with any BaseGSEngine implementation
    - Extensible: Easy to add new processing steps via callbacks
    - Configurable: Many parameters for trading off quality vs. speed
    
    Args:
        engine: A BaseGSEngine implementation (e.g., GraphdecoEngine)
        
        frame_source: A FrameSource that yields Frame objects
        
        rs_corrector: Optional RSCorrector for rolling shutter correction.
            If None, no RS correction is applied.
            
        steps_per_frame: Number of optimization steps after each new frame.
            Higher values = better quality but slower. Default: 5
            
        warmup_frames: Number of initial frames before starting optimization.
            Useful to accumulate enough views for stable optimization. Default: 1
            
        render_every: Render a preview image every N frames. 0 = never. Default: 0
        
        save_every: Save engine state every N frames. 0 = never. Default: 0
        
        output_dir: Directory for saving outputs (renders, checkpoints).
            If None, uses './output/online_gs/'
            
        config: Additional configuration passed to engine.initialize_scene()
        
    Attributes:
        engine: The 3DGS engine being used
        metrics_history: List of metrics from each optimization step
        frame_count: Number of frames processed
    """
    
    def __init__(
        self,
        engine: BaseGSEngine,
        frame_source: FrameSource,
        rs_corrector: Optional[RSCorrector] = None,
        steps_per_frame: int = 5,
        warmup_frames: int = 1,
        render_every: int = 0,
        save_every: int = 0,
        output_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.engine = engine
        self.frame_source = frame_source
        self.rs_corrector = rs_corrector or IdentityRSCorrector()
        self.steps_per_frame = steps_per_frame
        self.warmup_frames = max(1, warmup_frames)
        self.render_every = render_every
        self.save_every = save_every
        self.output_dir = Path(output_dir or "./output/online_gs")
        self.config = config or {}
        
        # State
        self._initialized = False
        self._intrinsics: Optional[Dict[str, float]] = None
        self.frame_count = 0
        self.metrics_history: List[Dict[str, float]] = []
        
        # Callbacks
        self._on_frame_callbacks: List[Callable[[Frame, Dict[str, float]], None]] = []
        self._on_render_callbacks: List[Callable[[np.ndarray, int], None]] = []
        
        # Timing
        self._total_time = 0.0
        self._frame_times: List[float] = []
    
    def _maybe_init_scene(self, frame: Frame) -> None:
        """
        Initialize the engine scene on the first frame.
        
        Uses the frame's intrinsics and the pipeline's config to set up
        the engine. This is called automatically on the first step().
        """
        if self._initialized:
            return
        
        self._intrinsics = frame.intrinsics.copy()
        
        logger.info(f"Initializing scene with intrinsics: {self._intrinsics}")
        logger.info(f"Engine config: {self.config}")
        
        self.engine.initialize_scene(self._intrinsics, self.config)
        self._initialized = True
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def step(self, frame: Frame) -> Dict[str, float]:
        """
        Process a single frame through the pipeline.
        
        This method:
        1. Initializes the scene if this is the first frame
        2. Applies rolling shutter correction (if configured)
        3. Adds the frame to the engine
        4. Runs optimization steps
        5. Optionally renders and saves outputs
        
        Args:
            frame: The Frame to process
            
        Returns:
            Dictionary of metrics from optimization, including:
            - 'loss': Training loss
            - 'num_gaussians': Current Gaussian count
            - 'frame_time': Processing time for this frame (seconds)
            - Additional engine-specific metrics
        """
        start_time = time.time()
        
        # Initialize on first frame
        self._maybe_init_scene(frame)
        
        # Apply rolling shutter correction
        rgb_corrected = self.rs_corrector.correct(
            frame.rgb,
            poses=frame.pose,
            timestamps=[frame.timestamp],
        )
        
        # Create corrected frame (preserving other attributes)
        if rgb_corrected is not frame.rgb:
            frame = Frame(
                idx=frame.idx,
                timestamp=frame.timestamp,
                rgb=rgb_corrected,
                depth=frame.depth,
                pose=frame.pose,
                intrinsics=frame.intrinsics,
                metadata=frame.metadata,
            )
        
        # Add frame to engine
        self.engine.add_frame(
            frame_id=frame.idx,
            rgb=frame.rgb,
            depth=frame.depth,
            pose_world_cam=frame.pose,
        )
        
        # Run optimization (skip during warmup)
        metrics: Dict[str, float] = {}
        
        if self.frame_count >= self.warmup_frames - 1:
            n_steps = self.steps_per_frame
            
            # Run more steps for early frames to establish good initialization
            if self.frame_count < self.warmup_frames + 5:
                n_steps = self.steps_per_frame * 2
            
            metrics = self.engine.optimize_step(n_steps)
        else:
            metrics = {
                'loss': 0.0,
                'num_gaussians': self.engine.get_num_gaussians(),
                'warmup': True,
            }
        
        # Record timing
        frame_time = time.time() - start_time
        metrics['frame_time'] = frame_time
        metrics['frame_idx'] = frame.idx
        
        self._frame_times.append(frame_time)
        self._total_time += frame_time
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.frame_count += 1
        
        # Render preview
        if self.render_every > 0 and self.frame_count % self.render_every == 0:
            self._render_preview(frame)
        
        # Save checkpoint
        if self.save_every > 0 and self.frame_count % self.save_every == 0:
            self._save_checkpoint()
        
        # Call user callbacks
        for callback in self._on_frame_callbacks:
            try:
                callback(frame, metrics)
            except Exception as e:
                logger.warning(f"Frame callback error: {e}")
        
        return metrics
    
    def run(
        self,
        max_frames: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on all frames from the source.
        
        Args:
            max_frames: Maximum number of frames to process.
                If None, processes all frames in the source.
                
            progress: If True, show progress bar (requires tqdm)
            
        Returns:
            Summary dictionary with:
            - 'num_frames': Total frames processed
            - 'total_time': Total processing time (seconds)
            - 'avg_frame_time': Average time per frame
            - 'final_num_gaussians': Final Gaussian count
            - 'final_loss': Last recorded loss
        """
        # Determine number of frames
        total_frames = len(self.frame_source)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Starting online GS pipeline for {total_frames} frames")
        logger.info(f"Steps per frame: {self.steps_per_frame}")
        logger.info(f"RS corrector: {self.rs_corrector.name}")
        
        # Setup progress bar
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(self.frame_source),
                    total=total_frames,
                    desc="Processing frames",
                )
            except ImportError:
                logger.info("tqdm not available, running without progress bar")
                iterator = enumerate(self.frame_source)
        else:
            iterator = enumerate(self.frame_source)
        
        # Process frames
        start_time = time.time()
        
        for idx, frame in iterator:
            if max_frames is not None and idx >= max_frames:
                break
            
            metrics = self.step(frame)
            
            # Update progress bar description
            if progress and 'tqdm' in str(type(iterator)):
                loss = metrics.get('loss', 0)
                n_gauss = metrics.get('num_gaussians', 0)
                iterator.set_postfix({
                    'loss': f"{loss:.4f}",
                    'gaussians': n_gauss,
                })
        
        total_time = time.time() - start_time
        
        # Compile summary
        summary = {
            'num_frames': self.frame_count,
            'total_time': total_time,
            'avg_frame_time': total_time / max(1, self.frame_count),
            'final_num_gaussians': self.engine.get_num_gaussians(),
            'final_loss': self.metrics_history[-1].get('loss', 0) if self.metrics_history else 0,
        }
        
        logger.info(f"Pipeline complete. Processed {self.frame_count} frames in {total_time:.1f}s")
        logger.info(f"Average frame time: {summary['avg_frame_time']*1000:.1f}ms")
        logger.info(f"Final Gaussian count: {summary['final_num_gaussians']}")
        
        return summary
    
    def _render_preview(self, frame: Frame) -> None:
        """Render and save a preview image with GT comparison."""
        # Skip rendering if we don't have enough Gaussians yet
        num_gaussians = self.engine.get_num_gaussians()
        if num_gaussians < 10000:
            logger.debug(f"Skipping render - only {num_gaussians} Gaussians")
            return
        
        try:
            import cv2
            import numpy as np
            
            rendered = self.engine.render_view(frame.pose, frame.image_size)
            
            # Check if render is mostly black (failed render)
            if rendered.mean() < 5:  # Almost black
                logger.warning(f"Render at frame {frame.idx} is mostly black, skipping save")
                return
            
            # Save directory
            render_dir = self.output_dir / "renders"
            render_dir.mkdir(exist_ok=True)
            
            # Save just the render
            render_path = render_dir / f"frame_{self.frame_count:06d}.png"
            cv2.imwrite(str(render_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            
            # Also save side-by-side comparison with GT
            gt = frame.get_rgb_uint8()
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            gt_labeled = gt.copy()
            rendered_labeled = rendered.copy()
            cv2.putText(gt_labeled, "Ground Truth", (10, 30), font, 0.8, (255, 255, 255), 2)
            cv2.putText(rendered_labeled, f"Rendered (frame {frame.idx})", (10, 30), font, 0.8, (255, 255, 255), 2)
            
            # Add metrics overlay
            if self.metrics_history:
                last_metrics = self.metrics_history[-1]
                loss = last_metrics.get('loss', 0)
                n_gauss = last_metrics.get('num_gaussians', 0)
                cv2.putText(rendered_labeled, f"Loss: {loss:.4f}", (10, 60), font, 0.6, (255, 255, 0), 1)
                cv2.putText(rendered_labeled, f"Gaussians: {n_gauss}", (10, 85), font, 0.6, (255, 255, 0), 1)
            
            comparison = np.concatenate([gt_labeled, rendered_labeled], axis=1)
            
            comparison_dir = self.output_dir / "comparisons"
            comparison_dir.mkdir(exist_ok=True)
            comparison_path = comparison_dir / f"comparison_{self.frame_count:06d}.png"
            cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            logger.debug(f"Saved render to {render_path}")
            
            # Call render callbacks
            for callback in self._on_render_callbacks:
                try:
                    callback(rendered, self.frame_count)
                except Exception as e:
                    logger.warning(f"Render callback error: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to render preview: {e}")
    
    def _save_checkpoint(self) -> None:
        """Save engine state checkpoint."""
        try:
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"checkpoint_{self.frame_count:06d}"
            self.engine.save_state(str(checkpoint_path))
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def on_frame(self, callback: Callable[[Frame, Dict[str, float]], None]) -> None:
        """
        Register a callback to be called after each frame is processed.
        
        Args:
            callback: Function taking (frame, metrics) arguments
        """
        self._on_frame_callbacks.append(callback)
    
    def on_render(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """
        Register a callback to be called after each render.
        
        Args:
            callback: Function taking (rendered_image, frame_count) arguments
        """
        self._on_render_callbacks.append(callback)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the optimization metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self.metrics_history:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for m in self.metrics_history:
            all_keys.update(m.keys())
        
        summary = {}
        for key in all_keys:
            values = [m.get(key, 0) for m in self.metrics_history if isinstance(m.get(key), (int, float))]
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        return summary
    
    def save_final(self, name: Optional[str] = None) -> str:
        """
        Save the final engine state.
        
        Args:
            name: Optional name for the save directory.
                If None, uses 'final'.
                
        Returns:
            Path to saved state
        """
        save_name = name or "final"
        save_path = self.output_dir / save_name
        
        self.engine.save_state(str(save_path))
        
        # Also save metrics history
        import json
        metrics_path = save_path / "metrics.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_history = []
        for m in self.metrics_history:
            serializable_history.append({
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in m.items()
            })
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Saved final state to {save_path}")
        return str(save_path)
    
    def reset(self) -> None:
        """Reset the pipeline to initial state."""
        self.engine.reset()
        self._initialized = False
        self._intrinsics = None
        self.frame_count = 0
        self.metrics_history.clear()
        self._frame_times.clear()
        self._total_time = 0.0
        
        logger.info("Pipeline reset")
