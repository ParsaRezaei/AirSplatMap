#!/usr/bin/env python3
"""
Online 3DGS Pipeline Demo - TUM RGB-D with Graphdeco Engine
============================================================

This script demonstrates the online 3D Gaussian Splatting pipeline using:
- GraphdecoEngine: Wrapping the original Graphdeco 3DGS implementation
- TumRGBDSource: Reading from TUM RGB-D benchmark datasets

Usage:
    python scripts/online_tum_graphdeco.py [options]

Options:
    --dataset-root PATH     Path to datasets directory (default: ../datasets)
    --sequence NAME         TUM sequence name (e.g., rgbd_dataset_freiburg1_desk)
    --max-frames N          Maximum frames to process (default: 200)
    --steps-per-frame N     Optimization steps per frame (default: 5)
    --render-every N        Render preview every N frames (default: 50)
    --output-dir PATH       Output directory (default: ./output/online_gs)
    --verbose               Enable verbose logging

Example:
    python scripts/online_tum_graphdeco.py --max-frames 100 --render-every 20
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )


def find_dataset_root() -> Path:
    """
    Find the datasets directory.
    
    Searches:
    1. ../datasets (relative to AirSplatMap)
    2. ./datasets (inside AirSplatMap)
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    candidates = [
        project_root.parent / "datasets",  # ../datasets
        project_root / "datasets",          # ./datasets
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Return default even if doesn't exist (will give helpful error later)
    return project_root.parent / "datasets"


def main():
    parser = argparse.ArgumentParser(
        description="Online 3DGS Pipeline Demo with TUM RGB-D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Path to datasets directory',
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='TUM sequence name (e.g., rgbd_dataset_freiburg1_desk)',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=200,
        help='Maximum frames to process (default: 200)',
    )
    parser.add_argument(
        '--steps-per-frame',
        type=int,
        default=5,
        help='Optimization steps per frame (default: 5)',
    )
    parser.add_argument(
        '--render-every',
        type=int,
        default=50,
        help='Render preview every N frames (0 to disable, default: 50)',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=0,
        help='Save checkpoint every N frames (0 to disable)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (auto-generated if not specified)',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Custom name for this run (used in output directory)',
    )
    parser.add_argument(
        '--sh-degree',
        type=int,
        default=3,
        help='Spherical harmonics degree (default: 3)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging',
    )
    parser.add_argument(
        '--quality',
        type=str,
        choices=['fast', 'balanced', 'quality'],
        default='balanced',
        help='Quality preset: fast (quick preview), balanced (default), quality (best results)',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Find dataset root
    dataset_root = Path(args.dataset_root) if args.dataset_root else find_dataset_root()
    logger.info(f"Dataset root: {dataset_root}")
    
    # Import pipeline components
    try:
        from src.engines import GraphdecoEngine
        from src.pipeline import OnlineGSPipeline
        from src.pipeline.frames import TumRGBDSource
    except ImportError as e:
        logger.error(f"Failed to import pipeline components: {e}")
        logger.error("Make sure you're running from the AirSplatMap directory")
        sys.exit(1)
    
    # Create frame source
    logger.info("Setting up TUM RGB-D source...")
    try:
        source = TumRGBDSource(
            dataset_root=str(dataset_root),
            sequence=args.sequence,
        )
        
        if len(source) == 0:
            logger.error("No frames found in dataset!")
            logger.error(f"Searched in: {dataset_root}")
            logger.error("Please ensure you have a TUM RGB-D dataset with:")
            logger.error("  - rgb/ directory with images")
            logger.error("  - rgb.txt with timestamps")
            logger.error("  - groundtruth.txt with poses")
            sys.exit(1)
        
        logger.info(f"Found {len(source)} frames in dataset")
        
        # Extract sequence name for output naming
        seq_name = args.sequence or source._dataset_path.name if source._dataset_path else "unknown"
        
    except Exception as e:
        logger.error(f"Failed to create frame source: {e}")
        sys.exit(1)
    
    # Create engine
    logger.info("Initializing Graphdeco engine...")
    try:
        engine = GraphdecoEngine()
    except Exception as e:
        logger.error(f"Failed to create Graphdeco engine: {e}")
        logger.error("Make sure the gaussian-splatting repo is available and CUDA extensions are compiled")
        sys.exit(1)
    
    # Generate meaningful output directory name
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.name:
            output_name = args.name
        else:
            # Format: tum_<sequence>_<frames>f_<steps>s_<timestamp>
            seq_short = seq_name.replace("rgbd_dataset_", "").replace("freiburg", "fr")
            output_name = f"tum_{seq_short}_{args.max_frames}f_{args.steps_per_frame}s_{timestamp}"
        output_dir = f"./output/{output_name}"
    else:
        output_dir = args.output_dir
    
    # Engine configuration - tuned for online mapping
    # Quality presets affect key parameters
    quality_configs = {
        'fast': {
            'steps_multiplier': 1,
            'depth_subsample': 4,  # Sparser points
            'add_gaussians_every': 15,
            'add_gaussians_subsample': 16,
            'densification_interval': 100,
            'densify_grad_threshold': 0.0002,
            'position_lr_mult': 1.5,
            'max_gaussians': 150000,
        },
        'balanced': {
            'steps_multiplier': 2,
            'depth_subsample': 2,  # Good density
            'add_gaussians_every': 8,
            'add_gaussians_subsample': 10,
            'densification_interval': 50,
            'densify_grad_threshold': 0.00008,  # More aggressive splitting
            'position_lr_mult': 2.5,  # Faster convergence
            'max_gaussians': 350000,
        },
        'quality': {
            'steps_multiplier': 3,
            'depth_subsample': 2,
            'add_gaussians_every': 6,
            'add_gaussians_subsample': 6,
            'densification_interval': 40,
            'densify_grad_threshold': 0.00006,
            'position_lr_mult': 3.0,
            'max_gaussians': 500000,
        },
    }
    
    qc = quality_configs[args.quality]
    actual_steps = args.steps_per_frame * qc['steps_multiplier']
    
    config = {
        'sh_degree': args.sh_degree,
        'white_background': False,
        # Densification settings
        'densify_grad_threshold': qc['densify_grad_threshold'],
        'densify_from_iter': 100,  # Start after some convergence
        'densify_until_iter': 100000,
        'densification_interval': qc['densification_interval'],
        'opacity_reset_interval': 1000,
        'percent_dense': 0.01,
        # Loss weighting
        'lambda_dssim': 0.2,
        # Learning rates - higher for faster online convergence
        'position_lr_init': 0.00016 * qc['position_lr_mult'],
        'position_lr_final': 1.6e-06 * qc['position_lr_mult'],
        'feature_lr': 0.0025 * qc['position_lr_mult'],
        'opacity_lr': 0.05,
        'scaling_lr': 0.005 * qc['position_lr_mult'],
        'rotation_lr': 0.001 * qc['position_lr_mult'],
        # For online mapping, favor recent frames
        'recency_weight': 0.7,
        # Depth-based Gaussian addition
        'depth_subsample': qc['depth_subsample'],
        'add_gaussians_per_frame': True,
        'add_gaussians_every': qc['add_gaussians_every'],
        'add_gaussians_subsample': qc['add_gaussians_subsample'],
        # Memory management
        'max_gaussians': qc['max_gaussians'],
    }
    
    logger.info(f"Quality preset: {args.quality}")
    logger.info(f"  Effective steps/frame: {actual_steps}")
    logger.info(f"  Depth subsample: {qc['depth_subsample']}")
    logger.info(f"  Densification interval: {qc['densification_interval']}")
    logger.info(f"  Max Gaussians: {qc['max_gaussians']}")
    
    # Create pipeline
    logger.info("Creating online pipeline...")
    pipeline = OnlineGSPipeline(
        engine=engine,
        frame_source=source,
        rs_corrector=None,  # No RS correction for now
        steps_per_frame=actual_steps,  # Use quality-adjusted steps
        warmup_frames=3,
        render_every=args.render_every,
        save_every=args.save_every,
        output_dir=output_dir,
        config=config,
    )
    
    # Optional: Add callback to log progress
    def on_frame_callback(frame, metrics):
        if frame.idx % 10 == 0:
            loss = metrics.get('loss', 0)
            n_gauss = metrics.get('num_gaussians', 0)
            frame_time = metrics.get('frame_time', 0) * 1000
            logger.debug(
                f"Frame {frame.idx}: loss={loss:.4f}, "
                f"gaussians={n_gauss}, time={frame_time:.1f}ms"
            )
    
    pipeline.on_frame(on_frame_callback)
    
    # Run pipeline
    logger.info("=" * 60)
    logger.info("Starting online 3DGS mapping")
    logger.info(f"  Dataset: {seq_name}")
    logger.info(f"  Max frames: {args.max_frames}")
    logger.info(f"  Steps per frame: {args.steps_per_frame}")
    logger.info(f"  Render every: {args.render_every}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)
    
    try:
        summary = pipeline.run(max_frames=args.max_frames, progress=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        summary = {
            'num_frames': pipeline.frame_count,
            'final_num_gaussians': engine.get_num_gaussians(),
        }
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Frames processed: {summary.get('num_frames', 0)}")
    logger.info(f"  Total time: {summary.get('total_time', 0):.1f}s")
    logger.info(f"  Avg frame time: {summary.get('avg_frame_time', 0)*1000:.1f}ms")
    logger.info(f"  Final Gaussians: {summary.get('final_num_gaussians', 0)}")
    logger.info(f"  Final loss: {summary.get('final_loss', 0):.4f}")
    logger.info("=" * 60)
    
    # Save final state
    logger.info("Saving final state...")
    save_path = pipeline.save_final()
    logger.info(f"Saved to: {save_path}")
    
    # Render a few novel views if we have enough Gaussians
    if engine.get_num_gaussians() > 100:
        logger.info("Rendering final views...")
        import numpy as np
        
        try:
            import cv2
            
            # Render from sampled training views throughout the sequence
            render_dir = Path(output_dir) / "final_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect only frames that were actually used for training (have depth)
            processed_frames = []
            for i, frame in enumerate(source):
                if i >= args.max_frames:
                    break
                # Only include frames that have depth (these were used for training)
                if frame.depth is not None:
                    processed_frames.append(frame)
            
            # Sample frames evenly throughout the sequence
            n_renders = min(10, len(processed_frames))
            if len(processed_frames) > 0:
                indices = np.linspace(0, len(processed_frames) - 1, n_renders, dtype=int)
                
                for idx in indices:
                    frame = processed_frames[idx]
                    rendered = engine.render_view(frame.pose, frame.image_size)
                    
                    # Save rendered and ground truth side by side
                    gt = frame.get_rgb_uint8()
                    
                    # Add labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    gt_labeled = gt.copy()
                    rendered_labeled = rendered.copy()
                    cv2.putText(gt_labeled, f"GT (frame {frame.idx})", (10, 30), font, 0.8, (255, 255, 255), 2)
                    cv2.putText(rendered_labeled, f"Rendered", (10, 30), font, 0.8, (255, 255, 255), 2)
                    
                    comparison = np.concatenate([gt_labeled, rendered_labeled], axis=1)
                    
                    save_path = render_dir / f"comparison_frame{frame.idx:04d}.png"
                    cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Final renders saved to: {render_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to render final views: {e}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
