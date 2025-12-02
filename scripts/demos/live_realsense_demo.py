#!/usr/bin/env python3
"""
Live RealSense 3D Gaussian Splatting Demo

Real-time 3DGS mapping using Intel RealSense D400-series cameras.
Supports D415, D435, D435i (with IMU), D455, etc.

Features:
- Real-time RGB-D capture with metric depth
- Visual odometry pose estimation (ORB/Flow)
- Optional IMU data for D435i/D455
- Live rendering comparison
- Background threaded capture for low latency

Usage:
    # Basic usage (first available camera)
    python scripts/demos/live_realsense_demo.py
    
    # With specific settings
    python scripts/demos/live_realsense_demo.py --width 1280 --height 720 --fps 30
    
    # List available cameras
    python scripts/demos/live_realsense_demo.py --list-devices
    
    # Use specific camera by serial number
    python scripts/demos/live_realsense_demo.py --serial 12345678
"""

import os
# Set CUDA arch for gsplat JIT compilation
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')
os.environ.setdefault('MAX_JOBS', '2')
if 'CONDA_PREFIX' in os.environ:
    cuda_include = os.path.join(os.environ['CONDA_PREFIX'], 'targets', 'x86_64-linux', 'include')
    if os.path.exists(cuda_include):
        current_cpath = os.environ.get('CPATH', '')
        if cuda_include not in current_cpath:
            os.environ['CPATH'] = f"{cuda_include}:{current_cpath}" if current_cpath else cuda_include

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def draw_metrics_overlay(img, metrics):
    """Draw metrics overlay on image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    line_height = 25
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (300, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Metrics
    texts = [
        (f"FPS: {metrics.get('fps', 0):.1f}", (0, 255, 0)),
        (f"Frame: {metrics.get('frame_idx', 0)}", (255, 255, 255)),
        (f"Gaussians: {metrics.get('num_gaussians', 0):,}", (0, 200, 255)),
        (f"Loss: {metrics.get('loss', 0):.4f}", (255, 200, 0)),
        (f"PSNR: {metrics.get('psnr', 0):.1f} dB", (255, 200, 0)),
        (f"GPU: {metrics.get('gpu_mb', 0):.0f} MB", (255, 150, 0)),
        (f"Tracking: {metrics.get('tracking', 'ok')}", (0, 255, 0) if metrics.get('tracking') == 'ok' else (0, 0, 255)),
    ]
    
    for text, color in texts:
        cv2.putText(img, text, (15, y), font, 0.6, color, 1, cv2.LINE_AA)
        y += line_height
    
    return img


def draw_imu_overlay(img, accel, gyro, x_offset=320):
    """Draw IMU data visualization."""
    if accel is None or gyro is None:
        return img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x_offset, 5), (x_offset + 200, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # IMU data
    y = 25
    cv2.putText(img, "IMU Data", (x_offset + 10, y), font, 0.5, (100, 255, 100), 1)
    y += 20
    cv2.putText(img, f"Accel: [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}]", 
                (x_offset + 10, y), font, 0.4, (200, 200, 200), 1)
    y += 18
    cv2.putText(img, f"Gyro: [{gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f}]", 
                (x_offset + 10, y), font, 0.4, (200, 200, 200), 1)
    
    # Gravity indicator (from accelerometer)
    y += 25
    cv2.putText(img, "Gravity:", (x_offset + 10, y), font, 0.4, (150, 150, 150), 1)
    
    # Draw gravity arrow
    center = (x_offset + 150, y - 5)
    accel_norm = np.linalg.norm(accel)
    if accel_norm > 0.1:
        accel_dir = accel / accel_norm
        arrow_len = 30
        end_point = (
            int(center[0] + accel_dir[0] * arrow_len),
            int(center[1] + accel_dir[1] * arrow_len)
        )
        cv2.arrowedLine(img, center, end_point, (0, 255, 255), 2)
    
    return img


def draw_depth_colormap(depth, min_depth=0.1, max_depth=5.0):
    """Convert depth to colormap visualization."""
    # Normalize to 0-255
    depth_viz = depth.copy()
    depth_viz = np.clip(depth_viz, min_depth, max_depth)
    depth_viz = ((depth_viz - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
    
    # Mark invalid depth as black
    depth_colored[depth < min_depth] = [0, 0, 0]
    
    return depth_colored


def main():
    parser = argparse.ArgumentParser(description="Live RealSense 3DGS Demo")
    
    # Camera settings
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial number")
    parser.add_argument("--list-devices", action="store_true", help="List available cameras and exit")
    
    # Processing settings
    parser.add_argument("--pose-model", type=str, default="orb", 
                       choices=["orb", "robust_flow", "sift"],
                       help="Pose estimation model")
    parser.add_argument("--engine", type=str, default="gsplat",
                       choices=["graphdeco", "gsplat"],
                       help="3DGS engine backend")
    parser.add_argument("--steps-per-frame", type=int, default=5, 
                       help="Optimization steps per frame")
    parser.add_argument("--max-frames", type=int, default=None, 
                       help="Maximum frames to process")
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Process every Nth frame (1=all, 2=every other, etc.)")
    
    # Output settings
    parser.add_argument("--output", type=str, default="./output/realsense_demo",
                       help="Output directory")
    parser.add_argument("--save-video", action="store_true", 
                       help="Save output video")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable live display (headless mode)")
    
    # Quality presets
    parser.add_argument("--quality", choices=["fast", "balanced", "quality"], 
                       default="fast", help="Quality preset")
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        from src.pipeline.frames import RealSenseSource
        
        print("\nAvailable RealSense devices:")
        print("-" * 50)
        devices = RealSenseSource.list_devices()
        if not devices:
            print("No RealSense devices found!")
            print("\nTroubleshooting:")
            print("  1. Check USB connection (use USB 3.0 port)")
            print("  2. Install librealsense2: sudo apt install librealsense2-*")
            print("  3. Add udev rules for RealSense")
        else:
            for i, dev in enumerate(devices):
                print(f"  [{i}] {dev['name']}")
                print(f"      Serial: {dev['serial']}")
                print(f"      Firmware: {dev['firmware']}")
        print()
        return
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quality presets
    presets = {
        'fast': {
            'sh_degree': 2,
            'steps_mult': 1,
            'depth_subsample': 4,
            'max_gaussians': 80000,
            'add_every': 3,
        },
        'balanced': {
            'sh_degree': 3,
            'steps_mult': 2,
            'depth_subsample': 3,
            'max_gaussians': 120000,
            'add_every': 5,
        },
        'quality': {
            'sh_degree': 3,
            'steps_mult': 3,
            'depth_subsample': 2,
            'max_gaussians': 150000,
            'add_every': 7,
        }
    }
    preset = presets[args.quality]
    
    # Initialize RealSense source
    logger.info("Initializing RealSense camera...")
    from src.pipeline.frames import RealSenseSource
    
    try:
        source = RealSenseSource(
            serial_number=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            pose_model=args.pose_model,
            max_frames=args.max_frames,
            enable_imu=True,
            align_depth=True,
            exposure_auto=False,  # Fixed exposure for consistent tracking
        )
    except Exception as e:
        logger.error(f"Failed to initialize RealSense: {e}")
        logger.info("\nRun with --list-devices to see available cameras")
        return 1
    
    intrinsics = source.get_intrinsics()
    width = int(intrinsics['width'])
    height = int(intrinsics['height'])
    has_imu = source.has_imu()
    
    logger.info(f"Camera: {width}x{height} @ {args.fps} FPS")
    logger.info(f"IMU available: {has_imu}")
    logger.info(f"Pose model: {args.pose_model}")
    
    # Initialize 3DGS engine
    logger.info(f"Initializing {args.engine} engine...")
    from src.engines import get_engine
    
    try:
        engine = get_engine(args.engine)
    except Exception as e:
        logger.warning(f"Engine {args.engine} not available: {e}")
        logger.info("Falling back to graphdeco engine")
        engine = get_engine('graphdeco')
    
    # Engine configuration
    config = {
        'sh_degree': preset['sh_degree'],
        'white_background': False,
        'densify_grad_threshold': 0.0002,
        'densify_from_iter': 50,
        'densify_until_iter': 50000,
        'densification_interval': 100,
        'opacity_reset_interval': 500,
        'percent_dense': 0.01,
        'lambda_dssim': 0.2,
        'position_lr_init': 0.0005,
        'position_lr_final': 0.000005,
        'feature_lr': 0.005,
        'opacity_lr': 0.05,
        'scaling_lr': 0.01,
        'rotation_lr': 0.002,
        'recency_weight': 0.5,  # Balance between recent and all frames
        'depth_subsample': preset['depth_subsample'],
        'add_gaussians_per_frame': True,
        'add_gaussians_every': preset['add_every'],
        'add_gaussians_subsample': 8,
        'max_gaussians': preset['max_gaussians'],
    }
    
    engine.initialize_scene(intrinsics, config)
    steps_per_frame = args.steps_per_frame * preset['steps_mult']
    
    # Video writer
    video_writer = None
    if args.save_video:
        video_path = output_dir / "realsense_live.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 15, (width * 3, height))
        logger.info(f"Recording to: {video_path}")
    
    # Main loop
    logger.info("=" * 60)
    logger.info("Starting live RealSense 3DGS demo")
    logger.info(f"  Quality: {args.quality}")
    logger.info(f"  Engine: {args.engine}")
    logger.info(f"  Steps/frame: {steps_per_frame}")
    logger.info("  Press 'q' to quit, 's' to save snapshot")
    logger.info("=" * 60)
    
    frame_count = 0
    total_time = 0
    fps_history = []
    
    try:
        for frame in source:
            frame_start = time.time()
            
            # Skip frames if requested
            if args.skip_frames > 1 and frame.idx % args.skip_frames != 0:
                continue
            
            frame_count += 1
            
            # Get data
            rgb = frame.get_rgb_uint8()
            depth = frame.depth
            pose = frame.pose
            tracking_status = frame.metadata.get('tracking_status', 'unknown')
            
            # Get IMU data
            accel = frame.metadata.get('accel')
            gyro = frame.metadata.get('gyro')
            
            # Add frame to engine
            engine.add_frame(frame.idx, rgb, depth, pose)
            
            # Optimize
            metrics = engine.optimize_step(steps_per_frame)
            
            loss = metrics.get('loss', 0)
            num_gaussians = metrics.get('num_gaussians', 0)
            
            # Render current view
            rendered = engine.render_view(pose, (width, height))
            
            # Calculate timing
            elapsed = time.time() - frame_start
            total_time += elapsed
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(current_fps)
            avg_fps = sum(fps_history[-30:]) / min(len(fps_history), 30)
            
            # Calculate PSNR
            mse = np.mean((rgb.astype(float) - rendered.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / (mse + 1e-10)) if mse > 0 else 40
            
            # Create visualization
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            depth_viz = draw_depth_colormap(depth)
            
            # Draw metrics
            display_metrics = {
                'fps': current_fps,
                'frame_idx': frame_count,
                'num_gaussians': num_gaussians,
                'loss': loss,
                'psnr': psnr,
                'gpu_mb': get_gpu_memory_mb(),
                'tracking': tracking_status,
            }
            
            rgb_bgr = draw_metrics_overlay(rgb_bgr, display_metrics)
            
            # Draw IMU overlay if available
            if has_imu and accel is not None:
                rgb_bgr = draw_imu_overlay(rgb_bgr, accel, gyro)
            
            # Labels
            cv2.putText(rgb_bgr, "INPUT", (width - 80, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(rendered_bgr, "3DGS RENDER", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_viz, "DEPTH", (width - 80, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine views
            display = np.concatenate([rgb_bgr, rendered_bgr, depth_viz], axis=1)
            
            # Save video frame
            if video_writer:
                video_writer.write(display)
            
            # Display
            if not args.no_display:
                cv2.imshow('RealSense 3DGS', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save snapshot
                    snap_path = output_dir / f"snapshot_{frame_count:05d}.png"
                    cv2.imwrite(str(snap_path), display)
                    logger.info(f"Saved snapshot: {snap_path}")
            
            # Log progress
            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}: {num_gaussians:,} Gaussians, "
                           f"loss={loss:.4f}, PSNR={psnr:.1f}dB, FPS={current_fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        source.stop()
        
        if video_writer:
            video_writer.release()
        
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Final stats
        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info("=" * 60)
        logger.info("Session complete!")
        logger.info(f"  Frames processed: {frame_count}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Average FPS: {avg_fps:.2f}")
        logger.info(f"  Final Gaussians: {engine.get_num_gaussians():,}")
        logger.info("=" * 60)
        
        # Save final model
        if engine.get_num_gaussians() > 0:
            final_path = output_dir / "final"
            engine.save_state(str(final_path))
            logger.info(f"Model saved to: {final_path}")
        
        # Save session info
        info_path = output_dir / "session_info.txt"
        with open(info_path, 'w') as f:
            f.write("RealSense 3DGS Session\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Camera: {source._device_name}\n")
            f.write(f"Serial: {source._device_serial}\n")
            f.write(f"Resolution: {width}x{height} @ {args.fps} FPS\n")
            f.write(f"IMU enabled: {has_imu}\n\n")
            f.write(f"Engine: {args.engine}\n")
            f.write(f"Pose model: {args.pose_model}\n")
            f.write(f"Quality: {args.quality}\n\n")
            f.write(f"Frames: {frame_count}\n")
            f.write(f"Time: {total_time:.1f}s\n")
            f.write(f"Avg FPS: {avg_fps:.2f}\n")
            f.write(f"Final Gaussians: {engine.get_num_gaussians():,}\n")
        
        logger.info(f"Session info saved to: {info_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
