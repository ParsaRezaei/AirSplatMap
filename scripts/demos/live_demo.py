#!/usr/bin/env python3
"""
Live 3D Gaussian Splatting Demo

Processes a video file or webcam feed in real-time, building a 3DGS scene
and rendering from the current viewpoint.

Usage:
    # From video file
    python scripts/live_demo.py --video path/to/video.mp4
    
    # From webcam
    python scripts/live_demo.py --webcam
    
    # From webcam with specific camera index
    python scripts/live_demo.py --webcam --camera-id 0
"""

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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def estimate_pose_from_optical_flow(prev_gray, curr_gray, prev_pose, K):
    """
    Estimate camera pose change using optical flow.
    This is a simple approximation - real systems use full SLAM.
    """
    # Detect features
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    if p0 is None or len(p0) < 10:
        return prev_pose
    
    # Calculate optical flow
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    
    if p1 is None:
        return prev_pose
    
    # Select good points
    good_old = p0[status == 1]
    good_new = p1[status == 1]
    
    if len(good_old) < 8:
        return prev_pose
    
    # Estimate essential matrix
    E, mask = cv2.findEssentialMat(good_new, good_old, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None:
        return prev_pose
    
    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, good_new, good_old, K)
    
    # Build transformation matrix (camera motion)
    delta_pose = np.eye(4)
    delta_pose[:3, :3] = R
    delta_pose[:3, 3] = t.flatten() * 0.05  # Scale factor for translation
    
    # Apply to previous pose
    new_pose = prev_pose @ delta_pose
    
    return new_pose


def create_depth_estimator():
    """Create a simple depth estimator using Depth-Anything if available."""
    try:
        # Try to import Depth-Anything
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Depth-Anything-3"))
        from src.depth_anything_3 import DepthAnything3
        
        logger.info("Loading Depth-Anything-3 model...")
        model = DepthAnything3.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        def estimate_depth(rgb):
            with torch.no_grad():
                # Preprocess
                img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img = img.unsqueeze(0)
                if torch.cuda.is_available():
                    img = img.cuda()
                
                # Predict
                depth = model(img)
                depth = depth.squeeze().cpu().numpy()
                
                # Scale to metric (approximate)
                depth = depth * 5.0  # Rough scaling
                
                return depth
        
        logger.info("Depth-Anything-3 loaded successfully")
        return estimate_depth
        
    except Exception as e:
        logger.warning(f"Could not load Depth-Anything: {e}")
        logger.warning("Using constant depth (quality will be limited)")
        
        def constant_depth(rgb):
            h, w = rgb.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 2.0  # 2 meters
        
        return constant_depth


def main():
    parser = argparse.ArgumentParser(description="Live 3DGS Demo")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every N frames")
    parser.add_argument("--steps-per-frame", type=int, default=10, help="Optimization steps per frame")
    parser.add_argument("--output", type=str, default="./output/live_demo", help="Output directory")
    parser.add_argument("--no-display", action="store_true", help="Don't show live window")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    args = parser.parse_args()
    
    if not args.video and not args.webcam:
        logger.error("Please specify --video or --webcam")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video source
    if args.webcam:
        logger.info(f"Opening webcam {args.camera_id}...")
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    else:
        logger.info(f"Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        sys.exit(1)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    logger.info(f"Video: {width}x{height} @ {fps} FPS")
    
    # Setup intrinsics (approximate for webcam)
    # For better results, calibrate your camera
    fx = width * 0.8  # Approximate focal length
    fy = width * 0.8
    cx = width / 2
    cy = height / 2
    
    intrinsics = {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'width': width, 'height': height
    }
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    logger.info(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Create depth estimator
    estimate_depth = create_depth_estimator()
    
    # Initialize 3DGS engine
    logger.info("Initializing 3DGS engine...")
    from src.engines import GraphdecoEngine
    
    engine = GraphdecoEngine()
    
    config = {
        'sh_degree': 2,  # Lower for speed
        'white_background': False,
        'densify_grad_threshold': 0.0005,
        'densify_from_iter': 50,
        'densify_until_iter': 50000,
        'densification_interval': 50,
        'opacity_reset_interval': 500,
        'percent_dense': 0.01,
        'lambda_dssim': 0.2,
        'position_lr_init': 0.0005,
        'position_lr_final': 0.00001,
        'feature_lr': 0.005,
        'opacity_lr': 0.05,
        'scaling_lr': 0.01,
        'rotation_lr': 0.002,
        'recency_weight': 0.8,  # High for live
        'depth_subsample': 4,
        'add_gaussians_per_frame': True,
        'add_gaussians_every': 5,
        'add_gaussians_subsample': 16,
        'max_gaussians': 100000,
    }
    
    engine.initialize_scene(intrinsics, config)
    
    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = output_dir / "output.mp4"
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps / args.skip_frames, (width * 2, height))
        logger.info(f"Saving video to: {video_path}")
    
    # Main loop
    frame_idx = 0
    prev_gray = None
    current_pose = np.eye(4)  # Start at identity
    
    logger.info("=" * 60)
    logger.info("Starting live 3DGS demo")
    logger.info("Press 'q' to quit, 's' to save snapshot")
    logger.info("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    logger.info("End of video")
                    break
                else:
                    continue
            
            frame_idx += 1
            
            # Skip frames for speed
            if frame_idx % args.skip_frames != 0:
                continue
            
            start_time = time.time()
            
            # Convert to grayscale for pose estimation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Estimate pose from optical flow
            if prev_gray is not None:
                current_pose = estimate_pose_from_optical_flow(prev_gray, gray, current_pose, K)
            
            prev_gray = gray.copy()
            
            # Estimate depth
            depth = estimate_depth(frame)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add frame to engine
            engine.add_frame(frame_idx, rgb, depth, current_pose)
            
            # Optimize
            metrics = engine.optimize_step(args.steps_per_frame)
            loss = metrics.get('loss', 0)
            num_gaussians = metrics.get('num_gaussians', 0)
            
            # Render current view
            rendered = engine.render_view(current_pose, (width, height))
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            
            # Create side-by-side display
            display = np.concatenate([frame, rendered_bgr], axis=1)
            
            # Add text overlay
            cv2.putText(display, f"Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"3DGS Render", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Frame: {frame_idx}", (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Gaussians: {num_gaussians}", (10, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Loss: {loss:.4f}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"FPS: {current_fps:.1f}", (width + 10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save to video
            if video_writer:
                video_writer.write(display)
            
            # Display
            if not args.no_display:
                cv2.imshow("Live 3DGS Demo", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    snapshot_path = output_dir / f"snapshot_{frame_idx:06d}.png"
                    cv2.imwrite(str(snapshot_path), display)
                    logger.info(f"Saved snapshot: {snapshot_path}")
            
            # Log progress
            if frame_idx % 50 == 0:
                logger.info(f"Frame {frame_idx}: {num_gaussians} Gaussians, loss={loss:.4f}, FPS={current_fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Save final state
        logger.info("Saving final state...")
        final_path = output_dir / "final"
        engine.save_state(str(final_path))
        logger.info(f"Saved to: {final_path}")
        
        logger.info("Done!")


if __name__ == "__main__":
    main()
