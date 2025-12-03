#!/usr/bin/env python3
"""
Live ArduPilot 3D Gaussian Splatting Demo
=========================================

Real-time 3DGS mapping using camera + ArduPilot pose from drone/vehicle.

Supports:
- Any OpenCV camera (USB, IP camera, RTSP)
- ArduPilot over MAVLink (serial, UDP, TCP)
- Mission Planner / QGroundControl connections
- SITL simulation

Usage:
    # Connect to Pixhawk over serial
    python scripts/demos/live_ardupilot_demo.py --mavlink /dev/ttyUSB0
    
    # Connect to SITL simulation
    python scripts/demos/live_ardupilot_demo.py --mavlink tcp:127.0.0.1:5762 --camera 0
    
    # Drone with RTSP camera
    python scripts/demos/live_ardupilot_demo.py \\
        --mavlink udpin:0.0.0.0:14550 \\
        --camera "rtsp://192.168.1.100:8554/main" \\
        --camera-pitch -45
    
    # Raspberry Pi with CSI camera
    python scripts/demos/live_ardupilot_demo.py \\
        --mavlink /dev/ttyAMA0 \\
        --camera "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
        
    # View pose data only (no mapping)
    python scripts/demos/live_ardupilot_demo.py --mavlink udpin:0.0.0.0:14550 --pose-only

Requirements:
    pip install pymavlink opencv-python numpy torch
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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


def draw_pose_overlay(img: np.ndarray, state, pose: np.ndarray, fps: float) -> np.ndarray:
    """Draw pose and status overlay on image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (320, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Status info
    y = 25
    line_height = 20
    
    texts = [
        f"FPS: {fps:.1f}",
        f"Armed: {'YES' if state.armed else 'NO'}",
        f"GPS Fix: {state.gps_fix} ({state.satellites} sats)",
        f"Battery: {state.battery_voltage:.1f}V ({state.battery_remaining}%)",
        f"Roll: {np.degrees(state.roll):.1f} deg",
        f"Pitch: {np.degrees(state.pitch):.1f} deg",
        f"Yaw: {np.degrees(state.yaw):.1f} deg",
    ]
    
    if state.position_ned is not None:
        texts.append(f"Pos NED: [{state.position_ned[0]:.2f}, {state.position_ned[1]:.2f}, {state.position_ned[2]:.2f}]")
    
    # GPS position
    if state.lat != 0 or state.lon != 0:
        texts.append(f"GPS: {state.lat:.6f}, {state.lon:.6f}")
    
    for text in texts:
        cv2.putText(img, text, (10, y), font, font_scale, color, 1)
        y += line_height
    
    return img


def draw_horizon(img: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """Draw artificial horizon indicator."""
    img = img.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Draw horizon line
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    
    # Horizon offset based on pitch
    pitch_offset = int(pitch_deg * 3)  # Scale factor for visualization
    
    # Calculate horizon line endpoints
    line_len = 100
    dx = int(line_len * np.cos(roll))
    dy = int(line_len * np.sin(roll))
    
    pt1 = (cx - dx, cy + pitch_offset - dy)
    pt2 = (cx + dx, cy + pitch_offset + dy)
    
    cv2.line(img, pt1, pt2, (0, 255, 255), 2)
    
    # Draw aircraft symbol
    cv2.line(img, (cx - 30, cy), (cx - 10, cy), (255, 255, 0), 2)
    cv2.line(img, (cx + 10, cy), (cx + 30, cy), (255, 255, 0), 2)
    cv2.circle(img, (cx, cy), 5, (255, 255, 0), 2)
    
    return img


def pose_monitor_mode(args):
    """Run in pose monitor mode (no mapping, just display pose)."""
    from src.pose.ardupilot_mavlink import ArduPilotPoseProvider
    
    logger.info("Starting pose monitor mode...")
    logger.info(f"Connecting to: {args.mavlink}")
    
    provider = ArduPilotPoseProvider(
        connection_string=args.mavlink,
    )
    
    if not provider.start():
        logger.error("Failed to connect to ArduPilot")
        return
    
    logger.info("Connected! Press 'q' to quit, 's' for snapshot")
    
    # Open camera if specified
    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.warning(f"Could not open camera {args.camera}, continuing without video")
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    fps_counter = []
    last_time = time.time()
    
    # Create window
    cv2.namedWindow("ArduPilot Pose Monitor", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            now = time.time()
            
            # Get state
            state = provider.get_state()
            pose = provider.get_pose()
            
            # Calculate FPS
            fps_counter.append(now)
            fps_counter = [t for t in fps_counter if now - t < 1.0]
            fps = len(fps_counter)
            
            # Display
            if cap:
                ret, frame = cap.read()
                if ret:
                    frame = draw_pose_overlay(frame, state, pose, fps)
                    frame = draw_horizon(frame, state.roll, state.pitch)
                    cv2.imshow("ArduPilot Pose Monitor", frame)
            else:
                # Create a blank display with pose info
                display = np.zeros((400, 500, 3), dtype=np.uint8)
                display = draw_pose_overlay(display, state, pose, fps)
                cv2.imshow("ArduPilot Pose Monitor", display)
                
                # Also console output
                if now - last_time >= 0.5:
                    print(f"\rRoll: {np.degrees(state.roll):6.1f}° | "
                          f"Pitch: {np.degrees(state.pitch):6.1f}° | "
                          f"Yaw: {np.degrees(state.yaw):6.1f}° | "
                          f"GPS: {state.gps_fix} ({state.satellites} sats) | "
                          f"Armed: {state.armed}", end="", flush=True)
                    last_time = now
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') and cap:
                # Save snapshot
                filename = f"ardupilot_snapshot_{int(now)}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Saved: {filename}")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        provider.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()


def mapping_mode(args):
    """Run full 3DGS mapping mode."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        return
    
    from src.pipeline.ardupilot_source import ArduPilotSource
    
    # Check for CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, mapping will be slow")
    
    logger.info("Starting mapping mode...")
    
    # Create source
    source = ArduPilotSource(
        camera_source=args.camera if args.camera is not None else 0,
        mavlink_connection=args.mavlink,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fov_deg=args.fov,
        camera_pitch_deg=args.camera_pitch,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        use_depth_estimation=not args.no_depth,
    )
    
    # Try to initialize pipeline engine
    engine = None
    try:
        from src.engines.gsplat_engine import GSplatEngine, GSplatConfig
        
        config = GSplatConfig(
            max_gaussians=args.max_gaussians,
            init_opacity=0.5,
            learning_rate_position=0.001,
            learning_rate_opacity=0.01,
            sh_degree=2,
        )
        
        engine = GSplatEngine(config)
        engine.initialize_from_intrinsics(source.get_intrinsics())
        logger.info("GSplat engine initialized")
        
    except ImportError as e:
        logger.warning(f"GSplat engine not available: {e}")
        logger.info("Running in preview mode (no 3DGS)")
    except Exception as e:
        logger.warning(f"Failed to init engine: {e}")
    
    # Main loop
    cv2.namedWindow("ArduPilot 3DGS Mapping", cv2.WINDOW_NORMAL)
    
    fps_counter = []
    output_dir = Path(args.output) if args.output else Path("output/ardupilot_mapping")
    
    try:
        for frame in source:
            now = time.time()
            
            # Step pipeline if available
            render = None
            if engine:
                engine.step(frame)
                render = engine.render(frame.pose, frame.intrinsics)
            
            # Calculate FPS
            fps_counter.append(now)
            fps_counter = [t for t in fps_counter if now - t < 1.0]
            fps = len(fps_counter)
            
            # Create display
            rgb_uint8 = frame.get_rgb_uint8()
            rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            
            if render is not None:
                render_uint8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
                render_bgr = cv2.cvtColor(render_uint8, cv2.COLOR_RGB2BGR)
                display = np.hstack([rgb_bgr, render_bgr])
            else:
                display = rgb_bgr
            
            # Draw pose overlay if ArduPilot connected
            state = source.get_ardupilot_state()
            if state:
                display = draw_pose_overlay(display, state, frame.pose, fps)
            
            # Add info text
            num_gs = engine.num_gaussians if engine else 0
            info_text = f"Frame {frame.idx} | FPS: {fps:.1f} | Gaussians: {num_gs:,}"
            cv2.putText(display, info_text, (10, display.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # MAVLink status
            if source.is_mavlink_connected():
                status_text = "MAVLink: Connected"
                status_color = (0, 255, 0)
            else:
                status_text = "MAVLink: Fallback VO"
                status_color = (0, 165, 255)
            cv2.putText(display, status_text, (10, display.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            cv2.imshow("ArduPilot 3DGS Mapping", display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') and engine:
                # Save checkpoint
                output_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = output_dir / f"checkpoint_{frame.idx}.ply"
                engine.save(ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
            elif key == ord('r') and engine:
                # Reset
                engine.reset()
                logger.info("Reset engine")
                
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        source.stop()
        cv2.destroyAllWindows()
        
        # Final save
        if engine and args.output:
            output_dir.mkdir(parents=True, exist_ok=True)
            final_path = output_dir / "final.ply"
            engine.save(final_path)
            logger.info(f"Saved final model: {final_path}")


def test_connection(args):
    """Test ArduPilot connection."""
    from src.pose.ardupilot_mavlink import ArduPilotPoseProvider
    
    print(f"Testing connection to: {args.mavlink}")
    print("Waiting for heartbeat (timeout 30s)...")
    
    provider = ArduPilotPoseProvider(connection_string=args.mavlink)
    
    if provider.start():
        print("✓ Connection successful!")
        print()
        
        # Wait a bit for data
        time.sleep(2)
        
        state = provider.get_state()
        print("Vehicle Status:")
        print(f"  Armed: {state.armed}")
        print(f"  GPS Fix: {state.gps_fix} ({state.satellites} satellites)")
        print(f"  Battery: {state.battery_voltage:.1f}V ({state.battery_remaining}%)")
        print(f"  Attitude: Roll={np.degrees(state.roll):.1f}° Pitch={np.degrees(state.pitch):.1f}° Yaw={np.degrees(state.yaw):.1f}°")
        
        if state.position_ned is not None:
            print(f"  Position (NED): [{state.position_ned[0]:.2f}, {state.position_ned[1]:.2f}, {state.position_ned[2]:.2f}]")
        
        if state.lat != 0 or state.lon != 0:
            print(f"  GPS: {state.lat:.6f}, {state.lon:.6f}, {state.alt:.1f}m")
        
        provider.stop()
    else:
        print("✗ Connection failed!")
        print("Check:")
        print("  - Connection string is correct")
        print("  - ArduPilot is powered and running")
        print("  - Serial port permissions (Linux: sudo usermod -a -G dialout $USER)")


def main():
    parser = argparse.ArgumentParser(
        description="Live ArduPilot 3D Gaussian Splatting Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test connection
  python live_ardupilot_demo.py --test --mavlink udpin:0.0.0.0:14550

  # Pixhawk over USB (Linux)
  python live_ardupilot_demo.py --mavlink /dev/ttyUSB0 --camera 0

  # Pixhawk over USB (Windows)
  python live_ardupilot_demo.py --mavlink COM3 --camera 0

  # SITL simulation
  python live_ardupilot_demo.py --mavlink tcp:127.0.0.1:5762

  # Mission Planner / QGC connection
  python live_ardupilot_demo.py --mavlink udpin:0.0.0.0:14550

  # Drone with RTSP camera
  python live_ardupilot_demo.py --mavlink udpin:0.0.0.0:14550 \\
      --camera rtsp://192.168.1.100:8554/main

  # Pose monitor only (no 3DGS)
  python live_ardupilot_demo.py --mavlink udpin:0.0.0.0:14550 --pose-only
        """
    )
    
    # Connection arguments
    parser.add_argument("--mavlink", type=str, default="udpin:0.0.0.0:14550",
                       help="MAVLink connection string (default: udpin:0.0.0.0:14550)")
    parser.add_argument("--camera", type=str, default=None,
                       help="Camera source (index, URL, or GStreamer pipeline)")
    
    # Camera settings
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera FOV in degrees")
    parser.add_argument("--camera-pitch", type=float, default=-45.0,
                       help="Camera pitch angle in degrees (negative = looking down)")
    
    # Processing settings
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target processing FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--max-gaussians", type=int, default=100000, help="Maximum Gaussians")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation")
    
    # Output
    parser.add_argument("--output", type=str, default=None, help="Output directory for model")
    
    # Modes
    parser.add_argument("--pose-only", action="store_true",
                       help="Monitor pose only (no mapping)")
    parser.add_argument("--test", action="store_true",
                       help="Test ArduPilot connection and exit")
    
    args = parser.parse_args()
    
    # Auto-detect camera type
    if args.camera is not None:
        # Try to parse as int (webcam index)
        try:
            args.camera = int(args.camera)
        except ValueError:
            pass  # Keep as string (URL or GStreamer)
    
    # Run appropriate mode
    if args.test:
        test_connection(args)
    elif args.pose_only:
        pose_monitor_mode(args)
    else:
        mapping_mode(args)


if __name__ == "__main__":
    main()
