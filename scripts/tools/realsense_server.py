#!/usr/bin/env python3
"""
Intel RealSense Live Server - Serves synchronized RGB, depth, pose, and IMU via HTTP.

Captures live data from RealSense D400-series cameras (D415, D435, D435i, D455)
and serves it via HTTP endpoints compatible with the AirSplatMap pipeline.

Endpoints:
  GET /          - MJPEG video stream
  GET /stream    - Same as / (alias)
  GET /frame     - Single synchronized bundle (image + pose + depth + intrinsics)
  GET /pose      - Current computed pose (4x4 matrix from visual odometry)
  GET /depth     - Current depth map (base64 PNG)
  GET /intrinsics - Camera intrinsics
  GET /imu       - Latest IMU data (accelerometer + gyroscope)
  GET /status    - Server status info
  GET /pointcloud - Current point cloud (subsampled, for visualization)

Usage:
  python realsense_server.py --port 8554 --fps 30
  python realsense_server.py --pose-model orb --width 640 --height 480
  
Then use in pipeline:
  LiveVideoSource("http://localhost:8554/stream", use_server_pose=True)
"""

import argparse
import base64
import cv2
import json
import numpy as np
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from queue import Queue, Empty

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


class RealSenseCapture:
    """
    Captures RGB-D and IMU from Intel RealSense with pose estimation.
    
    Features:
    - Hardware-synchronized RGB and depth
    - Metric depth directly from sensor
    - IMU streaming (on D435i, D455)
    - Visual odometry pose estimation
    - Thread-safe access to current frame data
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_depth: bool = True,
        enable_imu: bool = True,
        pose_model: str = 'orb',
        depth_min: float = 0.1,
        depth_max: float = 10.0,
        serial_number: Optional[str] = None,
        auto_exposure: bool = True,
        manual_exposure: Optional[int] = None,
    ):
        try:
            import pyrealsense2 as rs
            self._rs = rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 not installed. Install with: pip install pyrealsense2"
            )
        
        self._width = width
        self._height = height
        self._fps = fps
        self._align_depth = align_depth
        self._enable_imu = enable_imu
        self._pose_model_name = pose_model
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._serial_number = serial_number
        self._auto_exposure = auto_exposure
        self._manual_exposure = manual_exposure
        
        # Hardware reset the device first to clear any stale state
        ctx = rs.context()
        devices = list(ctx.devices)
        if not devices:
            raise RuntimeError("No RealSense devices found")
        
        print(f"Found {len(devices)} device(s), resetting...")
        devices[0].hardware_reset()
        time.sleep(3)  # Wait for device to come back
        
        # Initialize RealSense pipeline
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        
        # Select specific camera if serial provided
        if serial_number:
            self._config.enable_device(serial_number)
        
        # Configure streams
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Enable infrared stereo streams for stereo VIO
        self._config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # Left IR
        self._config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # Right IR
        
        # Enable IMU if requested
        self._has_imu = False
        if enable_imu:
            try:
                self._config.enable_stream(rs.stream.accel)
                self._config.enable_stream(rs.stream.gyro)
                self._has_imu = True
            except Exception as e:
                print(f"IMU not available: {e}")
        
        # Start pipeline
        try:
            self._profile = self._pipeline.start(self._config)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense: {e}")
        
        # Get device info
        device = self._profile.get_device()
        self._device_name = device.get_info(rs.camera_info.name)
        self._device_serial = device.get_info(rs.camera_info.serial_number)
        print(f"Connected to {self._device_name} (S/N: {self._device_serial})")
        
        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self._depth_scale}")
        
        # Get stereo baseline from depth sensor
        try:
            depth_baseline = depth_sensor.get_option(rs.option.stereo_baseline)
            self._stereo_baseline = depth_baseline / 1000.0  # Convert mm to meters
            print(f"Stereo baseline: {self._stereo_baseline:.4f}m")
        except:
            self._stereo_baseline = 0.05  # Default 50mm
            print(f"Stereo baseline: {self._stereo_baseline:.4f}m (default)")
        
        # Configure color sensor - start with auto exposure, lock after warmup
        self._color_sensor = device.first_color_sensor()
        self._auto_exposure = auto_exposure
        self._manual_exposure = manual_exposure
        self._exposure_locked = False
        
        if manual_exposure is not None:
            # Use manual exposure immediately
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            try:
                self._color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            except:
                pass
            self._color_sensor.set_option(rs.option.exposure, manual_exposure)
            self._exposure_locked = True
            print(f"Manual exposure: {manual_exposure}")
        else:
            # Start with auto-exposure, will lock after warmup
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("Auto-exposure: enabled (will lock after warmup)")
        
        # Get intrinsics from color stream
        color_stream = self._profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        self._intrinsics = {
            'fx': color_intrinsics.fx,
            'fy': color_intrinsics.fy,
            'cx': color_intrinsics.ppx,
            'cy': color_intrinsics.ppy,
            'width': color_intrinsics.width,
            'height': color_intrinsics.height,
            'baseline': self._stereo_baseline,
        }
        print(f"Intrinsics: fx={self._intrinsics['fx']:.1f}, fy={self._intrinsics['fy']:.1f}")
        
        # Get IR intrinsics for stereo VIO
        try:
            ir_stream = self._profile.get_stream(rs.stream.infrared, 1)
            ir_intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
            self._ir_intrinsics = {
                'fx': ir_intrinsics.fx,
                'fy': ir_intrinsics.fy,
                'cx': ir_intrinsics.ppx,
                'cy': ir_intrinsics.ppy,
                'width': ir_intrinsics.width,
                'height': ir_intrinsics.height,
                'baseline': self._stereo_baseline,
            }
            print(f"IR Intrinsics: fx={self._ir_intrinsics['fx']:.1f}, fy={self._ir_intrinsics['fy']:.1f}")
        except:
            self._ir_intrinsics = self._intrinsics.copy()
        
        # Setup alignment
        self._align = rs.align(rs.stream.color) if align_depth else None
        
        # Current frame data (thread-safe)
        self._lock = threading.Lock()
        self._current_rgb: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._current_depth_raw: Optional[np.ndarray] = None
        self._current_ir_left: Optional[np.ndarray] = None
        self._current_ir_right: Optional[np.ndarray] = None
        self._current_pose: Optional[np.ndarray] = np.eye(4)
        self._current_timestamp: float = 0.0
        self._current_accel: Optional[np.ndarray] = None
        self._current_gyro: Optional[np.ndarray] = None
        
        # Pose estimator
        self._pose_estimator = None
        self._vio = None
        self._init_pose_estimator()
        
        # Frame counter
        self._frame_count = 0
        self._start_time = time.time()
        
        # Warmup - grab frames to let auto-exposure stabilize
        print("Warming up camera...")
        for i in range(30):
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except:
                pass
        print("Camera ready!")
        
        # Capture thread
        self._running = True
        self._pipeline_ok = True  # Track pipeline state
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _init_pose_estimator(self):
        """Initialize visual odometry."""
        # Always init fallback first (in case VO fails mid-run)
        self._init_fallback_pose()
        self._stereo_vio = None
        self._vio = None
        
        # Use FilteredVO - better pose filtering for 3DGS
        try:
            from src.pose.filtered_vo import FilteredVO
            self._stereo_vio = FilteredVO(
                baseline=self._stereo_baseline,
                max_features=400,
                motion_threshold=0.8,  # Pixels - below this = stationary
                pose_filter_alpha=0.5,  # Moderate smoothing
                max_translation_per_frame=0.10,  # 10cm max per frame
                max_rotation_per_frame=0.12,  # ~7 degrees max per frame
                min_inliers=15,
            )
            self._stereo_vio.set_intrinsics_from_dict(self._ir_intrinsics)
            print(f"Initialized FilteredVO (baseline={self._stereo_baseline:.4f}m)")
            return
        except Exception as e:
            print(f"FilteredVO not available: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback to SimpleVO
        try:
            from src.pose.simple_vo import SimpleVO
            self._stereo_vio = SimpleVO(
                baseline=self._stereo_baseline,
                max_features=300,
                motion_threshold=1.0,
            )
            self._stereo_vio.set_intrinsics_from_dict(self._ir_intrinsics)
            print(f"Initialized SimpleVO (baseline={self._stereo_baseline:.4f}m)")
            return
        except Exception as e:
            print(f"SimpleVO not available: {e}")
        
        print("Using fallback ORB-based odometry")
    
    def _init_fallback_pose(self):
        """Initialize simple ORB-based pose estimation."""
        self._orb = cv2.ORB_create(nfeatures=2000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._K = np.array([
            [self._intrinsics['fx'], 0, self._intrinsics['cx']],
            [0, self._intrinsics['fy'], self._intrinsics['cy']],
            [0, 0, 1]
        ])
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None
        self._accumulated_pose = np.eye(4)
    
    def _estimate_pose_fallback(self, rgb: np.ndarray) -> np.ndarray:
        """Fallback ORB-based pose estimation."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        
        if self._prev_desc is None or desc is None or len(kp) < 10:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return self._accumulated_pose.copy()
        
        matches = self._bf.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        
        if len(matches) >= 8:
            pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
            
            E, mask = cv2.findEssentialMat(pts1, pts2, self._K, method=cv2.RANSAC, 
                                           prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self._K)
                
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = t.flatten() * 0.1  # Scale translation
                
                self._accumulated_pose = self._accumulated_pose @ np.linalg.inv(T_rel)
        
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        
        return self._accumulated_pose.copy()
    
    def _capture_loop(self):
        """Background capture thread."""
        rs = self._rs
        error_count = 0
        exposure_lock_frame = 90  # Lock exposure after 90 frames (~3 sec) - give more time to settle
        
        while self._running:
            if not self._pipeline_ok:
                time.sleep(0.1)
                continue
                
            try:
                # Wait for frames
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                error_count = 0  # Reset on success
                timestamp = time.time()
                
                # Lock exposure after warmup
                if not self._exposure_locked and self._frame_count == exposure_lock_frame:
                    try:
                        current_exp = self._color_sensor.get_option(rs.option.exposure)
                        # Ensure minimum exposure of 500 for decent brightness
                        final_exp = max(500, current_exp)
                        self._color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                        self._color_sensor.set_option(rs.option.exposure, final_exp)
                        self._exposure_locked = True
                        print(f"Exposure locked at {final_exp:.0f} (was {current_exp:.0f})")
                    except Exception as e:
                        print(f"Failed to lock exposure: {e}")
                
                # Process IMU
                accel = None
                gyro = None
                if self._has_imu:
                    for frame in frames:
                        if frame.is_motion_frame():
                            motion = frame.as_motion_frame()
                            motion_data = motion.get_motion_data()
                            
                            if frame.get_profile().stream_type() == rs.stream.accel:
                                accel = np.array([motion_data.x, motion_data.y, motion_data.z])
                            elif frame.get_profile().stream_type() == rs.stream.gyro:
                                gyro = np.array([motion_data.x, motion_data.y, motion_data.z])
                
                # Get IR frames for stereo VIO
                ir_left = None
                ir_right = None
                try:
                    ir_left_frame = frames.get_infrared_frame(1)
                    ir_right_frame = frames.get_infrared_frame(2)
                    if ir_left_frame and ir_right_frame:
                        ir_left = np.asanyarray(ir_left_frame.get_data())
                        ir_right = np.asanyarray(ir_right_frame.get_data())
                except:
                    pass
                
                # Align depth to color
                if self._align:
                    frames = self._align.process(frames)
                
                # Get frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy
                rgb = np.asanyarray(color_frame.get_data())  # BGR format
                depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
                
                # Convert depth to meters
                depth = depth_raw.astype(np.float32) * self._depth_scale
                depth[(depth < self._depth_min) | (depth > self._depth_max)] = 0
                
                # Estimate pose - prefer stereo VIO
                pose = None
                
                if hasattr(self, '_stereo_vio') and self._stereo_vio is not None and ir_left is not None:
                    try:
                        vio_result = self._stereo_vio.process(
                            ir_left, ir_right, timestamp, depth, accel, gyro
                        )
                        # Use raw pose for 3DGS (camera-to-world, OpenCV convention)
                        pose = self._stereo_vio.get_pose()
                        
                        # Debug: print pose occasionally
                        if self._frame_count % 60 == 0:  # Less frequent logging
                            t = pose[:3, 3]
                            status = vio_result.tracking_status
                            proc_ms = getattr(vio_result, 'processing_time_ms', 0)
                            print(f"Frame {self._frame_count}: pos=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}], status={status}, inliers={vio_result.num_inliers}, vio_ms={proc_ms:.1f}")
                    except Exception as e:
                        if self._frame_count < 5:
                            print(f"StereoVIO error: {e}")
                            import traceback
                            traceback.print_exc()
                
                if pose is None and self._vio is not None:
                    try:
                        rgb_for_vio = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        vio_result = self._vio.process(
                            rgb_for_vio, depth, timestamp, accel, gyro
                        )
                        # Use raw pose
                        pose = self._vio.get_pose()
                        
                        if self._frame_count % 30 == 0:
                            t = pose[:3, 3]
                            print(f"Frame {self._frame_count}: pos=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], mono inliers={vio_result.num_inliers}")
                    except Exception as e:
                        if self._frame_count < 5:
                            print(f"VIO error: {e}")
                
                if pose is None:
                    pose = self._estimate_pose_fallback(rgb)
                
                # Update current data (thread-safe)
                with self._lock:
                    self._current_rgb = rgb
                    self._current_depth = depth
                    self._current_depth_raw = depth_raw
                    self._current_ir_left = ir_left
                    self._current_ir_right = ir_right
                    self._current_pose = pose
                    self._current_timestamp = timestamp
                    self._current_accel = accel
                    self._current_gyro = gyro
                    self._frame_count += 1
                
            except Exception as e:
                if self._running:
                    error_count += 1
                    if error_count == 1 or error_count % 30 == 0:
                        print(f"Capture error ({error_count}): {e}")
                    time.sleep(0.033)  # Wait ~1 frame
    
    def get_synchronized_data(self) -> Dict:
        """Get all current frame data (synchronized)."""
        with self._lock:
            return {
                'rgb': self._current_rgb.copy() if self._current_rgb is not None else None,
                'depth': self._current_depth.copy() if self._current_depth is not None else None,
                'depth_raw': self._current_depth_raw.copy() if self._current_depth_raw is not None else None,
                'pose': self._current_pose.copy() if self._current_pose is not None else None,
                'timestamp': self._current_timestamp,
                'accel': self._current_accel.copy() if self._current_accel is not None else None,
                'gyro': self._current_gyro.copy() if self._current_gyro is not None else None,
                'frame_idx': self._frame_count,
            }
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current RGB frame (BGR format)."""
        with self._lock:
            return self._current_rgb.copy() if self._current_rgb is not None else None
    
    def get_current_depth_raw(self) -> Optional[np.ndarray]:
        """Get current raw depth (uint16)."""
        with self._lock:
            return self._current_depth_raw.copy() if self._current_depth_raw is not None else None
    
    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get current estimated pose."""
        with self._lock:
            return self._current_pose.copy() if self._current_pose is not None else None
    
    def get_intrinsics(self) -> Dict:
        """Get camera intrinsics."""
        return self._intrinsics.copy()
    
    def get_imu_data(self) -> Optional[Dict]:
        """Get latest IMU data."""
        with self._lock:
            if self._current_accel is None and self._current_gyro is None:
                return None
            return {
                'accel': self._current_accel.tolist() if self._current_accel is not None else None,
                'gyro': self._current_gyro.tolist() if self._current_gyro is not None else None,
                'timestamp': self._current_timestamp,
            }
    
    def get_point_cloud(self, subsample: int = 8, max_points: int = 10000) -> Optional[Dict]:
        """Get current point cloud (subsampled for visualization), transformed to world frame."""
        with self._lock:
            if self._current_depth is None or self._current_rgb is None:
                return None
            
            depth = self._current_depth.copy()
            rgb = self._current_rgb.copy()
            pose = self._current_pose.copy() if self._current_pose is not None else np.eye(4)
        
        h, w = depth.shape
        fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
        cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
        
        # Create pixel grid (subsampled)
        u = np.arange(0, w, subsample)
        v = np.arange(0, h, subsample)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        
        # Get depth and color at these pixels
        d = depth[v, u]
        colors = rgb[v, u]  # BGR
        
        # Filter valid depth
        valid = (d > self._depth_min) & (d < self._depth_max)
        u, v, d = u[valid], v[valid], d[valid]
        colors = colors[valid]
        
        if len(d) == 0:
            return None
        
        # Limit points
        if len(d) > max_points:
            idx = np.random.choice(len(d), max_points, replace=False)
            u, v, d = u[idx], v[idx], d[idx]
            colors = colors[idx]
        
        # Unproject to 3D (camera frame)
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        
        points_cam = np.stack([x, y, z], axis=1)
        
        # Transform to world frame using pose (camera-to-world)
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        # Convert BGR to RGB and normalize
        colors_rgb = colors[:, ::-1] / 255.0
        
        return {
            'points': points_world.tolist(),
            'colors': colors_rgb.tolist(),
            'count': len(points_world),
        }
    
    def get_status(self) -> Dict:
        """Get capture status."""
        elapsed = time.time() - self._start_time
        with self._lock:
            return {
                'device': self._device_name,
                'serial': self._device_serial,
                'resolution': f"{self._width}x{self._height}",
                'target_fps': self._fps,
                'actual_fps': self._frame_count / elapsed if elapsed > 0 else 0,
                'frames_captured': self._frame_count,
                'has_imu': self._has_imu,
                'pose_model': self._pose_model_name,
                'depth_scale': self._depth_scale,
                'running': self._running,
            }
    
    def stop(self):
        """Stop capture."""
        self._running = False
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        try:
            self._pipeline.stop()
        except:
            pass
        print("RealSense capture stopped")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""
    daemon_threads = True  # Don't wait for threads on shutdown


class RealSenseRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RealSense server."""
    
    capture: RealSenseCapture = None
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        path = self.path.split('?')[0]
        
        if path in ('/', '/stream'):
            self._handle_mjpeg_stream()
        elif path == '/frame':
            self._handle_synchronized_frame()
        elif path == '/pose':
            self._handle_pose()
        elif path == '/depth':
            self._handle_depth()
        elif path == '/intrinsics':
            self._handle_intrinsics()
        elif path == '/imu':
            self._handle_imu()
        elif path == '/pointcloud':
            self._handle_pointcloud()
        elif path == '/status':
            self._handle_status()
        elif path == '/reset':
            self._handle_reset()
        else:
            self.send_error(404, 'Not Found')
    
    def _handle_reset(self):
        """Reset the visual odometry to identity pose."""
        if hasattr(self.capture, '_stereo_vio') and self.capture._stereo_vio is not None:
            self.capture._stereo_vio.reset()
            msg = "VO reset to identity"
        else:
            msg = "No VO to reset"
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok', 'message': msg}).encode())
    
    def _handle_mjpeg_stream(self):
        """Stream MJPEG video."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        last_frame_idx = -1
        
        try:
            while True:
                # Get frame and check if it's new
                with self.capture._lock:
                    frame = self.capture._current_rgb
                    frame_idx = self.capture._frame_count
                
                # Only send if we have a new frame
                if frame is not None and frame_idx != last_frame_idx:
                    last_frame_idx = frame_idx
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                
                time.sleep(0.016)  # ~60Hz polling, actual rate limited by camera
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
    
    def _handle_synchronized_frame(self):
        """Return synchronized bundle: frame + pose + depth + intrinsics."""
        data = self.capture.get_synchronized_data()
        
        result = {
            'frame_idx': data['frame_idx'],
            'timestamp': data['timestamp'],
            'has_frame': data['rgb'] is not None,
            'has_pose': data['pose'] is not None,
            'has_depth': data['depth_raw'] is not None,
        }
        
        if data['rgb'] is not None:
            _, jpeg = cv2.imencode('.jpg', data['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 90])
            result['frame_jpg_b64'] = base64.b64encode(jpeg.tobytes()).decode()
        
        if data['pose'] is not None:
            result['pose'] = data['pose'].tolist()
        
        if data['depth_raw'] is not None:
            _, png = cv2.imencode('.png', data['depth_raw'])
            result['depth_png_b64'] = base64.b64encode(png.tobytes()).decode()
            result['depth_scale'] = int(1.0 / self.capture._depth_scale)  # Convert to integer scale
        
        result['intrinsics'] = self.capture.get_intrinsics()
        
        if data['accel'] is not None:
            result['accel'] = data['accel'].tolist()
        if data['gyro'] is not None:
            result['gyro'] = data['gyro'].tolist()
        
        self._send_json(result)
    
    def _handle_pose(self):
        """Return current pose."""
        pose = self.capture.get_current_pose()
        
        result = {
            'has_pose': pose is not None,
            'timestamp': time.time(),
        }
        if pose is not None:
            result['pose'] = pose.tolist()
        
        self._send_json(result)
    
    def _handle_depth(self):
        """Return current depth map."""
        depth = self.capture.get_current_depth_raw()
        
        result = {
            'has_depth': depth is not None,
            'scale_factor': int(1.0 / self.capture._depth_scale),
        }
        if depth is not None:
            _, png = cv2.imencode('.png', depth)
            result['depth_png_b64'] = base64.b64encode(png.tobytes()).decode()
            result['shape'] = list(depth.shape)
        
        self._send_json(result)
    
    def _handle_intrinsics(self):
        """Return camera intrinsics."""
        self._send_json(self.capture.get_intrinsics())
    
    def _handle_imu(self):
        """Return IMU data."""
        imu = self.capture.get_imu_data()
        if imu is None:
            result = {'has_imu': False}
        else:
            result = {'has_imu': True, **imu}
        self._send_json(result)
    
    def _handle_pointcloud(self):
        """Return current point cloud."""
        pc = self.capture.get_point_cloud()
        if pc is None:
            result = {'has_pointcloud': False, 'count': 0}
        else:
            result = {'has_pointcloud': True, **pc}
        self._send_json(result)
    
    def _handle_status(self):
        """Return server status."""
        self._send_json(self.capture.get_status())
    
    def _send_json(self, data: Dict):
        """Send JSON response."""
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


def list_realsense_devices():
    """List available RealSense devices."""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = []
        for dev in ctx.devices:
            devices.append({
                'name': dev.get_info(rs.camera_info.name),
                'serial': dev.get_info(rs.camera_info.serial_number),
                'firmware': dev.get_info(rs.camera_info.firmware_version),
            })
        return devices
    except ImportError:
        return []


def main():
    parser = argparse.ArgumentParser(description='Intel RealSense Live Server')
    parser.add_argument('--port', '-p', type=int, default=8554, help='HTTP port (default: 8554)')
    parser.add_argument('--width', '-W', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', '-H', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--pose-model', '-m', default='orb', 
                       choices=['orb', 'sift', 'robust_flow', 'loftr'],
                       help='Pose estimation model (default: orb)')
    parser.add_argument('--no-imu', action='store_true', help='Disable IMU streaming')
    parser.add_argument('--no-align', action='store_true', help='Disable depth-to-color alignment')
    parser.add_argument('--serial', '-s', type=str, default=None, help='Camera serial number')
    parser.add_argument('--list-devices', '-l', action='store_true', help='List available devices and exit')
    parser.add_argument('--auto-exposure', '-a', action='store_true', default=True, help='Auto-exposure then lock (default: on)')
    parser.add_argument('--exposure', '-e', type=int, default=None, help='Manual exposure value (skips auto)')
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found (or pyrealsense2 not installed)")
        else:
            print(f"Found {len(devices)} RealSense device(s):")
            for d in devices:
                print(f"  {d['name']} (S/N: {d['serial']}, FW: {d['firmware']})")
        return
    
    print("="*60)
    print("Intel RealSense Live Server")
    print("="*60)
    
    # Initialize capture
    try:
        capture = RealSenseCapture(
            width=args.width,
            height=args.height,
            fps=args.fps,
            align_depth=not args.no_align,
            enable_imu=not args.no_imu,
            pose_model=args.pose_model,
            serial_number=args.serial,
            auto_exposure=args.auto_exposure,
            manual_exposure=args.exposure,
        )
    except Exception as e:
        print(f"Failed to initialize RealSense: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure a RealSense camera is connected")
        print("  2. Install pyrealsense2: pip install pyrealsense2")
        print("  3. Try running: python -c \"import pyrealsense2 as rs; print(rs.context().devices)\"")
        return 1
    
    # Set capture for handler
    RealSenseRequestHandler.capture = capture
    
    # Start server
    server = ThreadedHTTPServer(('0.0.0.0', args.port), RealSenseRequestHandler)
    
    print(f"\nServer running on http://0.0.0.0:{args.port}")
    print(f"\nEndpoints:")
    print(f"  Stream:      http://localhost:{args.port}/stream")
    print(f"  Sync Frame:  http://localhost:{args.port}/frame")
    print(f"  Pose:        http://localhost:{args.port}/pose")
    print(f"  Depth:       http://localhost:{args.port}/depth")
    print(f"  IMU:         http://localhost:{args.port}/imu")
    print(f"  Point Cloud: http://localhost:{args.port}/pointcloud")
    print(f"  Intrinsics:  http://localhost:{args.port}/intrinsics")
    print(f"  Status:      http://localhost:{args.port}/status")
    print(f"\nUse in pipeline:")
    print(f"  LiveVideoSource('http://localhost:{args.port}/stream', use_server_pose=True, depth_model='ground_truth')")
    print(f"\nPress Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        capture.stop()
        server.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
