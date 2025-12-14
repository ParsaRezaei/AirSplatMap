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
        
        # Check for devices
        ctx = rs.context()
        devices = list(ctx.devices)
        if not devices:
            raise RuntimeError("No RealSense devices found")
        
        print(f"Found {len(devices)} device(s)")
        
        # Hardware reset to clear any stale state (required for IMU to work)
        print("Hardware reset to clear device state...")
        devices[0].hardware_reset()
        time.sleep(4)  # Wait for device to come back
        
        # Re-query after reset
        ctx = rs.context()
        devices = list(ctx.devices)
        if not devices:
            raise RuntimeError("Device not found after reset")
        
        # Get device serial for both pipelines
        device_serial = serial_number or devices[0].get_info(rs.camera_info.serial_number)
        
        # ============================================
        # PIPELINE 1: IMU FIRST (like DonkeyCar approach)
        # Start IMU before video to avoid resource conflicts
        # ============================================
        self._imu_pipeline = None
        self._has_imu = False
        self._imu_lock = threading.Lock()
        self._latest_accel = None
        self._latest_gyro = None
        
        if enable_imu:
            try:
                self._imu_pipeline = rs.pipeline()
                imu_config = rs.config()
                imu_config.enable_device(device_serial)
                # Use lowest available frequencies (100Hz accel, 200Hz gyro)
                # Available: Accel @ 400/200/100 Hz, Gyro @ 400/200 Hz
                imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
                imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                
                self._imu_profile = self._imu_pipeline.start(imu_config)
                
                # Warmup IMU - eat some frames to let it settle
                print("IMU pipeline starting, warming up...")
                for i in range(5):
                    try:
                        self._imu_pipeline.wait_for_frames(1000)
                    except:
                        pass
                
                self._has_imu = True
                print("IMU pipeline started (accel@63Hz, gyro@200Hz)")
            except Exception as e:
                print(f"IMU pipeline failed: {e}")
                self._imu_pipeline = None
        
        # ============================================
        # PIPELINE 2: Video (RGB + Depth + IR) at 30Hz
        # Started AFTER IMU pipeline
        # ============================================
        self._video_pipeline = rs.pipeline()
        self._video_config = rs.config()
        self._video_config.enable_device(device_serial)
        
        # Video streams
        self._video_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._video_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Enable IR stereo streams
        self._has_ir = False
        try:
            self._video_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            self._video_config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
            self._has_ir = True
            print("IR stereo streams enabled")
        except Exception as e:
            print(f"IR streams not available: {e}")
        
        # Start video pipeline
        try:
            self._video_profile = self._video_pipeline.start(self._video_config)
        except Exception as e:
            # If video fails and IMU was started, stop IMU
            if self._imu_pipeline:
                self._imu_pipeline.stop()
            raise RuntimeError(f"Failed to start video pipeline: {e}")
        
        # Get device info from video pipeline
        device = self._video_profile.get_device()
        self._device_name = device.get_info(rs.camera_info.name)
        self._device_serial = device.get_info(rs.camera_info.serial_number)
        print(f"Connected to {self._device_name} (S/N: {self._device_serial})")
        
        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self._depth_scale}")
        
        # Get stereo baseline
        try:
            depth_baseline = depth_sensor.get_option(rs.option.stereo_baseline)
            self._stereo_baseline = depth_baseline / 1000.0
            print(f"Stereo baseline: {self._stereo_baseline:.4f}m")
        except:
            self._stereo_baseline = 0.05
            print(f"Stereo baseline: {self._stereo_baseline:.4f}m (default)")
        
        # Configure exposure
        self._color_sensor = device.first_color_sensor()
        self._exposure_locked = False
        
        if manual_exposure is not None:
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            try:
                self._color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            except:
                pass
            self._color_sensor.set_option(rs.option.exposure, manual_exposure)
            self._exposure_locked = True
            print(f"Manual exposure: {manual_exposure}")
        else:
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("Auto-exposure: enabled (will lock after warmup)")
        
        # Get intrinsics
        color_stream = self._video_profile.get_stream(rs.stream.color)
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
        
        # Get IR intrinsics
        try:
            ir_stream = self._video_profile.get_stream(rs.stream.infrared, 1)
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
        
        # ============================================
        # Let camera warm up (like DonkeyCar does)
        # ============================================
        print("Letting cameras warm up...")
        time.sleep(2)
        
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
        
        # Running state
        self._running = True
        self._pipeline_ok = True
        
        # Brief video warmup
        print("Warming up video pipeline...")
        for i in range(10):
            try:
                frames = self._video_pipeline.wait_for_frames(timeout_ms=1000)
            except:
                pass
        print("Camera ready!")
        
        # Start IMU capture thread if IMU is available
        if self._has_imu:
            self._imu_thread = threading.Thread(target=self._imu_capture_loop, daemon=True)
            self._imu_thread.start()
        
        # Start video capture thread
        self._capture_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _imu_capture_loop(self):
        """Separate thread for IMU pipeline polling."""
        rs = self._rs
        
        while self._running and self._imu_pipeline is not None:
            try:
                # Poll IMU pipeline
                imu_frames = self._imu_pipeline.wait_for_frames(100)
                
                # Extract accel and gyro
                accel_frame = imu_frames.first_or_default(rs.stream.accel)
                gyro_frame = imu_frames.first_or_default(rs.stream.gyro)
                
                with self._imu_lock:
                    if accel_frame:
                        accel_data = accel_frame.as_motion_frame().get_motion_data()
                        self._latest_accel = np.array([accel_data.x, accel_data.y, accel_data.z])
                    if gyro_frame:
                        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                        self._latest_gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z])
                        
            except Exception as e:
                # Timeout is normal
                pass
    
    def _video_capture_loop(self):
        """Video capture loop at 30Hz - IMU data from separate thread."""
        rs = self._rs
        error_count = 0
        
        while self._running:
            if not self._pipeline_ok:
                time.sleep(0.1)
                continue
                
            try:
                # Wait for video frames from video pipeline
                frames = self._video_pipeline.wait_for_frames(timeout_ms=100)
                error_count = 0
                timestamp = time.time()
                
                # Get IMU data from separate IMU thread
                accel = None
                gyro = None
                if self._has_imu:
                    with self._imu_lock:
                        accel = self._latest_accel.copy() if self._latest_accel is not None else None
                        gyro = self._latest_gyro.copy() if self._latest_gyro is not None else None
                
                # Lock exposure after warmup
                if not self._exposure_locked and self._frame_count == 90:
                    try:
                        current_exp = self._color_sensor.get_option(rs.option.exposure)
                        final_exp = max(500, current_exp)
                        self._color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                        self._color_sensor.set_option(rs.option.exposure, final_exp)
                        self._exposure_locked = True
                    except:
                        pass
                
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
                aligned = frames
                if self._align:
                    aligned = self._align.process(frames)
                
                # Get video frames
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy
                rgb = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                
                # Convert depth to meters
                depth = depth_raw.astype(np.float32) * self._depth_scale
                depth[(depth < self._depth_min) | (depth > self._depth_max)] = 0
                
                # Estimate pose with IMU data from separate thread
                pose = self._estimate_pose(ir_left, ir_right, rgb, depth, timestamp, accel, gyro)
                
                # Update current data
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
                
            except RuntimeError as e:
                # Timeout is expected when polling
                if "Frame didn't arrive" not in str(e):
                    error_count += 1
                    if error_count == 1 or error_count % 30 == 0:
                        print(f"Capture error ({error_count}): {e}")
            except Exception as e:
                if self._running:
                    error_count += 1
                    if error_count == 1 or error_count % 30 == 0:
                        print(f"Capture error ({error_count}): {e}")
                    time.sleep(0.033)
    
    def _estimate_pose(self, ir_left, ir_right, rgb, depth, timestamp, accel, gyro):
        """Estimate pose using available data."""
        pose = None
        
        # Try StereoVIO first
        if self._stereo_vio is not None and ir_left is not None:
            try:
                vio_result = self._stereo_vio.process(
                    ir_left, ir_right, timestamp, depth, accel, gyro
                )
                pose = self._stereo_vio.get_pose()
            except Exception as e:
                if self._frame_count < 5:
                    print(f"StereoVIO error: {e}")
        
        # Try RGBD VO
        if pose is None and self._vio is not None:
            try:
                rgb_for_vio = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                vio_result = self._vio.process(
                    rgb_for_vio, depth, timestamp, accel, gyro
                )
                pose = self._vio.get_pose()
            except Exception as e:
                if self._frame_count < 5:
                    print(f"VIO error: {e}")
        
        # Fallback to ORB
        if pose is None:
            pose = self._estimate_pose_fallback(rgb)
        
        return pose
    
    def _init_pose_estimator(self):
        """Initialize visual odometry."""
        # Always init fallback first (in case VO fails mid-run)
        self._init_fallback_pose()
        self._stereo_vio = None
        self._vio = None
        
        # Try StereoVIO from src.pose with better parameters for stability
        try:
            from src.pose.stereo_vio import StereoVIO
            self._stereo_vio = StereoVIO(
                baseline=self._stereo_baseline,
                max_features=500,  # More features for robustness
                min_features=80,
                use_imu=self._has_imu,
                pose_smoothing=0.5,  # Strong smoothing to reduce jitter
                motion_threshold=1.5,  # Higher threshold for stationary detection
                max_translation=0.05,  # Max 5cm per frame (~1.5m/s at 30fps)
                max_rotation=0.1,  # Max ~5.7 degrees per frame
            )
            self._stereo_vio.set_intrinsics_from_dict(self._ir_intrinsics)
            print(f"Initialized StereoVIO (baseline={self._stereo_baseline:.4f}m, imu={self._has_imu})")
            return
        except Exception as e:
            print(f"StereoVIO not available: {e}")
            import traceback
            traceback.print_exc()
        
        # Try RGBD VO
        try:
            from src.pose.rgbd_vo import RGBDVO
            self._vio = RGBDVO()
            self._vio.set_intrinsics_from_dict(self._intrinsics)
            print(f"Initialized RGBDVO")
            return
        except Exception as e:
            print(f"RGBDVO not available: {e}")
        
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
        """Background capture thread (used when IMU is disabled)."""
        rs = self._rs
        error_count = 0
        
        while self._running:
            if not self._pipeline_ok:
                time.sleep(0.1)
                continue
                
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                error_count = 0
                timestamp = time.time()
                
                # Lock exposure after warmup
                if not self._exposure_locked and self._frame_count == 90:
                    try:
                        current_exp = self._color_sensor.get_option(rs.option.exposure)
                        final_exp = max(500, current_exp)
                        self._color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                        self._color_sensor.set_option(rs.option.exposure, final_exp)
                        self._exposure_locked = True
                    except:
                        pass
                
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
                rgb = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                
                # Convert depth to meters
                depth = depth_raw.astype(np.float32) * self._depth_scale
                depth[(depth < self._depth_min) | (depth > self._depth_max)] = 0
                
                # Estimate pose (no IMU data in this mode)
                pose = self._estimate_pose(ir_left, ir_right, rgb, depth, timestamp, None, None)
                
                # Update current data
                with self._lock:
                    self._current_rgb = rgb
                    self._current_depth = depth
                    self._current_depth_raw = depth_raw
                    self._current_ir_left = ir_left
                    self._current_ir_right = ir_right
                    self._current_pose = pose
                    self._current_timestamp = timestamp
                    self._frame_count += 1
                
            except Exception as e:
                if self._running:
                    error_count += 1
                    if error_count == 1 or error_count % 30 == 0:
                        print(f"Capture error ({error_count}): {e}")
                    time.sleep(0.033)
    
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
        
        # Wait for video capture thread
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        # Wait for IMU thread if it exists
        if self._has_imu and hasattr(self, '_imu_thread') and self._imu_thread.is_alive():
            self._imu_thread.join(timeout=1.0)
        
        # Stop IMU pipeline first (started first, stop last is cleaner but either works)
        if self._imu_pipeline is not None:
            try:
                self._imu_pipeline.stop()
            except:
                pass
        
        # Stop video pipeline
        try:
            self._video_pipeline.stop()
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
