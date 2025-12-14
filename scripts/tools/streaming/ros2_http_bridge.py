#!/usr/bin/env python3
"""
ROS2 HTTP Bridge for AirSplatMap
================================

Bridges ROS2 topics (cuVSLAM + RealSense) to HTTP endpoints
compatible with the AirSplatMap dashboard and Gaussian splatting pipeline.

Topics subscribed:
- /visual_slam/tracking/odometry - Pose from cuVSLAM
- /realsense/color/image_raw - RGB image
- /realsense/aligned_depth_to_color/image_raw - Aligned depth

Endpoints (same interface as realsense_server.py):
- GET /stream - MJPEG stream with pose in headers
- GET /frame - Single synchronized frame (JSON)
- GET /pose - Current pose (JSON)
- GET /depth - Depth image (PNG)
- GET /status - Server status (JSON)
- GET /intrinsics - Camera intrinsics (JSON)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import numpy as np
import cv2
import json
import time
import threading
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
import argparse


class ROS2Bridge(Node):
    """ROS2 node that collects data from topics."""
    
    def __init__(self):
        super().__init__('airsplatmap_bridge')
        
        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Data storage with locks
        self._lock = threading.Lock()
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._pose: np.ndarray = np.eye(4)
        self._timestamp: float = 0.0
        self._intrinsics: Optional[dict] = None
        self._depth_scale: float = 0.001  # RealSense default
        
        # Frame counters
        self._rgb_count = 0
        self._depth_count = 0
        self._pose_count = 0
        self._start_time = time.time()
        
        # Subscribe to RGB
        self.create_subscription(
            Image,
            '/realsense/color/image_raw',
            self._rgb_callback,
            sensor_qos
        )
        
        # Subscribe to aligned depth
        self.create_subscription(
            Image,
            '/realsense/aligned_depth_to_color/image_raw',
            self._depth_callback,
            sensor_qos
        )
        
        # Subscribe to cuVSLAM odometry
        self.create_subscription(
            Odometry,
            '/visual_slam/tracking/odometry',
            self._odom_callback,
            sensor_qos
        )
        
        # Subscribe to camera info for intrinsics
        self.create_subscription(
            CameraInfo,
            '/realsense/color/camera_info',
            self._camera_info_callback,
            sensor_qos
        )
        
        self.get_logger().info('ROS2 Bridge initialized')
    
    def _rgb_callback(self, msg: Image):
        """Handle RGB image."""
        try:
            # Convert ROS image to numpy
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            else:
                self.get_logger().warn(f'Unknown RGB encoding: {msg.encoding}')
                return
            
            with self._lock:
                self._rgb = img
                self._timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self._rgb_count += 1
        except Exception as e:
            self.get_logger().error(f'RGB callback error: {e}')
    
    def _depth_callback(self, msg: Image):
        """Handle depth image."""
        try:
            if msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            elif msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                depth = (depth * 1000).astype(np.uint16)  # Convert to mm
            else:
                self.get_logger().warn(f'Unknown depth encoding: {msg.encoding}')
                return
            
            with self._lock:
                self._depth = depth
                self._depth_count += 1
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')
    
    def _odom_callback(self, msg: Odometry):
        """Handle odometry from cuVSLAM."""
        try:
            # Extract position
            pos = msg.pose.pose.position
            quat = msg.pose.pose.orientation
            
            # Convert quaternion to rotation matrix
            # q = [w, x, y, z] but ROS uses [x, y, z, w]
            x, y, z, w = quat.x, quat.y, quat.z, quat.w
            
            # Rotation matrix from quaternion
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
            ])
            
            # Build 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = [pos.x, pos.y, pos.z]
            
            with self._lock:
                self._pose = pose
                self._pose_count += 1
        except Exception as e:
            self.get_logger().error(f'Odometry callback error: {e}')
    
    def _camera_info_callback(self, msg: CameraInfo):
        """Handle camera info for intrinsics."""
        try:
            K = np.array(msg.k).reshape(3, 3)
            with self._lock:
                self._intrinsics = {
                    'fx': float(K[0, 0]),
                    'fy': float(K[1, 1]),
                    'cx': float(K[0, 2]),
                    'cy': float(K[1, 2]),
                    'width': msg.width,
                    'height': msg.height,
                }
        except Exception as e:
            self.get_logger().error(f'Camera info callback error: {e}')
    
    def get_frame(self):
        """Get current synchronized frame data."""
        with self._lock:
            return {
                'rgb': self._rgb.copy() if self._rgb is not None else None,
                'depth': self._depth.copy() if self._depth is not None else None,
                'pose': self._pose.copy(),
                'timestamp': self._timestamp,
            }
    
    def get_pose(self):
        """Get current pose."""
        with self._lock:
            return self._pose.copy(), self._timestamp
    
    def get_intrinsics(self):
        """Get camera intrinsics."""
        with self._lock:
            return self._intrinsics
    
    def get_status(self):
        """Get server status."""
        elapsed = time.time() - self._start_time
        with self._lock:
            return {
                'device': 'ROS2 cuVSLAM Bridge',
                'serial': 'ros2_bridge',
                'resolution': f'{self._intrinsics["width"]}x{self._intrinsics["height"]}' if self._intrinsics else 'unknown',
                'target_fps': 30,
                'actual_fps': self._rgb_count / elapsed if elapsed > 0 else 0,
                'frames_captured': self._rgb_count,
                'has_imu': True,  # cuVSLAM uses IMU
                'pose_model': 'cuvslam',
                'depth_scale': self._depth_scale,
                'running': True,
                'pose_count': self._pose_count,
            }


class BridgeHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler."""
    
    bridge: ROS2Bridge = None
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        path = self.path.split('?')[0]
        
        if path == '/stream':
            self._handle_stream()
        elif path == '/frame':
            self._handle_frame()
        elif path == '/pose':
            self._handle_pose()
        elif path == '/depth':
            self._handle_depth()
        elif path == '/status':
            self._handle_status()
        elif path == '/intrinsics':
            self._handle_intrinsics()
        else:
            self.send_error(404)
    
    def _send_json(self, data):
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)
    
    def _handle_stream(self):
        """MJPEG stream with pose in headers."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            while True:
                frame = self.bridge.get_frame()
                rgb = frame['rgb']
                
                if rgb is None:
                    time.sleep(0.033)
                    continue
                
                # Encode JPEG
                _, jpeg = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Pose as JSON in header
                pose = frame['pose']
                pose_json = json.dumps({
                    'pose': pose.tolist(),
                    'timestamp': frame['timestamp'],
                })
                
                # Build frame
                header = (
                    f'--frame\r\n'
                    f'Content-Type: image/jpeg\r\n'
                    f'Content-Length: {len(jpeg)}\r\n'
                    f'X-Pose: {pose_json}\r\n'
                    f'\r\n'
                )
                
                self.wfile.write(header.encode())
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def _handle_frame(self):
        """Single synchronized frame."""
        frame = self.bridge.get_frame()
        
        if frame['rgb'] is None:
            self._send_json({'has_frame': False})
            return
        
        # Encode images
        _, rgb_jpg = cv2.imencode('.jpg', frame['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        rgb_b64 = base64.b64encode(rgb_jpg).decode()
        
        depth_b64 = None
        has_depth = False
        if frame['depth'] is not None:
            _, depth_png = cv2.imencode('.png', frame['depth'])
            depth_b64 = base64.b64encode(depth_png).decode()
            has_depth = True
        
        # Get intrinsics
        intrinsics = self.bridge.get_intrinsics()
        
        self._send_json({
            'has_frame': True,
            'has_pose': True,
            'timestamp': frame['timestamp'],
            'frame_jpg_b64': rgb_b64,  # Dashboard expects this key
            'depth_png_b64': depth_b64,  # frames.py expects this key
            'has_depth': has_depth,
            'depth_scale': 1000,  # Depth in mm, scale to meters
            'pose': frame['pose'].tolist(),
            'intrinsics': intrinsics,
        })
    
    def _handle_pose(self):
        """Current pose."""
        pose, timestamp = self.bridge.get_pose()
        self._send_json({
            'has_pose': True,
            'timestamp': timestamp,
            'pose': pose.tolist(),
        })
    
    def _handle_depth(self):
        """Depth image as PNG."""
        frame = self.bridge.get_frame()
        
        if frame['depth'] is None:
            self.send_error(404)
            return
        
        # Colorize depth
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(frame['depth'], alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        _, png = cv2.imencode('.png', depth_vis)
        
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', len(png))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(png.tobytes())
    
    def _handle_status(self):
        """Server status."""
        self._send_json(self.bridge.get_status())
    
    def _handle_intrinsics(self):
        """Camera intrinsics."""
        intrinsics = self.bridge.get_intrinsics()
        if intrinsics:
            self._send_json(intrinsics)
        else:
            self._send_json({'error': 'Intrinsics not available yet'})


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description='ROS2 HTTP Bridge for AirSplatMap')
    parser.add_argument('--port', type=int, default=8554, help='HTTP port')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ROS2 HTTP Bridge for AirSplatMap")
    print("=" * 60)
    
    # Initialize ROS2
    rclpy.init()
    bridge = ROS2Bridge()
    
    # Set up HTTP handler
    BridgeHTTPHandler.bridge = bridge
    
    # Start HTTP server
    server = ThreadedHTTPServer(('0.0.0.0', args.port), BridgeHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    print(f"\nHTTP server running on http://0.0.0.0:{args.port}")
    print("\nEndpoints:")
    print(f"  Stream:     http://localhost:{args.port}/stream")
    print(f"  Frame:      http://localhost:{args.port}/frame")
    print(f"  Pose:       http://localhost:{args.port}/pose")
    print(f"  Depth:      http://localhost:{args.port}/depth")
    print(f"  Status:     http://localhost:{args.port}/status")
    print(f"  Intrinsics: http://localhost:{args.port}/intrinsics")
    print("\nWaiting for ROS2 data...")
    
    try:
        rclpy.spin(bridge)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        try:
            bridge.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass
        try:
            server.shutdown()
        except:
            pass
        print("\nShutdown complete")


if __name__ == '__main__':
    main()
