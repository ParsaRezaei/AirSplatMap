#!/usr/bin/env python3
"""
ArduPilot HTTP Server
=====================

Serves camera + ArduPilot pose data over HTTP for remote 3DGS mapping.
Run this on a companion computer (Raspberry Pi, Jetson, etc.) and
connect from a workstation for GPU-accelerated mapping.

Endpoints:
  GET /           - MJPEG video stream
  GET /frame      - Current frame as JPEG with pose in headers
  GET /bundle     - JSON bundle with base64 image + pose
  GET /pose       - Current pose (4x4 matrix as JSON)
  GET /status     - Vehicle status (armed, GPS, battery)
  GET /intrinsics - Camera intrinsics

Usage (on companion computer):
  python ardupilot_server.py --port 8554 --mavlink /dev/ttyAMA0 --camera 0

Usage (on workstation):
  # View stream in browser
  http://companion:8554/
  
  # Get single frame with pose
  curl http://companion:8554/bundle
  
  # Connect with AirSplatMap
  python live_web_demo.py --url http://companion:8554

Requirements:
  pip install pymavlink opencv-python numpy
"""

import argparse
import base64
import cv2
import json
import logging
import math
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional, Dict, Any

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for concurrent connections."""
    daemon_threads = True
    allow_reuse_address = True


class ArduPilotDataProvider:
    """
    Captures camera and ArduPilot data for HTTP serving.
    """
    
    def __init__(
        self,
        camera_source: Any = 0,
        mavlink_connection: str = "udpin:0.0.0.0:14550",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fov_deg: float = 60.0,
        camera_pitch_deg: float = -45.0,
        jpeg_quality: int = 85,
    ):
        self._camera_source = camera_source
        self._mavlink_connection = mavlink_connection
        self._width = width
        self._height = height
        self._fps = fps
        self._fov_deg = fov_deg
        self._camera_pitch_deg = camera_pitch_deg
        self._jpeg_quality = jpeg_quality
        
        # Computed intrinsics
        self._intrinsics = self._compute_intrinsics()
        
        # Components
        self._cap: Optional[cv2.VideoCapture] = None
        self._ardupilot = None
        
        # Current data (thread-safe)
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._current_jpeg: Optional[bytes] = None
        self._current_pose: Optional[np.ndarray] = None
        self._current_state = None
        self._frame_idx = 0
        self._timestamp = 0.0
        
        # Threading
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
    
    def _compute_intrinsics(self) -> Dict[str, float]:
        fx = self._width / (2.0 * np.tan(np.radians(self._fov_deg) / 2.0))
        return {
            'fx': fx,
            'fy': fx,
            'cx': self._width / 2.0,
            'cy': self._height / 2.0,
            'width': self._width,
            'height': self._height,
        }
    
    def start(self) -> bool:
        """Start camera and ArduPilot connection."""
        # Open camera
        logger.info(f"Opening camera: {self._camera_source}")
        self._cap = cv2.VideoCapture(self._camera_source)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera: {self._camera_source}")
            return False
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        
        # Get actual resolution
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h}")
        
        if actual_w != self._width or actual_h != self._height:
            self._width = actual_w
            self._height = actual_h
            self._intrinsics = self._compute_intrinsics()
        
        # Connect to ArduPilot
        try:
            from src.pose.ardupilot_mavlink import ArduPilotPoseProvider
            
            # Compute camera rotation
            pitch_rad = np.radians(self._camera_pitch_deg)
            cr, sr = math.cos(pitch_rad), math.sin(pitch_rad)
            camera_rotation = np.array([
                [1, 0, 0],
                [0, cr, -sr],
                [0, sr, cr]
            ]) @ np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])
            
            self._ardupilot = ArduPilotPoseProvider(
                connection_string=self._mavlink_connection,
                camera_rotation=camera_rotation,
            )
            
            if self._ardupilot.start():
                logger.info("ArduPilot connected")
            else:
                logger.warning("ArduPilot connection failed, poses will be identity")
                self._ardupilot = None
                
        except ImportError:
            logger.warning("pymavlink not installed, ArduPilot unavailable")
            self._ardupilot = None
        except Exception as e:
            logger.warning(f"ArduPilot init error: {e}")
            self._ardupilot = None
        
        # Start capture thread
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        return True
    
    def stop(self):
        """Stop all components."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._ardupilot:
            self._ardupilot.stop()
            self._ardupilot = None
    
    def _capture_loop(self):
        """Background capture loop."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Get pose from ArduPilot
            pose = None
            state = None
            if self._ardupilot:
                pose = self._ardupilot.get_pose()
                state = self._ardupilot.get_state()
            
            if pose is None:
                pose = np.eye(4, dtype=np.float64)
            
            # Encode JPEG
            _, jpeg = cv2.imencode('.jpg', frame, encode_params)
            jpeg_bytes = jpeg.tobytes()
            
            # Update shared state
            with self._lock:
                self._current_frame = frame
                self._current_jpeg = jpeg_bytes
                self._current_pose = pose
                self._current_state = state
                self._timestamp = time.time()
                self._frame_idx += 1
            
            # Rate limit
            time.sleep(1.0 / self._fps)
    
    def get_jpeg(self) -> Optional[bytes]:
        """Get current frame as JPEG."""
        with self._lock:
            return self._current_jpeg
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame as numpy array (BGR)."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def get_pose(self) -> np.ndarray:
        """Get current 4x4 pose matrix."""
        with self._lock:
            return self._current_pose.copy() if self._current_pose is not None else np.eye(4)
    
    def get_state(self):
        """Get current ArduPilot state."""
        with self._lock:
            return self._current_state
    
    def get_bundle(self) -> Dict[str, Any]:
        """Get synchronized frame bundle with image and pose."""
        with self._lock:
            if self._current_jpeg is None:
                return None
            
            bundle = {
                'frame_idx': self._frame_idx,
                'timestamp': self._timestamp,
                'image_base64': base64.b64encode(self._current_jpeg).decode('ascii'),
                'pose': self._current_pose.tolist() if self._current_pose is not None else np.eye(4).tolist(),
                'intrinsics': self._intrinsics,
            }
            
            if self._current_state:
                bundle['vehicle_state'] = {
                    'armed': self._current_state.armed,
                    'roll': self._current_state.roll,
                    'pitch': self._current_state.pitch,
                    'yaw': self._current_state.yaw,
                    'gps_fix': self._current_state.gps_fix,
                    'satellites': self._current_state.satellites,
                    'battery_voltage': self._current_state.battery_voltage,
                    'battery_remaining': self._current_state.battery_remaining,
                    'lat': self._current_state.lat,
                    'lon': self._current_state.lon,
                    'alt': self._current_state.alt,
                }
                if self._current_state.position_ned is not None:
                    bundle['vehicle_state']['position_ned'] = self._current_state.position_ned.tolist()
            
            return bundle
    
    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics."""
        return self._intrinsics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get server and vehicle status."""
        status = {
            'server_time': time.time(),
            'frame_idx': self._frame_idx,
            'camera_connected': self._cap is not None and self._cap.isOpened(),
            'ardupilot_connected': self._ardupilot is not None,
            'resolution': {'width': self._width, 'height': self._height},
            'fps': self._fps,
        }
        
        state = self.get_state()
        if state:
            status['vehicle'] = {
                'armed': state.armed,
                'gps_fix': state.gps_fix,
                'satellites': state.satellites,
                'battery_voltage': state.battery_voltage,
                'battery_remaining': state.battery_remaining,
            }
        
        return status


# Global data provider
_data_provider: Optional[ArduPilotDataProvider] = None


class ArduPilotRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ArduPilot data."""
    
    protocol_version = 'HTTP/1.1'
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
    
    def _send_jpeg(self, jpeg: bytes):
        """Send JPEG image."""
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(jpeg))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(jpeg)
    
    def do_GET(self):
        """Handle GET requests."""
        global _data_provider
        
        if _data_provider is None:
            self._send_json({'error': 'Server not initialized'}, 500)
            return
        
        path = self.path.split('?')[0]  # Remove query string
        
        try:
            if path == '/' or path == '/stream':
                # MJPEG stream
                self._handle_mjpeg_stream()
            
            elif path == '/frame':
                # Single JPEG frame with pose in headers
                jpeg = _data_provider.get_jpeg()
                if jpeg:
                    pose = _data_provider.get_pose()
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('X-Pose', json.dumps(pose.tolist()))
                    self.send_header('X-Timestamp', str(time.time()))
                    self.end_headers()
                    self.wfile.write(jpeg)
                else:
                    self._send_json({'error': 'No frame available'}, 503)
            
            elif path == '/bundle':
                # Full bundle with base64 image and pose
                bundle = _data_provider.get_bundle()
                if bundle:
                    self._send_json(bundle)
                else:
                    self._send_json({'error': 'No data available'}, 503)
            
            elif path == '/pose':
                # Current pose matrix
                pose = _data_provider.get_pose()
                self._send_json({'pose': pose.tolist()})
            
            elif path == '/intrinsics':
                # Camera intrinsics
                self._send_json(_data_provider.get_intrinsics())
            
            elif path == '/status':
                # Server and vehicle status
                self._send_json(_data_provider.get_status())
            
            elif path == '/health':
                # Simple health check
                self._send_json({'status': 'ok'})
            
            else:
                self._send_json({'error': 'Not found'}, 404)
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            self._send_json({'error': str(e)}, 500)
    
    def _handle_mjpeg_stream(self):
        """Stream MJPEG video."""
        global _data_provider
        
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            while True:
                jpeg = _data_provider.get_jpeg()
                if jpeg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                
                time.sleep(1.0 / 30)  # 30 fps max
                
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected


def run_server(args):
    """Run the HTTP server."""
    global _data_provider
    
    # Create data provider
    _data_provider = ArduPilotDataProvider(
        camera_source=args.camera,
        mavlink_connection=args.mavlink,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fov_deg=args.fov,
        camera_pitch_deg=args.camera_pitch,
        jpeg_quality=args.quality,
    )
    
    if not _data_provider.start():
        logger.error("Failed to start data provider")
        return
    
    # Create server
    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, ArduPilotRequestHandler)
    
    logger.info(f"ArduPilot HTTP Server starting on http://{args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info(f"  GET /        - MJPEG video stream")
    logger.info(f"  GET /frame   - Single JPEG with pose in headers")
    logger.info(f"  GET /bundle  - JSON bundle (base64 image + pose)")
    logger.info(f"  GET /pose    - Current pose matrix")
    logger.info(f"  GET /status  - Server and vehicle status")
    logger.info(f"  GET /intrinsics - Camera intrinsics")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        _data_provider.stop()
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="ArduPilot HTTP Server for remote 3DGS mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with USB camera and MAVLink
  python ardupilot_server.py --camera 0 --mavlink udpin:0.0.0.0:14550

  # Raspberry Pi with Pixhawk
  python ardupilot_server.py --camera 0 --mavlink /dev/ttyAMA0 --port 8554

  # RTSP camera from companion
  python ardupilot_server.py --camera rtsp://localhost:8554/main --mavlink udpin:0.0.0.0:14550

  # Higher resolution
  python ardupilot_server.py --camera 0 --width 1280 --height 720 --mavlink udpin:0.0.0.0:14550
        """
    )
    
    # Server settings
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8554,
                       help="Server port (default: 8554)")
    
    # Camera settings
    parser.add_argument("--camera", type=str, default="0",
                       help="Camera source (index, URL, or GStreamer)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera FOV")
    parser.add_argument("--camera-pitch", type=float, default=-45.0,
                       help="Camera pitch (negative = down)")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality (0-100)")
    
    # MAVLink settings
    parser.add_argument("--mavlink", type=str, default="udpin:0.0.0.0:14550",
                       help="MAVLink connection string")
    
    args = parser.parse_args()
    
    # Parse camera source
    try:
        args.camera = int(args.camera)
    except ValueError:
        pass
    
    run_server(args)


if __name__ == "__main__":
    main()
