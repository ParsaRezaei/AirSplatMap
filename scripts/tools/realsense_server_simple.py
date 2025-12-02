#!/usr/bin/env python3
"""
Simple RealSense HTTP Server - Minimal implementation.
No background threads, no pose estimation, just raw camera feed.
"""

import cv2
import json
import numpy as np
import sys
import time
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

# Global state
rs = None
pipeline = None
align = None
intrinsics = None
depth_scale = 1.0
frame_count = 0
accumulated_pose = np.eye(4)


def init_camera():
    """Initialize RealSense camera."""
    global rs, pipeline, align, intrinsics, depth_scale
    
    import pyrealsense2 as rs_module
    rs = rs_module
    
    # Create context and check devices
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise RuntimeError("No RealSense devices found")
    
    print(f"Found {len(devices)} device(s)")
    for dev in devices:
        print(f"  - {dev.get_info(rs.camera_info.name)} (S/N: {dev.get_info(rs.camera_info.serial_number)})")
    
    # Hardware reset the device first
    print("Resetting device...")
    devices[0].hardware_reset()
    time.sleep(3)  # Wait for device to come back
    
    # Re-query after reset
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise RuntimeError("Device not found after reset")
    
    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Use 640x480 @ 30fps - most compatible
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    print("Starting pipeline...")
    profile = pipeline.start(config)
    
    # Get device and enable auto-exposure
    device = profile.get_device()
    color_sensor = device.first_color_sensor()
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Alignment
    align = rs.align(rs.stream.color)
    
    # Get intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    intrinsics = {
        'fx': intr.fx, 'fy': intr.fy,
        'cx': intr.ppx, 'cy': intr.ppy,
        'width': intr.width, 'height': intr.height
    }
    
    print(f"Camera ready: {intr.width}x{intr.height}")
    print(f"Intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}")
    print(f"Depth scale: {depth_scale}")
    
    # Warm up - grab a few frames
    print("Warming up...")
    for i in range(30):
        pipeline.wait_for_frames(timeout_ms=1000)
    print("Ready!")


def get_frame():
    """Get a frame - blocking call."""
    global pipeline, align, frame_count
    
    if pipeline is None:
        return None, None
    
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        frames = align.process(frames)
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame:
            return None, None
        
        frame_count += 1
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        
        return color, depth
    except Exception as e:
        print(f"Frame error: {e}")
        return None, None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        path = self.path.split('?')[0]
        
        if path in ('/', '/stream'):
            self.do_stream()
        elif path == '/frame':
            self.do_single_frame()
        elif path == '/pose':
            self.do_pose()
        elif path == '/intrinsics':
            self.send_json(intrinsics or {})
        elif path == '/status':
            self.send_json({'running': pipeline is not None, 'frames': frame_count})
        else:
            self.send_error(404)
    
    def do_stream(self):
        """MJPEG stream."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgbound')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        print(f"[Stream] Client connected")
        sent = 0
        t0 = time.time()
        
        try:
            while True:
                color, _ = get_frame()
                
                if color is not None:
                    _, jpeg = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    self.wfile.write(b'--jpgbound\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    
                    sent += 1
                    if sent % 30 == 0:
                        fps = sent / (time.time() - t0)
                        print(f"[Stream] {sent} frames, {fps:.1f} FPS")
                        
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            fps = sent / (time.time() - t0) if time.time() > t0 else 0
            print(f"[Stream] Client disconnected after {sent} frames ({fps:.1f} FPS)")
    
    def do_single_frame(self):
        """Single synchronized frame with pose and depth."""
        color, depth = get_frame()
        
        result = {
            'frame_idx': frame_count,
            'timestamp': time.time(),
            'has_frame': color is not None,
            'has_depth': depth is not None,
            'has_pose': True,
        }
        
        if color is not None:
            _, jpeg = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 90])
            result['frame_jpg_b64'] = base64.b64encode(jpeg.tobytes()).decode()
        
        if depth is not None:
            _, png = cv2.imencode('.png', depth)
            result['depth_png_b64'] = base64.b64encode(png.tobytes()).decode()
            result['depth_scale'] = int(1.0 / depth_scale)
        
        # Simple identity pose for now
        result['pose'] = accumulated_pose.tolist()
        result['intrinsics'] = intrinsics
        
        self.send_json(result)
    
    def do_pose(self):
        """Return current pose."""
        self.send_json({
            'has_pose': True,
            'pose': accumulated_pose.tolist(),
            'timestamp': time.time()
        })
    
    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8554)
    args = parser.parse_args()
    
    print("="*50)
    print("Simple RealSense Server")
    print("="*50)
    
    try:
        init_camera()
    except Exception as e:
        print(f"FATAL: {e}")
        print("\nTry:")
        print("  1. Unplug and replug the camera")
        print("  2. Close Intel RealSense Viewer if open")
        print("  3. Use a USB 3.0 port (blue)")
        return 1
    
    server = ThreadedHTTPServer(('0.0.0.0', args.port), Handler)
    print(f"\nServer: http://localhost:{args.port}")
    print(f"Stream: http://localhost:{args.port}/stream")
    print(f"Frame:  http://localhost:{args.port}/frame")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        if pipeline:
            pipeline.stop()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
