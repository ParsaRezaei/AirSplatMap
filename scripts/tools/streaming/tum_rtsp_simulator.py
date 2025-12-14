#!/usr/bin/env python3
"""
TUM RGBD Dataset Simulator - Serves synchronized frames, poses, and depth via HTTP.

Key Feature: All endpoints serve data for the SAME current frame, ensuring
synchronization between video stream, pose, and depth queries.

Endpoints:
  GET /          - MJPEG video stream
  GET /stream    - Same as / (alias)
  GET /frame     - Single synchronized bundle (image + pose + depth + intrinsics)
  GET /pose      - Current frame's ground truth pose (4x4 matrix)
  GET /depth     - Current frame's depth map (base64 PNG)
  GET /intrinsics - Camera intrinsics
  GET /status    - Server status info

Usage:
  python tum_rtsp_simulator.py --dataset /path/to/tum --fps 15
"""

import argparse
import base64
import cv2
import json
import numpy as np
import os
import re
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


class TUMDataset:
    """Load and serve TUM RGBD dataset with synchronized access."""
    
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.name = self.path.name
        
        # Load associations
        self.rgb_files: List[str] = []
        self.depth_files: List[str] = []
        self.timestamps: List[float] = []
        self.poses: List[np.ndarray] = []
        
        self._load_dataset()
        
        # TUM Freiburg camera intrinsics (default for fr1/fr2/fr3)
        if 'freiburg1' in self.name:
            self.fx, self.fy = 517.3, 516.5
            self.cx, self.cy = 318.6, 255.3
        elif 'freiburg2' in self.name:
            self.fx, self.fy = 520.9, 521.0
            self.cx, self.cy = 325.1, 249.7
        elif 'freiburg3' in self.name:
            self.fx, self.fy = 535.4, 539.2
            self.cx, self.cy = 320.1, 247.6
        else:
            # Default to fr1
            self.fx, self.fy = 517.3, 516.5
            self.cx, self.cy = 318.6, 255.3
    
    def _load_dataset(self):
        """Load RGB-D associations and ground truth poses."""
        # Try to find association file
        assoc_file = self.path / 'associations.txt'
        if not assoc_file.exists():
            # Try to create it from rgb.txt and depth.txt
            self._create_associations()
        
        if assoc_file.exists():
            self._load_associations(assoc_file)
        else:
            # Fallback: just load RGB files
            rgb_dir = self.path / 'rgb'
            if rgb_dir.exists():
                for f in sorted(rgb_dir.glob('*.png')):
                    self.rgb_files.append(str(f))
                    ts = float(f.stem)
                    self.timestamps.append(ts)
        
        # Load ground truth poses
        self._load_groundtruth()
        
        print(f"Loaded {len(self.rgb_files)} frames from {self.name}")
        print(f"  Poses available: {len(self.poses)}")
    
    def _create_associations(self):
        """Create associations.txt from rgb.txt and depth.txt."""
        rgb_txt = self.path / 'rgb.txt'
        depth_txt = self.path / 'depth.txt'
        
        if not rgb_txt.exists() or not depth_txt.exists():
            return
        
        # Load timestamps
        rgb_times = {}
        depth_times = {}
        
        with open(rgb_txt) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    rgb_times[float(parts[0])] = parts[1]
        
        with open(depth_txt) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    depth_times[float(parts[0])] = parts[1]
        
        # Associate by nearest timestamp (within 0.02s)
        associations = []
        for rgb_ts, rgb_file in sorted(rgb_times.items()):
            best_depth_ts = None
            best_diff = float('inf')
            for depth_ts in depth_times:
                diff = abs(rgb_ts - depth_ts)
                if diff < best_diff and diff < 0.02:
                    best_diff = diff
                    best_depth_ts = depth_ts
            
            if best_depth_ts is not None:
                associations.append((rgb_ts, rgb_file, best_depth_ts, depth_times[best_depth_ts]))
        
        # Write associations file
        with open(self.path / 'associations.txt', 'w') as f:
            for rgb_ts, rgb_file, depth_ts, depth_file in associations:
                f.write(f"{rgb_ts} {rgb_file} {depth_ts} {depth_file}\n")
    
    def _load_associations(self, assoc_file: Path):
        """Load pre-computed associations."""
        with open(assoc_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    rgb_ts = float(parts[0])
                    rgb_file = str(self.path / parts[1])
                    depth_file = str(self.path / parts[3])
                    
                    if os.path.exists(rgb_file) and os.path.exists(depth_file):
                        self.timestamps.append(rgb_ts)
                        self.rgb_files.append(rgb_file)
                        self.depth_files.append(depth_file)
    
    def _load_groundtruth(self):
        """Load ground truth poses from groundtruth.txt."""
        gt_file = self.path / 'groundtruth.txt'
        if not gt_file.exists():
            return
        
        gt_data = []
        with open(gt_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    ts = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    gt_data.append((ts, tx, ty, tz, qx, qy, qz, qw))
        
        # Match GT to RGB timestamps
        for rgb_ts in self.timestamps:
            best_gt = None
            best_diff = float('inf')
            for gt in gt_data:
                diff = abs(rgb_ts - gt[0])
                if diff < best_diff:
                    best_diff = diff
                    best_gt = gt
            
            if best_gt is not None and best_diff < 0.1:
                pose = self._quat_to_matrix(best_gt[1:4], best_gt[4:8])
                self.poses.append(pose)
            else:
                # Use identity if no matching GT
                self.poses.append(np.eye(4))
    
    def _quat_to_matrix(self, trans: Tuple[float, float, float], 
                        quat: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert translation + quaternion to 4x4 matrix."""
        tx, ty, tz = trans
        qx, qy, qz, qw = quat
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T
    
    def __len__(self):
        return len(self.rgb_files)
    
    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """Get RGB frame at index."""
        if 0 <= idx < len(self.rgb_files):
            img = cv2.imread(self.rgb_files[idx])
            return img
        return None
    
    def get_depth(self, idx: int) -> Optional[np.ndarray]:
        """Get depth map at index (in meters)."""
        if 0 <= idx < len(self.depth_files):
            depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)
            if depth is not None:
                # TUM depth is in millimeters, convert to meters
                # Actually TUM uses 5000 scale factor
                return depth.astype(np.float32) / 5000.0
        return None
    
    def get_depth_raw(self, idx: int) -> Optional[np.ndarray]:
        """Get raw depth (uint16) for efficient transfer."""
        if 0 <= idx < len(self.depth_files):
            return cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)
        return None
    
    def get_pose(self, idx: int) -> Optional[np.ndarray]:
        """Get ground truth pose at index."""
        if 0 <= idx < len(self.poses):
            return self.poses[idx]
        return None
    
    def get_timestamp(self, idx: int) -> Optional[float]:
        """Get timestamp at index."""
        if 0 <= idx < len(self.timestamps):
            return self.timestamps[idx]
        return None
    
    def get_intrinsics(self) -> Dict:
        """Get camera intrinsics."""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': 640,
            'height': 480
        }


class TUMSimulator:
    """Simulates a live TUM dataset stream with synchronized access."""
    
    def __init__(self, dataset: TUMDataset, fps: float = 15.0, loop: bool = True):
        self.dataset = dataset
        self.fps = fps
        self.loop = loop
        self.frame_interval = 1.0 / fps
        
        # Current frame state (protected by lock)
        self._lock = threading.Lock()
        self._current_idx = 0
        self._direction = 1  # 1 = forward, -1 = backward
        self._frames_played = 0
        self._running = True
        
        # Cache current frame data
        self._current_frame: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._current_pose: Optional[np.ndarray] = None
        self._current_timestamp: Optional[float] = None
        
        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
    
    def _playback_loop(self):
        """Background thread that advances frames at target FPS."""
        while self._running:
            time.sleep(self.frame_interval)
            self._advance_frame()
    
    def _advance_frame(self):
        """Advance to next frame and cache data."""
        with self._lock:
            # Load and cache current frame data
            idx = self._current_idx
            self._current_frame = self.dataset.get_frame(idx)
            self._current_depth = self.dataset.get_depth_raw(idx)
            self._current_pose = self.dataset.get_pose(idx)
            self._current_timestamp = self.dataset.get_timestamp(idx)
            
            # Advance index
            self._current_idx += self._direction
            self._frames_played += 1
            
            # Handle bounds
            if self._current_idx >= len(self.dataset):
                if self.loop:
                    self._direction = -1
                    self._current_idx = len(self.dataset) - 2
                else:
                    self._current_idx = len(self.dataset) - 1
            elif self._current_idx < 0:
                if self.loop:
                    self._direction = 1
                    self._current_idx = 1
                else:
                    self._current_idx = 0
    
    def get_synchronized_data(self) -> Dict:
        """Get synchronized frame, depth, pose for current frame."""
        with self._lock:
            return {
                'frame': self._current_frame,
                'depth': self._current_depth,
                'pose': self._current_pose,
                'timestamp': self._current_timestamp,
                'frame_idx': self._current_idx,
            }
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current RGB frame."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def get_current_depth(self) -> Optional[np.ndarray]:
        """Get current depth map (raw uint16)."""
        with self._lock:
            return self._current_depth.copy() if self._current_depth is not None else None
    
    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get current ground truth pose."""
        with self._lock:
            return self._current_pose.copy() if self._current_pose is not None else None
    
    def get_current_timestamp(self) -> Optional[float]:
        """Get current timestamp."""
        with self._lock:
            return self._current_timestamp
    
    def get_status(self) -> Dict:
        """Get simulator status."""
        with self._lock:
            return {
                'current_frame': self._current_idx,
                'total_frames': len(self.dataset),
                'frames_played': self._frames_played,
                'direction': 'forward' if self._direction > 0 else 'backward',
                'fps': self.fps,
                'dataset': self.dataset.name,
            }
    
    def stop(self):
        """Stop the simulator."""
        self._running = False


class TUMRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for TUM simulator."""
    
    simulator: TUMSimulator = None
    
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
        elif path == '/status':
            self._handle_status()
        else:
            self.send_error(404, 'Not Found')
    
    def _handle_mjpeg_stream(self):
        """Stream MJPEG video."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        try:
            while True:
                frame = self.simulator.get_current_frame()
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                time.sleep(1.0 / self.simulator.fps)
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def _handle_synchronized_frame(self):
        """Return synchronized bundle: frame + pose + depth + timestamp."""
        data = self.simulator.get_synchronized_data()
        
        result = {
            'frame_idx': data['frame_idx'],
            'timestamp': data['timestamp'],
            'has_frame': data['frame'] is not None,
            'has_pose': data['pose'] is not None,
            'has_depth': data['depth'] is not None,
        }
        
        if data['frame'] is not None:
            _, jpeg = cv2.imencode('.jpg', data['frame'], [cv2.IMWRITE_JPEG_QUALITY, 90])
            result['frame_jpg_b64'] = base64.b64encode(jpeg.tobytes()).decode()
        
        if data['pose'] is not None:
            result['pose'] = data['pose'].tolist()
        
        if data['depth'] is not None:
            _, png = cv2.imencode('.png', data['depth'])
            result['depth_png_b64'] = base64.b64encode(png.tobytes()).decode()
            result['depth_scale'] = 5000  # TUM scale factor
        
        result['intrinsics'] = self.simulator.dataset.get_intrinsics()
        
        self._send_json(result)
    
    def _handle_pose(self):
        """Return current ground truth pose."""
        pose = self.simulator.get_current_pose()
        timestamp = self.simulator.get_current_timestamp()
        
        result = {
            'has_pose': pose is not None,
            'timestamp': timestamp,
        }
        if pose is not None:
            result['pose'] = pose.tolist()
        
        self._send_json(result)
    
    def _handle_depth(self):
        """Return current depth map."""
        depth = self.simulator.get_current_depth()
        
        result = {
            'has_depth': depth is not None,
            'scale_factor': 5000,
        }
        if depth is not None:
            _, png = cv2.imencode('.png', depth)
            result['depth_png_b64'] = base64.b64encode(png.tobytes()).decode()
            result['shape'] = list(depth.shape)
        
        self._send_json(result)
    
    def _handle_intrinsics(self):
        """Return camera intrinsics."""
        self._send_json(self.simulator.dataset.get_intrinsics())
    
    def _handle_status(self):
        """Return server status."""
        self._send_json(self.simulator.get_status())
    
    def _send_json(self, data: Dict):
        """Send JSON response."""
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


def find_tum_datasets(base_path: str) -> List[str]:
    """Find TUM RGBD datasets in a directory."""
    datasets = []
    base = Path(base_path)
    
    # Look for directories containing rgb/ and depth/
    for d in base.iterdir():
        if d.is_dir():
            if (d / 'rgb').exists() and (d / 'depth').exists():
                datasets.append(str(d))
    
    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(description='TUM RGBD Dataset Simulator')
    parser.add_argument('--dataset', '-d', required=True, help='Path to TUM dataset or parent directory')
    parser.add_argument('--fps', '-f', type=float, default=15.0, help='Playback FPS (default: 15)')
    parser.add_argument('--port', '-p', type=int, default=8554, help='HTTP port (default: 8554)')
    parser.add_argument('--sequence', '-s', type=int, default=0, help='Sequence index if multiple datasets')
    args = parser.parse_args()
    
    # Find datasets
    dataset_path = args.dataset
    if os.path.isdir(os.path.join(dataset_path, 'rgb')):
        # Direct dataset path
        datasets = [dataset_path]
    else:
        # Parent directory - find all datasets
        datasets = find_tum_datasets(dataset_path)
    
    if not datasets:
        print(f"No TUM datasets found in {dataset_path}")
        return
    
    print(f"Found {len(datasets)} dataset(s):")
    for i, d in enumerate(datasets):
        print(f"  [{i}] {os.path.basename(d)}")
    
    # Select dataset
    idx = min(args.sequence, len(datasets) - 1)
    selected = datasets[idx]
    print(f"\nUsing dataset: {os.path.basename(selected)}")
    
    # Load dataset
    dataset = TUMDataset(selected)
    if len(dataset) == 0:
        print("Failed to load dataset")
        return
    
    # Create simulator
    simulator = TUMSimulator(dataset, fps=args.fps)
    TUMRequestHandler.simulator = simulator
    
    # Start threaded server (handles concurrent requests)
    server = ThreadedHTTPServer(('0.0.0.0', args.port), TUMRequestHandler)
    print(f"\nTUM Simulator running on http://0.0.0.0:{args.port}")
    print(f"  Stream:     http://localhost:{args.port}/stream")
    print(f"  Sync Frame: http://localhost:{args.port}/frame")
    print(f"  Pose:       http://localhost:{args.port}/pose")
    print(f"  Depth:      http://localhost:{args.port}/depth")
    print(f"  Intrinsics: http://localhost:{args.port}/intrinsics")
    print(f"  Status:     http://localhost:{args.port}/status")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        simulator.stop()
        server.shutdown()


if __name__ == '__main__':
    main()
