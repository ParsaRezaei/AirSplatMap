#!/usr/bin/env python3
"""
Start RealSense camera and add it to the web dashboard as a live source.

This script:
1. Detects connected RealSense cameras
2. Starts the RealSense HTTP server (realsense_server.py)
3. Connects to the web dashboard WebSocket API
4. Adds the RealSense stream as a live source

Usage:
  python scripts/tools/start_realsense_live.py
  python scripts/tools/start_realsense_live.py --port 8554 --dashboard-port 9003
  python scripts/tools/start_realsense_live.py --width 640 --height 480 --fps 30
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def detect_realsense():
    """Detect connected RealSense cameras."""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = []
        for dev in ctx.devices:
            try:
                devices.append({
                    'name': dev.get_info(rs.camera_info.name),
                    'serial': dev.get_info(rs.camera_info.serial_number),
                    'firmware': dev.get_info(rs.camera_info.firmware_version),
                })
            except Exception:
                pass
        return devices
    except ImportError:
        print("ERROR: pyrealsense2 not installed")
        return []
    except Exception as e:
        print(f"ERROR: Failed to detect RealSense: {e}")
        return []


def start_realsense_server(port=8554, width=640, height=480, fps=30):
    """Start the RealSense HTTP server as a subprocess."""
    script_path = project_root / 'scripts' / 'tools' / 'realsense_server.py'
    
    if not script_path.exists():
        print(f"ERROR: realsense_server.py not found at {script_path}")
        return None
    
    # Set LD_LIBRARY_PATH for rsusb backend
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"/usr/local/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    cmd = [
        sys.executable, str(script_path),
        '--port', str(port),
        '--width', str(width),
        '--height', str(height),
        '--fps', str(fps),
    ]
    
    print(f"Starting RealSense server: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            bufsize=1,
            universal_newlines=True,
        )
        return process
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        return None


async def add_to_dashboard(dashboard_host='127.0.0.1', dashboard_port=9003, 
                           realsense_port=8554, name='RealSense D435i'):
    """Connect to dashboard WebSocket and add RealSense as live source."""
    try:
        import websockets
    except ImportError:
        print("ERROR: websockets not installed. Run: pip install websockets")
        return False
    
    uri = f'ws://{dashboard_host}:{dashboard_port}'
    print(f"Connecting to dashboard at {uri}...")
    
    try:
        async with websockets.connect(uri, ping_timeout=10) as ws:
            # Wait for init message
            init_msg = await asyncio.wait_for(ws.recv(), timeout=5)
            init_data = json.loads(init_msg)
            
            if init_data.get('type') != 'init':
                print(f"Unexpected init message: {init_data.get('type')}")
                return False
            
            # Check if already added
            datasets = init_data.get('datasets', [])
            existing = [d['name'] for d in datasets if d.get('live')]
            if name in existing:
                print(f"'{name}' already exists in dashboard")
                return True
            
            print(f"Current datasets: {[d['name'] for d in datasets]}")
            
            # Add RealSense as live source
            stream_url = f'http://127.0.0.1:{realsense_port}/stream'
            await ws.send(json.dumps({
                'cmd': 'add_live',
                'name': name,
                'source': stream_url,
                'type': 'REALSENSE',
                'pose_method': 'ground_truth',
                'depth_method': 'ground_truth',
            }))
            
            # Wait for response
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            resp_data = json.loads(resp)
            
            if resp_data.get('type') == 'datasets':
                new_datasets = resp_data.get('datasets', [])
                print(f"Success! Live sources: {[d['name'] for d in new_datasets if d.get('live')]}")
                return True
            else:
                print(f"Unexpected response: {resp_data}")
                return False
                
    except asyncio.TimeoutError:
        print("ERROR: Timeout connecting to dashboard")
        return False
    except ConnectionRefusedError:
        print(f"ERROR: Cannot connect to dashboard at {uri}")
        print("Make sure the dashboard is running: python dashboard/web_dashboard.py")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def wait_for_server(port, timeout=10):
    """Wait for the RealSense server to become available."""
    import urllib.request
    import urllib.error
    
    url = f'http://127.0.0.1:{port}/status'
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                data = json.loads(resp.read().decode())
                if data.get('running'):
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(0.5)
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Start RealSense camera and add to web dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Auto-detect and start with defaults
  %(prog)s --port 8554               # Use specific port
  %(prog)s --width 1280 --height 720 # Higher resolution
  %(prog)s --no-dashboard            # Only start server, don't add to dashboard
        """
    )
    parser.add_argument('--port', '-p', type=int, default=8554,
                        help='RealSense server port (default: 8554)')
    parser.add_argument('--width', '-W', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', '-H', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--dashboard-host', default='127.0.0.1',
                        help='Dashboard WebSocket host (default: 127.0.0.1)')
    parser.add_argument('--dashboard-port', type=int, default=9003,
                        help='Dashboard WebSocket port (default: 9003)')
    parser.add_argument('--name', default='RealSense D435i',
                        help='Name for the live source (default: RealSense D435i)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Only start server, do not add to dashboard')
    parser.add_argument('--list-devices', '-l', action='store_true',
                        help='List RealSense devices and exit')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RealSense Live Stream Setup")
    print("=" * 60)
    
    # Detect cameras
    print("\nDetecting RealSense cameras...")
    devices = detect_realsense()
    
    if not devices:
        print("No RealSense cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check USB connection (use USB 3.0 port)")
        print("  2. Run: lsusb | grep Intel")
        print("  3. Check pyrealsense2: python -c 'import pyrealsense2'")
        return 1
    
    print(f"Found {len(devices)} camera(s):")
    for d in devices:
        print(f"  • {d['name']} (S/N: {d['serial']}, FW: {d['firmware']})")
    
    if args.list_devices:
        return 0
    
    # Start server
    print(f"\nStarting RealSense server on port {args.port}...")
    process = start_realsense_server(args.port, args.width, args.height, args.fps)
    
    if process is None:
        return 1
    
    # Wait for server to be ready
    print("Waiting for server to initialize...")
    time.sleep(3)  # Give it time to initialize camera
    
    if process.poll() is not None:
        # Process died
        print("ERROR: Server failed to start")
        stdout, _ = process.communicate()
        if stdout:
            print(stdout)
        return 1
    
    if not wait_for_server(args.port, timeout=15):
        print("ERROR: Server not responding")
        process.terminate()
        return 1
    
    print(f"✓ RealSense server running at http://127.0.0.1:{args.port}")
    print(f"  Stream URL: http://127.0.0.1:{args.port}/stream")
    print(f"  Status:     http://127.0.0.1:{args.port}/status")
    
    # Add to dashboard
    if not args.no_dashboard:
        print(f"\nAdding to dashboard...")
        success = asyncio.run(add_to_dashboard(
            args.dashboard_host, args.dashboard_port,
            args.port, args.name
        ))
        
        if success:
            print(f"✓ Added '{args.name}' to dashboard")
        else:
            print("✗ Failed to add to dashboard (server still running)")
    
    print("\n" + "=" * 60)
    print("RealSense live stream is ready!")
    print("=" * 60)
    print(f"\nOpen dashboard: http://localhost:9002")
    print(f"Select '{args.name}' and click Start to begin 3DGS")
    print("\nPress Ctrl+C to stop the server")
    
    # Keep running until interrupted
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        if hasattr(os, 'killpg'):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
        print("Server stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stream server output
    try:
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                print(f"[server] {line.rstrip()}")
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
