#!/usr/bin/env python3
"""
Live demo with web viewer for AirSplatMap.

Runs Gaussian Splatting with real-time web-based visualization:
- 3D point cloud view (Three.js)
- Live rendered images
- Training metrics
- Camera input stream

Usage:
    python scripts/live_web_demo.py --engine gsplat --dataset fr1_desk
    
Then open http://localhost:8766 in your browser.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Live web viewer demo")
    parser.add_argument("--engine", "-e", default="gsplat", 
                       help="Engine to use (gsplat, graphdeco, gslam, etc.)")
    parser.add_argument("--dataset", "-d", default="rgbd_dataset_freiburg1_desk",
                       help="Dataset name")
    parser.add_argument("--dataset-root", default="../datasets/tum",
                       help="Dataset root directory")
    parser.add_argument("--max-frames", type=int, default=0,
                       help="Maximum frames to process (0 = all frames)")
    parser.add_argument("--ws-port", type=int, default=8765,
                       help="WebSocket server port")
    parser.add_argument("--http-port", type=int, default=8766,
                       help="HTTP server port for viewer")
    parser.add_argument("--update-rate", type=float, default=5.0,
                       help="Max viewer updates per second")
    
    args = parser.parse_args()
    
    # Import components
    from src.engines import get_engine, list_engines
    from src.pipeline.frames import TumRGBDSource
    from src.viewer import GaussianViewerServer, serve_viewer
    
    print("\n" + "="*60)
    print("AirSplatMap Live Web Viewer Demo")
    print("="*60)
    
    # Print available engines
    print("\nAvailable engines:")
    for name, info in list_engines().items():
        status = "✅" if info['available'] else "❌"
        print(f"  {status} {name}: {info['speed']}")
    
    print(f"\nUsing engine: {args.engine}")
    
    # Start web viewer server
    print(f"\nStarting web viewer...")
    viewer_server = GaussianViewerServer(
        port=args.ws_port,
        update_rate=args.update_rate
    )
    viewer_server.start()
    
    # Start HTTP server for viewer UI
    serve_viewer(port=args.http_port, ws_port=args.ws_port)
    
    print(f"\n🌐 Open viewer at: http://localhost:{args.http_port}")
    print("   Press Ctrl+C to stop\n")
    
    # Give servers time to start
    time.sleep(1)
    
    # Load dataset
    dataset_path = Path(args.dataset_root) / args.dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Available datasets:")
        for d in Path(args.dataset_root).iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return 1
    
    print(f"Loading dataset: {dataset_path}")
    source = TumRGBDSource(str(dataset_path))
    
    # Limit frames if requested
    all_frames = list(source)
    if args.max_frames > 0 and len(all_frames) > args.max_frames:
        all_frames = all_frames[:args.max_frames]
    
    # Get intrinsics from source or first frame
    intrinsics = source.get_intrinsics()
    if all_frames:
        intrinsics = all_frames[0].intrinsics
    
    width = int(intrinsics['width'])
    height = int(intrinsics['height'])
    
    print(f"  Frames: {len(all_frames)}")
    print(f"  Resolution: {width}x{height}")
    
    # Create engine
    engine = get_engine(args.engine)
    
    # Initialize
    engine.initialize_scene(intrinsics, {'num_frames': len(all_frames)})
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    total_loss = 0
    num_frames = len(all_frames)
    
    try:
        for i, frame in enumerate(all_frames):
            frame_start = time.time()
            
            # Add frame
            engine.add_frame(
                frame_id=frame.idx,
                rgb=frame.rgb,
                depth=frame.depth,
                pose_world_cam=frame.pose
            )
            
            # Optimize
            metrics = engine.optimize_step(n_steps=5)
            total_loss += metrics.get('loss', 0)
            
            # Update viewer
            if i % 2 == 0:  # Update every 2 frames
                # Update Gaussians
                pts = engine.get_point_cloud()
                if pts is not None and len(pts) > 0:
                    colors = getattr(engine, 'get_gaussian_colors', lambda: None)()
                    if colors is None:
                        colors = np.ones_like(pts) * 0.7  # Gray default
                    viewer_server.update_gaussians(pts, colors)
                
                # Update render
                rendered = engine.render_view(frame.pose, (width, height))
                if rendered is not None and rendered.max() > 0:
                    viewer_server.update_render(rendered, "render")
                
                # Update camera input
                viewer_server.update_camera_image(frame.rgb)
                
                # Update metrics
                elapsed = time.time() - start_time
                viewer_server.update_metrics({
                    'loss': metrics.get('loss', 0),
                    'psnr': metrics.get('psnr', 0),
                    'iteration': i,
                    'num_gaussians': engine.get_num_gaussians(),
                    'fps': (i + 1) / elapsed if elapsed > 0 else 0,
                })
            
            # Progress
            fps = 1.0 / (time.time() - frame_start)
            print(f"\rFrame {i+1}/{num_frames} | "
                  f"Gaussians: {engine.get_num_gaussians():,} | "
                  f"Loss: {metrics.get('loss', 0):.4f} | "
                  f"FPS: {fps:.1f} | "
                  f"Clients: {viewer_server.client_count}", end="")
        
        print("\n\nTraining complete!")
        
        # Final refinement
        print("Running final refinement...")
        for i in range(100):
            metrics = engine.optimize_step(n_steps=1)
            if i % 10 == 0:
                pts = engine.get_point_cloud()
                if pts is not None:
                    colors = getattr(engine, 'get_gaussian_colors', lambda: None)()
                    if colors is None:
                        colors = np.ones_like(pts) * 0.7
                    viewer_server.update_gaussians(pts, colors)
                viewer_server.update_metrics({
                    'loss': metrics.get('loss', 0),
                    'iteration': num_frames + i,
                    'num_gaussians': engine.get_num_gaussians(),
                })
        
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f}s")
        print(f"Average FPS: {num_frames/elapsed:.1f}")
        print(f"Final Gaussians: {engine.get_num_gaussians():,}")
        
        # Keep server running
        print(f"\n🎉 Training complete! Viewer remains active.")
        print(f"   Open http://localhost:{args.http_port} to explore")
        print("   Press Ctrl+C to exit\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    viewer_server.stop()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
