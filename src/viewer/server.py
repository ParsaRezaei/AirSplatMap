"""
Web-based Gaussian Splatting Viewer Server.

Provides real-time streaming of:
- Gaussian point cloud data
- Rendered views from training
- Live camera input
- Training metrics

Uses WebSockets for low-latency communication.
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging
import threading
import time
import base64

logger = logging.getLogger(__name__)

# Try to import websockets
try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not installed. Install with: pip install websockets")


class GaussianViewerServer:
    """
    WebSocket server for streaming Gaussian Splatting data to web viewer.
    
    Features:
    - Stream Gaussian positions/colors in real-time
    - Stream rendered images from training
    - Stream live camera input
    - Broadcast training metrics
    
    Example:
        server = GaussianViewerServer(port=8765)
        server.start()
        
        # During training:
        server.update_gaussians(positions, colors)
        server.update_render(rendered_image)
        server.update_metrics({'loss': 0.1, 'psnr': 25.0})
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_gaussians: int = 100000,
        update_rate: float = 10.0  # Max updates per second
    ):
        """
        Initialize viewer server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            max_gaussians: Maximum Gaussians to stream (downsampled if more)
            update_rate: Maximum updates per second to clients
        """
        self.host = host
        self.port = port
        self.max_gaussians = max_gaussians
        self.min_update_interval = 1.0 / update_rate
        
        self._clients: set = set()
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Current state
        self._gaussians: Optional[Dict] = None
        self._render: Optional[np.ndarray] = None
        self._camera_image: Optional[np.ndarray] = None
        self._metrics: Dict[str, Any] = {}
        self._camera_pose: Optional[np.ndarray] = None
        
        # Timing
        self._last_gaussian_update = 0
        self._last_render_update = 0
        self._last_camera_update = 0
        
        # Callbacks
        self._on_camera_move: Optional[Callable] = None
    
    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start server: websockets not installed")
            return
        
        if self._running:
            logger.warning("Server already running")
            return
        
        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        
        logger.info(f"Viewer server starting on ws://{self.host}:{self.port}")
        logger.info(f"Open viewer at http://localhost:{self.port + 1}")
    
    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
    
    def _run_server(self) -> None:
        """Run the async server in its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self._loop.close()
    
    async def _serve(self) -> None:
        """Async server main loop."""
        async with serve(self._handle_client, self.host, self.port):
            logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
            while self._running:
                await asyncio.sleep(0.1)
    
    async def _handle_client(self, websocket) -> None:
        """Handle a connected client."""
        self._clients.add(websocket)
        logger.info(f"Client connected, total: {len(self._clients)}")
        
        try:
            # Send current state
            if self._gaussians:
                await self._send_gaussians(websocket)
            if self._render is not None:
                await self._send_render(websocket)
            if self._metrics:
                await self._send_metrics(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"Client disconnected, total: {len(self._clients)}")
    
    async def _handle_message(self, websocket, message: str) -> None:
        """Handle message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'camera_move':
                # Client moved camera - request render
                if self._on_camera_move:
                    pose = np.array(data['pose']).reshape(4, 4)
                    self._on_camera_move(pose)
            
            elif msg_type == 'request_state':
                # Client requesting current state
                if self._gaussians:
                    await self._send_gaussians(websocket)
                if self._render is not None:
                    await self._send_render(websocket)
            
            elif msg_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {message[:100]}")
        except Exception as e:
            logger.warning(f"Error handling message: {e}")
    
    def update_gaussians(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        opacities: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None
    ) -> None:
        """
        Update Gaussian point cloud data.
        
        Args:
            positions: Nx3 array of positions
            colors: Nx3 array of colors (0-1 or 0-255)
            opacities: Optional Nx1 array of opacities
            scales: Optional Nx3 array of scales
        """
        now = time.time()
        if now - self._last_gaussian_update < self.min_update_interval:
            return
        
        # Normalize colors to 0-255 range
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        
        # Downsample if too many
        n = len(positions)
        if n > self.max_gaussians:
            indices = np.random.choice(n, self.max_gaussians, replace=False)
            positions = positions[indices]
            colors = colors[indices]
            if opacities is not None:
                opacities = opacities[indices]
            if scales is not None:
                scales = scales[indices]
        
        self._gaussians = {
            'positions': positions.astype(np.float32),
            'colors': colors.astype(np.uint8),
            'opacities': opacities.astype(np.float32) if opacities is not None else None,
            'scales': scales.astype(np.float32) if scales is not None else None,
            'count': len(positions),
        }
        
        self._last_gaussian_update = now
        self._broadcast_gaussians()
    
    def update_render(self, image: np.ndarray, label: str = "render") -> None:
        """
        Update rendered image.
        
        Args:
            image: HxWx3 RGB image (uint8)
            label: Label for this render (e.g., "render", "ground_truth")
        """
        now = time.time()
        if now - self._last_render_update < self.min_update_interval:
            return
        
        self._render = image
        self._last_render_update = now
        self._broadcast_render(label)
    
    def update_camera_image(self, image: np.ndarray) -> None:
        """
        Update live camera image.
        
        Args:
            image: HxWx3 RGB image (uint8)
        """
        now = time.time()
        if now - self._last_camera_update < self.min_update_interval:
            return
        
        self._camera_image = image
        self._last_camera_update = now
        self._broadcast_render("camera")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update training metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        self._metrics.update(metrics)
        self._broadcast_metrics()
    
    def set_camera_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Set callback for when client moves camera.
        
        Args:
            callback: Function taking 4x4 pose matrix
        """
        self._on_camera_move = callback
    
    def _broadcast_gaussians(self) -> None:
        """Broadcast Gaussians to all clients."""
        if not self._clients or not self._gaussians:
            return
        
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_gaussians_async(),
                self._loop
            )
    
    async def _broadcast_gaussians_async(self) -> None:
        """Async broadcast of Gaussians."""
        for client in list(self._clients):
            try:
                await self._send_gaussians(client)
            except Exception:
                pass
    
    async def _send_gaussians(self, websocket) -> None:
        """Send Gaussians to a specific client."""
        if not self._gaussians:
            return
        
        # Pack data efficiently
        data = {
            'type': 'gaussians',
            'count': self._gaussians['count'],
            'positions': base64.b64encode(
                self._gaussians['positions'].tobytes()
            ).decode('ascii'),
            'colors': base64.b64encode(
                self._gaussians['colors'].tobytes()
            ).decode('ascii'),
        }
        
        if self._gaussians['opacities'] is not None:
            data['opacities'] = base64.b64encode(
                self._gaussians['opacities'].tobytes()
            ).decode('ascii')
        
        await websocket.send(json.dumps(data))
    
    def _broadcast_render(self, label: str) -> None:
        """Broadcast render to all clients."""
        if not self._clients:
            return
        
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_render_async(label),
                self._loop
            )
    
    async def _broadcast_render_async(self, label: str) -> None:
        """Async broadcast of render."""
        for client in list(self._clients):
            try:
                await self._send_render(client, label)
            except Exception:
                pass
    
    async def _send_render(self, websocket, label: str = "render") -> None:
        """Send render to a specific client."""
        image = self._camera_image if label == "camera" else self._render
        if image is None:
            return
        
        # Compress image to JPEG
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode('ascii')
        except ImportError:
            # Fallback: send raw (less efficient)
            image_b64 = base64.b64encode(image.tobytes()).decode('ascii')
        
        data = {
            'type': 'render',
            'label': label,
            'width': image.shape[1],
            'height': image.shape[0],
            'image': image_b64,
        }
        
        await websocket.send(json.dumps(data))
    
    def _broadcast_metrics(self) -> None:
        """Broadcast metrics to all clients."""
        if not self._clients:
            return
        
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_metrics_async(),
                self._loop
            )
    
    async def _broadcast_metrics_async(self) -> None:
        """Async broadcast of metrics."""
        data = {
            'type': 'metrics',
            **self._metrics
        }
        message = json.dumps(data)
        
        for client in list(self._clients):
            try:
                await client.send(message)
            except Exception:
                pass
    
    async def _send_metrics(self, websocket) -> None:
        """Send metrics to a specific client."""
        data = {
            'type': 'metrics',
            **self._metrics
        }
        await websocket.send(json.dumps(data))
    
    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)
    
    @property
    def is_running(self) -> bool:
        """Whether the server is running."""
        return self._running
