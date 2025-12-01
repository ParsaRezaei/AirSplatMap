"""
HTTP server for serving the web viewer client.
"""

import http.server
import socketserver
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# HTML content for the viewer
VIEWER_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AirSplatMap Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee; 
            overflow: hidden;
        }
        #container { display: flex; height: 100vh; }
        
        /* 3D Viewer */
        #viewer { flex: 1; position: relative; }
        #canvas3d { width: 100%; height: 100%; }
        
        /* Sidebar */
        #sidebar {
            width: 320px;
            background: #16213e;
            padding: 16px;
            overflow-y: auto;
            border-left: 1px solid #0f3460;
        }
        
        .panel {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .panel h3 {
            font-size: 14px;
            color: #e94560;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Status */
        #status {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff6b6b;
        }
        .status-dot.connected { background: #51cf66; }
        
        /* Metrics */
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #0f3460;
        }
        .metric:last-child { border-bottom: none; }
        .metric-value { 
            color: #4ecdc4; 
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        
        /* Renders */
        .render-container {
            position: relative;
            margin-bottom: 8px;
        }
        .render-container img {
            width: 100%;
            border-radius: 4px;
            background: #000;
        }
        .render-label {
            position: absolute;
            top: 4px;
            left: 4px;
            background: rgba(0,0,0,0.7);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
        }
        
        /* Controls */
        .control-group {
            margin-bottom: 8px;
        }
        .control-group label {
            display: block;
            font-size: 12px;
            margin-bottom: 4px;
            color: #aaa;
        }
        input[type="range"] {
            width: 100%;
        }
        button {
            background: #e94560;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            margin-top: 8px;
        }
        button:hover { background: #ff6b9d; }
        
        /* Info overlay */
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="viewer">
            <canvas id="canvas3d"></canvas>
            <div id="info">
                <div>Gaussians: <span id="gaussianCount">0</span></div>
                <div>FPS: <span id="fps">0</span></div>
                <div>Drag to rotate, scroll to zoom</div>
            </div>
        </div>
        <div id="sidebar">
            <div class="panel">
                <h3>Connection</h3>
                <div id="status">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">Disconnected</span>
                </div>
                <button id="connectBtn">Connect</button>
            </div>
            
            <div class="panel">
                <h3>Training Metrics</h3>
                <div id="metrics">
                    <div class="metric">
                        <span>Loss</span>
                        <span class="metric-value" id="metricLoss">-</span>
                    </div>
                    <div class="metric">
                        <span>PSNR</span>
                        <span class="metric-value" id="metricPsnr">-</span>
                    </div>
                    <div class="metric">
                        <span>Iteration</span>
                        <span class="metric-value" id="metricIter">-</span>
                    </div>
                    <div class="metric">
                        <span>Gaussians</span>
                        <span class="metric-value" id="metricGaussians">-</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Live Render</h3>
                <div class="render-container">
                    <img id="renderImage" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
                    <div class="render-label">Training View</div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Camera Input</h3>
                <div class="render-container">
                    <img id="cameraImage" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
                    <div class="render-label">Live Camera</div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Display Options</h3>
                <div class="control-group">
                    <label>Point Size</label>
                    <input type="range" id="pointSize" min="1" max="10" value="3">
                </div>
                <div class="control-group">
                    <label>Opacity</label>
                    <input type="range" id="opacity" min="0" max="100" value="100">
                </div>
            </div>
        </div>
    </div>

    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Configuration
        const WS_PORT = ''' + str(8765) + ''';
        const WS_URL = `ws://${window.location.hostname}:${WS_PORT}`;
        
        // Three.js setup
        let scene, camera, renderer, controls, pointCloud;
        let gaussianPositions = null;
        let gaussianColors = null;
        
        // WebSocket
        let ws = null;
        let connected = false;
        
        // FPS counter
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        
        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            const canvas = document.getElementById('canvas3d');
            camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.01, 1000);
            camera.position.set(0, 2, 5);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Axes helper
            scene.add(new THREE.AxesHelper(1));
            
            // Grid
            const grid = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
            scene.add(grid);
            
            // Initial point cloud (placeholder)
            createPointCloud([0,0,0], [255,255,255]);
            
            // Handle resize
            window.addEventListener('resize', onResize);
            
            // Start animation
            animate();
        }
        
        function createPointCloud(positions, colors) {
            // Remove old
            if (pointCloud) scene.remove(pointCloud);
            
            const geometry = new THREE.BufferGeometry();
            
            if (positions.length > 0) {
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            }
            
            const material = new THREE.PointsMaterial({
                size: parseFloat(document.getElementById('pointSize').value) * 0.01,
                vertexColors: true,
                transparent: true,
                opacity: parseFloat(document.getElementById('opacity').value) / 100,
                sizeAttenuation: true
            });
            
            pointCloud = new THREE.Points(geometry, material);
            scene.add(pointCloud);
        }
        
        function updatePointCloud(positions, colors) {
            gaussianPositions = positions;
            gaussianColors = colors;
            
            const geometry = pointCloud.geometry;
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.attributes.position.needsUpdate = true;
            geometry.attributes.color.needsUpdate = true;
            geometry.computeBoundingSphere();
            
            document.getElementById('gaussianCount').textContent = (positions.length / 3).toLocaleString();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
            
            // FPS counter
            frameCount++;
            const now = Date.now();
            if (now - lastFpsUpdate > 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
            }
        }
        
        function onResize() {
            const canvas = document.getElementById('canvas3d');
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }
        
        // WebSocket handling
        function connect() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
                return;
            }
            
            ws = new WebSocket(WS_URL);
            
            ws.onopen = () => {
                connected = true;
                document.getElementById('statusDot').classList.add('connected');
                document.getElementById('statusText').textContent = 'Connected';
                document.getElementById('connectBtn').textContent = 'Disconnect';
                console.log('Connected to server');
            };
            
            ws.onclose = () => {
                connected = false;
                document.getElementById('statusDot').classList.remove('connected');
                document.getElementById('statusText').textContent = 'Disconnected';
                document.getElementById('connectBtn').textContent = 'Connect';
                console.log('Disconnected from server');
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };
        }
        
        function handleMessage(data) {
            switch (data.type) {
                case 'gaussians':
                    handleGaussians(data);
                    break;
                case 'render':
                    handleRender(data);
                    break;
                case 'metrics':
                    handleMetrics(data);
                    break;
                case 'pong':
                    console.log('Pong received');
                    break;
            }
        }
        
        function handleGaussians(data) {
            // Decode base64 data
            const posBytes = Uint8Array.from(atob(data.positions), c => c.charCodeAt(0));
            const colBytes = Uint8Array.from(atob(data.colors), c => c.charCodeAt(0));
            
            const positions = new Float32Array(posBytes.buffer);
            const colors = new Uint8Array(colBytes.buffer);
            
            // Convert colors to 0-1 range
            const colorsNorm = new Float32Array(colors.length);
            for (let i = 0; i < colors.length; i++) {
                colorsNorm[i] = colors[i] / 255.0;
            }
            
            updatePointCloud(positions, colorsNorm);
        }
        
        function handleRender(data) {
            const imgElement = data.label === 'camera' ? 
                document.getElementById('cameraImage') : 
                document.getElementById('renderImage');
            
            imgElement.src = 'data:image/jpeg;base64,' + data.image;
        }
        
        function handleMetrics(data) {
            if (data.loss !== undefined) {
                document.getElementById('metricLoss').textContent = data.loss.toFixed(4);
            }
            if (data.psnr !== undefined) {
                document.getElementById('metricPsnr').textContent = data.psnr.toFixed(2) + ' dB';
            }
            if (data.iteration !== undefined) {
                document.getElementById('metricIter').textContent = data.iteration.toLocaleString();
            }
            if (data.num_gaussians !== undefined) {
                document.getElementById('metricGaussians').textContent = data.num_gaussians.toLocaleString();
            }
        }
        
        // Event listeners
        document.getElementById('connectBtn').addEventListener('click', connect);
        
        document.getElementById('pointSize').addEventListener('input', (e) => {
            if (pointCloud) {
                pointCloud.material.size = parseFloat(e.target.value) * 0.01;
            }
        });
        
        document.getElementById('opacity').addEventListener('input', (e) => {
            if (pointCloud) {
                pointCloud.material.opacity = parseFloat(e.target.value) / 100;
            }
        });
        
        // Initialize
        init();
        
        // Auto-connect after a short delay
        setTimeout(connect, 500);
    </script>
</body>
</html>
'''


class ViewerHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves the viewer HTML."""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(VIEWER_HTML.encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress logs


def serve_viewer(host: str = "0.0.0.0", port: int = 8766, ws_port: int = 8765) -> threading.Thread:
    """
    Start HTTP server for viewer in background thread.
    
    Args:
        host: Host to bind to
        port: HTTP port
        ws_port: WebSocket port (embedded in HTML)
        
    Returns:
        Server thread
    """
    global VIEWER_HTML
    # Update WS port in HTML
    VIEWER_HTML = VIEWER_HTML.replace("''' + str(8765) + '''", str(ws_port))
    
    handler = ViewerHTTPHandler
    server = socketserver.TCPServer((host, port), handler)
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    logger.info(f"Viewer HTTP server running at http://{host}:{port}")
    return thread
