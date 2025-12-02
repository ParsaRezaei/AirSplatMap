#!/usr/bin/env python3
"""Add RealSense camera to web dashboard via WebSocket API."""
import asyncio
import websockets
import json

async def main():
    print("Connecting to dashboard...")
    # Use 127.0.0.1 to avoid Windows DNS delay
    async with websockets.connect('ws://127.0.0.1:9003') as ws:
        # Get init message
        init = json.loads(await ws.recv())
        print(f"Connected. Current datasets: {[d['name'] for d in init.get('datasets', [])]}")
        
        # Add RealSense source with 127.0.0.1 (avoids Windows localhost DNS delay)
        await ws.send(json.dumps({
            'cmd': 'add_live',
            'name': 'RealSense D435I',
            'source': 'http://127.0.0.1:8554/stream',
            'type': 'LIVE',
            'pose_method': 'ground_truth',
            'depth_method': 'ground_truth'
        }))
        
        # Get response
        resp = json.loads(await ws.recv())
        if resp.get('type') == 'datasets':
            print(f"Success! Datasets: {[d['name'] for d in resp.get('datasets', [])]}")
        else:
            print(f"Response: {resp}")

if __name__ == '__main__':
    asyncio.run(main())
