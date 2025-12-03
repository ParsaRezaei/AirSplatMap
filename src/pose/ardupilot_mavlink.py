#!/usr/bin/env python3
"""
ArduPilot MAVLink Pose Provider
===============================

Connects to ArduPilot (or any MAVLink autopilot) to get real-time pose data
from drones, rovers, boats, etc.

Supports:
- Position (LOCAL_POSITION_NED, GLOBAL_POSITION_INT)
- Attitude (ATTITUDE, ATTITUDE_QUATERNION)
- GPS (GPS_RAW_INT)
- System status

Connection options:
- Serial: /dev/ttyUSB0, COM3, etc.
- UDP: udpin:0.0.0.0:14550, udpout:localhost:14550
- TCP: tcp:localhost:5760
- SITL: tcp:127.0.0.1:5762

Usage:
    from src.pose.ardupilot_mavlink import ArduPilotPoseProvider
    
    # Connect to SITL
    provider = ArduPilotPoseProvider("tcp:127.0.0.1:5762")
    provider.start()
    
    # Get current pose
    pose = provider.get_pose()  # 4x4 camera-to-world matrix
    
    # Get raw data
    position = provider.get_position()  # [x, y, z] in NED frame
    attitude = provider.get_attitude()  # [roll, pitch, yaw] in radians
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArduPilotState:
    """Current state from ArduPilot."""
    timestamp: float = 0.0
    
    # Position in NED frame (meters)
    position_ned: Optional[np.ndarray] = None  # [north, east, down]
    velocity_ned: Optional[np.ndarray] = None  # [vn, ve, vd]
    
    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # Attitude as quaternion [w, x, y, z]
    quaternion: Optional[np.ndarray] = None
    
    # GPS (if available)
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    gps_fix: int = 0
    satellites: int = 0
    
    # Status
    armed: bool = False
    mode: str = ""
    battery_voltage: float = 0.0
    battery_remaining: int = 0


class ArduPilotPoseProvider:
    """
    Provides real-time pose from ArduPilot via MAVLink.
    
    Converts ArduPilot's NED (North-East-Down) coordinate frame to
    the camera coordinate frame used by AirSplatMap.
    
    Frame conventions:
    - ArduPilot: NED (North-East-Down), X=North, Y=East, Z=Down
    - AirSplatMap: Camera frame, X=Right, Y=Down, Z=Forward (OpenCV convention)
    """
    
    def __init__(
        self,
        connection_string: str = "udpin:0.0.0.0:14550",
        source_system: int = 255,
        source_component: int = 0,
        camera_offset: Optional[np.ndarray] = None,
        camera_rotation: Optional[np.ndarray] = None,
        use_visual_odometry: bool = False,
    ):
        """
        Initialize ArduPilot connection.
        
        Args:
            connection_string: MAVLink connection string
                - Serial: "/dev/ttyUSB0" or "COM3"
                - UDP in: "udpin:0.0.0.0:14550"
                - UDP out: "udpout:localhost:14550"
                - TCP: "tcp:localhost:5760"
            source_system: MAVLink system ID
            source_component: MAVLink component ID
            camera_offset: [x, y, z] offset from vehicle center to camera (meters, body frame)
            camera_rotation: 3x3 rotation from body to camera frame
            use_visual_odometry: Send visual odometry back to ArduPilot (for EKF fusion)
        """
        self._connection_string = connection_string
        self._source_system = source_system
        self._source_component = source_component
        self._use_visual_odometry = use_visual_odometry
        
        # Camera mounting transform (body -> camera)
        self._camera_offset = camera_offset if camera_offset is not None else np.zeros(3)
        if camera_rotation is not None:
            self._camera_rotation = camera_rotation
        else:
            # Default: camera pointing forward, typical drone gimbal mount
            # Rotates from body frame (X=forward, Y=right, Z=down) to camera (X=right, Y=down, Z=forward)
            self._camera_rotation = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ], dtype=np.float64)
        
        # State
        self._state = ArduPilotState()
        self._state_lock = threading.Lock()
        
        # Connection
        self._mavlink = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # Origin for local positioning
        self._home_position: Optional[np.ndarray] = None
        self._origin_set = False
        
        # Pose history for velocity estimation
        self._pose_history: List[Tuple[float, np.ndarray]] = []
        self._max_history = 10
        
    def start(self) -> bool:
        """Start the MAVLink connection and receive thread."""
        try:
            from pymavlink import mavutil
        except ImportError:
            raise ImportError(
                "pymavlink not installed. Install with: pip install pymavlink"
            )
        
        logger.info(f"Connecting to ArduPilot at {self._connection_string}")
        
        try:
            self._mavlink = mavutil.mavlink_connection(
                self._connection_string,
                source_system=self._source_system,
                source_component=self._source_component,
            )
            
            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            self._mavlink.wait_heartbeat(timeout=30)
            logger.info(f"Connected to system {self._mavlink.target_system}, "
                       f"component {self._mavlink.target_component}")
            
            # Request data streams
            self._request_data_streams()
            
            # Start receive thread
            self._running = True
            self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._recv_thread.start()
            
            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def stop(self):
        """Stop the connection."""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        if self._mavlink:
            self._mavlink.close()
            self._mavlink = None
        logger.info("ArduPilot connection closed")
    
    def _request_data_streams(self):
        """Request necessary data streams from ArduPilot."""
        if not self._mavlink:
            return
        
        from pymavlink import mavutil
        
        # Request position and attitude at high rate
        self._mavlink.mav.request_data_stream_send(
            self._mavlink.target_system,
            self._mavlink.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10,  # 10 Hz
            1    # Enable
        )
        
        self._mavlink.mav.request_data_stream_send(
            self._mavlink.target_system,
            self._mavlink.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # Attitude
            50,  # 50 Hz
            1
        )
        
        self._mavlink.mav.request_data_stream_send(
            self._mavlink.target_system,
            self._mavlink.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
            2,   # 2 Hz
            1
        )
        
        logger.info("Requested data streams")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        from pymavlink import mavutil
        
        while self._running and self._mavlink:
            try:
                self._mavlink.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
            time.sleep(1.0)
    
    def _receive_loop(self):
        """Main receive loop for MAVLink messages."""
        while self._running and self._mavlink:
            try:
                msg = self._mavlink.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                
                with self._state_lock:
                    self._state.timestamp = time.time()
                    
                    if msg_type == 'ATTITUDE':
                        self._state.roll = msg.roll
                        self._state.pitch = msg.pitch
                        self._state.yaw = msg.yaw
                        
                    elif msg_type == 'ATTITUDE_QUATERNION':
                        self._state.quaternion = np.array([msg.q1, msg.q2, msg.q3, msg.q4])
                        
                    elif msg_type == 'LOCAL_POSITION_NED':
                        self._state.position_ned = np.array([msg.x, msg.y, msg.z])
                        self._state.velocity_ned = np.array([msg.vx, msg.vy, msg.vz])
                        
                    elif msg_type == 'GLOBAL_POSITION_INT':
                        self._state.lat = msg.lat / 1e7
                        self._state.lon = msg.lon / 1e7
                        self._state.alt = msg.alt / 1000.0
                        
                        # Convert to local if we have home
                        if not self._origin_set:
                            self._home_position = np.array([
                                self._state.lat, self._state.lon, self._state.alt
                            ])
                            self._origin_set = True
                            logger.info(f"Set origin: {self._home_position}")
                        
                    elif msg_type == 'GPS_RAW_INT':
                        self._state.gps_fix = msg.fix_type
                        self._state.satellites = msg.satellites_visible
                        
                    elif msg_type == 'HEARTBEAT':
                        self._state.armed = (msg.base_mode & 128) != 0
                        
                    elif msg_type == 'SYS_STATUS':
                        self._state.battery_voltage = msg.voltage_battery / 1000.0
                        self._state.battery_remaining = msg.battery_remaining
                        
            except Exception as e:
                if self._running:
                    logger.debug(f"Receive error: {e}")
    
    def get_state(self) -> ArduPilotState:
        """Get current state (thread-safe copy)."""
        with self._state_lock:
            return ArduPilotState(
                timestamp=self._state.timestamp,
                position_ned=self._state.position_ned.copy() if self._state.position_ned is not None else None,
                velocity_ned=self._state.velocity_ned.copy() if self._state.velocity_ned is not None else None,
                roll=self._state.roll,
                pitch=self._state.pitch,
                yaw=self._state.yaw,
                quaternion=self._state.quaternion.copy() if self._state.quaternion is not None else None,
                lat=self._state.lat,
                lon=self._state.lon,
                alt=self._state.alt,
                gps_fix=self._state.gps_fix,
                satellites=self._state.satellites,
                armed=self._state.armed,
                mode=self._state.mode,
                battery_voltage=self._state.battery_voltage,
                battery_remaining=self._state.battery_remaining,
            )
    
    def get_position(self) -> Optional[np.ndarray]:
        """Get position in NED frame (meters)."""
        with self._state_lock:
            return self._state.position_ned.copy() if self._state.position_ned is not None else None
    
    def get_attitude(self) -> Tuple[float, float, float]:
        """Get attitude as (roll, pitch, yaw) in radians."""
        with self._state_lock:
            return (self._state.roll, self._state.pitch, self._state.yaw)
    
    def get_pose(self) -> np.ndarray:
        """
        Get current camera pose as 4x4 transformation matrix.
        
        Returns camera-to-world transform suitable for 3DGS mapping.
        Converts from ArduPilot NED frame to camera frame.
        """
        state = self.get_state()
        
        # Build rotation from attitude
        if state.quaternion is not None:
            R_body = self._quat_to_rotation_matrix(state.quaternion)
        else:
            R_body = self._euler_to_rotation_matrix(state.roll, state.pitch, state.yaw)
        
        # Get position (use origin-relative if available)
        if state.position_ned is not None:
            pos_ned = state.position_ned
        else:
            pos_ned = np.zeros(3)
        
        # Convert NED to world frame (flip Z for up)
        # NED: X=North, Y=East, Z=Down
        # World: X=East, Y=North, Z=Up (common 3D convention)
        pos_world = np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])
        
        # Convert body rotation to world frame
        R_ned_to_world = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ], dtype=np.float64)
        
        R_world = R_ned_to_world @ R_body
        
        # Apply camera mounting transform
        R_camera = R_world @ self._camera_rotation.T
        t_camera = pos_world + R_world @ self._camera_offset
        
        # Build 4x4 transformation matrix
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_camera
        pose[:3, 3] = t_camera
        
        return pose
    
    def send_visual_odometry(self, pose: np.ndarray, timestamp_us: int):
        """
        Send visual odometry to ArduPilot for EKF fusion.
        
        Args:
            pose: 4x4 camera-to-world transformation
            timestamp_us: Timestamp in microseconds
        """
        if not self._mavlink or not self._use_visual_odometry:
            return
        
        # Extract position and orientation
        pos = pose[:3, 3]
        R = pose[:3, :3]
        
        # Convert to NED frame
        R_world_to_ned = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ], dtype=np.float64)
        
        pos_ned = R_world_to_ned @ pos
        R_ned = R_world_to_ned @ R @ self._camera_rotation
        
        # Send VISION_POSITION_ESTIMATE
        self._mavlink.mav.vision_position_estimate_send(
            timestamp_us,
            pos_ned[0], pos_ned[1], pos_ned[2],
            *self._rotation_matrix_to_euler(R_ned),
            covariance=None
        )
    
    def is_connected(self) -> bool:
        """Check if connected to ArduPilot."""
        return self._running and self._mavlink is not None
    
    @staticmethod
    def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ], dtype=np.float64)
        
        return R
    
    @staticmethod
    def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        
        return R
    
    @staticmethod
    def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
        pitch = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
        
        if abs(R[2, 0]) < 0.999:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            yaw = 0.0
        
        return roll, pitch, yaw


# Convenience function
def create_ardupilot_provider(
    connection: str = "udpin:0.0.0.0:14550",
    camera_pitch_deg: float = -45.0,
) -> ArduPilotPoseProvider:
    """
    Create an ArduPilot pose provider with common settings.
    
    Args:
        connection: MAVLink connection string
        camera_pitch_deg: Camera pitch angle (negative = looking down)
    
    Returns:
        Configured ArduPilotPoseProvider
    """
    pitch_rad = np.radians(camera_pitch_deg)
    camera_rotation = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ]) @ np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])
    
    return ArduPilotPoseProvider(
        connection_string=connection,
        camera_rotation=camera_rotation,
    )
