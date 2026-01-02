"""State estimation operator benchmarks.

Benchmarks for Kalman filters, scene graph managers, and other
state estimation components.
"""

import numpy as np
from .base import OperatorBenchmark


class KalmanFilter2DBboxBenchmark(OperatorBenchmark):
    """Benchmark for 2D bounding box Kalman filter.

    Used in tracking for motion prediction and smoothing.
    """

    def __init__(
        self,
        num_filters: int = 50,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ):
        """Initialize Kalman filter benchmark.

        Args:
            num_filters: Number of parallel filter instances
            process_noise: Process noise covariance scale
            measurement_noise: Measurement noise covariance scale
        """
        super().__init__(
            operator_id="kalman_filter_2d_bbox",
            config={
                "num_filters": num_filters,
                "process_noise": process_noise,
                "measurement_noise": measurement_noise,
            },
        )
        self.num_filters = num_filters
        self.filters = []

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize filters."""
        if execution_target != "cpu":
            print(f"  Warning: Kalman filter runs on CPU, ignoring target={execution_target}")

        # Create filter instances
        self.filters = [
            SimpleKalmanFilter2D(
                process_noise=self.config["process_noise"],
                measurement_noise=self.config["measurement_noise"],
            )
            for _ in range(self.num_filters)
        ]
        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample bbox measurements."""
        # [x, y, w, h] for each filter
        bboxes = np.random.rand(self.num_filters, 4).astype(np.float32) * 640
        return {"bboxes": bboxes}

    def run_once(self, inputs: dict) -> dict:
        """Run predict and update for all filters."""
        bboxes = inputs["bboxes"]
        predictions = []

        for i, (kf, bbox) in enumerate(zip(self.filters, bboxes)):
            # Predict
            pred = kf.predict()
            # Update with measurement
            kf.update(bbox)
            predictions.append(pred)

        return {"predictions": np.array(predictions)}

    def teardown(self) -> None:
        """Clean up."""
        self.filters = []


class SimpleKalmanFilter2D:
    """Simple Kalman filter for 2D bounding box tracking.

    State: [cx, cy, s, r, vx, vy, vs, vr]
    - cx, cy: center position
    - s: scale (area)
    - r: aspect ratio
    - vx, vy, vs, vr: velocities
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 1.0):
        # State dimension
        self.dim_x = 8
        self.dim_z = 4

        # State vector
        self.x = np.zeros(self.dim_x, dtype=np.float32)

        # State covariance
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10

        # State transition matrix
        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[:4, 4:] = np.eye(4)

        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[:4, :4] = np.eye(4)

        # Process noise
        self.Q = np.eye(self.dim_x, dtype=np.float32) * process_noise

        # Measurement noise
        self.R = np.eye(self.dim_z, dtype=np.float32) * measurement_noise

    def predict(self) -> np.ndarray:
        """Predict next state."""
        # x = F @ x
        self.x = self.F @ self.x
        # P = F @ P @ F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z: np.ndarray) -> None:
        """Update with measurement."""
        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P


class SceneGraphManagerBenchmark(OperatorBenchmark):
    """Benchmark for scene graph manager.

    Manages 3D object state, trajectory history, and spatial relationships.
    """

    def __init__(
        self,
        num_objects: int = 50,
        max_history: int = 100,
        ttl_seconds: float = 5.0,
    ):
        """Initialize scene graph benchmark.

        Args:
            num_objects: Number of tracked objects
            max_history: Maximum trajectory history length
            ttl_seconds: Time-to-live for objects
        """
        super().__init__(
            operator_id="scene_graph_manager",
            config={
                "num_objects": num_objects,
                "max_history": max_history,
                "ttl_seconds": ttl_seconds,
            },
        )
        self.num_objects = num_objects
        self.manager = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize scene graph."""
        if execution_target != "cpu":
            print(f"  Warning: Scene graph runs on CPU, ignoring target={execution_target}")

        self.manager = SimpleSceneGraph(
            max_history=self.config["max_history"],
            ttl_seconds=self.config["ttl_seconds"],
        )

        # Pre-populate with objects
        for i in range(self.num_objects):
            self.manager.add_object(
                track_id=i,
                position=np.random.rand(3).astype(np.float32) * 10,
                bbox=np.random.rand(4).astype(np.float32) * 640,
                class_id=np.random.randint(0, 80),
            )

        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample track updates."""
        tracks = []
        for i in range(self.num_objects):
            tracks.append({
                "track_id": i,
                "bbox": np.random.rand(4).astype(np.float32) * 640,
                "class_id": np.random.randint(0, 80),
                "confidence": np.random.rand(),
            })
        return {"tracks": tracks}

    def run_once(self, inputs: dict) -> dict:
        """Update scene graph with tracks."""
        tracks = inputs["tracks"]

        for track in tracks:
            # Estimate 3D position from bbox (simplified)
            bbox = track["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            depth = 640 / (bbox[3] - bbox[1] + 1)  # Inverse height heuristic

            self.manager.update_object(
                track_id=track["track_id"],
                position=np.array([cx * depth / 640, cy * depth / 640, depth]),
                bbox=track["bbox"],
            )

        # Get current objects
        objects = self.manager.get_objects()

        return {"objects": objects}

    def teardown(self) -> None:
        """Clean up."""
        self.manager = None


class SimpleSceneGraph:
    """Simplified scene graph for benchmarking."""

    def __init__(self, max_history: int = 100, ttl_seconds: float = 5.0):
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self.objects: dict[int, dict] = {}

    def add_object(
        self,
        track_id: int,
        position: np.ndarray,
        bbox: np.ndarray,
        class_id: int,
    ) -> None:
        """Add new object to scene graph."""
        self.objects[track_id] = {
            "track_id": track_id,
            "position": position,
            "bbox": bbox,
            "class_id": class_id,
            "trajectory": [position.copy()],
            "velocity": np.zeros(3, dtype=np.float32),
        }

    def update_object(
        self,
        track_id: int,
        position: np.ndarray,
        bbox: np.ndarray,
    ) -> None:
        """Update existing object."""
        if track_id not in self.objects:
            self.add_object(track_id, position, bbox, 0)
            return

        obj = self.objects[track_id]

        # Update velocity
        old_pos = obj["position"]
        obj["velocity"] = position - old_pos

        # Update position
        obj["position"] = position
        obj["bbox"] = bbox

        # Update trajectory
        obj["trajectory"].append(position.copy())
        if len(obj["trajectory"]) > self.max_history:
            obj["trajectory"] = obj["trajectory"][-self.max_history:]

    def get_objects(self) -> dict:
        """Get all objects."""
        return self.objects


class EKF6DOFBenchmark(OperatorBenchmark):
    """Benchmark for 6-DOF Extended Kalman Filter.

    Fuses IMU and other sensor data for pose estimation.
    """

    def __init__(
        self,
        imu_rate_hz: float = 100.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        """Initialize EKF benchmark.

        Args:
            imu_rate_hz: IMU update rate
            process_noise: Process noise scale
            measurement_noise: Measurement noise scale
        """
        super().__init__(
            operator_id="ekf_6dof",
            config={
                "imu_rate_hz": imu_rate_hz,
                "process_noise": process_noise,
                "measurement_noise": measurement_noise,
            },
        )
        self.dt = 1.0 / imu_rate_hz
        self.ekf = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize EKF."""
        if execution_target != "cpu":
            print(f"  Warning: EKF runs on CPU, ignoring target={execution_target}")

        self.ekf = SimpleEKF6DOF(
            dt=self.dt,
            process_noise=self.config["process_noise"],
            measurement_noise=self.config["measurement_noise"],
        )
        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample IMU data."""
        # [ax, ay, az, gx, gy, gz]
        accel = np.random.randn(3).astype(np.float32) * 0.1
        accel[2] += 9.81  # Add gravity
        gyro = np.random.randn(3).astype(np.float32) * 0.01
        return {"accel": accel, "gyro": gyro}

    def run_once(self, inputs: dict) -> dict:
        """Run EKF predict and update."""
        accel = inputs["accel"]
        gyro = inputs["gyro"]

        # Predict with gyro
        self.ekf.predict(gyro)

        # Update with accel
        self.ekf.update(accel)

        return {"pose": self.ekf.get_pose()}

    def teardown(self) -> None:
        """Clean up."""
        self.ekf = None


class SimpleEKF6DOF:
    """Simplified 6-DOF EKF for benchmarking.

    State: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    - Position (x, y, z)
    - Velocity (vx, vy, vz)
    - Orientation quaternion (qw, qx, qy, qz)
    """

    def __init__(
        self,
        dt: float = 0.01,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self.dt = dt

        # State dimension
        self.dim_x = 10

        # State vector
        self.x = np.zeros(self.dim_x, dtype=np.float32)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)

        # State covariance
        self.P = np.eye(self.dim_x, dtype=np.float32) * 0.1

        # Process noise
        self.Q = np.eye(self.dim_x, dtype=np.float32) * process_noise

        # Measurement noise
        self.R = np.eye(3, dtype=np.float32) * measurement_noise

    def predict(self, gyro: np.ndarray) -> None:
        """Predict step with gyroscope."""
        # Update position with velocity
        self.x[:3] += self.x[3:6] * self.dt

        # Update quaternion with gyro (simplified)
        omega = gyro * self.dt
        q = self.x[6:10]
        dq = self._omega_to_quaternion(omega)
        self.x[6:10] = self._quaternion_multiply(q, dq)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

        # Update covariance
        self.P = self.P + self.Q

    def update(self, accel: np.ndarray) -> None:
        """Update step with accelerometer."""
        # Expected gravity in body frame
        g_world = np.array([0, 0, 9.81], dtype=np.float32)
        g_body = self._rotate_vector(g_world, self.x[6:10])

        # Innovation
        y = accel - g_body

        # Simplified Kalman update
        K = 0.1  # Fixed gain for simplicity
        # Update velocity estimate
        self.x[3:6] += K * y

        # Update covariance
        self.P *= (1 - K)

    def get_pose(self) -> np.ndarray:
        """Get current pose [x, y, z, qw, qx, qy, qz]."""
        return np.concatenate([self.x[:3], self.x[6:10]])

    def _omega_to_quaternion(self, omega: np.ndarray) -> np.ndarray:
        """Convert angular velocity to quaternion delta."""
        angle = np.linalg.norm(omega)
        if angle < 1e-6:
            return np.array([1, 0, 0, 0], dtype=np.float32)
        axis = omega / angle
        return np.array(
            [
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
            ],
            dtype=np.float32,
        )

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    def _rotate_vector(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion."""
        # Simplified rotation
        qw, qx, qy, qz = q
        # Rotation matrix from quaternion
        R = np.array(
            [
                [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
                [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
            ],
            dtype=np.float32,
        )
        return R @ v
