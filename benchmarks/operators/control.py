"""Control operator benchmarks.

Benchmarks for PID controllers, path planners, and motion control.
"""

import numpy as np
from .base import OperatorBenchmark
import heapq


class PIDControllerBenchmark(OperatorBenchmark):
    """Benchmark for PID controller.

    Runs at high frequency (100Hz+) for real-time control.
    """

    def __init__(
        self,
        num_controllers: int = 6,  # Typical for 6-DOF robot
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.01,
        dt: float = 0.01,
    ):
        """Initialize PID benchmark.

        Args:
            num_controllers: Number of parallel controllers
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Sample time in seconds
        """
        super().__init__(
            operator_id="pid_controller",
            config={
                "num_controllers": num_controllers,
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "dt": dt,
            },
        )
        self.num_controllers = num_controllers
        self.controllers = []

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize controllers."""
        if execution_target != "cpu":
            print(f"  Warning: PID runs on CPU, ignoring target={execution_target}")

        self.controllers = [
            SimplePID(
                kp=self.config["kp"],
                ki=self.config["ki"],
                kd=self.config["kd"],
                dt=self.config["dt"],
            )
            for _ in range(self.num_controllers)
        ]
        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample setpoints and measurements."""
        setpoints = np.random.rand(self.num_controllers).astype(np.float32)
        measurements = np.random.rand(self.num_controllers).astype(np.float32)
        return {"setpoints": setpoints, "measurements": measurements}

    def run_once(self, inputs: dict) -> dict:
        """Run all controllers."""
        setpoints = inputs["setpoints"]
        measurements = inputs["measurements"]

        outputs = np.zeros(self.num_controllers, dtype=np.float32)
        for i, (pid, sp, meas) in enumerate(
            zip(self.controllers, setpoints, measurements)
        ):
            outputs[i] = pid.compute(sp, meas)

        return {"outputs": outputs}

    def teardown(self) -> None:
        """Clean up."""
        self.controllers = []


class SimplePID:
    """Simple PID controller implementation."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.01,
        output_min: float = -1.0,
        output_max: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute control output."""
        error = setpoint - measurement

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral,
            self.output_min / (self.ki + 1e-6),
            self.output_max / (self.ki + 1e-6),
        )
        i_term = self.ki * self.integral

        # Derivative
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error

        # Total output
        output = p_term + i_term + d_term
        return np.clip(output, self.output_min, self.output_max)


class PathPlannerAStarBenchmark(OperatorBenchmark):
    """Benchmark for A* path planner.

    Plans paths on occupancy grids.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (100, 100),
        obstacle_density: float = 0.2,
        diagonal_movement: bool = True,
    ):
        """Initialize A* benchmark.

        Args:
            grid_size: Grid dimensions (width, height)
            obstacle_density: Fraction of cells that are obstacles
            diagonal_movement: Allow diagonal moves
        """
        super().__init__(
            operator_id="path_planner_astar",
            config={
                "grid_size": grid_size,
                "obstacle_density": obstacle_density,
                "diagonal_movement": diagonal_movement,
            },
        )
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.diagonal = diagonal_movement
        self.grid = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize occupancy grid."""
        if execution_target != "cpu":
            print(f"  Warning: A* runs on CPU, ignoring target={execution_target}")

        # Create random occupancy grid
        self.grid = np.random.rand(*self.grid_size) < self.obstacle_density
        self.grid = self.grid.astype(np.uint8) * 255

        # Ensure start and goal are free
        self.grid[0, 0] = 0
        self.grid[-1, -1] = 0

        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample start and goal."""
        # Random start and goal (ensuring they're free)
        while True:
            start = (
                np.random.randint(0, self.grid_size[0] // 4),
                np.random.randint(0, self.grid_size[1] // 4),
            )
            if self.grid[start[0], start[1]] == 0:
                break

        while True:
            goal = (
                np.random.randint(3 * self.grid_size[0] // 4, self.grid_size[0]),
                np.random.randint(3 * self.grid_size[1] // 4, self.grid_size[1]),
            )
            if self.grid[goal[0], goal[1]] == 0:
                break

        return {"start": start, "goal": goal, "grid": self.grid}

    def run_once(self, inputs: dict) -> dict:
        """Run A* planning."""
        start = inputs["start"]
        goal = inputs["goal"]
        grid = inputs["grid"]

        path, success = self._astar(grid, start, goal)

        return {
            "path": path,
            "success": success,
            "path_length": len(path) if path else 0,
        }

    def _astar(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[list | None, bool]:
        """A* path planning algorithm."""
        rows, cols = grid.shape

        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        came_from: dict[tuple, tuple] = {}

        g_score: dict[tuple, float] = {start: 0}
        f_score: dict[tuple, float] = {start: self._heuristic(start, goal)}

        closed_set: set[tuple] = set()

        # Movement directions
        if self.diagonal:
            directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
            costs = [1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414]
        else:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            costs = [1, 1, 1, 1]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path)), True

            if current in closed_set:
                continue
            closed_set.add(current)

            for (dr, dc), cost in zip(directions, costs):
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue

                # Check obstacle
                if grid[neighbor[0], neighbor[1]] > 0:
                    continue

                # Check if already processed
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None, False

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def teardown(self) -> None:
        """Clean up."""
        self.grid = None


class TrajectoryFollowerBenchmark(OperatorBenchmark):
    """Benchmark for trajectory following controller.

    Computes control commands to follow a reference trajectory.
    """

    def __init__(
        self,
        trajectory_length: int = 100,
        lookahead_distance: float = 1.0,
        dt: float = 0.01,
    ):
        """Initialize trajectory follower benchmark.

        Args:
            trajectory_length: Number of waypoints
            lookahead_distance: Lookahead for pure pursuit
            dt: Control timestep
        """
        super().__init__(
            operator_id="trajectory_follower",
            config={
                "trajectory_length": trajectory_length,
                "lookahead_distance": lookahead_distance,
                "dt": dt,
            },
        )
        self.trajectory_length = trajectory_length
        self.lookahead = lookahead_distance
        self.dt = dt
        self.trajectory = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize reference trajectory."""
        if execution_target != "cpu":
            print(f"  Warning: Trajectory follower runs on CPU, ignoring target={execution_target}")

        # Generate smooth reference trajectory
        t = np.linspace(0, 2 * np.pi, self.trajectory_length)
        self.trajectory = np.column_stack([
            5 * np.cos(t),  # x
            5 * np.sin(t),  # y
            t,              # heading
        ]).astype(np.float32)

        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample robot state."""
        # Current pose: [x, y, theta]
        pose = np.array([
            np.random.rand() * 10 - 5,
            np.random.rand() * 10 - 5,
            np.random.rand() * 2 * np.pi,
        ], dtype=np.float32)

        return {"pose": pose, "trajectory": self.trajectory}

    def run_once(self, inputs: dict) -> dict:
        """Compute control command."""
        pose = inputs["pose"]
        trajectory = inputs["trajectory"]

        # Find lookahead point
        lookahead_point = self._find_lookahead(pose, trajectory)

        # Pure pursuit control
        v, omega = self._pure_pursuit(pose, lookahead_point)

        return {"velocity": v, "angular_velocity": omega}

    def _find_lookahead(
        self,
        pose: np.ndarray,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """Find lookahead point on trajectory."""
        x, y, _ = pose

        # Compute distances to all waypoints
        dx = trajectory[:, 0] - x
        dy = trajectory[:, 1] - y
        distances = np.sqrt(dx**2 + dy**2)

        # Find closest point
        closest_idx = np.argmin(distances)

        # Find lookahead point
        for i in range(closest_idx, len(trajectory)):
            if distances[i] >= self.lookahead:
                return trajectory[i]

        # If no point far enough, use last point
        return trajectory[-1]

    def _pure_pursuit(
        self,
        pose: np.ndarray,
        lookahead: np.ndarray,
    ) -> tuple[float, float]:
        """Pure pursuit steering law."""
        x, y, theta = pose
        lx, ly, _ = lookahead

        # Transform to robot frame
        dx = lx - x
        dy = ly - y
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        # Curvature
        L = np.sqrt(local_x**2 + local_y**2)
        if L < 0.01:
            return 0.0, 0.0

        curvature = 2 * local_y / (L**2)

        # Control outputs
        v = 1.0  # Constant velocity
        omega = v * curvature

        return float(v), float(omega)

    def teardown(self) -> None:
        """Clean up."""
        self.trajectory = None
