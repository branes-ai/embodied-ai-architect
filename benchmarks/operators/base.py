"""Base classes for operator benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time
import statistics


@dataclass
class OperatorBenchmarkResult:
    """Result from benchmarking an operator.

    Matches the OperatorPerfProfile schema in embodied-schemas for
    direct integration with the operator catalog.
    """

    operator_id: str
    hardware_id: str
    execution_target: str  # cpu, gpu, npu

    # Timing metrics
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Resource metrics
    memory_mb: float | None = None
    power_w: float | None = None
    throughput_fps: float | None = None

    # Metadata
    conditions: str = ""
    iterations: int = 100
    warmup_iterations: int = 10
    measured: bool = True

    # Raw timings for analysis
    raw_timings_ms: list[float] = field(default_factory=list)

    def to_perf_profile(self) -> dict:
        """Convert to operator catalog perf_profile format."""
        return {
            "hardware_id": self.hardware_id,
            "execution_target": self.execution_target,
            "latency_ms": round(self.mean_latency_ms, 2),
            "memory_mb": self.memory_mb,
            "power_w": self.power_w,
            "throughput_fps": round(self.throughput_fps, 1) if self.throughput_fps else None,
            "conditions": self.conditions,
            "measured": self.measured,
        }


class OperatorBenchmark(ABC):
    """Base class for operator benchmarks.

    Subclasses implement specific operators (YOLO, ByteTrack, etc.)
    and can be run on different execution targets.
    """

    def __init__(self, operator_id: str, config: dict | None = None):
        self.operator_id = operator_id
        self.config = config or {}
        self._is_setup = False

    @abstractmethod
    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize the operator for the specified execution target.

        Args:
            execution_target: Where to run (cpu, gpu, npu)
        """
        pass

    @abstractmethod
    def create_sample_input(self) -> dict:
        """Create sample input data for benchmarking.

        Returns:
            Dictionary of input port name -> data
        """
        pass

    @abstractmethod
    def run_once(self, inputs: dict) -> dict:
        """Execute the operator once.

        Args:
            inputs: Input data dictionary

        Returns:
            Output data dictionary
        """
        pass

    def teardown(self) -> None:
        """Clean up resources after benchmarking."""
        pass

    def benchmark(
        self,
        hardware_id: str,
        execution_target: str = "cpu",
        iterations: int = 100,
        warmup_iterations: int = 10,
        conditions: str = "",
    ) -> OperatorBenchmarkResult:
        """Run benchmark and collect metrics.

        Args:
            hardware_id: ID of the hardware platform
            execution_target: Execution target (cpu, gpu, npu)
            iterations: Number of timed iterations
            warmup_iterations: Number of warmup iterations
            conditions: Description of test conditions

        Returns:
            OperatorBenchmarkResult with timing metrics
        """
        # Setup if needed
        if not self._is_setup:
            self.setup(execution_target)
            self._is_setup = True

        # Create sample input
        inputs = self.create_sample_input()

        # Warmup
        for _ in range(warmup_iterations):
            self.run_once(inputs)

        # Synchronize if GPU/NPU
        self._synchronize(execution_target)

        # Timed runs
        timings_ns = []
        for _ in range(iterations):
            self._synchronize(execution_target)
            start = time.perf_counter_ns()
            self.run_once(inputs)
            self._synchronize(execution_target)
            timings_ns.append(time.perf_counter_ns() - start)

        # Convert to ms
        timings_ms = [t / 1e6 for t in timings_ns]

        # Calculate statistics
        sorted_timings = sorted(timings_ms)
        p50_idx = int(0.50 * len(sorted_timings))
        p95_idx = int(0.95 * len(sorted_timings))
        p99_idx = int(0.99 * len(sorted_timings))

        mean_latency = statistics.mean(timings_ms)
        throughput = 1000.0 / mean_latency if mean_latency > 0 else 0

        return OperatorBenchmarkResult(
            operator_id=self.operator_id,
            hardware_id=hardware_id,
            execution_target=execution_target,
            mean_latency_ms=mean_latency,
            std_latency_ms=statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0,
            min_latency_ms=min(timings_ms),
            max_latency_ms=max(timings_ms),
            p50_latency_ms=sorted_timings[p50_idx],
            p95_latency_ms=sorted_timings[p95_idx],
            p99_latency_ms=sorted_timings[min(p99_idx, len(sorted_timings) - 1)],
            memory_mb=self._get_memory_mb(),
            power_w=None,  # Requires external measurement
            throughput_fps=throughput,
            conditions=conditions,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            raw_timings_ms=timings_ms,
        )

    def _synchronize(self, execution_target: str) -> None:
        """Synchronize execution for accurate timing."""
        if execution_target == "gpu":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        # NPU synchronization would go here

    def _get_memory_mb(self) -> float | None:
        """Get current memory usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e6
        except ImportError:
            pass
        return None
