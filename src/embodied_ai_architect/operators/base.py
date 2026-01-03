"""Base classes for runnable operators.

Provides the Operator interface for composable pipeline components
that can be benchmarked and executed on different hardware targets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time
import statistics


@dataclass
class OperatorResult:
    """Result of an operator benchmark run."""

    operator_id: str
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    iterations: int
    warmup_iterations: int
    execution_target: str


@dataclass
class OperatorConfig:
    """Configuration for an operator instance."""

    operator_id: str
    execution_target: str = "cpu"
    rate_hz: float | None = None
    params: dict[str, Any] = field(default_factory=dict)


class Operator(ABC):
    """Base class for runnable pipeline operators.

    Operators are composable processing units that can be:
    - Configured with parameters
    - Executed on different hardware targets (CPU, GPU, NPU)
    - Benchmarked for latency and throughput
    - Connected via dataflow edges in an architecture
    """

    def __init__(self, operator_id: str):
        """Initialize operator with its catalog ID.

        Args:
            operator_id: Operator ID matching embodied-schemas catalog
        """
        self.operator_id = operator_id
        self._is_setup = False
        self._execution_target = "cpu"
        self._config: dict[str, Any] = {}

    @property
    def execution_target(self) -> str:
        """Get the current execution target."""
        return self._execution_target

    @abstractmethod
    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize operator with configuration.

        Args:
            config: Operator-specific configuration parameters
            execution_target: Target device (cpu, gpu, npu)
        """
        pass

    @abstractmethod
    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process inputs and return outputs.

        Args:
            inputs: Dictionary of input values keyed by port name

        Returns:
            Dictionary of output values keyed by port name
        """
        pass

    def teardown(self) -> None:
        """Release resources. Override if cleanup is needed."""
        pass

    def benchmark(
        self,
        sample_input: dict[str, Any],
        iterations: int = 100,
        warmup: int = 10,
    ) -> OperatorResult:
        """Run benchmark and return timing statistics.

        Args:
            sample_input: Representative input for benchmarking
            iterations: Number of timed iterations
            warmup: Number of warmup iterations (not timed)

        Returns:
            OperatorResult with timing statistics
        """
        if not self._is_setup:
            raise RuntimeError(f"Operator {self.operator_id} not setup")

        # Warmup
        for _ in range(warmup):
            self.process(sample_input)

        # Timed runs
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            self.process(sample_input)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            latencies.append(elapsed_ms)

        # Calculate statistics
        latencies.sort()
        n = len(latencies)

        return OperatorResult(
            operator_id=self.operator_id,
            mean_latency_ms=statistics.mean(latencies),
            std_latency_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            min_latency_ms=latencies[0],
            max_latency_ms=latencies[-1],
            p50_latency_ms=latencies[int(n * 0.50)],
            p95_latency_ms=latencies[int(n * 0.95)],
            p99_latency_ms=latencies[int(n * 0.99)],
            iterations=iterations,
            warmup_iterations=warmup,
            execution_target=self._execution_target,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.operator_id}, target={self._execution_target})"
