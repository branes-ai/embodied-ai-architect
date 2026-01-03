"""Benchmark harness for embodied AI architectures.

Provides infrastructure for:
- Executing complete architecture pipelines
- Collecting timing metrics per operator
- Monitoring power consumption
- Generating benchmark reports
"""

from .runner import (
    ArchitectureRunner,
    ArchitectureBenchmarkResult,
    OperatorTiming,
    PipelineTiming,
)
from .power import (
    PowerMonitor,
    PowerMetrics,
    get_power_monitor,
)

__all__ = [
    "ArchitectureRunner",
    "ArchitectureBenchmarkResult",
    "OperatorTiming",
    "PipelineTiming",
    "PowerMonitor",
    "PowerMetrics",
    "get_power_monitor",
]
