"""Operator benchmarking framework for embodied AI systems.

This module provides infrastructure for profiling operators on different
hardware platforms and collecting performance metrics for the operator catalog.
"""

from .runner import OperatorBenchmarkRunner, BenchmarkConfig
from .base import OperatorBenchmark, OperatorBenchmarkResult

__all__ = [
    "OperatorBenchmarkRunner",
    "BenchmarkConfig",
    "OperatorBenchmark",
    "OperatorBenchmarkResult",
]
