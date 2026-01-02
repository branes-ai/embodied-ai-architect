"""Operator benchmarking framework for embodied AI systems.

This module provides infrastructure for profiling operators on different
hardware platforms and collecting performance metrics for the operator catalog.

Features:
- Automatic hardware detection (CPU, GPU, NPU)
- Standardized timing methodology with warmup and percentiles
- Integration with embodied-schemas operator catalog
- Support for CPU, GPU (CUDA/ROCm), and NPU (ONNX Runtime) targets
"""

from .runner import OperatorBenchmarkRunner, BenchmarkConfig
from .base import OperatorBenchmark, OperatorBenchmarkResult
from .hardware_detect import detect_hardware, print_hardware_info

__all__ = [
    "OperatorBenchmarkRunner",
    "BenchmarkConfig",
    "OperatorBenchmark",
    "OperatorBenchmarkResult",
    "detect_hardware",
    "print_hardware_info",
]
