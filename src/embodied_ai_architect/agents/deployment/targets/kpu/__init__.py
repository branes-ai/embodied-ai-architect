"""Stillwater KPU deployment target.

This package provides the integration layer for deploying models to the
Stillwater Knowledge Processing Unit (KPU) via software simulation.

The KPU is designed for efficient inference of neural networks with:
- Configurable precision (INT8, FP16, custom posit formats)
- Tiled execution for large tensors
- Fused operator support
- Deterministic latency for real-time applications

Requirements:
    PyTorch 2.0+ with torch.compile support

Architecture:
    PyTorch Model
        ↓
    torch.compile with KPU backend
        ↓
    KPUCompiler (FX graph → KPU ops, tiling, scheduling)
        ↓
    KPUProgram (executable representation)
        ↓
    KPURuntime (simulation or hardware execution)
        ↓
    Results + Metrics
"""

from .target import StillwaterKPUTarget
from .spec import (
    # Core contracts
    KPUCompilerInterface,
    KPURuntimeInterface,
    KPUSimulatorInterface,
    # Data models
    KPUConfig,
    KPUProgram,
    KPUTensor,
    KPUOp,
    TileConfig,
    MemoryLayout,
    # Execution results
    KPUExecutionResult,
    KPUPerformanceMetrics,
)

__all__ = [
    # Target
    "StillwaterKPUTarget",
    # Contracts
    "KPUCompilerInterface",
    "KPURuntimeInterface",
    "KPUSimulatorInterface",
    # Models
    "KPUConfig",
    "KPUProgram",
    "KPUTensor",
    "KPUOp",
    "TileConfig",
    "MemoryLayout",
    # Results
    "KPUExecutionResult",
    "KPUPerformanceMetrics",
]
