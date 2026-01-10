"""NVIDIA NVDLA Virtual Platform deployment target.

NVDLA (NVIDIA Deep Learning Accelerator) is an open-source accelerator IP.
This package provides integration with the NVDLA Virtual Platform for
simulation-based deployment and validation.

Architecture:
    ONNX Model
        ↓
    (ONNX → Caffe conversion)
        ↓
    NVDLA Compiler (nvdla_compiler)
        ↓
    Loadable File (.nvdla)
        ↓
    NVDLA Runtime (nvdla_runtime on VP)
        ↓
    Results + Metrics

The Virtual Platform is a SystemC-based simulation that runs a full
Linux system with NVDLA hardware models. It supports three modes:
- Pure SystemC simulation (slow but accurate)
- RTL co-simulation (most accurate)
- AWS FPGA execution (hardware)

References:
- https://nvdla.org/
- https://github.com/nvdla/vp
- https://github.com/nvdla/sw
"""

from .target import NVDLATarget
from .spec import (
    NVDLAConfig,
    NVDLALoadable,
    NVDLACompilerInterface,
    NVDLARuntimeInterface,
    NVDLAExecutionResult,
    NVDLAPerformanceMetrics,
    NVDLAPrecision,
    NVDLAVariant,
)

__all__ = [
    "NVDLATarget",
    "NVDLAConfig",
    "NVDLALoadable",
    "NVDLACompilerInterface",
    "NVDLARuntimeInterface",
    "NVDLAExecutionResult",
    "NVDLAPerformanceMetrics",
    "NVDLAPrecision",
    "NVDLAVariant",
]
