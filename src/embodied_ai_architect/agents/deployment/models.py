"""Pydantic models for deployment operations."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DeploymentPrecision(str, Enum):
    """Supported quantization precisions."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class DeploymentStatus(str, Enum):
    """Status of a deployment operation."""

    PENDING = "pending"
    EXPORTING = "exporting"
    QUANTIZING = "quantizing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class CalibrationConfig(BaseModel):
    """Configuration for INT8 calibration."""

    data_path: Path
    num_samples: int = 100
    batch_size: int = 1
    input_shape: tuple[int, ...] | None = None
    preprocessing: str = "imagenet"  # "imagenet", "coco", "yolo", "none"


class ValidationConfig(BaseModel):
    """Configuration for deployment validation."""

    test_data_path: Path | None = None
    num_samples: int = 100
    tolerance_percent: float = 1.0  # Max accuracy drop allowed
    compare_outputs: bool = True
    latency_check: bool = True


class DeploymentArtifact(BaseModel):
    """Result of a successful deployment."""

    engine_path: Path
    precision: DeploymentPrecision
    target: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None = None
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of deployment validation."""

    passed: bool
    baseline_accuracy: float | None = None
    deployed_accuracy: float | None = None
    accuracy_delta: float | None = None
    baseline_latency_ms: float | None = None
    deployed_latency_ms: float | None = None
    speedup: float | None = None
    max_output_diff: float | None = None
    samples_compared: int = 0
    errors: list[str] = Field(default_factory=list)


class DeploymentResult(BaseModel):
    """Complete result of deployment operation."""

    success: bool
    status: DeploymentStatus
    artifact: DeploymentArtifact | None = None
    validation: ValidationResult | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
