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


class PowerConfig(BaseModel):
    """Configuration for power measurement and validation."""

    enabled: bool = True
    measurement_duration_sec: float = 5.0  # Duration to measure power
    warmup_iterations: int = 10  # Warmup before measurement
    measurement_iterations: int = 50  # Iterations during measurement
    power_budget_watts: float | None = None  # Target power budget
    tolerance_percent: float = 20.0  # Allowed deviation from prediction


class ValidationConfig(BaseModel):
    """Configuration for deployment validation."""

    test_data_path: Path | None = None
    num_samples: int = 100
    tolerance_percent: float = 1.0  # Max accuracy drop allowed
    compare_outputs: bool = True
    latency_check: bool = True
    power_validation: PowerConfig | None = None  # Power measurement config


class DeploymentArtifact(BaseModel):
    """Result of a successful deployment."""

    engine_path: Path
    precision: DeploymentPrecision
    target: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None = None
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class PowerMetrics(BaseModel):
    """Power measurement results."""

    measured_watts: float  # Actual measured power
    predicted_watts: float | None = None  # Model-predicted power
    deviation_percent: float | None = None  # (measured - predicted) / predicted * 100
    within_budget: bool | None = None  # True if under power_budget_watts
    energy_per_inference_mj: float | None = None  # millijoules per inference
    inferences_per_joule: float | None = None  # Efficiency metric
    measurement_method: str = "unknown"  # e.g., "tegrastats", "rapl", "nvidia-smi"
    gpu_power_watts: float | None = None  # GPU-specific power if available
    cpu_power_watts: float | None = None  # CPU-specific power if available
    total_power_watts: float | None = None  # System total if available


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
    # Power metrics
    power: PowerMetrics | None = None
    power_validation_passed: bool | None = None
    errors: list[str] = Field(default_factory=list)


class DeploymentResult(BaseModel):
    """Complete result of deployment operation."""

    success: bool
    status: DeploymentStatus
    artifact: DeploymentArtifact | None = None
    validation: ValidationResult | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
