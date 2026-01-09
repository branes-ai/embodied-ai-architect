"""Pydantic models for pipeline requirements.

Defines structured requirements for perception pipelines including
task specifications, accuracy constraints, and hardware requirements.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported perception task types."""

    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    FACE_DETECTION = "face_detection"
    HAND_TRACKING = "hand_tracking"
    TRACKING = "tracking"


class ExecutionTarget(str, Enum):
    """Hardware execution targets."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    EDGE = "edge"


class PerceptionRequirements(BaseModel):
    """Requirements for perception tasks."""

    tasks: list[TaskType] = Field(
        default_factory=list,
        description="Required perception tasks",
    )
    target_classes: list[str] = Field(
        default_factory=list,
        description="Target object classes to detect (e.g., person, car)",
    )
    min_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy (mAP, top-1, etc.) as decimal",
    )
    max_latency_ms: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum inference latency in milliseconds",
    )
    min_fps: Optional[float] = Field(
        default=None,
        gt=0,
        description="Minimum frames per second",
    )
    input_resolution: Optional[tuple[int, int]] = Field(
        default=None,
        description="Input resolution (width, height)",
    )


class HardwareRequirements(BaseModel):
    """Hardware constraints for deployment."""

    execution_target: ExecutionTarget = Field(
        default=ExecutionTarget.CPU,
        description="Target hardware for inference",
    )
    max_power_watts: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum power budget in watts",
    )
    max_memory_mb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum memory usage in MB",
    )
    max_params_millions: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum model parameters in millions",
    )


class DeploymentRequirements(BaseModel):
    """Deployment environment requirements."""

    runtime: Optional[str] = Field(
        default=None,
        description="Target runtime (onnxruntime, tensorrt, openvino)",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization mode (fp32, fp16, int8)",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Inference batch size",
    )


class PipelineRequirements(BaseModel):
    """Complete pipeline requirements specification.

    This is the top-level model that captures all requirements
    for synthesizing a perception pipeline.
    """

    name: str = Field(
        description="Pipeline name/identifier",
    )
    description: Optional[str] = Field(
        default=None,
        description="Pipeline description",
    )
    perception: PerceptionRequirements = Field(
        default_factory=PerceptionRequirements,
        description="Perception task requirements",
    )
    hardware: HardwareRequirements = Field(
        default_factory=HardwareRequirements,
        description="Hardware constraints",
    )
    deployment: DeploymentRequirements = Field(
        default_factory=DeploymentRequirements,
        description="Deployment configuration",
    )
    use_case: Optional[str] = Field(
        default=None,
        description="Reference use case ID from embodied-schemas",
    )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Pipeline: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")

        if self.perception.tasks:
            tasks = ", ".join(t.value for t in self.perception.tasks)
            lines.append(f"  Tasks: {tasks}")

        if self.perception.target_classes:
            classes = ", ".join(self.perception.target_classes)
            lines.append(f"  Classes: {classes}")

        constraints = []
        if self.perception.min_accuracy:
            constraints.append(f"accuracy≥{self.perception.min_accuracy*100:.0f}%")
        if self.perception.max_latency_ms:
            constraints.append(f"latency≤{self.perception.max_latency_ms}ms")
        if self.hardware.max_power_watts:
            constraints.append(f"power≤{self.hardware.max_power_watts}W")
        if constraints:
            lines.append(f"  Constraints: {', '.join(constraints)}")

        lines.append(f"  Target: {self.hardware.execution_target.value}")

        return "\n".join(lines)
