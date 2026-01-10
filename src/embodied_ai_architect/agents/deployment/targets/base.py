"""Base class for deployment targets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch.nn as nn

from ..models import (
    CalibrationConfig,
    DeploymentArtifact,
    DeploymentPrecision,
    ValidationConfig,
    ValidationResult,
)


class DeploymentTarget(ABC):
    """Abstract base class for deployment targets.

    Implementations handle specific deployment platforms like
    Jetson (TensorRT), Coral (Edge TPU), OpenVINO, etc.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this target is available on the current system.

        Returns:
            True if deployment tools are installed and accessible
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return information about target capabilities.

        Returns:
            Dictionary with supported precisions, features, etc.
        """
        pass

    @abstractmethod
    def deploy(
        self,
        model: nn.Module | Path,
        precision: DeploymentPrecision,
        output_path: Path,
        input_shape: tuple[int, ...],
        calibration: CalibrationConfig | None = None,
        **kwargs,
    ) -> DeploymentArtifact:
        """Deploy a model to this target.

        Args:
            model: PyTorch model or path to ONNX model
            precision: Target precision (fp32, fp16, int8)
            output_path: Path for output engine file
            input_shape: Model input shape
            calibration: Calibration config for INT8
            **kwargs: Target-specific options

        Returns:
            DeploymentArtifact with deployed model info

        Raises:
            ValueError: If precision not supported or calibration missing for INT8
            RuntimeError: If deployment fails
        """
        pass

    @abstractmethod
    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: nn.Module | Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate deployed model against baseline.

        Args:
            deployed_artifact: The deployed model artifact
            baseline_model: Original model for comparison
            config: Validation configuration

        Returns:
            ValidationResult with accuracy/performance comparison
        """
        pass

    def supports_precision(self, precision: DeploymentPrecision) -> bool:
        """Check if target supports a precision level."""
        caps = self.get_capabilities()
        return precision.value in caps.get("supported_precisions", [])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
