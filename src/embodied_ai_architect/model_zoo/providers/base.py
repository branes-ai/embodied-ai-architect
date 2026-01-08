"""Base classes for model providers.

Defines the abstract interface that all model providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ModelFormat(str, Enum):
    """Supported model formats."""

    PYTORCH = "pytorch"  # .pt file (state_dict or full model)
    TORCHSCRIPT = "torchscript"  # .pt file (JIT traced/scripted)
    ONNX = "onnx"  # .onnx file
    TENSORRT = "tensorrt"  # .engine file
    OPENVINO = "openvino"  # OpenVINO IR format
    COREML = "coreml"  # .mlmodel or .mlpackage


@dataclass
class ModelQuery:
    """Query parameters for model discovery."""

    # Task filtering
    task: Optional[str] = None  # "detection", "classification", "segmentation", etc.
    subtask: Optional[str] = None  # "object_detection", "pose_estimation", etc.

    # Size constraints
    min_params: Optional[int] = None
    max_params: Optional[int] = None
    min_flops: Optional[int] = None
    max_flops: Optional[int] = None

    # Quality filtering
    min_accuracy: Optional[float] = None  # mAP, top-1, etc.
    benchmarked: bool = False  # Only return models with benchmark data

    # Architecture filtering
    architecture: Optional[str] = None  # "resnet", "yolo", "vit", etc.
    family: Optional[str] = None  # "cnn", "transformer", etc.

    # Provider filtering
    provider: Optional[str] = None  # Limit to specific provider

    # Search
    query: Optional[str] = None  # Free-text search

    def matches(self, model_info: dict[str, Any]) -> bool:
        """Check if a model matches this query."""
        if self.task and model_info.get("task") != self.task:
            return False

        if self.subtask and model_info.get("subtask") != self.subtask:
            return False

        params = model_info.get("parameters")
        if params:
            if self.min_params and params < self.min_params:
                return False
            if self.max_params and params > self.max_params:
                return False

        flops = model_info.get("flops")
        if flops:
            if self.min_flops and flops < self.min_flops:
                return False
            if self.max_flops and flops > self.max_flops:
                return False

        if self.min_accuracy:
            accuracy = model_info.get("accuracy", model_info.get("map50", model_info.get("top1")))
            if accuracy and accuracy < self.min_accuracy:
                return False

        if self.benchmarked and not model_info.get("benchmarked"):
            return False

        if self.architecture and model_info.get("architecture") != self.architecture:
            return False

        if self.family and model_info.get("family") != self.family:
            return False

        if self.query:
            query_lower = self.query.lower()
            name = model_info.get("name", "").lower()
            desc = model_info.get("description", "").lower()
            if query_lower not in name and query_lower not in desc:
                return False

        return True


@dataclass
class ModelArtifact:
    """Result of downloading a model."""

    # Identity
    model_id: str
    provider: str
    format: ModelFormat

    # File info
    path: Path
    size_bytes: int

    # Metadata
    name: str
    version: Optional[str] = None
    task: Optional[str] = None
    parameters: Optional[int] = None
    input_shape: Optional[tuple[int, ...]] = None

    # Benchmark data (if available)
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None

    # Extra metadata from provider
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """Abstract base class for model providers.

    Implementations handle specific model sources like Ultralytics,
    TorchVision, or HuggingFace Hub.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'ultralytics', 'torchvision')."""
        ...

    @property
    @abstractmethod
    def supported_formats(self) -> list[ModelFormat]:
        """List of formats this provider can export to."""
        ...

    @abstractmethod
    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available models, optionally filtered.

        Args:
            query: Optional filter criteria

        Returns:
            List of model info dictionaries with at least:
                - id: Unique model identifier
                - name: Human-readable name
                - task: Primary task (detection, classification, etc.)
                - parameters: Parameter count (if known)
        """
        ...

    @abstractmethod
    def download(
        self,
        model_id: str,
        format: ModelFormat,
        cache_dir: Path,
    ) -> ModelArtifact:
        """Download a model in the specified format.

        Args:
            model_id: Model identifier (from list_models)
            format: Target format for export
            cache_dir: Directory to store downloaded model

        Returns:
            ModelArtifact with path and metadata

        Raises:
            ValueError: If model_id is not found
            ValueError: If format is not supported
            RuntimeError: If download fails
        """
        ...

    @abstractmethod
    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed information about a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model metadata

        Raises:
            ValueError: If model_id is not found
        """
        ...

    def supports_format(self, format: ModelFormat) -> bool:
        """Check if this provider supports a format."""
        return format in self.supported_formats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
