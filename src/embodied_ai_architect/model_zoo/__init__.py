"""Model Zoo - Unified model acquisition and discovery.

Provides a unified interface for acquiring models from multiple sources:
- Ultralytics (YOLO models)
- TorchVision (ResNet, EfficientNet, etc.)
- HuggingFace Hub (transformers, vision models)

Example usage:
    from embodied_ai_architect.model_zoo import acquire, discover

    # Download a model
    path = acquire("yolov8n", format="onnx")

    # Discover available models
    models = discover(task="detection", max_params=10_000_000)
"""

from .acquisition import ModelAcquisition, acquire, acquire_for_operator
from .cache import ModelCache
from .discovery import ModelDiscoveryService, discover

__all__ = [
    "ModelAcquisition",
    "ModelCache",
    "ModelDiscoveryService",
    "acquire",
    "acquire_for_operator",
    "discover",
]
