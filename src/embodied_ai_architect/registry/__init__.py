"""Model registry for managing and analyzing PyTorch models.

This module provides a persistent registry for PyTorch models with:
- Static analysis (parameters, FLOPs, architecture detection)
- Safe loading (state_dict, JIT, ONNX, full model)
- Query/filter capabilities

Usage:
    from embodied_ai_architect.registry import ModelRegistry, ModelAnalyzer

    # Analyze a model
    analyzer = ModelAnalyzer()
    metadata = analyzer.analyze("model.pt", input_shape=(1, 3, 224, 224))

    # Register in registry
    registry = ModelRegistry()
    registry.register("model.pt", name="My Model", tags=["perception"])

    # Query models
    models = registry.query(architecture="cnn", max_params=10_000_000)
"""

from embodied_ai_architect.registry.model_registry import (
    ModelMetadata,
    ModelRegistry,
)
from embodied_ai_architect.registry.analyzer import ModelAnalyzer
from embodied_ai_architect.registry.exceptions import (
    RegistryError,
    ModelLoadError,
    ModelNotFoundError,
    ModelAlreadyExistsError,
    AnalysisError,
)

__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "ModelAnalyzer",
    "RegistryError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelAlreadyExistsError",
    "AnalysisError",
]
