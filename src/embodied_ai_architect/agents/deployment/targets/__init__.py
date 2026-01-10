"""Deployment targets for different hardware platforms."""

from .base import DeploymentTarget

_available_targets = ["DeploymentTarget"]

# Jetson/TensorRT target (optional: tensorrt, pycuda)
try:
    from .jetson import JetsonTarget

    _available_targets.append("JetsonTarget")
except ImportError:
    pass

__all__ = _available_targets
