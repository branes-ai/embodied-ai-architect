"""Deployment targets for different hardware platforms."""

from .base import DeploymentTarget

_available_targets = ["DeploymentTarget"]

# Jetson/TensorRT target (optional: tensorrt, pycuda)
try:
    from .jetson import JetsonTarget

    _available_targets.append("JetsonTarget")
except ImportError:
    pass

# OpenVINO target (optional: openvino, nncf)
try:
    from .openvino import OpenVINOTarget

    _available_targets.append("OpenVINOTarget")
except ImportError:
    pass

__all__ = _available_targets
