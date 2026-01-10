"""Deployment agent for edge/embedded model deployment."""

from .agent import DeploymentAgent
from .models import (
    CalibrationConfig,
    DeploymentArtifact,
    DeploymentPrecision,
    DeploymentResult,
    DeploymentStatus,
    ValidationConfig,
    ValidationResult,
)

_exports = [
    "DeploymentAgent",
    "DeploymentPrecision",
    "DeploymentStatus",
    "CalibrationConfig",
    "ValidationConfig",
    "DeploymentArtifact",
    "DeploymentResult",
    "ValidationResult",
]

# Import targets with optional dependencies
try:
    from .targets.jetson import JetsonTarget

    _exports.append("JetsonTarget")
except ImportError:
    pass

try:
    from .targets.openvino import OpenVINOTarget

    _exports.append("OpenVINOTarget")
except ImportError:
    pass

__all__ = _exports
