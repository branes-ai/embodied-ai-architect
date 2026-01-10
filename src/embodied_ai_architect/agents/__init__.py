"""Agent implementations for the Embodied AI Architect system."""

from .base import BaseAgent
from .model_analyzer import ModelAnalyzerAgent
from .benchmark import BenchmarkAgent
from .hardware_profile import HardwareProfileAgent
from .report_synthesis import ReportSynthesisAgent

_exports = [
    "BaseAgent",
    "ModelAnalyzerAgent",
    "BenchmarkAgent",
    "HardwareProfileAgent",
    "ReportSynthesisAgent",
]

# Deployment agent (optional: tensorrt)
try:
    from .deployment import DeploymentAgent

    _exports.append("DeploymentAgent")
except ImportError:
    pass

__all__ = _exports
