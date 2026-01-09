"""Pipeline requirements definition and wizard.

This module provides tools for defining perception pipeline requirements
through an interactive wizard or YAML configuration files.

Example usage:
    # Interactive wizard
    from embodied_ai_architect.requirements import run_wizard, save_requirements
    requirements = run_wizard()
    save_requirements(requirements, "requirements.yaml")

    # Load from file
    from embodied_ai_architect.requirements import load_requirements
    requirements = load_requirements("requirements.yaml")

    # From use case
    from embodied_ai_architect.requirements import from_usecase
    requirements = from_usecase("drone_obstacle_avoidance")
"""

from .loader import load_requirements, requirements_to_yaml, save_requirements
from .models import (
    DeploymentRequirements,
    ExecutionTarget,
    HardwareRequirements,
    PerceptionRequirements,
    PipelineRequirements,
    TaskType,
)
from .wizard import RequirementsWizard, from_usecase, run_wizard

__all__ = [
    # Models
    "PipelineRequirements",
    "PerceptionRequirements",
    "HardwareRequirements",
    "DeploymentRequirements",
    "TaskType",
    "ExecutionTarget",
    # Wizard
    "RequirementsWizard",
    "run_wizard",
    "from_usecase",
    # Loader
    "load_requirements",
    "save_requirements",
    "requirements_to_yaml",
]
