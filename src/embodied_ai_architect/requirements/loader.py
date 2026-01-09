"""YAML loader and saver for pipeline requirements.

Handles serialization and deserialization of PipelineRequirements
to/from YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .models import PipelineRequirements


def load_requirements(path: Union[str, Path]) -> PipelineRequirements:
    """Load pipeline requirements from a YAML file.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed PipelineRequirements

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If YAML doesn't match schema
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    return PipelineRequirements.model_validate(data)


def save_requirements(
    requirements: PipelineRequirements,
    path: Union[str, Path],
    include_defaults: bool = False,
) -> Path:
    """Save pipeline requirements to a YAML file.

    Args:
        requirements: Requirements to save
        path: Output path
        include_defaults: Whether to include default values

    Returns:
        Path to saved file
    """
    path = Path(path)

    # Convert to dict, excluding None values and defaults if requested
    data = requirements.model_dump(
        exclude_none=True,
        exclude_defaults=not include_defaults,
        mode="json",
    )

    # Always include name
    data["name"] = requirements.name

    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return path


def requirements_to_yaml(
    requirements: PipelineRequirements,
    include_defaults: bool = False,
) -> str:
    """Convert requirements to YAML string.

    Args:
        requirements: Requirements to convert
        include_defaults: Whether to include default values

    Returns:
        YAML string
    """
    data = requirements.model_dump(
        exclude_none=True,
        exclude_defaults=not include_defaults,
        mode="json",
    )
    data["name"] = requirements.name

    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
