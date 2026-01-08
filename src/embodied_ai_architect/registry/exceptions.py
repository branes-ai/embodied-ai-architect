"""Exceptions for the model registry."""


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class ModelLoadError(RegistryError):
    """Failed to load a model file."""

    pass


class ModelNotFoundError(RegistryError):
    """Model not found in registry."""

    pass


class ModelAlreadyExistsError(RegistryError):
    """Model with this ID already exists."""

    pass


class AnalysisError(RegistryError):
    """Failed to analyze model."""

    pass
