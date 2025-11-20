"""Sensor abstractions for different camera types."""

from .base import BaseSensor
from .monocular import MonocularCamera

__all__ = ['BaseSensor', 'MonocularCamera']
