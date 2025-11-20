"""Sensor abstractions for different camera types."""

from .base import BaseSensor
from .monocular import MonocularCamera
from .stereo import StereoCamera, StereoRecordedCamera

__all__ = ['BaseSensor', 'MonocularCamera', 'StereoCamera', 'StereoRecordedCamera']
