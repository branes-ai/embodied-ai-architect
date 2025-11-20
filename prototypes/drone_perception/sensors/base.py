"""Base sensor interface."""

from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Add parent directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import Frame


class BaseSensor(ABC):
    """Abstract base class for all sensors."""

    @abstractmethod
    def get_frame(self) -> Frame:
        """
        Capture and return a single frame.

        Returns:
            Frame with image, optional depth, and metadata
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if sensor is ready to capture frames."""
        pass

    @abstractmethod
    def release(self):
        """Release sensor resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
