"""Image preprocessing operator.

Handles resizing, normalization, and format conversion for vision pipelines.
"""

from typing import Any

import numpy as np

from ..base import Operator


class ImagePreprocessor(Operator):
    """Image preprocessing for vision pipelines.

    Operations:
    - Resize to target size
    - Normalize pixel values
    - Convert color space
    - Pad to maintain aspect ratio
    """

    def __init__(self):
        super().__init__(operator_id="image_preprocessor")
        self.target_size = (640, 640)
        self.normalize = True
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]
        self.letterbox = False
        self._use_gpu = False

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize preprocessor.

        Args:
            config: Configuration with optional keys:
                - output_size: Target (width, height) tuple
                - normalize: Whether to normalize to [0, 1]
                - mean: Per-channel mean for normalization
                - std: Per-channel std for normalization
                - letterbox: Pad to maintain aspect ratio
            execution_target: cpu or gpu
        """
        self._execution_target = execution_target
        self._config = config

        self.target_size = tuple(config.get("output_size", (640, 640)))
        self.normalize = config.get("normalize", True)
        self.mean = config.get("mean", [0.0, 0.0, 0.0])
        self.std = config.get("std", [1.0, 1.0, 1.0])
        self.letterbox = config.get("letterbox", False)

        # GPU preprocessing with OpenCV CUDA if available
        self._use_gpu = execution_target == "gpu"
        if self._use_gpu:
            try:
                import cv2

                if not cv2.cuda.getCudaEnabledDeviceCount():
                    print("[ImagePreprocessor] CUDA not available, falling back to CPU")
                    self._use_gpu = False
            except Exception:
                self._use_gpu = False

        self._is_setup = True
        print(f"[ImagePreprocessor] Ready on {execution_target} (output: {self.target_size})")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Preprocess input image.

        Args:
            inputs: Dictionary with 'image' key containing RGB uint8 array (H, W, 3)

        Returns:
            Dictionary with 'processed' key containing preprocessed image
        """
        image = inputs["image"]

        if self._use_gpu:
            processed = self._process_gpu(image)
        else:
            processed = self._process_cpu(image)

        return {"processed": processed}

    def _process_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU preprocessing using OpenCV."""
        import cv2

        if self.letterbox:
            processed = self._letterbox_resize(image)
        else:
            processed = cv2.resize(image, self.target_size)

        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
            processed = (processed - self.mean) / self.std

        return processed

    def _process_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU preprocessing using OpenCV CUDA."""
        import cv2

        # Upload to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_image, self.target_size)

        # Download result
        processed = gpu_resized.download()

        # Normalize on CPU (CUDA normalize is complex)
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
            processed = (processed - self.mean) / self.std

        return processed

    def _letterbox_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize with letterboxing to maintain aspect ratio."""
        import cv2

        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image
        if self.normalize:
            padded = np.full((target_h, target_w, 3), 114 / 255.0, dtype=np.float32)
        else:
            padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # Center the image
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        padded[top : top + new_h, left : left + new_w] = resized

        return padded
