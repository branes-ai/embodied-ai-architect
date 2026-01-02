"""Perception operator benchmarks.

Benchmarks for object detection, tracking, and preprocessing operators.
"""

import numpy as np
from .base import OperatorBenchmark


class YOLODetectorBenchmark(OperatorBenchmark):
    """Benchmark for YOLO object detector variants.

    Supports YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x.
    Can run on CPU, GPU (PyTorch/CUDA), or NPU (ONNX Runtime).
    """

    def __init__(
        self,
        variant: str = "n",
        input_size: int = 640,
        conf_threshold: float = 0.25,
        model_path: str | None = None,
    ):
        """Initialize YOLO benchmark.

        Args:
            variant: Model variant (n, s, m, l, x)
            input_size: Input resolution (default 640)
            conf_threshold: Confidence threshold
            model_path: Optional path to ONNX model for NPU
        """
        super().__init__(
            operator_id=f"yolo_detector_{variant}",
            config={
                "variant": variant,
                "input_size": input_size,
                "conf_threshold": conf_threshold,
            },
        )
        self.variant = variant
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.model = None
        self.session = None
        self.execution_target = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize model for the specified target."""
        self.execution_target = execution_target

        if execution_target == "npu":
            self._setup_npu()
        else:
            self._setup_pytorch(execution_target)

        self._is_setup = True

    def _setup_pytorch(self, target: str) -> None:
        """Setup PyTorch/Ultralytics model."""
        try:
            from ultralytics import YOLO

            model_name = f"yolov8{self.variant}.pt"
            self.model = YOLO(model_name)

            if target == "gpu":
                self.model.to("cuda")
            else:
                self.model.to("cpu")

        except ImportError:
            raise RuntimeError("ultralytics package required for YOLO benchmark")

    def _setup_npu(self) -> None:
        """Setup ONNX Runtime with NPU execution provider."""
        try:
            import onnxruntime as ort

            # Use provided model path or default
            if self.model_path:
                model_path = self.model_path
            else:
                model_path = f"yolov8{self.variant}.onnx"

            # Configure for NPU
            providers = ["RyzenAIExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            providers = [p for p in providers if p in available]

            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"  Using providers: {self.session.get_providers()}")

        except ImportError:
            raise RuntimeError("onnxruntime package required for NPU benchmark")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def create_sample_input(self) -> dict:
        """Create sample RGB image."""
        # Random image data
        image = np.random.randint(
            0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8
        )
        return {"image": image}

    def run_once(self, inputs: dict) -> dict:
        """Run inference once."""
        image = inputs["image"]

        if self.execution_target == "npu":
            return self._run_npu(image)
        else:
            return self._run_pytorch(image)

    def _run_pytorch(self, image: np.ndarray) -> dict:
        """Run inference with PyTorch model."""
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        return {"detections": results[0].boxes if results else []}

    def _run_npu(self, image: np.ndarray) -> dict:
        """Run inference with ONNX Runtime."""
        # Preprocess
        blob = image.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch dim

        # Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})

        return {"detections": outputs}

    def teardown(self) -> None:
        """Clean up resources."""
        self.model = None
        self.session = None


class ByteTrackBenchmark(OperatorBenchmark):
    """Benchmark for ByteTrack multi-object tracker.

    Runs on CPU only (tracking is association logic, not DNN inference).
    """

    def __init__(
        self,
        num_detections: int = 100,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
    ):
        """Initialize ByteTrack benchmark.

        Args:
            num_detections: Number of detections per frame for benchmark
            track_high_thresh: High confidence threshold
            track_low_thresh: Low confidence threshold
            match_thresh: IOU matching threshold
        """
        super().__init__(
            operator_id="bytetrack",
            config={
                "num_detections": num_detections,
                "track_high_thresh": track_high_thresh,
                "track_low_thresh": track_low_thresh,
                "match_thresh": match_thresh,
            },
        )
        self.num_detections = num_detections
        self.tracker = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Initialize tracker."""
        if execution_target != "cpu":
            print(f"  Warning: ByteTrack runs on CPU, ignoring target={execution_target}")

        # Simple tracker implementation for benchmarking
        # In production, would use actual ByteTrack
        self.tracker = SimpleTracker(
            track_high_thresh=self.config["track_high_thresh"],
            match_thresh=self.config["match_thresh"],
        )
        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample detections."""
        # Generate random detections: [x1, y1, x2, y2, conf, class]
        detections = np.random.rand(self.num_detections, 6).astype(np.float32)
        # Scale bbox coordinates
        detections[:, :4] *= 640
        # Ensure x2 > x1, y2 > y1
        detections[:, 2] = detections[:, 0] + np.abs(detections[:, 2] - detections[:, 0])
        detections[:, 3] = detections[:, 1] + np.abs(detections[:, 3] - detections[:, 1])
        return {"detections": detections}

    def run_once(self, inputs: dict) -> dict:
        """Run tracking once."""
        detections = inputs["detections"]
        tracks = self.tracker.update(detections)
        return {"tracks": tracks}

    def teardown(self) -> None:
        """Clean up."""
        self.tracker = None


class SimpleTracker:
    """Simplified tracker for benchmarking ByteTrack-like performance.

    Implements the core association logic without full ByteTrack complexity.
    """

    def __init__(self, track_high_thresh: float = 0.5, match_thresh: float = 0.8):
        self.track_high_thresh = track_high_thresh
        self.match_thresh = match_thresh
        self.tracks = []
        self.frame_id = 0

    def update(self, detections: np.ndarray) -> list:
        """Update tracks with new detections."""
        self.frame_id += 1

        if len(detections) == 0:
            return self.tracks

        # Split by confidence
        high_conf = detections[detections[:, 4] > self.track_high_thresh]
        low_conf = detections[detections[:, 4] <= self.track_high_thresh]

        # Compute IOU matrix (simplified)
        if len(self.tracks) > 0 and len(high_conf) > 0:
            iou_matrix = self._compute_iou_matrix(
                np.array([t["bbox"] for t in self.tracks]), high_conf[:, :4]
            )
            # Simple greedy matching
            matched = self._greedy_match(iou_matrix, self.match_thresh)
        else:
            matched = []

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx]["bbox"] = high_conf[det_idx, :4]
            self.tracks[track_idx]["conf"] = high_conf[det_idx, 4]

        # Create new tracks for unmatched detections
        matched_det = set(d for _, d in matched)
        for i, det in enumerate(high_conf):
            if i not in matched_det:
                self.tracks.append(
                    {
                        "id": len(self.tracks),
                        "bbox": det[:4],
                        "conf": det[4],
                        "class": int(det[5]),
                    }
                )

        return self.tracks

    def _compute_iou_matrix(
        self, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IOU between two sets of boxes."""
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2), dtype=np.float32)

        for i in range(n1):
            for j in range(n2):
                iou[i, j] = self._iou(boxes1[i], boxes2[j])

        return iou

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IOU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _greedy_match(
        self, iou_matrix: np.ndarray, thresh: float
    ) -> list[tuple[int, int]]:
        """Greedy matching based on IOU."""
        matched = []
        used_tracks = set()
        used_dets = set()

        # Sort by IOU descending
        indices = np.unravel_index(
            np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape
        )

        for t, d in zip(indices[0], indices[1]):
            if t in used_tracks or d in used_dets:
                continue
            if iou_matrix[t, d] < thresh:
                break
            matched.append((t, d))
            used_tracks.add(t)
            used_dets.add(d)

        return matched


class ImagePreprocessorBenchmark(OperatorBenchmark):
    """Benchmark for image preprocessing (resize, normalize).

    Can run on CPU or GPU.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (1920, 1080),
        output_size: tuple[int, int] = (640, 640),
        normalize: bool = True,
    ):
        super().__init__(
            operator_id="image_preprocessor",
            config={
                "input_size": input_size,
                "output_size": output_size,
                "normalize": normalize,
            },
        )
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = normalize
        self.execution_target = None

    def setup(self, execution_target: str = "cpu") -> None:
        """Setup preprocessing."""
        self.execution_target = execution_target

        if execution_target == "gpu":
            try:
                import torch

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                self.device = None
        else:
            self.device = None

        self._is_setup = True

    def create_sample_input(self) -> dict:
        """Create sample high-res image."""
        image = np.random.randint(
            0, 255, (self.input_size[1], self.input_size[0], 3), dtype=np.uint8
        )
        return {"image": image}

    def run_once(self, inputs: dict) -> dict:
        """Run preprocessing."""
        image = inputs["image"]

        if self.execution_target == "gpu" and self.device is not None:
            return self._run_gpu(image)
        else:
            return self._run_cpu(image)

    def _run_cpu(self, image: np.ndarray) -> dict:
        """CPU preprocessing with OpenCV."""
        import cv2

        # Resize
        resized = cv2.resize(image, self.output_size)

        # Normalize
        if self.normalize:
            processed = resized.astype(np.float32) / 255.0
        else:
            processed = resized

        return {"processed": processed}

    def _run_gpu(self, image: np.ndarray) -> dict:
        """GPU preprocessing with PyTorch."""
        import torch
        import torch.nn.functional as F

        # Convert to tensor
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()

        # Resize
        resized = F.interpolate(
            tensor, size=self.output_size, mode="bilinear", align_corners=False
        )

        # Normalize
        if self.normalize:
            processed = resized / 255.0
        else:
            processed = resized

        return {"processed": processed.cpu().numpy()}
