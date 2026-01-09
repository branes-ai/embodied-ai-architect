"""Accuracy metrics for model validation.

Provides standard metrics for evaluating perception models:
- Detection: mAP, IoU, precision, recall
- Classification: accuracy, top-k, confusion matrix
- Segmentation: mIoU, pixel accuracy, dice score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class MetricType(str, Enum):
    """Supported metric types."""

    # Detection metrics
    MAP = "mAP"
    MAP50 = "mAP@50"
    MAP75 = "mAP@75"
    IOU = "IoU"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

    # Classification metrics
    ACCURACY = "accuracy"
    TOP1 = "top1"
    TOP5 = "top5"

    # Segmentation metrics
    MIOU = "mIoU"
    PIXEL_ACCURACY = "pixel_accuracy"
    DICE = "dice"

    # Latency metrics
    LATENCY_MS = "latency_ms"
    THROUGHPUT_FPS = "throughput_fps"


@dataclass
class MetricResult:
    """Result of a metric computation."""

    metric: MetricType
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.threshold is not None:
            self.passed = self.value >= self.threshold


@dataclass
class ValidationResult:
    """Complete validation result for a model."""

    model_id: str
    dataset: str
    metrics: list[MetricResult] = field(default_factory=list)
    passed: bool = True
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Overall pass if all metrics with thresholds pass
        for metric in self.metrics:
            if metric.passed is False:
                self.passed = False
                break

    def get_metric(self, metric_type: MetricType) -> Optional[MetricResult]:
        """Get a specific metric result."""
        for m in self.metrics:
            if m.metric == metric_type:
                return m
        return None

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [f"Validation: {status}", f"Model: {self.model_id}", f"Dataset: {self.dataset}"]

        for m in self.metrics:
            threshold_str = f" (≥{m.threshold:.1%})" if m.threshold else ""
            status_icon = ""
            if m.passed is True:
                status_icon = " ✓"
            elif m.passed is False:
                status_icon = " ✗"

            if m.metric in (MetricType.LATENCY_MS,):
                lines.append(f"  {m.metric.value}: {m.value:.1f}ms{status_icon}")
            elif m.metric in (MetricType.THROUGHPUT_FPS,):
                lines.append(f"  {m.metric.value}: {m.value:.1f} FPS{status_icon}")
            else:
                lines.append(f"  {m.metric.value}: {m.value:.1%}{threshold_str}{status_icon}")

        return "\n".join(lines)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_map(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Compute mean Average Precision for detection.

    Args:
        predictions: List of {boxes, scores, labels}
        ground_truth: List of {boxes, labels}
        iou_threshold: IoU threshold for matching

    Returns:
        mAP score
    """
    if not predictions or not ground_truth:
        return 0.0

    # Collect all predictions and ground truths
    all_preds = []
    all_gt = []

    for pred, gt in zip(predictions, ground_truth):
        pred_boxes = pred.get("boxes", [])
        pred_scores = pred.get("scores", [])
        pred_labels = pred.get("labels", [])

        gt_boxes = gt.get("boxes", [])
        gt_labels = gt.get("labels", [])

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            all_preds.append({"box": box, "score": score, "label": label, "matched": False})

        for box, label in zip(gt_boxes, gt_labels):
            all_gt.append({"box": box, "label": label, "matched": False})

    if not all_gt:
        return 0.0

    # Sort predictions by score
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    # Compute precision-recall curve
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for pred in all_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(all_gt):
            if gt["matched"] or gt["label"] != pred["label"]:
                continue

            iou = compute_iou(np.array(pred["box"]), np.array(gt["box"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            all_gt[best_gt_idx]["matched"] = True
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(all_gt)

        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = 0.0
        for r, pr in zip(recalls, precisions):
            if r >= t:
                p = max(p, pr)
        ap += p / 11

    return ap


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted class indices
        labels: Ground truth class indices

    Returns:
        Accuracy score
    """
    if len(predictions) == 0:
        return 0.0
    return np.mean(predictions == labels)


def compute_top_k_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> float:
    """Compute top-k classification accuracy.

    Args:
        logits: Model output logits [N, num_classes]
        labels: Ground truth class indices [N]
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy score
    """
    if len(logits) == 0:
        return 0.0

    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels[:, np.newaxis], axis=1)
    return np.mean(correct)


def compute_miou(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mean Intersection over Union for segmentation.

    Args:
        predictions: Predicted segmentation masks [N, H, W]
        labels: Ground truth masks [N, H, W]
        num_classes: Number of classes
        ignore_index: Label index to ignore

    Returns:
        mIoU score
    """
    ious = []

    for cls in range(num_classes):
        pred_mask = predictions == cls
        gt_mask = labels == cls

        # Ignore specified index
        valid_mask = labels != ignore_index
        pred_mask = pred_mask & valid_mask
        gt_mask = gt_mask & valid_mask

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)

        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def compute_dice(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute Dice coefficient for segmentation.

    Args:
        predictions: Predicted binary masks
        labels: Ground truth binary masks

    Returns:
        Dice score
    """
    intersection = np.sum(predictions & labels)
    total = np.sum(predictions) + np.sum(labels)

    if total == 0:
        return 1.0  # Both empty

    return 2 * intersection / total
