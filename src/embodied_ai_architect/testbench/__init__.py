"""Testbench for model validation and drift monitoring.

Provides tools for validating model accuracy against ground truth
datasets and monitoring performance drift over time.

Example usage:
    # Validate a model
    from embodied_ai_architect.testbench import ModelValidator, MetricType

    validator = ModelValidator(
        model_id="yolov8n",
        task="detection",
        thresholds={MetricType.MAP50: 0.5},
    )
    result = validator.validate(model_fn, dataset)
    print(result.summary())

    # Record for drift monitoring
    from embodied_ai_architect.testbench import record_validation, check_drift

    record_validation(result)
    drift_report = check_drift("yolov8n")
    if drift_report:
        print(drift_report.summary())
"""

from .drift import (
    DriftMonitor,
    DriftReport,
    DriftStatus,
    check_drift,
    get_monitor,
    record_validation,
)
from .metrics import (
    MetricResult,
    MetricType,
    ValidationResult,
    compute_accuracy,
    compute_dice,
    compute_iou,
    compute_map,
    compute_miou,
    compute_top_k_accuracy,
)
from .validation import ModelValidator, validate_model

__all__ = [
    # Metrics
    "MetricType",
    "MetricResult",
    "ValidationResult",
    "compute_iou",
    "compute_map",
    "compute_accuracy",
    "compute_top_k_accuracy",
    "compute_miou",
    "compute_dice",
    # Validation
    "ModelValidator",
    "validate_model",
    # Drift
    "DriftMonitor",
    "DriftReport",
    "DriftStatus",
    "get_monitor",
    "record_validation",
    "check_drift",
]
