# Session: Operator Benchmarking Infrastructure with Graphs Auto-Detect Integration

**Date**: 2026-01-02
**Focus**: Phase 2 of Ryzen AI NUC Demo - Operator Profiling Infrastructure

## Summary

Completed the operator benchmarking framework for profiling embodied AI operators on heterogeneous hardware (CPU, GPU, NPU). Integrated with the existing `graphs` repository's comprehensive hardware auto-detection system for consistent SHA fingerprinting of hw/sw configurations.

## Key Accomplishments

### 1. Benchmark Framework (`benchmarks/operators/`)

Created a systematic operator benchmarking infrastructure:

- **`base.py`**: `OperatorBenchmark` base class with standardized timing methodology (warmup, iterations, percentiles), `OperatorBenchmarkResult` dataclass matching embodied-schemas `OperatorPerfProfile`

- **`runner.py`**: `BenchmarkConfig` with hardware fingerprints, `OperatorBenchmarkRunner` for batch execution across targets, YAML/JSON output with fingerprints

- **`perception.py`**: Perception operator benchmarks
  - `YOLODetectorBenchmark` - PyTorch and ONNX Runtime (NPU) support
  - `ByteTrackBenchmark` - Multi-object tracker
  - `ImagePreprocessorBenchmark` - CPU/GPU preprocessing

- **`state_estimation.py`**: State estimation operator benchmarks
  - `KalmanFilter2DBboxBenchmark` - 2D bbox tracking filters
  - `SceneGraphManagerBenchmark` - 3D scene graph management
  - `EKF6DOFBenchmark` - 6-DOF pose estimation

- **`control.py`**: Control operator benchmarks
  - `PIDControllerBenchmark` - Multi-axis PID control
  - `PathPlannerAStarBenchmark` - A* path planning on occupancy grids
  - `TrajectoryFollowerBenchmark` - Pure pursuit trajectory following

- **`catalog_updater.py`**: Updates embodied-schemas operator YAML files with benchmark results

- **`run_operator_benchmarks.py`**: Main CLI script with auto-detection and catalog updates

### 2. Hardware Auto-Detection Integration

Updated `hardware_detect.py` to integrate with `graphs.hardware.calibration.auto_detect`:

```python
# When graphs is available, provides:
{
    "hardware_fingerprint": "c3f840a080356806",  # SHA256 of hw identity
    "software_fingerprint": "6286d41799837f08",  # SHA256 of sw stack
    "cpu_model": "12th Gen Intel(R) Core(TM) i7-12700K",
    "cpu_stepping": 2,
    "cpu_microcode": "0x3a",
    "calibration_context": CalibrationContext,  # Full context for advanced use
    ...
}
```

The SHA fingerprints enable:
- **Reproducibility**: Same hardware + same software = same fingerprints
- **Tracking**: Can track performance changes across software updates
- **Consistency**: Uses the same identification system as graphs calibration

Falls back to simplified detection when graphs is not installed.

### 3. NPU Detection

Extended detection beyond graphs to include NPU availability:
- ONNX Runtime providers (RyzenAI, QNN, CoreML)
- AMD XDNA driver detection (`/dev/accel/accel0`)

## Example Output

```
$ python benchmarks/run_operator_benchmarks.py --detect

Detected Hardware:
  HW Fingerprint: c3f840a080356806
  SW Fingerprint: 6286d41799837f08
  CPU: 12th Gen Intel(R) Core(TM) i7-12700K
  Vendor: intel
  Stepping: 2, Microcode: 0x3a
  Cores: 12P / 20L
  NPU: Available (XDNA)
  Memory: 31.1 GB
  Execution targets: cpu, npu
  Catalog match: generic_intel_cpu
```

## Files Created/Modified

### Created
- `benchmarks/operators/__init__.py` - Package exports
- `benchmarks/operators/base.py` - Base classes and result types
- `benchmarks/operators/runner.py` - Benchmark runner and config
- `benchmarks/operators/perception.py` - Perception operator benchmarks
- `benchmarks/operators/state_estimation.py` - State estimation benchmarks
- `benchmarks/operators/control.py` - Control operator benchmarks
- `benchmarks/operators/catalog_updater.py` - Catalog YAML updater
- `benchmarks/operators/hardware_detect.py` - Hardware auto-detection with graphs integration
- `benchmarks/run_operator_benchmarks.py` - Main CLI script

### Integration Points
- Uses `graphs.hardware.calibration.auto_detect.CalibrationContext` when available
- Updates `embodied-schemas` operator YAML files with `--update-catalog` flag
- Fingerprints stored in benchmark results for reproducibility

## User Feedback Addressed

**User**: "why is there a --hardware flag? Why is this not auto_detect?"

**Response**: Removed manual hardware specification in favor of auto-detection. The `--hardware` flag is now optional override only.

**User**: "we had an extensive auto_detect in ../graphs that yields a SHA has to uniquely identify and track complex hw/sw compositions. What you just did takes advantage of that and thus is consistent, correct?"

**Response**: Initially created a duplicate - fixed by integrating with `graphs.hardware.calibration.auto_detect.CalibrationContext.detect()`. Now properly uses:
- `context.hardware.fingerprint` for hardware identity SHA
- `context.software.fingerprint` for software stack SHA
- Full `CalibrationContext` for advanced access

## Architecture Decisions

1. **Fingerprint Integration**: Wraps graphs auto_detect rather than duplicating
2. **Fallback Support**: Works without graphs installed (shows note about missing fingerprints)
3. **NPU Extension**: Extends graphs detection with NPU-specific checks
4. **Result Format**: Fingerprints included in saved YAML/JSON results

## Next Steps

1. Continue with Phase 3: Reference Architecture creation
2. Run actual benchmarks on AMD Ryzen AI NUC hardware
3. Add more operator benchmarks (trajectory predictor, collision detector)
4. Consider adding NPU detection to graphs auto_detect upstream
