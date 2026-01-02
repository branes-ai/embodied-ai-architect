#!/usr/bin/env python3
"""Run operator benchmarks for AMD Ryzen AI NUCs.

Usage:
    python run_operator_benchmarks.py --hardware amd_ryzen_7_8845hs_nuc --targets cpu gpu npu
    python run_operator_benchmarks.py --hardware amd_ryzen_9_8945hs_nuc --tdp 35W
    python run_operator_benchmarks.py --list  # List available benchmarks
"""

import argparse
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from operators import (
    BenchmarkConfig,
    OperatorBenchmarkRunner,
)
from operators.perception import (
    YOLODetectorBenchmark,
    ByteTrackBenchmark,
    ImagePreprocessorBenchmark,
)
from operators.state_estimation import (
    KalmanFilter2DBboxBenchmark,
    SceneGraphManagerBenchmark,
    EKF6DOFBenchmark,
)
from operators.control import (
    PIDControllerBenchmark,
    PathPlannerAStarBenchmark,
    TrajectoryFollowerBenchmark,
)
from operators.catalog_updater import CatalogUpdater


# Available benchmarks
BENCHMARKS = {
    # Perception
    "yolo_detector_n": lambda: YOLODetectorBenchmark(variant="n"),
    "yolo_detector_s": lambda: YOLODetectorBenchmark(variant="s"),
    "yolo_detector_m": lambda: YOLODetectorBenchmark(variant="m"),
    "bytetrack": lambda: ByteTrackBenchmark(num_detections=100),
    "image_preprocessor": lambda: ImagePreprocessorBenchmark(),
    # State estimation
    "kalman_filter_2d_bbox": lambda: KalmanFilter2DBboxBenchmark(num_filters=50),
    "scene_graph_manager": lambda: SceneGraphManagerBenchmark(num_objects=50),
    "ekf_6dof": lambda: EKF6DOFBenchmark(),
    # Control
    "pid_controller": lambda: PIDControllerBenchmark(num_controllers=6),
    "path_planner_astar": lambda: PathPlannerAStarBenchmark(grid_size=(100, 100)),
    "trajectory_follower": lambda: TrajectoryFollowerBenchmark(),
}

# Default operators for PGN&C pipeline
PGNC_OPERATORS = [
    "yolo_detector_n",
    "bytetrack",
    "scene_graph_manager",
    "ekf_6dof",
    "path_planner_astar",
    "pid_controller",
]


def main():
    parser = argparse.ArgumentParser(description="Run operator benchmarks")

    parser.add_argument(
        "--hardware",
        type=str,
        default="amd_ryzen_7_8845hs_nuc",
        help="Hardware ID (default: amd_ryzen_7_8845hs_nuc)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["cpu"],
        help="Execution targets: cpu, gpu, npu (default: cpu)",
    )
    parser.add_argument(
        "--operators",
        type=str,
        nargs="+",
        default=None,
        help="Operators to benchmark (default: PGN&C pipeline operators)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--tdp",
        type=str,
        default=None,
        help="TDP mode for documentation (e.g., 28W, 45W)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--update-catalog",
        type=Path,
        default=None,
        help="Path to embodied-schemas operators directory to update",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--pgnc",
        action="store_true",
        help="Run PGN&C pipeline operators only",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available benchmarks:")
        for name in sorted(BENCHMARKS.keys()):
            print(f"  - {name}")
        print("\nPGN&C pipeline operators:")
        for name in PGNC_OPERATORS:
            print(f"  - {name}")
        return 0

    # Determine operators to run
    if args.pgnc:
        operators = PGNC_OPERATORS
    elif args.operators:
        operators = args.operators
    else:
        operators = PGNC_OPERATORS

    # Validate operators
    for op in operators:
        if op not in BENCHMARKS:
            print(f"Error: Unknown operator '{op}'")
            print(f"Available: {', '.join(sorted(BENCHMARKS.keys()))}")
            return 1

    # Create config
    config = BenchmarkConfig(
        hardware_id=args.hardware,
        execution_targets=args.targets,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        tdp_mode=args.tdp,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("OPERATOR BENCHMARK SUITE")
    print("=" * 60)
    print(f"Hardware: {config.hardware_id}")
    print(f"Targets: {', '.join(config.execution_targets)}")
    print(f"TDP Mode: {config.tdp_mode or 'Not specified'}")
    print(f"Operators: {len(operators)}")
    print(f"Iterations: {config.iterations}")
    print("=" * 60)

    # Create runner
    runner = OperatorBenchmarkRunner(config)

    # Register benchmarks
    for op_name in operators:
        benchmark = BENCHMARKS[op_name]()
        runner.register_benchmark(benchmark)
        print(f"Registered: {op_name}")

    print()

    # Run benchmarks
    results = runner.run_all()

    # Print summary
    runner.print_summary()

    # Save results
    output_file = runner.save_results(format="yaml")

    # Update catalog if requested
    if args.update_catalog:
        print("\nUpdating operator catalog...")
        updater = CatalogUpdater(args.update_catalog)

        # Generate report
        report = updater.generate_update_report(results)
        report_path = args.output_dir / "update_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Update report: {report_path}")

        # Apply updates
        status = updater.update_batch(results)
        success = sum(1 for v in status.values() if v)
        print(f"Updated {success}/{len(status)} profiles")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
