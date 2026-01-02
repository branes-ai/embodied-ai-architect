"""Update operator catalog with benchmark results.

Reads benchmark results and updates the perf_profiles in operator YAML files.
"""

from pathlib import Path
from typing import Any
import yaml
import datetime

from .base import OperatorBenchmarkResult


class CatalogUpdater:
    """Updates operator catalog YAML files with benchmark results."""

    def __init__(self, catalog_dir: Path):
        """Initialize updater.

        Args:
            catalog_dir: Path to operators data directory in embodied-schemas
        """
        self.catalog_dir = Path(catalog_dir)
        if not self.catalog_dir.exists():
            raise ValueError(f"Catalog directory not found: {catalog_dir}")

    def find_operator_file(self, operator_id: str) -> Path | None:
        """Find YAML file for an operator.

        Args:
            operator_id: Operator ID (e.g., yolo_detector_n)

        Returns:
            Path to YAML file or None if not found
        """
        for yaml_file in self.catalog_dir.rglob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                    if data and data.get("id") == operator_id:
                        return yaml_file
            except Exception:
                continue
        return None

    def update_operator(
        self,
        result: OperatorBenchmarkResult,
        replace_existing: bool = True,
    ) -> bool:
        """Update operator YAML with benchmark result.

        Args:
            result: Benchmark result to add
            replace_existing: Replace existing profile for same hw/target

        Returns:
            True if updated, False if operator not found
        """
        yaml_file = self.find_operator_file(result.operator_id)
        if not yaml_file:
            print(f"  Warning: Operator {result.operator_id} not found in catalog")
            return False

        # Load existing YAML
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        # Get or create perf_profiles list
        if "perf_profiles" not in data:
            data["perf_profiles"] = []

        profiles = data["perf_profiles"]

        # Create new profile entry
        new_profile = result.to_perf_profile()

        # Check for existing entry
        existing_idx = None
        for i, p in enumerate(profiles):
            if (
                p.get("hardware_id") == result.hardware_id
                and p.get("execution_target") == result.execution_target
            ):
                existing_idx = i
                break

        if existing_idx is not None:
            if replace_existing:
                profiles[existing_idx] = new_profile
                print(f"  Updated existing profile for {result.operator_id}")
            else:
                print(f"  Skipping existing profile for {result.operator_id}")
                return False
        else:
            profiles.append(new_profile)
            print(f"  Added new profile for {result.operator_id}")

        # Update last_updated
        data["last_updated"] = datetime.date.today().isoformat()

        # Write back
        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return True

    def update_batch(
        self,
        results: list[OperatorBenchmarkResult],
        replace_existing: bool = True,
    ) -> dict[str, bool]:
        """Update multiple operators.

        Args:
            results: List of benchmark results
            replace_existing: Replace existing profiles

        Returns:
            Dictionary of operator_id -> success status
        """
        status = {}
        for result in results:
            key = f"{result.operator_id}:{result.hardware_id}:{result.execution_target}"
            status[key] = self.update_operator(result, replace_existing)
        return status

    def generate_update_report(
        self,
        results: list[OperatorBenchmarkResult],
    ) -> str:
        """Generate markdown report of proposed updates.

        Args:
            results: Benchmark results

        Returns:
            Markdown report string
        """
        lines = ["# Operator Catalog Update Report", ""]
        lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
        lines.append("")

        # Group by operator
        by_operator: dict[str, list[OperatorBenchmarkResult]] = {}
        for r in results:
            if r.operator_id not in by_operator:
                by_operator[r.operator_id] = []
            by_operator[r.operator_id].append(r)

        for operator_id, op_results in sorted(by_operator.items()):
            lines.append(f"## {operator_id}")
            lines.append("")

            yaml_file = self.find_operator_file(operator_id)
            if yaml_file:
                lines.append(f"File: `{yaml_file.relative_to(self.catalog_dir.parent)}`")
            else:
                lines.append("**Warning: Operator not found in catalog**")
            lines.append("")

            lines.append("| Hardware | Target | Latency (ms) | Throughput (fps) | Memory (MB) |")
            lines.append("|----------|--------|--------------|------------------|-------------|")

            for r in op_results:
                mem = f"{r.memory_mb:.1f}" if r.memory_mb else "N/A"
                fps = f"{r.throughput_fps:.1f}" if r.throughput_fps else "N/A"
                lines.append(
                    f"| {r.hardware_id} | {r.execution_target} | "
                    f"{r.mean_latency_ms:.2f} ± {r.std_latency_ms:.2f} | {fps} | {mem} |"
                )

            lines.append("")

        return "\n".join(lines)


def load_results_from_file(filepath: Path) -> list[OperatorBenchmarkResult]:
    """Load benchmark results from JSON/YAML file.

    Args:
        filepath: Path to results file

    Returns:
        List of OperatorBenchmarkResult
    """
    import json

    with open(filepath) as f:
        if filepath.suffix == ".yaml":
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    results = []
    for r in data.get("results", []):
        results.append(
            OperatorBenchmarkResult(
                operator_id=r["operator_id"],
                hardware_id=r["hardware_id"],
                execution_target=r["execution_target"],
                mean_latency_ms=r["mean_latency_ms"],
                std_latency_ms=r["std_latency_ms"],
                min_latency_ms=r["min_latency_ms"],
                max_latency_ms=r["max_latency_ms"],
                p50_latency_ms=r["p50_latency_ms"],
                p95_latency_ms=r["p95_latency_ms"],
                p99_latency_ms=r.get("p99_latency_ms", r["p95_latency_ms"]),
                memory_mb=r.get("memory_mb"),
                power_w=r.get("power_w"),
                throughput_fps=r.get("throughput_fps"),
                conditions=r.get("conditions", ""),
                iterations=r.get("iterations", 100),
                measured=r.get("measured", True),
            )
        )

    return results
