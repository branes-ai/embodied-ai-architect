"""Operator benchmark runner for systematic profiling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import yaml
import datetime

from .base import OperatorBenchmark, OperatorBenchmarkResult


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    hardware_id: str
    execution_targets: list[str] = field(default_factory=lambda: ["cpu"])
    iterations: int = 100
    warmup_iterations: int = 10
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))

    # TDP/power mode for documentation
    tdp_mode: str | None = None
    ambient_temp_c: float | None = None

    # SHA fingerprints from graphs auto_detect for reproducibility
    hardware_fingerprint: str | None = None
    software_fingerprint: str | None = None

    # Software versions
    python_version: str | None = None
    torch_version: str | None = None
    onnxruntime_version: str | None = None


class OperatorBenchmarkRunner:
    """Runs benchmarks for multiple operators and collects results."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[OperatorBenchmarkResult] = []
        self._collect_system_info()

    def _collect_system_info(self) -> None:
        """Collect system information for reproducibility."""
        import sys
        import platform

        self.system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # PyTorch version
        try:
            import torch

            self.system_info["torch_version"] = torch.__version__
            self.system_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.system_info["cuda_device"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        # ONNX Runtime version
        try:
            import onnxruntime as ort

            self.system_info["onnxruntime_version"] = ort.__version__
            self.system_info["onnxruntime_providers"] = ort.get_available_providers()
        except ImportError:
            pass

    def register_benchmark(self, benchmark: OperatorBenchmark) -> None:
        """Register a benchmark to run.

        Args:
            benchmark: OperatorBenchmark instance
        """
        if not hasattr(self, "_benchmarks"):
            self._benchmarks: list[OperatorBenchmark] = []
        self._benchmarks.append(benchmark)

    def run_all(self) -> list[OperatorBenchmarkResult]:
        """Run all registered benchmarks on all configured targets.

        Returns:
            List of all benchmark results
        """
        if not hasattr(self, "_benchmarks"):
            return []

        for benchmark in self._benchmarks:
            for target in self.config.execution_targets:
                print(f"Benchmarking {benchmark.operator_id} on {target}...")

                try:
                    conditions = self._build_conditions(target)
                    result = benchmark.benchmark(
                        hardware_id=self.config.hardware_id,
                        execution_target=target,
                        iterations=self.config.iterations,
                        warmup_iterations=self.config.warmup_iterations,
                        conditions=conditions,
                    )
                    self.results.append(result)
                    self._print_result(result)

                except Exception as e:
                    print(f"  Error: {e}")

                finally:
                    benchmark.teardown()
                    benchmark._is_setup = False

        return self.results

    def _build_conditions(self, target: str) -> str:
        """Build conditions string for result."""
        parts = []

        if self.config.tdp_mode:
            parts.append(f"TDP={self.config.tdp_mode}")

        parts.append(f"target={target}")

        if self.config.ambient_temp_c:
            parts.append(f"ambient={self.config.ambient_temp_c}C")

        return ", ".join(parts)

    def _print_result(self, result: OperatorBenchmarkResult) -> None:
        """Print a single result."""
        print(f"  Mean: {result.mean_latency_ms:.2f}ms ± {result.std_latency_ms:.2f}ms")
        print(f"  P50: {result.p50_latency_ms:.2f}ms, P95: {result.p95_latency_ms:.2f}ms")
        if result.throughput_fps:
            print(f"  Throughput: {result.throughput_fps:.1f} fps")
        if result.memory_mb:
            print(f"  Memory: {result.memory_mb:.1f} MB")

    def save_results(self, format: str = "json") -> Path:
        """Save results to file.

        Args:
            format: Output format (json or yaml)

        Returns:
            Path to saved file
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.config.hardware_id}_{timestamp}"

        # Build config dict with fingerprints if available
        config_dict = {
            "hardware_id": self.config.hardware_id,
            "execution_targets": self.config.execution_targets,
            "iterations": self.config.iterations,
            "warmup_iterations": self.config.warmup_iterations,
            "tdp_mode": self.config.tdp_mode,
        }

        # Include SHA fingerprints for reproducibility tracking
        if self.config.hardware_fingerprint:
            config_dict["hardware_fingerprint"] = self.config.hardware_fingerprint
        if self.config.software_fingerprint:
            config_dict["software_fingerprint"] = self.config.software_fingerprint

        data = {
            "config": config_dict,
            "system_info": self.system_info,
            "results": [self._result_to_dict(r) for r in self.results],
        }

        if format == "yaml":
            output_path = self.config.output_dir / f"{filename}.yaml"
            with open(output_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            output_path = self.config.output_dir / f"{filename}.json"
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def _result_to_dict(self, result: OperatorBenchmarkResult) -> dict:
        """Convert result to dictionary, excluding raw timings."""
        return {
            "operator_id": result.operator_id,
            "hardware_id": result.hardware_id,
            "execution_target": result.execution_target,
            "mean_latency_ms": round(result.mean_latency_ms, 3),
            "std_latency_ms": round(result.std_latency_ms, 3),
            "min_latency_ms": round(result.min_latency_ms, 3),
            "max_latency_ms": round(result.max_latency_ms, 3),
            "p50_latency_ms": round(result.p50_latency_ms, 3),
            "p95_latency_ms": round(result.p95_latency_ms, 3),
            "p99_latency_ms": round(result.p99_latency_ms, 3),
            "memory_mb": result.memory_mb,
            "power_w": result.power_w,
            "throughput_fps": round(result.throughput_fps, 1) if result.throughput_fps else None,
            "conditions": result.conditions,
            "iterations": result.iterations,
            "measured": result.measured,
        }

    def generate_perf_profiles(self) -> dict[str, list[dict]]:
        """Generate perf_profiles entries for updating operator YAML files.

        Returns:
            Dictionary mapping operator_id to list of perf_profile entries
        """
        profiles: dict[str, list[dict]] = {}

        for result in self.results:
            if result.operator_id not in profiles:
                profiles[result.operator_id] = []

            profiles[result.operator_id].append(result.to_perf_profile())

        return profiles

    def print_summary(self) -> None:
        """Print summary of all results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Hardware: {self.config.hardware_id}")
        if self.config.hardware_fingerprint:
            print(f"HW Fingerprint: {self.config.hardware_fingerprint}")
            print(f"SW Fingerprint: {self.config.software_fingerprint}")
        print(f"Targets: {', '.join(self.config.execution_targets)}")
        print(f"Iterations: {self.config.iterations}")
        print()

        # Group by operator
        by_operator: dict[str, list[OperatorBenchmarkResult]] = {}
        for result in self.results:
            if result.operator_id not in by_operator:
                by_operator[result.operator_id] = []
            by_operator[result.operator_id].append(result)

        for operator_id, results in by_operator.items():
            print(f"\n{operator_id}:")
            for result in results:
                print(
                    f"  [{result.execution_target}] "
                    f"{result.mean_latency_ms:.2f}ms ± {result.std_latency_ms:.2f}ms "
                    f"({result.throughput_fps:.1f} fps)"
                )
