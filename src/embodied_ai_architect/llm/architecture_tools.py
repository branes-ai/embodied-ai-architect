"""Tools for architecture-level analysis of embodied AI systems.

Provides LLM-callable tools for:
- End-to-end architecture analysis on hardware targets
- Multi-rate scheduling feasibility checking
- Bottleneck identification at the architecture level
- Architecture listing and visualization
"""

import json
import traceback
from datetime import datetime
from typing import Any

# Import from embodied-schemas
try:
    from embodied_schemas import (
        Registry,
        SoftwareArchitecture,
        OperatorEntry,
        Verdict,
        Confidence,
        architecture_to_mermaid,
    )
    from embodied_schemas.scheduling import (
        ArchitectureAnalysisResult,
        SchedulingAnalysisResult,
        OperatorTiming,
        RateFeasibility,
        ExecutionTargetUtilization,
        DataMovementAnalysis,
        DataTransfer,
    )
    HAS_SCHEMAS = True
except ImportError:
    HAS_SCHEMAS = False
    Registry = None
    SoftwareArchitecture = None


def _check_schemas_available() -> str | None:
    """Check if embodied-schemas is available."""
    if not HAS_SCHEMAS:
        return (
            "Error: embodied-schemas not installed or outdated. "
            "Install with: pip install -e ../embodied-schemas"
        )
    return None


def _get_registry() -> "Registry":
    """Get or create the global registry instance."""
    return Registry.load()


def get_architecture_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for architecture analysis.

    Returns:
        List of tool definitions in Anthropic format
    """
    return [
        {
            "name": "list_architectures",
            "description": (
                "List all available software architectures in the catalog. "
                "Returns architecture IDs, names, platform types, and operator counts."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "platform_type": {
                        "type": "string",
                        "description": (
                            "Filter by platform type: drone, vehicle, manipulator, etc. "
                            "If not specified, returns all architectures."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "show_architecture",
            "description": (
                "Show detailed information about a specific architecture, including "
                "operators, dataflow edges, variants, and a Mermaid diagram."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID (e.g., 'drone_perception_v1')",
                    },
                },
                "required": ["architecture_id"],
            },
        },
        {
            "name": "analyze_architecture",
            "description": (
                "Analyze a software architecture on a specific hardware target. "
                "Returns end-to-end latency, power, memory, critical path, and bottleneck. "
                "Use this to determine if an architecture meets its timing requirements."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID (e.g., 'drone_perception_v1')",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID (e.g., 'Jetson-Orin-Nano')",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID to use instead of base architecture",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
        {
            "name": "check_scheduling",
            "description": (
                "Check if all operators in an architecture can meet their target rates "
                "on a specific hardware platform. Returns PASS/FAIL verdict with "
                "per-operator feasibility analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "architecture_id": {
                        "type": "string",
                        "description": "Architecture ID",
                    },
                    "hardware_id": {
                        "type": "string",
                        "description": "Hardware ID",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Optional variant ID",
                    },
                },
                "required": ["architecture_id", "hardware_id"],
            },
        },
    ]


def list_architectures(platform_type: str | None = None) -> str:
    """List available architectures in the catalog."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()

        if platform_type:
            archs = registry.get_architectures_by_platform(platform_type)
        else:
            archs = list(registry.architectures)

        output = {
            "total": len(archs),
            "architectures": [
                {
                    "id": arch.id,
                    "name": arch.name,
                    "platform_type": arch.platform_type,
                    "num_operators": len(arch.operators),
                    "num_variants": len(arch.variants),
                    "end_to_end_latency_ms": arch.end_to_end_latency_ms,
                    "power_budget_w": arch.power_budget_w,
                }
                for arch in archs
            ],
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2)


def show_architecture(architecture_id: str) -> str:
    """Show detailed information about an architecture."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps({
                "error": f"Architecture '{architecture_id}' not found",
                "available": list(registry.architectures.keys()),
            }, indent=2)

        # Get operator details
        operator_details = []
        for op_inst in arch.operators:
            op_entry = registry.operators.get(op_inst.operator_id)
            operator_details.append({
                "instance_id": op_inst.id,
                "operator_id": op_inst.operator_id,
                "operator_name": op_entry.name if op_entry else "Unknown",
                "category": op_entry.category.value if op_entry else "unknown",
                "rate_hz": op_inst.rate_hz,
                "execution_target": op_inst.execution_target,
                "config": op_inst.config if op_inst.config else None,
            })

        # Generate Mermaid diagram
        mermaid = architecture_to_mermaid(arch)

        output = {
            "id": arch.id,
            "name": arch.name,
            "description": arch.description,
            "platform_type": arch.platform_type,
            "sensors": arch.sensors,
            "actuators": arch.actuators,
            "requirements": {
                "end_to_end_latency_ms": arch.end_to_end_latency_ms,
                "min_throughput_fps": arch.min_throughput_fps,
                "power_budget_w": arch.power_budget_w,
                "memory_budget_mb": arch.memory_budget_mb,
            },
            "operators": operator_details,
            "dataflow": [
                {
                    "from": f"{edge.source_op}.{edge.source_port}",
                    "to": f"{edge.target_op}.{edge.target_port}",
                }
                for edge in arch.dataflow
            ],
            "variants": [
                {
                    "id": var.id,
                    "name": var.name,
                    "description": var.description,
                    "operator_overrides": var.operator_overrides,
                    "target_hardware": var.target_hardware,
                    "expected_latency_ms": var.expected_latency_ms,
                }
                for var in arch.variants
            ],
            "mermaid_diagram": mermaid,
            "reference_impl": arch.reference_impl,
            "tags": arch.tags,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2)


def _get_operator_perf(
    op_entry: "OperatorEntry",
    hardware_id: str,
) -> tuple[float, float, float | None]:
    """Get performance metrics for an operator on specific hardware.

    Returns:
        Tuple of (latency_ms, memory_mb, power_w)
    """
    # Try to find a matching perf profile
    for profile in op_entry.perf_profiles:
        if profile.hardware_id == hardware_id:
            return (
                profile.latency_ms or op_entry.typical_latency_ms or 1.0,
                profile.memory_mb or op_entry.typical_memory_mb or 10.0,
                profile.power_w,
            )

    # Fall back to reference/typical values
    return (
        op_entry.typical_latency_ms or 1.0,
        op_entry.typical_memory_mb or 10.0,
        None,
    )


def _apply_variant(
    arch: "SoftwareArchitecture",
    variant_id: str,
    registry: "Registry",
) -> tuple[dict[str, str], dict[str, dict]]:
    """Apply variant overrides to get effective operator mapping.

    Returns:
        Tuple of (operator_id_map, config_map) where:
        - operator_id_map: instance_id -> effective operator_id
        - config_map: instance_id -> effective config
    """
    # Start with base architecture
    operator_id_map = {op.id: op.operator_id for op in arch.operators}
    config_map = {op.id: op.config for op in arch.operators}

    # Find and apply variant if specified
    if variant_id:
        for var in arch.variants:
            if var.id == variant_id:
                # Apply operator overrides
                for inst_id, new_op_id in var.operator_overrides.items():
                    if inst_id in operator_id_map:
                        operator_id_map[inst_id] = new_op_id

                # Apply config overrides
                for inst_id, config_override in var.config_overrides.items():
                    if inst_id in config_map:
                        merged = dict(config_map[inst_id])
                        merged.update(config_override)
                        config_map[inst_id] = merged
                break

    return operator_id_map, config_map


def analyze_architecture(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
) -> str:
    """Analyze an architecture on specific hardware."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps({
                "error": f"Architecture '{architecture_id}' not found",
            }, indent=2)

        # Check if hardware exists
        hardware = registry.hardware.get(hardware_id)
        hardware_name = hardware.name if hardware else hardware_id

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Analyze each operator
        operator_timings = []
        total_latency_ms = 0.0
        total_memory_mb = 0.0
        total_power_w = 0.0
        bottleneck_op = None
        bottleneck_latency = 0.0
        critical_path = []

        cpu_ops = []
        gpu_ops = []

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if not op_entry:
                # Unknown operator - use estimates
                latency_ms = 1.0
                memory_mb = 10.0
                power_w = None
            else:
                latency_ms, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)

            # Calculate rate feasibility
            target_rate = op_inst.rate_hz
            if target_rate:
                max_rate = 1000.0 / latency_ms if latency_ms > 0 else float('inf')
                rate_feasible = max_rate >= target_rate
            else:
                max_rate = None
                rate_feasible = True

            # Track by execution target
            exec_target = op_inst.execution_target or "cpu"
            if exec_target == "gpu":
                gpu_ops.append(op_inst.id)
            else:
                cpu_ops.append(op_inst.id)

            timing = OperatorTiming(
                operator_instance_id=op_inst.id,
                operator_id=effective_op_id,
                execution_target=exec_target,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                power_w=power_w,
                target_rate_hz=target_rate,
                achievable_rate_hz=max_rate,
                rate_feasible=rate_feasible,
                is_critical_path=True,  # Simplified: assume all on critical path
                limiting_factor=None,
            )
            operator_timings.append(timing)

            # Accumulate totals (simplified: sequential execution)
            total_latency_ms += latency_ms
            total_memory_mb = max(total_memory_mb, memory_mb)  # Peak concurrent
            if power_w:
                total_power_w += power_w

            # Track bottleneck (longest operator)
            if latency_ms > bottleneck_latency:
                bottleneck_latency = latency_ms
                bottleneck_op = op_inst.id

            critical_path.append(op_inst.id)

        # Calculate throughput
        throughput_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

        # Determine verdict based on constraints
        latency_target = arch.end_to_end_latency_ms
        power_budget = arch.power_budget_w

        if latency_target and total_latency_ms > latency_target:
            verdict = Verdict.FAIL
            latency_margin = ((latency_target - total_latency_ms) / latency_target) * 100
        elif latency_target:
            verdict = Verdict.PASS
            latency_margin = ((latency_target - total_latency_ms) / latency_target) * 100
        else:
            verdict = Verdict.PASS
            latency_margin = None

        if power_budget:
            power_margin = ((power_budget - total_power_w) / power_budget) * 100
            if total_power_w > power_budget:
                verdict = Verdict.FAIL
        else:
            power_margin = None

        # Generate summary
        if verdict == Verdict.PASS:
            summary = (
                f"Architecture meets requirements on {hardware_id}: "
                f"{total_latency_ms:.1f}ms latency, {throughput_fps:.1f} FPS"
            )
        else:
            summary = (
                f"Architecture FAILS on {hardware_id}: "
                f"{total_latency_ms:.1f}ms latency exceeds {latency_target}ms target. "
                f"Bottleneck: {bottleneck_op} ({bottleneck_latency:.1f}ms)"
            )

        # Build suggestions
        suggestions = []
        if verdict == Verdict.FAIL:
            if bottleneck_op:
                suggestions.append(
                    f"Optimize or replace '{bottleneck_op}' operator ({bottleneck_latency:.1f}ms)"
                )
            if latency_margin and latency_margin < -20:
                suggestions.append(
                    "Consider using a faster variant or more powerful hardware"
                )
            suggestions.append(
                "Check if operators can run in parallel to reduce critical path"
            )

        # Create result
        result = ArchitectureAnalysisResult(
            verdict=verdict,
            confidence=Confidence.MEDIUM,  # Estimates, not measured
            summary=summary,
            architecture_id=architecture_id,
            architecture_name=arch.name,
            hardware_id=hardware_id,
            hardware_name=hardware_name,
            variant_id=variant_id,
            timestamp=datetime.now(),
            end_to_end_latency_ms=total_latency_ms,
            throughput_fps=throughput_fps,
            total_power_w=total_power_w,
            total_memory_mb=total_memory_mb,
            latency_target_ms=latency_target,
            latency_margin_pct=latency_margin,
            power_budget_w=power_budget,
            power_margin_pct=power_margin,
            operator_timings=operator_timings,
            critical_path=critical_path,
            critical_path_latency_ms=total_latency_ms,
            bottleneck_operator=bottleneck_op,
            bottleneck_latency_ms=bottleneck_latency,
            bottleneck_type="latency",
            suggestions=suggestions,
            warnings=["Analysis uses estimated performance data, not measurements"],
        )

        # Convert to JSON
        output = {
            "verdict": result.verdict.value,
            "confidence": result.confidence.value,
            "summary": result.summary,
            "architecture_id": result.architecture_id,
            "hardware_id": result.hardware_id,
            "variant_id": result.variant_id,
            "metrics": {
                "end_to_end_latency_ms": round(result.end_to_end_latency_ms, 2),
                "throughput_fps": round(result.throughput_fps, 1),
                "total_power_w": round(result.total_power_w, 1),
                "total_memory_mb": round(result.total_memory_mb, 1),
            },
            "constraints": {
                "latency_target_ms": result.latency_target_ms,
                "latency_margin_pct": round(result.latency_margin_pct, 1) if result.latency_margin_pct else None,
                "power_budget_w": result.power_budget_w,
                "power_margin_pct": round(result.power_margin_pct, 1) if result.power_margin_pct else None,
            },
            "bottleneck": {
                "operator": result.bottleneck_operator,
                "latency_ms": round(result.bottleneck_latency_ms, 2) if result.bottleneck_latency_ms else None,
                "type": result.bottleneck_type,
            },
            "critical_path": result.critical_path,
            "operator_timings": [
                {
                    "instance_id": t.operator_instance_id,
                    "operator_id": t.operator_id,
                    "target": t.execution_target,
                    "latency_ms": round(t.latency_ms, 2),
                    "memory_mb": round(t.memory_mb, 1),
                    "rate_feasible": t.rate_feasible,
                }
                for t in result.operator_timings
            ],
            "suggestions": result.suggestions,
            "warnings": result.warnings,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({
            "verdict": "UNKNOWN",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2)


def check_scheduling(
    architecture_id: str,
    hardware_id: str,
    variant_id: str | None = None,
) -> str:
    """Check scheduling feasibility for an architecture on hardware."""
    error = _check_schemas_available()
    if error:
        return error

    try:
        registry = _get_registry()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            return json.dumps({
                "error": f"Architecture '{architecture_id}' not found",
            }, indent=2)

        # Apply variant if specified
        operator_id_map, config_map = _apply_variant(arch, variant_id, registry)

        # Analyze rate feasibility for each operator
        rate_analysis = []
        all_feasible = True
        total_cpu_util = 0.0
        total_gpu_util = 0.0
        worst_latency = 0.0

        cpu_util_list = []
        gpu_util_list = []

        for op_inst in arch.operators:
            effective_op_id = operator_id_map[op_inst.id]
            op_entry = registry.operators.get(effective_op_id)

            if op_entry:
                latency_ms, memory_mb, power_w = _get_operator_perf(op_entry, hardware_id)
            else:
                latency_ms = 1.0
                memory_mb = 10.0

            target_rate = op_inst.rate_hz
            if target_rate:
                # Maximum rate = 1 / latency
                max_rate = 1000.0 / latency_ms if latency_ms > 0 else float('inf')
                achievable = max_rate >= target_rate
                margin = ((max_rate - target_rate) / target_rate) * 100 if target_rate > 0 else 100

                # Calculate utilization at target rate
                # Utilization = (time per invocation) * (invocations per second)
                utilization = (latency_ms / 1000.0) * target_rate  # fraction of time busy

                exec_target = op_inst.execution_target or "cpu"
                if exec_target == "gpu":
                    gpu_util_list.append(utilization)
                else:
                    cpu_util_list.append(utilization)
            else:
                max_rate = None
                achievable = True
                margin = None
                utilization = 0.0

            if not achievable:
                all_feasible = False
                limiting = "compute" if latency_ms > (1000.0 / target_rate) else "dependency"
            else:
                limiting = None

            rate_analysis.append(RateFeasibility(
                operator_instance_id=op_inst.id,
                target_rate_hz=target_rate or 0.0,
                achievable=achievable,
                actual_rate_hz=max_rate,
                margin_pct=margin,
                limiting_factor=limiting,
            ))

            if latency_ms > worst_latency:
                worst_latency = latency_ms

        # Aggregate utilization
        total_cpu_util = sum(cpu_util_list) * 100 if cpu_util_list else 0.0
        total_gpu_util = sum(gpu_util_list) * 100 if gpu_util_list else 0.0

        # Determine verdict
        if all_feasible and total_cpu_util <= 100 and total_gpu_util <= 100:
            verdict = Verdict.PASS
            summary = (
                f"All operator rates achievable on {hardware_id}. "
                f"CPU: {total_cpu_util:.1f}%, GPU: {total_gpu_util:.1f}% utilization."
            )
        else:
            verdict = Verdict.FAIL
            infeasible = [r for r in rate_analysis if not r.achievable]
            if infeasible:
                summary = (
                    f"Scheduling FAILS: {len(infeasible)} operators cannot meet target rate. "
                    f"Worst case: {infeasible[0].operator_instance_id}"
                )
            else:
                summary = (
                    f"Scheduling FAILS: Utilization exceeds 100%. "
                    f"CPU: {total_cpu_util:.1f}%, GPU: {total_gpu_util:.1f}%"
                )

        # Build suggestions
        suggestions = []
        if not all_feasible:
            for r in rate_analysis:
                if not r.achievable:
                    suggestions.append(
                        f"Reduce target rate for '{r.operator_instance_id}' "
                        f"(max: {r.actual_rate_hz:.1f}Hz, target: {r.target_rate_hz:.1f}Hz)"
                    )
        if total_cpu_util > 100:
            suggestions.append("CPU is oversubscribed - reduce rates or offload to GPU")
        if total_gpu_util > 100:
            suggestions.append("GPU is oversubscribed - consider more powerful hardware")

        # Create target utilization entries
        target_utilization = []
        if cpu_util_list:
            target_utilization.append(ExecutionTargetUtilization(
                target_id="cpu",
                utilization_pct=total_cpu_util,
                assigned_operators=[op.id for op in arch.operators
                                    if (op.execution_target or "cpu") == "cpu"],
            ))
        if gpu_util_list:
            target_utilization.append(ExecutionTargetUtilization(
                target_id="gpu",
                utilization_pct=total_gpu_util,
                assigned_operators=[op.id for op in arch.operators
                                    if op.execution_target == "gpu"],
            ))

        result = SchedulingAnalysisResult(
            verdict=verdict,
            confidence=Confidence.MEDIUM,
            summary=summary,
            architecture_id=architecture_id,
            hardware_id=hardware_id,
            variant_id=variant_id,
            timestamp=datetime.now(),
            target_utilization=target_utilization,
            rate_analysis=rate_analysis,
            all_rates_feasible=all_feasible,
            total_cpu_utilization_pct=total_cpu_util,
            total_accelerator_utilization_pct=total_gpu_util if gpu_util_list else None,
            worst_case_latency_ms=worst_latency,
            suggestions=suggestions,
        )

        # Convert to JSON
        output = {
            "verdict": result.verdict.value,
            "confidence": result.confidence.value,
            "summary": result.summary,
            "architecture_id": result.architecture_id,
            "hardware_id": result.hardware_id,
            "all_rates_feasible": result.all_rates_feasible,
            "utilization": {
                "cpu_pct": round(result.total_cpu_utilization_pct, 1),
                "gpu_pct": round(result.total_accelerator_utilization_pct, 1) if result.total_accelerator_utilization_pct else None,
            },
            "worst_case_latency_ms": round(result.worst_case_latency_ms, 2),
            "rate_analysis": [
                {
                    "operator": r.operator_instance_id,
                    "target_hz": r.target_rate_hz,
                    "achievable": r.achievable,
                    "max_hz": round(r.actual_rate_hz, 1) if r.actual_rate_hz else None,
                    "margin_pct": round(r.margin_pct, 1) if r.margin_pct else None,
                    "limiting_factor": r.limiting_factor,
                }
                for r in result.rate_analysis
            ],
            "suggestions": result.suggestions,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({
            "verdict": "UNKNOWN",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2)


def create_architecture_tool_executors() -> dict[str, callable]:
    """Create executor functions for architecture tools.

    Returns:
        Dictionary mapping tool names to executor functions
    """
    return {
        "list_architectures": list_architectures,
        "show_architecture": show_architecture,
        "analyze_architecture": analyze_architecture,
        "check_scheduling": check_scheduling,
    }
