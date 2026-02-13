"""Specialist agent executors for the agentic SoC designer.

Each specialist conforms to the AgentExecutor protocol:
    (task: TaskNode, state: SoCDesignState) -> dict[str, Any]

Specialists read from state (constraints, previous task results) and return
a result dict. If the result contains a '_state_updates' key, the Dispatcher
merges those into the top-level SoCDesignState.

Usage:
    from embodied_ai_architect.graphs.specialists import create_default_dispatcher

    dispatcher = create_default_dispatcher()
    state = dispatcher.run(state)
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.dispatcher import Dispatcher
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    PPAMetrics,
    SoCDesignState,
    get_constraints,
    get_dependency_results,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workload Analyzer
# ---------------------------------------------------------------------------


def workload_analyzer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Analyze AI workload: operator graph, compute/memory requirements.

    Wraps the existing ModelAnalyzerAgent when a model path is available.
    Falls back to constraint-based estimation from the goal description.

    Writes to state: workload_profile
    """
    goal = state.get("goal", "")
    use_case = state.get("use_case", "")
    constraints = get_constraints(state)

    # Try to use existing ModelAnalyzerAgent if available
    model_analysis = _try_model_analysis(state)

    if model_analysis:
        profile = _build_profile_from_model(model_analysis, use_case)
    else:
        profile = _estimate_workload_from_goal(goal, use_case, constraints)

    return {
        "summary": f"Workload analyzed for {use_case or 'target application'}",
        "workload_profile": profile,
        "_state_updates": {"workload_profile": profile},
    }


def _try_model_analysis(state: SoCDesignState) -> dict[str, Any] | None:
    """Try to run ModelAnalyzerAgent if a model path is in state."""
    try:
        from embodied_ai_architect.agents.model_analyzer import ModelAnalyzerAgent

        model_path = state.get("workload_profile", {}).get("model_path")
        if not model_path:
            return None

        agent = ModelAnalyzerAgent()
        result = agent.execute({"model": model_path})
        if result.success:
            return result.data
    except (ImportError, Exception) as e:
        logger.debug("ModelAnalyzerAgent not available: %s", e)
    return None


def _build_profile_from_model(
    model_data: dict[str, Any], use_case: str
) -> dict[str, Any]:
    """Build workload profile from ModelAnalyzerAgent output."""
    layer_counts = model_data.get("layer_type_counts", {})
    total_params = model_data.get("total_parameters", 0)

    # Classify operator types
    operators = []
    if layer_counts.get("Conv2d", 0) > 0:
        operators.append({"type": "convolution", "count": layer_counts["Conv2d"]})
    if layer_counts.get("Linear", 0) > 0:
        operators.append({"type": "matrix_multiply", "count": layer_counts["Linear"]})
    if layer_counts.get("MultiheadAttention", 0) > 0:
        operators.append({"type": "attention", "count": layer_counts["MultiheadAttention"]})
    for ltype, count in layer_counts.items():
        if ltype not in ("Conv2d", "Linear", "MultiheadAttention"):
            operators.append({"type": ltype.lower(), "count": count})

    # Estimate compute requirements
    params_m = total_params / 1e6
    estimated_gflops = params_m * 2.0  # rough: 2 FLOPs per parameter per inference
    estimated_memory_mb = (total_params * 4) / (1024 * 1024)  # FP32

    return {
        "model_type": model_data.get("model_type", "unknown"),
        "total_parameters": total_params,
        "operators": operators,
        "estimated_gflops": round(estimated_gflops, 2),
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "dominant_op": operators[0]["type"] if operators else "unknown",
        "use_case": use_case,
        "source": "model_analysis",
    }


def _estimate_workload_from_goal(
    goal: str, use_case: str, constraints: DesignConstraints
) -> dict[str, Any]:
    """Estimate workload profile from goal description when no model is available.

    Uses heuristics based on common embodied AI workloads.
    """
    goal_lower = goal.lower()

    # Detect workload types from keywords
    workloads = []
    if any(kw in goal_lower for kw in ("detection", "yolo", "object")):
        workloads.append({
            "name": "object_detection",
            "model_class": "YOLOv8",
            "operators": [
                {"type": "convolution", "count": 75},
                {"type": "batch_norm", "count": 70},
                {"type": "activation", "count": 70},
            ],
            "estimated_gflops": 8.7,
            "estimated_memory_mb": 12.0,
            "estimated_params_m": 3.2,
        })
    if any(kw in goal_lower for kw in ("tracking", "track", "bytetrack")):
        workloads.append({
            "name": "object_tracking",
            "model_class": "ByteTrack",
            "operators": [
                {"type": "matrix_multiply", "count": 10},
                {"type": "sort", "count": 1},
            ],
            "estimated_gflops": 0.5,
            "estimated_memory_mb": 2.0,
            "estimated_params_m": 0.0,
        })
    if any(kw in goal_lower for kw in ("slam", "localization", "mapping")):
        workloads.append({
            "name": "visual_slam",
            "model_class": "ORB-SLAM3",
            "operators": [
                {"type": "feature_extraction", "count": 1},
                {"type": "matrix_multiply", "count": 20},
                {"type": "sparse_solve", "count": 1},
            ],
            "estimated_gflops": 3.0,
            "estimated_memory_mb": 50.0,
            "estimated_params_m": 0.0,
        })
    if any(kw in goal_lower for kw in ("perception", "vision", "camera")):
        workloads.append({
            "name": "visual_perception",
            "model_class": "CNN-based",
            "operators": [
                {"type": "convolution", "count": 50},
                {"type": "pooling", "count": 10},
            ],
            "estimated_gflops": 6.0,
            "estimated_memory_mb": 10.0,
            "estimated_params_m": 2.5,
        })
    if any(kw in goal_lower for kw in ("voice", "speech", "audio")):
        workloads.append({
            "name": "voice_recognition",
            "model_class": "Whisper-tiny",
            "operators": [
                {"type": "attention", "count": 4},
                {"type": "convolution", "count": 2},
                {"type": "matrix_multiply", "count": 20},
            ],
            "estimated_gflops": 1.5,
            "estimated_memory_mb": 150.0,
            "estimated_params_m": 39.0,
        })
    if any(kw in goal_lower for kw in ("lidar", "point cloud")):
        workloads.append({
            "name": "lidar_processing",
            "model_class": "PointPillars",
            "operators": [
                {"type": "convolution", "count": 30},
                {"type": "scatter", "count": 1},
            ],
            "estimated_gflops": 4.0,
            "estimated_memory_mb": 20.0,
            "estimated_params_m": 1.8,
        })

    # Default if nothing matched
    if not workloads:
        workloads.append({
            "name": "general_inference",
            "model_class": "Unknown",
            "operators": [{"type": "general_purpose", "count": 1}],
            "estimated_gflops": 5.0,
            "estimated_memory_mb": 20.0,
            "estimated_params_m": 2.0,
        })

    # Add scheduling field per workload (concurrent/sequential/time_shared)
    for w in workloads:
        if "scheduling" not in w:
            w["scheduling"] = _infer_scheduling(w.get("name", ""))

    total_gflops = aggregate_workload_requirements(workloads)
    total_memory_mb = sum(w["estimated_memory_mb"] for w in workloads)

    # Determine dominant operation type
    all_ops: dict[str, int] = {}
    for w in workloads:
        for op in w["operators"]:
            all_ops[op["type"]] = all_ops.get(op["type"], 0) + op["count"]
    dominant_op = max(all_ops, key=all_ops.get) if all_ops else "unknown"  # type: ignore[arg-type]

    return {
        "workloads": workloads,
        "total_estimated_gflops": round(total_gflops, 2),
        "total_estimated_memory_mb": round(total_memory_mb, 2),
        "dominant_op": dominant_op,
        "workload_count": len(workloads),
        "use_case": use_case,
        "source": "goal_estimation",
    }


def _infer_scheduling(workload_name: str) -> str:
    """Infer scheduling mode from workload name."""
    concurrent = {"object_detection", "visual_perception", "visual_slam"}
    sequential = {"voice_recognition"}
    if workload_name in concurrent:
        return "concurrent"
    if workload_name in sequential:
        return "sequential"
    return "time_shared"


def aggregate_workload_requirements(workloads: list[dict]) -> float:
    """Aggregate GFLOPS across workloads considering scheduling mode.

    Concurrent workloads sum their GFLOPS (peak demand).
    Sequential workloads take the max (only one active at a time).
    Time-shared workloads contribute 70% of their GFLOPS (multiplexed).

    Args:
        workloads: List of workload dicts with estimated_gflops and scheduling.

    Returns:
        Peak aggregate GFLOPS requirement.
    """
    concurrent_gflops = 0.0
    sequential_gflops: list[float] = []
    time_shared_gflops = 0.0

    for w in workloads:
        gflops = w.get("estimated_gflops", 0.0)
        scheduling = w.get("scheduling", "concurrent")
        if scheduling == "concurrent":
            concurrent_gflops += gflops
        elif scheduling == "sequential":
            sequential_gflops.append(gflops)
        else:  # time_shared
            time_shared_gflops += gflops * 0.7

    return concurrent_gflops + (max(sequential_gflops) if sequential_gflops else 0.0) + time_shared_gflops


def map_workloads_to_accelerators(
    workloads: list[dict], hardware_candidates: list[dict]
) -> list[dict]:
    """Map workloads to available accelerators for heterogeneous execution.

    Each workload is assigned to the best-matching accelerator based on
    operator type compatibility and available compute.

    Args:
        workloads: List of workload dicts.
        hardware_candidates: Available hardware platforms.

    Returns:
        List of mapping dicts with workload, accelerator, and precision.
    """
    if not hardware_candidates:
        return []

    mappings = []
    for w in workloads:
        best_hw = hardware_candidates[0]  # default to top-ranked
        dominant_op = ""
        if w.get("operators"):
            dominant_op = max(w["operators"], key=lambda o: o.get("count", 0)).get("type", "")

        # Find best match for this workload's dominant operator
        for hw in hardware_candidates:
            strengths = hw.get("strengths", [])
            if dominant_op == "convolution" and any(
                s in strengths for s in ("dataflow", "int8", "gpu_acceleration")
            ):
                best_hw = hw
                break
            if dominant_op == "attention" and any(
                s in strengths for s in ("gpu_acceleration", "flexible")
            ):
                best_hw = hw
                break

        precision = "int8" if best_hw.get("peak_tops_int8", 0) > 0 else "fp16"
        mappings.append({
            "workload": w.get("name", "unknown"),
            "model_class": w.get("model_class", "unknown"),
            "accelerator": best_hw.get("name", "unknown"),
            "precision": precision,
            "scheduling": w.get("scheduling", "concurrent"),
        })

    return mappings


# ---------------------------------------------------------------------------
# Hardware Explorer
# ---------------------------------------------------------------------------


def hw_explorer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Enumerate and score hardware candidates against constraints.

    Wraps the existing HardwareProfileAgent when available, and adds
    constraint-based filtering and scoring.

    Writes to state: hardware_candidates
    """
    constraints = get_constraints(state)
    dep_results = get_dependency_results(state, task)

    # Collect workload profile from dependencies
    workload = None
    for dep_id, result in dep_results.items():
        if "workload_profile" in result:
            workload = result["workload_profile"]
            break

    # Also check top-level state
    if workload is None:
        workload = state.get("workload_profile", {})

    # Use static catalog for consistent field naming and KPU inclusion.
    # The existing HardwareProfileAgent uses a different schema — it will be
    # integrated once field names are normalized across embodied-schemas.
    candidates = _estimate_hardware_candidates(workload, constraints)

    # Filter by constraints
    filtered = _filter_by_constraints(candidates, constraints)

    return {
        "summary": f"Explored {len(filtered)} hardware candidates from {len(candidates)} evaluated",
        "hardware_candidates": filtered,
        "total_evaluated": len(candidates),
        "_state_updates": {"hardware_candidates": filtered},
    }


def _try_hardware_agent(
    workload: dict[str, Any], constraints: DesignConstraints
) -> list[dict[str, Any]]:
    """Try to use existing HardwareProfileAgent."""
    try:
        from embodied_ai_architect.agents.hardware_profile.agent import HardwareProfileAgent

        agent = HardwareProfileAgent()
        input_data: dict[str, Any] = {
            "model_analysis": {
                "total_parameters": workload.get("total_parameters", 0),
                "total_layers": 0,
                "layer_type_counts": {},
            },
            "constraints": {},
            "top_n": 10,
        }
        if constraints.max_latency_ms:
            input_data["constraints"]["max_latency_ms"] = constraints.max_latency_ms
        if constraints.max_power_watts:
            input_data["constraints"]["max_power_watts"] = constraints.max_power_watts
        if constraints.max_cost_usd:
            input_data["constraints"]["max_cost_usd"] = constraints.max_cost_usd

        result = agent.execute(input_data)
        if result.success:
            return result.data.get("recommendations", [])
    except (ImportError, Exception) as e:
        logger.debug("HardwareProfileAgent not available: %s", e)
    return []


def _estimate_hardware_candidates(
    workload: dict[str, Any], constraints: DesignConstraints
) -> list[dict[str, Any]]:
    """Generate hardware candidates from heuristics when agent not available."""
    total_gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))
    dominant_op = workload.get("dominant_op", "general_purpose")

    # Static catalog of representative hardware platforms
    catalog = [
        {
            "name": "Stillwater KPU",
            "vendor": "Stillwater Supercomputing",
            "type": "kpu",
            "compute_paradigm": "dataflow",
            "peak_tops_int8": 20.0,
            "peak_tflops_fp16": 5.0,
            "memory_gb": 2.0,
            "tdp_watts": 5.0,
            "cost_usd": 25.0,
            "strengths": ["low_power", "dataflow", "int8", "custom_operators"],
            "suitable_for": ["edge", "drone", "amr"],
        },
        {
            "name": "NVIDIA Jetson Orin Nano",
            "vendor": "NVIDIA",
            "type": "gpu",
            "compute_paradigm": "simd",
            "peak_tops_int8": 40.0,
            "peak_tflops_fp16": 20.0,
            "memory_gb": 8.0,
            "tdp_watts": 15.0,
            "cost_usd": 199.0,
            "strengths": ["gpu_acceleration", "cuda", "tensorrt", "wide_framework_support"],
            "suitable_for": ["edge", "drone", "amr", "quadruped"],
        },
        {
            "name": "Google Coral Edge TPU",
            "vendor": "Google",
            "type": "tpu",
            "compute_paradigm": "systolic_array",
            "peak_tops_int8": 4.0,
            "peak_tflops_fp16": 0.0,
            "memory_gb": 0.008,
            "tdp_watts": 2.0,
            "cost_usd": 35.0,
            "strengths": ["ultra_low_power", "int8_only", "fast_inference"],
            "suitable_for": ["edge", "iot", "camera"],
        },
        {
            "name": "AMD Ryzen AI (Xilinx NPU)",
            "vendor": "AMD",
            "type": "npu",
            "compute_paradigm": "reconfigurable",
            "peak_tops_int8": 16.0,
            "peak_tflops_fp16": 8.0,
            "memory_gb": 16.0,
            "tdp_watts": 25.0,
            "cost_usd": 150.0,
            "strengths": ["heterogeneous", "cpu_plus_npu", "flexible"],
            "suitable_for": ["edge", "amr", "quadruped"],
        },
        {
            "name": "Raspberry Pi 5",
            "vendor": "Raspberry Pi Foundation",
            "type": "cpu",
            "compute_paradigm": "von_neumann",
            "peak_tops_int8": 0.5,
            "peak_tflops_fp16": 0.1,
            "memory_gb": 8.0,
            "tdp_watts": 8.0,
            "cost_usd": 80.0,
            "strengths": ["general_purpose", "low_cost", "ecosystem"],
            "suitable_for": ["edge", "education", "prototype"],
        },
        {
            "name": "Hailo-8",
            "vendor": "Hailo",
            "type": "npu",
            "compute_paradigm": "dataflow",
            "peak_tops_int8": 26.0,
            "peak_tflops_fp16": 0.0,
            "memory_gb": 0.0,  # No on-chip memory, uses host
            "tdp_watts": 2.5,
            "cost_usd": 70.0,
            "strengths": ["high_efficiency", "low_power", "dataflow"],
            "suitable_for": ["edge", "camera", "drone"],
        },
    ]

    # Score each candidate
    scored = []
    for hw in catalog:
        score = _score_hardware(hw, total_gflops, dominant_op, constraints)
        scored.append({**hw, "score": score, "rank": 0})

    # Sort by score descending and assign ranks
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, hw in enumerate(scored):
        hw["rank"] = i + 1

    return scored


def _score_hardware(
    hw: dict[str, Any],
    required_gflops: float,
    dominant_op: str,
    constraints: DesignConstraints,
) -> float:
    """Score a hardware candidate (0-100) based on workload fit and constraints."""
    score = 50.0  # base score

    # Performance adequacy (can it handle the workload?)
    peak_int8 = hw.get("peak_tops_int8", 0)
    if peak_int8 > 0:
        perf_ratio = peak_int8 / max(required_gflops / 1000, 0.001)
        score += min(perf_ratio * 10, 20)  # up to +20 for headroom

    # Power fit
    if constraints.max_power_watts and hw.get("tdp_watts"):
        if hw["tdp_watts"] <= constraints.max_power_watts:
            power_headroom = 1 - (hw["tdp_watts"] / constraints.max_power_watts)
            score += power_headroom * 15  # up to +15 for power headroom
        else:
            score -= 30  # heavy penalty for exceeding power budget

    # Cost fit
    if constraints.max_cost_usd and hw.get("cost_usd"):
        if hw["cost_usd"] <= constraints.max_cost_usd:
            cost_headroom = 1 - (hw["cost_usd"] / constraints.max_cost_usd)
            score += cost_headroom * 10  # up to +10 for cost headroom
        else:
            score -= 20  # penalty for exceeding cost budget

    # Operation type match
    strengths = hw.get("strengths", [])
    if dominant_op == "convolution" and any(s in strengths for s in ("gpu_acceleration", "dataflow", "int8")):
        score += 10
    if dominant_op == "attention" and any(s in strengths for s in ("gpu_acceleration", "flexible")):
        score += 10
    if dominant_op == "general_purpose" and "general_purpose" in strengths:
        score += 5

    # Multi-workload bonus: heterogeneous platforms handle diverse workloads better
    if "heterogeneous" in strengths or "flexible" in strengths:
        score += 5

    return round(max(0, min(100, score)), 1)


def _filter_by_constraints(
    candidates: list[dict[str, Any]], constraints: DesignConstraints
) -> list[dict[str, Any]]:
    """Add constraint verdict annotations to candidates."""
    for c in candidates:
        verdicts = {}
        if constraints.max_power_watts and c.get("tdp_watts"):
            verdicts["power"] = "PASS" if c["tdp_watts"] <= constraints.max_power_watts else "FAIL"
        if constraints.max_cost_usd and c.get("cost_usd"):
            verdicts["cost"] = "PASS" if c["cost_usd"] <= constraints.max_cost_usd else "FAIL"
        c["constraint_verdicts"] = verdicts
    return candidates


# ---------------------------------------------------------------------------
# Architecture Composer
# ---------------------------------------------------------------------------


def architecture_composer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Compose SoC architecture from workload analysis and hardware candidates.

    Maps workload operators to hardware accelerators, designs memory hierarchy,
    and defines interconnect topology.

    Writes to state: selected_architecture, ip_blocks, memory_map, interconnect
    """
    constraints = get_constraints(state)
    dep_results = get_dependency_results(state, task)

    # Gather inputs from dependencies and state
    workload = state.get("workload_profile", {})
    candidates = state.get("hardware_candidates", [])
    for result in dep_results.values():
        if "workload_profile" in result:
            workload = result["workload_profile"]
        if "hardware_candidates" in result:
            candidates = result["hardware_candidates"]

    # Select top hardware candidate that passes constraints
    selected_hw = None
    for hw in candidates:
        verdicts = hw.get("constraint_verdicts", {})
        if all(v == "PASS" for v in verdicts.values()):
            selected_hw = hw
            break
    if selected_hw is None and candidates:
        selected_hw = candidates[0]  # fallback to best-scored

    if selected_hw is None:
        raise ValueError("No hardware candidates available for architecture composition")

    # Compose architecture
    architecture = _compose_architecture(workload, selected_hw, constraints)
    ip_blocks = _define_ip_blocks(workload, selected_hw)
    memory_map = _define_memory_map(ip_blocks, selected_hw)
    interconnect = _define_interconnect(ip_blocks)

    return {
        "summary": f"Composed SoC architecture around {selected_hw['name']}",
        "selected_architecture": architecture,
        "ip_blocks": ip_blocks,
        "memory_map": memory_map,
        "interconnect": interconnect,
        "_state_updates": {
            "selected_architecture": architecture,
            "ip_blocks": ip_blocks,
            "memory_map": memory_map,
            "interconnect": interconnect,
        },
    }


def _compose_architecture(
    workload: dict[str, Any],
    hw: dict[str, Any],
    constraints: DesignConstraints,
) -> dict[str, Any]:
    """Build top-level architecture description."""
    return {
        "name": f"SoC-{hw['name'].replace(' ', '-')}",
        "primary_compute": hw["name"],
        "compute_type": hw.get("type", "unknown"),
        "compute_paradigm": hw.get("compute_paradigm", "unknown"),
        "target_process_nm": constraints.target_process_nm or 28,
        "workload_mapping": _map_workloads_to_hw(workload, hw),
        "design_rationale": (
            f"Selected {hw['name']} as primary compute for "
            f"{workload.get('dominant_op', 'general')} workload "
            f"with {hw.get('tdp_watts', '?')}W TDP"
        ),
    }


def _map_workloads_to_hw(
    workload: dict[str, Any], hw: dict[str, Any]
) -> list[dict[str, str]]:
    """Map workload operators/subworkloads to hardware execution targets."""
    mappings = []
    workloads = workload.get("workloads", [])
    if not workloads:
        # Single-model workload
        mappings.append({
            "workload": workload.get("use_case", "main"),
            "target": hw["name"],
            "precision": "int8" if hw.get("peak_tops_int8", 0) > 0 else "fp16",
        })
    else:
        for w in workloads:
            precision = "int8" if hw.get("peak_tops_int8", 0) > 0 else "fp16"
            mappings.append({
                "workload": w["name"],
                "model_class": w.get("model_class", "unknown"),
                "target": hw["name"],
                "precision": precision,
            })
    return mappings


def _define_ip_blocks(
    workload: dict[str, Any], hw: dict[str, Any]
) -> list[dict[str, Any]]:
    """Define IP blocks for the SoC."""
    blocks = [
        {
            "name": "cpu_subsystem",
            "type": "cpu",
            "description": "Application processor for control and orchestration",
            "config": {"cores": 2, "isa": "RISC-V", "frequency_mhz": 1000},
        },
        {
            "name": "compute_engine",
            "type": hw.get("type", "accelerator"),
            "description": f"{hw['name']} compute engine",
            "config": {
                "peak_tops_int8": hw.get("peak_tops_int8", 0),
                "peak_tflops_fp16": hw.get("peak_tflops_fp16", 0),
                "paradigm": hw.get("compute_paradigm", "unknown"),
            },
        },
        {
            "name": "memory_controller",
            "type": "memory",
            "description": "DDR/LPDDR memory controller",
            "config": {
                "type": "LPDDR4X",
                "capacity_gb": min(hw.get("memory_gb", 2), 4),
                "bandwidth_gbps": 25.6,
            },
        },
        {
            "name": "io_subsystem",
            "type": "io",
            "description": "Camera, sensor, and communication interfaces",
            "config": {"mipi_csi": 2, "i2c": 4, "spi": 2, "uart": 2, "gpio": 32},
        },
    ]

    # Add ISP if vision workload
    dominant = workload.get("dominant_op", "")
    if dominant in ("convolution", "feature_extraction") or any(
        w.get("name", "") in ("object_detection", "visual_perception", "visual_slam")
        for w in workload.get("workloads", [])
    ):
        blocks.append({
            "name": "isp",
            "type": "isp",
            "description": "Image Signal Processor for camera input",
            "config": {"max_resolution": "4K", "pipeline_stages": 8},
        })

    return blocks


def _define_memory_map(
    ip_blocks: list[dict[str, Any]], hw: dict[str, Any]
) -> dict[str, Any]:
    """Define address space layout."""
    base = 0x0000_0000
    regions = []
    for block in ip_blocks:
        size = 0x0100_0000  # 16MB per block default
        regions.append({
            "name": block["name"],
            "base_address": f"0x{base:08X}",
            "size_bytes": size,
        })
        base += size

    # DRAM region
    dram_gb = min(hw.get("memory_gb", 2), 4)
    regions.append({
        "name": "dram",
        "base_address": f"0x{0x8000_0000:08X}",
        "size_bytes": int(dram_gb * 1024 * 1024 * 1024),
    })

    return {"regions": regions, "address_width_bits": 32}


def _define_interconnect(ip_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Define bus/NoC topology."""
    return {
        "type": "AXI4" if len(ip_blocks) <= 6 else "NoC",
        "data_width_bits": 128,
        "frequency_mhz": 500,
        "topology": "star" if len(ip_blocks) <= 4 else "ring",
        "connected_blocks": [b["name"] for b in ip_blocks],
    }


# ---------------------------------------------------------------------------
# PPA Assessor
# ---------------------------------------------------------------------------


def ppa_assessor(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Evaluate a proposed architecture against PPA constraints.

    Estimates power, performance (latency), and area from the architecture
    and hardware specs. Produces verdicts per constraint.

    Writes to state: ppa_metrics
    """
    constraints = get_constraints(state)
    architecture = state.get("selected_architecture", {})
    ip_blocks = state.get("ip_blocks", [])
    workload = state.get("workload_profile", {})
    candidates = state.get("hardware_candidates", [])

    # Also check dependency results
    dep_results = get_dependency_results(state, task)
    for result in dep_results.values():
        if "selected_architecture" in result:
            architecture = result["selected_architecture"]
        if "ip_blocks" in result:
            ip_blocks = result["ip_blocks"]

    # Find the selected hardware
    selected_hw = None
    for hw in candidates:
        if hw.get("name") == architecture.get("primary_compute"):
            selected_hw = hw
            break
    if selected_hw is None and candidates:
        selected_hw = candidates[0]

    # Estimate PPA
    power_w = _estimate_power(ip_blocks, selected_hw, workload)
    latency_ms = _estimate_latency(workload, selected_hw)
    area_mm2 = _estimate_area(ip_blocks, constraints)
    cost_usd = selected_hw.get("cost_usd", 0) if selected_hw else 0
    memory_mb = sum(
        b.get("config", {}).get("capacity_gb", 0) * 1024
        for b in ip_blocks
        if b.get("type") == "memory"
    )

    # Generate verdicts
    verdicts = {}
    bottlenecks = []
    suggestions = []

    if constraints.max_power_watts and power_w is not None:
        if power_w <= constraints.max_power_watts:
            verdicts["power"] = "PASS"
        else:
            verdicts["power"] = "FAIL"
            bottlenecks.append(f"Power {power_w:.1f}W exceeds {constraints.max_power_watts}W budget")
            suggestions.append("Consider lower-power accelerator or reducing clock frequency")

    if constraints.max_latency_ms and latency_ms is not None:
        if latency_ms <= constraints.max_latency_ms:
            verdicts["latency"] = "PASS"
        else:
            verdicts["latency"] = "FAIL"
            bottlenecks.append(
                f"Latency {latency_ms:.1f}ms exceeds {constraints.max_latency_ms}ms target"
            )
            suggestions.append("Consider INT8 quantization or model pruning")

    if constraints.max_cost_usd and cost_usd is not None:
        if cost_usd <= constraints.max_cost_usd:
            verdicts["cost"] = "PASS"
        else:
            verdicts["cost"] = "FAIL"
            bottlenecks.append(f"Cost ${cost_usd:.0f} exceeds ${constraints.max_cost_usd:.0f} budget")

    if constraints.max_area_mm2 and area_mm2 is not None:
        if area_mm2 <= constraints.max_area_mm2:
            verdicts["area"] = "PASS"
        else:
            verdicts["area"] = "FAIL"
            bottlenecks.append(f"Area {area_mm2:.1f}mm² exceeds {constraints.max_area_mm2}mm² target")

    all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False
    overall = "PASS" if all_pass else "FAIL"

    ppa = PPAMetrics(
        power_watts=power_w,
        latency_ms=latency_ms,
        area_mm2=area_mm2,
        cost_usd=cost_usd,
        memory_mb=memory_mb,
        verdicts=verdicts,
        bottlenecks=bottlenecks,
        suggestions=suggestions,
    )

    return {
        "summary": f"PPA assessment: {overall} ({len(verdicts)} constraints checked)",
        "overall_verdict": overall,
        "ppa_metrics": ppa.model_dump(),
        "_state_updates": {"ppa_metrics": ppa.model_dump()},
    }


def _estimate_power(
    ip_blocks: list[dict[str, Any]],
    hw: dict[str, Any] | None,
    workload: dict[str, Any] | None = None,
) -> float:
    """Estimate total SoC power consumption.

    Baseline power is TDP * utilization + fixed overhead.
    When the optimizer applies strategies (tracked in workload["optimizations_applied"]),
    compute power is reduced proportionally to the cumulative GFLOPS reduction.
    Clock scaling in IP blocks also reduces compute power.
    """
    if hw is None:
        return 10.0  # default estimate

    tdp = hw.get("tdp_watts", 10.0)
    base_utilization = 0.8  # baseline: 80% of TDP under full workload

    # When optimizations have been applied, the workload's GFLOPS have been reduced
    # by the optimizer. Compute the ratio vs. the original GFLOPS to scale power.
    optimization_scale = 1.0
    if workload and workload.get("optimizations_applied"):
        # The optimizer reduces total_estimated_gflops each time it applies a strategy.
        # We infer the original by working backwards, but the simpler approach:
        # count the number of workloads and check if gflops is much lower than expected.
        # For robustness, use the ratio of current gflops to a reference value.
        current_gflops = workload.get(
            "total_estimated_gflops", workload.get("estimated_gflops", 0)
        )
        # Reference: typical perception workload is ~15 GFLOPS unoptimized
        # We use per-use-case reference to stay general
        reference_gflops = _reference_gflops(workload)
        if reference_gflops > 0 and current_gflops > 0:
            optimization_scale = min(current_gflops / reference_gflops, 1.0)
            optimization_scale = max(optimization_scale, 0.20)  # floor: 20% of base

    # Check for clock scaling in IP blocks
    clock_scale = 1.0
    for block in ip_blocks:
        if block.get("type") in ("kpu", "gpu", "npu", "tpu", "accelerator"):
            config = block.get("config", {})
            freq = config.get("frequency_mhz", 1000)
            if freq < 1000:
                clock_scale = freq / 1000.0

    compute_w = tdp * base_utilization * optimization_scale * clock_scale
    cpu_w = 1.0  # control CPU
    io_w = 0.5  # I/O subsystem
    memory_w = 0.8  # memory controller

    return round(compute_w + cpu_w + io_w + memory_w, 1)


def _reference_gflops(workload: dict[str, Any]) -> float:
    """Get reference GFLOPS for a workload type (used for power scaling)."""
    # Sum up the workloads' GFLOPS from the model-class heuristics
    workloads = workload.get("workloads", [])
    if workloads:
        # Reference = sum of standard model GFLOPS for each workload type
        ref_map = {
            "object_detection": 8.7,
            "object_tracking": 0.5,
            "visual_perception": 6.0,
            "visual_slam": 3.0,
            "voice_recognition": 1.5,
            "lidar_processing": 4.0,
            "general_inference": 5.0,
        }
        total = sum(ref_map.get(w.get("name", ""), 5.0) for w in workloads)
        return total
    # Single-workload fallback
    return workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))


def _estimate_latency(
    workload: dict[str, Any], hw: dict[str, Any] | None
) -> float:
    """Estimate end-to-end inference latency."""
    if hw is None:
        return 50.0  # default estimate

    total_gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))
    peak_tops = hw.get("peak_tops_int8", 1.0)

    if peak_tops <= 0:
        peak_tops = hw.get("peak_tflops_fp16", 1.0)

    # Rough model: latency = compute / (throughput * utilization)
    utilization = 0.3  # typical 30% utilization
    if peak_tops > 0:
        latency_s = (total_gflops / 1000) / (peak_tops * utilization)
        latency_ms = latency_s * 1000
    else:
        latency_ms = 100.0

    # Add overhead for memory access and I/O
    overhead_ms = 2.0
    return round(latency_ms + overhead_ms, 1)


def _estimate_area(
    ip_blocks: list[dict[str, Any]], constraints: DesignConstraints
) -> float:
    """Estimate die area in mm²."""
    process_nm = constraints.target_process_nm or 28

    # Rough area estimates per block type at 28nm
    area_per_type = {
        "cpu": 4.0,
        "kpu": 8.0,
        "gpu": 15.0,
        "npu": 10.0,
        "tpu": 12.0,
        "memory": 3.0,
        "io": 2.0,
        "isp": 3.0,
        "accelerator": 10.0,
    }

    # Scale by process node (area scales roughly with node²)
    scale = (process_nm / 28) ** 2

    total = sum(area_per_type.get(b.get("type", ""), 5.0) for b in ip_blocks)
    total *= scale
    total *= 1.2  # 20% overhead for interconnect and padding

    return round(total, 1)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


def critic(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Review overall design quality and identify weaknesses.

    Acts as the "senior engineer" who challenges assumptions and
    identifies potential issues.

    Does not write to state — returns critique for planner/human review.
    """
    architecture = state.get("selected_architecture", {})
    ppa = state.get("ppa_metrics", {})
    constraints = get_constraints(state)
    workload = state.get("workload_profile", {})
    ip_blocks = state.get("ip_blocks", [])

    issues = []
    recommendations = []
    strengths = []

    # Check PPA verdicts
    verdicts = ppa.get("verdicts", {})
    failing = [k for k, v in verdicts.items() if v == "FAIL"]
    passing = [k for k, v in verdicts.items() if v == "PASS"]

    if failing:
        issues.append(f"Failing constraints: {', '.join(failing)}")
    if passing:
        strengths.append(f"Meeting constraints: {', '.join(passing)}")

    # Check for single point of failure
    compute_blocks = [b for b in ip_blocks if b.get("type") in ("kpu", "gpu", "npu", "tpu")]
    if len(compute_blocks) == 1:
        issues.append(
            "Single compute engine — no fallback if accelerator fails or is overloaded"
        )
        recommendations.append(
            "Consider CPU fallback path for critical inference workloads"
        )

    # Check power margin
    power_w = ppa.get("power_watts")
    max_power = constraints.max_power_watts
    if power_w and max_power:
        margin_pct = (1 - power_w / max_power) * 100
        if margin_pct < 10:
            issues.append(f"Tight power margin: only {margin_pct:.0f}% headroom")
            recommendations.append("Add power management modes (sleep/throttle)")
        elif margin_pct > 50:
            strengths.append(f"Generous power margin: {margin_pct:.0f}% headroom")

    # Check memory adequacy
    workload_mem = workload.get("total_estimated_memory_mb", workload.get("estimated_memory_mb", 0))
    available_mem = sum(
        b.get("config", {}).get("capacity_gb", 0) * 1024
        for b in ip_blocks
        if b.get("type") == "memory"
    )
    if workload_mem > 0 and available_mem > 0:
        if workload_mem > available_mem * 0.8:
            issues.append(
                f"Memory pressure: workload needs ~{workload_mem:.0f}MB, "
                f"available {available_mem:.0f}MB"
            )
            recommendations.append("Consider model compression or streaming execution")

    # Safety check
    if constraints.safety_critical and "redundancy" not in str(ip_blocks).lower():
        issues.append("Safety-critical application but no redundancy in architecture")
        recommendations.append("Add dual-lockstep CPU and ECC memory for safety compliance")

    # Overall assessment
    if not failing and not issues:
        assessment = "STRONG"
    elif not failing:
        assessment = "ADEQUATE"
    elif len(failing) <= 1:
        assessment = "NEEDS_WORK"
    else:
        assessment = "SIGNIFICANT_ISSUES"

    return {
        "summary": f"Design review: {assessment} ({len(issues)} issues, {len(strengths)} strengths)",
        "assessment": assessment,
        "issues": issues,
        "recommendations": recommendations,
        "strengths": strengths,
    }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


def report_generator(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Generate a design report summarizing all artifacts and decisions.

    Produces a structured report dict (no file I/O for now).
    """
    architecture = state.get("selected_architecture", {})
    ppa = state.get("ppa_metrics", {})
    workload = state.get("workload_profile", {})
    candidates = state.get("hardware_candidates", [])
    ip_blocks = state.get("ip_blocks", [])
    history = state.get("history", [])
    rationale = state.get("design_rationale", [])

    # Find critic results in dependencies
    dep_results = get_dependency_results(state, task)
    critique = {}
    for result in dep_results.values():
        if "assessment" in result:
            critique = result

    report = {
        "title": f"SoC Design Report: {state.get('goal', 'Unknown')}",
        "session_id": state.get("session_id", "unknown"),
        "use_case": state.get("use_case", ""),
        "platform": state.get("platform", ""),
        "sections": {
            "executive_summary": {
                "goal": state.get("goal", ""),
                "architecture": architecture.get("name", "N/A"),
                "primary_compute": architecture.get("primary_compute", "N/A"),
                "overall_verdict": ppa.get("verdicts", {}),
                "assessment": critique.get("assessment", "N/A"),
            },
            "workload_analysis": workload,
            "hardware_exploration": {
                "candidates_evaluated": len(candidates),
                "top_candidates": candidates[:3] if candidates else [],
            },
            "architecture": {
                "description": architecture,
                "ip_blocks": ip_blocks,
                "memory_map": state.get("memory_map", {}),
                "interconnect": state.get("interconnect", {}),
            },
            "ppa_assessment": ppa,
            "design_review": critique,
            "decision_history": history,
            "design_rationale": rationale,
        },
    }

    return {
        "summary": "Design report generated",
        "report": report,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_dispatcher() -> Dispatcher:
    """Create a Dispatcher pre-loaded with all specialist agents.

    Returns a Dispatcher with these agents registered:
    - workload_analyzer
    - hw_explorer
    - architecture_composer
    - ppa_assessor
    - critic
    - report_generator
    - design_optimizer
    - design_explorer (Pareto front)
    - safety_detector
    - experience_retriever
    """
    from embodied_ai_architect.graphs.optimizer import design_optimizer
    from embodied_ai_architect.graphs.kpu_specialists import (
        bandwidth_validator,
        floorplan_validator,
        kpu_configurator,
        kpu_optimizer,
    )
    from embodied_ai_architect.graphs.rtl_specialists import (
        rtl_generator,
        rtl_ppa_assessor,
    )
    from embodied_ai_architect.graphs.pareto import design_explorer
    from embodied_ai_architect.graphs.safety import safety_detector
    from embodied_ai_architect.graphs.experience_specialist import experience_retriever

    dispatcher = Dispatcher()
    dispatcher.register_many({
        "workload_analyzer": workload_analyzer,
        "hw_explorer": hw_explorer,
        "architecture_composer": architecture_composer,
        "ppa_assessor": ppa_assessor,
        "critic": critic,
        "report_generator": report_generator,
        "design_optimizer": design_optimizer,
        # KPU micro-architecture specialists
        "kpu_configurator": kpu_configurator,
        "floorplan_validator": floorplan_validator,
        "bandwidth_validator": bandwidth_validator,
        "kpu_optimizer": kpu_optimizer,
        # RTL specialists
        "rtl_generator": rtl_generator,
        "rtl_ppa_assessor": rtl_ppa_assessor,
        # Phase 4 specialists
        "design_explorer": design_explorer,
        "safety_detector": safety_detector,
        "experience_retriever": experience_retriever,
    })
    return dispatcher
