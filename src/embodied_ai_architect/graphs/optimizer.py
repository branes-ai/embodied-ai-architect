"""Design optimizer specialist for iterative SoC refinement.

Uses a deterministic strategy catalog (no LLM) to fix failing PPA constraints.
Each strategy has applicability conditions and estimated reduction factors.
The optimizer reads failing verdicts from ppa_metrics, filters out already-tried
strategies from working memory, selects the best applicable one, and applies it
by modifying workload_profile/architecture/ip_blocks via _state_updates.

Usage:
    from embodied_ai_architect.graphs.optimizer import design_optimizer

    result = design_optimizer(task, state)
    # result["_state_updates"] contains modified workload_profile, etc.
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.memory import WorkingMemoryStore
from embodied_ai_architect.graphs.soc_state import (
    SoCDesignState,
    get_constraints,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimization strategy catalog
# ---------------------------------------------------------------------------

OPTIMIZATION_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "quantize_int8",
        "description": "Quantize model weights and activations from FP16/FP32 to INT8",
        "applicable_when": ["power"],
        "power_reduction_factor": 0.20,  # ~20% power reduction
        "latency_reduction_factor": 0.15,  # ~15% latency reduction
        "accuracy_impact": "minor",
        "applies_to": "workload_profile",
    },
    {
        "name": "reduce_resolution",
        "description": "Reduce input resolution (e.g. 640x640 -> 480x480)",
        "applicable_when": ["power", "latency"],
        "power_reduction_factor": 0.25,  # ~25% power reduction (quadratic in resolution)
        "latency_reduction_factor": 0.30,  # ~30% latency reduction
        "accuracy_impact": "moderate",
        "applies_to": "workload_profile",
    },
    {
        "name": "clock_scaling",
        "description": "Reduce accelerator clock frequency for power savings",
        "applicable_when": ["power"],
        "power_reduction_factor": 0.15,  # ~15% power reduction
        "latency_reduction_factor": -0.10,  # latency increases ~10%
        "accuracy_impact": "none",
        "applies_to": "ip_blocks",
    },
    {
        "name": "model_pruning",
        "description": "Structured pruning to reduce model size by ~30%",
        "applicable_when": ["power", "latency"],
        "power_reduction_factor": 0.18,  # ~18% power reduction
        "latency_reduction_factor": 0.20,  # ~20% latency reduction
        "accuracy_impact": "moderate",
        "applies_to": "workload_profile",
    },
    {
        "name": "smaller_model",
        "description": "Switch to a smaller model variant (e.g. YOLOv8n -> YOLOv8p)",
        "applicable_when": ["power", "latency", "cost"],
        "power_reduction_factor": 0.35,  # ~35% power reduction
        "latency_reduction_factor": 0.40,  # ~40% latency reduction
        "accuracy_impact": "significant",
        "applies_to": "workload_profile",
    },
]


def design_optimizer(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Apply an optimization strategy to fix failing PPA constraints.

    Reads failing verdicts from ppa_metrics, filters applicable strategies
    excluding already-tried ones (from working memory), selects best,
    and applies by modifying state artifacts via _state_updates.

    Args:
        task: Current task node.
        state: Current SoC design state with ppa_metrics containing verdicts.

    Returns:
        Result dict with _state_updates for modified artifacts.
    """
    ppa = state.get("ppa_metrics", {})
    verdicts = ppa.get("verdicts", {})
    failing = [k for k, v in verdicts.items() if v == "FAIL"]

    if not failing:
        return {
            "summary": "No failing constraints — no optimization needed",
            "strategy": None,
            "applied": False,
        }

    # Load working memory to see what was already tried
    wm_data = state.get("working_memory", {})
    store = WorkingMemoryStore(**wm_data) if wm_data else WorkingMemoryStore()
    already_tried = store.get_tried_descriptions("design_optimizer")

    # Filter applicable strategies
    applicable = []
    for strat in OPTIMIZATION_STRATEGIES:
        if strat["name"] in already_tried:
            continue
        if any(f in strat["applicable_when"] for f in failing):
            applicable.append(strat)

    if not applicable:
        return {
            "summary": f"No untried strategies for failing constraints: {failing}",
            "strategy": None,
            "applied": False,
            "failing_constraints": failing,
            "already_tried": already_tried,
        }

    # Select best strategy: prefer highest power reduction for power failures
    if "power" in failing:
        applicable.sort(key=lambda s: s["power_reduction_factor"], reverse=True)
    elif "latency" in failing:
        applicable.sort(key=lambda s: s["latency_reduction_factor"], reverse=True)
    else:
        applicable.sort(key=lambda s: s["power_reduction_factor"], reverse=True)

    selected = applicable[0]
    logger.info(
        "Optimizer selected strategy '%s' for failing constraints %s",
        selected["name"],
        failing,
    )

    # Apply strategy
    state_updates = _apply_strategy(selected, state)

    # Record attempt in working memory
    iteration = state.get("iteration", 0)
    store.record_attempt(
        agent_name="design_optimizer",
        description=selected["name"],
        outcome=f"Applied {selected['description']} at iteration {iteration}",
        iteration=iteration,
    )
    store.record_decision("design_optimizer", f"Applied {selected['name']}: {selected['description']}")
    state_updates["working_memory"] = store.model_dump()

    return {
        "summary": f"Applied optimization: {selected['description']}",
        "strategy": selected["name"],
        "applied": True,
        "failing_constraints": failing,
        "power_reduction_factor": selected["power_reduction_factor"],
        "latency_reduction_factor": selected["latency_reduction_factor"],
        "_state_updates": state_updates,
    }


def _apply_strategy(strategy: dict[str, Any], state: SoCDesignState) -> dict[str, Any]:
    """Apply a strategy by modifying copies of state artifacts.

    Returns a dict of state keys to update.
    """
    updates: dict[str, Any] = {}
    ppa = dict(state.get("ppa_metrics", {}))

    power_factor = strategy["power_reduction_factor"]
    latency_factor = strategy["latency_reduction_factor"]

    if strategy["applies_to"] == "workload_profile":
        workload = dict(state.get("workload_profile", {}))

        # Reduce estimated compute requirements
        if "total_estimated_gflops" in workload:
            workload["total_estimated_gflops"] = round(
                workload["total_estimated_gflops"] * (1 - power_factor), 2
            )
        if "estimated_gflops" in workload:
            workload["estimated_gflops"] = round(
                workload["estimated_gflops"] * (1 - power_factor), 2
            )

        # Scale sub-workloads
        for w in workload.get("workloads", []):
            if "estimated_gflops" in w:
                w["estimated_gflops"] = round(
                    w["estimated_gflops"] * (1 - power_factor), 2
                )

        # Record optimization applied
        optimizations = workload.get("optimizations_applied", [])
        optimizations.append(strategy["name"])
        workload["optimizations_applied"] = optimizations

        updates["workload_profile"] = workload

    elif strategy["applies_to"] == "ip_blocks":
        ip_blocks = [dict(b) for b in state.get("ip_blocks", [])]

        for block in ip_blocks:
            if block.get("type") in ("kpu", "gpu", "npu", "tpu", "accelerator"):
                config = dict(block.get("config", {}))
                if "frequency_mhz" in config:
                    config["frequency_mhz"] = int(config["frequency_mhz"] * (1 - power_factor))
                block["config"] = config

        updates["ip_blocks"] = ip_blocks

    # Adjust PPA estimates based on reduction factors
    if ppa.get("power_watts") is not None:
        ppa["power_watts"] = round(ppa["power_watts"] * (1 - power_factor), 2)
    if ppa.get("latency_ms") is not None:
        ppa["latency_ms"] = round(ppa["latency_ms"] * (1 - latency_factor), 2)

    # Clear verdicts — they'll be recomputed by ppa_assessor
    ppa["verdicts"] = {}
    ppa["bottlenecks"] = []
    ppa["suggestions"] = []
    updates["ppa_metrics"] = ppa

    return updates
