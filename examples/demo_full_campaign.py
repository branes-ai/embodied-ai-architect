#!/usr/bin/env python3
"""Demo 7: Full Campaign — Quadruped Robot with 4 Workloads.

Quadruped robot with Visual SLAM, object detection, LiDAR processing,
and voice recognition. 15W envelope, $50 BOM, 10K volume.
Exercises the full pipeline including multi-workload analysis and DSE.

Usage:
    python examples/demo_full_campaign.py
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
    get_task_graph,
)
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.specialists import create_default_dispatcher
from embodied_ai_architect.graphs.governance import GovernancePolicy

# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------

DEMO_7_GOAL = (
    "Design an SoC for a quadruped robot that must concurrently run: "
    "visual SLAM for real-time localization and navigation mapping, "
    "YOLO object detection for obstacle avoidance and terrain classification, "
    "LiDAR point cloud processing for 3D terrain surface reconstruction, "
    "and voice recognition for natural-language command input. "
    "All four workloads must be scheduled concurrently with deterministic priorities. "
    "Power budget: 15W, BOM cost: $50 at 10K volume, "
    "perception pipeline latency: <50ms end-to-end."
)

DEMO_7_CONSTRAINTS = DesignConstraints(
    max_power_watts=15.0,
    max_latency_ms=50.0,
    max_cost_usd=50.0,
    max_area_mm2=200.0,
    target_volume=10_000,
)

DEMO_7_PLAN = [
    {
        "id": "t1",
        "name": "Analyze multi-workload (SLAM + detection + LiDAR + voice)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Enumerate hardware candidates under 15W/$50",
        "agent": "hw_explorer",
        "dependencies": ["t1"],
    },
    {
        "id": "t3",
        "name": "Explore design space via Pareto front analysis",
        "agent": "design_explorer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Compose SoC architecture for multi-workload execution",
        "agent": "architecture_composer",
        "dependencies": ["t3"],
    },
    {
        "id": "t5",
        "name": "Assess PPA metrics against quadruped constraints",
        "agent": "ppa_assessor",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Review design for multi-workload adequacy and risks",
        "agent": "critic",
        "dependencies": ["t5"],
    },
    {
        "id": "t7",
        "name": "Generate comprehensive design report",
        "agent": "report_generator",
        "dependencies": ["t6"],
    },
]

W = 72


def banner(text: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {text}")
    print(f"{'=' * W}")


def section(text: str) -> None:
    print(f"\n--- {text} {'-' * max(0, W - len(text) - 5)}")


def kv(key: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:.<30s} {value}")


def run_demo() -> dict:
    """Run Demo 7 and return final state."""
    banner("Agentic SoC Designer — Demo 7: Full Campaign (Quadruped Robot)")
    print(f"\n  Goal:")
    print(textwrap.fill(DEMO_7_GOAL, width=W - 4, initial_indent="    ", subsequent_indent="    "))

    governance = GovernancePolicy(
        iteration_limit=10,
        cost_budget_tokens=0,  # unlimited for demo
    )

    state = create_initial_soc_state(
        goal=DEMO_7_GOAL,
        constraints=DEMO_7_CONSTRAINTS,
        use_case="quadruped_robot",
        platform="quadruped",
        max_iterations=10,
        governance=governance.model_dump(),
    )

    # Plan
    section("Planning")
    planner = PlannerNode(static_plan=DEMO_7_PLAN)
    plan_updates = planner(state)
    state = {**state, **plan_updates}

    graph = get_task_graph(state)
    print(f"  Task graph ({len(graph.nodes)} tasks):\n")
    for tid in graph.execution_order():
        task = graph.nodes[tid]
        deps = ", ".join(task.dependencies) if task.dependencies else "(none)"
        print(f"    [{tid}] {task.agent}: {task.name}")
        print(f"         deps: {deps}")

    # Dispatch
    section("Executing Pipeline")
    dispatcher = create_default_dispatcher()
    t0 = time.perf_counter()

    step = 0
    completed_ids: set[str] = set()
    while True:
        graph = get_task_graph(state)
        if graph.is_complete:
            state = {**state, "status": "complete"}
            break

        ready = graph.ready_tasks()
        if not ready:
            print(f"\n  Pipeline stuck! Status: {state.get('status')}")
            break

        for task in ready:
            print(f"\n  [{task.id}] {task.agent}: {task.name}")

        state = dispatcher.step(state)
        step += 1

        graph = get_task_graph(state)
        for task in graph.nodes.values():
            if task.id not in completed_ids and task.status.value == "completed":
                completed_ids.add(task.id)
                summary = (task.result or {}).get("summary", "")
                if summary:
                    print(f"         -> {summary}")

    exec_time = time.perf_counter() - t0
    print(f"\n  Pipeline completed in {step} steps, {exec_time:.2f}s")

    # Workload analysis
    wp = state.get("workload_profile", {})
    if wp:
        section("Multi-Workload Analysis")
        kv("Total GFLOPS", f"{wp.get('total_estimated_gflops', '?')}")
        kv("Total Memory", f"{wp.get('total_estimated_memory_mb', '?')} MB")
        kv("Workload Count", str(wp.get("workload_count", 0)))
        kv("Dominant Op", wp.get("dominant_op", "?"))

        workloads = wp.get("workloads", [])
        if workloads:
            print(f"\n  {'Workload':<25} {'Model':<15} {'GFLOPS':>8} {'Sched':<12}")
            print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*12}")
            for w in workloads:
                print(
                    f"  {w['name']:<25} {w.get('model_class', '?'):<15} "
                    f"{w.get('estimated_gflops', 0):>8.1f} {w.get('scheduling', '?'):<12}"
                )

    # Pareto results
    pareto = state.get("pareto_results", {})
    if pareto:
        section("Pareto Front Analysis")
        kv("Design points", str(pareto.get("total", 0)))
        kv("Non-dominated", str(pareto.get("non_dominated_count", 0)))

        knee = pareto.get("knee_point")
        if knee:
            kv("Knee point", knee.get("hardware_name", "N/A"))
            kv("  Power", f"{knee.get('power', 0):.1f}W")
            kv("  Latency", f"{knee.get('latency', 0):.1f}ms")
            kv("  Cost", f"${knee.get('cost', 0):.0f}")

    # Architecture
    arch = state.get("selected_architecture", {})
    if arch:
        section("Selected Architecture")
        kv("Name", arch.get("name", "N/A"))
        kv("Primary Compute", arch.get("primary_compute", "N/A"))
        kv("Paradigm", arch.get("compute_paradigm", "N/A"))

    # PPA
    ppa = state.get("ppa_metrics", {})
    if ppa:
        section("PPA Assessment")
        kv("Process Node", f"{ppa.get('process_nm', 28)}nm")
        for metric, unit in [("power_watts", "W"), ("latency_ms", "ms"),
                             ("area_mm2", "mm2"), ("cost_usd", "$")]:
            val = ppa.get(metric)
            if val is not None:
                kv(metric.replace("_", " ").title(), f"{val:.1f} {unit}")

        verdicts = ppa.get("verdicts", {})
        if verdicts:
            print()
            all_pass = True
            for k, v in verdicts.items():
                status = "PASS" if v == "PASS" else "** FAIL **"
                if v != "PASS":
                    all_pass = False
                print(f"    {k:<15} {status}")

            kv("\nOverall", "ALL PASS" if all_pass else "HAS FAILURES")

    # Decision trail
    history = state.get("history", [])
    if history:
        section("Decision Trail")
        for i, d in enumerate(history, 1):
            agent = d.get("agent", "?")
            action = d.get("action", "?")
            print(f"    {i}. [{agent}] {action}")

    banner("Demo 7 Complete")
    total = exec_time
    print(f"  Total time: {total:.2f}s  |  Status: {state.get('status')}")
    print(f"  Decisions recorded: {len(history)}")
    print()
    return state


if __name__ == "__main__":
    run_demo()
