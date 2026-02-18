#!/usr/bin/env python3
"""Demo 2: Design Space Exploration — 3-way Hardware Comparison (Pareto).

Warehouse AMR with MobileNetV2 + SLAM workload. Runs the pipeline with
the Pareto design explorer to compare hardware across power/latency/cost,
identifies the knee point on the Pareto front.

Usage:
    python examples/demo_dse_pareto.py
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

# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------

DEMO_2_GOAL = (
    "Design an SoC for a warehouse AMR (autonomous mobile robot) that must: "
    "run MobileNetV2 object detection for pallet and obstacle recognition, "
    "concurrent visual SLAM for localization and mapping in GPS-denied indoor aisles, "
    "schedule both workloads with deterministic latency guarantees, "
    "consume less than 15 watts, cost less than $100 at 10K volume, "
    "and achieve end-to-end latency under 50ms."
)

DEMO_2_CONSTRAINTS = DesignConstraints(
    max_power_watts=15.0,
    max_latency_ms=50.0,
    max_cost_usd=100.0,
    target_volume=10_000,
)

DEMO_2_PLAN = [
    {
        "id": "t1",
        "name": "Analyze AMR workload (detection + SLAM concurrent)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Enumerate hardware candidates under 15W/$100",
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
        "name": "Compose SoC architecture using Pareto knee point",
        "agent": "architecture_composer",
        "dependencies": ["t3"],
    },
    {
        "id": "t5",
        "name": "Assess PPA metrics against AMR constraints",
        "agent": "ppa_assessor",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Review design quality and trade-offs",
        "agent": "critic",
        "dependencies": ["t5"],
    },
    {
        "id": "t7",
        "name": "Generate design report with Pareto analysis",
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
    """Run Demo 2 and return final state."""
    banner("Agentic SoC Designer — Demo 2: DSE Pareto (Warehouse AMR)")
    print(f"\n  Goal:")
    print(textwrap.fill(DEMO_2_GOAL, width=W - 4, initial_indent="    ", subsequent_indent="    "))

    state = create_initial_soc_state(
        goal=DEMO_2_GOAL,
        constraints=DEMO_2_CONSTRAINTS,
        use_case="warehouse_amr",
        platform="amr",
    )

    # Plan
    section("Planning")
    planner = PlannerNode(static_plan=DEMO_2_PLAN)
    plan_updates = planner(state)
    state = {**state, **plan_updates}

    graph = get_task_graph(state)
    for tid in graph.execution_order():
        task = graph.nodes[tid]
        print(f"    [{tid}] {task.agent}: {task.name}")

    # Dispatch
    section("Executing Pipeline")
    dispatcher = create_default_dispatcher()
    t0 = time.perf_counter()

    state = dispatcher.run(state)
    exec_time = time.perf_counter() - t0
    print(f"\n  Pipeline completed in {exec_time:.2f}s")

    # Pareto results
    pareto = state.get("pareto_results", {})
    if pareto:
        section("Pareto Front Analysis")
        front = pareto.get("front", [])
        kv("Total design points", str(pareto.get("total", 0)))
        kv("Non-dominated", str(pareto.get("non_dominated_count", 0)))

        print(f"\n  {'Name':<25} {'Power':>7} {'Latency':>9} {'Cost':>7} {'Dom?':>5} {'Knee':>5}")
        print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*7} {'-'*5} {'-'*5}")
        for p in front:
            dom = "yes" if p.get("dominated") else "no"
            knee = "*" if p.get("knee_point") else ""
            print(
                f"  {p.get('hardware_name', '?'):<25} "
                f"{p.get('power', 0):>6.1f}W {p.get('latency', 0):>8.1f}ms "
                f"${p.get('cost', 0):>5.0f} {dom:>5} {knee:>5}"
            )

        knee_point = pareto.get("knee_point")
        if knee_point:
            print(f"\n  Knee point: {knee_point.get('hardware_name', 'N/A')}")

    # PPA
    ppa = state.get("ppa_metrics", {})
    if ppa:
        section("PPA Assessment")
        for metric, unit in [("power_watts", "W"), ("latency_ms", "ms"),
                             ("area_mm2", "mm2"), ("cost_usd", "$")]:
            val = ppa.get(metric)
            if val is not None:
                kv(metric.replace("_", " ").title(), f"{val:.1f} {unit}")

        verdicts = ppa.get("verdicts", {})
        if verdicts:
            print()
            for k, v in verdicts.items():
                status = "PASS" if v == "PASS" else "** FAIL **"
                print(f"    {k:<15} {status}")

    banner("Demo 2 Complete")
    print(f"  Status: {state.get('status')}")
    print()
    return state


if __name__ == "__main__":
    run_demo()
