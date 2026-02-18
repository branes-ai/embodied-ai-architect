#!/usr/bin/env python3
"""Demo 5: HITL Safety — Surgical Robot with IEC 62304.

Safety-critical surgical robot requiring 1ms force-feedback latency and
IEC 62304 compliance. Demonstrates safety detection, redundancy injection,
and governance approval gates.

Usage:
    python examples/demo_hitl_safety.py
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

DEMO_5_GOAL = (
    "Design an SoC for a surgical robot arm controller that must: "
    "process force-feedback sensor data with sub-millisecond latency for haptic control, "
    "run visual guidance with object detection for surgical instrument tracking, "
    "support voice and speech recognition for hands-free surgeon commands, "
    "comply with IEC 62304 Class C safety requirements including "
    "dual-redundant lockstep execution and hardware watchdog timers, "
    "consume less than 25 watts, and cost less than $500."
)

DEMO_5_CONSTRAINTS = DesignConstraints(
    max_power_watts=25.0,
    max_latency_ms=1.0,
    max_cost_usd=500.0,
    max_area_mm2=300.0,
    safety_critical=True,
    safety_standard="IEC 62304 Class C",
)

DEMO_5_PLAN = [
    {
        "id": "t1",
        "name": "Detect safety-critical requirements (IEC 62304)",
        "agent": "safety_detector",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Analyze surgical workload (force-feedback + visual guidance)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t3",
        "name": "Enumerate hardware candidates under 25W/$500",
        "agent": "hw_explorer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Compose SoC architecture with safety redundancy",
        "agent": "architecture_composer",
        "dependencies": ["t1", "t3"],
    },
    {
        "id": "t5",
        "name": "Assess PPA metrics against surgical constraints",
        "agent": "ppa_assessor",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Review design for safety compliance and risks",
        "agent": "critic",
        "dependencies": ["t5"],
    },
    {
        "id": "t7",
        "name": "Generate safety-aware design report",
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
    """Run Demo 5 and return final state."""
    banner("Agentic SoC Designer — Demo 5: HITL Safety (Surgical Robot)")
    print(f"\n  Goal:")
    print(textwrap.fill(DEMO_5_GOAL, width=W - 4, initial_indent="    ", subsequent_indent="    "))

    # Set up governance with safety approval requirements
    governance = GovernancePolicy(
        approval_required_actions=["change_safety_architecture", "override_safety_constraint"],
        iteration_limit=5,
        require_human_approval_on_fail=True,
        fail_iteration_threshold=2,
    )

    state = create_initial_soc_state(
        goal=DEMO_5_GOAL,
        constraints=DEMO_5_CONSTRAINTS,
        use_case="surgical_robot",
        platform="medical",
        governance=governance.model_dump(),
    )

    # Plan
    section("Planning")
    planner = PlannerNode(static_plan=DEMO_5_PLAN)
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

    # Safety analysis
    safety = state.get("safety_analysis", {})
    if safety:
        section("Safety Analysis")
        kv("Safety Critical", str(safety.get("is_safety_critical", False)))
        kv("Standard", safety.get("detected_standard", "N/A"))
        reasons = safety.get("reasons", [])
        if reasons:
            print("\n  Detection reasons:")
            for r in reasons:
                print(f"    - {r}")

        reqs = safety.get("redundancy_requirements", [])
        if reqs:
            print("\n  Redundancy requirements:")
            for r in reqs:
                print(f"    - {r.get('description', r.get('type', '?'))}")

    # Governance
    gov = state.get("governance", {})
    if gov:
        section("Governance Policy (Safety-Enhanced)")
        approval_actions = gov.get("approval_required_actions", [])
        if approval_actions:
            print("  Actions requiring approval:")
            for a in approval_actions:
                print(f"    - {a}")

        safety_actions = gov.get("safety_critical_actions", [])
        if safety_actions:
            print("  Safety-critical actions:")
            for a in safety_actions:
                print(f"    - {a}")

    # Audit log
    audit = state.get("audit_log", [])
    if audit:
        section("Audit Log")
        for entry in audit:
            print(f"    [{entry.get('agent', '?')}] {entry.get('action', '?')}")

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
            for k, v in verdicts.items():
                status = "PASS" if v == "PASS" else "** FAIL **"
                print(f"    {k:<15} {status}")

    banner("Demo 5 Complete")
    print(f"  Status: {state.get('status')}")
    print()
    return state


if __name__ == "__main__":
    run_demo()
