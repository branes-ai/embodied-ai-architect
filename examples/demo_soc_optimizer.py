#!/usr/bin/env python3
"""Demo 3: SoC Design Optimizer — Iterative Power Convergence.

Demonstrates the optimization loop: the initial KPU design exceeds the 5W
power budget (6.3W), and the optimizer iteratively applies strategies
(INT8 quantization, resolution reduction) until the design passes.

Expected output:
    Iteration 0: KPU selected, power ~6.3W → FAIL
    Iteration 1: Apply optimization → power reduced → re-evaluate
    Iteration 2: Apply optimization → power within budget → PASS
    Result: Converged in 2 optimization iterations.

Usage:
    python examples/demo_soc_optimizer.py
    python examples/demo_soc_optimizer.py --max-iterations 5
    python examples/demo_soc_optimizer.py --power 4.0   # tighter budget
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check langgraph availability before importing
try:
    import langgraph  # noqa: F401
except ImportError:
    print("ERROR: langgraph is required for this demo.")
    print("Install it with: pip install -e '.[langgraph]'")
    sys.exit(1)

from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.specialists import create_default_dispatcher
from embodied_ai_architect.graphs.governance import GovernanceGuard, GovernancePolicy
from embodied_ai_architect.graphs.soc_graph import build_soc_design_graph

# ---------------------------------------------------------------------------
# Static plan (same as Demo 1)
# ---------------------------------------------------------------------------

DEMO_PLAN = [
    {
        "id": "t1",
        "name": "Analyze perception workload (detection + tracking at 30fps)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Enumerate feasible hardware under power/cost constraints",
        "agent": "hw_explorer",
        "dependencies": ["t1"],
    },
    {
        "id": "t3",
        "name": "Compose SoC architecture with selected compute engine",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Assess PPA metrics against drone constraints",
        "agent": "ppa_assessor",
        "dependencies": ["t3"],
    },
    {
        "id": "t5",
        "name": "Review design for risks and improvement opportunities",
        "agent": "critic",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Generate design report with trade-off analysis",
        "agent": "report_generator",
        "dependencies": ["t5"],
    },
]

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

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


def verdict_str(v: str) -> str:
    return "** FAIL **" if v == "FAIL" else v


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def run_demo(max_power: float, max_latency: float, max_cost: float, max_iterations: int) -> None:
    banner("SoC Design Optimizer — Demo 3")
    goal_text = (
        f"Design an SoC for a last-mile delivery drone that must: "
        f"run a visual perception pipeline with camera-based object detection "
        f"and multi-object tracking at 30fps through a tightly power-constrained "
        f"compute subsystem, consume less than {max_power} watts with no thermal "
        f"throttling headroom, cost less than ${max_cost} in BOM at volume. "
        f"The camera ISP must feed the detection pipeline with minimal buffering latency."
    )
    print(f"\n  Goal:")
    print(textwrap.fill(goal_text, width=W - 4, initial_indent="    ", subsequent_indent="    "))
    print(f"  Power budget: {max_power}W | Latency: {max_latency}ms | Cost: ${max_cost}")
    print(f"  Max optimization iterations: {max_iterations}")

    constraints = DesignConstraints(
        max_power_watts=max_power,
        max_latency_ms=max_latency,
        max_cost_usd=max_cost,
        max_area_mm2=100.0,
        target_volume=100_000,
    )

    state = create_initial_soc_state(
        goal=goal_text,
        constraints=constraints,
        use_case="delivery_drone",
        platform="drone",
        max_iterations=max_iterations,
    )

    # Build the graph
    planner = PlannerNode(static_plan=DEMO_PLAN)
    dispatcher = create_default_dispatcher()
    governance = GovernanceGuard(GovernancePolicy(iteration_limit=max_iterations))
    graph = build_soc_design_graph(
        dispatcher=dispatcher,
        planner=planner,
        governance=governance,
    )

    # Run the optimization loop
    section("Running Optimization Loop")
    t0 = time.perf_counter()

    result = graph.invoke(state, config={"recursion_limit": max_iterations * 5 + 20})

    elapsed = time.perf_counter() - t0

    # Print results
    section("Optimization History")
    opt_history = result.get("optimization_history", [])
    for entry in opt_history:
        it = entry["iteration"]
        snap = entry["ppa_snapshot"]
        verdicts = entry["verdicts"]
        power = snap.get("power_watts")
        power_str = f"{power:.1f}W" if power is not None else "N/A"
        verdict_list = " ".join(f"{k}:{verdict_str(v)}" for k, v in verdicts.items())
        print(f"  Iteration {it}: power={power_str}  {verdict_list}")

    section("Final PPA")
    ppa = result.get("ppa_metrics", {})
    kv("Process Node", f"{ppa.get('process_nm', 28)}nm")
    if ppa.get("power_watts") is not None:
        kv("Power", f"{ppa['power_watts']:.2f} W")
    if ppa.get("latency_ms") is not None:
        kv("Latency", f"{ppa['latency_ms']:.1f} ms")
    if ppa.get("area_mm2") is not None:
        kv("Area", f"{ppa['area_mm2']:.1f} mm2")
    if ppa.get("cost_usd") is not None:
        kv("Cost", f"${ppa['cost_usd']:.0f}")

    cost_bd = ppa.get("cost_breakdown")
    if cost_bd:
        print()
        kv("Die Cost", f"${cost_bd.get('die_cost_usd', 0):.2f}")
        kv("Package", f"${cost_bd.get('package_cost_usd', 0):.2f}")
        kv("Test", f"${cost_bd.get('test_cost_usd', 0):.2f}")
        kv("NRE/unit", f"${cost_bd.get('nre_per_unit_usd', 0):.2f}")
        kv("Yield", f"{cost_bd.get('yield_percent', 0):.1f}%")
        kv("Dies/wafer", str(cost_bd.get('dies_per_wafer', 0)))

    verdicts = ppa.get("verdicts", {})
    if verdicts:
        print()
        for k, v in verdicts.items():
            kv(f"  {k}", verdict_str(v))

    section("Design Decisions")
    for entry in result.get("design_rationale", []):
        print(f"    {entry}")

    # --- Outcome analysis ---
    all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False
    iterations = result.get("iteration", 0)

    if all_pass:
        banner("Design Converged")
        print(f"\n  All constraints PASS after {iterations} optimization iteration"
              f"{'s' if iterations != 1 else ''}.")
    else:
        failing = [k for k, v in verdicts.items() if v == "FAIL"]
        banner("Design Did Not Converge")
        print(f"\n  The optimizer exhausted {iterations} iterations without satisfying"
              f" all constraints.")
        print(f"  Failing: {', '.join(failing)}")

        # Per-constraint failure analysis
        constraints_d = result.get("constraints", {})
        print()
        for fc in failing:
            _print_failure_analysis(fc, ppa, constraints_d)

        # Actionable next steps
        section("Suggested Next Steps")
        suggestions = _suggest_next_steps(failing, ppa, constraints_d)
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")

    print(f"\n  Optimization iterations: {iterations}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Status: {result.get('status')}")
    print()


def _print_failure_analysis(constraint: str, ppa: dict, constraints: dict) -> None:
    """Print why a specific constraint cannot be met."""
    if constraint == "cost":
        actual = ppa.get("cost_usd", 0)
        target = constraints.get("max_cost_usd", 0)
        bd = ppa.get("cost_breakdown", {})
        nre = bd.get("nre_per_unit_usd", 0)
        die = bd.get("die_cost_usd", 0)
        volume = bd.get("volume", 0)
        node = ppa.get("process_nm", 28)
        print(f"  cost: ${actual:.0f} vs ${target:.0f} target")
        if nre > 0 and actual > 0:
            nre_pct = nre / actual * 100
            print(f"    NRE dominates at {nre_pct:.0f}% of unit cost "
                  f"(${nre:.0f}/unit at {volume:,} volume)")
            print(f"    Die cost is only ${die:.2f} — variable cost is not the problem")
            if target > 0:
                # Compute break-even volume: total_nre / (target - variable_costs)
                variable = actual - nre
                headroom = target - variable
                if headroom > 0:
                    total_nre = nre * volume
                    breakeven = int(total_nre / headroom)
                    print(f"    Break-even volume at ${target:.0f} target: "
                          f"~{breakeven:,} units (at {node}nm)")
    elif constraint == "power":
        actual = ppa.get("power_watts")
        target = constraints.get("max_power_watts")
        if actual and target:
            overshoot = (actual / target - 1) * 100
            print(f"  power: {actual:.1f}W vs {target:.1f}W target (+{overshoot:.0f}% over)")
            print(f"    All workload/clock strategies exhausted")
    elif constraint == "area":
        actual = ppa.get("area_mm2")
        target = constraints.get("max_area_mm2")
        if actual and target:
            print(f"  area: {actual:.1f}mm² vs {target:.1f}mm² target")
            node = ppa.get("process_nm", 28)
            print(f"    At {node}nm — shrinking process node would reduce area")
    elif constraint == "latency":
        actual = ppa.get("latency_ms")
        target = constraints.get("max_latency_ms")
        if actual and target:
            print(f"  latency: {actual:.1f}ms vs {target:.1f}ms target")


def _suggest_next_steps(
    failing: list[str], ppa: dict, constraints: dict
) -> list[str]:
    """Generate actionable suggestions based on which constraints fail."""
    suggestions: list[str] = []
    bd = ppa.get("cost_breakdown", {})

    if "cost" in failing:
        nre = bd.get("nre_per_unit_usd", 0)
        actual = ppa.get("cost_usd", 0)
        if nre > 0 and actual > 0 and nre / actual > 0.5:
            volume = bd.get("volume", 0)
            suggestions.append(
                f"Increase production volume (currently {volume:,}) — "
                f"NRE amortization dominates unit cost"
            )
        target = constraints.get("max_cost_usd", 0)
        node = ppa.get("process_nm", 28)
        if node <= 40:
            suggestions.append(
                f"Consider a more mature process node (currently {node}nm) — "
                f"65nm or 90nm have much lower NRE"
            )
        suggestions.append(
            "Relax cost target or use an existing COTS accelerator "
            "instead of custom silicon"
        )

    if "power" in failing:
        suggestions.append(
            "Consider a more aggressive process node (lower Vdd) "
            "or hardware with lower TDP"
        )
        suggestions.append(
            "Re-evaluate workload: can any pipeline stages run at lower frame rate?"
        )

    if "area" in failing:
        suggestions.append(
            "Shrink process node to reduce die area, or remove IP blocks"
        )

    if not suggestions:
        suggestions.append("Review constraint targets — they may be mutually infeasible")

    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SoC Design Optimizer — Demo 3: Power Convergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--power", type=float, default=5.0, help="Max power budget in watts")
    parser.add_argument("--latency", type=float, default=33.3, help="Max latency in ms")
    parser.add_argument("--cost", type=float, default=30.0, help="Max cost in USD")
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Max optimization iterations"
    )
    args = parser.parse_args()

    run_demo(
        max_power=args.power,
        max_latency=args.latency,
        max_cost=args.cost,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
