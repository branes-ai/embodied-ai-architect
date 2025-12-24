"""
Critic Node - Analyze PPA metrics and make routing decisions.

This node is the "decision maker" that determines:
- Did we meet constraints?
- Should we iterate?
- Have we converged?
"""

import sys
from pathlib import Path
from typing import Callable

# Handle both package and script imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from ..state import SoCDesignState, ActionType, OptimizationStep
except ImportError:
    from state import SoCDesignState, ActionType, OptimizationStep


def create_critic_node() -> Callable[[SoCDesignState], dict]:
    """
    Create the critic node.

    The critic node:
    1. Compares current metrics against constraints
    2. Compares against baseline for improvement tracking
    3. Decides whether to iterate or sign off
    """

    def critic_node(state: SoCDesignState) -> dict:
        metrics = state.get("current_metrics", {})
        constraints = state.get("constraints", {})
        baseline = state.get("baseline_metrics", {})
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]

        # Step record
        step = OptimizationStep(
            iteration=iteration,
            agent="critic",
            action="analyze_ppa",
            metrics_before=baseline,
            metrics_after=metrics
        )

        # Check iteration limit
        if iteration >= max_iterations:
            step.success = False
            step.error = "Max iterations reached"
            step.reasoning = f"Stopping after {iteration} iterations"

            return {
                "next_action": ActionType.ABORT.value,
                "last_error": f"Max iterations ({max_iterations}) reached without meeting constraints",
                "history": state["history"] + [step.to_dict()]
            }

        # Analyze metrics vs constraints
        issues = []
        improvements = []

        # Check area constraint
        max_area = constraints.get("max_area_cells")
        current_area = metrics.get("area_cells", 0)
        baseline_area = baseline.get("area_cells", 0)

        if max_area and current_area > max_area:
            issues.append(f"Area {current_area} exceeds max {max_area}")

        if baseline_area > 0 and current_area > 0:
            area_change = (baseline_area - current_area) / baseline_area * 100
            if area_change > 0:
                improvements.append(f"Area reduced by {area_change:.1f}%")
            elif area_change < -5:  # More than 5% worse
                issues.append(f"Area increased by {-area_change:.1f}%")

        # Check timing constraint (if available - requires more advanced synthesis)
        target_clock = constraints.get("target_clock_ns")
        wns = metrics.get("wns_ps")
        if target_clock and wns is not None and wns < 0:
            issues.append(f"Timing violation: WNS = {wns}ps")

        # Decision logic
        if issues:
            # Need to iterate
            step.success = False
            step.reasoning = f"Issues: {'; '.join(issues)}"

            # Increment iteration for next round
            return {
                "next_action": ActionType.ARCHITECT.value,
                "iteration": iteration + 1,
                "last_error": issues[0],
                "history": state["history"] + [step.to_dict()]
            }

        # Success!
        step.success = True
        step.reasoning = f"Constraints met. {'; '.join(improvements) if improvements else 'Baseline achieved'}"

        return {
            "next_action": ActionType.SIGN_OFF.value,
            "last_error": None,
            "history": state["history"] + [step.to_dict()]
        }

    return critic_node
