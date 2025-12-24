"""
Validation Node - Run functional simulation.

This node ensures that optimized designs remain functionally correct.
It's the "ground truth" that catches semantic errors LLMs might introduce.
"""

import sys
import time
from pathlib import Path
from typing import Callable

# Handle both package and script imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from ..state import SoCDesignState, ActionType, OptimizationStep
    from ..tools import SimulationTool
except ImportError:
    from state import SoCDesignState, ActionType, OptimizationStep
    from tools import SimulationTool


def create_validation_node(work_dir: Path) -> Callable[[SoCDesignState], dict]:
    """
    Create the validation node.

    The validation node:
    1. Runs functional simulation with testbench
    2. Checks that all tests pass
    3. Routes to critic on success, architect on failure
    """
    sim_tool = SimulationTool(work_dir / "sim")

    def validation_node(state: SoCDesignState) -> dict:
        # Skip if no testbench
        if not state.get("testbench"):
            return {
                "next_action": ActionType.CRITIQUE.value,
                "history": state["history"] + [{
                    "iteration": state["iteration"],
                    "agent": "validation",
                    "action": "skipped",
                    "reasoning": "No testbench provided",
                    "success": True
                }]
            }

        start_time = time.time()

        # Run simulation
        result = sim_tool.run(
            state["rtl_code"],
            state["testbench"],
            state["top_module"]
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Create step record
        step = OptimizationStep(
            iteration=state["iteration"],
            agent="validation",
            action="run_simulation",
            success=result.get("success", False),
            duration_ms=duration_ms
        )

        if not result.get("success", False):
            error_msg = result.get("error_message", "Validation failed")
            tests_failed = result.get("tests_failed", 0)
            step.error = f"{error_msg} ({tests_failed} tests failed)"

            return {
                "next_action": ActionType.ARCHITECT.value,
                "validation_errors": [error_msg],
                "last_error": f"Validation failed: {error_msg}",
                "history": state["history"] + [step.to_dict()]
            }

        # Success
        tests_passed = result.get("tests_passed", 0)
        step.reasoning = f"{tests_passed} tests passed"

        return {
            "next_action": ActionType.CRITIQUE.value,
            "validation_errors": [],
            "last_error": None,
            "history": state["history"] + [step.to_dict()]
        }

    return validation_node
