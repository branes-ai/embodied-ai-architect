"""
Linter Node - Fast syntax checking before synthesis.

This is the "fast-fail" node that catches errors quickly
before expensive synthesis runs.
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
    from ..tools import RTLLintTool
except ImportError:
    from state import SoCDesignState, ActionType, OptimizationStep
    from tools import RTLLintTool


def create_linter_node(work_dir: Path) -> Callable[[SoCDesignState], dict]:
    """
    Create the linter node.

    The linter node:
    1. Runs fast syntax checking on the current RTL
    2. Routes back to architect on errors
    3. Routes to synthesis on success
    """
    lint_tool = RTLLintTool(work_dir / "lint")

    def linter_node(state: SoCDesignState) -> dict:
        start_time = time.time()

        # Run linting
        result = lint_tool.run(state["rtl_code"])
        duration_ms = int((time.time() - start_time) * 1000)

        # Create step record
        step = OptimizationStep(
            iteration=state["iteration"],
            agent="linter",
            action="syntax_check",
            success=result.get("success", False),
            duration_ms=duration_ms
        )

        if not result.get("success", False):
            errors = result.get("errors", ["Unknown lint error"])
            step.error = "; ".join(errors[:3])  # First 3 errors

            return {
                "next_action": ActionType.ARCHITECT.value,
                "lint_errors": errors,
                "last_error": f"Lint failed: {errors[0] if errors else 'Unknown error'}",
                "history": state["history"] + [step.to_dict()]
            }

        # Success - proceed to synthesis
        return {
            "next_action": ActionType.SYNTHESIZE.value,
            "lint_errors": [],
            "last_error": None,
            "history": state["history"] + [step.to_dict()]
        }

    return linter_node
