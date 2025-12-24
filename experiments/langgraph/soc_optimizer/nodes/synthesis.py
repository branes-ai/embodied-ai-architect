"""
Synthesis Node - Run EDA synthesis and extract PPA metrics.

This node runs Yosys synthesis and extracts area metrics.
On first run, it stores baseline metrics for comparison.
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
    from ..tools import RTLSynthesisTool
except ImportError:
    from state import SoCDesignState, ActionType, OptimizationStep
    from tools import RTLSynthesisTool


def create_synthesis_node(work_dir: Path) -> Callable[[SoCDesignState], dict]:
    """
    Create the synthesis node.

    The synthesis node:
    1. Runs Yosys synthesis on the RTL
    2. Extracts PPA metrics (cell count, wire count, etc.)
    3. Stores baseline metrics on first run
    4. Routes to validation on success, architect on failure
    """
    synth_tool = RTLSynthesisTool(work_dir / "synth")

    def synthesis_node(state: SoCDesignState) -> dict:
        start_time = time.time()

        # Run synthesis
        result = synth_tool.run(
            state["rtl_code"],
            state["top_module"],
            optimize=True
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Create step record
        step = OptimizationStep(
            iteration=state["iteration"],
            agent="synthesis",
            action="run_yosys",
            success=result.get("success", False),
            metrics_after=result if result.get("success") else None,
            duration_ms=duration_ms
        )

        if not result.get("success", False):
            errors = result.get("errors", ["Unknown synthesis error"])
            step.error = "; ".join(errors[:3])

            return {
                "next_action": ActionType.ARCHITECT.value,
                "synthesis_errors": errors,
                "last_error": f"Synthesis failed: {errors[0] if errors else 'Unknown'}",
                "history": state["history"] + [step.to_dict()]
            }

        # Extract metrics
        metrics = {
            "area_cells": result.get("area_cells", 0),
            "area_um2": result.get("area_um2", 0.0),
            "num_wires": result.get("num_wires", 0),
            "num_cells": result.get("num_cells", 0),
            "cell_counts": result.get("cell_counts", {}),
        }

        # Build update dict
        update = {
            "current_metrics": metrics,
            "synthesis_errors": [],
            "last_error": None,
            "history": state["history"] + [step.to_dict()]
        }

        # Store baseline on first successful synthesis
        if state.get("baseline_metrics") is None:
            update["baseline_metrics"] = metrics
            step.reasoning = "First synthesis - storing as baseline"

        # Route to validation if we have a testbench, otherwise to critic
        if state.get("testbench"):
            update["next_action"] = ActionType.VALIDATE.value
        else:
            update["next_action"] = ActionType.CRITIQUE.value

        return update

    return synthesis_node
