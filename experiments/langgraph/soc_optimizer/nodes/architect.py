"""
Architect Node - LLM-powered RTL optimization.

This node uses an LLM to analyze the current design and
propose optimizations based on the critic's feedback.
"""

import sys
import re
from pathlib import Path
from typing import Callable, Optional

# Handle both package and script imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from ..state import SoCDesignState, ActionType, OptimizationStep
except ImportError:
    from state import SoCDesignState, ActionType, OptimizationStep


# System prompt for the RTL Architect
ARCHITECT_SYSTEM_PROMPT = """You are a Senior RTL Architect specializing in Verilog/SystemVerilog optimization.

Your task is to improve digital designs to meet PPA (Power, Performance, Area) constraints.

## Optimization Strategies

1. **Pipelining**: Insert registers to break long combinational paths
   - Identify deep logic chains (multipliers, adders in series)
   - Add pipeline stages with proper enable/valid signals

2. **Resource Sharing**: Combine similar operations
   - Share multipliers/adders across different operations
   - Use muxes to select operands

3. **Logic Restructuring**: Simplify combinational logic
   - Flatten nested if-else into parallel logic
   - Convert priority encoders to one-hot where appropriate

4. **Strength Reduction**: Replace expensive operations
   - Multiplication by constants -> shifts and adds
   - Division by powers of 2 -> right shifts

## Rules

- PRESERVE the module interface (ports must remain identical)
- PRESERVE functional correctness
- Output ONLY valid Verilog code
- Add comments explaining significant changes
- Do NOT add unnecessary complexity

## Output Format

Return ONLY the modified Verilog code, wrapped in ```verilog ... ``` markers.
"""


ARCHITECT_USER_PROMPT = """## Current Design

```verilog
{rtl_code}
```

## Constraints
{constraints}

## Current Metrics
{metrics}

## Issues to Address
{issues}

## Previous Attempts
{history}

Please provide an optimized version of this design that addresses the issues."""


def create_architect_node(
    llm_client: Optional[object] = None,
    mock_mode: bool = False
) -> Callable[[SoCDesignState], dict]:
    """
    Create the architect node.

    Args:
        llm_client: An LLM client with a .invoke(prompt) method.
                   If None and not mock_mode, will try to create one.
        mock_mode: If True, return the original code unchanged (for testing).

    The architect node:
    1. Analyzes current design and metrics
    2. Uses LLM to propose optimizations
    3. Extracts modified RTL from response
    4. Routes to linter for validation
    """

    def architect_node(state: SoCDesignState) -> dict:
        # Architect is only called when critic decides optimization is needed
        # We should always have baseline metrics at this point
        if state.get("baseline_metrics") is None:
            return {
                "next_action": ActionType.ABORT.value,
                "last_error": "Architect called without baseline metrics",
                "history": state["history"] + [{
                    "iteration": state["iteration"],
                    "agent": "architect",
                    "action": "error",
                    "error": "No baseline metrics available",
                    "success": False
                }]
            }

        # Create step record
        step = OptimizationStep(
            iteration=state["iteration"],
            agent="architect",
            action="optimize_rtl"
        )

        # Mock mode - return original code
        if mock_mode or llm_client is None:
            step.reasoning = "Mock mode - no changes made"
            step.success = True

            return {
                "next_action": ActionType.LINT.value,
                "history": state["history"] + [step.to_dict()]
            }

        # Build prompt
        issues = []
        if state.get("last_error"):
            issues.append(state["last_error"])
        if state.get("lint_errors"):
            issues.extend(state["lint_errors"][:3])
        if state.get("synthesis_errors"):
            issues.extend(state["synthesis_errors"][:3])
        if state.get("validation_errors"):
            issues.extend(state["validation_errors"][:3])

        # Get recent history
        recent_history = state["history"][-5:] if state["history"] else []
        history_str = "\n".join([
            f"- Iteration {h.get('iteration', '?')}: {h.get('action', '?')} - {h.get('reasoning', h.get('error', 'no info'))}"
            for h in recent_history
        ])

        user_prompt = ARCHITECT_USER_PROMPT.format(
            rtl_code=state["rtl_code"],
            constraints=state.get("constraints", {}),
            metrics=state.get("current_metrics", {}),
            issues="\n".join(f"- {i}" for i in issues) if issues else "None",
            history=history_str if history_str else "None"
        )

        try:
            # Call LLM
            response = llm_client.invoke([
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])

            # Extract Verilog from response
            new_rtl = extract_verilog(response.content)

            if new_rtl and new_rtl != state["rtl_code"]:
                step.success = True
                step.reasoning = "Generated optimized RTL"

                return {
                    "rtl_code": new_rtl,
                    "next_action": ActionType.LINT.value,
                    "history": state["history"] + [step.to_dict()]
                }
            else:
                step.success = False
                step.error = "No valid RTL changes extracted from LLM response"

                return {
                    "next_action": ActionType.ABORT.value,
                    "last_error": "Architect failed to produce valid changes",
                    "history": state["history"] + [step.to_dict()]
                }

        except Exception as e:
            step.success = False
            step.error = str(e)

            return {
                "next_action": ActionType.ABORT.value,
                "last_error": f"Architect error: {e}",
                "history": state["history"] + [step.to_dict()]
            }

    return architect_node


def extract_verilog(text: str) -> Optional[str]:
    """Extract Verilog code from LLM response."""
    # Try to find code in ```verilog ... ``` blocks
    pattern = r'```(?:verilog|systemverilog|sv)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Return the longest match (usually the full module)
        return max(matches, key=len).strip()

    # Try to find module ... endmodule
    module_pattern = r'(module\s+\w+.*?endmodule)'
    module_matches = re.findall(module_pattern, text, re.DOTALL)

    if module_matches:
        return max(module_matches, key=len).strip()

    return None
