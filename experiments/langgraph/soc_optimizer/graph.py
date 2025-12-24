"""
LangGraph State Machine for SoC Optimization.

This module assembles the nodes into a complete optimization loop
with conditional routing based on success/failure at each stage.

Graph Structure:
    architect → linter → synthesis → validation → critic
         ↑         │          │           │          │
         └─────────┴──────────┴───────────┴──────────┘
                        (on failure)

Exit conditions:
- SIGN_OFF: Constraints met, optimization successful
- ABORT: Max iterations or unrecoverable error
"""

from pathlib import Path
from typing import Optional

try:
    from .state import SoCDesignState, ActionType
    from .nodes import (
        create_architect_node,
        create_linter_node,
        create_synthesis_node,
        create_validation_node,
        create_critic_node,
    )
except ImportError:
    # Running as script
    from state import SoCDesignState, ActionType
    from nodes import (
        create_architect_node,
        create_linter_node,
        create_synthesis_node,
        create_validation_node,
        create_critic_node,
    )

# Check for LangGraph availability
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None


def build_soc_optimizer_graph(
    work_dir: Path,
    llm_client: Optional[object] = None,
    mock_mode: bool = False,
    checkpointer: Optional[object] = None,
) -> "StateGraph":
    """
    Build the SoC optimization state machine.

    Args:
        work_dir: Directory for EDA tool outputs
        llm_client: LLM client for architect node (optional)
        mock_mode: If True, architect returns unchanged code (for testing)
        checkpointer: LangGraph checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is not installed. Install with: pip install langgraph"
        )

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create nodes
    architect = create_architect_node(llm_client, mock_mode)
    linter = create_linter_node(work_dir)
    synthesis = create_synthesis_node(work_dir)
    validation = create_validation_node(work_dir)
    critic = create_critic_node()

    # Build graph
    workflow = StateGraph(SoCDesignState)

    # Add nodes
    workflow.add_node("architect", architect)
    workflow.add_node("linter", linter)
    workflow.add_node("synthesis", synthesis)
    workflow.add_node("validation", validation)
    workflow.add_node("critic", critic)

    # Set entry point - start with linter to validate input, then get baseline PPA
    workflow.set_entry_point("linter")

    # =========== Edge Definitions ===========

    # Architect -> always lint the proposed changes
    workflow.add_edge("architect", "linter")

    # Linter -> conditional routing
    workflow.add_conditional_edges(
        "linter",
        lambda s: s["next_action"],
        {
            ActionType.ARCHITECT.value: "architect",   # Syntax error - retry
            ActionType.SYNTHESIZE.value: "synthesis",  # Clean - proceed
        }
    )

    # Synthesis -> conditional routing
    workflow.add_conditional_edges(
        "synthesis",
        lambda s: s["next_action"],
        {
            ActionType.ARCHITECT.value: "architect",   # Synthesis failed
            ActionType.VALIDATE.value: "validation",   # Success - validate
            ActionType.CRITIQUE.value: "critic",       # No testbench - skip to critic
        }
    )

    # Validation -> conditional routing
    workflow.add_conditional_edges(
        "validation",
        lambda s: s["next_action"],
        {
            ActionType.ARCHITECT.value: "architect",   # Validation failed
            ActionType.CRITIQUE.value: "critic",       # Success - analyze
        }
    )

    # Critic -> the main routing decision
    workflow.add_conditional_edges(
        "critic",
        lambda s: s["next_action"],
        {
            ActionType.ARCHITECT.value: "architect",   # PPA not met - iterate
            ActionType.SIGN_OFF.value: END,            # Success!
            ActionType.ABORT.value: END,               # Give up
        }
    )

    # Compile with optional checkpointing
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


def run_optimization(
    rtl_code: str,
    top_module: str,
    work_dir: Path,
    constraints: Optional[dict] = None,
    testbench: Optional[str] = None,
    max_iterations: int = 5,
    llm_client: Optional[object] = None,
    mock_mode: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the SoC optimization loop.

    This is a convenience function that:
    1. Creates the graph
    2. Initializes state
    3. Runs to completion
    4. Returns final state and results

    Args:
        rtl_code: Verilog source code
        top_module: Top-level module name
        work_dir: Directory for outputs
        constraints: PPA constraints (optional)
        testbench: Testbench code (optional)
        max_iterations: Maximum optimization iterations
        llm_client: LLM client for architect
        mock_mode: Skip LLM calls (for testing)
        verbose: Print progress

    Returns:
        Dictionary with final state and summary
    """
    try:
        from .state import create_initial_state, DesignConstraints, state_summary
    except ImportError:
        from state import create_initial_state, DesignConstraints, state_summary

    # Build graph
    graph = build_soc_optimizer_graph(
        work_dir=work_dir,
        llm_client=llm_client,
        mock_mode=mock_mode,
    )

    # Create initial state
    constraint_obj = DesignConstraints(**(constraints or {}))
    initial_state = create_initial_state(
        rtl_code=rtl_code,
        top_module=top_module,
        constraints=constraint_obj,
        testbench=testbench,
        max_iterations=max_iterations,
    )

    # Run graph
    # Set recursion_limit high enough for max_iterations * 5 nodes per iteration + buffer
    recursion_limit = (max_iterations + 1) * 6 + 10
    config = {
        "configurable": {"thread_id": initial_state["project_id"]},
        "recursion_limit": recursion_limit,
    }

    # Accumulate state (stream returns partial updates per node)
    accumulated_state = dict(initial_state)
    for step in graph.stream(initial_state, config):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        # Merge node output into accumulated state
        accumulated_state.update(node_output)

        if verbose:
            action = node_output.get("next_action", "done")
            error = node_output.get("last_error", "")
            print(f"[{node_name}] → {action}" + (f" ({error})" if error else ""))

    final_state = accumulated_state

    # Generate summary
    if verbose and final_state:
        print("\n" + state_summary(final_state))

    return {
        "final_state": final_state,
        "success": final_state.get("next_action") == ActionType.SIGN_OFF.value if final_state else False,
        "iterations": final_state.get("iteration", 0) if final_state else 0,
        "baseline_metrics": final_state.get("baseline_metrics") if final_state else None,
        "final_metrics": final_state.get("current_metrics") if final_state else None,
    }


# Alternative: Simple loop-based execution without LangGraph
def run_optimization_simple(
    rtl_code: str,
    top_module: str,
    work_dir: Path,
    constraints: Optional[dict] = None,
    testbench: Optional[str] = None,
    max_iterations: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run optimization without LangGraph (for testing/comparison).

    This implements the same logic as the graph but in a simple loop,
    useful when LangGraph isn't installed or for debugging.
    """
    try:
        from .state import create_initial_state, DesignConstraints, ActionType, state_summary
        from .tools import RTLLintTool, RTLSynthesisTool, SimulationTool
    except ImportError:
        from state import create_initial_state, DesignConstraints, ActionType, state_summary
        from tools import RTLLintTool, RTLSynthesisTool, SimulationTool

    work_dir = Path(work_dir).resolve()

    # Create tools
    lint_tool = RTLLintTool(work_dir / "lint")
    synth_tool = RTLSynthesisTool(work_dir / "synth")
    sim_tool = SimulationTool(work_dir / "sim")

    # Initialize state
    constraint_obj = DesignConstraints(**(constraints or {}))
    state = create_initial_state(
        rtl_code=rtl_code,
        top_module=top_module,
        constraints=constraint_obj,
        testbench=testbench,
        max_iterations=max_iterations,
    )

    iteration = 0
    while iteration < max_iterations:
        if verbose:
            print(f"\n=== Iteration {iteration} ===")

        # Lint
        if verbose:
            print("[lint] Checking syntax...")
        lint_result = lint_tool.run(state["rtl_code"])
        if not lint_result.get("success"):
            if verbose:
                print(f"[lint] FAILED: {lint_result.get('errors', [])[:2]}")
            break

        # Synthesis
        if verbose:
            print("[synthesis] Running Yosys...")
        synth_result = synth_tool.run(state["rtl_code"], state["top_module"])
        if not synth_result.get("success"):
            if verbose:
                print(f"[synthesis] FAILED: {synth_result.get('errors', [])[:2]}")
            break

        metrics = {
            "area_cells": synth_result.get("area_cells", 0),
            "num_cells": synth_result.get("num_cells", 0),
        }

        # Store baseline
        if state.get("baseline_metrics") is None:
            state["baseline_metrics"] = metrics
            if verbose:
                print(f"[synthesis] Baseline: {metrics['area_cells']} cells")

        state["current_metrics"] = metrics
        if verbose:
            print(f"[synthesis] Current: {metrics['area_cells']} cells")

        # Validation (if testbench)
        if state.get("testbench"):
            if verbose:
                print("[validation] Running simulation...")
            sim_result = sim_tool.run(
                state["rtl_code"],
                state["testbench"],
                state["top_module"]
            )
            if not sim_result.get("success"):
                if verbose:
                    print(f"[validation] FAILED: {sim_result.get('error_message', 'Unknown')}")
                break
            if verbose:
                print(f"[validation] PASSED: {sim_result.get('tests_passed', 0)} tests")

        # Check constraints (simple version - just check area)
        max_area = (constraints or {}).get("max_area_cells")
        if max_area and metrics["area_cells"] > max_area:
            if verbose:
                print(f"[critic] Area {metrics['area_cells']} > max {max_area}")
                print("[architect] Skipped (simple mode - no LLM optimization)")
            iteration += 1
            # In simple mode, we can't actually optimize - just re-run same code
            # This will always fail if constraints aren't met on first try
            continue

        # Success!
        if verbose:
            print("\n[critic] Constraints met!")
        state["next_action"] = ActionType.SIGN_OFF.value
        break

    else:
        state["next_action"] = ActionType.ABORT.value

    if verbose:
        print("\n" + state_summary(state))

    return {
        "final_state": state,
        "success": state.get("next_action") == ActionType.SIGN_OFF.value,
        "iterations": iteration,
        "baseline_metrics": state.get("baseline_metrics"),
        "final_metrics": state.get("current_metrics"),
    }
