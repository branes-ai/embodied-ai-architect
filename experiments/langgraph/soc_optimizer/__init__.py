"""
LangGraph SoC Optimizer - Agentic optimization loop for SoC design.

This package implements a LangGraph-based optimization workflow for
System-on-Chip design exploration and PPA optimization.

Example:
    from soc_optimizer import run_optimization, DesignConstraints

    result = run_optimization(
        rtl_code=open("design.v").read(),
        top_module="my_design",
        work_dir="./output",
        constraints={"target_clock_ns": 1.0},
    )
"""

from .state import (
    SoCDesignState,
    DesignConstraints,
    PPAMetrics,
    OptimizationStep,
    ActionType,
    create_initial_state,
    state_summary,
)

from .graph import (
    build_soc_optimizer_graph,
    run_optimization,
    run_optimization_simple,
)

__all__ = [
    # State
    "SoCDesignState",
    "DesignConstraints",
    "PPAMetrics",
    "OptimizationStep",
    "ActionType",
    "create_initial_state",
    "state_summary",
    # Graph
    "build_soc_optimizer_graph",
    "run_optimization",
    "run_optimization_simple",
]

__version__ = "0.1.0"
