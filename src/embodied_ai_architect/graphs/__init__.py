"""LangGraph-based operator orchestration for embodied AI pipelines.

This module provides LangGraph-based orchestration for embodied AI operators.
Requires the 'langgraph' optional dependency:

    pip install -e ".[langgraph]"

Usage:
    from embodied_ai_architect.graphs import (
        build_perception_graph,
        build_autonomy_graph,
        PipelineRunner,
    )

    graph = build_perception_graph()
    runner = PipelineRunner(graph)
    result = runner.run_batch(frame=image)
"""

from embodied_ai_architect.graphs.state import (
    EmbodiedPipelineState,
    PipelineStage,
    create_initial_state,
    format_timing_summary,
    get_total_latency_ms,
    is_over_budget,
)
from embodied_ai_architect.graphs.task_graph import (
    CycleError,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    DesignDecision,
    DesignStatus,
    PPAMetrics,
    SoCDesignState,
    create_initial_soc_state,
    get_constraints,
    get_iteration_summary,
    get_ppa_metrics,
    get_task_graph,
    is_design_complete,
    is_over_iteration_limit,
    record_decision,
    set_task_graph,
)
from embodied_ai_architect.graphs.planner import (
    PlannerNode,
    create_planner_node,
    parse_plan_json,
    tasks_to_graph,
)
from embodied_ai_architect.graphs.dispatcher import (
    Dispatcher,
    DispatchError,
)
from embodied_ai_architect.graphs.specialists import (
    architecture_composer,
    create_default_dispatcher,
    critic,
    hw_explorer,
    ppa_assessor,
    report_generator,
    workload_analyzer,
)
from embodied_ai_architect.graphs.soc_state import (
    get_dependency_results,
    get_task_result,
)

__all__ = [
    # Perception pipeline state
    "EmbodiedPipelineState",
    "PipelineStage",
    "create_initial_state",
    "format_timing_summary",
    "get_total_latency_ms",
    "is_over_budget",
    # Task graph engine
    "CycleError",
    "TaskGraph",
    "TaskNode",
    "TaskStatus",
    # SoC design state
    "DesignConstraints",
    "DesignDecision",
    "DesignStatus",
    "PPAMetrics",
    "SoCDesignState",
    "create_initial_soc_state",
    "get_constraints",
    "get_iteration_summary",
    "get_ppa_metrics",
    "get_task_graph",
    "is_design_complete",
    "is_over_iteration_limit",
    "record_decision",
    "set_task_graph",
    # Planner
    "PlannerNode",
    "create_planner_node",
    "parse_plan_json",
    "tasks_to_graph",
    # Dispatcher
    "Dispatcher",
    "DispatchError",
    # Specialists
    "architecture_composer",
    "create_default_dispatcher",
    "critic",
    "hw_explorer",
    "ppa_assessor",
    "report_generator",
    "workload_analyzer",
    # State helpers
    "get_dependency_results",
    "get_task_result",
]

# Lazy imports for langgraph-dependent modules
def __getattr__(name):
    """Lazy import for langgraph-dependent modules."""
    if name == "PipelineRunner":
        from embodied_ai_architect.graphs.runner import PipelineRunner
        return PipelineRunner
    elif name == "build_perception_graph":
        from embodied_ai_architect.graphs.pipelines.perception import build_perception_graph
        return build_perception_graph
    elif name == "build_autonomy_graph":
        from embodied_ai_architect.graphs.pipelines.autonomy import build_autonomy_graph
        return build_autonomy_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
