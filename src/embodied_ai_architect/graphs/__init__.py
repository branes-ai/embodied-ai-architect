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

__all__ = [
    # State
    "EmbodiedPipelineState",
    "PipelineStage",
    "create_initial_state",
    "format_timing_summary",
    "get_total_latency_ms",
    "is_over_budget",
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
