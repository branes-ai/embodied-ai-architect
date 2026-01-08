"""Perception-only LangGraph pipeline."""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from embodied_ai_architect.graphs.state import EmbodiedPipelineState, PipelineStage
from embodied_ai_architect.graphs.nodes.perception import (
    create_preprocess_node,
    create_detect_node,
    create_track_node,
    create_scene_graph_node,
)
from embodied_ai_architect.graphs.utils import (
    route_by_next_stage,
    create_error_handler_node,
)


def build_perception_graph(
    execution_target: str = "cpu",
    yolo_variant: str = "s",
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """
    Build perception pipeline graph.

    Pipeline flow:
        preprocess → detect → track → scene_graph → END
                        ↓
                   error_handler → END

    Args:
        execution_target: Hardware target (cpu, gpu, npu)
        yolo_variant: YOLO model variant (n, s, m, l, x)
        checkpointer: Optional checkpoint saver for persistence

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(EmbodiedPipelineState)

    # Create nodes
    workflow.add_node("preprocess", create_preprocess_node(execution_target))
    workflow.add_node("detect", create_detect_node(yolo_variant, execution_target))
    workflow.add_node("track", create_track_node())
    workflow.add_node("scene_graph", create_scene_graph_node())
    workflow.add_node("error_handler", create_error_handler_node())

    # Set entry point
    workflow.set_entry_point("preprocess")

    # Linear edges for happy path
    workflow.add_edge("preprocess", "detect")

    # Conditional routing after detect
    workflow.add_conditional_edges(
        "detect",
        route_by_next_stage,
        {
            PipelineStage.TRACK.value: "track",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # Conditional routing after track
    workflow.add_conditional_edges(
        "track",
        route_by_next_stage,
        {
            PipelineStage.SCENE_GRAPH.value: "scene_graph",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # Conditional routing after scene_graph
    workflow.add_conditional_edges(
        "scene_graph",
        route_by_next_stage,
        {
            PipelineStage.COMPLETE.value: END,
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # Error handler always terminates
    workflow.add_edge("error_handler", END)

    return workflow.compile(checkpointer=checkpointer)
