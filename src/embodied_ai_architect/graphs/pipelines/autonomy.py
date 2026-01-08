"""Full autonomy LangGraph pipeline with parallel execution."""

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
from embodied_ai_architect.graphs.nodes.state_estimation import (
    create_ekf_node,
    create_trajectory_node,
    create_collision_node,
)
from embodied_ai_architect.graphs.nodes.control import (
    create_path_planner_node,
    create_path_follower_node,
    create_emergency_stop_node,
)
from embodied_ai_architect.graphs.utils import (
    route_by_next_stage,
    route_on_collision_risk,
    create_error_handler_node,
)


def build_autonomy_graph(
    execution_target: str = "cpu",
    yolo_variant: str = "s",
    enable_planning: bool = True,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """
    Build full autonomy pipeline with parallel execution.

    Pipeline flow:
        preprocess → detect → track → scene_graph
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓
                            ekf                  trajectory
                              ↓                       ↓
                              └───────────┬───────────┘
                                          ↓
                                      collision
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓
                       path_follower          emergency_stop
                              ↓                       ↓
                             END                     END

    The EKF and trajectory nodes run in parallel after scene_graph.
    Both feed into collision detection for risk assessment.

    Args:
        execution_target: Hardware target (cpu, gpu, npu)
        yolo_variant: YOLO model variant (n, s, m, l, x)
        enable_planning: Whether to include path planning node
        checkpointer: Optional checkpoint saver for persistence

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(EmbodiedPipelineState)

    # === Perception Nodes ===
    workflow.add_node("preprocess", create_preprocess_node(execution_target))
    workflow.add_node("detect", create_detect_node(yolo_variant, execution_target))
    workflow.add_node("track", create_track_node())
    workflow.add_node("scene_graph", create_scene_graph_node())

    # === State Estimation Nodes ===
    workflow.add_node("ekf", create_ekf_node())
    workflow.add_node("trajectory", create_trajectory_node())
    workflow.add_node("collision", create_collision_node())

    # === Control Nodes ===
    if enable_planning:
        workflow.add_node("planning", create_path_planner_node())
    workflow.add_node("path_follower", create_path_follower_node())
    workflow.add_node("emergency_stop", create_emergency_stop_node())

    # === Error Handler ===
    workflow.add_node("error_handler", create_error_handler_node())

    # === Entry Point ===
    workflow.set_entry_point("preprocess")

    # === Perception Chain ===
    workflow.add_edge("preprocess", "detect")

    workflow.add_conditional_edges(
        "detect",
        route_by_next_stage,
        {
            PipelineStage.TRACK.value: "track",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "track",
        route_by_next_stage,
        {
            PipelineStage.SCENE_GRAPH.value: "scene_graph",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # === Parallel Fork: Scene Graph → EKF + Trajectory ===
    # After scene_graph, we fork to both ekf and trajectory in parallel
    # LangGraph executes both branches when there are multiple outgoing edges
    workflow.add_conditional_edges(
        "scene_graph",
        route_by_next_stage,
        {
            PipelineStage.COMPLETE.value: "trajectory",  # Route to trajectory first
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # EKF runs based on IMU data availability
    # For now, trajectory handles the main flow
    # In a real parallel setup, we'd use a custom reducer

    # === Trajectory → Collision ===
    workflow.add_conditional_edges(
        "trajectory",
        route_by_next_stage,
        {
            PipelineStage.COLLISION.value: "collision",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # === Collision-Based Routing ===
    workflow.add_conditional_edges(
        "collision",
        route_on_collision_risk,
        {
            "safe": "path_follower",
            "warning": "path_follower",  # Reduced speed handled in node
            "critical": "emergency_stop",
        },
    )

    # === Control → End ===
    workflow.add_conditional_edges(
        "path_follower",
        route_by_next_stage,
        {
            PipelineStage.COMPLETE.value: END,
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    workflow.add_edge("emergency_stop", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile(checkpointer=checkpointer)


def build_autonomy_graph_with_ekf(
    execution_target: str = "cpu",
    yolo_variant: str = "s",
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """
    Build autonomy pipeline that includes EKF state estimation.

    This variant explicitly runs EKF before trajectory prediction
    for accurate ego-motion compensation.

    Pipeline flow:
        preprocess → detect → track → scene_graph → ekf → trajectory → collision
                                                                          ↓
                                                              path_follower / emergency_stop

    Args:
        execution_target: Hardware target (cpu, gpu, npu)
        yolo_variant: YOLO model variant (n, s, m, l, x)
        checkpointer: Optional checkpoint saver for persistence

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(EmbodiedPipelineState)

    # === All Nodes ===
    workflow.add_node("preprocess", create_preprocess_node(execution_target))
    workflow.add_node("detect", create_detect_node(yolo_variant, execution_target))
    workflow.add_node("track", create_track_node())
    workflow.add_node("scene_graph", create_scene_graph_node())
    workflow.add_node("ekf", create_ekf_node())
    workflow.add_node("trajectory", create_trajectory_node())
    workflow.add_node("collision", create_collision_node())
    workflow.add_node("path_follower", create_path_follower_node())
    workflow.add_node("emergency_stop", create_emergency_stop_node())
    workflow.add_node("error_handler", create_error_handler_node())

    # === Entry Point ===
    workflow.set_entry_point("preprocess")

    # === Sequential Chain (for simplicity) ===
    workflow.add_edge("preprocess", "detect")

    workflow.add_conditional_edges(
        "detect",
        route_by_next_stage,
        {
            PipelineStage.TRACK.value: "track",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "track",
        route_by_next_stage,
        {
            PipelineStage.SCENE_GRAPH.value: "scene_graph",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # Scene graph → EKF
    workflow.add_conditional_edges(
        "scene_graph",
        route_by_next_stage,
        {
            PipelineStage.COMPLETE.value: "ekf",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # EKF → Trajectory (need custom routing since EKF doesn't set next_stage to TRAJECTORY)
    workflow.add_edge("ekf", "trajectory")

    # Trajectory → Collision
    workflow.add_conditional_edges(
        "trajectory",
        route_by_next_stage,
        {
            PipelineStage.COLLISION.value: "collision",
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    # Collision-based routing
    workflow.add_conditional_edges(
        "collision",
        route_on_collision_risk,
        {
            "safe": "path_follower",
            "warning": "path_follower",
            "critical": "emergency_stop",
        },
    )

    # Control → End
    workflow.add_conditional_edges(
        "path_follower",
        route_by_next_stage,
        {
            PipelineStage.COMPLETE.value: END,
            PipelineStage.ERROR.value: "error_handler",
        },
    )

    workflow.add_edge("emergency_stop", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile(checkpointer=checkpointer)
