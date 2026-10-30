# Plan: LangGraph Operator Orchestration

## Overview

Create a LangGraph-based orchestration system for the embodied AI operators (perception, state estimation, control) that enables:
- Declarative pipeline definition with conditional routing
- Built-in checkpointing and state persistence
- Integration with existing benchmark harness
- Parallelism support for independent operators

## Current State Analysis

### Operators (12 total)
```
Perception:        ImagePreprocessor → YOLOv8 → ByteTrack → SceneGraphManager
State Estimation:  EKF6DOF, TrajectoryPredictor, CollisionDetector
Control:           PathPlannerAStar, PathFollower, PIDController
```

All operators implement:
- `setup(config, execution_target)` - Initialize
- `process(inputs: dict) -> dict` - Execute
- `teardown()` - Cleanup
- `benchmark()` - Performance measurement

### LangGraph Pattern (from SoC Optimizer)
- State: TypedDict with `next_action` routing field
- Nodes: Factory functions returning state updates
- Routing: Conditional edges via `lambda s: s["next_action"]`
- Execution: `graph.stream()` for real-time updates

### Dependencies
Already in `pyproject.toml`:
```toml
langgraph = ["langgraph>=0.2.0", "langchain-anthropic>=0.3.0"]
```

---

## Proposed Architecture

### 1. State Schema

```python
# src/embodied_ai_architect/graphs/state.py

class PipelineStage(str, Enum):
    PREPROCESS = "preprocess"
    DETECT = "detect"
    TRACK = "track"
    SCENE_GRAPH = "scene_graph"
    STATE_ESTIMATION = "state_estimation"
    TRAJECTORY = "trajectory"
    COLLISION = "collision"
    PLANNING = "planning"
    CONTROL = "control"
    COMPLETE = "complete"
    ERROR = "error"

class EmbodiedPipelineState(TypedDict):
    # Input data
    frame: Optional[dict]           # {"image": ndarray, "timestamp": float}
    imu_data: Optional[dict]        # {"accel": [...], "gyro": [...]}

    # Intermediate results (accumulated)
    preprocessed: Optional[dict]
    detections: Optional[dict]
    tracks: Optional[dict]
    scene_objects: Optional[dict]
    ego_pose: Optional[dict]
    trajectories: Optional[dict]
    collision_risks: Optional[dict]
    planned_path: Optional[dict]
    control_output: Optional[dict]

    # Routing control
    next_stage: str                 # PipelineStage value
    enabled_stages: List[str]       # Which stages to run

    # Configuration
    operator_configs: dict          # Per-operator config overrides
    execution_target: str           # "cpu", "gpu", "npu"

    # Tracking
    iteration: int
    frame_id: int
    timing: dict                    # Per-stage timing
    errors: List[str]

    # Metadata
    pipeline_id: str
    created_at: str
```

### 2. Directory Structure

```
src/embodied_ai_architect/graphs/
├── __init__.py
├── state.py                    # EmbodiedPipelineState TypedDict
├── nodes/
│   ├── __init__.py
│   ├── perception.py           # preprocess, detect, track, scene_graph nodes
│   ├── state_estimation.py     # ekf, trajectory, collision nodes
│   └── control.py              # planning, path_follow, pid nodes
├── pipelines/
│   ├── __init__.py
│   ├── perception.py           # Perception-only subgraph
│   ├── autonomy.py             # Full autonomy pipeline
│   └── benchmark.py            # Benchmark-integrated pipeline
└── utils.py                    # Routing helpers, timing utilities
```

### 3. Node Factory Pattern

```python
# src/embodied_ai_architect/graphs/nodes/perception.py

def create_detect_node(
    variant: str = "s",
    execution_target: str = "cpu"
) -> Callable[[EmbodiedPipelineState], dict]:
    """Factory for YOLO detection node."""

    operator = None  # Lazy initialization

    def detect_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if PipelineStage.DETECT.value not in state["enabled_stages"]:
            return {"next_stage": PipelineStage.TRACK.value}

        # Lazy init
        if operator is None:
            operator = create_operator(
                f"yolo_detector_{variant}",
                state.get("operator_configs", {}).get("detect", {}),
                execution_target
            )

        start = time.perf_counter_ns()
        try:
            result = operator.process({"image": state["preprocessed"]["processed"]})
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6

            return {
                "detections": result,
                "next_stage": PipelineStage.TRACK.value,
                "timing": {**state.get("timing", {}), "detect": elapsed_ms},
            }
        except Exception as e:
            return {
                "next_stage": PipelineStage.ERROR.value,
                "errors": state.get("errors", []) + [f"detect: {e}"],
            }

    return detect_node
```

### 4. Graph Construction

```python
# src/embodied_ai_architect/graphs/pipelines/perception.py

def build_perception_graph(
    execution_target: str = "cpu",
    yolo_variant: str = "s",
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledGraph:
    """Build perception pipeline graph."""

    workflow = StateGraph(EmbodiedPipelineState)

    # Create nodes
    workflow.add_node("preprocess", create_preprocess_node(execution_target))
    workflow.add_node("detect", create_detect_node(yolo_variant, execution_target))
    workflow.add_node("track", create_track_node())
    workflow.add_node("scene_graph", create_scene_graph_node())
    workflow.add_node("error_handler", create_error_handler_node())

    # Entry point
    workflow.set_entry_point("preprocess")

    # Edges
    workflow.add_edge("preprocess", "detect")

    # Conditional routing after detect
    workflow.add_conditional_edges(
        "detect",
        lambda s: s["next_stage"],
        {
            PipelineStage.TRACK.value: "track",
            PipelineStage.ERROR.value: "error_handler",
        }
    )

    workflow.add_conditional_edges(
        "track",
        lambda s: s["next_stage"],
        {
            PipelineStage.SCENE_GRAPH.value: "scene_graph",
            PipelineStage.COMPLETE.value: END,
            PipelineStage.ERROR.value: "error_handler",
        }
    )

    workflow.add_edge("scene_graph", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile(checkpointer=checkpointer)
```

### 5. Full Autonomy Pipeline

```python
# src/embodied_ai_architect/graphs/pipelines/autonomy.py

def build_autonomy_graph(...) -> CompiledGraph:
    """
    Full autonomy loop:

    [Perception] → [State Est] → [Collision] → [Control]
         ↓              ↓            ↓            ↓
      objects       ego_pose    collision?    commands
                                    │
                                    ├─ safe → PathFollower
                                    └─ risk → EmergencyStop
    """

    workflow = StateGraph(EmbodiedPipelineState)

    # Perception nodes
    workflow.add_node("preprocess", create_preprocess_node())
    workflow.add_node("detect", create_detect_node())
    workflow.add_node("track", create_track_node())
    workflow.add_node("scene_graph", create_scene_graph_node())

    # State estimation nodes
    workflow.add_node("ekf", create_ekf_node())
    workflow.add_node("trajectory", create_trajectory_node())
    workflow.add_node("collision", create_collision_node())

    # Control nodes
    workflow.add_node("path_follower", create_path_follower_node())
    workflow.add_node("emergency_stop", create_emergency_stop_node())

    # Perception chain
    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "detect")
    workflow.add_edge("detect", "track")
    workflow.add_edge("track", "scene_graph")

    # Fork: perception feeds both EKF and trajectory
    workflow.add_edge("scene_graph", "ekf")
    workflow.add_edge("scene_graph", "trajectory")  # Parallel if supported

    # Trajectory → Collision assessment
    workflow.add_edge("trajectory", "collision")

    # Collision-based routing
    workflow.add_conditional_edges(
        "collision",
        route_on_collision_risk,
        {
            "safe": "path_follower",
            "warning": "path_follower",  # Reduced speed
            "critical": "emergency_stop",
        }
    )

    workflow.add_edge("path_follower", END)
    workflow.add_edge("emergency_stop", END)

    return workflow.compile()
```

### 6. Parallel Execution Pattern

LangGraph supports parallel node execution via branching edges:

```python
# Parallel fork: scene_graph feeds both EKF and trajectory simultaneously
workflow.add_edge("scene_graph", "ekf")
workflow.add_edge("scene_graph", "trajectory")

# Both must complete before collision assessment
# Use a "join" node that waits for both inputs
workflow.add_node("join_state", create_join_node(["ekf", "trajectory"]))
workflow.add_edge("ekf", "join_state")
workflow.add_edge("trajectory", "join_state")
workflow.add_edge("join_state", "collision")
```

### 7. Streaming Execution Mode

For continuous video processing:

```python
# src/embodied_ai_architect/graphs/runner.py

class PipelineRunner:
    """Executes pipeline in batch or streaming mode."""

    def __init__(self, graph: CompiledGraph):
        self.graph = graph

    def run_batch(self, frame: np.ndarray) -> EmbodiedPipelineState:
        """Single frame execution."""
        state = create_initial_state(frame=frame)
        for step in self.graph.stream(state):
            pass
        return step[list(step.keys())[0]]

    async def run_stream(
        self,
        source: VideoSource,
        callback: Callable[[EmbodiedPipelineState], None]
    ):
        """Continuous video stream execution."""
        frame_id = 0
        async for frame in source:
            state = create_initial_state(
                frame=frame,
                frame_id=frame_id,
            )
            result = self.run_batch(frame)
            callback(result)
            frame_id += 1
```

### 8. Optional LLM Architect Node

For runtime optimization experiments:

```python
# src/embodied_ai_architect/graphs/nodes/architect.py

def create_architect_node(
    llm_client: Optional[LLMClient] = None,
    enabled: bool = False,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Optional LLM-powered optimization node.

    Analyzes pipeline metrics and suggests:
    - Operator config adjustments (thresholds, model variants)
    - Stage skipping for performance
    - Early exit conditions
    """

    def architect_node(state: EmbodiedPipelineState) -> dict:
        if not enabled or llm_client is None:
            return {"next_stage": state["next_stage"]}  # Pass-through

        # Analyze recent timing
        timing = state.get("timing", {})
        total_ms = sum(timing.values())

        # If over budget, ask LLM for optimization suggestions
        if total_ms > state.get("latency_budget_ms", float("inf")):
            suggestions = llm_client.chat([
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": format_optimization_request(state)},
            ])
            return apply_suggestions(state, suggestions)

        return {"next_stage": state["next_stage"]}

    return architect_node
```

### 9. CLI Integration

```python
# src/embodied_ai_architect/cli/pipeline.py

@click.command()
@click.option("--pipeline", type=click.Choice(["perception", "autonomy"]))
@click.option("--input", type=click.Path(exists=True), help="Input image/video")
@click.option("--execution-target", default="cpu")
@click.option("--stream", is_flag=True, help="Streaming mode for video")
@click.option("--architect", is_flag=True, help="Enable LLM architect node")
@click.option("--checkpoint", is_flag=True, help="Enable checkpointing")
def pipeline(pipeline_name, input_path, execution_target, stream, architect, checkpoint):
    """Run operator pipeline with LangGraph orchestration."""

    # Build graph
    graph = build_pipeline(
        pipeline_name,
        execution_target=execution_target,
        enable_architect=architect,
        checkpointer=SqliteSaver.from_conn_string(":memory:") if checkpoint else None,
    )

    runner = PipelineRunner(graph)

    if stream:
        # Video streaming mode
        source = VideoSource(input_path)
        asyncio.run(runner.run_stream(source, print_results))
    else:
        # Batch mode
        frame = load_image(input_path)
        result = runner.run_batch(frame)
        print_results(result)
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (~200 LOC)
1. Create `src/embodied_ai_architect/graphs/` directory structure
2. Implement `state.py` with `EmbodiedPipelineState` and `PipelineStage` enum
3. Implement `utils.py` with routing helpers, timing utilities, and join node

### Phase 2: Perception Pipeline (~400 LOC)
4. Implement perception nodes in `nodes/perception.py`:
   - `create_preprocess_node()`
   - `create_detect_node()`
   - `create_track_node()`
   - `create_scene_graph_node()`
5. Build perception graph in `pipelines/perception.py`
6. Add unit tests for perception pipeline

### Phase 3: State Estimation & Control (~350 LOC)
7. Implement state estimation nodes in `nodes/state_estimation.py`:
   - `create_ekf_node()`
   - `create_trajectory_node()`
   - `create_collision_node()`
8. Implement control nodes in `nodes/control.py`:
   - `create_path_planner_node()`
   - `create_path_follower_node()`
   - `create_emergency_stop_node()`
9. Build full autonomy graph with parallel execution in `pipelines/autonomy.py`

### Phase 4: Runner & Streaming (~200 LOC)
10. Implement `PipelineRunner` class in `runner.py`:
    - `run_batch()` for single-frame execution
    - `run_stream()` for continuous video processing
11. Add `VideoSource` wrapper for OpenCV video capture

### Phase 5: Optional LLM Architect (~150 LOC)
12. Implement `nodes/architect.py`:
    - `create_architect_node()` with pass-through when disabled
    - `ARCHITECT_SYSTEM_PROMPT` for optimization guidance
    - `apply_suggestions()` for runtime config modification

### Phase 6: CLI & Integration (~150 LOC)
13. Add `pipeline` CLI command in `cli/pipeline.py`
14. Integrate with existing benchmark harness
15. Add end-to-end tests

**Estimated Total: ~1,450 LOC**

---

## Key Files to Create/Modify

**New Files:**
- `src/embodied_ai_architect/graphs/__init__.py`
- `src/embodied_ai_architect/graphs/state.py`
- `src/embodied_ai_architect/graphs/utils.py`
- `src/embodied_ai_architect/graphs/runner.py`
- `src/embodied_ai_architect/graphs/nodes/__init__.py`
- `src/embodied_ai_architect/graphs/nodes/perception.py`
- `src/embodied_ai_architect/graphs/nodes/state_estimation.py`
- `src/embodied_ai_architect/graphs/nodes/control.py`
- `src/embodied_ai_architect/graphs/nodes/architect.py`
- `src/embodied_ai_architect/graphs/pipelines/__init__.py`
- `src/embodied_ai_architect/graphs/pipelines/perception.py`
- `src/embodied_ai_architect/graphs/pipelines/autonomy.py`
- `src/embodied_ai_architect/cli/pipeline.py`
- `tests/graphs/test_perception_pipeline.py`
- `tests/graphs/test_autonomy_pipeline.py`

**Modified Files:**
- `src/embodied_ai_architect/cli/__init__.py` - Register pipeline command
- `pyproject.toml` - Move langgraph from optional to default deps

---

## Design Decisions (Confirmed)

1. **Parallel execution**: Yes - perception and state estimation will run in parallel using LangGraph's native parallelism
2. **Execution modes**: Both batch (single frame) and streaming (video) supported from the start
3. **LLM integration**: Optional architect node for runtime optimization experiments
