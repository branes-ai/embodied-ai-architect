# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embodied AI Architect is a design environment for creating and evaluating autonomous agents. It provides:
- Model analysis and benchmarking across different hardware targets
- Hardware profiling with recommendations for edge/cloud deployment
- Multi-hardware benchmark execution (local CPU, remote SSH, Kubernetes)
- Report generation for model-to-hardware fit analysis

## Build & Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Install with optional dependencies
pip install -e ".[dev,remote,kubernetes]"

# Run tests
pytest tests/

# Run single test
pytest tests/test_file.py::test_function -v

# Linting and formatting
black src/ tests/ --line-length 100
ruff check src/ tests/

# Run CLI
embodied-ai --help
embodied-ai workflow run model.pt
embodied-ai analyze model.pt
embodied-ai benchmark model.pt --backend local
```

## Architecture

### Core System (`src/embodied_ai_architect/`)

**Orchestrator Pattern**: The `Orchestrator` class coordinates agent execution in a pipeline:
1. ModelAnalyzer → analyzes PyTorch model structure
2. HardwareProfile → recommends hardware based on model characteristics
3. Benchmark → measures actual performance on target backends
4. ReportSynthesis → generates HTML/JSON reports

**Agent System** (`agents/`):
- All agents extend `BaseAgent` and implement `execute(input_data) -> AgentResult`
- Agents are registered with the Orchestrator and executed in sequence
- Each agent produces an `AgentResult` with success status, data, and optional error

**Benchmark Backends** (`agents/benchmark/backends/`):
- `LocalCPUBackend`: Local CPU inference
- `RemoteSSHBackend`: Execute on remote machines via SSH (requires `paramiko`)
- `KubernetesBackend`: Distributed benchmarking on K8s (requires `kubernetes`)

**CLI** (`cli/`): Click-based CLI with subcommands:
- `workflow` - Run full analysis pipeline
- `analyze` - Model structure analysis only
- `benchmark` - Performance benchmarking
- `report` - View/manage reports
- `backends` - Manage benchmark backends
- `secrets` - Manage credentials

### Prototypes (`prototypes/`)

**drone_perception/**: Real-time perception pipeline for drones
- Sensors: monocular, stereo (RealSense, OAK-D), wide-angle, LiDAR
- Detection: YOLOv8 wrapper
- Tracking: ByteTrack with Kalman filtering
- Reasoning: trajectory prediction, collision detection, spatial analysis, behavior classification
- Scene graph: 3D state management with object persistence

```bash
# Run drone perception pipeline
cd prototypes/drone_perception
pip install -r requirements.txt
python examples/full_pipeline.py --video 0  # webcam
python examples/reasoning_pipeline.py --camera 0 --model s
```

**multi_rate_framework/**: Multi-rate control system using Zenoh pub/sub
- Components run at different frequencies (1Hz, 10Hz, 100Hz)
- `@control_loop` decorator for rate-specified execution

```bash
cd prototypes/multi_rate_framework
pip install eclipse-zenoh
python example_multirate.py
```

## Key Design Patterns

- **Pydantic models** for data validation (`AgentResult`, `WorkflowResult`)
- **Optional dependencies** with try/except imports (see `backends/__init__.py`)
- **Rich console** output for CLI
- **Jinja2 templates** for HTML report generation

## Code Style

- Line length: 100 characters
- Python target: 3.9+
- Use type hints
- Format with Black, lint with Ruff
