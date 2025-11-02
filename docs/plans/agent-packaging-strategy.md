# Agent Packaging Strategy

**Date**: 2025-11-02
**Status**: Design Proposal

## Overview

As agents become more complex and function-specific, they should be developed as independent packages/repositories. This allows for:
- Specialized development teams per agent
- Independent versioning and release cycles
- Complex dependencies isolated from core system
- Community contributions to specific agents
- Mix-and-match agent capabilities

## Package Architecture

### Core Package: `embodied-ai-architect`

**Purpose**: Provides the orchestration framework and agent interfaces

**Contains**:
- Orchestrator class
- BaseAgent abstract class
- AgentResult and message schemas
- Agent registry and discovery
- Workflow management

**Minimal dependencies**:
- pydantic (for data models)
- Basic Python stdlib

### Agent Packages: Independent Repositories

Each agent is a separate Python package following the naming convention:
`embodied-ai-architect-{agent-name}`

**Example agent packages**:
- `embodied-ai-architect-model-analyzer` - Model analysis capabilities
- `embodied-ai-architect-benchmark` - Benchmarking and profiling
- `embodied-ai-architect-hardware-profile` - Hardware knowledge base
- `embodied-ai-architect-transpiler-cpp` - C++ code generation
- `embodied-ai-architect-transpiler-rust` - Rust code generation
- `embodied-ai-architect-validator` - Accuracy/robustness validation

## Agent Package Structure

```
embodied-ai-architect-benchmark/
├── pyproject.toml
├── README.md
├── src/
│   └── embodied_ai_architect_benchmark/
│       ├── __init__.py
│       ├── agent.py              # Main BenchmarkAgent class
│       ├── backends/             # Execution backends
│       │   ├── __init__.py
│       │   ├── base.py           # Backend interface
│       │   ├── local_cpu.py
│       │   ├── local_gpu.py
│       │   └── remote.py
│       ├── profilers/            # Specialized profilers
│       │   ├── latency.py
│       │   ├── throughput.py
│       │   ├── memory.py
│       │   └── energy.py
│       └── utils/
├── tests/
└── examples/
```

## Backend Plugin Architecture

Agents that need to dispatch work (like benchmarking) should use a **backend plugin pattern**:

```python
# Backend interface
class BenchmarkBackend(ABC):
    @abstractmethod
    def execute_benchmark(self, model, config) -> BenchmarkResult:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

# Specific backends can be separate sub-packages
class LocalCPUBackend(BenchmarkBackend):
    """Simple local CPU execution"""
    pass

class RemoteClusterBackend(BenchmarkBackend):
    """Dispatch to Kubernetes cluster"""
    pass

class RobotBackend(BenchmarkBackend):
    """Deploy to physical robot for real-world testing"""
    pass
```

## Installation Patterns

### Basic Installation
```bash
pip install embodied-ai-architect
pip install embodied-ai-architect-model-analyzer
pip install embodied-ai-architect-benchmark
```

### With Extras
```bash
# Install benchmark agent with all backends
pip install embodied-ai-architect-benchmark[remote,robot,gpu]

# Install only core + local CPU benchmarking
pip install embodied-ai-architect-benchmark
```

### Monorepo Alternative (Development)
For development, you can use a monorepo structure:
```
embodied-ai-architect/
├── packages/
│   ├── core/
│   ├── agent-model-analyzer/
│   ├── agent-benchmark/
│   ├── agent-hardware-profile/
│   └── agent-transpiler-cpp/
```

## Agent Discovery & Registration

### Manual Registration
```python
from embodied_ai_architect import Orchestrator
from embodied_ai_architect_benchmark import BenchmarkAgent
from embodied_ai_architect_model_analyzer import ModelAnalyzerAgent

orchestrator = Orchestrator()
orchestrator.register_agent(ModelAnalyzerAgent())
orchestrator.register_agent(BenchmarkAgent(backend="local_cpu"))
```

### Auto-discovery (Future)
```python
# Agents register themselves via entry points in pyproject.toml
orchestrator = Orchestrator()
orchestrator.discover_agents()  # Finds all installed agent packages
```

## Agent Interface Contract

All agents must:
1. Inherit from `BaseAgent` (from core package)
2. Implement `execute(input_data: Dict) -> AgentResult`
3. Define clear input/output schemas using Pydantic
4. Provide `is_available()` method to check if dependencies are met
5. Include comprehensive tests

## Dependency Management

### Core Package Dependencies
- Minimal dependencies only
- No ML framework dependencies (PyTorch, TensorFlow, etc.)

### Agent Package Dependencies
- Can have heavy dependencies (PyTorch, CUDA, vendor SDKs)
- Must declare all dependencies in pyproject.toml
- Use extras for optional features

## Version Compatibility

Agents declare compatible core versions:
```toml
[project]
name = "embodied-ai-architect-benchmark"
dependencies = [
    "embodied-ai-architect>=0.1.0,<1.0.0"
]
```

## Benefits of This Architecture

1. **Modularity**: Install only what you need
2. **Scalability**: Agents can have complex internal architectures
3. **Independence**: Different teams can maintain different agents
4. **Flexibility**: Mix agents from different sources
5. **Testing**: Test agents independently
6. **Distribution**: Share agents via PyPI or private registries

## Example: Complex Benchmark Agent

The benchmark agent is a good example of complex functionality:

**Internal Architecture**:
- **Agent Core**: Coordinates benchmarking workflow
- **Backends**: Local, remote cluster, edge device, robot
- **Profilers**: Latency, throughput, memory, energy, thermal
- **Result Aggregators**: Statistical analysis, visualization
- **Remote Communication**: gRPC, REST APIs, ROS2 messages
- **Hardware Interfaces**: CUDA, ROCm, Intel oneAPI, vendor SDKs

This complexity justifies a separate repository with:
- Dedicated documentation
- Backend-specific examples
- Integration tests with real hardware
- CI/CD for multiple platforms

## Migration Path

**Phase 1**: Monorepo (current)
- All agents in single repo
- Rapid prototyping
- Easier to iterate on interfaces

**Phase 2**: Split critical agents
- Extract benchmark agent to separate repo
- Extract transpiler agents (complex dependencies)
- Keep simple agents in core

**Phase 3**: Full plugin ecosystem
- All agents as separate packages
- Auto-discovery mechanism
- Community agent marketplace
