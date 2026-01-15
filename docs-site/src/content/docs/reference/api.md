---
title: Python API Reference
description: Reference for the Python API.
---

## Agents

### ModelAnalyzerAgent

Analyzes PyTorch model structure.

```python
from embodied_ai_architect.agents import ModelAnalyzerAgent

agent = ModelAnalyzerAgent()
result = agent.execute({"model": "path/to/model.pt"})

if result.success:
    print(f"Parameters: {result.data['parameters']}")
    print(f"FLOPs: {result.data['flops']}")
```

### HardwareProfileAgent

Recommends hardware based on model analysis.

```python
from embodied_ai_architect.agents import HardwareProfileAgent

agent = HardwareProfileAgent()
result = agent.execute({
    "model_analysis": model_analysis,
    "constraints": {
        "power_watts": 15,
        "latency_ms": 33,
    },
    "top_n": 5,
})
```

### BenchmarkAgent

Benchmarks model on various backends.

```python
from embodied_ai_architect.agents import BenchmarkAgent

agent = BenchmarkAgent()
result = agent.execute({
    "model": "model.pt",
    "backends": ["local"],
    "iterations": 100,
})
```

### DeploymentAgent

Deploys model to target hardware.

```python
from embodied_ai_architect.agents import DeploymentAgent

agent = DeploymentAgent()
result = agent.execute({
    "model": "model.pt",
    "target": "jetson",
    "precision": "int8",
    "input_shape": (1, 3, 640, 640),
    "calibration_data": "./calibration/",
})
```

## LLM Integration

### ArchitectAgent

Interactive agent with tool use.

```python
from embodied_ai_architect.llm import LLMClient, ArchitectAgent

llm = LLMClient()
agent = ArchitectAgent(llm=llm)

response = agent.run("Can YOLOv8n run at 30fps on Jetson Orin Nano?")
print(response)
```

## Orchestrator

### Orchestrator

Coordinates agent execution.

```python
from embodied_ai_architect import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.run_workflow(
    model_path="model.pt",
    target_hardware="jetson-orin-nano",
)
```
