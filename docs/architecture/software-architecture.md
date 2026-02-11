# Embodied AI Architect Software Architecture

## System Overview

The diagram below shows the top level architecture.

User → Orchestrator → Agents → (LLM/Bench/Deploy/Schemas)

---
Diagram A: High-Level System Context`
---
```mermaid
graph TB
    User["User<br/>(CLI/Chat)"]
    Orchestrator["Orchestrator<br/>(Workflow Hub)"]
    
    subgraph "Agent Tier"
        Agents["5 Specialized Agents"]
    end
    
    subgraph "LLM Tier"
        LLM["Claude LLM<br/>+ Tools"]
    end
    
    subgraph "Execution Tier"
        Bench["Benchmarking<br/>Infrastructure"]
        Deploy["Deployment<br/>Targets"]
    end
    
    subgraph "Data Tier"
        Schemas["embodied-schemas<br/>(Shared Models)"]
    end
    
    User -->|Commands| Orchestrator
    Orchestrator -->|Dispatch| Agents
    Agents -->|Query/Execute| LLM
    Agents -->|Measure| Bench
    Agents -->|Compile| Deploy
    Agents -->|Read| Schemas
    LLM -->|Reference| Schemas
    
    style Orchestrator fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#000
    style Agents fill:#4ecdc4,stroke:#1098ad,color:#000
    style LLM fill:#ffe66d,stroke:#f0a500,color:#000
    style Bench fill:#95e1d3,stroke:#2d9b8f,color:#000
    style Deploy fill:#c7ceea,stroke:#9775fa,color:#000
    style Schemas fill:#ffc9c9,stroke:#fa5252,color:#000
```


## Component Responsibilities
See: `Diagram B: Agent Responsibilities`


## Agent Responsibilities

Each agent has a single responsibility. We have a collection of agents that follow this data dependency chain.
```mermaid
graph LR
    subgraph "Model Analysis"
        MA["ModelAnalyzerAgent<br/>- Parse DNN architectures<br/>- Extract operator graphs<br/>- Validate against schemas"]
    end
    
    subgraph "Hardware Profiling"
        HP["HardwareProfileAgent<br/>- Query hardware DB<br/>- Match constraints<br/>- Recommend targets"]
    end
    
    subgraph "Benchmarking"
        BA["BenchmarkAgent<br/>- Execute on backends<br/>- Collect metrics<br/>- Measure power"]
    end
    
    subgraph "Deployment"
        DA["DeploymentAgent<br/>- Compile to targets<br/>- Quantize models<br/>- Generate binaries"]
    end
    
    subgraph "Reporting"
        RA["ReportSynthesisAgent<br/>- Aggregate results<br/>- Generate artifacts<br/>- Format output"]
    end
    
    MA -->|"Output: Model Graph"| BA
    HP -->|"Constraints"| BA
    BA -->|"Perf Data"| RA
    DA -->|"Binary Artifacts"| RA
    
    style MA fill:#a8dadc,color:#000
    style HP fill:#f1faee,color:#000
    style BA fill:#e63946,color:#000
    style DA fill:#f4a261,color:#000
    style RA fill:#2a9d8f,color:#000
```


## Execution Flow
1. User submits query via CLI or chat
2. Orchestrator dispatches to relevant agents
3. Agents call LLM tools for decision-making
4. Tools query `embodied-schemas` and run benchmarks
5. Results aggregated into report

Each target provides a concrete implementation.
```mermaid
graph TD
    Base["DeploymentTarget<br/>Abstract Base<br/>- Model loading<br/>- Inference engine<br/>- Metrics collection"]
    
    OpenVINO["OpenVINO<br/>Intel x86/ARM<br/>- FP32/FP16/INT8<br/>- CPU/GPU optimization"]
    
    Coral["Coral Edge TPU<br/>Google<br/>- INT8 quantized<br/>- USB/PCIe"]
    
    Jetson["Jetson Orin<br/>NVIDIA ARM<br/>- CUDA/TensorRT<br/>- 8-16GB VRAM"]
    
    KPU["Stillwater KPU<br/>Custom ISA<br/>- Proprietary compiler<br/>- Low power"]
    
    NVDLA["NVDLA<br/>Open Standard<br/>- DLA + Falcon"]
    
    Base -->|Implements| OpenVINO
    Base -->|Implements| Coral
    Base -->|Implements| Jetson
    Base -->|Implements| KPU
    Base -->|Implements| NVDLA
    
    style Base fill:#ffb3ba,stroke:#c92a2a,stroke-width:2px,color:#000
    style OpenVINO fill:#bae1ff,color:#000
    style Coral fill:#ffffba,color:#000
    style Jetson fill:#baffc9,color:#000
    style KPU fill:#e0bbff,color:#000
    style NVDLA fill:#ffc0cb,color:#000
```

## Adding a New Deployment Target
1. Subclass `DeploymentTarget` (Diagram C)
2. Implement: `load()`, `infer()`, `cleanup()`
3. Register in agent configuration


## Benchmarking Pipeline
See: `Diagram D: Benchmarking Data Flow`

---
Diagram D: Benchmarking Data Flow
---
```mermaid
graph LR
    Model["DNN Model<br/>(ONNX/TensorFlow)"]
    Operators["Operator Library<br/>YOLO, ByteTrack<br/>EKF, PID Control"]
    Graph["Graph Pipeline<br/>(LangGraph)"]
    
    Runner["ArchitectureRunner<br/>- Load model<br/>- Instantiate ops<br/>- Execute"]
    
    subgraph "Backends"
        LocalCPU["LocalCPU<br/>Direct execution"]
        RemoteSSH["RemoteSSH<br/>Deploy to edge device"]
        K8s["Kubernetes<br/>Distributed"]
    end
    
    PowerMon["PowerMonitor<br/>- RAPL counters<br/>- AMD SMU<br/>- GPIO sampling"]
    
    Metrics["Performance Report<br/>- Latency (ms)<br/>- Throughput (FPS)<br/>- Energy (J)<br/>- Memory (MB)"]
    
    Model -->|"Parsed by"| Runner
    Operators -->|"Used by"| Graph
    Graph -->|"Executed by"| Runner
    Runner -->|"Dispatch to"| LocalCPU
    Runner -->|"Dispatch to"| RemoteSSH
    Runner -->|"Dispatch to"| K8s
    Runner -->|"Monitored by"| PowerMon
    LocalCPU -->|"Collect"| Metrics
    RemoteSSH -->|"Collect"| Metrics
    K8s -->|"Collect"| Metrics
    PowerMon -->|"Feed"| Metrics
    
    style Runner fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px,color:#000
    style Metrics fill:#51cf66,stroke:#2b8a3e,stroke-width:2px,color:#000
    style PowerMon fill:#ffd43b,stroke:#cc8800,color:#000
```

## Adding a New Tool
1. Define in `llm/tools.py`
2. Register with LLMClient
3. Update agent to call it


## LLM Integration & Tool Ecosystem

The diagram shows the tool calling pattern. The LLM is orchestrating the tool calls to gather and resolve design information.
```mermaid
graph TB
    User["User Query<br/>(Multimodal)"]
    
    LLMClient["LLMClient<br/>Claude API<br/>- Streaming<br/>- Tool calling"]
    
    subgraph "Tool Registry"
        CoreTools["Core Tools<br/>- analyze_model<br/>- query_hardware<br/>- run_benchmark<br/>- deploy_model"]
        GraphTools["Graph Tools<br/>(Optional)<br/>- visualize_dag<br/>- optimize_graph"]
        ArchTools["Architecture Tools<br/>- query_schemas<br/>- recommend_target"]
    end
    
    Schemas["embodied-schemas<br/>Hardware catalog<br/>Model registry"]
    
    Agents["Agent Implementations<br/>(Consume tool outputs)"]
    
    Output["Report<br/>JSON/Markdown"]
    
    User -->|Query| LLMClient
    LLMClient -->|"Call"| CoreTools
    LLMClient -->|"Call"| GraphTools
    LLMClient -->|"Call"| ArchTools
    CoreTools -->|Read| Schemas
    ArchTools -->|Read| Schemas
    LLMClient -->|Direct control| Agents
    Agents -->|Results| Output
    
    style LLMClient fill:#ffe66d,stroke:#f0a500,stroke-width:2px,color:#000
    style CoreTools fill:#4ecdc4,color:#000
    style GraphTools fill:#95e1d3,color:#000
    style ArchTools fill:#b3a2d6,color:#000
    style Schemas fill:#ffc9c9,stroke:#fa5252,stroke-width:2px,color:#000
    style Output fill:#51cf66,color:#000
```

## All together
```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI["CLI Commands<br/>cli/commands/"]
        Chat["Interactive Chat<br/>llm/agent.py"]
    end

    subgraph "Orchestration Layer"
        Orchestrator["Orchestrator<br/>orchestrator.py<br/>(Workflow Coordination)"]
    end

    subgraph "Agent Layer"
        ModelAnalyzer["ModelAnalyzerAgent<br/>agents/model_analyzer/"]
        HardwareProfile["HardwareProfileAgent<br/>agents/hardware_profile/"]
        Benchmark["BenchmarkAgent<br/>agents/benchmark/"]
        Deployment["DeploymentAgent<br/>agents/deployment/"]
        ReportSynthesis["ReportSynthesisAgent<br/>agents/report_synthesis/"]
    end

    subgraph "LLM Integration"
        LLMClient["LLMClient<br/>llm/client.py<br/>(Anthropic API)"]
        Tools["Tool Definitions<br/>llm/tools.py"]
        GraphsTools["Graphs Tools<br/>llm/graphs_tools.py<br/>(Optional)"]
        ArchTools["Architecture Tools<br/>llm/architecture_tools.py"]
    end

    subgraph "Deployment Targets"
        OpenVINO["OpenVINO Target"]
        Coral["Coral Edge TPU"]
        Jetson["Jetson Target"]
        KPU["Stillwater KPU"]
        NVDLA["NVDLA Target"]
        Base["DeploymentTarget<br/>Base Class"]
    end

    subgraph "Benchmarking Infrastructure"
        ArchRunner["ArchitectureRunner<br/>benchmark/runner.py"]
        LocalCPU["LocalCPUBackend"]
        RemoteSSH["RemoteSSHBackend"]
        Kubernetes["KubernetesBackend"]
        PowerMonitor["PowerMonitor<br/>benchmark/power.py<br/>(RAPL, SMU)"]
    end

    subgraph "Operators"
        OperatorBase["Operator Base<br/>operators/base.py"]
        YOLOOp["YOLOv8 ONNX<br/>operators/perception/"]
        ByteTrackOp["ByteTrack<br/>operators/perception/"]
        EKFOp["EKF 6-DoF<br/>operators/state_estimation/"]
        ControlOp["PID Controller<br/>operators/control/"]
    end

    subgraph "Graph Pipelines"
        PerceptionGraph["Perception Pipeline<br/>graphs/pipelines/"]
        AutonomyGraph["Autonomy Pipeline<br/>graphs/pipelines/"]
    end

    subgraph "Data & Schemas"
        EmbodiedSchemas["embodied-schemas<br/>(Shared Package)<br/>Registry, Models, Hardware"]
        HardwareKB["Hardware Knowledge Base<br/>agents/hardware_profile/"]
    end

    subgraph "Output & Reporting"
        Reports["Report Synthesis<br/>reports/"]
        JSONOutput["JSON Results<br/>Validation Reports"]
    end

    %% User interactions
    CLI -->|"Commands"| Orchestrator
    Chat -->|"LLM Queries"| LLMClient

    %% Orchestrator coordinates agents
    Orchestrator -->|"Dispatch"| ModelAnalyzer
    Orchestrator -->|"Dispatch"| HardwareProfile
    Orchestrator -->|"Dispatch"| Benchmark
    Orchestrator -->|"Dispatch"| Deployment
    Orchestrator -->|"Dispatch"| ReportSynthesis

    %% LLM Integration
    LLMClient -->|"Uses"| Tools
    Tools -->|"Calls"| GraphsTools
    Tools -->|"Calls"| ArchTools
    ArchTools -->|"Reads"| EmbodiedSchemas

    %% Agents use LLM
    Chat -->|"Leverage"| Tools
    Chat -->|"Queries"| HardwareProfile

    %% Agent implementations
    ModelAnalyzer -->|"Analyzes"| EmbodiedSchemas
    HardwareProfile -->|"Provides"| HardwareKB
    Benchmark -->|"Uses"| ArchRunner
    Deployment -->|"Uses"| Base
    ReportSynthesis -->|"Generates"| Reports

    %% Deployment target hierarchy
    Base -->|"Implements"| OpenVINO
    Base -->|"Implements"| Coral
    Base -->|"Implements"| Jetson
    Base -->|"Implements"| KPU
    Base -->|"Implements"| NVDLA

    %% Benchmarking
    ArchRunner -->|"Loads"| EmbodiedSchemas
    ArchRunner -->|"Instantiates"| OperatorBase
    ArchRunner -->|"Measures Power"| PowerMonitor
    Benchmark -->|"Dispatches to"| LocalCPU
    Benchmark -->|"Dispatches to"| RemoteSSH
    Benchmark -->|"Dispatches to"| Kubernetes

    %% Operators
    OperatorBase -->|"Implements"| YOLOOp
    OperatorBase -->|"Implements"| ByteTrackOp
    OperatorBase -->|"Implements"| EKFOp
    OperatorBase -->|"Implements"| ControlOp

    %% Graph Pipelines
    PerceptionGraph -->|"Uses"| YOLOOp
    PerceptionGraph -->|"Uses"| ByteTrackOp
    AutonomyGraph -->|"Orchestrates"| PerceptionGraph
    AutonomyGraph -->|"Uses"| EKFOp
    AutonomyGraph -->|"Uses"| ControlOp

    %% Output
    ReportSynthesis -->|"Produces"| JSONOutput
    Benchmark -->|"Produces"| JSONOutput

    %% Shared data
    HardwareProfile -->|"References"| EmbodiedSchemas
    Benchmark -->|"References"| EmbodiedSchemas

    style Orchestrator fill:#ff6b6b,color:#000
    style LLMClient fill:#4ecdc4,color:#000
    style EmbodiedSchemas fill:#ffe66d,color:#000
    style ArchRunner fill:#95e1d3,color:#000
    style Base fill:#c7ceea,color:#000
```