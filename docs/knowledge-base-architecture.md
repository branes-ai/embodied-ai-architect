# Knowledge Base Architecture for Embodied AI Architect

## Executive Summary

This document defines an **LLM-native** architecture for the knowledge base powering the Embodied AI Architect agent. The core principle: **text in context is the interface**. The LLM reasons directly over injected markdown/YAML content rather than querying databases through tool indirection.

## Design Principles

1. **Text is the API** - LLMs reason over text; databases add unnecessary indirection
2. **Context injection over tool calls** - Inject relevant content directly; tools only for computation
3. **Chunked for retrieval** - Content split into context-friendly units (~500-2000 tokens each)
4. **Semantic selection** - Vector similarity determines what to inject
5. **Computed on-demand** - Roofline analysis via tools; static knowledge via context

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM-NATIVE KNOWLEDGE BASE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  USER QUERY                                                      │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  RETRIEVAL LAYER (Vector Search)                        │    │
│  │  • Embed query                                          │    │
│  │  • Find relevant chunks (top-k by similarity)           │    │
│  │  • Respect token budget                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  CONTEXT ASSEMBLY                                        │    │
│  │  • Inject retrieved chunks as markdown                  │    │
│  │  • Add system context (constraint tiers, etc.)          │    │
│  │  • Format for LLM consumption                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LLM REASONING                                           │    │
│  │  • Reasons over injected context                        │    │
│  │  • Calls tools ONLY for computation (roofline, etc.)    │    │
│  │  • Generates response                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Content Tiers

### Tier 1: Structured Catalog (Markdown/YAML)

**Source**: embodied-schemas (135 YAML files)
**Format**: Markdown summaries + full YAML for detail
**Access**: Direct context injection via semantic retrieval

```
embodied-schemas/
├── data/
│   ├── hardware/           # 14 platform specs
│   ├── chips/              # 12 SoC specs
│   ├── gpus/               # 22 GPU specs
│   ├── cpus/               # 36 CPU specs
│   ├── npus/               # 4 NPU specs
│   ├── models/             # 10 ML model specs
│   ├── sensors/            # 4 sensor specs
│   ├── operators/          # 20 operator definitions
│   ├── architectures/      # 3 reference architectures
│   └── usecases/           # 4 use case templates
```

**Chunk Strategy**: One file = one chunk. Each chunk includes:
- Markdown summary (human-readable, ~200 tokens)
- Full YAML (machine-parseable, ~500-1500 tokens)

**Example chunk** (hardware/nvidia_jetson_orin_nano.md):
```markdown
# NVIDIA Jetson Orin Nano 8GB

**Category**: Edge GPU Module
**Vendor**: NVIDIA
**Target**: Drones, small robots, edge AI

## Key Specs
- **Compute**: 40 INT8 TOPS, 20 FP16 TFLOPS
- **Memory**: 8GB LPDDR5 @ 68 GB/s
- **Power**: 7-15W configurable TDP
- **Form Factor**: 70x45mm module

## Strengths
- Best perf/watt in module form factor
- TensorRT support for optimized inference
- Mature software ecosystem (JetPack)

## Limitations
- Memory bandwidth limited for large models
- Requires active cooling at 15W

## Compatible Models
- YOLOv8n/s (INT8): 30+ FPS
- MiDaS small: 15+ FPS
- ByteTrack: 60+ FPS

---
<details>
<summary>Full YAML Specification</summary>

```yaml
id: nvidia_jetson_orin_nano_8gb
name: NVIDIA Jetson Orin Nano 8GB
vendor: nvidia
category: gpu
# ... full spec
```
</details>
```

### Tier 2: Analytical Data (Computed via Tools)

**Source**: graphs library
**Format**: Tool results returned as formatted text
**Access**: Tool calls (only tier requiring tools for data access)

This is the **only tier that uses tools** because:
- Roofline analysis requires computation over model + hardware
- Results depend on query parameters (batch size, precision)
- Cannot be pre-computed for all combinations

**Tools**:
```python
# Returns formatted text, not raw data
analyze_roofline(model_id, hardware_id, precision) -> str
"""
## Roofline Analysis: YOLOv8n on Jetson Orin Nano

**Bottleneck**: Memory-bound (62% of ops)
**Arithmetic Intensity**: 12.4 FLOPs/byte
**Utilization**: 34% compute, 78% bandwidth

### Interpretation
This model is memory-bound on this hardware. The GPU spends
most time waiting for data. Consider:
- INT8 quantization to reduce memory traffic
- Operator fusion to reduce intermediate tensors
- Smaller batch size if latency-critical
"""

check_constraint(model_id, hardware_id, metric, threshold) -> str
"""
## Constraint Check: Latency ≤ 33ms

**Verdict**: PASS ✓
**Measured**: 28.4ms (14% margin)
**Confidence**: High (based on roofline model)

The model meets the 30 FPS requirement on this hardware.
"""
```

### Tier 3: Experiential Knowledge (Markdown Articles)

**Source**: Authored content (deployment guides, troubleshooting, etc.)
**Format**: Pure markdown
**Access**: Direct context injection via semantic retrieval

**Directory Structure**:
```
knowledge/
├── deployment/
│   ├── yolo-jetson-tensorrt.md
│   ├── yolo-coral-edgetpu.md
│   ├── yolo-hailo8.md
│   └── ...
├── optimization/
│   ├── int8-quantization-guide.md
│   ├── operator-fusion-tensorrt.md
│   └── ...
├── troubleshooting/
│   ├── latency-too-high.md
│   ├── int8-accuracy-drop.md
│   ├── oom-on-edge.md
│   └── ...
├── best-practices/
│   ├── model-selection-edge.md
│   ├── power-budgeting.md
│   └── ...
└── case-studies/
    ├── drone-obstacle-avoidance.md
    ├── warehouse-amr.md
    └── ...
```

**Article Template**:
```markdown
---
id: deploy-yolo-jetson-tensorrt
title: Deploying YOLOv8 on Jetson Orin with TensorRT
category: deployment
hardware_tags: [jetson-orin-nano, jetson-orin-nx, jetson-orin-agx]
model_tags: [yolov8n, yolov8s, yolov8m]
use_case_tags: [detection, real-time, edge]
skill_level: intermediate
---

# Deploying YOLOv8 on Jetson Orin with TensorRT

This guide covers converting YOLOv8 to TensorRT and optimizing
for real-time inference on NVIDIA Jetson Orin platforms.

## Prerequisites
- JetPack 5.1+ installed
- YOLOv8 model exported to ONNX
- At least 4GB free storage

## Steps

### 1. Export to ONNX
```bash
yolo export model=yolov8n.pt format=onnx opset=17
```

### 2. Convert to TensorRT
...

## Common Issues

### "Unsupported operator" error
TensorRT doesn't support all ONNX operators. Solution:
- Use opset 17 (best compatibility)
- Check operator support matrix
- Consider custom plugin for unsupported ops

## Performance Results

| Model | Precision | Latency | Throughput |
|-------|-----------|---------|------------|
| YOLOv8n | FP16 | 8.2ms | 122 FPS |
| YOLOv8n | INT8 | 5.1ms | 196 FPS |
| YOLOv8s | FP16 | 14.3ms | 70 FPS |
```

---

## Retrieval System

### Vector Index

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, fast)
**Index**: FAISS or Qdrant (in-memory for <10K chunks)

**What gets embedded**:
- Chunk title + summary + tags (not full content)
- Keeps embeddings focused on semantic meaning

**Retrieval Flow**:
```python
def retrieve_context(query: str, token_budget: int = 8000) -> str:
    """Retrieve relevant chunks for context injection."""

    # 1. Embed query
    query_embedding = embed(query)

    # 2. Find similar chunks
    chunks = vector_index.search(query_embedding, top_k=20)

    # 3. Select within token budget
    selected = []
    tokens_used = 0
    for chunk in chunks:
        if tokens_used + chunk.tokens <= token_budget:
            selected.append(chunk)
            tokens_used += chunk.tokens

    # 4. Format as markdown context
    return format_context(selected)
```

### Context Assembly

```python
def assemble_context(query: str, retrieved_chunks: list[Chunk]) -> str:
    """Assemble context for LLM injection."""

    context = """
## Relevant Knowledge

The following information has been retrieved based on your query.
Use this context to answer the user's question.

"""

    # Group by type
    hardware = [c for c in retrieved_chunks if c.type == "hardware"]
    models = [c for c in retrieved_chunks if c.type == "model"]
    guides = [c for c in retrieved_chunks if c.type == "knowledge"]

    if hardware:
        context += "### Hardware Specifications\n\n"
        for chunk in hardware:
            context += chunk.content + "\n\n"

    if models:
        context += "### Model Information\n\n"
        for chunk in models:
            context += chunk.content + "\n\n"

    if guides:
        context += "### Guides & Documentation\n\n"
        for chunk in guides:
            context += chunk.content + "\n\n"

    return context
```

---

## Token Budget Management

**Typical context allocation**:
```
Total budget: 100K tokens (Claude)
├── System prompt: 2K
├── Conversation history: 10K
├── Retrieved context: 8K (default, expandable)
├── Tool results: 2K
└── Response generation: 78K available
```

**Chunk sizing guidelines**:
| Content Type | Target Size | Rationale |
|--------------|-------------|-----------|
| Hardware summary | 200-400 tokens | Quick reference |
| Hardware full | 800-1500 tokens | Complete spec |
| Model summary | 200-400 tokens | Quick reference |
| Knowledge article | 1000-2000 tokens | Self-contained guide |
| Troubleshooting | 500-1000 tokens | Focused solution |

---

## Content Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Convert existing catalog to LLM-ready markdown chunks.

| Task | Source | Output |
|------|--------|--------|
| Generate hardware summaries | embodied-schemas YAML | 50+ markdown chunks |
| Generate model summaries | embodied-schemas YAML | 10+ markdown chunks |
| Generate sensor summaries | embodied-schemas YAML | 4 markdown chunks |
| Generate operator summaries | embodied-schemas YAML | 20 markdown chunks |
| Generate use case summaries | embodied-schemas YAML | 4 markdown chunks |
| Build vector index | All chunks | FAISS index |
| Implement retrieval | New code | retrieve_context() |

**Deliverable**: Query "what hardware for drone obstacle avoidance" returns relevant Jetson, Coral, Hailo specs + drone use case template.

### Phase 2: Analytical Integration (Week 3-4)

**Goal**: Connect graphs roofline as text-returning tools.

| Task | Output |
|------|--------|
| Wrap UnifiedAnalyzer | analyze_roofline() tool |
| Wrap constraint checker | check_constraint() tool |
| Format results as markdown | Human-readable analysis text |
| Add interpretation logic | Recommendations in plain English |

**Deliverable**: "Will YOLOv8n run at 30fps on Jetson Orin Nano?" returns formatted analysis with verdict.

### Phase 3: Knowledge Articles (Week 5-8)

**Goal**: Author experiential content for common scenarios.

#### Deployment Guides (Priority Order)

| # | Article | Hardware | Model |
|---|---------|----------|-------|
| 1 | YOLOv8 + TensorRT on Jetson | Jetson Orin family | YOLOv8n/s/m |
| 2 | YOLOv8 on Coral Edge TPU | Coral USB/PCIe | YOLOv8n |
| 3 | YOLOv8 on Hailo-8 | Hailo-8 M.2/PCIe | YOLOv8n/s |
| 4 | ONNX Runtime on Intel NPU | Meteor Lake | YOLOv8n |
| 5 | TFLite on Raspberry Pi | RPi 5 | YOLOv8n |
| 6 | Multi-model pipeline | Jetson Orin | YOLO + tracker |
| 7 | Depth estimation | Jetson Orin | MiDaS |

#### Optimization Recipes

| # | Article | Technique |
|---|---------|-----------|
| 1 | INT8 quantization with calibration | Post-training quantization |
| 2 | TensorRT optimization guide | Layer fusion, precision |
| 3 | Reducing memory footprint | Activation checkpointing |
| 4 | Batch size tuning | Latency vs throughput |

#### Troubleshooting

| # | Article | Symptom |
|---|---------|---------|
| 1 | Model too slow on edge | High latency diagnosis |
| 2 | INT8 accuracy degradation | Calibration issues |
| 3 | Out of memory errors | Memory optimization |
| 4 | TensorRT build failures | Operator compatibility |

### Phase 4: Benchmark Data (Week 9-10)

**Goal**: Add measured performance data as markdown tables.

**Format** (embedded in hardware/model chunks):
```markdown
## Benchmark Results

Measured on JetPack 5.1.2, TensorRT 8.5.2, batch=1

| Model | Precision | Latency (ms) | FPS | Power (W) |
|-------|-----------|--------------|-----|-----------|
| YOLOv8n | FP16 | 8.2 | 122 | 11.2 |
| YOLOv8n | INT8 | 5.1 | 196 | 9.8 |
| YOLOv8s | FP16 | 14.3 | 70 | 12.4 |
| YOLOv8s | INT8 | 8.8 | 114 | 10.6 |

*Conditions: 640x640 input, warmup=50, iterations=1000*
```

---

## File Organization

```
branes_mcp/
├── knowledge/                    # All LLM-readable content
│   ├── catalog/                  # Generated from embodied-schemas
│   │   ├── hardware/
│   │   │   ├── nvidia-jetson-orin-nano.md
│   │   │   ├── nvidia-jetson-orin-nx.md
│   │   │   ├── google-coral-edge-tpu.md
│   │   │   └── ...
│   │   ├── models/
│   │   │   ├── yolov8n.md
│   │   │   ├── yolov8s.md
│   │   │   └── ...
│   │   ├── sensors/
│   │   ├── operators/
│   │   └── usecases/
│   │
│   ├── guides/                   # Authored content
│   │   ├── deployment/
│   │   ├── optimization/
│   │   ├── troubleshooting/
│   │   └── best-practices/
│   │
│   └── reference/                # Static reference material
│       ├── constraint-tiers.md
│       ├── precision-formats.md
│       └── glossary.md
│
├── src/branes_mcp/
│   ├── retrieval/
│   │   ├── index.py             # Vector index management
│   │   ├── chunker.py           # Content chunking
│   │   └── retriever.py         # Retrieval logic
│   │
│   ├── tools/
│   │   ├── roofline.py          # analyze_roofline() - calls graphs
│   │   └── constraints.py       # check_constraint() - calls graphs
│   │
│   └── context/
│       └── assembler.py         # Context assembly for LLM
│
└── scripts/
    └── generate_catalog.py      # YAML → Markdown conversion
```

---

## When to Use Tools vs Context

| Query Type | Method | Rationale |
|------------|--------|-----------|
| "What is Jetson Orin Nano?" | Context injection | Static fact lookup |
| "Compare Jetson vs Coral" | Context injection | Both specs in context |
| "Best hardware for 30fps detection" | Context injection | Filter in LLM reasoning |
| "Will YOLOv8n hit 30fps on Orin?" | **Tool call** | Requires roofline computation |
| "How to deploy YOLO on Jetson" | Context injection | Retrieve deployment guide |
| "Why is my model slow?" | Context + Tool | Guide for diagnosis + roofline analysis |

**Rule**: Use tools only when computation is required. Everything else is context.

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Retrieval relevance | >80% relevant chunks | Human evaluation |
| Context efficiency | <8K tokens for typical query | Token counting |
| Tool call reduction | <20% queries need tools | Usage analytics |
| Response quality | Answers without "I don't know" | User feedback |
| Latency (retrieval) | <200ms | Performance monitoring |

---

## Next Steps

1. **Write generate_catalog.py** - Convert embodied-schemas YAML to markdown chunks
2. **Build vector index** - Embed all chunks, create FAISS index
3. **Implement retriever** - retrieve_context() function
4. **Wrap graphs tools** - Text-formatted roofline analysis
5. **Author first 5 deployment guides** - Start with Jetson + YOLO
6. **Test end-to-end** - Query → retrieval → context → response
