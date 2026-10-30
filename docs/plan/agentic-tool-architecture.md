# Agentic Tool Architecture for Embodied AI Codesign

## Executive Summary

This document captures architectural decisions for building an Agentic AI system optimized for hardware/software codesign of complex embodied AI systems (drones, quadrupeds, bipeds, AMRs) and edge AI perception applications.

---

## 1. Tool Registration Patterns

### Current State (This Codebase)

Two distinct registration patterns exist:

1. **Orchestrator Pattern** (`orchestrator.py`)
   - Manual dictionary-based registration: `orchestrator.register_agent(agent)`
   - Hard-coded execution order in `process()` method
   - No validation—missing agents are silently skipped

2. **LLM Tool Pattern** (`llm/tools.py`)
   - Dual registration: definitions (JSON schema) + executors (callables)
   - Must be kept in sync manually
   - Runtime error if tool called but executor missing

### Key Gap

No mechanism to validate tool completeness before execution. The system discovers missing capabilities at runtime.

---

## 2. Tool Granularity Design Decision

### Options Analyzed

| Design | Speed | Accuracy | Robustness | LLM Cognitive Load |
|--------|-------|----------|------------|---------------------|
| Generic + args | Medium | Lower | Fragile | High |
| Task-specific by name | Fast (parallel) | High | High | Medium |
| Context-adaptive | Slow | Unpredictable | Low | Low |
| Do-everything | Slow | Noisy | Medium | Low |

### Recommendation: Dimension-Specific Tools

Create named tools per analytical dimension:

```python
analyze_accuracy(model, dataset, threshold?)
analyze_energy(model, hardware, power_budget?)
analyze_latency(model, hardware, target_ms?)
analyze_memory(model, hardware, memory_limit?)
analyze_cost(model, hardware, budget?)
```

**Rationale:**
- LLMs excel at semantic matching (goal → tool name)
- LLMs struggle with complex parameterization (goal → correct arg combinations)
- Parallel execution when dimensions are independent
- Partial results possible if some analyses fail

---

## 3. Tool Output Schema

### Principle: Pre-Digested Judgment

Tools should perform domain reasoning, not the LLM. The tool knows thresholds and interpretations; the LLM receives a verdict it can trust.

### Recommended Output Schema

```json
{
  "verdict": "PASS | FAIL | PARTIAL | UNKNOWN",
  "confidence": "high | medium | low",
  "summary": "One sentence: what was checked, what was found",
  "metric": {
    "name": "string",
    "measured": "number",
    "required": "number | null",
    "unit": "string",
    "margin": "+/- percentage from requirement"
  },
  "evidence": "Brief description of how this was determined",
  "suggestion": "If not PASS, actionable next step (or null)"
}
```

### Why This Works

| LLM Executor Need | Schema Feature |
|-------------------|----------------|
| Subtask complete? | `verdict` field—no parsing required |
| Can I trust it? | `confidence` field |
| Compose results? | Uniform schema across tools |
| Report to user? | `summary` field |
| Recover from failure? | `suggestion` field |

---

## 4. Domain Knowledge Architecture

### Options Evaluated

| Approach | Query Capability | Relationships | Remote Access | Best For |
|----------|------------------|---------------|---------------|----------|
| File tree (YAML/JSON) | Load & search | Implicit | Git sync | Static, structured facts |
| RDF/Knowledge Graph | SPARQL, inference | Explicit, typed | Yes | Rich relationships |
| Relational DB | SQL | Foreign keys | Yes | Tabular data |
| System prompt | Always available | None | N/A | Small, critical knowledge |
| Vector embeddings | Semantic similarity | None | Yes | Unstructured, fuzzy matching |

### Industry Trends (2025)

Research shows hybrid architectures outperforming single-approach systems:

- **Temporal Knowledge Graphs** (Zep/Graphiti): 18.5% higher accuracy, 90% lower latency vs vector-only
- **Hybrid Retrieval**: semantic embeddings + keyword search + graph traversal
- **Structured Memory Notes** (A-MEM): Zettelkasten-style linked notes with metadata

### Pure Vector RAG Limitations

- **Lost constraints**: top-k retrieval misses critical facts
- **Semantic drift**: matches on topic but wrong on specifics
- **Context dilution**: too many partially-relevant chunks

For hardware specs and requirements, these are fatal flaws. Exact facts need graph/relational lookup, not vector similarity.

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Tool Query Layer                      │
│   get_hardware_spec(hw, spec) → exact lookup            │
│   get_usecase_requirements(usecase) → structured list   │
│   find_similar_usecases(description) → semantic search  │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│  Graph Store │   │ Vector Index │   │  YAML/JSON Files │
│  (relations) │   │  (semantic)  │   │  (version ctrl)  │
└──────────────┘   └──────────────┘   └──────────────────┘
```

### Implementation Phases

**Phase 1**: YAML files as source of truth
```yaml
# hardware/jetson_nano.yaml
name: Jetson Nano
specs:
  compute_gflops: 472
  memory_gb: 4
  power_watts: 10
```

**Phase 2**: Load into lightweight graph (NetworkX or SQLite)
- Enables relational queries
- Supports relationship traversal

**Phase 3**: Add vector index for semantic queries (optional)
- Fuzzy use-case matching
- Embed descriptions for similarity search

---

## 5. Subtask Decomposition

### Core Question: Who Decomposes?

| Approach | Trade-offs |
|----------|------------|
| LLM-driven | Flexible but opaque, hard to validate |
| Template workflows | Predictable but rigid |
| Goal-oriented | Powerful but complex |
| Hybrid | Balance of flexibility and structure |

### Recommended Pattern

With dimension-specific tools, decomposition becomes:

```
User: "Can YOLOv8s run on Jetson Nano for drone obstacle avoidance?"

Decomposition:
1. Identify relevant dimensions → latency, energy, memory (from use case)
2. Map to tools → analyze_latency, analyze_energy, analyze_memory
3. Execute (parallel)
4. Synthesize: all PASS? → yes. any FAIL? → explain which and why
```

The decomposition focuses on **which dimensions matter**, not how to parameterize tools.

---

## 6. Key Design Principles

1. **Tool semantics over arguments**: Make selection the hard part (LLM handles well), not parameterization

2. **Verdict-first outputs**: Tools return judgments, not raw data for LLM to interpret

3. **Structured facts, graph lookup**: Don't use vector similarity for exact knowledge

4. **Knowledge outside prompts**: Tool descriptions in prompt, domain facts in knowledge store

5. **Parallel when independent**: Dimension analyses can run concurrently

---

## References

- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- [Zep: Temporal Knowledge Graph Architecture](https://arxiv.org/html/2501.13956v1)
- [Graphiti: Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [Comparing Memory Systems for LLM Agents](https://www.marktechpost.com/2025/11/10/comparing-memory-systems-for-llm-agents-vector-graph-and-event-logs/)
- [AWS: Persistent Memory for Agentic AI](https://aws.amazon.com/blogs/database/build-persistent-memory-for-agentic-ai-applications-with-mem0-open-source-amazon-elasticache-for-valkey-and-amazon-neptune-analytics/)

---

*Document created: December 2024*
*Status: Design Discussion*
