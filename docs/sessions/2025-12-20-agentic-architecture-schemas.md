# Session Log: Agentic Architecture and Shared Schemas

**Date**: December 20, 2025
**Focus**: Designing the agentic tool architecture and creating the shared schema repository

---

## Summary

This session established the foundational architecture for the Embodied AI codesign agent system, including tool design patterns, knowledge base architecture, and a new shared schema repository that will be used by both `graphs` and `embodied-ai-architect`.

---

## Key Decisions Made

### 1. Tool Granularity: Dimension-Specific by Name

**Decision**: Create named tools per analytical dimension rather than generic tools with complex arguments.

```
analyze_latency(model, hardware, requirement?)
analyze_power(model, hardware, budget?)
analyze_memory(model, hardware, limit?)
analyze_accuracy(model, dataset, threshold?)
analyze_cost(model, hardware, budget?)
```

**Rationale**:
- LLMs excel at semantic matching (goal → tool name)
- LLMs struggle with complex parameterization
- Enables parallel execution when dimensions are independent
- Partial results possible if some analyses fail

### 2. Tool Output Schema: Verdict-First

**Decision**: Tools return pre-digested judgments, not raw data for the LLM to interpret.

```json
{
  "verdict": "PASS | FAIL | PARTIAL | UNKNOWN",
  "confidence": "high | medium | low",
  "summary": "One sentence description",
  "metric": {"name": "latency", "measured": 45.2, "required": 33.3, "unit": "ms"},
  "evidence": "Measured over 100 inference runs",
  "suggestion": "Consider YOLOv8n (~22ms) or TensorRT optimization"
}
```

**Rationale**: The tool knows domain thresholds; the LLM receives a verdict it can trust.

### 3. Knowledge Architecture: Hybrid Storage

**Decision**: YAML files as source of truth with optional graph/vector layers.

| Query Type | Backend |
|------------|---------|
| Exact fact lookup | YAML/Graph |
| Requirement inference | Graph |
| Fuzzy matching | Vector (optional) |

**Rationale**: Structured facts need exact lookup, not vector similarity.

### 4. Shared Schema Repository

**Decision**: Create `branes-ai/embodied-schemas` as a separate package imported by both repos.

**Dependency flow**:
```
embodied-schemas (schemas + facts)
       ↑              ↑
       │              │
   graphs      embodied-ai-architect
```

### 5. Data Split: graphs vs embodied-schemas

**Decision**: Datasheet specs → embodied-schemas, roofline/calibration → graphs

| embodied-schemas | graphs |
|------------------|--------|
| Vendor specs (memory, TDP, cores) | ops_per_clock |
| Power profiles | theoretical_peaks |
| Physical/environmental specs | Calibration data |
| Interface specs | Operation profiles |

---

## Planning Documents Created

### 1. docs/plans/agentic-tool-architecture.md
- Tool registration patterns analysis
- Tool granularity decision (dimension-specific)
- Output schema design (verdict-first)
- Knowledge base architecture options
- Industry research (A-MEM, Zep/Graphiti, Mem0)

### 2. docs/plans/embodied-ai-codesign-subtasks.md
- 70+ enumerated subtasks in 7 categories
- Knowledge Base (21 subtasks)
- Analysis Tools (19 subtasks)
- Recommendation Tools (12 subtasks)
- Synthesis Tools (11 subtasks)
- Decomposition & Planning (14 subtasks)
- Validation (10 subtasks)
- Phased implementation plan

### 3. docs/plans/knowledge-base-schema.md
- Complete Pydantic schema designs
- Hardware: physical, environmental, power, interfaces
- Models: architecture, variants, accuracy
- Sensors: cameras, depth, LiDAR
- Use cases: constraints, success criteria
- Benchmarks: verdict-first output
- Directory structure and query interface

### 4. docs/plans/shared-schema-repo-architecture.md
- Repository decision and rationale
- Package configuration
- API design patterns
- Data split with graphs/hardware_registry
- Migration strategy
- Field mapping table

---

## Implementation: branes-ai/embodied-schemas

Created and committed the new shared schema repository:

### Repository Structure
```
embodied-schemas/
├── src/embodied_schemas/
│   ├── __init__.py      # Package exports
│   ├── hardware.py      # HardwareEntry, HardwareCapability, 20+ models
│   ├── models.py        # ModelEntry, ModelVariant, accuracy benchmarks
│   ├── sensors.py       # SensorEntry, CameraSpec, DepthSpec, LidarSpec
│   ├── usecases.py      # UseCaseEntry, Constraint, SuccessCriterion
│   ├── benchmarks.py    # BenchmarkResult, LatencyMetrics, verdict format
│   ├── constraints.py   # LatencyTier, PowerClass, tier utilities
│   ├── loaders.py       # YAML loading with Pydantic validation
│   ├── registry.py      # Unified data access API
│   └── data/            # YAML data directories (32 subdirectories)
├── tests/
│   └── test_schemas.py  # 15 passing tests
├── docs/
│   └── sessions/
│       └── 2024-12-20-initial-setup.md
├── CHANGELOG.md
├── pyproject.toml
├── README.md
└── LICENSE
```

### Key Schema Highlights

**HardwareEntry** - Extended for embodied AI:
- `PhysicalSpec`: weight, dimensions, form factor
- `EnvironmentalSpec`: temp range, IP rating, vibration/shock
- `InterfaceSpec`: CSI, USB, PCIe, CAN bus counts
- `PowerSpec`: Multiple power modes with frequencies

**ModelEntry** - Perception-focused:
- Architecture: backbone, neck, head, params, FLOPs
- Variants: fp32, fp16, int8 with accuracy delta
- Accuracy benchmarks per dataset

**UseCaseEntry** - Constraint templates:
- Hard/soft constraints with criticality levels
- Success criteria for validation
- Recommended hardware/model/sensor configs

**BenchmarkResult** - Verdict-first output:
- Latency, power, memory, thermal metrics
- Verdict + confidence + suggestion format

### Test Results
```
15 passed in 0.12s
```

---

## Research Findings

### Current Agentic Memory Systems (2025)

- **A-MEM**: Zettelkasten-style linked notes with dynamic indexing
- **Zep/Graphiti**: Temporal knowledge graphs, 18.5% accuracy improvement over vector-only
- **Mem0**: Hybrid vector + graph + key-value for persistent memory
- **FalkorDB**: 90% hallucination reduction with knowledge graphs

### Key Insight
Pure vector RAG has limitations for structured facts:
- Lost constraints (top-k misses critical facts)
- Semantic drift (matches topic but wrong specifics)
- Context dilution (too many partially-relevant chunks)

**Recommendation**: Graph/relational lookup for exact facts, vectors only for fuzzy semantic matching.

---

## graphs/hardware_registry Analysis

Explored existing structure (43+ hardware profiles):

```
hardware_registry/
├── cpu/           # 18 devices
├── gpu/           # 10 devices
├── accelerator/   # 12 devices
├── dsp/           # 2 devices
└── boards/        # 2 devices
    └── {device_id}/
        ├── spec.json
        └── calibrations/*.json
```

### What Stays in graphs
- `ops_per_clock` - Roofline model parameters
- `theoretical_peaks` - Computed from ops_per_clock × frequency
- Calibration data - Measured performance, efficiency curves
- Operation profiles - GEMM, CONV, attention benchmarks

### What Migrates to embodied-schemas
- Datasheet specs (memory, TDP, compute units)
- Power profiles (modes and limits)
- Physical/environmental/interface specs

---

## Next Steps

1. **Seed initial data** - Add YAML files for key hardware platforms and models
2. **Integrate with graphs** - Add embodied-schemas as dependency, update hardware registry to reference base_id
3. **Integrate with embodied-ai-architect** - Migrate knowledge_base.py to use shared schemas
4. **Implement analysis tools** - Build dimension-specific tools using the verdict-first output schema

---

## Files Changed

### Created (embodied-schemas repo)
- `pyproject.toml` - Package configuration
- `README.md` - Usage documentation
- `LICENSE` - MIT license
- `.gitignore` - Python ignores
- `CHANGELOG.md` - Version history
- `src/embodied_schemas/*.py` - 9 Python modules
- `tests/test_schemas.py` - 15 tests
- `docs/sessions/2024-12-20-initial-setup.md` - Session log

### Created (embodied-ai-architect repo)
- `docs/plans/agentic-tool-architecture.md`
- `docs/plans/embodied-ai-codesign-subtasks.md`
- `docs/plans/knowledge-base-schema.md`
- `docs/plans/shared-schema-repo-architecture.md`

---

## Statistics

- Planning documents: 4 files, ~2,500 lines
- Schema code: 9 modules, ~1,500 lines
- Tests: 15 passing
- Total discussion topics: 7 major architectural decisions

---

*Session duration: ~3 hours*
