# Best-in-Class MCP Architectures for Value-Added Services

## The Core Problem

MCP's context window challenge is fundamental: **every tool definition and every intermediate result flows through the LLM context window**, causing:

- **Token explosion**: 100+ tools × schema definitions = 100K+ tokens before any work begins
- **Cost scaling**: Each API call re-transmits the full tool catalog
- **Latency degradation**: Larger contexts = slower inference
- **Accuracy loss**: Anthropic research (Sept 2025) confirms model accuracy *decreases* as context grows

This document synthesizes three production-proven architectures that address these challenges.

---

## Architecture 1: Code-First / Tools-as-Code Pattern

**Source**: [Anthropic Engineering](https://www.anthropic.com/engineering/code-execution-with-mcp)
**Production Validation**: [GitHub MCP Server - 98% Token Reduction](https://github.com/orgs/modelcontextprotocol/discussions/629)

### Core Insight

Instead of exposing tools as callable functions, expose them as **code APIs on disk**. The agent writes and executes code rather than calling tools directly.

```
Traditional MCP                    Code-First MCP
─────────────────                  ─────────────────
┌─────────────┐                    ┌─────────────┐
│   LLM       │                    │   LLM       │
│ (150K ctx)  │                    │ (2K ctx)    │
└──────┬──────┘                    └──────┬──────┘
       │ tool calls                       │ code
       ▼                                  ▼
┌─────────────┐                    ┌─────────────┐
│ MCP Server  │                    │  Sandbox    │
│ (112 tools) │                    │  Runtime    │
└─────────────┘                    └──────┬──────┘
                                          │ imports
                                          ▼
                                   ┌─────────────┐
                                   │ MCP Server  │
                                   │ (on-demand) │
                                   └─────────────┘
```

### Token Reduction Results

| Metric | Traditional | Code-First | Reduction |
|--------|-------------|------------|-----------|
| Tool definitions | 150,000 | 1,200 | 99.2% |
| Typical conversation | 80,000 | 6,000 | 92.5% |
| Simple queries | 75,000 | 2,000 | 97.3% |

### Implementation Pattern

```typescript
// servers/branes/hardware_analysis.ts
// Tool wrapper - agent loads only when needed
import { callMCPTool } from '../mcp-client';

export interface HardwareAnalysisInput {
  model_path: string;
  target_latency_ms: number;
  power_budget_w: number;
}

export interface HardwareRecommendation {
  hardware_id: string;
  predicted_latency_ms: number;
  predicted_power_w: number;
  roofline_efficiency: number;
  optimization_suggestions: string[];
}

export async function analyzeHardwareFit(
  input: HardwareAnalysisInput
): Promise<HardwareRecommendation[]> {
  return callMCPTool<HardwareRecommendation[]>(
    'branes__analyze_hardware_fit',
    input
  );
}
```

```typescript
// Agent-generated code (runs in sandbox)
import * as branes from './servers/branes';
import * as fs from 'fs';

// Load model, analyze, filter results - all in sandbox
const recommendations = await branes.analyzeHardwareFit({
  model_path: '/models/yolov8n.onnx',
  target_latency_ms: 33,
  power_budget_w: 5
});

// Process locally - never hits context window
const viable = recommendations.filter(r =>
  r.roofline_efficiency > 0.6 && r.predicted_power_w < 5
);

// Only return summary to LLM
fs.writeFileSync('/results/summary.json', JSON.stringify({
  count: viable.length,
  top_pick: viable[0],
  power_range: [Math.min(...viable.map(r => r.predicted_power_w)),
                Math.max(...viable.map(r => r.predicted_power_w))]
}));
```

### Key Design Decisions

1. **Single entry point**: One `execute_code` tool instead of 100+ tool definitions
2. **Progressive disclosure**: Agent explores filesystem to discover tools as needed
3. **Local filtering**: Large result sets processed in sandbox, not context
4. **Persistent state**: Results written to files, resumable across turns

### Applicability to Branes

**High fit** for:
- `graphs` roofline analysis (returns large datasets)
- Knowledge base queries (filters thousands of recipes)
- Batch hardware comparisons (iterates over many targets)

**Implementation cost**: Medium - requires sandbox runtime (Deno/Node)

---

## Architecture 2: Dual-Response ResourceLink Pattern

**Source**: [arXiv: Extending ResourceLink for Large Dataset Processing](https://arxiv.org/html/2510.05968v1)

### Core Insight

Separate **query construction** (needs LLM reasoning) from **data retrieval** (doesn't need LLM). Return two components:

1. **Preview**: Small sample for LLM to validate query correctness
2. **ResourceLink**: URI to fetch complete data out-of-band

```
┌──────────────────────────────────────────────────────────────────┐
│                      DUAL-RESPONSE PATTERN                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User: "Show me all hardware that can run YOLOv8 under 30ms"    │
│                               │                                  │
│                               ▼                                  │
│   ┌────────────────────────────────────────────────────────┐     │
│   │                    MCP SERVER                          │     │
│   │                                                        │     │
│   │   1. Execute query with LIMIT 5                        │     │
│   │   2. Execute COUNT(*) in parallel                      │     │
│   │   3. Generate ResourceLink URI                         │     │
│   │                                                        │     │
│   └────────────────────────────────────────────────────────┘     │
│                               │                                  │
│                ┌──────────────┴───────────────┐                  │
│                ▼                              ▼                  │
│       ┌─────────────────┐          ┌─────────────────────┐       │
│       │    PREVIEW      │          │    RESOURCE LINK    │       │
│       │  (→ LLM ctx)    │          │   (out-of-band)     │       │
│       │                 │          │                     │       │
│       │ • 5 sample rows │          │ • URI: /results/abc │       │
│       │ • Schema info   │          │ • Total: 47 records │       │
│       │ • Query echo    │          │ • Expires: 1hr      │       │
│       └─────────────────┘          └─────────────────────┘       │
│                │                              │                  │
│                ▼                              ▼                  │
│          LLM validates               Client fetches directly     │
│          "Looks correct,             GET /results/abc?page=1     │
│           proceed with               GET /results/abc?page=2     │
│           full export"               ...                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation Pattern

```python
# branes_mcp/tools/hardware_query.py

from mcp import Tool, ResourceLink
from pydantic import BaseModel

class HardwareQueryResult(BaseModel):
    # Preview for LLM context
    preview: list[dict]  # Max 5 items
    preview_count: int

    # Out-of-band retrieval
    resource_link: ResourceLink
    total_count: int
    schema: dict

@server.tool("query_hardware_catalog")
async def query_hardware_catalog(
    constraints: dict,
    preview_limit: int = 5
) -> HardwareQueryResult:
    """Query hardware catalog with preview + full results link."""

    # Execute with limit for preview
    preview = await db.query(
        constraints,
        limit=preview_limit
    )

    # Get total count (parallel)
    total = await db.count(constraints)

    # Generate persistent result link
    result_id = await cache.store_query(constraints, ttl=3600)

    return HardwareQueryResult(
        preview=preview,
        preview_count=len(preview),
        resource_link=ResourceLink(
            uri=f"branes://results/{result_id}",
            name="Full hardware query results",
            mimeType="application/json"
        ),
        total_count=total,
        schema=HardwareEntry.model_json_schema()
    )
```

```python
# Out-of-band REST endpoint (bypasses LLM entirely)
# branes_mcp/api/results.py

@router.get("/results/{result_id}")
async def get_results_metadata(result_id: str):
    """Return total count, schema, expiration."""
    return await cache.get_metadata(result_id)

@router.post("/results/{result_id}")
async def get_results_page(
    result_id: str,
    offset: int = 0,
    limit: int = 100
):
    """Paginated retrieval - never touches LLM."""
    return await cache.get_page(result_id, offset, limit)
```

### Token Impact Analysis

| Operation | Traditional | Dual-Response | Savings |
|-----------|-------------|---------------|---------|
| 50-row query | 5,000 tokens | 800 tokens | 84% |
| 500-row query | 50,000 tokens | 800 tokens | 98.4% |
| 5,000-row query | Fails (too large) | 800 tokens | ∞ |

### Key Design Decisions

1. **Preview semantics preserved**: Sample uses same query logic with LIMIT
2. **Explicit lifecycle**: Client can PIN results for persistence or let them expire
3. **Security isolation**: Multi-tenant filters applied at query time, not retrieval
4. **Discovery via .well-known**: Clients discover REST endpoints automatically

### Applicability to Branes

**High fit** for:
- Knowledge base search (thousands of deployment recipes)
- Hardware catalog queries (growing dataset)
- Benchmark history retrieval (large result sets over time)

**Implementation cost**: Low - standard REST + caching layer

---

## Architecture 3: Context Manager with Relevance Scoring

**Source**: [Implementing MCP in Multi-Agent Systems](https://subhadipmitra.com/blog/2025/implementing-model-context-protocol/)

### Core Insight

Not all context is equally valuable. Implement a **dedicated Context Manager** that:
- Scores relevance of each context block dynamically
- Enforces token budgets with intelligent pruning
- Applies time-based decay to stale information

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT MANAGER PATTERN                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Incoming Context Blocks                                       │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │
│   │SYST │ │USER │ │TOOL │ │KNOW │ │MEM  │ │TOOL │               │
│   │1.0  │ │0.9  │ │0.7  │ │0.4  │ │0.3  │ │0.2  │  ← relevance  │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘               │
│      │       │       │       │       │       │                  │
│      └───────┴───────┴───────┼───────┴───────┘                  │
│                              ▼                                  │
│              ┌───────────────────────────────┐                  │
│              │      CONTEXT MANAGER          │                  │
│              │                               │                  │
│              │  • Relevance scoring (TF-IDF) │                  │
│              │  • Time decay (exponential)   │                  │
│              │  • Token budget enforcement   │                  │
│              │  • Priority: SYS > USER > TOOL│                  │
│              │  • Saturation threshold: 90%  │                  │
│              │                               │                  │
│              └───────────────┬───────────────┘                  │
│                              │                                  │
│                              ▼                                  │
│              ┌───────────────────────────────┐                  │
│              │    OPTIMIZED CONTEXT          │                  │
│              │    (fits in window)           │                  │
│              │                               │                  │
│              │  [SYSTEM] + [USER] + [TOOL]   │                  │
│              │  ← only high-relevance blocks │                  │
│              └───────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Relevance Scoring Algorithm

```python
# branes_mcp/context/manager.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

class ContextBlock(BaseModel):
    content: str
    block_type: Literal["SYSTEM", "USER", "AGENT", "TOOL", "MEMORY", "KNOWLEDGE"]
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.0

class ContextManager:
    def __init__(
        self,
        max_tokens: int = 100_000,
        saturation_threshold: float = 0.9,
        decay_half_life_seconds: float = 300.0
    ):
        self.max_tokens = max_tokens
        self.saturation_threshold = saturation_threshold
        self.decay_half_life = decay_half_life_seconds
        self.vectorizer = TfidfVectorizer()

    def score_relevance(
        self,
        blocks: list[ContextBlock],
        current_query: str
    ) -> list[ContextBlock]:
        """Score blocks by relevance to current query."""

        # TF-IDF similarity
        corpus = [b.content for b in blocks] + [current_query]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[-1]

        for i, block in enumerate(blocks):
            # Base relevance from content similarity
            similarity = cosine_similarity(
                tfidf_matrix[i], query_vector
            )[0][0]

            # Time decay (non-SYSTEM blocks)
            if block.block_type != "SYSTEM":
                age_seconds = (datetime.now() - block.timestamp).total_seconds()
                decay = math.exp(-age_seconds / self.decay_half_life)
                similarity *= decay

            # Recency boost for recent turns
            if block.block_type in ("USER", "AGENT"):
                recency_boost = 1.5 if age_seconds < 60 else 1.0
                similarity *= recency_boost

            block.relevance_score = similarity

        return blocks

    def optimize_context(
        self,
        blocks: list[ContextBlock],
        current_query: str
    ) -> list[ContextBlock]:
        """Prune to fit token budget while preserving high-value blocks."""

        scored = self.score_relevance(blocks, current_query)

        # Priority ordering
        priority = {"SYSTEM": 0, "USER": 1, "AGENT": 2, "TOOL": 3, "MEMORY": 4, "KNOWLEDGE": 5}
        sorted_blocks = sorted(
            scored,
            key=lambda b: (priority[b.block_type], -b.relevance_score)
        )

        # Fill until budget exhausted
        budget = int(self.max_tokens * self.saturation_threshold)
        selected = []
        used_tokens = 0

        for block in sorted_blocks:
            if used_tokens + block.token_count <= budget:
                selected.append(block)
                used_tokens += block.token_count
            elif block.block_type == "SYSTEM":
                # Always include system blocks
                selected.append(block)
                used_tokens += block.token_count

        return selected
```

### Integration with MCP Tools

```python
# branes_mcp/server.py

class BranesMCPServer:
    def __init__(self):
        self.context_manager = ContextManager(max_tokens=100_000)
        self.tool_results_cache = {}

    @server.tool("analyze_hardware_fit")
    async def analyze_hardware_fit(self, input: dict) -> dict:
        # Store result with metadata for context management
        result = await self._do_analysis(input)

        # Create context block for result
        result_block = ContextBlock(
            content=self._summarize_result(result),  # Compressed summary
            block_type="TOOL",
            timestamp=datetime.now(),
            token_count=count_tokens(result)
        )

        # Full result available via ResourceLink if needed
        result_id = self._cache_full_result(result)

        return {
            "summary": result_block.content,  # Goes to context
            "resource_link": f"branes://results/{result_id}",  # Out-of-band
            "recommendations": result["top_3"]  # Key insights only
        }

    def _summarize_result(self, result: dict) -> str:
        """Create token-efficient summary for context."""
        return f"""Hardware analysis: {result['total_candidates']} options evaluated.
Top recommendation: {result['top_pick']['name']}
- Latency: {result['top_pick']['latency_ms']}ms
- Power: {result['top_pick']['power_w']}W
- Efficiency: {result['top_pick']['roofline_efficiency']:.1%}"""
```

### Token Impact Analysis

| Scenario | Without Manager | With Manager | Savings |
|----------|-----------------|--------------|---------|
| 10-turn conversation | 50,000 | 20,000 | 60% |
| Multi-tool workflow | 80,000 | 25,000 | 69% |
| Long research session | 150,000+ | 40,000 | 73%+ |

### Key Design Decisions

1. **Type-based priority**: SYSTEM blocks always preserved
2. **Exponential decay**: Old tool results fade unless re-referenced
3. **Recency boost**: Recent conversation turns get priority
4. **Saturation threshold**: Leave 10% headroom for response generation
5. **Summary + link**: Return compressed summaries, link to full data

### Applicability to Branes

**High fit** for:
- Long design sessions (multi-turn hardware exploration)
- Knowledge base Q&A (accumulating context over time)
- Agentic workflows (multiple tool calls building on each other)

**Implementation cost**: Medium - requires context tracking infrastructure

---

## Comparative Analysis

| Dimension | Code-First | Dual-Response | Context Manager |
|-----------|------------|---------------|-----------------|
| **Token Reduction** | 92-99% | 84-98% | 60-73% |
| **Implementation Complexity** | High (sandbox) | Low (REST cache) | Medium (scoring) |
| **Best For** | Many tools, complex workflows | Large result sets | Long conversations |
| **LLM Capability Required** | Code generation | None | None |
| **Latency Impact** | Higher (code exec) | Lower (parallel) | Minimal |
| **Auditability** | High (code logs) | Medium | Low |

---

## Recommended Hybrid Architecture for Branes

Combine all three patterns based on operation type:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BRANES MCP HYBRID ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     CONTEXT MANAGER (Always Active)                 │   │
│   │   • Relevance scoring for all context blocks                        │   │
│   │   • Token budget enforcement (90K of 100K)                          │   │
│   │   • Time decay for stale tool results                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│            ┌───────────────────────┼───────────────────────┐                │
│            │                       │                       │                │
│            ▼                       ▼                       ▼                │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│    │   CODE-FIRST    │    │  DUAL-RESPONSE  │    │   DIRECT TOOL   │        │
│    │                 │    │                 │    │                 │        │
│    │ Complex multi-  │    │ Large dataset   │    │ Simple lookups  │        │
│    │ step analysis:  │    │ queries:        │    │ and mutations:  │        │
│    │                 │    │                 │    │                 │        │
│    │ • Batch compare │    │ • KB search     │    │ • Get HW spec   │        │
│    │ • Custom filter │    │ • History query │    │ • Store result  │        │
│    │ • Optimization  │    │ • Catalog list  │    │ • Config update │        │
│    │   loops         │    │                 │    │                 │        │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
│   Tool Selection Logic:                                                     │
│   ─────────────────────                                                     │
│   if operation.involves_iteration or operation.tool_count > 3:              │
│       use CODE_FIRST                                                        │
│   elif operation.result_size > 1000 or operation.is_search:                 │
│       use DUAL_RESPONSE                                                     │
│   else:                                                                     │
│       use DIRECT_TOOL                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

**Phase 1**: Dual-Response + Context Manager
- Immediate token savings for knowledge base and catalog queries
- Lower implementation risk
- Works with current LLM capabilities

**Phase 2**: Add Code-First for Power Users
- Sandbox runtime for complex analysis workflows
- Progressive rollout to validate safety
- Premium tier feature (higher compute cost)

### Branes-Specific Tool Categorization

| Tool Category | Pattern | Rationale |
|---------------|---------|-----------|
| `analyze_hardware_fit` | Dual-Response | Returns ranked list, needs preview |
| `query_knowledge_base` | Dual-Response | Search results, needs pagination |
| `get_hardware_spec` | Direct | Single record lookup |
| `compare_hardware_batch` | Code-First | Iterates over multiple targets |
| `estimate_silicon_requirements` | Direct | Computed result, small output |
| `run_benchmark_suite` | Code-First | Multi-step, needs local filtering |
| `export_report` | Dual-Response | Large output, needs link |

---

## Sources

- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [GitHub MCP Server: 98% Token Reduction](https://github.com/orgs/modelcontextprotocol/discussions/629)
- [arXiv: ResourceLink Patterns for Large Datasets](https://arxiv.org/html/2510.05968v1)
- [MCP Best Practices Guide](https://modelcontextprotocol.info/docs/best-practices/)
- [Implementing MCP in Multi-Agent Systems](https://subhadipmitra.com/blog/2025/implementing-model-context-protocol/)
- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
