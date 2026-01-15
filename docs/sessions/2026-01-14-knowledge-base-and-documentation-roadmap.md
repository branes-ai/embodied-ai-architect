# Session: Knowledge Base Architecture and Documentation Roadmap

**Date**: 2026-01-14
**Focus**: LLM-native knowledge base design and Astro Starlight documentation roadmap

## Summary

Resumed from a previous session on branes_mcp integration. Designed a comprehensive LLM-native knowledge base architecture and created a documentation roadmap using Astro Starlight. Also completed missing roadmap documentation.

## Key Accomplishments

### 1. Knowledge Base Architecture (LLM-Native)

Created `docs/knowledge-base-architecture.md` with a fundamentally different approach than traditional database-backed systems:

**Design Principles:**
- **Text is the API** - LLMs reason over markdown/YAML, not SQL query results
- **Context injection over tool calls** - Inject relevant content directly into LLM context
- **Tools only for computation** - Roofline analysis requires tools; everything else is text
- **Chunked for retrieval** - Content split into ~500-2000 token units

**Three-Tier Content Model:**

| Tier | Source | Access Method |
|------|--------|---------------|
| Structured Catalog | embodied-schemas YAML (135 files) | Vector search → context injection |
| Analytical Data | graphs library (roofline) | Tool calls (computation required) |
| Experiential Knowledge | Authored markdown articles | Vector search → context injection |

**Key Insight**: The user correctly identified that hiding structured catalog data behind SQL would thwart LLM capabilities. Revised the architecture to use direct markdown files with semantic retrieval.

### 2. Documentation Roadmap (Astro Starlight)

Created `docs/plans/roadmap-documentation.md` with a 12-week plan:

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| Foundation | Weeks 1-2 | Starlight site, landing page, getting started |
| Catalog | Weeks 3-4 | Auto-generated pages from embodied-schemas |
| Tutorials | Weeks 5-8 | 7 step-by-step deployment/optimization guides |
| Reference | Weeks 9-10 | CLI, MCP tools, API documentation |
| Interactive | Weeks 11-12 | Hardware finder, compatibility matrix |

**LLM Integration Features:**
- `llms.txt` - Lightweight sitemap for AI agent discovery
- `llms-full.txt` - Full content for RAG ingestion
- Single source of truth: YAML → website + knowledge base + LLM metadata

### 3. Complete Roadmap Documentation

**Created:**
- `docs/plans/roadmap-phase-1.md` - Missing Phase 1 (Digital Twin Foundation) with user stories and acceptance criteria
- `docs/plans/roadmap-summary.md` - Complete overview of all 3 phases + supporting infrastructure

**Updated:**
- `docs/plans/roadmap.md` - Added "Supporting Infrastructure Roadmaps" section linking to documentation, knowledge base, devteam, and system architecture roadmaps

### 4. Roadmap Summary

Consolidated the complete 12-month product roadmap:

| Phase | Timeline | Goal | Key Metric |
|-------|----------|------|------------|
| Phase 1 | Months 1-4 | Digital Twin Foundation | ±15% energy model accuracy |
| Phase 2 | Months 5-8 | Software Architect Agent | ≥20% Perf/Watt improvement |
| Phase 3 | Months 9-12 | Hardware Co-Designer | ≥50% flight time increase |

## Files Created

| File | Purpose |
|------|---------|
| `docs/knowledge-base-architecture.md` | LLM-native KB design with retrieval system |
| `docs/plans/roadmap-documentation.md` | Astro Starlight docs site roadmap |
| `docs/plans/roadmap-phase-1.md` | Phase 1 user stories and acceptance criteria |
| `docs/plans/roadmap-summary.md` | Complete roadmap overview |

## Files Modified

| File | Change |
|------|--------|
| `docs/plans/roadmap.md` | Added Supporting Infrastructure Roadmaps section |

## Architecture Decisions

### 1. LLM-Native vs Database-Backed

**Decision**: Use markdown files with vector retrieval instead of SQL database.

**Rationale**:
- LLMs can reason directly over text in context
- SQL adds tool call indirection and latency
- embodied-schemas is already 135 YAML files (~500KB total)
- Semantic retrieval selects relevant chunks within token budget

### 2. Astro Starlight for Documentation

**Decision**: Use Astro Starlight instead of MkDocs or Docusaurus.

**Rationale**:
- Modern, fast static site generator
- Built-in documentation features (search, sidebar, dark mode)
- Clean HTML output that LLMs can consume
- Native support for `llms.txt` generation
- MDX support for interactive components

### 3. Single Source of Truth

**Decision**: Generate docs, knowledge base, and LLM metadata from same source files.

**Rationale**:
- Prevents content drift between human docs and LLM context
- Changes propagate to all outputs automatically
- Reduces maintenance burden

## Content Flow

```
embodied-schemas (YAML)
        │
        ├─► docs-site/ (Starlight)     → Human-readable website
        ├─► branes_mcp/knowledge/      → LLM context injection
        └─► llms.txt / llms-full.txt   → AI agent discovery
```

## Next Steps

1. **branes_mcp session**: Implement Phase 1 of knowledge base
   - Write `generate_catalog.py` (YAML → markdown chunks)
   - Build vector index with FAISS
   - Implement `retrieve_context()` function
   - Wrap graphs tools to return formatted text

2. **Documentation site**: Initialize Astro Starlight project
   - `npm create astro@latest -- --template starlight`
   - Create landing page and getting started guide
   - Deploy to Vercel

3. **Content authoring**: Write first deployment guides
   - YOLOv8 on Jetson Orin with TensorRT
   - YOLOv8 on Coral Edge TPU
   - INT8 quantization guide

## Session Context

This session continued from a previous branes_mcp session that was interrupted. The previous session had created the branes_mcp repository structure with:
- Dual-Response pattern for token-efficient tool outputs
- Mock data for hardware catalog and knowledge base
- Context manager with TF-IDF scoring
- REST API for out-of-band retrieval

The next step identified was "Integrate with graphs library for real roofline analysis" and "Connect to real knowledge base (replace mock data)". This session focused on the knowledge base architecture and documentation infrastructure needed to support that integration.
