# Session Log: Cross-Repository Claude Documentation

**Date**: December 21, 2025
**Focus**: Establishing cross-repository awareness for Claude Code instances

---

## Summary

This session addressed how Claude Code instances working in different repositories understand the multi-repo architecture. We created CLAUDE.md for the new `embodied-schemas` repo and updated CLAUDE.md files in downstream repos to reference the shared schema dependency.

---

## Work Completed

### 1. Created embodied-schemas/CLAUDE.md

Created comprehensive documentation for Claude Code working in the shared schema repository:

- **Repository Role**: Explicitly states it's a shared dependency imported by both `graphs` and `embodied-ai-architect`
- **Data Split**: Documents what belongs here (datasheet specs) vs downstream repos (roofline/calibration)
- **Build/Test Commands**: Standard development workflow
- **Schema Design Patterns**:
  - Verdict-first output schema
  - Pydantic models with optional fields
  - ID conventions for hardware, models, sensors, use cases
- **Adding New Data**: Guidelines for adding hardware, models, making schema changes
- **Versioning**: Semantic versioning with downstream compatibility notes

### 2. Updated embodied-ai-architect/CLAUDE.md

Added "Related Repositories" section:

```
embodied-schemas (shared dependency)
       ↑              ↑
       │              │
   graphs      embodied-ai-architect (this repo)
```

- Documents imports from embodied-schemas (HardwareEntry, Registry, BenchmarkResult)
- Documents what graphs provides (roofline models, calibration, hardware mappers)
- Data split summary

### 3. Updated graphs/CLAUDE.md

Added "Related Repositories" section:

- Dependency diagram showing all three repos
- Documents imports from embodied-schemas
- Data split table:

| graphs | embodied-schemas |
|--------|------------------|
| ops_per_clock | Vendor specs |
| theoretical_peaks | Physical specs |
| Calibration data | Environmental specs |
| Operation profiles | Interface specs |
| Efficiency curves | Power profiles |

- Notes that `hardware_registry/` references `base_id` from embodied-schemas

### 4. Fixed Date Errors

Fixed remaining date error in `embodied-schemas/docs/sessions/2025-12-20-initial-setup.md`:
- Changed "December 20, 2024" to "December 20, 2025"

---

## Architecture Overview

The three-repo architecture ensures:

1. **Single Source of Truth**: `embodied-schemas` owns all Pydantic models and factual data
2. **Clean Dependencies**: Both downstream repos import schemas, no circular dependencies
3. **Separation of Concerns**:
   - `embodied-schemas`: Datasheet specs, vendor facts
   - `graphs`: Analysis tools, roofline models, calibration
   - `embodied-ai-architect`: LLM orchestration, agentic tools, CLI

4. **Cross-Repo Awareness**: Each repo's CLAUDE.md now documents:
   - The dependency graph
   - What that repo owns
   - What it imports from other repos
   - The `base_id` linking pattern

---

## Files Changed

### Created
- `embodied-schemas/CLAUDE.md`

### Modified
- `embodied-ai-architect/CLAUDE.md` - Added Related Repositories section
- `graphs/CLAUDE.md` - Added Related Repositories section
- `embodied-schemas/docs/sessions/2025-12-20-initial-setup.md` - Fixed date

---

## Key Decisions

### CLAUDE.md Purpose
Each repo's CLAUDE.md serves to:
1. Orient any Claude Code instance to the repo's role
2. Provide build/test/development commands
3. Document relationships with other repos
4. Establish data ownership boundaries

### Cross-Reference Pattern
Each repo explicitly documents:
- The dependency diagram
- What it imports from `embodied-schemas`
- The data split (what stays local vs shared)
- The `base_id` linking pattern for hardware

---

## Next Steps

1. **Commit changes** in all three repos
2. **Add embodied-schemas dependency** to pyproject.toml in graphs and embodied-ai-architect
3. **Seed initial data** in embodied-schemas with YAML files for key hardware platforms
4. **Migrate hardware registry** in graphs to use `base_id` references

---

*Session duration: ~30 minutes*
