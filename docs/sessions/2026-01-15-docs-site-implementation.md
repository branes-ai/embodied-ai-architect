# Session: Documentation Site Implementation

**Date**: 2026-01-15
**Focus**: Implementing Astro Starlight docs site and fixing duplicate tool name bug

## Summary

Implemented the Phase 1 documentation site using Astro Starlight, fixed a duplicate tool name bug in the LLM tools, and refined the landing page messaging to better convey the Embodied AI Architect's identity as an agentic AI for designing differentiated embodied AI solutions.

## Key Accomplishments

### 1. Fixed Duplicate Tool Name Bug

**Problem**: 18 tests in `test_verdict_tools_cli.py` were failing with error:
```
'tools: Tool names must be unique.'
```

**Root Cause**: Both `graphs_tools.py` and `architecture_tools.py` defined a tool named `identify_bottleneck`:
- `graphs_tools.py:171` - analyzes model-level bottleneck (compute-bound vs memory-bound)
- `architecture_tools.py:223` - identifies bottleneck operator in a pipeline

**Fix**: Renamed the architecture tool to `identify_architecture_bottleneck` in:
- Tool definition in `get_architecture_tool_definitions()`
- Executor registration in `create_architecture_tool_executors()`

**Result**: Tests improved from 18 failed to 1 failed (remaining failure is unrelated test expectation issue).

### 2. Implemented Astro Starlight Documentation Site

Created `docs-site/` directory with full Astro Starlight implementation:

**Structure:**
```
docs-site/
├── astro.config.mjs          # Configured with GitHub Pages support
├── public/llms.txt           # LLM-friendly sitemap
├── src/
│   ├── assets/               # Logo + hero SVGs
│   ├── styles/custom.css     # Brand colors
│   └── content/docs/
│       ├── index.mdx                    # Landing page
│       ├── getting-started/             # Introduction, Installation, Quickstart
│       ├── features/                    # 5 feature pages
│       ├── tutorials/yolo-on-jetson.md  # Sample tutorial
│       ├── catalog/                     # Hardware, Models, Sensors, Operators
│       ├── reference/                   # CLI, MCP, API, Constraints
│       └── troubleshooting/             # Common issues
```

**Pages Created (20 total):**
- Landing page with value proposition
- Getting Started: Introduction, Installation, Quickstart
- Features: Model Analysis, Roofline Analysis, Hardware Selection, Constraint Checking, Deployment
- Catalog: Hardware, Models, Sensors, Operators (placeholder for auto-generation)
- Reference: CLI, MCP Tools, Python API, Constraints
- Tutorials: YOLOv8 on Jetson Orin
- Troubleshooting: Common Issues

### 3. GitHub Pages Deployment Support

**Challenge**: Starlight doesn't automatically prepend `base` path to markdown links.

**Solution**: Added `rehype-rewrite` plugin to automatically rewrite internal links:
```javascript
// astro.config.mjs
const rehypeBaseLinks = isGitHubPages ? [
  rehypeRewrite,
  {
    rewrite: (node) => {
      if (node.type === 'element' && node.tagName === 'a' && node.properties?.href) {
        const href = node.properties.href;
        if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('/embodied-ai-architect')) {
          node.properties.href = '/embodied-ai-architect' + href;
        }
      }
    }
  }
] : [];
```

**Build modes:**
- Local: `npm run build` - links are `/getting-started/...`
- GitHub Pages: `DEPLOY_TARGET=github-pages npm run build` - links are `/embodied-ai-architect/getting-started/...`

**GitHub Actions workflow** created at `.github/workflows/deploy-docs.yml`.

### 4. Refined Value Proposition Messaging

Updated landing page and introduction to convey the Architect's identity:

**Key Messaging:**
1. **Agentic AI Identity**: "Your AI architect", "partners with product architects and engineers"
2. **Custom + COTS Differentiation**: Covers both commodity and custom hardware with pre-silicon modeling
3. **Tesla FSD Example**: Concrete example of why custom hardware creates competitive advantage
4. **Competitive Intelligence**: "Predict performance, cost, and energy of competitor systems"
5. **Build-vs-Buy Guidance**: Helps determine when custom silicon justifies investment
6. **Functional Requirements Emphasis**: Latency and energy are functional requirements in embodied AI

**User-refined opening:**
> The key difference of Embodied AI applications over their Datacenter relatives is that latency and energy targets are functional requirements:
> - Can't identify an object in 5msec, your autonomous vehicle will fail.
> - Can't execute your MPC optimization using less than 2.5W, your drone mission will fail.
> - Can't deliver reliable 30 frames per second throughput, your situational awareness will fail.

## Files Changed

### Bug Fix
- `src/embodied_ai_architect/llm/architecture_tools.py`: Renamed `identify_bottleneck` to `identify_architecture_bottleneck`

### New Files (docs-site/)
- `astro.config.mjs` - Site configuration with GitHub Pages support
- `package.json` - Dependencies (Astro 5.6.1, Starlight 0.37.3)
- `public/llms.txt` - LLM-friendly site description
- `src/assets/` - Logo SVGs, hero image
- `src/styles/custom.css` - Brand colors
- `src/content/docs/` - 19 markdown/MDX files

### New Files (workflows)
- `.github/workflows/deploy-docs.yml` - GitHub Pages deployment

## Technical Decisions

1. **Astro Starlight over alternatives**: Built-in docs features, Pagefind search, dark mode, MDX support
2. **Rehype plugin for base path**: More maintainable than manually updating all links
3. **Environment-based config**: `DEPLOY_TARGET=github-pages` for production builds
4. **Relative hero links**: Changed from `/path/` to `./path/` for Starlight compatibility

## Next Steps

1. **Deploy to GitHub Pages**: Push and enable Pages in repo settings
2. **Auto-generate catalog**: Build YAML → MDX generator for embodied-schemas
3. **Add more tutorials**: Coral deployment, INT8 quantization, multi-model pipelines
4. **Interactive features**: Hardware finder, compatibility matrix (Phase 5 roadmap)

## Commands Reference

```bash
# Local development
cd docs-site && npm run dev

# Local build
npm run build

# GitHub Pages build
DEPLOY_TARGET=github-pages npm run build

# Preview built site
npm run preview
```
