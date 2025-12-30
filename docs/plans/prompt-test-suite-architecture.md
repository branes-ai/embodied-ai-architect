# Reference Prompt Test Suite Architecture

This document describes the design, implementation, and maintenance strategy for the reference prompt test suite used to validate the verdict-first tools in the CLI chat interface.

## Purpose

The reference prompt test suite validates that:
1. **Tool Selection** - The LLM correctly maps user prompts to the appropriate tools
2. **Verdict Accuracy** - Tools return correct PASS/FAIL verdicts for known scenarios
3. **Response Quality** - Agent responses are actionable, include relevant information, and avoid errors
4. **Regression Prevention** - Changes to tools, prompts, or model behavior don't break expected outcomes

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Test Suite Components                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │  Test Cases      │    │  Test Runner     │    │  Assertions  │  │
│  │  (Declarative)   │───▶│  (Execution)     │───▶│  (Validation)│  │
│  └──────────────────┘    └──────────────────┘    └──────────────┘  │
│          │                       │                      │          │
│          ▼                       ▼                      ▼          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ verdict_tools_   │    │ VerdictTools     │    │ ToolCall     │  │
│  │ test_suite.md    │    │ Tester           │    │ Capture      │  │
│  │                  │    │                  │    │              │  │
│  │ - Categories     │    │ - Mock mode      │    │ - Records    │  │
│  │ - Prompts        │    │ - Live API mode  │    │ - Validates  │  │
│  │ - Expectations   │    │ - Agent wrapper  │    │ - Reports    │  │
│  └──────────────────┘    └──────────────────┘    └──────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Declarative Test Cases

Test cases are defined as data structures, not imperative code:

```python
@dataclass
class PromptTestCase:
    name: str                           # Unique identifier
    prompt: str                         # User input to test
    expected_tools: list[str]           # Tools that should be called
    expected_verdict: str | None        # PASS, FAIL, or None
    expected_in_response: list[str]     # Substrings to find
    not_expected_in_response: list[str] # Substrings that should NOT appear
    timeout_seconds: int = 60
```

**Rationale**: Declarative definitions are:
- Easy to read and audit
- Machine-parseable for reporting
- Extensible without code changes
- Version-controllable with clear diffs

### 2. Two-Tier Execution

| Mode | API Calls | Speed | Purpose |
|------|-----------|-------|---------|
| Unit (Mock) | None | Fast | Validate test infrastructure, CI gates |
| Live API | Real | Slow | Validate actual LLM behavior |

```bash
# Unit tests (no API key required)
pytest tests/test_verdict_tools_cli.py -v

# Live API tests (requires ANTHROPIC_API_KEY)
pytest tests/test_verdict_tools_cli.py -v --live-api
```

**Rationale**: Fast unit tests for CI, live tests for behavior validation.

### 3. Verdict-First Validation

Tools return structured verdicts, tests validate against them:

```json
{
  "verdict": "PASS | FAIL | UNKNOWN",
  "confidence": "high | medium | low",
  "summary": "One-sentence result",
  "metric": { "name": "latency", "measured": 0.4, "required": 10.0, "unit": "ms" }
}
```

Tests extract verdicts from both:
- Tool call results (structured JSON)
- Agent prose responses (keyword detection)

### 4. Flexible Tool Matching

A test passes if **any** expected tool is called, not all:

```python
expected_tools=["full_analysis", "check_latency"]  # Either is acceptable
```

**Rationale**: The LLM may choose different valid paths to answer the same question. A bottleneck query might use `full_analysis` or `identify_bottleneck` - both are correct.

## Test Categories

Tests are organized by functional area:

| Category | Tool(s) Tested | Example Prompt |
|----------|----------------|----------------|
| Latency Checks | `check_latency` | "Can ResNet-18 meet 10ms on H100?" |
| Power Checks | `check_power` | "Can I run within 15W?" |
| Memory Checks | `check_memory` | "Does it fit in 512MB?" |
| Full Analysis | `full_analysis` | "Give me complete analysis of ResNet-50" |
| Comparisons | `compare_hardware_targets` | "Compare H100 vs A100" |
| Listings | `list_available_hardware` | "What hardware is available?" |
| Error Handling | Various | "Analyze FooBarNet" (unknown model) |
| Multi-turn | Various | Follow-up questions with context |

## Implementation Details

### File Structure

```
tests/
├── conftest.py                    # Pytest configuration, --live-api flag
├── prompts/
│   └── verdict_tools_test_suite.md  # Human-readable test documentation
└── test_verdict_tools_cli.py      # Automated test implementation
```

### Key Components

#### PromptTestCase

Defines a single test scenario with expectations:

```python
PromptTestCase(
    name="latency_pass_easy",
    prompt="Can ResNet-18 meet a 10ms latency target on an H100?",
    expected_tools=["check_latency"],
    expected_verdict="PASS",
    expected_in_response=["PASS", "headroom"],
)
```

#### ToolCallCapture

Intercepts tool calls during agent execution:

```python
class ToolCallCapture:
    def on_tool_start(self, name: str, args: dict) -> None
    def on_tool_end(self, name: str, result: str) -> None
    def has_tool(self, name: str) -> bool
    def get_tool_names(self) -> list[str]
```

#### VerdictToolsTester

Orchestrates test execution:

```python
class VerdictToolsTester:
    def __init__(self, use_live_api: bool = False)
    def setup(self) -> None          # Initialize agent (mock or real)
    def run_test(self, test_case) -> dict  # Execute and validate
```

### Validation Logic

Each test validates:

1. **Tool Selection**: At least one expected tool was called
2. **Verdict Match**: If `expected_verdict` set, response contains it
3. **Content Presence**: All `expected_in_response` substrings found
4. **Content Absence**: No `not_expected_in_response` substrings found
5. **No Tracebacks**: Python exceptions don't leak to output

## Adding New Test Cases

### Step 1: Identify the Scenario

Determine what behavior you're testing:
- New tool coverage
- Edge case handling
- Regression prevention
- Error handling

### Step 2: Create the Test Case

Add to the appropriate category list in `test_verdict_tools_cli.py`:

```python
LATENCY_TEST_CASES = [
    # ... existing cases ...
    PromptTestCase(
        name="latency_quantization_impact",
        prompt="How much faster is ResNet-50 in INT8 vs FP32 on Jetson Orin?",
        expected_tools=["check_latency", "full_analysis"],
        expected_in_response=["INT8", "FP32", "speedup"],
    ),
]
```

### Step 3: Document in Test Suite

Add human-readable documentation to `tests/prompts/verdict_tools_test_suite.md`:

```markdown
### X.Y Quantization Comparison
\`\`\`
How much faster is ResNet-50 in INT8 vs FP32 on Jetson Orin?
\`\`\`
**Expected behavior:**
- Tool called: `check_latency` or `full_analysis` with precision parameter
- Response compares INT8 vs FP32 performance
- Speedup factor mentioned
```

### Step 4: Run and Validate

```bash
# Unit test (validates test structure)
pytest tests/test_verdict_tools_cli.py::TestVerdictToolsUnit -v

# Live test (validates actual behavior)
pytest tests/test_verdict_tools_cli.py -v --live-api -k "quantization"
```

## Maintenance Strategy

### Regular Validation Cadence

| Frequency | Test Set | Purpose |
|-----------|----------|---------|
| Every commit | Unit tests | Validate test infrastructure |
| Daily (CI) | Smoke tests (5 prompts) | Quick sanity check |
| Weekly | Full suite | Comprehensive validation |
| On model update | Full suite | Detect behavior changes |

### Handling Flaky Tests

LLM responses are non-deterministic. Strategies for handling flakiness:

1. **Flexible Expectations**: Accept multiple valid tool choices
2. **Semantic Matching**: Check for concepts, not exact wording
3. **Retry Logic**: For borderline cases, run 3x and pass if majority succeed
4. **Verdict Focus**: Trust tool verdicts over prose interpretation

### Updating for Model Changes

When upgrading Claude models:

1. Run full test suite with `--live-api`
2. Review failures for:
   - **True regressions**: Tool selection or verdict wrong → investigate
   - **Improved behavior**: Better response than expected → update expectations
   - **Style changes**: Same meaning, different words → relax string matching
3. Update test cases to match new baseline
4. Document changes in CHANGELOG

### Updating for Tool Changes

When modifying verdict-first tools:

1. Ensure tool output schema remains backward-compatible
2. Run affected test category
3. Add new test cases for new functionality
4. Update expected verdicts if thresholds change

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Prompt Test Suite
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run unit tests
        run: pytest tests/test_verdict_tools_cli.py -v

  smoke-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -e ".[dev,chat]"
      - name: Run smoke tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: pytest tests/test_verdict_tools_cli.py -v --live-api -k "smoke"
```

### Cost Management

Live API tests consume API credits. Strategies:

- **Smoke tests only in CI**: Run 5 prompts, not 50+
- **Full suite manually**: Run locally before releases
- **Caching**: Cache known-good responses for regression detection (future)

## Metrics and Reporting

### Test Results Schema

```python
{
    "test_run_id": "uuid",
    "timestamp": "ISO8601",
    "model": "claude-sonnet-4-20250514",
    "test_cases": [
        {
            "name": "latency_pass_easy",
            "status": "passed | failed | skipped",
            "tools_called": ["check_latency"],
            "verdict_found": "PASS",
            "duration_ms": 2340,
            "errors": []
        }
    ],
    "summary": {
        "passed": 45,
        "failed": 2,
        "skipped": 3,
        "pass_rate": 0.90
    }
}
```

### Quality Metrics

Track over time:
- **Pass rate**: % of tests passing
- **Tool accuracy**: Correct tool selected
- **Verdict accuracy**: Correct PASS/FAIL
- **Latency**: Response time per test
- **Cost**: API tokens consumed

## Future Enhancements

### Planned Improvements

1. **Golden Response Caching**
   - Cache known-good responses
   - Detect regressions via semantic diff
   - Reduce API costs for unchanged tests

2. **Prompt Mutation Testing**
   - Automatically generate prompt variants
   - "Can ResNet-18..." → "Is ResNet-18 able to..."
   - Validate robustness to phrasing

3. **Multi-Model Comparison**
   - Run same suite across Claude models
   - Compare Haiku vs Sonnet vs Opus
   - Track capability differences

4. **Visual Test Dashboard**
   - HTML report with pass/fail history
   - Trend graphs for quality metrics
   - Drill-down into failures

### Schema Evolution

If `PromptTestCase` needs extension:

```python
@dataclass
class PromptTestCase:
    # Existing fields...

    # Future fields (with defaults for backward compatibility)
    category: str = ""                    # For filtering
    priority: int = 1                     # 1=critical, 2=normal, 3=edge
    model_requirements: list[str] = None  # ["opus"] for complex reasoning
    retry_count: int = 1                  # For flaky tests
```

## Quick Reference

### Run Tests

```bash
# Unit tests (no API)
pytest tests/test_verdict_tools_cli.py::TestVerdictToolsUnit -v

# Smoke tests (live API)
pytest tests/test_verdict_tools_cli.py -v --live-api -k "smoke"

# Full suite (live API)
pytest tests/test_verdict_tools_cli.py::TestVerdictToolsLive -v --live-api

# Interactive runner
python tests/test_verdict_tools_cli.py
```

### Add a Test

1. Add `PromptTestCase` to appropriate `*_TEST_CASES` list
2. Document in `verdict_tools_test_suite.md`
3. Run unit tests to validate structure
4. Run live test to validate behavior

### Debug a Failure

1. Run with `-s` for verbose output: `pytest ... -v -s`
2. Check `tools_called` vs `expected_tools`
3. Check `verdict_found` vs `expected_verdict`
4. Review full response for context
5. Consider if expectation needs updating
