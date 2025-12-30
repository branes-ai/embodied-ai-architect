# Verdict-First Tools Test Prompt Suite

This document contains test prompts for validating the verdict-first graph analysis tools in the CLI chat interface.

## Prerequisites

```bash
# Ensure graphs is installed with schemas support
pip install -e ../graphs[schemas]

# Ensure embodied-schemas is installed
pip install -e ../embodied-schemas

# Start the chat
export ANTHROPIC_API_KEY=your-key
embodied-ai chat -v
```

---

## Category 1: Direct Latency Checks

These prompts should trigger the `check_latency` tool.

### 1.1 PASS Case - Easy Target
```
Can ResNet-18 meet a 10ms latency target on an H100?
```
**Expected behavior:**
- Tool called: `check_latency(model_name="resnet18", hardware_name="H100-SXM5-80GB", latency_target_ms=10.0)`
- Verdict: PASS (H100 should achieve ~0.4ms for ResNet-18)
- Response mentions significant headroom (~96%)

### 1.2 FAIL Case - Tight Target
```
Can ResNet-50 achieve 0.1ms latency on a Jetson Orin Nano?
```
**Expected behavior:**
- Tool called: `check_latency`
- Verdict: FAIL (Jetson Orin Nano cannot hit 0.1ms for ResNet-50)
- Response includes suggestions for meeting the target

### 1.3 Edge Case - Borderline Target
```
Check if MobileNetV2 can run in under 5ms on the Jetson Orin AGX
```
**Expected behavior:**
- Tool called: `check_latency`
- Verdict: PASS or FAIL with margin information
- Response explains the margin (how close to the target)

### 1.4 Natural Language Variant
```
I need YOLOv8n to run at 30fps on a Coral Edge TPU. Is that possible?
```
**Expected behavior:**
- Agent converts 30fps to ~33ms latency target
- Tool called: `check_latency(model_name="yolov8n", hardware_name="Coral-Edge-TPU", latency_target_ms=33.3)`
- Clear verdict with explanation

### 1.5 Precision Variant
```
What's the latency of EfficientNet-B0 on an A100 in FP16?
```
**Expected behavior:**
- Tool called: `check_latency` or `full_analysis` with precision="FP16"
- Response includes FP16-specific metrics

---

## Category 2: Direct Power Checks

These prompts should trigger the `check_power` tool.

### 2.1 PASS Case - Generous Budget
```
Can I run MobileNetV2 on a Jetson Orin Nano within a 15W power budget?
```
**Expected behavior:**
- Tool called: `check_power(model_name="mobilenet_v2", hardware_name="Jetson-Orin-Nano", power_budget_w=15.0)`
- Verdict: PASS (Jetson Orin Nano TDP is 15W)

### 2.2 FAIL Case - Tight Budget
```
I have a 5W power budget for running ResNet-50. Can the Jetson Orin AGX work?
```
**Expected behavior:**
- Tool called: `check_power`
- Verdict: FAIL (Orin AGX TDP is 60W)
- Suggestions for lower-power alternatives

### 2.3 Drone Use Case
```
I'm building a drone with a 10W power budget. Can I run YOLOv8s for object detection?
```
**Expected behavior:**
- Tool called: `check_power`
- Agent should consider multiple hardware options or ask clarifying questions
- Response addresses the power constraint specifically

### 2.4 Battery Life Context
```
Running on battery, I need inference to stay under 3W average. What can YOLOv8n achieve on a Coral Edge TPU?
```
**Expected behavior:**
- Tool called: `check_power(model_name="yolov8n", hardware_name="Coral-Edge-TPU", power_budget_w=3.0)`
- Verdict with power analysis

---

## Category 3: Direct Memory Checks

These prompts should trigger the `check_memory` tool.

### 3.1 PASS Case - Plenty of Memory
```
Does ResNet-18 fit in 512MB of memory on the Jetson Orin AGX?
```
**Expected behavior:**
- Tool called: `check_memory(model_name="resnet18", hardware_name="Jetson-Orin-AGX", memory_budget_mb=512.0)`
- Verdict: PASS with memory breakdown

### 3.2 FAIL Case - Constrained Memory
```
Can I run ResNet-152 in just 50MB of memory?
```
**Expected behavior:**
- Tool called: `check_memory`
- Verdict: FAIL (ResNet-152 weights alone are ~230MB)
- Suggestions for reducing memory (quantization, smaller model)

### 3.3 Microcontroller Context
```
I have a microcontroller with 2MB of SRAM. Can any vision model fit?
```
**Expected behavior:**
- Agent should try `check_memory` with small models or explain limitations
- Response addresses the extreme constraint

---

## Category 4: Full Analysis

These prompts should trigger the `full_analysis` tool.

### 4.1 Comprehensive Analysis Without Constraint
```
Give me a complete analysis of ResNet-50 on the H100
```
**Expected behavior:**
- Tool called: `full_analysis(model_name="resnet50", hardware_name="H100-SXM5-80GB")`
- Response includes: latency, throughput, energy, memory breakdown, bottleneck analysis

### 4.2 Analysis With Latency Constraint
```
Analyze ViT-B/16 on TPU v4 with a 20ms latency requirement
```
**Expected behavior:**
- Tool called: `full_analysis(model_name="vit_b_16", hardware_name="TPU-v4", constraint_metric="latency", constraint_threshold=20.0)`
- Verdict included along with full metrics

### 4.3 Analysis With Energy Constraint
```
How energy efficient is MobileNetV3-Small on the Hailo-8? I need under 5mJ per inference.
```
**Expected behavior:**
- Tool called: `full_analysis` with constraint_metric="energy"
- Response addresses energy efficiency specifically

### 4.4 Bottleneck Identification
```
Is ResNet-18 compute-bound or memory-bound on an A100?
```
**Expected behavior:**
- Tool called: `full_analysis` or `identify_bottleneck`
- Response clearly identifies bottleneck type with explanation

---

## Category 5: Hardware Comparison

These prompts should trigger `compare_hardware_targets` or multiple `check_*` calls.

### 5.1 Head-to-Head Comparison
```
Compare ResNet-18 performance on H100 vs A100 vs Jetson Orin AGX
```
**Expected behavior:**
- Tool called: `compare_hardware_targets` with specific hardware list
- Ranked comparison by latency and efficiency

### 5.2 Best Hardware for Constraint
```
What's the best hardware to run YOLOv8m under 10ms latency?
```
**Expected behavior:**
- Multiple `check_latency` calls or `compare_hardware_targets`
- Response recommends hardware that meets the constraint

### 5.3 Edge vs Cloud
```
Should I run EfficientNet-B4 on edge (Jetson) or in the cloud (A100)?
```
**Expected behavior:**
- Comparison analysis covering latency, power, and cost tradeoffs

---

## Category 6: Multi-Constraint Scenarios

These prompts involve multiple constraints.

### 6.1 Latency AND Power
```
I need ResNet-18 to run under 10ms AND under 15W. What are my options?
```
**Expected behavior:**
- Multiple tool calls checking both constraints
- Response identifies hardware meeting both requirements

### 6.2 Drone Deployment Scenario
```
I'm deploying on a drone. I need:
- Object detection at 30fps
- Under 10W power
- Under 500MB memory
Can YOLOv8n work on the Jetson Orin Nano?
```
**Expected behavior:**
- Multiple constraint checks: latency (~33ms), power (10W), memory (500MB)
- Comprehensive verdict addressing all constraints

### 6.3 Automotive Use Case
```
For an automotive ADAS system, I need:
- Latency under 50ms
- High reliability
- Works on TDA4VM
Is EfficientNet-B0 suitable?
```
**Expected behavior:**
- Analysis on TDA4VM hardware
- Response addresses automotive-specific concerns

---

## Category 7: Error Handling

These prompts test graceful error handling.

### 7.1 Unknown Model
```
Can you analyze FooBarNet on the H100?
```
**Expected behavior:**
- Tool returns error for unknown model
- Agent explains the model isn't in the database and suggests alternatives

### 7.2 Unknown Hardware
```
Check latency of ResNet-18 on the NVIDIA RTX 5090
```
**Expected behavior:**
- Tool returns error for unknown hardware
- Agent lists available hardware options

### 7.3 Invalid Constraint
```
Can ResNet-18 achieve -5ms latency?
```
**Expected behavior:**
- Agent should recognize invalid input
- Graceful error message

### 7.4 Missing graphs Package
```
# (Test with graphs uninstalled)
Analyze ResNet-18 on H100
```
**Expected behavior:**
- Clear error message about missing dependency
- Installation instructions provided

---

## Category 8: Conversational Follow-ups

These test multi-turn conversations.

### 8.1 Iterative Refinement
```
Turn 1: "Analyze ResNet-50 on the Jetson Orin AGX"
Turn 2: "What if I use INT8 quantization?"
Turn 3: "How about on the Jetson Orin Nano instead?"
```
**Expected behavior:**
- Agent maintains context across turns
- Each response builds on previous analysis

### 8.2 Clarification Request
```
Turn 1: "I need fast inference for my model"
```
**Expected behavior:**
- Agent asks clarifying questions: what model? what hardware? what's "fast"?

### 8.3 Constraint Relaxation
```
Turn 1: "Can YOLOv8m run under 5ms on Jetson Orin Nano?"
Turn 2: "What if I relax it to 20ms?"
```
**Expected behavior:**
- First check should FAIL
- Second check may PASS with relaxed constraint

---

## Category 9: List and Discovery

These prompts test hardware/model discovery.

### 9.1 List Hardware
```
What hardware targets are available for analysis?
```
**Expected behavior:**
- Tool called: `list_available_hardware`
- Response shows categorized hardware list

### 9.2 Hardware by Category
```
Show me all edge GPUs you can analyze
```
**Expected behavior:**
- Tool called: `list_available_hardware(category="edge_gpu")`
- Response lists Jetson variants

### 9.3 Model Suggestions
```
What's a good lightweight model for edge deployment?
```
**Expected behavior:**
- Agent recommends models like MobileNetV2, EfficientNet-B0, YOLOv8n
- May offer to analyze specific options

---

## Validation Checklist

For each test prompt, verify:

- [ ] Correct tool(s) called
- [ ] Appropriate parameters passed
- [ ] Verdict is present and correct (PASS/FAIL/UNKNOWN)
- [ ] Confidence level reported
- [ ] Summary is human-readable
- [ ] Suggestions provided on FAIL
- [ ] Metrics are plausible (not obviously wrong)
- [ ] No Python tracebacks in output
- [ ] Response is actionable for the user

---

## Quick Smoke Test

Run these 5 prompts for a quick validation:

1. `Can ResNet-18 meet 10ms latency on H100?` → Expect PASS
2. `Can ResNet-50 run under 1ms on Jetson Orin Nano?` → Expect FAIL
3. `Full analysis of MobileNetV2 on A100` → Expect detailed metrics
4. `List available hardware` → Expect categorized list
5. `Compare ResNet-18 on H100 vs A100` → Expect ranked comparison

---

## Automated Test Runner

For automated testing, see `tests/test_verdict_tools_cli.py` (to be created).

```python
# Example test structure
TEST_CASES = [
    {
        "prompt": "Can ResNet-18 meet 10ms latency on H100?",
        "expected_tool": "check_latency",
        "expected_verdict": "PASS",
        "expected_in_response": ["headroom", "margin"],
    },
    # ... more cases
]
```
