# DeploymentAgent Implementation Plan

## Overview

Add a `DeploymentAgent` that deploys models to edge devices (Jetson first) with INT8 quantization and accuracy validation.

## Target Outcome

```bash
# Deploy YOLOv8n to Jetson with INT8 quantization
branes deploy run yolov8n.pt --target jetson --precision int8 \
  --calibration-data ./calib_images --input-shape 1,3,640,640 \
  --test-data ./test_images --validate

# Output: model_int8.engine (TensorRT) + validation report
```

---

## File Structure

```
src/embodied_ai_architect/
├── agents/
│   ├── __init__.py                    # Add DeploymentAgent export
│   └── deployment/
│       ├── __init__.py                # Package init with conditional imports
│       ├── agent.py                   # DeploymentAgent class
│       ├── models.py                  # Pydantic models (DeploymentResult, etc.)
│       └── targets/
│           ├── __init__.py            # Target registry
│           ├── base.py                # DeploymentTarget ABC
│           └── jetson.py              # JetsonTarget (TensorRT)
├── cli/commands/
│   └── deploy.py                      # CLI: branes deploy run/list
└── llm/
    └── tools.py                       # Add deploy_model tool
```

---

## Implementation Steps

### Step 1: Data Models (`agents/deployment/models.py`)

Create Pydantic models:
- `DeploymentPrecision` - Enum: fp32, fp16, int8
- `CalibrationConfig` - data_path, num_samples, batch_size, preprocessing
- `ValidationConfig` - test_data_path, tolerance_percent, num_samples
- `DeploymentArtifact` - engine_path, precision, size_bytes, input/output shapes
- `ValidationResult` - passed, speedup, max_output_diff, latencies
- `DeploymentResult` - success, artifact, validation, logs

### Step 2: DeploymentTarget Base (`agents/deployment/targets/base.py`)

Abstract base class:
```python
class DeploymentTarget(ABC):
    def is_available(self) -> bool: ...
    def get_capabilities(self) -> Dict[str, Any]: ...
    def deploy(model, precision, output_path, calibration) -> DeploymentArtifact: ...
    def validate(artifact, baseline, config) -> ValidationResult: ...
```

### Step 3: JetsonTarget (`agents/deployment/targets/jetson.py`)

TensorRT implementation:
1. **is_available()**: Check `import tensorrt` succeeds
2. **deploy()**:
   - Parse ONNX with `trt.OnnxParser`
   - Configure precision flags (FP16/INT8)
   - For INT8: Create `IInt8EntropyCalibrator2` with calibration images
   - Build and serialize engine to `.engine` file
3. **validate()**:
   - Load baseline (ONNX Runtime) and deployed (TensorRT Runtime)
   - Compare outputs on test samples
   - Measure latency speedup
   - Return pass/fail based on tolerance

### Step 4: DeploymentAgent (`agents/deployment/agent.py`)

Orchestrates the workflow:
```python
def execute(input_data) -> AgentResult:
    1. Validate inputs (model, target, precision, input_shape)
    2. Export to ONNX if needed (torch.onnx.export)
    3. Call target.deploy() with calibration config
    4. Call target.validate() if test_data provided
    5. Return AgentResult with artifact and validation
```

### Step 5: CLI Command (`cli/commands/deploy.py`)

Click command group:
- `branes deploy run <model> --target --precision --calibration-data --input-shape`
- `branes deploy list` - Show available targets

### Step 6: LLM Tool (`llm/tools.py`)

Add `deploy_model` tool:
```python
deploy_model(model_path, input_shape, target="jetson", precision="int8",
             calibration_data=None, test_data=None)
```

### Step 7: Update Package Inits

- `agents/__init__.py` - Export DeploymentAgent (with try/except)
- `cli/__init__.py` - Register deploy command
- `pyproject.toml` - Add `jetson` optional dependency group

---

## INT8 Calibration Workflow

```
Calibration Images → Preprocessor → GPU Memory
                                        ↓
                              TensorRT Calibrator
                              (IInt8EntropyCalibrator2)
                                        ↓
                              Collect activation ranges
                                        ↓
                              Compute optimal scales
                                        ↓
                              Cache to calibration.cache
```

**Requirements:**
- 100-500 representative images
- Same preprocessing as inference
- Covers input distribution

---

## Validation Workflow

```
For each test sample:
  ├─ Baseline (ONNX Runtime) → output_baseline, latency_baseline
  └─ Deployed (TensorRT)     → output_deployed, latency_deployed
                                        ↓
                              Compare outputs (max diff)
                              Calculate speedup
                                        ↓
                              Pass if diff < tolerance (1%)
```

---

## Optional Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
jetson = [
    "tensorrt>=8.6.0",
    "pycuda>=2022.1",
    "onnxruntime-gpu>=1.16.0",
]
```

Install: `pip install embodied-ai-architect[jetson]`

---

## Key Files to Modify

| File | Change |
|------|--------|
| `agents/__init__.py` | Add DeploymentAgent export |
| `cli/__init__.py` | Register deploy command |
| `llm/tools.py` | Add deploy_model tool definition + executor |
| `pyproject.toml` | Add jetson dependency group |

---

## Verification

1. **Unit test**: Create `tests/agents/test_deployment.py`
   - Test model export to ONNX
   - Test target registration (mocked TensorRT)
   - Test validation logic

2. **Integration test** (requires TensorRT):
   ```bash
   branes deploy run tests/fixtures/resnet18.pt \
     --target jetson --precision int8 \
     --calibration-data tests/fixtures/calib \
     --input-shape 1,3,224,224 \
     --test-data tests/fixtures/test
   ```

3. **CLI smoke test**:
   ```bash
   branes deploy list  # Should show available targets
   ```

4. **LLM tool test**:
   ```python
   from embodied_ai_architect.llm.tools import create_tool_executors
   executors = create_tool_executors()
   result = executors["deploy_model"](
       model_path="model.onnx",
       input_shape=[1, 3, 224, 224],
       target="jetson",
       precision="fp16"  # No calibration needed
   )
   ```
