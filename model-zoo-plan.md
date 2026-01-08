# Model Zoo Plan

 Phase 1 of the Model Zoo is complete. Here's what was implemented:

  Model Zoo Implementation Summary

  New Files Created

  | File                               | Purpose                                                   |
  |------------------------------------|-----------------------------------------------------------|
  | model_zoo/__init__.py              | Public API exports acquire(), discover()                  |
  | model_zoo/providers/base.py        | ModelProvider ABC, ModelFormat, ModelQuery, ModelArtifact |
  | model_zoo/providers/ultralytics.py | YOLO models catalog (15 models with specs)                |
  | model_zoo/cache.py                 | Versioned cache manager at ~/.cache/branes/models/        |
  | model_zoo/discovery.py             | ModelDiscoveryService with provider aggregation           |
  | model_zoo/acquisition.py           | Unified acquire() API with provider auto-detection        |
  | cli/commands/zoo.py                | CLI commands: search, download, list, info, clear         |

  CLI Commands Available

  branes zoo search --task detection --max-params 5000000
  branes zoo download yolov8n --format onnx
  branes zoo list --cached
  branes zoo info yolov8n
  branes zoo clear

  Key Features

  - 15 YOLO models in catalog (v5, v8, v11 variants for detection, segmentation, pose, classification)
  - Automatic caching at ~/.cache/branes/models/
  - Format support: ONNX, PyTorch, TorchScript, TensorRT, CoreML, OpenVINO
  - Provider auto-detection from model names
  - embodied-schemas integration hooks (ready for catalog enrichment)

  Remaining Work (Phases 2-4)

  - TorchVision and HuggingFace providers
  - Operator integration (update YOLOv8ONNX.setup() to use acquire())
  - Requirements wizard (branes design new)
  - Testbench integration for accuracy validation


## YOLO operator integration complete.

The YOLOv8ONNX operator now uses the model zoo for automatic model acquisition:

```python
  # Before (old code - removed)
  def _get_or_download_model(self) -> Path:
      cache_dir = Path.home() / ".cache" / "embodied-ai" / "models"
      # ... 30 lines of download logic

  # After (new code)
  from embodied_ai_architect.model_zoo import acquire
  model_path = acquire(f"yolov8{self.variant}", format="onnx")
```

Key benefits:
  - Unified caching at ~/.cache/branes/models/ (not scattered across locations)
  - Automatic cache management with version tracking
  - CLI visibility - downloaded models appear in branes zoo list --cached
  - Simplified operator code - removed 33 lines of download logic

Verified working:
```bash
  $ python -c "from embodied_ai_architect.operators.perception.yolo_onnx import YOLOv8ONNX; op = YOLOv8ONNX('n'); op.setup({})"
  [YOLOv8ONNX] Loading ~/.cache/branes/models/ultralytics/yolov8n.onnx with providers: ['CPUExecutionProvider']
  [YOLOv8ONNX] Ready on cpu (input: [1, 3, 640, 640])
```

