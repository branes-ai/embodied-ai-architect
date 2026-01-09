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

## TorchVisionProvider

TorchVision provider added. Here's the summary:

File created: model_zoo/providers/torchvision.py

  Model Catalog (34 models)

  | Family       | Models                                               |
  |--------------|------------------------------------------------------|
  | ResNet       | resnet18, resnet34, resnet50, resnet101, resnet152   |
  | MobileNet    | mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large |
  | EfficientNet | efficientnet_b0 through b4                           |
  | VGG          | vgg11, vgg16, vgg19                                  |
  | DenseNet     | densenet121, densenet169, densenet201                |
  | SqueezeNet   | squeezenet1_0, squeezenet1_1                         |
  | ShuffleNet   | shufflenet_v2_x0_5, shufflenet_v2_x1_0               |
  | RegNet       | regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf       |
  | ConvNeXt     | convnext_tiny, convnext_small                        |
  | ViT          | vit_b_16, vit_b_32, vit_l_16                         |
  | Swin         | swin_t, swin_s, swin_b                               |

Features

  - Pretrained ImageNet weights from torchvision
  - Export to PyTorch, TorchScript, ONNX formats
  - Full accuracy metrics (top-1, top-5)
  - Parameter counts and FLOPs

Usage

```bash
  # Search small classification models
  branes zoo search --task classification --max-params 5000000

  # Download ResNet-18 as ONNX
  branes zoo download resnet18 --format onnx

  # Get model info
  branes zoo info efficientnet_b0
```

Current Cache

```text
  Cached Models (3)
  ├── mobilenet_v3_small │ torchvision │  9.7 MB
  ├── yolov8n            │ ultralytics │ 12.3 MB
  └── yolov8s            │ ultralytics │ 42.8 MB
  Total: 64.8 MB
```


## HuggingFaceProvider

HuggingFace provider added. Here's the summary:

File created: model_zoo/providers/huggingface.py

Model Catalog (27 models)

  | Task             | Models                                                                             |
  |------------------|------------------------------------------------------------------------------------|
  | Classification   | ViT (base/large), DeiT (tiny/small/base), BEiT, MobileViT (xx-small/x-small/small) |
  | Detection        | DETR (resnet-50/101), Conditional DETR, YOLOS (tiny/small/base)                    |
  | Segmentation     | SegFormer B0-B3 (ADE & Cityscapes)                                                 |
  | Depth Estimation | DPT (hybrid/large), GLPN (KITTI/NYU), Depth Anything (small/base)                  |

Model Zoo Summary

  | Provider    | Models | Focus                            |
  |-------------|--------|----------------------------------|
  | ultralytics | 15     | YOLO detection/segmentation/pose |
  | torchvision | 34     | Classic CNNs, ViT, Swin          |
  | huggingface | 27     | Transformers for vision tasks    |
  | Total       | 76     |                                  |

Current Cache

```text
  Cached Models (4)
  ├── apple/mobilevit-xx-small │ huggingface │  5.1 MB
  ├── mobilenet_v3_small       │ torchvision │  9.7 MB
  ├── yolov8n                  │ ultralytics │ 12.3 MB
  └── yolov8s                  │ ultralytics │ 42.8 MB
  Total: 69.9 MB
```

Example Usage

```bash
  # Search depth estimation models
  branes zoo search --task depth_estimation

  # Download SegFormer for urban scenes
  branes zoo download nvidia/segformer-b0-finetuned-cityscapes-512-1024

  # Get info on a HuggingFace model
  branes zoo info facebook/detr-resnet-50
```

Phase 2 is now complete. Remaining work:
  - Phase 3: Requirements wizard (branes design new)
  - Phase 4: Testbench and pipeline synthesis

## Examples

```bash
(p311) stillwater@sw-21:~/dev/branes/clones/embodied-ai-architect$ branes zoo list
╭─────────────────────────────╮
│ Branes Embodied AI Platform │
│ Version 0.3.0               │
╰─────────────────────────────╯
                                   Available Models (49)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ ID                 ┃ Name                        ┃ Task           ┃ Params ┃ Provider    ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ squeezenet1_1      │ SqueezeNet 1.1              │ classification │   1.2M │ torchvision │
│ squeezenet1_0      │ SqueezeNet 1.0              │ classification │   1.2M │ torchvision │
│ shufflenet_v2_x0_5 │ ShuffleNet V2 x0.5          │ classification │   1.4M │ torchvision │
│ yolov5n            │ YOLOv5 Nano                 │ detection      │   1.9M │ ultralytics │
│ shufflenet_v2_x1_0 │ ShuffleNet V2 x1.0          │ classification │   2.3M │ torchvision │
│ mobilenet_v3_small │ MobileNet V3 Small          │ classification │   2.5M │ torchvision │
│ yolo11n            │ YOLO11 Nano                 │ detection      │   2.6M │ ultralytics │
│ yolov8n-cls        │ YOLOv8 Nano Classification  │ classification │   2.7M │ ultralytics │
│ yolov8n            │ YOLOv8 Nano                 │ detection      │   3.2M │ ultralytics │
│ yolov8n-pose       │ YOLOv8 Nano Pose            │ pose           │   3.3M │ ultralytics │
│ yolov8n-seg        │ YOLOv8 Nano Segmentation    │ segmentation   │   3.4M │ ultralytics │
│ mobilenet_v2       │ MobileNet V2                │ classification │   3.5M │ torchvision │
│ regnet_y_400mf     │ RegNet Y 400MF              │ classification │   4.3M │ torchvision │
│ efficientnet_b0    │ EfficientNet B0             │ classification │   5.3M │ torchvision │
│ mobilenet_v3_large │ MobileNet V3 Large          │ classification │   5.5M │ torchvision │
│ yolov8s-cls        │ YOLOv8 Small Classification │ classification │   6.4M │ ultralytics │
│ regnet_y_800mf     │ RegNet Y 800MF              │ classification │   6.4M │ torchvision │
│ yolov5s            │ YOLOv5 Small                │ detection      │   7.2M │ ultralytics │
│ efficientnet_b1    │ EfficientNet B1             │ classification │   7.8M │ torchvision │
│ densenet121        │ DenseNet-121                │ classification │   8.0M │ torchvision │
│ efficientnet_b2    │ EfficientNet B2             │ classification │   9.1M │ torchvision │
│ yolo11s            │ YOLO11 Small                │ detection      │   9.4M │ ultralytics │
│ yolov8s            │ YOLOv8 Small                │ detection      │  11.2M │ ultralytics │
│ regnet_y_1_6gf     │ RegNet Y 1.6GF              │ classification │  11.2M │ torchvision │
│ yolov8s-pose       │ YOLOv8 Small Pose           │ pose           │  11.6M │ ultralytics │
│ resnet18           │ ResNet-18                   │ classification │  11.7M │ torchvision │
│ yolov8s-seg        │ YOLOv8 Small Segmentation   │ segmentation   │  11.8M │ ultralytics │
│ efficientnet_b3    │ EfficientNet B3             │ classification │  12.2M │ torchvision │
│ densenet169        │ DenseNet-169                │ classification │  14.1M │ torchvision │
│ efficientnet_b4    │ EfficientNet B4             │ classification │  19.3M │ torchvision │
│ densenet201        │ DenseNet-201                │ classification │  20.0M │ torchvision │
│ resnet34           │ ResNet-34                   │ classification │  21.8M │ torchvision │
│ resnet50           │ ResNet-50                   │ classification │  25.6M │ torchvision │
│ yolov8m            │ YOLOv8 Medium               │ detection      │  25.9M │ ultralytics │
│ swin_t             │ Swin Transformer Tiny       │ classification │  28.3M │ torchvision │
│ convnext_tiny      │ ConvNeXt Tiny               │ classification │  28.6M │ torchvision │
│ yolov8l            │ YOLOv8 Large                │ detection      │  43.7M │ ultralytics │
│ resnet101          │ ResNet-101                  │ classification │  44.5M │ torchvision │
│ swin_s             │ Swin Transformer Small      │ classification │  49.6M │ torchvision │
│ convnext_small     │ ConvNeXt Small              │ classification │  50.2M │ torchvision │
│ resnet152          │ ResNet-152                  │ classification │  60.2M │ torchvision │
│ yolov8x            │ YOLOv8 Extra-Large          │ detection      │  68.2M │ ultralytics │
│ vit_b_16           │ ViT Base 16                 │ classification │  86.6M │ torchvision │
│ swin_b             │ Swin Transformer Base       │ classification │  87.8M │ torchvision │
│ vit_b_32           │ ViT Base 32                 │ classification │  88.2M │ torchvision │
│ vgg11              │ VGG-11                      │ classification │ 132.9M │ torchvision │
│ vgg16              │ VGG-16                      │ classification │ 138.4M │ torchvision │
│ vgg19              │ VGG-19                      │ classification │ 143.7M │ torchvision │
│ vit_l_16           │ ViT Large 16                │ classification │ 304.3M │ torchvision │
└────────────────────┴─────────────────────────────┴────────────────┴────────┴─────────────┘
```

After adding the huggingface provider:

```bash
(p311) stillwater@sw-21:~/dev/branes/clones/embodied-ai-architect$ branes zoo list --provider huggingface
╭─────────────────────────────╮
│ Branes Embodied AI Platform │
│ Version 0.3.0               │
╰─────────────────────────────╯
                                                   Available Models (27)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ ID                                                ┃ Name                       ┃ Task             ┃ Params ┃ Provider    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ apple/mobilevit-xx-small                          │ MobileViT XX-Small         │ classification   │   1.3M │ huggingface │
│ apple/mobilevit-x-small                           │ MobileViT X-Small          │ classification   │   2.3M │ huggingface │
│ nvidia/segformer-b0-finetuned-ade-512-512         │ SegFormer B0 ADE           │ segmentation     │   3.7M │ huggingface │
│ nvidia/segformer-b0-finetuned-cityscapes-512-1024 │ SegFormer B0 Cityscapes    │ segmentation     │   3.7M │ huggingface │
│ apple/mobilevit-small                             │ MobileViT Small            │ classification   │   5.6M │ huggingface │
│ facebook/deit-tiny-patch16-224                    │ DeiT Tiny                  │ classification   │   5.7M │ huggingface │
│ hustvl/yolos-tiny                                 │ YOLOS Tiny                 │ detection        │   6.5M │ huggingface │
│ nvidia/segformer-b1-finetuned-ade-512-512         │ SegFormer B1 ADE           │ segmentation     │  13.7M │ huggingface │
│ facebook/deit-small-patch16-224                   │ DeiT Small                 │ classification   │  22.0M │ huggingface │
│ LiheYoung/depth-anything-small-hf                 │ Depth Anything Small       │ depth_estimation │  24.8M │ huggingface │
│ vinvino02/glpn-kitti                              │ GLPN KITTI                 │ depth_estimation │  26.0M │ huggingface │
│ vinvino02/glpn-nyu                                │ GLPN NYU                   │ depth_estimation │  26.0M │ huggingface │
│ nvidia/segformer-b2-finetuned-ade-512-512         │ SegFormer B2 ADE           │ segmentation     │  27.4M │ huggingface │
│ hustvl/yolos-small                                │ YOLOS Small                │ detection        │  30.0M │ huggingface │
│ facebook/detr-resnet-50                           │ DETR ResNet-50             │ detection        │  41.0M │ huggingface │
│ microsoft/conditional-detr-resnet-50              │ Conditional DETR ResNet-50 │ detection        │  44.0M │ huggingface │
│ nvidia/segformer-b3-finetuned-ade-512-512         │ SegFormer B3 ADE           │ segmentation     │  47.3M │ huggingface │
│ facebook/detr-resnet-101                          │ DETR ResNet-101            │ detection        │  60.0M │ huggingface │
│ google/vit-base-patch16-224                       │ ViT Base Patch16 224       │ classification   │  86.0M │ huggingface │
│ facebook/deit-base-patch16-224                    │ DeiT Base                  │ classification   │  86.0M │ huggingface │
│ microsoft/beit-base-patch16-224                   │ BEiT Base                  │ classification   │  86.0M │ huggingface │
│ google/vit-base-patch32-224                       │ ViT Base Patch32 224       │ classification   │  88.0M │ huggingface │
│ LiheYoung/depth-anything-base-hf                  │ Depth Anything Base        │ depth_estimation │  97.5M │ huggingface │
│ Intel/dpt-hybrid-midas                            │ DPT Hybrid MiDaS           │ depth_estimation │ 123.0M │ huggingface │
│ hustvl/yolos-base                                 │ YOLOS Base                 │ detection        │ 127.0M │ huggingface │
│ google/vit-large-patch16-224                      │ ViT Large Patch16 224      │ classification   │ 304.0M │ huggingface │
│ Intel/dpt-large                                   │ DPT Large                  │ depth_estimation │ 343.0M │ huggingface │
└───────────────────────────────────────────────────┴────────────────────────────┴──────────────────┴────────┴─────────────┘
```

```bash
(p311) stillwater@sw-21:~/dev/branes/clones/embodied-ai-architect$ branes zoo list --provider ultralytics
╭─────────────────────────────╮
│ Branes Embodied AI Platform │
│ Version 0.3.0               │
╰─────────────────────────────╯
                                Available Models (15)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ ID           ┃ Name                        ┃ Task           ┃ Params ┃ Provider    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ yolov5n      │ YOLOv5 Nano                 │ detection      │   1.9M │ ultralytics │
│ yolo11n      │ YOLO11 Nano                 │ detection      │   2.6M │ ultralytics │
│ yolov8n-cls  │ YOLOv8 Nano Classification  │ classification │   2.7M │ ultralytics │
│ yolov8n      │ YOLOv8 Nano                 │ detection      │   3.2M │ ultralytics │
│ yolov8n-pose │ YOLOv8 Nano Pose            │ pose           │   3.3M │ ultralytics │
│ yolov8n-seg  │ YOLOv8 Nano Segmentation    │ segmentation   │   3.4M │ ultralytics │
│ yolov8s-cls  │ YOLOv8 Small Classification │ classification │   6.4M │ ultralytics │
│ yolov5s      │ YOLOv5 Small                │ detection      │   7.2M │ ultralytics │
│ yolo11s      │ YOLO11 Small                │ detection      │   9.4M │ ultralytics │
│ yolov8s      │ YOLOv8 Small                │ detection      │  11.2M │ ultralytics │
│ yolov8s-pose │ YOLOv8 Small Pose           │ pose           │  11.6M │ ultralytics │
│ yolov8s-seg  │ YOLOv8 Small Segmentation   │ segmentation   │  11.8M │ ultralytics │
│ yolov8m      │ YOLOv8 Medium               │ detection      │  25.9M │ ultralytics │
│ yolov8l      │ YOLOv8 Large                │ detection      │  43.7M │ ultralytics │
│ yolov8x      │ YOLOv8 Extra-Large          │ detection      │  68.2M │ ultralytics │
└──────────────┴─────────────────────────────┴────────────────┴────────┴─────────────┘
```

## Other Providers

Are there other Model Zoo providers of interest besides HuggingFace, TorchVision, and Ultralytics?

Here are other notable model sources relevant for embodied AI:

High-Value Providers

  | Provider                    | Models | Why It's Useful                                                                                    |
  |-----------------------------|--------|----------------------------------------------------------------------------------------------------|
  | Timm (PyTorch Image Models) | 700+   | Ross Wightman's library. Most comprehensive vision model collection. Many edge-optimized variants. |
  | ONNX Model Zoo              | 50+    | Already in ONNX format. Validated, deployment-ready.                                               |
  | OpenMMLab                   | 200+   | MMDetection, MMSegmentation, MMPose. State-of-the-art detection/segmentation.                      |
  | NVIDIA NGC                  | 100+   | TensorRT-optimized. TAO pretrained models for Jetson.                                              |
  | MediaPipe                   | 20+    | Google's on-device models. Hands, face, pose, object detection. Very lightweight.                  |

Edge/Embedded Focused

  | Provider                  | Focus                         |
  |---------------------------|-------------------------------|
  | Qualcomm AI Hub           | Snapdragon-optimized models   |
  | Intel OpenVINO Zoo        | Intel CPU/VPU optimized       |
  | Apple CoreML Models       | Apple Neural Engine           |
  | TensorFlow Lite Model Zoo | Mobile/embedded TFLite models |

Robotics-Specific

  | Provider      | Focus                      |
  |---------------|----------------------------|
  | ROS Model Zoo | Robotics perception models |
  | NVIDIA Isaac  | Sim-to-real, manipulation  |
  | Open3D-ML     | 3D point cloud models      |

### Recommendation

For this project, I'd prioritize:

  1. Timm - Massive value, easy integration (it's just PyTorch)
  2. ONNX Model Zoo - Already deployment-ready
  3. MediaPipe - Excellent for real-time embodied AI (pose, hands, objects)

## TIMM Provider

Timm provider added. Here's the summary:

File created: model_zoo/providers/timm.py

Curated Model Catalog (53 models)

  | Family          | Models                        | Notable                             |
  |-----------------|-------------------------------|-------------------------------------|
  | EfficientNet    | B0-B3, Lite0, V2-S/M          | Google's efficient CNNs             |
  | FastViT         | T8, T12, S12, SA12, SA24      | Apple's state-of-the-art mobile ViT |
  | EdgeNeXt        | XX-Small, X-Small, Small      | Samsung's edge architecture         |
  | ConvNeXtV2      | Atto, Femto, Pico, Nano, Tiny | Meta's modern pure ConvNet          |
  | EfficientFormer | L1, L3, V2-S0/S1/S2           | Snap's efficient ViT                |
  | MobileViTV2     | 0.5, 0.75, 1.0, 1.5           | Apple's improved MobileViT          |
  | MobileNetV3/V4  | Small, Large, V4 variants     | Google's mobile CNNs                |
  | GhostNet        | V1, V2 (1.0/1.3/1.6)          | Huawei's efficient features         |
  | RepVGG          | A0, A1, A2, B0                | Fast inference via reparam          |
  | TinyNet         | A, B, C, D, E                 | NAS-optimized tiny models           |
  | LCNet           | 0.5, 0.75, 1.0                | Baidu's CPU-optimized               |
  | MaxViT          | Tiny, Small                   | Multi-axis attention                |

Model Zoo Summary

  | Provider    | Models | Focus                                |
  |-------------|--------|--------------------------------------|
  | ultralytics | 15     | YOLO detection/segmentation/pose     |
  | torchvision | 34     | Classic CNNs, ViT, Swin              |
  | huggingface | 27     | Transformers for vision tasks        |
  | timm        | 53     | Edge-optimized, modern architectures |
  | Total       | 129    |                                      |

Edge-Optimized Models (< 3M params)

```
  $ branes zoo search --max-params 3000000
  Available Models (22 found)
  ├── edgenext_xx_small     │ 1.3M │ 71.2% │ timm
  ├── mobilevitv2_050       │ 1.4M │ 70.3% │ timm
  ├── tinynet_d             │ 1.8M │ 66.7% │ timm
  ├── apple/mobilevit-xx-sm │ 1.3M │ 69.0% │ huggingface
  └── ...
```

Current Cache

```
  Cached Models (5)
  ├── apple/mobilevit-xx-small │ huggingface │  5.1 MB
  ├── edgenext_xx_small.in1k   │ timm        │  5.2 MB
  ├── mobilenet_v3_small       │ torchvision │  9.7 MB
  ├── yolov8n                  │ ultralytics │ 12.3 MB
  └── yolov8s                  │ ultralytics │ 42.8 MB
  Total: 75.1 MB
```

## ONNX Model Zoo

ONNX Model Zoo provider added. Here's the summary:

File created: model_zoo/providers/onnx_zoo.py

Model Catalog (30 models)

  | Task             | Models                                                                                            |
  |------------------|---------------------------------------------------------------------------------------------------|
  | Classification   | ResNet (18/50/101 v1/v2), MobileNetV2, SqueezeNet, ShuffleNet, VGG, GoogLeNet, EfficientNet Lite4 |
  | Detection        | YOLOv3, YOLOv4, Tiny-YOLOv3, SSD-MobileNet, Faster R-CNN, RetinaNet                               |
  | Segmentation     | FCN-ResNet50, FCN-ResNet101                                                                       |
  | Face Detection   | UltraFace (320/640) - ultra-lightweight                                                           |
  | Face Recognition | ArcFace                                                                                           |
  | Pose Estimation  | HRNet                                                                                             |
  | Depth Estimation | MiDaS v2, MiDaS v2 Small                                                                          |
  | Style Transfer   | Neural Style (mosaic, candy, pointilism)                                                          |
  | Super Resolution | ESPCN                                                                                             |

Key Advantage

  No conversion needed - models are already in validated ONNX format, ready for deployment.

Model Zoo Summary

  | Provider    | Models | Focus                            |
  |-------------|--------|----------------------------------|
  | ultralytics | 15     | YOLO detection/segmentation/pose |
  | torchvision | 34     | Classic CNNs, ViT, Swin          |
  | huggingface | 27     | Transformers for vision          |
  | timm        | 53     | Edge-optimized architectures     |
  | onnx-zoo    | 30     | Deployment-ready ONNX            |
  | Total       | 159    |                                  |

Current Cache

```
  Cached Models (6)
  ├── ultraface-320            │ onnx-zoo    │  1.2 MB
  ├── apple/mobilevit-xx-small │ huggingface │  5.1 MB
  ├── edgenext_xx_small.in1k   │ timm        │  5.2 MB
  ├── mobilenet_v3_small       │ torchvision │  9.7 MB
  ├── yolov8n                  │ ultralytics │ 12.3 MB
  └── yolov8s                  │ ultralytics │ 42.8 MB
  Total: 76.3 MB
```
