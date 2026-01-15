---
title: Model Catalog
description: Browse supported neural network models.
---

This catalog lists neural network models with pre-computed analysis data.

:::note
This page will be auto-generated from the `embodied-schemas` model catalog.
:::

## Object Detection

| Model | Parameters | FLOPs | Accuracy | Status |
|-------|------------|-------|----------|--------|
| YOLOv8n | 3.2M | 8.7G | 37.3 mAP | Full support |
| YOLOv8s | 11.2M | 28.6G | 44.9 mAP | Full support |
| YOLOv8m | 25.9M | 78.9G | 50.2 mAP | Full support |
| RT-DETR-L | 32M | 110G | 53.0 mAP | Full support |

## Classification

| Model | Parameters | FLOPs | Top-1 Acc | Status |
|-------|------------|-------|-----------|--------|
| ResNet-18 | 11.7M | 1.8G | 69.8% | Full support |
| ResNet-50 | 25.6M | 4.1G | 76.1% | Full support |
| MobileNetV2 | 3.4M | 0.3G | 72.0% | Full support |
| EfficientNet-B0 | 5.3M | 0.4G | 77.1% | Full support |
| ViT-B/16 | 86M | 17.6G | 81.8% | Full support |

## Segmentation

| Model | Parameters | FLOPs | mIoU | Status |
|-------|------------|-------|------|--------|
| DeepLabV3-ResNet50 | 42M | 178G | 77.9% | Partial support |
| SegFormer-B0 | 3.8M | 8.4G | 76.2% | Partial support |

## Depth Estimation

| Model | Parameters | FLOPs | Status |
|-------|------------|-------|--------|
| MiDaS-Small | 21M | 14G | Partial support |
| DPT-Hybrid | 123M | 47G | Partial support |

---

## Adding New Models

To add a model to the catalog:

1. Create YAML in `embodied-schemas/data/models/`
2. Include architecture info, accuracy benchmarks
3. Submit PR with paper/model zoo links
