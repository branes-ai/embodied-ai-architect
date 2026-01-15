---
title: CLI Reference
description: Complete reference for the embodied-ai command-line interface.
---

## Global Options

```bash
embodied-ai [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help and exit |

## Commands

### analyze

Analyze a model's structure and characteristics.

```bash
embodied-ai analyze MODEL [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--hardware TEXT` | Target hardware for analysis |
| `--input-shape TEXT` | Input tensor shape (e.g., 1,3,640,640) |
| `--batch-size INT` | Batch size for analysis |
| `--precision TEXT` | Precision (fp32, fp16, int8) |

### benchmark

Benchmark model performance on target backends.

```bash
embodied-ai benchmark MODEL [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--backend TEXT` | Backend (local, ssh, kubernetes) |
| `--iterations INT` | Number of iterations |
| `--warmup INT` | Warmup iterations |

### chat

Start interactive AI architect session.

```bash
embodied-ai chat [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--model TEXT` | Claude model to use |
| `--verbose` | Show tool calls |

Requires `ANTHROPIC_API_KEY` environment variable.

### workflow

Run the full analysis workflow.

```bash
embodied-ai workflow run MODEL [OPTIONS]
```

### deploy

Deploy a model to target hardware.

```bash
embodied-ai deploy MODEL [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--target TEXT` | Target (jetson, openvino, coral) |
| `--precision TEXT` | Precision (fp32, fp16, int8) |
| `--input-shape TEXT` | Input shape |
| `--calibration-data PATH` | Calibration images for INT8 |
| `--output-dir PATH` | Output directory |

### report

View and manage analysis reports.

```bash
embodied-ai report [list|view|export] [OPTIONS]
```

### backends

Manage benchmark backends.

```bash
embodied-ai backends [list|add|remove] [OPTIONS]
```

### secrets

Manage credentials for remote backends.

```bash
embodied-ai secrets [set|get|delete] [OPTIONS]
```
