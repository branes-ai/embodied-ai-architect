# AMD Ryzen AI NUC Benchmark Procedures

Baseline benchmark procedures for measuring operator performance on AMD Ryzen AI NUC platforms (Ryzen 7 8845HS and Ryzen 9 8945HS).

## Target Platforms

| Platform | CPU | GPU | NPU | TDP Range |
|----------|-----|-----|-----|-----------|
| amd_ryzen_7_8845hs_nuc | 8-core Zen 4 (3.8-5.1 GHz) | Radeon 780M (12 CUs) | XDNA 16 TOPS | 15-45W |
| amd_ryzen_9_8945hs_nuc | 8-core Zen 4 (4.0-5.2 GHz) | Radeon 780M (12 CUs) | XDNA 16 TOPS | 15-54W |

## Software Stack Requirements

### Base System

```bash
# Ubuntu 22.04 or 24.04
uname -a

# Python 3.10+
python3 --version
```

### ROCm Installation (GPU)

```bash
# ROCm 6.0+ for AMD GPU compute
# Check installation
rocminfo
hipconfig --full
```

### Ryzen AI SDK Installation (NPU)

```bash
# Check NPU driver
ls /dev/ryzen_ai*

# Verify ONNX Runtime providers
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should include: 'RyzenAIExecutionProvider'
```

## Benchmark Methodology

### General Principles

1. **Warmup**: Run 10 iterations before timing
2. **Iterations**: Minimum 100 iterations for statistics
3. **Isolation**: Close other applications during measurement
4. **TDP Mode**: Document the power mode (15W, 28W, 45W, etc.)
5. **Thermal**: Allow system to reach thermal equilibrium

### Timing Measurement

```python
import time
import statistics

def benchmark_operator(operator, inputs, warmup=10, iterations=100):
    """Run benchmark with proper warmup and statistics."""

    # Warmup
    for _ in range(warmup):
        operator.process(inputs)

    # Timed runs
    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operator.process(inputs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        timings.append(elapsed_ms)

    return {
        "mean_ms": statistics.mean(timings),
        "std_ms": statistics.stdev(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": sorted(timings)[int(0.95 * len(timings))],
        "min_ms": min(timings),
        "max_ms": max(timings),
    }
```

## CPU Benchmarks

### Single-Thread Latency

Measures CPU-bound operator performance on a single core.

```bash
# Pin to single core for consistency
taskset -c 0 python benchmark_cpu.py
```

**Operators to benchmark:**
- ByteTrack tracker
- Kalman filter (2D bbox)
- Scene graph manager
- PID controller
- A* path planner

### Multi-Thread Scaling

Measure how operators scale with thread count.

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Then 2, 4, 8
```

### Memory Bandwidth (STREAM)

```bash
# Install and run STREAM benchmark
./stream_c.exe
# Record: Copy, Scale, Add, Triad bandwidth (GB/s)
```

Expected DDR5-5600 dual-channel: ~89.6 GB/s theoretical, ~70-75 GB/s measured.

## GPU Benchmarks (ROCm/HIP)

### Setup

```python
import torch

# Verify ROCm
print(torch.cuda.is_available())  # True for ROCm
print(torch.cuda.get_device_name(0))  # AMD Radeon 780M
```

### Inference Latency

```python
def benchmark_gpu_model(model_path, input_shape, precision="fp16"):
    import torch

    device = torch.device("cuda")
    model = load_model(model_path).to(device)
    if precision == "fp16":
        model = model.half()

    dummy_input = torch.randn(input_shape, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
    torch.cuda.synchronize()

    # Timed
    timings = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.no_grad():
            model(dummy_input)
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start) / 1e6)

    return statistics.mean(timings)
```

### Models to Benchmark

| Model | Input Shape | Precision | Notes |
|-------|-------------|-----------|-------|
| YOLOv8n | (1, 3, 640, 640) | FP16 | Nano detector |
| YOLOv8s | (1, 3, 640, 640) | FP16 | Small detector |
| ResNet-50 | (1, 3, 224, 224) | FP16 | Classification |
| MobileNet-V3 | (1, 3, 224, 224) | FP16 | Lightweight |

### Batch Size Sensitivity

```python
for batch_size in [1, 2, 4, 8, 16]:
    input_shape = (batch_size, 3, 640, 640)
    latency = benchmark_gpu_model(model, input_shape)
    throughput = batch_size * 1000 / latency
    print(f"Batch {batch_size}: {latency:.2f}ms, {throughput:.1f} fps")
```

## NPU Benchmarks (ONNX Runtime + Ryzen AI EP)

### Setup

```python
import onnxruntime as ort

# Configure NPU execution
providers = ['RyzenAIExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# Verify NPU is being used
print(session.get_providers())
```

### INT8 Quantized Models

```bash
# Quantize model for NPU
python -m onnxruntime.quantization.quantize \
    --input model.onnx \
    --output model_int8.onnx \
    --per_channel \
    --quant_format QDQ
```

### Inference Benchmark

```python
def benchmark_npu_model(onnx_path, input_shape):
    providers = ['RyzenAIExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Timed
    timings = []
    for _ in range(100):
        start = time.perf_counter_ns()
        session.run(None, {input_name: dummy_input})
        timings.append((time.perf_counter_ns() - start) / 1e6)

    return statistics.mean(timings)
```

### Supported vs Fallback Ops

Not all operations run on NPU. Monitor which ops fall back to CPU:

```python
# Enable verbose logging
ort.set_default_logger_severity(0)

# Check execution plan
session = ort.InferenceSession(model_path, providers=providers)
# Review logs for "Falling back to CPU" messages
```

### Models to Benchmark

| Model | Precision | Expected NPU Ops |
|-------|-----------|------------------|
| YOLOv8n | INT8 | Conv2D, MatMul |
| ResNet-50 | INT8 | Conv2D, BatchNorm, Add |
| MobileNet-V3 | INT8 | Conv2D, Hardswish, SE |
| BERT-base | INT8 | MatMul, Attention, LayerNorm |

## Power Measurement

### RAPL (Software)

```python
def read_rapl_energy():
    """Read CPU package energy via RAPL."""
    with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj") as f:
        return int(f.read())

def measure_power(func, duration_s=5.0):
    """Measure average power during function execution."""
    start_energy = read_rapl_energy()
    start_time = time.time()

    while time.time() - start_time < duration_s:
        func()

    end_energy = read_rapl_energy()
    end_time = time.time()

    energy_j = (end_energy - start_energy) / 1e6
    power_w = energy_j / (end_time - start_time)
    return power_w
```

### Ryzen SMU (Hardware)

```bash
# Install ryzen_smu for AMD-specific monitoring
sudo ryzen_smu --power-monitor --interval 100
```

### External Power Meter (Ground Truth)

For accurate total system power, use USB power meter on DC input:
- Record idle power
- Record load power during benchmark
- Calculate delta for component power

## TDP Mode Testing

Test at multiple power modes to characterize power/performance trade-offs:

```bash
# Set TDP via BIOS or OS tools
# Ryzen 7: 15W, 28W, 45W
# Ryzen 9: 15W, 35W, 54W

# Document TDP mode for each benchmark run
```

## Results Template

```yaml
benchmark_result:
  hardware_id: amd_ryzen_7_8845hs_nuc
  operator_id: yolo_detector_n
  execution_target: npu

  conditions:
    tdp_mode: "28W"
    ambient_temp_c: 22
    input_resolution: "640x640"
    batch_size: 1
    precision: INT8
    runtime: "ONNX Runtime 1.18 + Ryzen AI EP"

  results:
    mean_latency_ms: 12.5
    std_latency_ms: 0.8
    p50_latency_ms: 12.3
    p95_latency_ms: 14.1
    memory_mb: 180
    power_w: 3.2
    throughput_fps: 80

  measured: true
  date: "2026-01-01"
  notes: "Production Ryzen AI SDK 1.2"
```

## Automation Scripts

### Run Full Benchmark Suite

```bash
#!/bin/bash
# benchmark_suite.sh

export HARDWARE_ID="amd_ryzen_7_8845hs_nuc"
export TDP_MODE="28W"

# CPU operators
python benchmark_cpu.py --operator bytetrack
python benchmark_cpu.py --operator kalman_filter_2d
python benchmark_cpu.py --operator scene_graph
python benchmark_cpu.py --operator pid_controller
python benchmark_cpu.py --operator path_planner_astar

# GPU operators
python benchmark_gpu.py --model yolov8n --precision fp16
python benchmark_gpu.py --model yolov8s --precision fp16

# NPU operators
python benchmark_npu.py --model yolov8n_int8.onnx
python benchmark_npu.py --model yolov8s_int8.onnx

# Generate report
python generate_report.py --hardware $HARDWARE_ID
```

## Validation Checklist

Before reporting results:

- [ ] System is thermally stable (ran for >5 minutes)
- [ ] TDP mode is documented
- [ ] No other heavy processes running
- [ ] ONNX Runtime/ROCm versions documented
- [ ] Multiple runs show consistent results (std < 10% of mean)
- [ ] Power measurement method documented
- [ ] Input shapes/conditions match operator catalog

## References

- [AMD Ryzen AI SDK Documentation](https://www.amd.com/en/developer/resources/ryzen-ai-software.html)
- [ONNX Runtime Ryzen AI EP](https://onnxruntime.ai/docs/execution-providers/RyzenAI-ExecutionProvider.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
