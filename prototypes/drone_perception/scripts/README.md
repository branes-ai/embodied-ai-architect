# Test Scripts

Automation scripts for testing and validation of the drone perception pipeline.

## Available Scripts

### 1. download_test_videos.py

Download test videos from public sources.

**Usage:**

```bash
# Quick test suite (recommended for getting started)
python scripts/download_test_videos.py --suite quick

# Download specific video
python scripts/download_test_videos.py --id traffic_001

# Download all traffic videos
python scripts/download_test_videos.py --category traffic

# Download everything
python scripts/download_test_videos.py --all
```

**Test Suites:**
- `quick`: 3 videos, ~50MB, 5-10 min processing time
- `traffic_focus`: Traffic scenarios only
- `pedestrian_focus`: Pedestrian tracking scenarios
- `comprehensive`: Full test suite, ~200MB, 30+ min processing

**Notes:**
- Videos are saved to `test_data/videos/`
- Some videos require manual download (VisDrone dataset)
- For YouTube videos, install: `pip install yt-dlp`

### 2. run_test_suite.py

Run the perception pipeline on test videos and collect metrics.

**Usage:**

```bash
# Run quick test suite
python scripts/run_test_suite.py --suite quick

# Run on specific video
python scripts/run_test_suite.py --video test_data/videos/traffic/traffic_001.mp4

# Use GPU with larger model
python scripts/run_test_suite.py --suite quick --device cuda --model s

# Run comprehensive suite
python scripts/run_test_suite.py --suite comprehensive --device cuda
```

**Output:**
- Results saved to `test_data/results/`
- JSON format with detailed metrics
- Per-frame statistics included

**Metrics Collected:**
- FPS (frames per second)
- Detection counts per frame
- Track counts per frame
- 3D object counts per frame
- Processing time per frame

### 3. create_baseline.py (TODO)

Create baseline results for regression testing.

**Usage:**
```bash
python scripts/create_baseline.py --suite quick --output baselines/v1.0.json
```

### 4. compare_results.py (TODO)

Compare current results against baseline for regression detection.

**Usage:**
```bash
python scripts/compare_results.py \
    --baseline baselines/v1.0.json \
    --current test_data/results/results_quick_20250101_120000.json \
    --output comparison.html
```

### 5. generate_report.py (TODO)

Generate HTML report from test results.

**Usage:**
```bash
python scripts/generate_report.py \
    --results test_data/results/ \
    --output test_report.html
```

## Typical Workflow

### Initial Setup

```bash
# 1. Download test videos
python scripts/download_test_videos.py --suite quick

# 2. Verify downloads
ls -lh test_data/videos/*/*.mp4
```

### Development Testing

```bash
# Quick smoke test during development
python scripts/run_test_suite.py --suite quick --device cpu --model n

# Check results
cat test_data/results/results_quick_*.json
```

### Pre-Commit Validation

```bash
# Run comprehensive suite before committing changes
python scripts/run_test_suite.py --suite comprehensive --device cuda --model n

# Compare with baseline
python scripts/compare_results.py \
    --baseline baselines/current.json \
    --current test_data/results/results_comprehensive_*.json
```

### Benchmarking

```bash
# Benchmark different models
for model in n s m; do
    python scripts/run_test_suite.py \
        --suite quick \
        --device cuda \
        --model $model \
        --output test_data/results/benchmark_${model}/
done

# Generate comparison report
python scripts/generate_report.py --results test_data/results/benchmark_*
```

## Adding Custom Scripts

When adding new test scripts:

1. Place in `scripts/` directory
2. Make executable: `chmod +x scripts/your_script.py`
3. Add shebang: `#!/usr/bin/env python3`
4. Add to this README
5. Update `.gitignore` if generating large files

## Dependencies

Core scripts require:
- `pyyaml` - For reading catalog
- `urllib` - For downloading files (built-in)

Optional:
- `yt-dlp` - For YouTube video downloads
- `matplotlib` - For report generation
- `pandas` - For result analysis

Install all:
```bash
pip install pyyaml yt-dlp matplotlib pandas
```

## CI/CD Integration

For continuous integration, use:

```bash
# In CI pipeline
python scripts/download_test_videos.py --suite quick
python scripts/run_test_suite.py --suite quick --device cpu --model n
python scripts/compare_results.py --baseline baseline.json --current results/
```

Exit codes:
- `0` - Success
- `1` - Failure (downloads failed, metrics degraded, etc.)

## Troubleshooting

**Downloads fail:**
- Check internet connection
- Some sources may require API keys or cookies
- Use `--force` to re-download

**Out of disk space:**
- Videos are large (50MB-500MB per video)
- Clean old results: `rm -rf test_data/results/*`
- Download only needed suites

**Slow processing:**
- Use GPU: `--device cuda`
- Use smaller model: `--model n`
- Reduce test suite size

**Results inconsistent:**
- Different random seeds (YOLO augmentation)
- CPU vs GPU differences
- Use same config for comparisons
