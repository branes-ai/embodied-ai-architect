# Test Data Setup - Quick Reference

Complete guide for setting up and using test videos.

## Quick Start (5 minutes)

```bash
cd prototypes/drone_perception

# 1. Install dependencies (including yaml)
pip install -r requirements.txt

# 2. Download quick test suite (~50MB, 3 videos)
python scripts/download_test_videos.py --suite quick

# 3. Run tests
python scripts/run_test_suite.py --suite quick

# 4. Check results
ls -lh test_data/results/
```

## Video Sources

We've cataloged test videos from multiple sources:

### 1. **VisDrone Dataset** (Academic Benchmark)
- **URL**: https://github.com/VisDrone/VisDrone-Dataset
- **Content**: 288 video clips with ground truth annotations
- **Classes**: Cars, pedestrians, bicycles, tricycles
- **Format**: MP4, 1920x1080, 30fps
- **Use**: Quantitative evaluation, regression testing
- **License**: Academic use
- **Note**: Requires manual download and registration

### 2. **Pixabay** (Free Stock Footage)
- **URL**: https://pixabay.com/videos/
- **Content**: Drone traffic, pedestrian, aerial footage
- **Quality**: HD/4K, royalty-free
- **Use**: Development testing, demos
- **License**: Pixabay License (Free for commercial use)

### 3. **Mixkit** (Free Stock Videos)
- **URL**: https://mixkit.co/free-stock-video/
- **Content**: Traffic, city scenes, intersections
- **Quality**: HD, no watermark
- **Use**: Real-world validation
- **License**: Mixkit License (Free)

### 4. **Videezy** (Creative Commons)
- **URL**: https://www.videezy.com/
- **Content**: Parking lots, roads, aerial views
- **Quality**: HD
- **Use**: Various scenarios
- **License**: CC BY 3.0 / Videezy

## Available Test Suites

### Quick Suite (Recommended for Development)
```bash
python scripts/download_test_videos.py --suite quick
```

**Contents:**
- `traffic_001`: Highway traffic (18s, cars at various speeds)
- `pedestrian_001`: Crosswalk scene (12s, multiple people)
- `mixed_001`: Urban intersection (20s, cars + people + bikes)

**Total**: ~50MB, processes in 5-10 minutes
**Best for**: Daily development, quick validation

### Traffic Focus
```bash
python scripts/download_test_videos.py --suite traffic_focus
```

**Contents:**
- Highway scenes
- Intersections
- Parking lots

**Total**: ~45MB
**Best for**: Vehicle tracking algorithms

### Pedestrian Focus
```bash
python scripts/download_test_videos.py --suite pedestrian_focus
```

**Contents:**
- Crosswalks
- Parks
- Mixed urban

**Total**: ~55MB
**Best for**: Person tracking, occlusion handling

### Comprehensive (Full Validation)
```bash
python scripts/download_test_videos.py --suite comprehensive
```

**Contents**: All videos across all categories
**Total**: ~200MB, processes in 30+ minutes
**Best for**: Pre-release validation, benchmarking

## Test Video Catalog

Full catalog with metadata: `test_data/video_catalog.yaml`

Example entry:
```yaml
- id: traffic_001
  name: "Highway Traffic Overhead"
  category: traffic
  source: "Pixabay"
  url: "https://pixabay.com/videos/..."
  duration: 18
  resolution: "1920x1080"
  fps: 30
  features:
    - Multiple vehicles
    - Constant speeds
    - Good for speed estimation
  license: "Pixabay License (Free)"
```

## Directory Structure

```
test_data/
├── README.md              # Overview of test data
├── video_catalog.yaml     # Video metadata catalog
├── videos/                # Downloaded videos (git-ignored)
│   ├── traffic/
│   │   ├── traffic_001.mp4
│   │   ├── traffic_002.mp4
│   │   └── traffic_003.mp4
│   ├── pedestrian/
│   │   ├── pedestrian_001.mp4
│   │   └── pedestrian_002.mp4
│   ├── mixed/
│   │   ├── mixed_001.mp4
│   │   └── mixed_002.mp4
│   └── visdrone/
│       └── (manual downloads)
├── annotations/           # Ground truth (if available)
│   └── visdrone/
├── results/              # Test results (git-ignored)
│   └── results_quick_20250119.json
└── recordings/           # Scene graph recordings (git-ignored)
```

## Running Tests

### Single Video
```bash
python scripts/run_test_suite.py \
    --video test_data/videos/traffic/traffic_001.mp4 \
    --model n \
    --device cpu
```

### Test Suite
```bash
python scripts/run_test_suite.py \
    --suite quick \
    --model n \
    --device cpu
```

### With GPU (Faster)
```bash
python scripts/run_test_suite.py \
    --suite quick \
    --model s \
    --device cuda
```

## Results Format

Results are saved as JSON with detailed metrics:

```json
{
  "suite": "quick",
  "timestamp": "2025-01-19T12:00:00",
  "config": {
    "model_size": "n",
    "device": "cpu",
    "conf_threshold": 0.3
  },
  "results": [
    {
      "video": "traffic_001.mp4",
      "metrics": {
        "frame_count": 540,
        "total_time_sec": 27.5,
        "avg_fps": 19.6,
        "avg_detections_per_frame": 8.2,
        "avg_tracks_per_frame": 7.5,
        "avg_3d_objects_per_frame": 6.8
      },
      "per_frame": {
        "detection_counts": [8, 9, 7, ...],
        "track_counts": [7, 8, 7, ...],
        "frame_times_sec": [0.051, 0.049, ...]
      }
    }
  ]
}
```

## Expected Performance

Baseline targets (on reference hardware):

### Detection Accuracy
- Cars: 85%+ recall
- Persons: 80%+ recall
- Bicycles: 75%+ recall

### Tracking Quality
- ID switches: < 2 per 100 frames
- Track persistence: 25+ frames average
- Lost track recovery: 60%+

### Speed (YOLOv8n)
- Laptop CPU (i7): 20-30 FPS
- Laptop GPU (RTX 3060): 60-100 FPS
- Jetson Orin Nano: 30-40 FPS

## Manual Downloads (VisDrone)

Some videos require manual download:

1. Visit: https://github.com/VisDrone/VisDrone-Dataset
2. Register for academic access
3. Download sample videos
4. Place in: `test_data/videos/visdrone/`
5. Rename to match catalog IDs

## Git Ignore

Videos are **not committed** to git (too large).

`.gitignore` excludes:
- `test_data/videos/**/*.mp4`
- `test_data/results/**/*.json`
- `test_data/recordings/**/*.h5`

Download on-demand using scripts.

## Troubleshooting

### Downloads fail
```bash
# Check if URL is still valid
# Some free sites rotate content

# For YouTube (if any):
pip install yt-dlp
```

### Out of space
```bash
# Videos are 10-50MB each
# Clean old results
rm -rf test_data/results/*
rm -rf test_data/recordings/*
```

### Inconsistent results
- Use same model size
- Use same device (CPU vs GPU differences)
- Fix random seed (not implemented yet)

## Adding Custom Videos

1. Place video in appropriate category folder
2. Add entry to `video_catalog.yaml`:
```yaml
- id: custom_001
  name: "My Custom Video"
  category: mixed
  source: "Custom"
  url: null
  download_method: "manual"
  duration: 30
  resolution: "1920x1080"
  fps: 30
  description: "Description here"
  features:
    - Feature 1
    - Feature 2
```

3. Run baseline:
```bash
python scripts/run_test_suite.py --video test_data/videos/mixed/custom_001.mp4
```

## CI/CD Integration

For automated testing:

```yaml
# .github/workflows/test.yml
- name: Download test videos
  run: python scripts/download_test_videos.py --suite quick

- name: Run tests
  run: python scripts/run_test_suite.py --suite quick --device cpu

- name: Check results
  run: python scripts/compare_results.py --baseline baseline.json
```

## Next Steps

After setting up test data:

1. **Baseline**: Run tests, save as baseline for regression
2. **Stereo**: Add stereo camera test videos
3. **Annotations**: Add ground truth for quantitative eval
4. **Comparison**: Implement result comparison script
5. **Reports**: Create HTML report generation

## References

- Test Data README: `test_data/README.md`
- Video Catalog: `test_data/video_catalog.yaml`
- Download Script: `scripts/download_test_videos.py`
- Test Runner: `scripts/run_test_suite.py`
- Scripts Guide: `scripts/README.md`
