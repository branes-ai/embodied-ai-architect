# Test Data Directory Structure

This directory contains test videos, annotations, and results for the drone perception pipeline.

## Directory Structure

```
test_data/
├── videos/              # Source video files
│   ├── visdrone/       # VisDrone dataset videos
│   ├── traffic/        # Traffic/road scenes
│   │   ├── *.mp4       # RGB videos (monocular)
│   │   ├── *_depth.mp4 # Depth map videos (stereo testing)
│   │   ├── *_output.mp4  # Monocular pipeline outputs
│   │   ├── *_stereo_output.mp4  # Stereo pipeline outputs
│   │   ├── *.log       # Pipeline logs
│   │   └── *_depth.yaml  # Depth map metadata
│   ├── pedestrian/     # Pedestrian tracking scenarios
│   ├── mixed/          # Mixed scenes (cars, people, bikes)
│   └── synthetic/      # Synthetic/simulation data
├── annotations/         # Ground truth annotations (if available)
│   └── visdrone/       # VisDrone annotations
├── results/            # Pipeline output results
│   ├── detections/     # Detection results (JSON)
│   ├── tracks/         # Tracking results (JSON)
│   └── visualizations/ # Output videos with overlays
└── recordings/         # Scene graph recordings (HDF5)
    └── DATE_VIDEO/     # One directory per recording
```

## Video Categories

### 1. VisDrone Dataset Videos
**Source**: https://github.com/VisDrone/VisDrone-Dataset
**Description**: Academic benchmark videos with annotations
**Use Cases**:
- Quantitative evaluation (have ground truth)
- Regression testing
- Algorithm comparison

### 2. Traffic Scenes
**Source**: Pixabay, Mixkit, Videezy
**Description**: Highway, intersection, parking lot footage
**Use Cases**:
- Car tracking validation
- Multi-object scenarios
- Speed estimation testing

### 3. Pedestrian Scenarios
**Source**: Free stock footage
**Description**: Parks, sidewalks, crosswalks
**Use Cases**:
- Person tracking
- Occlusion handling
- Crowd scenarios

### 4. Mixed Scenes
**Source**: Various
**Description**: Urban areas with cars, people, bikes, buses
**Use Cases**:
- Multi-class tracking
- Complex scenarios
- Real-world validation

### 5. Synthetic Data
**Source**: Simulation engines (future)
**Description**: Generated test cases
**Use Cases**:
- Edge case testing
- Controlled conditions
- Systematic validation

## Video Catalog

See `video_catalog.yaml` for detailed metadata about each test video.

## Downloading Test Videos

Use the provided script:

```bash
# Download all recommended test videos
python scripts/download_test_videos.py --all

# Download specific category
python scripts/download_test_videos.py --category traffic

# Download specific video by ID
python scripts/download_test_videos.py --id visdrone_sample_001
```

## Expected File Formats

### Videos
- Format: MP4 (H.264 codec preferred)
- Resolution: 720p or 1080p
- FPS: 25-30 fps
- Duration: 10-60 seconds per clip

### Annotations (VisDrone format)
```
<frame_id>, <target_id>, <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
```

### Results (Our format)
```json
{
  "video": "traffic_001.mp4",
  "frame_id": 42,
  "detections": [...],
  "tracks": [...],
  "scene_graph": {...}
}
```

## Testing Workflow

1. **Download test videos**:
   ```bash
   python scripts/download_test_videos.py --recommended
   ```

2. **Run pipeline on test set**:
   ```bash
   python scripts/run_test_suite.py --output results/
   ```

3. **Compare with baseline**:
   ```bash
   python scripts/compare_results.py --baseline baseline.json --current results/
   ```

4. **Generate regression report**:
   ```bash
   python scripts/generate_report.py --results results/ --output report.html
   ```

## Regression Testing

We maintain baseline results for each test video. When making changes:

1. Run on test suite
2. Compare with baseline (FPS, detection rate, tracking persistence)
3. Review significant deviations
4. Update baseline if intentional improvement

## Adding New Test Videos

1. Place video in appropriate category folder
2. Add entry to `video_catalog.yaml`
3. Run baseline: `python scripts/create_baseline.py --video your_video.mp4`
4. Commit metadata (not the video file if large)

## Size Management

- Videos are **not committed to git** (see .gitignore)
- Download on-demand using scripts
- Keep local cache in `test_data/videos/`
- Large files stored externally (Google Drive, S3, etc.)

## Recommended Test Set (Quick)

For rapid testing during development:

| ID | Description | Duration | Size | Key Features |
|----|-------------|----------|------|--------------|
| traffic_001 | Highway traffic | 15s | 10MB | Multiple cars, various speeds |
| pedestrian_001 | Crosswalk | 20s | 15MB | 5-10 people, occlusions |
| mixed_001 | City intersection | 30s | 25MB | Cars, people, bikes |
| visdrone_sample_001 | Benchmark clip | 30s | 20MB | Ground truth available |

Total: ~70MB

## Full Test Set (Comprehensive)

For thorough validation and benchmarking:

- 20+ videos across all categories
- Total: ~500MB
- Covers diverse scenarios, lighting, altitudes

Run: `python scripts/download_test_videos.py --full`

## Stereo Testing (Phase 2)

### Depth Map Generation

The stereo pipeline requires depth maps. Generate synthetic depth from RGB videos using MiDaS:

```bash
# Generate depth for all toll booth videos
python scripts/generate_depth_maps.py --batch toll_booth

# Generate depth for a specific video
python scripts/generate_depth_maps.py \
    --input test_data/videos/traffic/247589_tiny.mp4 \
    --max-depth 15.0
```

**Requirements:** PyTorch (install with `pip install torch torchvision`)

### Stereo Test Suite

Run automated stereo pipeline tests:

```bash
# Run all stereo tests (auto-generates depth if missing)
./run_stereo_test_suite.sh

# Compare monocular vs stereo results
python scripts/compare_mono_vs_stereo.py --batch
```

### Depth Map File Format

Generated depth maps are stored as:
- **Format:** MP4 video with colormap visualization
- **Scale:** 0.001 (1mm per unit)
- **Range:** 0-65.535 meters (stored in uint16)
- **Metadata:** Accompanying YAML file with generation parameters

Example metadata (`247589_tiny_depth.yaml`):
```yaml
source_video: 247589_tiny.mp4
depth_video: 247589_tiny_depth.mp4
model: MiDaS
frames: 258
resolution: 640x480
fps: 25.0
max_depth_meters: 15.0
depth_scale: 0.001  # meters per unit
```

### Testing Without Hardware

The `StereoRecordedCamera` class allows testing stereo pipeline features without physical stereo cameras:

```bash
python examples/stereo_pipeline.py \
    --backend recorded \
    --rgb-video test_data/videos/traffic/247589_tiny.mp4 \
    --depth-video test_data/videos/traffic/247589_tiny_depth.mp4
```

See [STEREO_TESTING.md](../STEREO_TESTING.md) for detailed instructions.
