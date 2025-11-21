# Camera + LiDAR pipeline

Files Created:

  1. sensors/lidar.py - LiDAR sensor interface supporting:
    - Velodyne (VLP-16, VLP-32, HDL-64E)
    - Ouster (OS0, OS1, OS2)
    - Livox (Avia, Mid-360)
    - ROS topics (sensor_msgs/PointCloud2)
    - Pre-recorded point cloud files (.pcd, .ply, .bin, .npy)
  2. examples/lidar_camera_pipeline.py - Complete fusion pipeline:
    - Synchronized LiDAR + Camera capture
    - LiDAR point projection to image plane
    - YOLO detection with metric LiDAR depth
    - 3D tracking and scene graph
    - Visualization with LiDAR overlay
  3. scripts/calibrate_lidar_camera.py - Calibration utility:
    - Camera intrinsic calibration (checkerboard)
    - LiDAR-camera extrinsic calibration framework
    - Calibration file generation

## Quick Start Guide:

  1. Create default calibration (for testing)
  python scripts/calibrate_lidar_camera.py --method default --output config/lidar_calib.json

  2. Run LiDAR + Camera pipeline
  python examples/lidar_camera_pipeline.py \
      --lidar velodyne \
      --camera 0 \
      --calibration config/lidar_calib.json \
      --model s \
      --classes 0 2 7

  3. For file-based testing (pre-recorded point clouds)
  python examples/lidar_camera_pipeline.py \
      --lidar file \
      --lidar-config '{"path": "test_data/point_clouds", "format": "bin"}' \
      --camera 0

## Architecture Highlights:

  - Point Cloud Projection: Projects 3D LiDAR points onto 2D camera image
  - Sparse Depth Maps: Creates depth maps from projected LiDAR points
  - Metric Accuracy: Real-world metric depth (0.1m - 100m+ range)
  - Modular Design: Easy to swap LiDAR sensors (Velodyne → Ouster → Livox)

## Next Steps for Production:

  1. Proper Calibration: Use --method checkerboard for camera intrinsics
  2. Extrinsic Calibration: Measure/calibrate LiDAR-to-camera transform
  3. Hardware Setup: Install LiDAR SDK for your specific sensor
  4. Test with Real Data: Run pipeline with your LiDAR + camera setup

