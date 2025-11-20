#!/usr/bin/env python3
"""
Download test videos for drone perception pipeline.

Usage:
    python scripts/download_test_videos.py --suite quick
    python scripts/download_test_videos.py --id traffic_001
    python scripts/download_test_videos.py --category traffic
    python scripts/download_test_videos.py --all
"""

import argparse
import sys
from pathlib import Path
import yaml
import urllib.request
import subprocess
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_color(message: str, color: str):
    """Print colored message."""
    print(f"{color}{message}{Colors.ENDC}")


def load_catalog(catalog_path: Path) -> dict:
    """Load video catalog from YAML."""
    with open(catalog_path, 'r') as f:
        return yaml.safe_load(f)


def download_direct(url: str, output_path: Path) -> bool:
    """
    Download file directly via HTTP.

    Args:
        url: Download URL
        output_path: Where to save file

    Returns:
        True if successful
    """
    try:
        print(f"  Downloading from: {url}")

        # Create progress bar
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            bar = '=' * (percent // 2) + '>' + '.' * (50 - percent // 2)
            print(f"\r  Progress: [{bar}] {percent}%", end='')

        urllib.request.urlretrieve(url, output_path, reporthook=show_progress)
        print()  # New line after progress
        return True

    except Exception as e:
        print_color(f"\n  Error downloading: {e}", Colors.FAIL)
        return False


def download_youtube(url: str, output_path: Path) -> bool:
    """
    Download video from YouTube using yt-dlp.

    Args:
        url: YouTube URL
        output_path: Where to save file

    Returns:
        True if successful
    """
    try:
        # Check if yt-dlp is installed
        result = subprocess.run(['which', 'yt-dlp'], capture_output=True)
        if result.returncode != 0:
            print_color("  Error: yt-dlp not installed. Install with: pip install yt-dlp", Colors.FAIL)
            return False

        print(f"  Downloading from YouTube: {url}")
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]',
            '-o', str(output_path),
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print_color(f"  Error: {result.stderr}", Colors.FAIL)
            return False

    except Exception as e:
        print_color(f"  Error: {e}", Colors.FAIL)
        return False


def download_video(video: Dict, output_dir: Path, force: bool = False) -> bool:
    """
    Download a single video.

    Args:
        video: Video metadata dict
        output_dir: Output directory
        force: Force re-download if exists

    Returns:
        True if successful
    """
    video_id = video['id']
    category = video['category']

    # Create category subdirectory
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_file = category_dir / f"{video_id}.mp4"

    # Check if already exists
    if output_file.exists() and not force:
        print_color(f"✓ {video_id}: Already exists, skipping", Colors.OKGREEN)
        return True

    print_color(f"\n→ Downloading: {video['name']} ({video_id})", Colors.OKBLUE)

    # Handle different download methods
    download_method = video.get('download_method', 'direct')

    if download_method == 'manual':
        print_color(f"  ⚠ Manual download required", Colors.WARNING)
        print(f"  Source: {video['source']}")
        print(f"  URL: {video['url']}")
        print(f"  Please download manually and place in: {output_file}")
        return False

    elif download_method == 'generate':
        print_color(f"  ⚠ Generated content (not available yet)", Colors.WARNING)
        return False

    elif download_method == 'direct':
        url = video.get('url')
        if not url:
            print_color(f"  Error: No URL provided", Colors.FAIL)
            return False

        # Determine download type
        if 'youtube.com' in url or 'youtu.be' in url:
            return download_youtube(url, output_file)
        else:
            return download_direct(url, output_file)

    else:
        print_color(f"  Error: Unknown download method: {download_method}", Colors.FAIL)
        return False


def main():
    parser = argparse.ArgumentParser(description='Download test videos for drone perception pipeline')
    parser.add_argument('--suite', type=str, help='Download test suite (quick, traffic_focus, comprehensive)')
    parser.add_argument('--id', type=str, help='Download specific video by ID')
    parser.add_argument('--category', type=str, help='Download all videos in category')
    parser.add_argument('--all', action='store_true', help='Download all available videos')
    parser.add_argument('--force', action='store_true', help='Force re-download existing files')
    parser.add_argument('--catalog', type=str, default='test_data/video_catalog.yaml',
                        help='Path to video catalog YAML')
    parser.add_argument('--output', type=str, default='test_data/videos',
                        help='Output directory for videos')

    args = parser.parse_args()

    # Load catalog
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print_color(f"Error: Catalog not found: {catalog_path}", Colors.FAIL)
        return 1

    catalog = load_catalog(catalog_path)
    videos = catalog['videos']
    test_suites = catalog.get('test_suites', {})

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_color("=" * 70, Colors.HEADER)
    print_color("Drone Perception Pipeline - Test Video Downloader", Colors.HEADER)
    print_color("=" * 70, Colors.HEADER)
    print()

    # Determine which videos to download
    videos_to_download = []

    if args.id:
        # Download specific video
        video = next((v for v in videos if v['id'] == args.id), None)
        if not video:
            print_color(f"Error: Video ID not found: {args.id}", Colors.FAIL)
            return 1
        videos_to_download = [video]
        print(f"Downloading video: {args.id}")

    elif args.suite:
        # Download test suite
        suite = test_suites.get(args.suite)
        if not suite:
            print_color(f"Error: Test suite not found: {args.suite}", Colors.FAIL)
            print(f"Available suites: {', '.join(test_suites.keys())}")
            return 1

        suite_video_ids = suite['videos']
        videos_to_download = [v for v in videos if v['id'] in suite_video_ids]

        print(f"Downloading test suite: {suite['name']}")
        print(f"Description: {suite['description']}")
        print(f"Videos: {len(videos_to_download)}")
        print(f"Total duration: ~{suite['total_duration']}s")
        print(f"Total size: ~{suite['total_size_mb']}MB")
        print()

    elif args.category:
        # Download category
        videos_to_download = [v for v in videos if v['category'] == args.category]
        if not videos_to_download:
            print_color(f"Error: No videos in category: {args.category}", Colors.FAIL)
            return 1
        print(f"Downloading category: {args.category} ({len(videos_to_download)} videos)")

    elif args.all:
        # Download all
        videos_to_download = videos
        print(f"Downloading all videos: {len(videos_to_download)}")

    else:
        # No selection - show help
        print_color("No videos selected. Use one of:", Colors.WARNING)
        print("  --suite SUITE    Download test suite (quick, comprehensive, etc.)")
        print("  --id VIDEO_ID    Download specific video")
        print("  --category CAT   Download all in category")
        print("  --all            Download everything")
        print()
        print("Recommended for getting started:")
        print("  python scripts/download_test_videos.py --suite quick")
        return 0

    # Download videos
    print()
    print_color(f"Starting download of {len(videos_to_download)} videos...", Colors.HEADER)
    print()

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, video in enumerate(videos_to_download, 1):
        print(f"[{i}/{len(videos_to_download)}]", end=' ')

        result = download_video(video, output_dir, force=args.force)

        if result:
            success_count += 1
        else:
            # Check if manual or not available
            if video.get('download_method') in ['manual', 'generate']:
                skip_count += 1
            else:
                fail_count += 1

    # Summary
    print()
    print_color("=" * 70, Colors.HEADER)
    print_color("Download Summary", Colors.HEADER)
    print_color("=" * 70, Colors.HEADER)
    print_color(f"✓ Successful: {success_count}", Colors.OKGREEN)
    if skip_count > 0:
        print_color(f"⊘ Skipped (manual/not available): {skip_count}", Colors.WARNING)
    if fail_count > 0:
        print_color(f"✗ Failed: {fail_count}", Colors.FAIL)
    print()
    print(f"Videos saved to: {output_dir}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
