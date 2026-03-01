# Image Annotation Tool

A lightweight, standalone image annotation tool built with Python and Flask. Quickly categorize thousands of images into bins (1, 2, or 3) with a responsive tile-based UI.

## Features

- **Fast & Lightweight**: Single Python script, no external services required
- **Persistent Annotations**: Saves to Parquet format with metadata (annotator, timestamp)
- **Lazy Loading**: Efficiently handles 4000+ images with batch loading (50 at a time)
- **Original Image Safety**: Thumbnails generated in-memory only—original files never modified
- **Web UI**: Modern, responsive interface with dark mode support
- **Progress Tracking**: Real-time stats on total, annotated, and pending images
- **Re-annotation Support**: Can review and re-label already annotated images
- **Cross-Platform**: Runs on Linux, macOS, Windows

## Installation

### Prerequisites

- Python 3.8 or higher
- Pop!_OS (Ubuntu-based) or any Linux distribution

### Setup

1. **Clone or extract the tool:**
   ```bash
   cd annotation-tool
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python3 annotation_tool.py /path/to/images
```

The tool will:
1. Scan the image directory
2. Open your default browser automatically
3. Prompt you to enter your name (annotator ID)
4. Start serving images for annotation

### Advanced Usage

```bash
# Specify output parquet file
python3 annotation_tool.py /path/to/images -o my_annotations.parquet

# Use a different port (default is 5000)
python3 annotation_tool.py /path/to/images --port 5001

# Don't open browser automatically
python3 annotation_tool.py /path/to/images --no-browser
```

### Access the UI

If the browser doesn't open automatically:
- Navigate to `http://localhost:5000` in your web browser

## How to Use

1. **Enter your name**: A prompt appears on first load asking for your annotator name
2. **View images**: Scroll through image tiles (200px thumbnails)
3. **Annotate**: Click one of the three category buttons (1, 2, or 3) under each image
4. **Track progress**: Stats at the top show total, annotated, and pending images
5. **Lazy load**: As you scroll, more images automatically load (50 at a time)
6. **Re-annotate**: Already-labeled images show their current label and can be re-labeled

## Output Format

Annotations are saved to a Parquet file with the following schema:

```
┌──────────────────┬───────────┬────────┬─────────────────────┐
│ image_filename   │ annotator │ label  │ timestamp           │
├──────────────────┼───────────┼────────┼─────────────────────┤
│ photo_001.jpg    │ Matt      │ 1      │ 2024-02-07T10:30... │
│ photo_002.jpg    │ Matt      │ 2      │ 2024-02-07T10:31... │
└──────────────────┴───────────┴────────┴─────────────────────┘
```

### Reading Annotations in Python

```python
import pandas as pd

df = pd.read_parquet('annotations.parquet')
print(df)

# Get all images labeled as category 1
category_1 = df[df['label'] == '1']

# Count by label
print(df['label'].value_counts())
```

## Supported Image Formats

- JPEG / JPG
- PNG
- GIF
- BMP
- WebP
- TIFF

## Important Notes

- **Original files are never modified**: Thumbnails are generated in-memory only
- **Concurrent access**: The tool is designed for single-user local use (one annotator at a time)
- **Performance**: Handles 4000+ images smoothly with lazy loading
- **Progress is persistent**: Reload the page anytime—your progress is saved

## Keyboard Shortcuts

No keyboard shortcuts currently (click-only interface for stability). Consider this for a future enhancement if needed.

## Troubleshooting

### Port already in use
```bash
python3 annotation_tool.py /path/to/images --port 5001
```

### Images not loading
- Check that the image directory path is correct
- Ensure you have read permissions on the directory
- Look for errors in the terminal where the tool is running

### Parquet file issues
If the annotations file becomes corrupted:
```bash
# Backup the old file
mv annotations.parquet annotations.parquet.bak
# Restart the tool—it will create a new file
python3 annotation_tool.py /path/to/images
```

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux (Pop!_OS recommended), macOS, or Windows |
| Python | 3.8+ |
| RAM | 512 MB minimum (comfortable with 2 GB+) |
| Disk | Space for original images + ~10% for parquet file |
| Browser | Any modern browser (Chrome, Firefox, Safari, Edge) |

## Performance

- **Image loading**: ~50 images per batch (configurable in code)
- **Thumbnail generation**: ~50-100ms per image (cached in browser)
- **Annotation save**: <100ms per click
- **Database writes**: Append-only, no locks or conflicts
