# Image annotation tool

A lightweight, standalone image annotation tool built with python and flask. quickly categorize thousands of images into bins (1, 2, or 3) with a responsive tile-based UI.

## Features

- **Fast & lightweight**: single python script, no external services required
- **Persistent annotations**: saves to parquet format with metadata (annotator, timestamp)
- **Lazy loading**: efficiently handles 4000+ images with batch loading (50 at a time)
- **Original image safety**: thumbnails generated in-memory only—original files never modified
- **Web UI**: modern, responsive interface with dark mode support
- **Progress tracking**: real-time stats on total, annotated, and pending images
- **Re-annotation support**: can review and re-label already annotated images
- **Cross-platform**: runs on linux, macos, windows

## Installation

### Prerequisites

- Python 3.8 or higher
- Pop!\_OS (ubuntu-based) or any linux distribution

### Setup

1. **Clone or extract the tool:**

   ```Bash
   Cd annotation-tool
   ```

2. **Create a virtual environment (recommended):**

   ```Bash
   Python3 -m venv venv
   Source venv/bin/activate  # on windows: venv\scripts\activate
   ```

3. **Install dependencies:**
   ```Bash
   Pip install -r requirements.txt
   ```

## Usage

### Basic usage

```Bash
Python3 annotation_tool.py /path/to/images
```

The tool will:

1. Scan the image directory
2. Open your default browser automatically
3. Prompt you to enter your name (annotator ID)
4. Start serving images for annotation

### Advanced usage

```Bash
# Specify output parquet file
Python3 annotation_tool.py /path/to/images -o my_annotations.parquet

# Use a different port (default is 5000)
Python3 annotation_tool.py /path/to/images --port 5001

# Don't open browser automatically
Python3 annotation_tool.py /path/to/images --no-browser
```

### Access the UI

If the browser doesn't open automatically:

- Navigate to `http://localhost:5000` in your web browser

## How to use

1. **Enter your name**: A prompt appears on first load asking for your annotator name
2. **View images**: scroll through image tiles (200px thumbnails)
3. **Annotate**: click one of the three category buttons (1, 2, or 3) under each image
4. **Track progress**: stats at the top show total, annotated, and pending images
5. **Lazy load**: as you scroll, more images automatically load (50 at a time)
6. **Re-annotate**: already-labeled images show their current label and can be re-labeled

## Output format

Annotations are saved to a parquet file with the following schema:

```
┌──────────────────┬───────────┬────────┬─────────────────────┐
│ Image_filename   │ annotator │ label  │ timestamp           │
├──────────────────┼───────────┼────────┼─────────────────────┤
│ Photo_001.jpg    │ matt      │ 1      │ 2024-02-07T10:30... │
│ Photo_002.jpg    │ matt      │ 2      │ 2024-02-07T10:31... │
└──────────────────┴───────────┴────────┴─────────────────────┘
```

### Reading annotations in python

```Python
Import pandas as pd

Df = pd.read_parquet('annotations.parquet')
Print(df)

# Get all images labeled as category 1
Category_1 = df[df['label'] == '1']

# Count by label
Print(df['label'].value_counts())
```

## Supported image formats

- JPEG / JPG
- PNG
- GIF
- BMP
- Webp
- TIFF

## Important notes

- **Original files are never modified**: thumbnails are generated in-memory only
- **Concurrent access**: the tool is designed for single-user local use (one annotator at a time)
- **Performance**: handles 4000+ images smoothly with lazy loading
- **Progress is persistent**: reload the page anytime—your progress is saved

## Keyboard shortcuts

No keyboard shortcuts currently (click-only interface for stability). consider this for a future enhancement if needed.

## Troubleshooting

### Port already in use

```Bash
Python3 annotation_tool.py /path/to/images --port 5001
```

### Images not loading

- Check that the image directory path is correct
- Ensure you have read permissions on the directory
- Look for errors in the terminal where the tool is running

### Parquet file issues

If the annotations file becomes corrupted:

```Bash
# Backup the old file
Mv annotations.parquet annotations.parquet.bak
# Restart the tool—it will create a new file
Python3 annotation_tool.py /path/to/images
```

## System requirements

| Component | requirement                                        |
| --------- | -------------------------------------------------- |
| OS        | linux (pop!\_OS recommended), macos, or windows    |
| Python    | 3.8+                                               |
| RAM       | 512 MB minimum (comfortable with 2 GB+)            |
| Disk      | space for original images + ~10% for parquet file  |
| Browser   | any modern browser (chrome, firefox, safari, edge) |

## Performance

- **Image loading**: ~50 images per batch (configurable in code)
- **Thumbnail generation**: ~50-100ms per image (cached in browser)
- **Annotation save**: <100ms per click
- **Database writes**: append-only, no locks or conflicts
