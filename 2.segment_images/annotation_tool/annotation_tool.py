#!/usr/bin/env python3
"""
Image Annotation Tool - 2D Image Grid Annotator
500 images per batch
6 label categories: globular, small, dissociated, elongated, blank, cluster
"""

import argparse
import sys
import threading
import webbrowser
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from flask import Flask, jsonify, request, send_file
from PIL import Image

app = Flask(__name__)

# Global state
IMAGE_DIR: Optional[Path] = None
PARQUET_FILE: Optional[Path] = None
ANNOTATOR: str = "default"
THUMBNAIL_SIZE = 140  # Larger for better visibility (was 100)
BATCH_SIZE = 500  # 500 images per batch
GRID_COLS = 15
GRID_ROWS = 10

# Label definitions
LABELS = {
    "globular": "1",
    "small": "2",
    "dissociated": "3",
    "elongated": "4",
    "blank": "5",
    "cluster": "6",
}

# Cache for existing annotations
ANNOTATIONS_CACHE: Dict[str, str] = {}


def get_image_files() -> List[Path]:
    """Get all image files from IMAGE_DIR, sorted by name."""
    if not IMAGE_DIR:
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    images = []

    # Find all image files (non-recursive for single directory)
    for file in IMAGE_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            images.append(file)

    if not images:
        print(f"Warning: No image files found in {IMAGE_DIR}")
        print(f"Looking for: {image_extensions}")

    return sorted(images)


def load_existing_annotations() -> None:
    """Load existing annotations from parquet file into memory cache."""
    global ANNOTATIONS_CACHE
    ANNOTATIONS_CACHE.clear()

    if not PARQUET_FILE or not PARQUET_FILE.exists():
        return

    try:
        table = pq.read_table(PARQUET_FILE)
        df = table.to_pandas()

        # Map filename to label
        for _, row in df.iterrows():
            ANNOTATIONS_CACHE[row["image_filename"]] = row["label"]
    except Exception as e:
        print(f"Warning: Could not load existing annotations: {e}")


def save_annotation(filename: str, label: str) -> None:
    """Append annotation to parquet file."""
    if not PARQUET_FILE:
        return

    annotation_data = {
        "image_filename": [filename],
        "annotator": [ANNOTATOR],
        "label": [label],
        "label_name": [next(k for k, v in LABELS.items() if v == label)],
        "timestamp": [datetime.now().isoformat()],
    }

    new_table = pa.table(annotation_data)

    try:
        if PARQUET_FILE.exists():
            existing_table = pq.read_table(PARQUET_FILE)
            combined_table = pa.concat_tables([existing_table, new_table])
        else:
            combined_table = new_table

        pq.write_table(combined_table, PARQUET_FILE)
        ANNOTATIONS_CACHE[filename] = label
    except Exception as e:
        print(f"Error saving annotation: {e}")


def generate_thumbnail(image_path: Path) -> bytes:
    """Generate a thumbnail (100×100 square)."""
    try:
        print(f"DEBUG: Opening image: {image_path}")
        with Image.open(image_path) as img:
            print(f"DEBUG: Image opened. Size: {img.size}, Mode: {img.mode}")
            # Create a copy
            img_copy = img.copy()

            # Make square thumbnail (fit to square, center crop if needed)
            img_copy.thumbnail(
                (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS
            )
            print(f"DEBUG: Thumbnail sized to {img_copy.size}")

            # Create square canvas
            square = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), (240, 240, 240))
            offset = (
                (THUMBNAIL_SIZE - img_copy.width) // 2,
                (THUMBNAIL_SIZE - img_copy.height) // 2,
            )

            # Convert to RGB if needed
            if img_copy.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img_copy.size, (240, 240, 240))
                rgb_img.paste(
                    img_copy,
                    mask=img_copy.split()[-1]
                    if img_copy.mode in ("RGBA", "LA")
                    else None,
                )
                img_copy = rgb_img
            elif img_copy.mode != "RGB":
                img_copy = img_copy.convert("RGB")

            print(f"DEBUG: Image converted to RGB")
            square.paste(img_copy, offset)

            # Save to bytes buffer
            buffer = BytesIO()
            square.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            print(
                f"DEBUG: Thumbnail saved to buffer, size: {len(buffer.getvalue())} bytes"
            )
            return buffer.getvalue()
    except Exception as e:
        print(f"ERROR generating thumbnail for {image_path}: {e}")
        import traceback

        traceback.print_exc()
        return b""


@app.route("/")
def index():
    """Serve the frontend HTML."""
    return INDEX_HTML


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return app configuration."""
    return jsonify(
        {
            "annotator": ANNOTATOR,
            "batch_size": BATCH_SIZE,
            "grid_cols": GRID_COLS,
            "grid_rows": GRID_ROWS,
            "thumbnail_size": THUMBNAIL_SIZE,
            "labels": LABELS,
        }
    )


@app.route("/api/images", methods=["GET"])
def get_images():
    """Return paginated image list with existing annotations."""
    offset = request.args.get("offset", 0, type=int)

    all_images = get_image_files()
    print(f"DEBUG: Total images found: {len(all_images)}")

    batch = all_images[offset : offset + BATCH_SIZE]
    print(f"DEBUG: Batch {offset}-{offset + len(batch)}, {len(batch)} images")

    images_data = []
    for img_path in batch:
        filename = img_path.name
        existing_label = ANNOTATIONS_CACHE.get(filename)
        images_data.append({"filename": filename, "existing_label": existing_label})
        print(f"DEBUG: Added image: {filename}")

    response = {
        "images": images_data,
        "offset": offset,
        "total": len(all_images),
        "has_more": (offset + BATCH_SIZE) < len(all_images),
    }
    print(f"DEBUG: Returning {len(images_data)} images")
    return jsonify(response)


@app.route("/api/thumbnail/<filename>", methods=["GET"])
def get_thumbnail(filename):
    """Return thumbnail for an image file."""
    if not IMAGE_DIR:
        print(f"ERROR: No IMAGE_DIR configured")
        return jsonify({"error": "No image directory configured"}), 400

    image_path = IMAGE_DIR / filename
    print(f"DEBUG: Thumbnail request for {filename}")
    print(f"DEBUG: Full path: {image_path}")
    print(f"DEBUG: File exists: {image_path.exists()}")

    # Security: prevent directory traversal
    if not image_path.exists() or not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}")
        return jsonify({"error": "Image not found"}), 404

    if not str(image_path.resolve()).startswith(str(IMAGE_DIR.resolve())):
        print(f"ERROR: Invalid path (directory traversal attempt): {image_path}")
        return jsonify({"error": "Invalid path"}), 403

    print(f"DEBUG: Generating thumbnail for {filename}")
    thumbnail_data = generate_thumbnail(image_path)

    if not thumbnail_data:
        print(f"ERROR: Could not generate thumbnail for {filename}")
        return jsonify({"error": "Could not generate thumbnail"}), 500

    print(f"DEBUG: Thumbnail generated, size: {len(thumbnail_data)} bytes")
    return send_file(
        BytesIO(thumbnail_data), mimetype="image/jpeg", as_attachment=False
    )


@app.route("/api/annotate", methods=["POST"])
def annotate():
    """Save an annotation."""
    data = request.get_json()

    if not data or "filename" not in data or "label" not in data:
        return jsonify({"error": "Missing filename or label"}), 400

    filename = data["filename"]
    label = data["label"]

    # Validate label
    if label not in LABELS.values():
        return jsonify(
            {"error": f"Invalid label. Must be one of: {list(LABELS.values())}"}
        ), 400

    save_annotation(filename, label)
    return jsonify({"success": True, "filename": filename, "label": label})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return annotation statistics."""
    all_images = get_image_files()
    annotated = sum(1 for img in all_images if img.name in ANNOTATIONS_CACHE)

    label_counts = {label_name: 0 for label_name in LABELS.keys()}
    for label in ANNOTATIONS_CACHE.values():
        label_name = next((k for k, v in LABELS.items() if v == label), None)
        if label_name:
            label_counts[label_name] += 1

    return jsonify(
        {
            "total_images": len(all_images),
            "annotated": annotated,
            "pending": len(all_images) - annotated,
            "by_label": label_counts,
        }
    )


@app.route("/api/set-annotator", methods=["POST"])
def set_annotator():
    """Set annotator name."""
    global ANNOTATOR
    data = request.get_json()
    if data and "annotator" in data:
        ANNOTATOR = data["annotator"]
        return jsonify({"success": True})
    return jsonify({"error": "Missing annotator"}), 400


# HTML/CSS/JS Frontend
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Annotation Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            overflow-x: hidden;
        }

        .header {
            background: #1a1a1a;
            border-bottom: 1px solid #333;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
        }

        .stats {
            display: flex;
            gap: 3rem;
            font-size: 0.9rem;
        }

        .stat {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d4ff;
        }

        .stat-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #888;
            letter-spacing: 0.5px;
            margin-top: 0.25rem;
        }

        .container {
            padding: 2rem;
            max-width: 1600px;
            margin: 0 auto;
        }

        .info-bar {
            background: #1a1a1a;
            border-left: 4px solid #00d4ff;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }

        .info-bar p {
            margin: 0.5rem 0;
            color: #bbb;
            font-size: 0.95rem;
        }

        .batch-info {
            font-weight: 600;
            color: #00d4ff;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 3rem;
            background: #111;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.5);
            max-height: 75vh;
            overflow-y: auto;
            padding-right: 1rem;
        }

        .grid::-webkit-scrollbar {
            width: 8px;
        }

        .grid::-webkit-scrollbar-track {
            background: #1a1a1a;
            border-radius: 4px;
        }

        .grid::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 4px;
        }

        .grid::-webkit-scrollbar-thumb:hover {
            background: #666;
        }

        .resize-controls {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 500;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            width: 200px;
        }

        .resize-controls label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #e0e0e0;
        }

        .resize-slider {
            width: 150px;
            cursor: pointer;
        }

        .resize-value {
            font-size: 0.9rem;
            font-weight: 600;
            color: #00d4ff;
            text-align: center;
        }

        .sort-menu {
            position: fixed;
            top: 100px;
            right: 2rem;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 510;
            display: none;
            flex-direction: column;
            min-width: 220px;
            max-height: 60vh;
            overflow-y: auto;
        }

        .sort-menu.show {
            display: flex;
        }

        .sort-menu button {
            padding: 0.75rem 1rem;
            border: none;
            background: #1a1a1a;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            text-align: left;
            transition: background 0.1s ease;
            border-bottom: 1px solid #f0f0f0;
        }

        .sort-menu button:last-child {
            border-bottom: none;
        }

        .sort-menu button:hover {
            background: #0a0a0a;
        }

        .image-tile img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(var(--brightness, 100%));
        }

        .image-tile {
            aspect-ratio: 1;
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .image-tile:hover {
            border-color: #00d4ff;
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
            transform: translateY(-2px);
        }

        .image-tile.selected {
            border-color: #ff4444;
            border-width: 3px;
            background: #330000;
            box-shadow: 0 4px 12px rgba(255, 68, 68, 0.3);
        }

        .image-tile img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(var(--brightness, 100%));
        }

        .image-tile.annotated {
            border-color: #22c55e;
            background: #0a2a0a;
        }

        .image-tile.annotated.selected {
            border-color: #e74c3c;
            background: #ffe6e6;
        }

        .selection-check {
            position: absolute;
            top: 0;
            left: 0;
            background: #e74c3c;
            color: white;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            border-radius: 0 0 6px 0;
            opacity: 0;
            transition: opacity 0.15s ease;
        }

        .image-tile.selected .selection-check {
            opacity: 1;
        }

        .label-overlay {
            position: absolute;
            top: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 0.25rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
            border-radius: 0 6px 0 4px;
            opacity: 0;
            transition: opacity 0.15s ease;
        }

        .image-tile.annotated .label-overlay {
            opacity: 1;
        }

        .tooltip {
            position: absolute;
            bottom: -40px;
            left: 50%;
            transform: translateX(-50%);
            background: #2c3e50;
            color: white;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease;
            z-index: 10;
        }

        .image-tile:hover .tooltip {
            opacity: 1;
        }

        .popup-menu {
            position: absolute;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            overflow: hidden;
            display: none;
        }

        .popup-menu.show {
            display: flex;
            flex-direction: column;
        }

        .popup-menu button {
            padding: 0.75rem 1.5rem;
            border: none;
            background: #1a1a1a;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            text-align: left;
            transition: background 0.1s ease;
            white-space: nowrap;
        }

        .popup-menu button:hover {
            background: #0a0a0a;
        }

        .popup-menu button:first-child {
            border-bottom: 1px solid #e9ecef;
        }

        .label-1 { color: #e74c3c; }  /* globular - red */
        .label-2 { color: #f39c12; }  /* small - orange */
        .label-3 { color: #9b59b6; }  /* dissociated - purple */
        .label-4 { color: #1abc9c; }  /* elongated - teal */
        .label-5 { color: #95a5a6; }  /* blank - gray */
        .label-6 { color: #3498db; }  /* cluster - blue */

        .buttons-row {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-globular {
            background: #e74c3c;
            color: white;
        }

        .btn-globular:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }

        .btn-small {
            background: #f39c12;
            color: white;
        }

        .btn-small:hover {
            background: #d68910;
            transform: translateY(-2px);
        }

        .btn-dissociated {
            background: #9b59b6;
            color: white;
        }

        .btn-dissociated:hover {
            background: #8e44ad;
            transform: translateY(-2px);
        }

        .btn-elongated {
            background: #1abc9c;
            color: white;
        }

        .btn-elongated:hover {
            background: #16a085;
            transform: translateY(-2px);
        }

        .btn-blank {
            background: #95a5a6;
            color: white;
        }

        .btn-blank:hover {
            background: #7f8c8d;
            transform: translateY(-2px);
        }

        .btn-cluster {
            background: #3498db;
            color: white;
        }

        .btn-cluster:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }

        .pagination {
            display: flex;
            justify-content: center;
            gap: 1rem;
            padding: 2rem;
            background: #1a1a1a;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }

        .btn-primary {
            background: #00d4ff;
            color: #000;
            font-weight: 600;
        }

        .btn-primary:hover {
            background: #00b8e6;
        }

        .btn-primary:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }

        .btn-save {
            background: #22c55e !important;
            color: #000 !important;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 0.6rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-save:hover {
            background: #16a34a !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(34, 197, 94, 0.3);
        }

        .btn-toggle-label {
            background: #9d4edd !important;
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 0.6rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-toggle-label:hover {
            background: #7b2cbf !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(157, 78, 221, 0.3);
        }

        .btn-toggle-label.wellFOV {
            background: #06aed5 !important;
        }

        .btn-toggle-label.wellFOV:hover {
            background: #048aa6 !important;
            box-shadow: 0 4px 8px rgba(6, 174, 213, 0.3);
        }

        .success-toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #22c55e;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        .loading {
            text-align: center;
            padding: 4rem 2rem;
            color: #888;
        }

        .spinner {
            border: 4px solid #ecf0f1;
            border-top-color: #00d4ff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            align-items: center;
            justify-content: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background-color: #1a1a1a;
            padding: 2.5rem;
            border-radius: 8px;
            width: 90%;
            max-width: 400px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid #333;
        }

        .modal-content h2 {
            margin-bottom: 1rem;
            color: #fff;
        }

        .modal-content p {
            color: #bbb;
            margin-bottom: 1.5rem;
        }

        .modal-content input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #444;
            background: #0a0a0a;
            color: #e0e0e0;
            border-radius: 6px;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .modal-buttons {
            display: flex;
            gap: 1rem;
            justify-content: flex-end;
        }

        .btn-cancel {
            background: #bdc3c7;
            color: white;
        }

        .btn-cancel:hover {
            background: #95a5a6;
        }

        .label-counts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .label-count-box {
            background: #1a1a1a;
            padding: 1rem;
            border-radius: 6px;
            border-left: 4px solid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .label-count-box.label-1 { border-color: #e74c3c; }
        .label-count-box.label-2 { border-color: #f39c12; }
        .label-count-box.label-3 { border-color: #9b59b6; }
        .label-count-box.label-4 { border-color: #1abc9c; }
        .label-count-box.label-5 { border-color: #95a5a6; }
        .label-count-box.label-6 { border-color: #3498db; }

        .label-count-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #e0e0e0;
        }

        .label-count-name {
            font-size: 0.85rem;
            text-transform: capitalize;
            color: #666;
            font-weight: 500;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📸 Image Annotation Tool</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="totalCount">-</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="annotatedCount">-</div>
                <div class="stat-label">Annotated</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="pendingCount">-</div>
                <div class="stat-label">Pending</div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="info-bar">
            <p class="batch-info">Batch <span id="batchNum">1</span> of <span id="batchTotal">1</span> (500 images)</p>
            <p>Click images to select, then click a label button to assign to all selected</p>
            <div style="margin-top: 0.5rem;">
                <span id="selectedCount" style="font-weight: 600; color: #e74c3c;">0 selected</span>
                <button onclick="selectAll()" style="margin-left: 1rem; padding: 0.5rem 1rem; background: #00d4ff; color: white; border: none; border-radius: 4px; cursor: pointer;">Select All (500)</button>
                <button onclick="deselectAll()" style="margin-left: 0.5rem; padding: 0.5rem 1rem; background: #95a5a6; color: white; border: none; border-radius: 4px; cursor: pointer;">Deselect All</button>
            </div>
        </div>

        <div class="label-counts" id="labelCounts"></div>

        <div class="bulk-buttons" style="display: flex; gap: 1rem; justify-content: center; margin-bottom: 2rem; flex-wrap: wrap;">
            <button class="btn-globular" onclick="bulkLabel('1')" title="Label all selected as Globular">🏷️ Globular</button>
            <button class="btn-small" onclick="bulkLabel('2')" title="Label all selected as Small">🏷️ Small</button>
            <button class="btn-dissociated" onclick="bulkLabel('3')" title="Label all selected as Dissociated">🏷️ Dissociated</button>
            <button class="btn-elongated" onclick="bulkLabel('4')" title="Label all selected as Elongated">🏷️ Elongated</button>
            <button class="btn-blank" onclick="bulkLabel('5')" title="Label all selected as Blank">🏷️ Blank</button>
            <button class="btn-cluster" onclick="bulkLabel('6')" title="Label all selected as Cluster">🏷️ Cluster</button>
        </div>

        <div class="grid" id="grid"></div>

        <div class="pagination">
            <button class="btn-primary" id="prevBtn" onclick="previousBatch()" disabled>← Previous</button>
            <span id="batchInfo" style="align-self: center; color: #bbb; font-weight: 500;"></span>
            <button class="btn-primary" id="nextBtn" onclick="nextBatch()">Next →</button>
        </div>
    </div>

    <div id="annotatorModal" class="modal show">
        <div class="modal-content">
            <h2>Welcome!</h2>
            <p>Enter your name (annotator ID):</p>
            <input type="text" id="annotatorInput" placeholder="Your name" autofocus>
            <div class="modal-buttons">
                <button class="btn-primary" onclick="setAnnotator()">Start</button>
            </div>
        </div>
    </div>

    <div id="imagePopup" class="popup-menu">
        <button onclick="selectLabel('1')"><span class="label-1">●</span> Globular</button>
        <button onclick="selectLabel('2')"><span class="label-2">●</span> Small</button>
        <button onclick="selectLabel('3')"><span class="label-3">●</span> Dissociated</button>
        <button onclick="selectLabel('4')"><span class="label-4">●</span> Elongated</button>
        <button onclick="selectLabel('5')"><span class="label-5">●</span> Blank</button>
        <button onclick="selectLabel('6')"><span class="label-6">●</span> Cluster</button>
    </div>

    <div class="resize-controls">
        <div>
            <label>Zoom:</label>
            <input type="range" id="zoomSlider" class="resize-slider" min="80" max="200" value="140" step="10">
            <div class="resize-value"><span id="zoomValue">140</span>px</div>
        </div>
        <div style="border-top: 1px solid #e9ecef; padding-top: 0.75rem;">
            <label>Brightness:</label>
            <input type="range" id="brightnessSlider" class="resize-slider" min="50" max="200" value="100" step="10">
            <div class="resize-value"><span id="brightnessValue">100</span>%</div>
        </div>
        <div style="border-top: 1px solid #e9ecef; padding-top: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem;">
            <button onclick="previousBatch()" class="btn-primary" style="width: 100%; padding: 0.6rem;">← Previous Batch</button>
            <button onclick="nextBatch()" class="btn-primary" style="width: 100%; padding: 0.6rem;">Next Batch →</button>
            <button onclick="toggleSortMenu()" class="btn-primary" style="width: 100%; padding: 0.6rem;">Sort by Label ▼</button>
            <button id="toggleLabelBtn" onclick="togglePatientWellFOV()" class="btn-toggle-label" style="width: 100%; padding: 0.6rem;">👤 Patient</button>
            <button onclick="saveAnnotations()" class="btn-save" style="width: 100%; padding: 0.6rem;">💾 Save All</button>
        </div>
    </div>

    <div id="sortMenu" class="sort-menu">
        <button onclick="sortImages('all')">All Images (Grouped)</button>
        <button onclick="sortImages('hide-labeled')">Hide Labeled (Unlabeled Only)</button>
        <button onclick="sortImages('globular')">Globular (1)</button>
        <button onclick="sortImages('small')">Small (2)</button>
        <button onclick="sortImages('dissociated')">Dissociated (3)</button>
        <button onclick="sortImages('elongated')">Elongated (4)</button>
        <button onclick="sortImages('blank')">Blank (5)</button>
        <button onclick="sortImages('cluster')">Cluster (6)</button>
    </div>

    <script>
        let currentBatch = 0;
        let totalBatches = 1;
        let config = {};
        let currentImageFilename = null;
        let labelNames = {};
        let selectedImages = new Set();  // Track selected images
        let allBatchImages = [];  // Store all images in current batch
        let currentSortFilter = 'all';  // Current sort filter

        async function loadConfig() {
            const res = await fetch('/api/config');
            config = await res.json();
            labelNames = config.labels;
        }

        function setAnnotator() {
            const name = document.getElementById('annotatorInput').value.trim();
            if (!name) {
                alert('Please enter a name');
                return;
            }

            fetch('/api/set-annotator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ annotator: name })
            }).then(() => {
                document.getElementById('annotatorModal').classList.remove('show');
                loadBatch(0);
            });
        }

        document.getElementById('annotatorInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') setAnnotator();
        });

        async function loadBatch(batchNum) {
            const offset = batchNum * config.batch_size;

            try {
                const res = await fetch(`/api/images?offset=${offset}`);
                const data = await res.json();

                // Store all images and reset sort filter
                allBatchImages = data.images;
                currentSortFilter = 'all';

                // Clear grid and selections
                document.getElementById('grid').innerHTML = '';
                selectedImages.clear();
                updateSelectedCount();

                // Render images plain (without grouping)
                renderPlainImages(allBatchImages);

                // Update batch info
                currentBatch = batchNum;
                totalBatches = Math.ceil(data.total / config.batch_size);
                updateBatchInfo(data);
                updateStats();
                updatePaginationButtons();
            } catch (e) {
                console.error('Error loading batch:', e);
                alert('Error loading images');
            }
        }

        function renderPlainImages(images) {
            const grid = document.getElementById('grid');
            grid.innerHTML = '';

            images.forEach(img => {
                const tile = createTile(img);
                grid.appendChild(tile);
            });
        }

        function renderImages(images) {
            const grid = document.getElementById('grid');
            grid.innerHTML = '';

            images.forEach(img => {
                // Only expect grouped images with headers here
                if (img.isHeader) {
                    const headerDiv = document.createElement('div');
                    headerDiv.style.cssText = `
                        grid-column: 1 / -1;
                        padding: 1rem 0;
                        margin: 1rem 0;
                        border-bottom: 2px solid #333;
                        font-size: 1.1rem;
                        font-weight: 700;
                        color: #00d4ff;
                    `;
                    headerDiv.textContent = `${img.label} (${img.count} images)`;
                    grid.appendChild(headerDiv);
                } else {
                    // Regular image tile
                    const tile = createTile(img);
                    grid.appendChild(tile);
                }
            });
        }

        function sortImages(filter) {
            console.log('DEBUG: sortImages called with filter:', filter);

            if (!allBatchImages || allBatchImages.length === 0) {
                console.error('ERROR: allBatchImages is empty or undefined!');
                alert('No images loaded. Please load a batch first.');
                return;
            }

            console.log('DEBUG: allBatchImages:', allBatchImages);
            console.log('DEBUG: labelNames:', labelNames);

            currentSortFilter = filter;
            let filtered = [...allBatchImages];
            console.log('DEBUG: Initial filtered count:', filtered.length);

            if (filter === 'unlabeled') {
                console.log('DEBUG: Filtering for unlabeled');
                filtered = filtered.filter(img => !img.existing_label);
            } else if (filter === 'hide-labeled') {
                console.log('DEBUG: Filtering - hiding labeled (showing unlabeled)');
                filtered = filtered.filter(img => !img.existing_label);
            } else if (filter !== 'all') {
                // Filter by label name (e.g., 'globular', 'small', etc.)
                // labelNames is like { 'globular': '1', 'small': '2', ... }
                const labelNum = labelNames[filter];
                console.log('DEBUG: Filtering for label:', filter, '-> number:', labelNum);
                if (labelNum) {
                    filtered = filtered.filter(img => {
                        const match = img.existing_label === labelNum;
                        console.log('DEBUG: Image', img.filename, 'label:', img.existing_label, 'match:', match);
                        return match;
                    });
                } else {
                    console.log('ERROR: Label not found in labelNames');
                }
            } else {
                console.log('DEBUG: Showing all images');
            }

            console.log('DEBUG: Final filtered count:', filtered.length);

            // Group by labels in snake pattern
            const grouped = groupImagesByLabel(filtered);

            renderImages(grouped);
            toggleSortMenu();  // Close menu after selection

            // Update info
            if (filter === 'all') {
                document.getElementById('batchInfo').textContent =
                    `Showing all ${filtered.length} images`;
            } else if (filter === 'unlabeled' || filter === 'hide-labeled') {
                document.getElementById('batchInfo').textContent =
                    `Showing ${filtered.length} unlabeled images`;
            } else {
                document.getElementById('batchInfo').textContent =
                    `Showing ${filtered.length} images labeled as "${filter}"`;
            }
        }

        function groupImagesByLabel(images) {
            // Group images by label
            const groups = {
                'unlabeled': [],
                '1': [],
                '2': [],
                '3': [],
                '4': [],
                '5': [],
                '6': []
            };

            images.forEach(img => {
                const label = img.existing_label || 'unlabeled';
                if (groups[label]) {
                    groups[label].push(img);
                }
            });

            // Order: unlabeled first, then 1, 2, 3, 4, 5, 6
            const labelOrder = ['unlabeled', '1', '2', '3', '4', '5', '6'];
            const ordered = [];

            labelOrder.forEach(label => {
                if (groups[label].length > 0) {
                    // Add label header
                    const labelName = label === 'unlabeled' ? 'Unlabeled' :
                        Object.keys(labelNames).find(k => labelNames[k] === label) || label;

                    ordered.push({
                        isHeader: true,
                        label: labelName,
                        count: groups[label].length
                    });

                    // Add all images for this label
                    ordered.push(...groups[label]);
                }
            });

            console.log('DEBUG: Grouped and ordered images:', ordered);
            return ordered;
        }

        function toggleSortMenu() {
            console.log('DEBUG: toggleSortMenu called');
            const menu = document.getElementById('sortMenu');
            console.log('DEBUG: menu element:', menu);

            if (!menu) {
                console.error('ERROR: sortMenu element not found!');
                return;
            }

            menu.classList.toggle('show');
            console.log('DEBUG: menu classes after toggle:', menu.className);

            // Position menu - simple approach: just stack above controls
            if (menu.classList.contains('show')) {
                menu.style.bottom = 'auto';
                menu.style.top = '100px';
                console.log('DEBUG: Menu positioned to top: 100px');
            }
        }

        // Close sort menu when clicking outside
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('sortMenu');
            const controls = document.querySelector('.resize-controls');
            if (!menu.contains(e.target) && !controls.contains(e.target)) {
                menu.classList.remove('show');
            }
        });

        function createTile(image) {
            const tile = document.createElement('div');
            tile.className = image.existing_label ? 'image-tile annotated' : 'image-tile';
            tile.id = `tile-${image.filename}`;
            tile.dataset.filename = image.filename;

            const img = document.createElement('img');
            img.src = `/api/thumbnail/${encodeURIComponent(image.filename)}`;
            img.alt = image.filename;
            tile.appendChild(img);

            // Add selection check mark
            const check = document.createElement('div');
            check.className = 'selection-check';
            check.textContent = '✓';
            tile.appendChild(check);

            if (image.existing_label) {
                const overlay = document.createElement('div');
                overlay.className = 'label-overlay';
                const labelName = Object.keys(labelNames).find(k => labelNames[k] === image.existing_label);
                overlay.textContent = labelName || image.existing_label;
                tile.appendChild(overlay);
            }

            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = image.filename;
            tile.appendChild(tooltip);

            // Toggle selection on click
            tile.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleSelection(image.filename, tile);
            });

            return tile;
        }

        function toggleSelection(filename, tile) {
            if (selectedImages.has(filename)) {
                selectedImages.delete(filename);
                tile.classList.remove('selected');
            } else {
                selectedImages.add(filename);
                tile.classList.add('selected');
            }
            updateSelectedCount();
        }

        function selectAll() {
            const tiles = document.querySelectorAll('.image-tile');
            tiles.forEach(tile => {
                const filename = tile.dataset.filename;
                selectedImages.add(filename);
                tile.classList.add('selected');
            });
            updateSelectedCount();
        }

        function deselectAll() {
            const tiles = document.querySelectorAll('.image-tile');
            tiles.forEach(tile => {
                tile.classList.remove('selected');
            });
            selectedImages.clear();
            updateSelectedCount();
        }

        function updateSelectedCount() {
            document.getElementById('selectedCount').textContent =
                `${selectedImages.size} selected`;
        }

        async function bulkLabel(label) {
            if (selectedImages.size === 0) {
                alert('Please select at least one image');
                return;
            }

            const labelName = Object.keys(labelNames).find(k => labelNames[k] === label);
            if (!confirm(`Label ${selectedImages.size} image(s) as "${labelName}"?`)) {
                return;
            }

            let success = 0;
            let failed = 0;

            // Label each selected image
            for (const filename of selectedImages) {
                try {
                    const res = await fetch('/api/annotate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename, label })
                    });

                    if (res.ok) {
                        const tile = document.getElementById(`tile-${filename}`);
                        tile.classList.add('annotated');

                        // Update overlay
                        let overlay = tile.querySelector('.label-overlay');
                        if (!overlay) {
                            overlay = document.createElement('div');
                            overlay.className = 'label-overlay';
                            tile.appendChild(overlay);
                        }
                        overlay.textContent = labelName || label;

                        success++;
                    } else {
                        failed++;
                    }
                } catch (e) {
                    console.error('Error labeling:', filename, e);
                    failed++;
                }
            }

            // Clear selection
            deselectAll();
            updateStats();

            // Show result
            alert(`✓ Labeled ${success} image(s)${failed > 0 ? `, ${failed} failed` : ''}`);
        }

        function showPopup(event) {
            const popup = document.getElementById('imagePopup');
            popup.classList.add('show');
            popup.style.position = 'fixed';
            popup.style.left = event.clientX + 'px';
            popup.style.top = event.clientY + 'px';
        }

        document.addEventListener('click', (e) => {
            const popup = document.getElementById('imagePopup');
            if (!popup.contains(e.target) && !e.target.closest('.image-tile')) {
                popup.classList.remove('show');
            }
        });

        async function selectLabel(label) {
            if (!currentImageFilename) return;

            try {
                const res = await fetch('/api/annotate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: currentImageFilename, label })
                });

                if (res.ok) {
                    const tile = document.getElementById(`tile-${currentImageFilename}`);
                    tile.classList.add('annotated');

                    // Update overlay
                    let overlay = tile.querySelector('.label-overlay');
                    if (!overlay) {
                        overlay = document.createElement('div');
                        overlay.className = 'label-overlay';
                        tile.appendChild(overlay);
                    }
                    const labelName = Object.keys(labelNames).find(k => labelNames[k] === label);
                    overlay.textContent = labelName || label;

                    document.getElementById('imagePopup').classList.remove('show');
                    updateStats();
                }
            } catch (e) {
                console.error('Error saving annotation:', e);
            }
        }

        function updateBatchInfo(data) {
            document.getElementById('batchNum').textContent = currentBatch + 1;
            document.getElementById('batchTotal').textContent = totalBatches;
            document.getElementById('batchInfo').textContent =
                `Showing ${data.images.length} images (${data.offset + 1}–${data.offset + data.images.length} of ${data.total})`;
        }

        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('totalCount').textContent = data.total_images;
                document.getElementById('annotatedCount').textContent = data.annotated;
                document.getElementById('pendingCount').textContent = data.pending;

                // Update label counts
                const countsHtml = Object.keys(labelNames).map(name => {
                    const count = data.by_label[name] || 0;
                    return `
                        <div class="label-count-box label-${labelNames[name]}">
                            <div class="label-count-value">${count}</div>
                            <div class="label-count-name">${name}</div>
                        </div>
                    `;
                }).join('');
                document.getElementById('labelCounts').innerHTML = countsHtml;
            } catch (e) {
                console.error('Error updating stats:', e);
            }
        }

        function saveAnnotations() {
            console.log('DEBUG: saveAnnotations called');

            // Get current stats
            fetch('/api/stats')
                .then(res => res.json())
                .then(data => {
                    const total = data.total_images;
                    const annotated = data.annotated;
                    const pending = data.pending;

                    // Show summary
                    alert(`📊 Annotation Summary:\n\n✓ Annotated: ${annotated}/${total}\n⏳ Pending: ${pending}\n\nAll annotations are automatically saved to:\nannotations.parquet`);

                    // Show toast with save confirmation
                    const toast = document.createElement('div');
                    toast.className = 'success-toast';
                    toast.textContent = `✓ ${annotated} annotations saved!`;
                    document.body.appendChild(toast);

                    setTimeout(() => toast.remove(), 3000);
                    console.log('DEBUG: Save confirmed. Total annotated:', annotated);
                })
                .catch(err => {
                    console.error('Error getting stats:', err);
                    alert('Error retrieving annotation statistics');
                });
        }

        let currentLabelMode = 'patient';  // 'patient' or 'wellFOV'

        function togglePatientWellFOV() {
            const btn = document.getElementById('toggleLabelBtn');
            const infoBar = document.querySelector('.info-bar');

            if (currentLabelMode === 'patient') {
                // Switch to wellFOV
                currentLabelMode = 'wellFOV';
                btn.textContent = '🔬 WellFOV';
                btn.classList.add('wellFOV');
                infoBar.style.borderLeftColor = '#06aed5';

                // Show toast
                const toast = document.createElement('div');
                toast.className = 'success-toast';
                toast.textContent = '🔬 Switched to WellFOV mode';
                document.body.appendChild(toast);
                setTimeout(() => toast.remove(), 2000);

                console.log('DEBUG: Switched to WellFOV mode');
            } else {
                // Switch to patient
                currentLabelMode = 'patient';
                btn.textContent = '👤 Patient';
                btn.classList.remove('wellFOV');
                infoBar.style.borderLeftColor = '#00d4ff';

                // Show toast
                const toast = document.createElement('div');
                toast.className = 'success-toast';
                toast.textContent = '👤 Switched to Patient mode';
                document.body.appendChild(toast);
                setTimeout(() => toast.remove(), 2000);

                console.log('DEBUG: Switched to Patient mode');
            }
        }

        function updatePaginationButtons() {
            document.getElementById('prevBtn').disabled = currentBatch === 0;
            document.getElementById('nextBtn').disabled = currentBatch >= totalBatches - 1;
        }

        function previousBatch() {
            if (currentBatch > 0) {
                loadBatch(currentBatch - 1);
                window.scrollTo(0, 0);
            }
        }

        function nextBatch() {
            if (currentBatch < totalBatches - 1) {
                loadBatch(currentBatch + 1);
                window.scrollTo(0, 0);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            await loadConfig();
            document.getElementById('annotatorInput').focus();

            // Setup resize slider
            const zoomSlider = document.getElementById('zoomSlider');
            const zoomValue = document.getElementById('zoomValue');
            const grid = document.getElementById('grid');

            zoomSlider.addEventListener('input', (e) => {
                const size = parseInt(e.target.value);
                zoomValue.textContent = size;
                grid.style.gridTemplateColumns = `repeat(auto-fit, minmax(${size}px, 1fr))`;
                // Adjust gap based on size
                const gap = Math.max(0.5, Math.min(1.5, size / 100));
                grid.style.gap = `${gap}rem`;
            });

            // Setup brightness slider
            const brightnessSlider = document.getElementById('brightnessSlider');
            const brightnessValue = document.getElementById('brightnessValue');

            brightnessSlider.addEventListener('input', (e) => {
                const brightness = parseInt(e.target.value);
                brightnessValue.textContent = brightness;

                // Apply brightness to all images
                const root = document.documentElement;
                root.style.setProperty('--brightness', brightness + '%');
            });
        });
    </script>
</body>
</html>
"""


def main():
    global IMAGE_DIR, PARQUET_FILE

    parser = argparse.ArgumentParser(
        description="Image Annotation Tool - 2D Grid Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 annotation_tool.py /path/to/images
  python3 annotation_tool.py ~/Pictures/my_images -o labels.parquet
  python3 annotation_tool.py ~/Pictures/my_images --port 5001
        """,
    )

    parser.add_argument(
        "image_dir", nargs="?", help="Directory containing images to annotate"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="annotations.parquet",
        help="Output parquet file (default: annotations.parquet)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run server on (default: 5000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )

    args = parser.parse_args()

    # Prompt for image directory if not provided
    if not args.image_dir:
        args.image_dir = input("Enter path to image directory: ").strip()

    # Expand ~ and resolve to absolute path
    IMAGE_DIR = Path(args.image_dir).expanduser().resolve()
    PARQUET_FILE = Path(args.output).resolve()

    if not IMAGE_DIR.exists():
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        sys.exit(1)

    if not IMAGE_DIR.is_dir():
        print(f"Error: Path is not a directory: {IMAGE_DIR}")
        sys.exit(1)

    # Load existing annotations
    load_existing_annotations()

    image_count = len(get_image_files())
    print(f"\n{'=' * 70}")
    print(f"Image Annotation Tool - 2D Grid Annotator")
    print(f"{'=' * 70}")
    print(f"Images found: {image_count}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Annotations file: {PARQUET_FILE}")
    print(f"Server running on: http://localhost:{args.port}")
    print(
        f"Grid layout: {GRID_COLS} columns × {GRID_ROWS} rows = {BATCH_SIZE} images per batch"
    )
    print(f"Labels: {', '.join(LABELS.keys())}")
    print(f"{'=' * 70}\n")

    if image_count == 0:
        print("Warning: No images found in directory!")

    # Open browser in a separate thread
    if not args.no_browser:

        def open_browser():
            import time

            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{args.port}")

        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()

    # Run Flask server
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
