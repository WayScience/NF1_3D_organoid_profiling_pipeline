#!/usr/bin/env python3
"""
Image Annotation Tool - entry point.

"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import state
from config import BATCH_SIZE, GRID_COLS, GRID_ROWS, LABELS
from flask import Flask
from image_utils import get_image_files
from routes import bp
from storage import load_existing_annotations

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(bp)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image Annotation Tool - 2D Grid Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 annotation_tool.py /path/to/images
  python3 annotation_tool.py ~/images -o labels.parquet
  python3 annotation_tool.py ~/images --port 5001
        """,
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        help="Directory containing images to annotate",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../image_labels/annotations.parquet",
        help="Output parquet file (default: ../image_labels/annotations.parquet)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically",
    )

    args = parser.parse_args()

    if not args.image_dir:
        args.image_dir = input("Enter path to image directory: ").strip()

    state.IMAGE_DIR = Path(args.image_dir).expanduser().resolve()
    state.PARQUET_FILE = Path(args.output).resolve()

    if not state.IMAGE_DIR.exists():
        print(f"Error: image directory not found: {state.IMAGE_DIR}")
        sys.exit(1)
    if not state.IMAGE_DIR.is_dir():
        print(f"Error: path is not a directory: {state.IMAGE_DIR}")
        sys.exit(1)

    load_existing_annotations()
    image_count = len(get_image_files())

    print(f"\n{'=' * 70}")
    print("Image Annotation Tool - 2D Grid Annotator")
    print(f"{'=' * 70}")
    print(f"Images found    : {image_count}")
    print(f"Image directory : {state.IMAGE_DIR}")
    print(f"Annotations file: {state.PARQUET_FILE}")
    print(f"Server          : http://localhost:{args.port}")
    print(
        f"Grid layout     : {GRID_COLS} cols × {GRID_ROWS} rows = {BATCH_SIZE} images/batch"
    )
    print(f"Labels          : {', '.join(LABELS)}")
    print(f"{'=' * 70}\n")

    if image_count == 0:
        print("Warning: no images found in directory!")

    if not args.no_browser:

        def _open_browser() -> None:
            import time

            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{args.port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    try:
        app.run(host="127.0.0.1", port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down…")
        sys.exit(0)


if __name__ == "__main__":
    main()
