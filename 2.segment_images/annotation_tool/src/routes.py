"""
Flask route definitions (registered as a Blueprint).
"""

from io import BytesIO

import state
from flask import Blueprint, jsonify, render_template, request, send_file
from image_utils import generate_thumbnail, get_image_files, get_sorted_images
from storage import save_annotation

from config import BATCH_SIZE, GRID_COLS, GRID_ROWS, LABELS, THUMBNAIL_SIZE

bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@bp.route("/")
def index():
    """Serve the single-page frontend."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API – configuration
# ---------------------------------------------------------------------------


@bp.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(
        {
            "annotator": state.ANNOTATOR,
            "batch_size": BATCH_SIZE,
            "grid_cols": GRID_COLS,
            "grid_rows": GRID_ROWS,
            "thumbnail_size": THUMBNAIL_SIZE,
            "labels": LABELS,
        }
    )


@bp.route("/api/set-annotator", methods=["POST"])
def set_annotator():
    data = request.get_json()
    if data and "annotator" in data:
        state.ANNOTATOR = data["annotator"]
        return jsonify({"success": True})
    return jsonify({"error": "Missing annotator"}), 400


# ---------------------------------------------------------------------------
# API – images
# ---------------------------------------------------------------------------


@bp.route("/api/images", methods=["GET"])
def get_images():
    """Return a paginated, optionally label-sorted list of images."""
    offset = request.args.get("offset", 0, type=int)

    if state.SORT_BY_LABEL_MODE:
        images_data, total = get_sorted_images(offset, BATCH_SIZE)
    else:
        all_images = get_image_files()
        batch = all_images[offset : offset + BATCH_SIZE]
        images_data = [
            {
                "filename": p.name,
                "existing_label": state.ANNOTATIONS_CACHE.get(p.name),
            }
            for p in batch
        ]
        total = len(all_images)

    return jsonify(
        {
            "images": images_data,
            "offset": offset,
            "total": total,
            "has_more": (offset + BATCH_SIZE) < total,
            "sort_by_label_mode": state.SORT_BY_LABEL_MODE,
        }
    )


@bp.route("/api/thumbnail/<filename>", methods=["GET"])
def get_thumbnail(filename: str):
    """Return a JPEG thumbnail for *filename*."""
    if not state.IMAGE_DIR:
        return jsonify({"error": "No image directory configured"}), 400

    image_path = state.IMAGE_DIR / filename

    # Security: reject directory-traversal attempts and missing files
    if not image_path.exists() or not image_path.is_file():
        return jsonify({"error": "Image not found"}), 404
    if not str(image_path.resolve()).startswith(str(state.IMAGE_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 403

    data = generate_thumbnail(image_path)
    if not data:
        return jsonify({"error": "Could not generate thumbnail"}), 500

    return send_file(BytesIO(data), mimetype="image/jpeg", as_attachment=False)


# ---------------------------------------------------------------------------
# API – annotations
# ---------------------------------------------------------------------------


@bp.route("/api/annotate", methods=["POST"])
def annotate():
    data = request.get_json()
    if not data or "filename" not in data or "label" not in data:
        return jsonify({"error": "Missing filename or label"}), 400

    label = str(data["label"])
    if label not in LABELS.values():
        return jsonify(
            {"error": f"Invalid label. Must be one of: {list(LABELS.values())}"}
        ), 400

    try:
        save_annotation(data["filename"], label)
        return jsonify({"success": True, "filename": data["filename"], "label": label})
    except Exception as e:
        return jsonify({"error": f"Failed to save annotation: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# API – stats & sort mode
# ---------------------------------------------------------------------------


@bp.route("/api/stats", methods=["GET"])
def get_stats():
    all_images = get_image_files()
    annotated = sum(1 for img in all_images if img.name in state.ANNOTATIONS_CACHE)

    label_counts = {name: 0 for name in LABELS}
    for label_val in state.ANNOTATIONS_CACHE.values():
        name = next((k for k, v in LABELS.items() if v == label_val), None)
        if name:
            label_counts[name] += 1

    return jsonify(
        {
            "total_images": len(all_images),
            "annotated": annotated,
            "pending": len(all_images) - annotated,
            "by_label": label_counts,
            "sort_by_label_mode": state.SORT_BY_LABEL_MODE,
        }
    )


@bp.route("/api/toggle-sort-by-label", methods=["POST"])
def toggle_sort_by_label():
    state.SORT_BY_LABEL_MODE = not state.SORT_BY_LABEL_MODE
    return jsonify({"sort_by_label_mode": state.SORT_BY_LABEL_MODE})
