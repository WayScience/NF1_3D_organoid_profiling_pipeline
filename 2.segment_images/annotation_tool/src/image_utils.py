"""
Image file discovery and thumbnail generation.
"""

import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import state
from PIL import Image

from config import BATCH_SIZE, THUMBNAIL_SIZE

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def get_image_files() -> List[Path]:
    """Return all image files in *state.IMAGE_DIR*, sorted by name."""
    if not state.IMAGE_DIR:
        return []

    images = [
        f
        for f in state.IMAGE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"Warning: no image files found in {state.IMAGE_DIR}")

    return sorted(images)


def generate_thumbnail(image_path: Path) -> bytes:
    """Return JPEG bytes of a square *THUMBNAIL_SIZE × THUMBNAIL_SIZE* thumbnail."""
    try:
        with Image.open(image_path) as img:
            img_copy = img.copy()
            img_copy.thumbnail(
                (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS
            )

            # Paste onto square canvas with neutral background
            canvas = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), (240, 240, 240))
            offset = (
                (THUMBNAIL_SIZE - img_copy.width) // 2,
                (THUMBNAIL_SIZE - img_copy.height) // 2,
            )

            if img_copy.mode in ("RGBA", "LA", "P"):
                rgb = Image.new("RGB", img_copy.size, (240, 240, 240))
                rgb.paste(
                    img_copy,
                    mask=img_copy.split()[-1]
                    if img_copy.mode in ("RGBA", "LA")
                    else None,
                )
                img_copy = rgb
            elif img_copy.mode != "RGB":
                img_copy = img_copy.convert("RGB")

            canvas.paste(img_copy, offset)

            buf = BytesIO()
            canvas.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return buf.getvalue()

    except Exception as exc:
        print(f"Error generating thumbnail for {image_path}: {exc}")
        traceback.print_exc()
        return b""


def get_sorted_images(
    offset: int, batch_size: int = BATCH_SIZE
) -> Tuple[List[Dict], int]:
    """
    Return (images_data, total) with images grouped/sorted by label.

    Images are ordered: unlabeled first, then labels 1–7.
    """
    all_images = get_image_files()

    groups: Dict[str, List[Path]] = {
        k: [] for k in ["unlabeled", "1", "2", "3", "4", "5", "6", "7"]
    }

    for img_path in all_images:
        label = state.ANNOTATIONS_CACHE.get(img_path.name) or "unlabeled"
        if label in groups:
            groups[label].append(img_path)

    sorted_images: List[Path] = []
    for key in ["unlabeled", "1", "2", "3", "4", "5", "6", "7"]:
        sorted_images.extend(groups[key])

    batch = sorted_images[offset : offset + batch_size]
    images_data = [
        {
            "filename": p.name,
            "existing_label": state.ANNOTATIONS_CACHE.get(p.name),
        }
        for p in batch
    ]

    return images_data, len(sorted_images)
