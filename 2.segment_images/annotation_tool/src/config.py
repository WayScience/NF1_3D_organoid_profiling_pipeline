"""
Annotation tool constants and configuration.
"""

from typing import Dict

# Label definitions  (display_name -> numeric string key)
LABELS: Dict[str, str] = {
    "globular": "1",
    "small": "2",
    "dissociated": "3",
    "elongated": "4",
    "blank": "5",
    "cluster": "6",
    "failed": "7",
}

THUMBNAIL_SIZE: int = 140  # px – square thumbnail side length
BATCH_SIZE: int = 500  # images per page/batch
GRID_COLS: int = 15
GRID_ROWS: int = 10
