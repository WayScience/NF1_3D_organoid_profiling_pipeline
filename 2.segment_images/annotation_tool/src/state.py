"""
Mutable runtime state shared across modules.

All variables are module-level so that imports like
    import state; state.IMAGE_DIR = ...
will be visible to every other module.
"""

from pathlib import Path
from typing import Dict, Optional

IMAGE_DIR: Optional[Path] = None
PARQUET_FILE: Optional[Path] = None
ANNOTATOR: str = "default"

# filename -> label-number string (e.g. "1")
ANNOTATIONS_CACHE: Dict[str, str] = {}

# When True, /api/images returns images globally sorted by label
SORT_BY_LABEL_MODE: bool = False
