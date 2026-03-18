"""
Annotation persistence: read/write parquet files.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import state

from config import LABELS


def load_existing_annotations() -> None:
    """Load annotations from *state.PARQUET_FILE* into *state.ANNOTATIONS_CACHE*."""
    state.ANNOTATIONS_CACHE.clear()

    if not state.PARQUET_FILE or not state.PARQUET_FILE.exists():
        return

    try:
        table = pq.read_table(state.PARQUET_FILE)
        df = table.to_pandas()
        for _, row in df.iterrows():
            # Labels may be stored as int in older parquet files; normalize to str.
            label = str(row["label"])
            # Skip invalid labels: 0 = unlabeled, only load valid labels (1-7)
            if label not in LABELS.values():
                continue
            state.ANNOTATIONS_CACHE[row["image_filename"]] = label
    except Exception as exc:
        print(f"Warning: could not load existing annotations: {exc}")


def save_annotation(filename: str, label: str) -> None:
    """Append a single annotation row to *state.PARQUET_FILE*."""
    if not state.PARQUET_FILE:
        return

    # Parse patient and well_fov from filename
    # Expected format: NF0014_T1_F11-2_405.tiff or NF0037_T1_CQ1_F11-2_405.tiff
    parts = filename.split("_")
    if "CQ1" in filename:
        patient = "_".join(parts[:3])  # NF0037_T1_CQ1
        well_fov = parts[3] if len(parts) > 3 else "unknown"
    else:
        patient = "_".join(parts[:2])  # NF0014_T1
        well_fov = parts[2] if len(parts) > 2 else "unknown"

    label_name = next((k for k, v in LABELS.items() if v == label), label)

    # Convert label to int to match existing parquet schema
    label_int = int(label)

    # Convert timestamp to pandas Timestamp for proper schema matching
    import pandas as pd

    timestamp = pd.Timestamp.now()

    # Column order matches existing parquet file
    new_table = pa.table(
        {
            "patient": [patient],
            "well_fov": [well_fov],
            "label_name": [label_name],
            "image_filename": [filename],
            "annotator": [state.ANNOTATOR],
            "label": [label_int],
            "timestamp": [timestamp],
        }
    )

    # normalize to str so the cache is always string-keyed by label.
    label = str(label)

    # Always update the in-memory cache so counts are correct even if the
    # file write fails (e.g. the parent directory does not yet exist).
    state.ANNOTATIONS_CACHE[filename] = label

    try:
        # Ensure the output directory exists before writing.
        state.PARQUET_FILE.parent.mkdir(parents=True, exist_ok=True)

        if state.PARQUET_FILE.exists():
            existing_table = pq.read_table(state.PARQUET_FILE)
            # Cast new table to match existing schema exactly
            new_table = new_table.cast(existing_table.schema)
            combined_table = pa.concat_tables([existing_table, new_table])
        else:
            combined_table = new_table

        # Atomic write: write to temp file first, then rename
        temp_file = state.PARQUET_FILE.with_suffix(".parquet.tmp")
        pq.write_table(combined_table, temp_file)
        temp_file.replace(state.PARQUET_FILE)  # Atomic on POSIX systems

        print(f"✓ Saved annotation: {filename} → {label_name} (label={label})")
    except Exception as exc:
        print(f"✗ Error saving annotation to disk: {exc}")
        import traceback

        traceback.print_exc()
        # Clean up temp file if it exists
        temp_file = state.PARQUET_FILE.with_suffix(".parquet.tmp")
        if temp_file.exists():
            temp_file.unlink()
        raise  # Re-raise so the Flask endpoint can return an error
