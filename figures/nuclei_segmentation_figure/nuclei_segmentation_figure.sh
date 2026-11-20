#!/bin/bash

set -euo pipefail

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/figures/nuclei_segmentation_figure/scripts/ "$git_root"/figures/nuclei_segmentation_figure/notebooks/*.ipynb

{
    conda deactivate 2>/dev/null &&
    deactivate 2>/dev/null
} || {
    echo "Note: no active conda/venv environment to deactivate" >&2
}

uv run python "$git_root"/figures/nuclei_segmentation_figure/scripts/segmentation_pipeline.py
