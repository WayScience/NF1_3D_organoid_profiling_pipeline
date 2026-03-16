#!/bin/bash

# run twice to ensure we are in a clean environment
# and not accidentally using an existing one
# try but will not fail if not in a conda environment
export SCRATCH_BASE="/scratch/$USER"
export UV_CACHE_DIR="$SCRATCH_BASE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_BASE/.local/share/uv/python"

loops=2
for _ in $(seq 1 $loops); do
    if command -v conda >/dev/null 2>&1; then
        conda deactivate >/dev/null 2>&1 || true
    fi
    deactivate >/dev/null 2>&1 || true
done
unset VIRTUAL_ENV


rm -f uv.lock
rm -rf .venv

uv venv /scratch/"$USER"/venvs/nf1-3d

# shellcheck disable=SC1090
source /scratch/"$USER"/venvs/nf1-3d/bin/activate
uv sync --frozen

# shellcheck disable=SC1091
source .venv/bin/activate

# shellcheck disable=SC1091
source .venv/bin/activate
# Use RELATIVE path - simple and reliable
uv pip install -e ./utils
uv pip install cosmicqc


echo "✓ Done! Activate with: source .venv/bin/activate"
