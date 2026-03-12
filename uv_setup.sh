#!/bin/bash


# run twice to ensure we are in a clean environment
# and not accidentally using an existing one
# try but will not fail if not in a conda environment
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

uv venv
uv sync
# shellcheck disable=SC1091
source .venv/bin/activate

# Use RELATIVE path - simple and reliable
uv pip install -e ./utils


echo "✓ Done! Activate with: source .venv/bin/activate"
