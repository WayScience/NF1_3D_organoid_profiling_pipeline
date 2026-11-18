#!/bin/bash

eu set -o pipefail

# this shell script runs the shipping of data to zip files for transfer
# and to sage synapse for public data availability

git_root=$(git rev-parse --show-toplevel)

module_dir="$git_root/figures/segmentation_workflow"
script_dir="$module_dir"/scripts

uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$module_dir/scripts/" "$module_dir/notebooks/"*.ipynb

uv run python "$script_dir"/cell_segmentation_decision_tree.py

echo "Segmentation workflow figure preparation completed."
