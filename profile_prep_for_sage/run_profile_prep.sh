#!/bin/bash

# this shell script runs the shipping of data to zip files for transfer
# and to sage synapse for public data availability

git_root=$(git rev-parse --show-toplevel)

module_dir="$git_root/profile_prep_for_sage"
script_dir="$module_dir"/scripts

uv run jupyter nbconvert --to=script --FilesWriter.build_directory="$module_dir/scripts/" "$module_dir/notebooks/"*.ipynb

uv run python "$script_dir"/prepare_notebooks_for_sage.py
uv run python "$script_dir"/zip_files_for_transfer.py

echo "Profile preparation for Sage completed."
