#!/bin/bash

# activate  cellprofiler environment
conda activate viz_env

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

# get all input directories in specified directory
z_stack_dir="${git_root}/data/NF0014/z-stack_images"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)

total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

# loop through all input directories
for dir in "${input_dirs[@]}"; do
    dir=${dir%*/}
    # get the base name of the directory
    well_fov=$(basename "$dir")
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    python "$git_root"/2.segment_images/scripts/6.animate_segmentation_and_raw_signal.py --well_fov "$well_fov"
done

# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
