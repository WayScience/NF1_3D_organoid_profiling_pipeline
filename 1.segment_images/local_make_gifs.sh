#!/bin/bash

# activate  cellprofiler environment
conda activate viz_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit
# get all input directories in specified directory
# z_stack_dir="../../data/NF0014/z-stack_images"
z_stack_dir="../../data/NF0014/test_dir/"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)

total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

touch segmentation.log
# loop through all input directories
for dir in "${input_dirs[@]}"; do
    dir=${dir%*/}
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    python 6.animate_segmentation_and_raw_signal.py --image_dir "$dir" >> segmentation.log
done

cd ../ || exit


# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
