#!/bin/bash


conda activate gff_preprocessing_env

# get the current directory
# get a list of all directories in the data/z-stack_images directory

dirs_list=$(ls -d ../../data/z-stack_images/*)

for dir in $dirs_list; do
    echo "Processing directory: $dir"
    python 2.z_slice_instensity_normalization.py --input_dir $dir
done

conda deactivate

echo "Z-slice intensity normalization complete"


