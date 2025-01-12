#!/bin/bash

# activate  cellprofiler environment
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit
# get all input directories in specified directory
z_stack_dir="../../data/z-stack_images"
# z_stack_dir="../../data/test_dir/"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)

total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0
compartments=( "nuclei" "cell" )

# loop through all input directories
for dir in "${input_dirs[@]}"; do
    dir=${dir%*/}
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    python 0.segment_nuclei_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.05
    python 1.segment_cells_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.1
    python 2.segment_whole_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.1
    for compartment in "${compartments[@]}"; do
        python 3.segmentation_decoupling.py --input_dir "$dir" --compartment "$compartment"
        python 4.reconstruct_3D_masks.py --input_dir "$dir" --compartment "$compartment" --radius_constraint 10
        python 6.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "$compartment"
    done
    python 5.create_cytoplasm_masks.py --input_dir "$dir"
    python 6.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "cytoplasm"
    python 7.clean_up_segmentation.py

done
echo -ne "\n"
cd ../ || exit


# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
