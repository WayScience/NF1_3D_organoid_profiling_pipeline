#!/bin/bash

NOTEBOOK=False

# activate  cellprofiler environment
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# run the notebooks with papermill
if [ "$NOTEBOOK" = True ]; then
    cd notebooks/ || exit
    papermill 0.segment_nuclei_organoids.ipynb 0.segment_nuclei_organoids.ipynb
    papermill 1.segment_cells_organoids.ipynb 1.segment_cells_organoids.ipynb
    papermill 2.segmentation_decoupling.ipynb 2.segmentation_decoupling.ipynb
    papermill 3.reconstruct_3D_masks.ipynb 3.reconstruct_3D_masks.ipynb
    papermill 4.create_cytoplasm_masks.ipynb 4.create_cytoplasm_masks.ipynb
    cd ../ || exit
    jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
fi

if [ "$NOTEBOOK" = False ]; then
    cd scripts/ || exit
    # get all input directories in specified directory
    # z_stack_dir="../../data/z-stack_images/"
    z_stack_dir="../../data/test_dir/"
    mapfile -t input_dirs < <(ls -d "$z_stack_dir"*)

    # subset the input directories for testing
    # input_dirs="${input_dirs[@]:0:1}"

    total_dirs=$(echo "$input_dirs" | wc -l)
    current_dir=0
    compartments=( "nuclei" "cell" )

    # loop through all input directories
    for dir in $input_dirs; do
        dir=${dir%*/}
        current_dir=$((current_dir + 1))
        echo -ne "Processing directory $current_dir of $total_dirs\r"
        python 0.segment_nuclei_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.05
        python 1.segment_cells_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.1
        for compartment in "${compartments[@]}"; do
            python 2.segmentation_decoupling.py --input_dir "$dir" --compartment "$compartment"
            python 3.reconstruct_3D_masks.py --input_dir "$dir" --compartment "$compartment" --radius_constraint 10
            python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "$compartment"
        done
        python 4.create_cytoplasm_masks.py --input_dir "$dir"
        python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "cytoplasm"
        python 6.clean_up_segmentation.py

    done
    echo -ne "\n"
    cd ../ || exit
fi

# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
