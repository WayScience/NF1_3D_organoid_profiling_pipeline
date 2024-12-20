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
    papermill 2.make_nuclei_segmentation_videos.ipynb 2.make_nuclei_segmentation_videos.ipynb
    papermill 3.make_cell_segmentation_videos.ipynb 3.make_cell_segmentation_videos.ipynb
    cd ../ || exit
fi

if [ "$NOTEBOOK" = False ]; then
    cd scripts/ || exit
    # get all input directories in specified directory
    z_stack_dir="../../data/z-stack_images/"
    input_dirs=$(ls -d $z_stack_dir*)


    total_dirs=$(echo "$input_dirs" | wc -l)
    current_dir=0

    # loop through all input directories
    for dir in $input_dirs; do
        dir=${dir%*/}
        current_dir=$((current_dir + 1))
        echo -ne "Processing directory $current_dir of $total_dirs\r"
        python 0.segment_nuclei_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.05
        python 1.segment_cells_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.1
        python 2.make_nuclei_segmentation_videos.py --input_dir "$dir"
        python 3.make_cell_segmentation_videos.py --input_dir "$dir"
    done
    echo -ne "\n"
    cd ../ || exit
fi

# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
