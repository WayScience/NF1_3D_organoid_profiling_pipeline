#!/bin/bash

# activate  cellprofiler environment
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit
# get all input directories in specified directory
z_stack_dir="../../data/NF0014/zstack_images/"
mapfile -t well_fovs < <(ls -d "$z_stack_dir"/*)

total_dirs=$(echo "${well_fovs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0
compartments=( "nuclei" "cell" "organoid" )

touch segmentation.log
# loop through all input directories
for well_fov in "${well_fovs[@]}"; do
    well_fov=${well_fov%*/}
    well_fov=$(basename "$well_fov")
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    echo "Reslicing images"
    python 00.reslice_images.py --well_fov "$well_fov"
    echo "Segmenting Nuclei"
    python 0.segment_nuclei_organoids.py --well_fov "$well_fov" --window_size 2 --clip_limit 0.05 >> segmentation.log
    echo "Completed Nuclei Segmentation"
    echo "Segmenting Cells"
    python 1.segment_cells_organoids.py --well_fov "$well_fov" --window_size 3 --clip_limit 0.1 >> segmentation.log
    echo "Completed Cell Segmentation"
    echo "Segmenting Organoids"
    python 2.segment_whole_organoids.py --well_fov "$well_fov" --window_size 4 --clip_limit 0.1 >> segmentation.log
    echo "Completed Organoid Segmentation"
    for compartment in "${compartments[@]}"; do
        echo "Decoupling $compartment"
        python 3.segmentation_decoupling.py --well_fov "$well_fov" --compartment "$compartment" >> segmentation.log
        python 4.reconstruct_3D_masks.py --well_fov "$well_fov" --compartment "$compartment" --radius_constraint 10 >> segmentation.log
    done
    python 5.create_cytoplasm_masks.py --well_fov "$well_fov" >> segmentation.log
done
echo "Cleaning up segmentation files"
python 7.clean_up_segmentation.py >> segmentation.log
echo -ne "\n"
cd ../ || exit


# deactivate cellprofiler environment
conda deactivate

source local_make_gifs.sh

echo "Segmentation complete"
