#!/bin/bash

# activate  cellprofiler environment
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit
# get all input directories in specified directory
patient="NF0014"
well_fov="E10-2"

compartments=( "nuclei" "organoid" )


python 0.segment_nuclei.py \
--patient "$patient" \
--well_fov "$well_fov" \
--window_size 3 \
--clip_limit 0.05

python 1.segment_whole_organoids.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 4 \
    --clip_limit 0.1

for compartment in "${compartments[@]}"; do

    if [ "$compartment" == "nuclei" ]; then
        window_size=3
    elif [ "$compartment" == "organoid" ]; then
        window_size=4
    else
        echo "Not specified compartment: $compartment"

    fi
    python 2.segmentation_decoupling.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment" \
        --window_size "$window_size"

    python 3.reconstruct_3D_masks.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"

    python 4.post-hoc_mask_refinement.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"
done

python 5.segment_cells_watershed_method.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.05

python 4.post-hoc_mask_refinement.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --compartment "cell"

python 6.post-hoc_reassignment.py \
    --patient "$patient" \
    --well_fov "$well_fov"

python 7.create_cytoplasm_masks.py \
    --patient "$patient" \
    --well_fov "$well_fov"

conda deactivate


cd ../ || exit

echo "Segmentation complete"
