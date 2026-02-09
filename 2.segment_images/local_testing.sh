#!/bin/bash
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit 1
patients=( "NF0030_T1" )
for patient in "${patients[@]}"; do
    data_dir="/home/lippincm/mnt/bandicoot/NF1_organoid_data/data/$patient/zstack_images/"
    for well_fov in "$data_dir"*/; do
        well_fov=$(basename "$well_fov")
        echo "Processing Patient: $patient, WellFOV: $well_fov"

        input_subparent_name="zstack_images"
        mask_subparent_name="segmentation_masks"

        echo "Patient: $patient, WellFOV: $well_fov,  Input Subparent Name: $input_subparent_name, Mask Subparent Name: $mask_subparent_name"

        echo "Beginning segmentation for $patient - $well_fov"
        conda activate GFF_segmentation_nuclei
        python 0.nuclei_segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name" --clip_limit 0.01
        conda deactivate ; conda activate GFF_segmentation
        python 1.cell_cyto_organoid_segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name" --clip_limit 0.01
        conda deactivate
    done
done
cd ../ || exit 1

echo "All segmentation child jobs ran"
