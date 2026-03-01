#!/bin/bash
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
conda activate GFF_segmentation_nuclei

cd scripts/ || exit 1
patients=( "NF0014_T1" "NF0016_T1" "NF0030_T1" "NF0037_T1" )
for patient in "${patients[@]}"; do
    data_dir="/home/lippincm/mnt/bandicoot/NF1_organoid_data/data/$patient/zstack_images/"
    for well_fov in "$data_dir"*/; do
        well_fov=$(basename "$well_fov")
        echo "Processing Patient: $patient, WellFOV: $well_fov"

#         input_subparent_name="zstack_images"
#         mask_subparent_name="segmentation_masks"

#         echo "Patient: $patient, WellFOV: $well_fov,  Input Subparent Name: $input_subparent_name, Mask Subparent Name: $mask_subparent_name"

        echo "Beginning segmentation for $patient - $well_fov"
        python 4.nuclei_segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name" --clip_limit 0.01
        python 5.cell_cyto_organoid_segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name" --clip_limit 0.01
    done
done
cd ../ || exit 1

conda deactivate

echo "All segmentation child jobs ran"
