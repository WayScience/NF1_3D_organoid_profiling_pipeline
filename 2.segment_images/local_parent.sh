#!/bin/bash

conda init bash
conda activate GFF_segmentation

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ "$git_root"/2.segment_images/notebooks/*.ipynb

# patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "NF0030" "NF0040" "SARCO219" "SARCO361" )
patient_array=( "NF0037_T1-Z-0.5" "NF0037_T1-Z0.2" "NF0037_T1-Z0.1" )

for patient in "${patient_array[@]}"; do

    # get all input directories in specified directory
    z_stack_dir="$git_root/data/$patient/zstack_images"
    mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)

    # loop through all input directories
    for well_fov in "${input_dirs[@]}"; do

        well_fov=$(basename "$well_fov")
        current_dir=$((current_dir + 1))
        echo "Beginning segmentation for $patient - $well_fov"
        bash "$git_root"/2.segment_images/child_segmentation.sh "$patient" "$well_fov" "zstack_images" "segmentation_masks"
    done

done


# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"

