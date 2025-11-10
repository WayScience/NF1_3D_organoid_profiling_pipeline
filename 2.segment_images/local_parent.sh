#!/bin/bash


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ "$git_root"/2.segment_images/notebooks/*.ipynb
BANDICOOT=TRUE
BANDICOOT_PATH="$HOME/mnt/bandicoot/NF1_organoid_data"
# patient_array=( "NF0014_T1" "NF0014_T2" "NF0016_T1" "NF0018_T1" "NF0021_T1" "NF0030_T1" "NF0031_T1" "NF0035_T1" "NF0037_T1-Z-1" "NF0037_T1-Z-0.5" "NF0037_T1-Z-0.2" "NF0037_T1-Z-0.1" "NF0040_T1" "SARCO219_T2" "SARCO361_T1" )

patient_array=( "NF0016_T1" "NF0018_T1" "NF0021_T1" "NF0030_T1" "NF0031_T1" "NF0035_T1" "NF0037_T1-Z-1" "NF0037_T1-Z-0.5" "NF0037_T1-Z-0.2" "NF0037_T1-Z-0.1" "NF0040_T1" "SARCO219_T2" "SARCO361_T1" )
for patient in "${patient_array[@]}"; do

    # get all input directories in specified directory
    if [ "$BANDICOOT" = TRUE ]; then
        git_root="$BANDICOOT_PATH"
    fi
    z_stack_dir="$git_root/data/$patient/zstack_images"
    mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)

    # loop through all input directories
    for well_fov in "${input_dirs[@]}"; do

        well_fov=$(basename "$well_fov")
        current_dir=$((current_dir + 1))
        echo "Beginning segmentation for $patient - $well_fov"
        bash child_segmentation.sh "$patient" "$well_fov" "zstack_images" "segmentation_masks"
    done

done


echo "All segmentation child jobs ran"

