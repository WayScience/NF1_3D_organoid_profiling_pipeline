#!/bin/bash

# """Run deep learning features on local machine using GPU."""

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


rerun=$1


if [ "$rerun" == "rerun" ]; then
    txt_file="${git_root}/3.cellprofiling/load_data/rerun_combinations.txt"
else
    txt_file="${git_root}/3.cellprofiling/load_data/input_combinations.txt"
fi

# Check if TXT file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: TXT file not found at $txt_file"
    exit 1
fi

# get the total number of SAMMed3D entries
total_sammed3d_entries=0
while IFS= read -r line; do
    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    feature="${parts[4]}"
    # check if the feature is SAMMed3D
    if [ "$feature" == "SAMMed3D" ]; then
        total_sammed3d_entries=$((total_sammed3d_entries + 1))
    fi
done < "$txt_file"

processed_entries=0
# parse the txt_file where each line contains
# patient, well_fov, compartment, channel, feature, processor_type
while IFS= read -r line; do

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    compartment="${parts[2]}"
    channel="${parts[3]}"
    feature="${parts[4]}"
    processor_type="${parts[5]}"
    input_subparent_name="${parts[6]}"
    mask_subparent_name="${parts[7]}"
    output_features_subparent_name="${parts[8]}"

    # check if the feature is SAMMed3D
    if [ "$feature" != "SAMMed3D" ]; then
        continue
    fi
    echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type"

    if [ "$feature" == "SAMMed3D" ] ; then
        echo "Running SAMMed3D feature extraction"
        # ignore shellcheck SC1091
        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_sammed3D_child.sh \
            "$patient" \
            "$well_fov" \
            "$compartment" \
            "$channel"  \
            "$input_subparent_name" \
            "$mask_subparent_name" \
            "$output_features_subparent_name"
    fi
    processed_entries=$((processed_entries + 1))
    echo "Processed $processed_entries/$total_sammed3d_entries"

done < <(tac "$txt_file")

echo "Featurization done"

