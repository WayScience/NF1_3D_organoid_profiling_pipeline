#!/bin/bash

# """Run deep learning features on local machine using GPU."""

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

txt_file="${git_root}/3.cellprofiling/load_data/load_combinations.txt"


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
    feature="${parts[2]}"
    # check if the feature is SAMMed3D
    if [ "$feature" == "SAMMed3D" ]; then
        total_sammed3d_entries=$((total_sammed3d_entries + 1))
    fi
done < "$txt_file"

log_file="${git_root}/3.cellprofiling/logs/featurization_log.txt"
# if log file doesn't exist, create it and the parent directory if it doesn't exist
if [ ! -f "$log_file" ]; then
    mkdir -p "$(dirname "$log_file")"
fi
touch "$log_file"

processed_entries=0
# parse the txt_file where each line contains
# patient, well_fov, compartment, channel, feature, processor_type
while IFS= read -r line; do

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    feature="${parts[2]}"
    compartment="${parts[3]}"
    channel="${parts[4]}"
    # shellcheck disable=SC2034
    processor_type="${parts[5]}"
    input_subparent_name="${parts[6]}"
    mask_subparent_name="${parts[7]}"
    output_features_subparent_name="${parts[8]}"

    # check if the feature is SAMMed3D
    if [ "$feature" != "SAMMed3D" ]; then
        continue
    fi

    {
    if [ "$feature" == "SAMMed3D" ] ; then
        if [ "$compartment" == "Nucleocentric" ] ; then
            # shellcheck disable=SC1091
            source "$git_root"/3.cellprofiling/slurm_scripts/run_nucleocentric_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel"  \
                "$input_subparent_name" \
                "$mask_subparent_name" \
                "$output_features_subparent_name"
        else
            echo "Running SAMMed3D feature extraction" >> "$log_file"
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
    fi
    } >> "$log_file" 2>&1
    processed_entries=$((processed_entries + 1))
    echo "Processed $processed_entries/$total_sammed3d_entries"

done < <(tac "$txt_file")
# done < "$txt_file"

echo "Featurization done"

