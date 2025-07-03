#!/bin/bash

conda activate gff_preprocessing_env
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


GPU_OPTIONS=( "TRUE" "FALSE" )

json_file="${git_root}/3.cellprofiling/load_data/input_combinations.json"

# Check if JSON file exists
if [ ! -f "$json_file" ]; then
    echo "Error: JSON file not found at $json_file"
    exit 1
fi

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi

for patient in "${patient_array[@]}"; do
    echo "Processing patient: $patient"
    # Check if the patient directory exists
    if [ ! -d "$git_root/data/$patient/zstack_images" ]; then
        echo "Error: Directory for patient $patient does not exist."
        continue
    fi

    parent_dir="${git_root}/data/${patient}/zstack_images"
    # get the list of all dirs in the parent_dir
    dirs=$(ls -d "$parent_dir"/*)
    # Read JSON array and process each object
    jq -r '.[] | "\(.feature) \(.compartment) \(.channel)"' "$json_file" | while read -r feature compartment channel; do
        for dir in $dirs; do
            echo "Processing feature: $feature, Compartment: $compartment, Channel: $channel"
            # neighbors and texture are always on CPU
            well_fov=$(basename "$dir")
            if [ "$feature" == "Neighbors" ] || [ "$feature" == "Texture" ]; then

                source "$git_root"/3.cellprofiling/slurm_scripts/run_neighbors_child.sh \
                        "$patient" \
                        "$well_fov" \
                        "$compartment" \
                        "$channel" \
                        "$use_GPU"
            fi
            if [ "$feature" == "Texture" ] ; then
                echo "Running texture feature extraction"
                source "$git_root"/3.cellprofiling/slurm_scripts/run_texture_child.sh \
                    "$patient" \
                    "$well_fov" \
                    "$compartment" \
                    "$channel" \
                    "$use_GPU"
            fi

            for use_GPU in "${GPU_OPTIONS[@]}"; do
            # iterate through for each option
                echo "$well_fov"
                echo "Feature: $feature, Compartment: $compartment, Channel: $channel"

                echo "Running GPU version"
                if [ "$feature" == "AreaSizeShape" ] ; then

                    source "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh \
                    "$patient" \
                    "$well_fov" \
                    "$compartment" \
                    "$channel" \
                    "$use_GPU"
                fi
                if [ "$feature" == "Colocalization" ] ; then
                    source "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh \
                        "$patient" \
                        "$well_fov" \
                        "$compartment" \
                        "$channel" \
                        "$use_GPU"
                fi
                if [ "$feature" == "Granularity" ] ; then
                    source "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh \
                        "$patient" \
                        "$well_fov" \
                        "$compartment" \
                        "$channel" \
                        "$use_GPU"
                fi
                if [ "$feature" == "Intensity" ] ; then
                    source "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh \
                        "$patient" \
                        "$well_fov" \
                        "$compartment" \
                        "$channel" \
                        "$use_GPU"
                fi
            done
        done
    done
done

echo "Featurization done"
