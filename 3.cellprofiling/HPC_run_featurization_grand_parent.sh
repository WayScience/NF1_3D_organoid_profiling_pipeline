#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=shared
#SBATCH --qos=long
#SBATCH --time=4-00:00:00 # D-HH:MM:SS
#SBATCH --output="logs/grand_parent/grand_parent-%j.out"


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

alpine_scratch_dir="/scratch/alpine"
anvil_scratch_dir="/anvil/scratch"

if [[ -e "$alpine_scratch_dir" ]]; then
    partition="amilan"
    qos="--qos=normal"
    max_jobs=990
    cluster="alpine"
elif [[ -e "$anvil_scratch_dir" ]]; then
    partition="shared"
    qos="--mail-type=all"
    max_cores=1280
    cluster="anvil"
else
    echo "Error: No known scratch directory found."
fi

txt_file="${git_root}/3.cellprofiling/load_data/load_combinations.txt"


# Check if TXT file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: TXT file not found at $txt_file"
    exit 1
fi


# parse the txt_file where each line contains
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

    echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type, InputSubparent: $input_subparent_name, MaskSubparent: $mask_subparent_name, OutputFeaturesSubparent: $output_features_subparent_name"


    # check that the number of jobs is less than 990
    # prior to submitting a job
    if [ "$cluster" == "alpine" ]; then
        number_of_jobs=$(squeue -u "$USER" | wc -l)
        while [ "$number_of_jobs" -gt "$max_jobs" ]; do
            sleep 1s
            number_of_jobs=$(squeue -u "$USER" | wc -l)
        done
    fi
    if [ "$cluster" == "anvil" ]; then
        number_of_cores=$(squeue -u "$USER" -h -o "%C" | awk '{sum += $1} END {print sum}')
        while [ "$number_of_cores" -gt "$max_cores" ]; do
            sleep 1s
            number_of_cores=$(squeue -u "$USER" -h -o "%C" | awk '{sum += $1} END {print sum}')
        done
    fi

    # shellcheck disable=SC1091
    source "$git_root"/3.cellprofiling/HPC_run_featurization_parent.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$feature" \
        "$processor_type" \
        "$input_subparent_name" \
        "$mask_subparent_name" \
        "$output_features_subparent_name"


done < "$txt_file"


echo "Featurization done"
