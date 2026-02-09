#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=3-00:00:00 # D-HH:MM:SS
#SBATCH --output="logs/grand_parent/grand_parent-%j.out"

module load anaconda
conda init bash
conda activate GFF_featurization

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

rerun=$1

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/3.cellprofiling/scripts/ "$git_root"/3.cellprofiling/notebooks/*.ipynb

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
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    bash "$git_root"/3.cellprofiling/HPC_run_featurization_parent.sh \
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
