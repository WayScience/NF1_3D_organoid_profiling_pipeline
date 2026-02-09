#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=60:00
#SBATCH --output=merge_features_parent%j.out


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/4.processing_image_based_profiles/scripts/ "$git_root"/4.processing_image_based_profiles/notebooks/*.ipynb

load_data_file_path="$git_root/4.processing_image_based_profiles/load_data/load_file.txt"


while IFS= read -r line; do

    IFS=$'\t' read -r -a parts <<< "$line"
    patient="${parts[0]}"
    well_fov="${parts[1]}"

    # check that the number of jobs is less than 990
    # prior to submitting a job
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=2:00 \
        --output=child_featurize-%j.out \
        "$git_root"/4.processing_image_based_profiles/merge_features_child.sh "$patient" "$well_fov"

done < "$load_data_file_path"


echo "All well_fov submitted for patient $patient"
