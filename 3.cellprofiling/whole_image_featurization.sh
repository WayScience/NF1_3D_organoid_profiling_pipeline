#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

conda activate GFF_segmentation_nuclei

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi

mkdir -p logs/whole_image_featurization/
cd scripts/ || exit 1

total_patients=${#patient_array[@]}
patient_counter=0
for patient_id in "${patient_array[@]}"; do
    well_fov_path="$HOME/mnt/bandicoot/NF1_organoid_data/data/$patient_id/zstack_images/"
    mapfile -t well_fovs < <(find "$well_fov_path" -maxdepth 1 -type d)
    echo "Processing patient: $patient_id"
    total_well_fovs=${#well_fovs[@]}
    well_fov_counter=0
    for well_fov in "${well_fovs[@]}"; do
        well_fov=$(basename "$well_fov")
        well_fov_log_path="../logs/whole_image_featurization/${patient_id}_${well_fov}_log.txt"
        touch "$well_fov_log_path"
        {
            python run_chami_75_on_all_images.py --patient "$patient_id" --well_fov "$well_fov" --input_subparent_name "zstack_images"
        } &> "$well_fov_log_path"
        ((well_fov_counter++))
        echo -e "$well_fov ($well_fov_counter/$total_well_fovs) - $patient_id ($patient_counter/$total_patients)"
    done
    ((patient_counter++))
done

conda deactivate
cd ../ || exit 1

echo "All done!"
