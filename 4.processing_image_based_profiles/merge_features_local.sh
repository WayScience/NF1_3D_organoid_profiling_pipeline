#!/bin/bash

git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    echo "Using Alpine environment"
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_uv_env/.venv"
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_uv_env/.venv"
else
    ENV_PATH="$git_root/.venv"
fi

PYTHON_BIN="$ENV_PATH/bin/python3"
jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/4.processing_image_based_profiles/scripts/ "$git_root"/4.processing_image_based_profiles/notebooks/*.ipynb

load_data_file_path="$git_root/4.processing_image_based_profiles/load_data/load_file.txt"
patient_ids_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs into an array
mapfile -t patient_array < "$patient_ids_file_path"
# setup the logs dir
if [ -d "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/" ]; then
    rm -rf "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/"
fi
mkdir -p "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/" # create the patients directory if it doesn't exist




while IFS= read -r line; do

    IFS=$'\t' read -r -a parts <<< "$line"
    patient="${parts[0]}"
    well_fov="${parts[1]}"

    echo "$patient - $well_fov"
    log_file="$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/${patient}_${well_fov}.log"
    touch "$log_file"  # create the log file if it doesn't exist
    {
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py --patient "$patient" --well_fov "$well_fov" --output_features_subparent_name "extracted_features" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/2.merge_sc.py --patient "$patient" --well_fov "$well_fov" --output_features_subparent_name "extracted_features" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py --patient "$patient" --well_fov "$well_fov" --output_features_subparent_name "extracted_features" --image_based_profiles_subparent_name "image_based_profiles"
    } >> "$log_file" 2>&1
done < "$load_data_file_path"

for patient in "${patient_array[@]}"; do
    echo "Processing patient: $patient"

    patient_log_file="$git_root/4.processing_image_based_profiles/logs/patients/${patient}.log"
    mkdir -p "$(dirname "$patient_log_file")"  # create the patients directory if it doesn't exist
    touch "$patient_log_file"  # create the patient log file if it doesn't exist
    {
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/5.combining_profiles.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/6.annotation.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/7a.organoid_qc.py  --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/7b.single_cell_qc.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/8.normalization.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/9.feature_selection.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/10.aggregation.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
    } >> "$patient_log_file" 2>&1

done


# PYTHON_BIN "$git_root"/4.processing_image_based_profiles/scripts/11.combine_patients.py --image_based_profiles_subparent_name "image_based_profiles"
# PYTHON_BIN "$git_root"/4.processing_image_based_profiles/scripts/0a.get_profiling_stats.py --image_based_profiles_subparent_name "image_based_profiles"
# Rscript "$git_root"/4.processing_image_based_profiles/scripts/0b.plot_profiling_stats.r

echo "All features merged for patients" "${patient_array[@]}"

