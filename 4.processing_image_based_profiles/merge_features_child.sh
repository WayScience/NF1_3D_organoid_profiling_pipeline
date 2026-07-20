#!/bin/bash

patient=$1
well_fov=$2

echo "Patient: $patient, Well/FOV: $well_fov"

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

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

log_file="$git_root/4.processing_image_based_profiles/logs/${patient}_${well_fov}.log"
# if the file is present, first remove it
if [ -f "$log_file" ]; then
    rm "$log_file"
fi
mkdir -p "$git_root/4.processing_image_based_profiles/logs"
{
    "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --output_features_subparent_name "extracted_features" \
        --image_based_profiles_subparent_name "image_based_profiles"
    "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/2.merge_sc.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --output_features_subparent_name "extracted_features" \
        --image_based_profiles_subparent_name "image_based_profiles"
    "$PYTHON_BIN" "$git_root"/4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --output_features_subparent_name "extracted_features" \
        --image_based_profiles_subparent_name "image_based_profiles"
} >> "$log_file" 2>&1

echo "Patient $patient well_fov $well_fov completed"
