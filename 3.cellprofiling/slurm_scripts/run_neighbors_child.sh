#!/bin/bash

patient=$1
well_fov=$2
compartment=$3
channel=$4
input_subparent_name=$5
mask_subparent_name=$6
output_features_subparent_name=$7

echo "Neighbors feature extraction for patient: $patient, WellFOV: $well_fov, Compartment: $compartment, Channel: $channel, UseGPU: CPU"

git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_uv_env/.venv"
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_uv_env/.venv"
else
    ENV_PATH="$git_root/.venv"
fi


# shellcheck disable=SC1091
source "$ENV_PATH"/bin/activate


uv run python "$git_root"/3.cellprofiling/scripts/neighbors.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --compartment "$compartment" \
    --channel "$channel" \
    --processor_type "CPU" \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name" \
    --output_features_subparent_name "$output_features_subparent_name"
