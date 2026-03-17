#!/bin/bash

git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_uv_env"
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_uv_env"
else
    ENV_PATH="$git_root/.venv"
fi


# shellcheck disable=SC1091
source "$ENV_PATH"/bin/activate

patient=$1
well_fov=$2
input_subparent_name=$3
mask_subparent_name=$4
echo "Processing well_fov $well_fov for patient $patient"

uv run python scripts/4.nuclei_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name" \
    --clip_limit 0.02

uv run python scripts/5.cell_cyto_organoid_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.03 \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name"

echo "Segmentation completed for well_fov $well_fov and patient $patient"
