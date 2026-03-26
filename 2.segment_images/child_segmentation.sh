#!/bin/bash

git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    echo "Using Alpine environment"
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_uv_env/.venv"
    module load cuda/11.3
elif [ -d "/anvil" ]; then
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_uv_env/.venv"
else
    ENV_PATH="$git_root/.venv"
fi

PYTHON_BIN="$ENV_PATH/bin/python3"

# shellcheck disable=SC1091
source "$ENV_PATH"/bin/activate

patient=$1
well_fov=$2
input_subparent_name=$3
mask_subparent_name=$4

log_dir="$git_root/2.segment_images/logs/child"
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
log_file="$log_dir/segmentation_child_${patient}_${well_fov}.log"


{
    echo "Processing well_fov $well_fov for patient $patient"
    "$PYTHON_BIN"  scripts/4.nuclei_segmentation.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --input_subparent_name "$input_subparent_name" \
        --mask_subparent_name "$mask_subparent_name" \
        --clip_limit 0.02

    "$PYTHON_BIN"  scripts/5.cell_segmentation.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --clip_limit 0.03 \
        --input_subparent_name "$input_subparent_name" \
        --mask_subparent_name "$mask_subparent_name"

    "$PYTHON_BIN"  scripts/6.organoid_segmentation.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --clip_limit 0.03 \
        --input_subparent_name "$input_subparent_name" \
        --mask_subparent_name "$mask_subparent_name"

    echo "Segmentation completed for well_fov $well_fov and patient $patient"

} &> "$log_file"
