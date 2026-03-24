#!/bin/bash

patient=$1
well_fov=$2
input_subparent_name=$3
mask_subparent_name=$4
echo "Processing well_fov $well_fov for patient $patient"

uv run scripts/4.nuclei_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name" \
    --clip_limit 0.02

uv run scripts/5.cell_cyto_organoid_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.03 \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name"

echo "Segmentation completed for well_fov $well_fov and patient $patient"
