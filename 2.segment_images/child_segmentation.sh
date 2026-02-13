#!/bin/bash


# activate cellprofiler environment
# The following environment activation commands are commented out.
# Ensure the required environment is activated manually before running this script,
# or confirm that activation is handled by a parent script or workflow.
# find the git repository root directory
# check if on slurms or local
# module load anaconda
# conda init

patient=$1
well_fov=$2
input_subparent_name=$3
mask_subparent_name=$4
echo "Processing well_fov $well_fov for patient $patient"

conda activate GFF_segmentation_nuclei

python scripts/0.nuclei_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name" \
    --clip_limit 0.01

conda deactivate
conda activate GFF_segmentation

python scripts/1.cell_cyto_organoid_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.01 \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name"


conda deactivate

echo "Segmentation completed for well_fov $well_fov and patient $patient"
