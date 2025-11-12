#!/bin/bash


# activate cellprofiler environment
# The following environment activation commands are commented out.
# Ensure the required environment is activated manually before running this script,
# or confirm that activation is handled by a parent script or workflow.
# find the git repository root directory
# check if on slurms or local
module load anaconda
conda init
conda activate GFF_segmentation_nuclei


patient=$1
well_fov=$2
input_subparent_name=$3
mask_subparent_name=$4
echo "Processing well_fov $well_fov for patient $patient"

start_time=$(date +%s)

python scripts/0.nuclei_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name" \
    --clip_limit 0.02

conda deactivate
conda activate GFF_segmentation

python scripts/1.segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.02 \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name"


# python scripts/5.clean_up_segmentation.py \
#     --patient "$patient" \
#     --well_fov "$well_fov" \
#     --input_subparent_name "$input_subparent_name" \
#     --mask_subparent_name "$mask_subparent_name"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo "Total process took ${hours}h ${minutes}m ${seconds}s (${elapsed_time} seconds)"

conda deactivate

echo "Segmentation completed for well_fov $well_fov and patient $patient"
