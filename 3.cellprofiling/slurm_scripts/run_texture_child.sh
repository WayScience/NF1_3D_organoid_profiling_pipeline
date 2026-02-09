#!/bin/bash

patient=$1
well_fov=$2
compartment=$3
channel=$4
input_subparent_name=$5
mask_subparent_name=$6
output_features_subparent_name=$7

echo "Texture feature extraction for patient: $patient, WellFOV: $well_fov, Compartment: $compartment, Channel: $channel, UseGPU: CPU"
module load miniforge
conda init bash
conda activate GFF_featurization

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

echo "Running featurization for $patient $well_fov"

# start the timer
start_timestamp=$(date +%s)
python "$git_root"/3.cellprofiling/scripts/texture.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --compartment "$compartment" \
    --channel "$channel" \
    --processor_type "CPU" \
    --input_subparent_name "$input_subparent_name" \
    --mask_subparent_name "$mask_subparent_name" \
    --output_features_subparent_name "$output_features_subparent_name"

end=$(date +%s)
echo "Time taken to run the featurization: (($end-$start_timestamp))"

conda deactivate
