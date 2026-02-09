#!/bin/bash
patient=$1
well_fov=$2
compartment=$3
channel=$4
processor_type=$5
input_subparent_name=$6
mask_subparent_name=$7
output_features_subparent_name=$8

echo "Colocalization feature extraction for patient: $patient, WellFOV: $well_fov, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type"
module load miniforge
conda init bash
conda activate GFF_featurization

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

# start the timer
start_timestamp=$(date +%s)
if [ "$processor_type" = "GPU" ]; then
    echo "Running GPU version"
    python "$git_root"/3.cellprofiling/scripts/colocalization.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment" \
        --channel "$channel" \
        --processor_type "GPU" \
        --input_subparent_name "$input_subparent_name" \
        --mask_subparent_name "$mask_subparent_name" \
        --output_features_subparent_name "$output_features_subparent_name"
else
    echo "Running CPU version"
    python "$git_root"/3.cellprofiling/scripts/colocalization.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment" \
        --channel "$channel" \
        --processor_type "CPU" \
        --input_subparent_name "$input_subparent_name" \
        --mask_subparent_name "$mask_subparent_name" \
        --output_features_subparent_name "$output_features_subparent_name"
fi

end=$(date +%s)
echo "Time taken to run the featurization: (($end-$start_timestamp))"

conda deactivate
