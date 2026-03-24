#!/bin/bash

patient=$1
well_fov=$2
compartment=$3
channel=$4
feature=$5
processor_type=$6
input_subparent_name=$7
mask_subparent_name=$8
output_features_subparent_name=$9

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi


echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type"
echo "InputSubparent: $input_subparent_name, MaskSubparent: $mask_subparent_name, OutputFeaturesSubparent: $output_features_subparent_name"
# regardless of the processor type, texture and neighbors features are run on CPU
if [ "$feature" == "Neighbors" ]; then
    sbatch \
    --nodes=1 \
    --ntasks=1 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=1:00 \
    --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
    --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_neighbors_child-%j.out" \
    "$git_root"/3.cellprofiling/slurm_scripts/run_neighbors_child.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$input_subparent_name" \
        "$mask_subparent_name" \
        "$output_features_subparent_name"
fi

if [ "$feature" == "Granularity" ] ; then
    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=5:00 \
        --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
        --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_granularity_child-%j.out" \
        "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "CPU" \
        "$input_subparent_name" \
        "$mask_subparent_name" \
        "$output_features_subparent_name"
fi

if [ "$feature" == "Texture" ] ; then
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=5:00 \
        --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
        --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_texture_child-%j.out" \
        "$git_root"/3.cellprofiling/slurm_scripts/run_texture_child.sh \
            "$patient" \
            "$well_fov" \
            "$compartment" \
            "$channel" \
            "$input_subparent_name" \
            "$mask_subparent_name" \
            "$output_features_subparent_name"
fi


# AreaSizeShape feature extraction
if [ "$feature" == "AreaSizeShape" ] ; then
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=00:00:30 \
        --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
        --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_area_shape_child-%j.out" \
        "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$processor_type" \
        "$input_subparent_name" \
        "$mask_subparent_name" \
        "$output_features_subparent_name"
fi
if [ "$feature" == "Colocalization" ] ; then
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=2:00 \
        --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
        --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_colocalization_child-%j.out" \
        "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$processor_type" \
        "$input_subparent_name" \
        "$mask_subparent_name" \
        "$output_features_subparent_name"
fi

if [ "$feature" == "Intensity" ] ; then
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=2:00 \
        --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
        --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_intensity_child-%j.out" \
        "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$processor_type" \
        "$input_subparent_name" \
        "$mask_subparent_name" \
        "$output_features_subparent_name"
fi

echo "All Parent Jobs submitted"

