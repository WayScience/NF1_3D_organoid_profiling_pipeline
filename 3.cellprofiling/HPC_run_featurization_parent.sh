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

ntasks_constant=2
time_constant="5:00"

echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type"
echo "InputSubparent: $input_subparent_name, MaskSubparent: $mask_subparent_name, OutputFeaturesSubparent: $output_features_subparent_name"
# regardless of the processor type, texture and neighbors features are run on CPU
if [ "$feature" == "Neighbors" ]; then
    echo "Running Neighbors feature extraction"
    sbatch \
    --nodes=1 \
    --ntasks=$ntasks_constant \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=$time_constant \
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
    echo "Running CPU version for Granularity"
    # might need to titrate time
    sbatch \
        --nodes=1 \
        --ntasks=$ntasks_constant \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=$time_constant \
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
    echo "Running texture feature extraction"
    sbatch \
        --nodes=1 \
        --ntasks=$ntasks_constant \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=$time_constant \
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
    if [ "$processor_type" == "CPU" ]; then
        echo "Running CPU version for AreaSizeShape"
        sbatch \
            --nodes=1 \
            --ntasks=$ntasks_constant \
            --partition=amilan \
            --qos=normal \
            --account=amc-general \
            --time=$time_constant \
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
    else
        echo "Running GPU version for AreaSizeShape"
        sbatch \
            --nodes=1 \
            --ntasks=$ntasks_constant \
            --partition=aa100 \
            --qos=normal \
            --gres=gpu:1 \
            --account=amc-general \
            --time=$time_constant \
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
fi
if [ "$feature" == "Colocalization" ] ; then
    if [ "$processor_type" == "CPU" ]; then
        echo "Running CPU version for Colocalization"
        sbatch \
            --nodes=1 \
            --ntasks=$ntasks_constant \
            --partition=amilan \
            --qos=normal \
            --account=amc-general \
            --time=$time_constant \
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
    else
        echo "Running GPU version for Colocalization"
        sbatch \
            --nodes=1 \
            --ntasks=$ntasks_constant \
            --partition=aa100 \
            --qos=normal \
            --gres=gpu:1 \
            --account=amc-general \
            --time=$time_constant \
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
fi

if [ "$feature" == "Intensity" ] ; then
    if [ "$processor_type" == "CPU" ]; then
        echo "Running CPU version for Intensity"
        sbatch \
            --nodes=1 \
            --ntasks=$ntasks_constant \
            --partition=amilan \
            --qos=normal \
            --account=amc-general \
            --time=$time_constant \
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
    else
        echo "Running GPU version for Intensity"
        sbatch \
            --nodes=1 \
            --ntasks=$ntasks_constant \
            --partition=aa100 \
            --qos=normal \
            --gres=gpu:1 \
            --account=amc-general \
            --time=$time_constant \
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
fi

echo "All Parent Jobs submitted"

