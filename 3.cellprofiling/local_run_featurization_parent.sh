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

# regardless of the processor type, texture and neighbors features are run on CPU
if [ "$feature" == "Neighbors" ]; then
    # shellcheck disable=SC1091
    source "$git_root"/3.cellprofiling/slurm_scripts/run_neighbors_child.sh \
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
    # shellcheck disable=SC1091
    source "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh \
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
    # shellcheck disable=SC1091
    source "$git_root"/3.cellprofiling/slurm_scripts/run_texture_child.sh \
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
        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh \
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
        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh \
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
        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh \
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
        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh \
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

        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh \
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
        # shellcheck disable=SC1091
        source "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh \
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

if [ "$feature" == "sammed3D" ] ; then
    echo "Running sammed3D feature extraction"
    # shellcheck disable=SC1091
    source "$git_root"/3.cellprofiling/slurm_scripts/run_sammed3D_child.sh \
            "$patient" \
            "$well_fov" \
            "$compartment" \
            "$channel"  \
            "$input_subparent_name" \
            "$mask_subparent_name" \
            "$output_features_subparent_name"
fi

echo "Featurization done"

