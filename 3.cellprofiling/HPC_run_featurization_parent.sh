#!/bin/bash

# patient=$1
# well_fov=$2
# feature=$3
# compartment=$4
# channel=$5
# processor_type=$6
# input_subparent_name=$7
# mask_subparent_name=$8
# output_features_subparent_name=$9

patient="NF0040_T1"
well_fov="B11-4"
feature="Texture"
compartment="Cell"
channel="DNA"
processor_type="CPU"
input_subparent_name="input_subparent"
mask_subparent_name="mask_subparent"
output_features_subparent_name="output_features"


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

alpine_scratch_dir="/scratch/alpine"
anvil_scratch_dir="/anvil/scratch"

if [[ -e "$alpine_scratch_dir" ]]; then
    partition="amilan"
    gpu_partition="aa100"
    gres="--gres=gpu:1" # only used for SAMMed3D feature extraction which is the only feature that uses GPU processing
    qos="--qos=normal"

elif [[ -e "$anvil_scratch_dir" ]]; then
    partition="shared"
    gpu_partition="gpu"
    qos="--mail-type=all"
    gres="--gpus-per-node=1"
else
    echo "Error: No known scratch directory found."
fi



echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type"
echo "InputSubparent: $input_subparent_name, MaskSubparent: $mask_subparent_name, OutputFeaturesSubparent: $output_features_subparent_name"
# regardless of the processor type, texture and neighbors features are run on CPU
if [ "$feature" == "Neighbors" ]; then
    sbatch \
    --nodes=1 \
    --mem=2G \
    --partition="$partition" \
    "$qos" \
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
        --mem=12G \
        --partition="$partition" \
        "$qos" \
        --time=7:00 \
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
        --mem=3G \
        --partition="$partition" \
        "$qos" \
        --time=3:00 \
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
        --mem=2G \
        --partition="$partition" \
        "$qos" \
        --time=00:01:30 \
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
        --mem=6G \
        --partition="$partition" \
        "$qos" \
        --time=5:00 \
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
        --mem=6G \
        --partition="$partition" \
        "$qos" \
        --time=5:00 \
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

# if [ "$feature" == "SAMMed3D" ] ; then
#     if [ "$compartment" == "Nucleocentric" ] ; then
#         sbatch \
#             --nodes=1 \
#             --mem=6G \
#             --partition="$gpu_partition" \
#             "$gres" \
#             "$qos" \
#             --time=10:00 \
#             --gres=gpu:1 \
#             --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
#             --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_sammed3d_child-%j.out" \
#             "$git_root"/3.cellprofiling/slurm_scripts/run_nucleocentric_child.sh \
#                 "$patient" \
#                 "$well_fov" \
#                 "$compartment" \
#                 "$channel"  \
#                 "$input_subparent_name" \
#                 "$mask_subparent_name" \
#                 "$output_features_subparent_name"
#     else
#         sbatch \
#             --nodes=1 \
#             --mem=6G \
#             --partition="$gpu_partition" \
#             "$gres" \
#             "$qos" \
#             --time=10:00 \
#             --gres=gpu:1 \
#             --export=patient="$patient",well_fov="$well_fov",compartment="$compartment",channel="$channel" \
#             --output="logs/child/${patient}_${well_fov}/${compartment}_${channel}_sammed3d_child-%j.out" \
#             "$git_root"/3.cellprofiling/slurm_scripts/run_sammed3D_child.sh \
#                 "$patient" \
#                 "$well_fov" \
#                 "$compartment" \
#                 "$channel"  \
#                 "$input_subparent_name" \
#                 "$mask_subparent_name" \
#                 "$output_features_subparent_name"
#     fi
# fi

echo "All Parent Jobs submitted"

