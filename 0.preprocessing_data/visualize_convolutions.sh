#!/bin/bash
# Visualize 2D and 3D convolution outputs for a specified well/FOV.

patient="NF0014_T1"
well_fov="C4-2"

conda activate viz_env

cd scripts || exit

# run each instance of the visualization in nViz
# one instance per script
# per convolution
for i in {1..25}; do
    python 7.visualize_image_convolutions_2d.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --input_subparent_name "convolution_${i}" \
        --mask_subparent_name "convolution_${i}_segmentation_masks" \
        --output_subparent_name "convolution_${i}"


    python 8.visualize_convolutions_3d.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --input_subparent_name "convolution_${i}" \
        --mask_subparent_name "convolution_${i}_segmentation_masks" \
        --output_subparent_name "convolution_${i}"
done

