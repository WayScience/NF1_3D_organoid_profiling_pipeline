#!/bin/bash

patient=$1
well_fov=$2

uv run python scripts/visualize_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --input_subparent_name "zstack_images" \
    --mask_subparent_name "segmentation_masks"
