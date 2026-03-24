#!/bin/bash

patient=$1
well_fov=$2

conda activate viz_env

cd scripts/ || exit

python visualize_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov"

cd ../ || exit

conda deactivate

