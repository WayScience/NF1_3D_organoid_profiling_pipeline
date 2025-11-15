#!/bin/bash
# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

patient="NF0014_T1"
well_fov="C4-2"

echo "Processing patient $patient"

# loop through all convolutions
for convolution in {1..25} 50 75 100; do
    input_subparent_name="convolution_${convolution}"
    mask_subparent_name="convolution_${convolution}_segmentation_masks"
    # shellcheck source=bash
    bash \
        "${git_root}"/2.segment_images/child_segmentation.sh \
        "$patient" \
        "$well_fov" \
        "$input_subparent_name" \
        "$mask_subparent_name"

done

# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"

