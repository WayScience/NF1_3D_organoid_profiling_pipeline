#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=7-00:00
#SBATCH --output=segmentation_parent-%j.out

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
rerun=$1

if [ "$rerun" == "rerun" ]; then
    txt_file="${git_root}/2.segment_images/load_data/rerun_combinations.txt"
else
    txt_file="${git_root}/2.segment_images/load_data/input_combinations.txt"
fi

# Check if TXT file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: TXT file not found at $txt_file"
    exit 1
fi

while IFS= read -r line; do
    # skip the header line
    if [[ "$line" == "patient"* ]]; then
        continue
    fi

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    input_subparent_name="${parts[2]}"
    mask_subparent_name="${parts[3]}"

    echo "Patient: $patient, WellFOV: $well_fov,  Input Subparent Name: $input_subparent_name, Mask Subparent Name: $mask_subparent_name"

    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done

    # requesting 4 nodes (3.75GB per node) for 15GB total memory requirement
    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --gres=gpu:1 \
        --qos=normal \
        --account=amc-general \
        --time=15:00 \
        --output=segmentation_child-%j.out \
        "${git_root}"/2.segment_images/child_segmentation.sh "$patient" "$well_fov" "$input_subparent_name" "$mask_subparent_name"

done < "$txt_file"

echo "All segmentation child jobs submitted"
